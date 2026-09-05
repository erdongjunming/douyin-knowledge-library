from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, Transcript, Work
from douyin_library.notes.state import classify_note_bundle
from douyin_library.transcription.service import _transcript_fingerprint


@dataclass(frozen=True, slots=True)
class AudioInventory:
    status: Literal["ready", "blocked", "missing"]
    path: str | None = None
    exists: bool = False
    bytes: int | None = None
    sha256: str | None = None
    recorded_bytes: int | None = None
    recorded_sha256: str | None = None
    probe_verified: bool = False
    verified_at: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptInventory:
    status: Literal["succeeded", "missing", "blocked"]
    transcript_id: int | None = None
    source_kind: str | None = None
    variant: str = "raw"
    segment_count: int = 0
    fingerprint: str | None = None
    file_consistent: bool = False
    parent_valid: bool | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class NoteInventory:
    quality_status: str
    provider: str | None = None
    model: str | None = None
    quality_gate: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class JobInventory:
    kind: str
    job_id: int
    status: str
    stage: str | None = None
    attempts: int = 0
    retry_scheduled: bool = False
    stale: bool = False
    error_code: str | None = None
    error_detail: str | None = None


@dataclass(frozen=True, slots=True)
class InventoryItem:
    aweme_id: str
    work_id: int
    audio: AudioInventory
    raw: TranscriptInventory
    cleaned: TranscriptInventory
    note: NoteInventory
    jobs: tuple[JobInventory, ...] = ()
    reason: str | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class InventoryReport:
    items: tuple[InventoryItem, ...]
    selected_ids: tuple[str, ...] = ()
    selection_reasons: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    selected_ids: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...] = ()

    def __iter__(self):
        return iter(self.selected_ids)


def _json(value: str | None):
    try:
        return json.loads(value) if value else None
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio(asset: Asset | None) -> AudioInventory:
    if asset is None:
        return AudioInventory("blocked", reason="audio_asset_missing")
    path = Path(asset.path)
    exists = path.is_file()
    if not exists:
        return AudioInventory("blocked", path=str(path), reason="audio_file_missing")
    try:
        size = path.stat().st_size
        digest = _sha256(path) if size else None
        readable = os.access(path, os.R_OK)
    except OSError:
        return AudioInventory(
            "blocked", path=str(path), exists=True, reason="audio_file_unreadable"
        )
    probe = _json(asset.probe_json)
    valid = (
        readable
        and size > 0
        and asset.bytes is not None
        and asset.sha256
        and probe
        and asset.verified_at
    )
    reason = None
    if not valid:
        reason = "audio_not_verified"
    elif size != asset.bytes or digest != asset.sha256:
        reason = "audio_asset_changed"
    return AudioInventory(
        "ready" if valid and reason is None else "blocked",
        str(path),
        True,
        size,
        digest,
        asset.bytes,
        asset.sha256,
        bool(valid and reason is None),
        asset.verified_at.isoformat() if asset.verified_at else None,
        reason,
    )


def _segments(row: Transcript):
    value = _json(row.segments_json)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(x, dict) and isinstance(x.get("text"), str) for x in value)
    ):
        return None
    return value


def _transcript(
    row: Transcript | None,
    layout: StorageLayout,
    aweme_id: str,
    *,
    parent: Transcript | None = None,
) -> TranscriptInventory:
    if row is None:
        return TranscriptInventory("missing", variant="raw")
    if row.status != "succeeded":
        return TranscriptInventory(
            "blocked",
            row.id,
            row.source_kind,
            row.variant,
            reason=f"transcript_status_{row.status}",
            parent_valid=parent is not None,
        )
    segs = _segments(row)
    if segs is None or not (row.text or "").strip():
        return TranscriptInventory(
            "blocked",
            row.id,
            row.source_kind,
            row.variant,
            reason="transcript_segments_invalid",
            parent_valid=parent is not None,
        )
    try:
        fp = _transcript_fingerprint(row)
    except Exception:
        fp = None
    if row.variant == "cleaned":
        meta = _json(row.word_timestamps_json)
        parent_fingerprint = None
        if parent is not None:
            try:
                parent_fingerprint = _transcript_fingerprint(parent)
            except (TypeError, ValueError, json.JSONDecodeError):
                parent_fingerprint = None
        if (
            parent is None
            or not isinstance(meta, dict)
            or parent_fingerprint is None
            or meta.get("raw_fingerprint") != parent_fingerprint
        ):
            return TranscriptInventory(
                "blocked",
                row.id,
                row.source_kind,
                row.variant,
                len(segs),
                fp,
                False,
                False,
                "cleaned_parent_invalid",
            )
        text_path, seg_path = (
            layout.transcript_cleaned_text_path(aweme_id),
            layout.transcript_cleaned_json_path(aweme_id),
        )
    else:
        text_path, seg_path = (
            layout.transcript_text_path(aweme_id),
            layout.transcript_segments_path(aweme_id),
        )
    consistent = False
    try:
        file_text = text_path.read_text(encoding="utf-8")
        file_segments = _json(seg_path.read_text(encoding="utf-8"))
        consistent = file_text == row.text and file_segments == segs
        if row.source_kind == "asr" and not layout.transcript_qwen_path(aweme_id).is_file():
            consistent = False
    except (OSError, UnicodeError):
        consistent = False
    if not consistent:
        return TranscriptInventory(
            "blocked",
            row.id,
            row.source_kind,
            row.variant,
            len(segs),
            fp,
            False,
            parent is not None,
            "transcript_files_inconsistent",
        )
    return TranscriptInventory(
        "succeeded",
        row.id,
        row.source_kind,
        row.variant,
        len(segs),
        fp,
        True,
        parent is not None if parent else None,
    )


def build_inventory(
    session: Session,
    layout: StorageLayout,
    *,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(minutes=15),
) -> InventoryReport:
    now = now or datetime.now(UTC)
    works = session.scalars(select(Work).order_by(Work.aweme_id)).all()
    items = []
    for work in works:
        asset = session.scalar(select(Asset).where(Asset.work_id == work.id, Asset.kind == "audio"))
        audio = _audio(asset)
        rows = session.scalars(select(Transcript).where(Transcript.work_id == work.id)).all()
        raws = [r for r in rows if r.variant == "raw"]
        raw_row = next(
            (r for r in sorted(raws, key=lambda r: r.id, reverse=True) if r.status == "succeeded"),
            None,
        )
        raw = _transcript(raw_row, layout, work.aweme_id)
        cleaned_row = next(
            (
                r
                for r in sorted(rows, key=lambda r: r.id, reverse=True)
                if r.variant == "cleaned" and r.status == "succeeded"
            ),
            None,
        )
        parent = (
            next(
                (
                    candidate
                    for candidate in raws
                    if candidate.id == cleaned_row.parent_transcript_id
                ),
                None,
            )
            if cleaned_row is not None and cleaned_row.parent_transcript_id is not None
            else None
        )
        cleaned = _transcript(cleaned_row, layout, work.aweme_id, parent=parent)
        note_state = classify_note_bundle(layout.note_dir(work.aweme_id))
        note = NoteInventory(
            note_state.quality_status,
            note_state.provider,
            note_state.model,
            note_state.quality_gate,
        )
        jobs = []
        for job in session.scalars(
            select(Job)
            .where(
                Job.work_id == work.id,
                Job.kind.in_(("transcribe_work", "clean_transcript_work", "generate_note")),
            )
            .order_by(Job.id)
        ).all():
            heartbeat = job.heartbeat_at or job.started_at
            heartbeat_utc = _as_utc(heartbeat)
            now_utc = _as_utc(now)
            stale = (
                job.status == "running"
                and heartbeat_utc is not None
                and now_utc is not None
                and now_utc - heartbeat_utc > stale_after
            )
            retry = (
                job.status == "pending"
                and job.next_retry_at is not None
                and job.next_retry_at > now
            )
            detail = (job.error_detail or "")[:500] or None
            jobs.append(
                JobInventory(
                    job.kind,
                    job.id,
                    job.status,
                    job.stage,
                    job.attempts,
                    retry,
                    stale,
                    job.error_code,
                    detail,
                )
            )
        blocked = (
            audio.reason
            if audio.status != "ready"
            else (raw.reason if raw.status == "blocked" else None)
        )
        items.append(
            InventoryItem(
                work.aweme_id,
                work.id,
                audio,
                raw,
                cleaned,
                note,
                tuple(jobs),
                reason=blocked,
                blocked_reason=blocked,
            )
        )
    return InventoryReport(tuple(items))


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def select_raw_asr_candidates(
    report: InventoryReport, aweme_ids=None, limit: int | None = None
) -> CandidateSelection:
    requested = None if aweme_ids is None else {str(x).strip() for x in aweme_ids if str(x).strip()}
    eligible = []
    reasons = []
    for item in report.items:
        if requested is not None and item.aweme_id not in requested:
            continue
        reason = None
        if item.audio.status != "ready":
            reason = "audio_not_ready"
        elif item.raw.status == "succeeded":
            reason = "raw_transcript_exists"
        elif item.raw.status == "blocked":
            reason = item.raw.reason or "raw_transcript_invalid"
        elif any(
            j.kind == "transcribe_work" and j.status in {"pending", "running"} for j in item.jobs
        ):
            reason = "transcribe_active"
        if reason:
            reasons.append((item.aweme_id, reason))
        else:
            eligible.append(item.aweme_id)
    eligible.sort()
    if limit is not None:
        eligible = eligible[: max(0, limit)]
    return CandidateSelection(tuple(eligible), tuple(sorted(reasons)))


__all__ = [
    "AudioInventory",
    "TranscriptInventory",
    "NoteInventory",
    "JobInventory",
    "InventoryItem",
    "InventoryReport",
    "CandidateSelection",
    "build_inventory",
    "select_raw_asr_candidates",
]
