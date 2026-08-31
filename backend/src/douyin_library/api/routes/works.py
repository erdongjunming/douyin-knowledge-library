from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, case, exists, func, or_, select
from sqlalchemy.orm import Session, aliased

from douyin_library.api.dependencies import get_runtime_session
from douyin_library.config import Settings
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Analysis, Asset, Creator, Job, Transcript, Work
from douyin_library.notes.state import NoteBundleState, classify_note_bundle
from douyin_library.security.cookie_store import CookieStore
from douyin_library.upstream.jiji_gateway import JijiGateway

router = APIRouter(prefix="/api/works", tags=["works"])


async def get_work_gateway() -> AsyncIterator[JijiGateway]:
    cookie = CookieStore().read()
    if not cookie:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="douyin_cookie_not_configured",
        )
    gateway = JijiGateway.from_cookie_header(cookie)
    try:
        yield gateway
    finally:
        await gateway.close()


def get_work_session(
    session: Annotated[Session, Depends(get_runtime_session)],
) -> Session:
    return session


@router.post("/{work_id}/process")
async def process_work(
    work_id: int,
    gateway: Annotated[JijiGateway, Depends(get_work_gateway)],
    session: Annotated[Session, Depends(get_work_session)],
) -> dict[str, Any]:
    work = session.scalar(select(Work).where(Work.id == work_id))
    if work is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="work_not_found")
    service = WorkMediaService(
        session,
        gateway,
        StorageLayout(Settings().storage_root),
    )
    try:
        result = await service.process(work)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {
        "work_id": work.id,
        "status": "succeeded",
        "audio": str(result.audio.path),
        "cover": str(result.cover),
        "metadata": str(result.metadata),
    }


@router.get("")
def list_works(
    session: Annotated[Session, Depends(get_work_session)],
    limit: int | None = Query(default=None, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    query: str | None = Query(default=None),
    failed_only: bool = Query(default=False),
) -> dict[str, Any]:
    effective_limit = limit or 100
    audio_exists = exists().where(Asset.work_id == Work.id, Asset.kind == "audio")
    transcript_exists = exists().where(
        Transcript.work_id == Work.id,
        Transcript.status == "succeeded",
    )
    download_history = aliased(Job)
    transcription_history = aliased(Job)
    latest_download_id = (
        select(func.max(download_history.id))
        .where(
            download_history.work_id == Work.id,
            download_history.kind == "download_work",
        )
        .correlate(Work)
        .scalar_subquery()
    )
    latest_transcription_id = (
        select(func.max(transcription_history.id))
        .where(
            transcription_history.work_id == Work.id,
            transcription_history.kind == "transcribe_work",
        )
        .correlate(Work)
        .scalar_subquery()
    )
    failed_download = exists().where(
        Job.work_id == Work.id,
        Job.id == latest_download_id,
        Job.status == "failed",
    )
    failed_transcription = exists().where(
        Job.work_id == Work.id,
        Job.id == latest_transcription_id,
        Job.status == "failed",
    )
    extraction_failure = or_(
        and_(failed_download, ~audio_exists),
        and_(failed_transcription, ~transcript_exists),
    )
    statement = (
        select(Work, Creator.nickname)
        .outerjoin(Creator, Work.creator_id == Creator.id)
        .order_by(
            case(
                (
                    exists().where(
                        Transcript.work_id == Work.id,
                        Transcript.status == "succeeded",
                    ),
                    0,
                ),
                else_=1,
            ),
            Work.published_at.desc().nullslast(),
            Work.id.desc(),
        )
    )
    if query and query.strip():
        pattern = f"%{query.strip()}%"
        statement = statement.where(
            or_(Work.description.ilike(pattern), Creator.nickname.ilike(pattern))
        )
    base_statement = statement
    failed_total = int(
        session.scalar(
            select(func.count()).select_from(
                base_statement.where(extraction_failure).order_by(None).subquery()
            )
        )
        or 0
    )
    if failed_only:
        statement = base_statement.where(extraction_failure)
    total = int(
        session.scalar(
            select(func.count()).select_from(statement.order_by(None).subquery())
        )
        or 0
    )

    rows = session.execute(statement.offset(offset).limit(effective_limit)).all()
    work_ids = [work.id for work, _ in rows]
    assets = (
        session.scalars(select(Asset).where(Asset.work_id.in_(work_ids))).all()
        if work_ids
        else []
    )
    assets_by_work: dict[int, list[str]] = {}
    for asset in assets:
        assets_by_work.setdefault(asset.work_id, []).append(asset.kind)

    analyses = session.scalars(
        select(Analysis).where(
            Analysis.work_id.in_(work_ids),
            Analysis.status == "succeeded",
        )
    ).all() if work_ids else []
    topics_by_work: dict[int, list[str]] = {}
    for analysis in analyses:
        try:
            topics = json.loads(analysis.topics_json or "[]")
        except json.JSONDecodeError:
            topics = []
        if isinstance(topics, list):
            topics_by_work.setdefault(analysis.work_id, []).extend(
                str(topic) for topic in topics if str(topic).strip()
            )

    transcripts = session.scalars(
        select(Transcript)
        .where(
            Transcript.work_id.in_(work_ids),
            Transcript.status == "succeeded",
        )
        .order_by(Transcript.id.desc())
    ).all() if work_ids else []
    transcript_by_work: dict[int, Transcript] = {}
    for transcript in transcripts:
        current = transcript_by_work.get(transcript.work_id)
        if current is None or _transcript_rank(transcript) > _transcript_rank(current):
            transcript_by_work[transcript.work_id] = transcript

    jobs = (
        session.scalars(
            select(Job)
            .where(
                Job.work_id.in_(work_ids),
                Job.kind.in_(
                    ("download_work", "transcribe_work", "clean_transcript_work", "generate_note")
                ),
            )
            .order_by(Job.id.desc())
        ).all()
        if work_ids
        else []
    )
    latest_jobs: dict[tuple[int, str], Job] = {}
    for job in jobs:
        latest_jobs.setdefault((job.work_id, job.kind), job)

    layout = StorageLayout(Settings().storage_root)
    items: list[dict[str, Any]] = []
    for work, nickname in rows:
        assets_for_work = sorted(set(assets_by_work.get(work.id, [])))
        transcript = transcript_by_work.get(work.id)
        note_path = layout.note_dir(work.aweme_id) / "learning_note.md"
        note_available = note_path.is_file()
        note_text = _read_text(note_path) if note_available else None
        note_state = classify_note_bundle(layout.note_dir(work.aweme_id))
        state = _work_state(
            work.id,
            assets_for_work,
            transcript is not None,
            note_available,
            latest_jobs,
            note_state,
        )
        note_metadata = _note_generation_metadata(layout.note_dir(work.aweme_id), note_state)
        items.append(
            {
                "id": work.id,
                "aweme_id": work.aweme_id,
                "description": work.description,
                "media_type": work.media_type,
                "availability": work.availability,
                "asset_kinds": assets_for_work,
                "creator_nickname": nickname,
                "published_at": work.published_at.isoformat() if work.published_at else None,
                "topics": sorted(set(topics_by_work.get(work.id, []))),
                "transcript_status": "succeeded" if transcript else None,
                "transcript_preview": _preview(transcript.text if transcript else None),
                **_transcript_provenance(session, work.id, transcript),
                "note_status": state["note_status"],
                "note_available": note_available,
                "note_preview": _preview(note_text),
                **_note_contract_fields(note_state, state, note_metadata),
                "extraction_status": state["extraction_status"],
                "failure": state["failure"],
            }
        )

    response: dict[str, Any] = {"items": items}
    # Keep the old empty default response compatible with existing clients. The web UI
    # always sends an explicit limit and receives the full pagination metadata.
    if items or limit is not None or offset > 0 or query or failed_only:
        response.update(
            {
                "total": total,
                "limit": effective_limit,
                "offset": offset,
                "has_more": offset + len(items) < total,
                "failed_total": failed_total,
            }
        )
    return response


@router.get("/{work_id}")
def get_work(
    work_id: int,
    session: Annotated[Session, Depends(get_work_session)],
) -> dict[str, Any]:
    work = session.get(Work, work_id)
    if work is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="work_not_found")
    transcript = _latest_transcript(session, work.id)
    segments: list[Any] = []
    if transcript and transcript.segments_json:
        try:
            parsed = json.loads(transcript.segments_json)
            segments = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            segments = []
    layout = StorageLayout(Settings().storage_root)
    note_path = layout.note_dir(work.aweme_id) / "learning_note.md"
    note_text = _read_text(note_path) if note_path.is_file() else None
    note_state = classify_note_bundle(layout.note_dir(work.aweme_id))
    assets = session.scalars(select(Asset).where(Asset.work_id == work.id)).all()
    jobs = session.scalars(
        select(Job)
        .where(
            Job.work_id == work.id,
            Job.kind.in_(
                ("download_work", "transcribe_work", "clean_transcript_work", "generate_note")
            ),
        )
        .order_by(Job.id.desc())
    ).all()
    latest_jobs: dict[tuple[int, str], Job] = {}
    for job in jobs:
        latest_jobs.setdefault((job.work_id, job.kind), job)
    state = _work_state(
        work.id,
        sorted({asset.kind for asset in assets}),
        transcript is not None,
        note_text is not None,
        latest_jobs,
        note_state,
    )
    note_metadata = _note_generation_metadata(layout.note_dir(work.aweme_id), note_state)
    return {
        "id": work.id,
        "aweme_id": work.aweme_id,
        "description": work.description,
        "note": {
            "path": str(note_path),
            "text": note_text,
        } if note_text is not None else None,
        "note_status": state["note_status"],
        "note_available": note_text is not None,
        **_note_contract_fields(note_state, state, note_metadata),
        "extraction_status": state["extraction_status"],
        "failure": state["failure"],
        "transcript": (
            {
                "engine": transcript.engine,
                "model": transcript.model,
                "language": transcript.language,
                "text": transcript.text,
                "segments": segments,
                **_transcript_provenance(session, work.id, transcript),
            }
            if transcript
            else None
        ),
        **_transcript_provenance(session, work.id, transcript),
    }


def _transcript_rank(transcript: Transcript) -> tuple[int, int]:
    return (1 if transcript.variant == "cleaned" else 0, transcript.id or 0)


def _latest_transcript(session: Session, work_id: int) -> Transcript | None:
    transcripts = session.scalars(
        select(Transcript)
        .where(Transcript.work_id == work_id, Transcript.status == "succeeded")
        .order_by(Transcript.id.desc())
    ).all()
    return max(transcripts, key=_transcript_rank, default=None)


def _transcript_provenance(
    session: Session,
    work_id: int,
    transcript: Transcript | None,
) -> dict[str, Any]:
    if transcript is None:
        return {
            "variant": None,
            "source_kind": None,
            "transcript_variant": None,
            "transcript_source_kind": None,
            "cleaner_model": None,
            "cleaner_prompt_version": None,
            "cleaning_fallback": None,
            "cleaning_fallback_reason": None,
        }
    try:
        parsed_metadata = json.loads(transcript.word_timestamps_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed_metadata = {}
    metadata = parsed_metadata if isinstance(parsed_metadata, dict) else {}
    cleaner_model = (
        metadata.get("cleaner_model") if isinstance(metadata.get("cleaner_model"), str) else None
    )
    prompt_version = (
        metadata.get("prompt_version") if isinstance(metadata.get("prompt_version"), str) else None
    )
    fallback = None
    if transcript.variant != "cleaned":
        failed = session.scalar(
            select(Job)
            .where(
                Job.work_id == work_id,
                Job.kind == "clean_transcript_work",
                Job.status == "failed",
            )
            .order_by(Job.id.desc())
        )
        fallback = _failure_reason(failed, "cleaned_unavailable") if failed else "cleaned_missing"
    return {
        "variant": transcript.variant,
        "source_kind": transcript.source_kind,
        "transcript_variant": transcript.variant,
        "transcript_source_kind": transcript.source_kind,
        "cleaner_model": cleaner_model,
        "cleaner_prompt_version": prompt_version,
        "cleaning_fallback": fallback,
        "cleaning_fallback_reason": fallback,
        "cleaner": {
            "engine": "ollama-cleaner",
            "model": cleaner_model,
            "prompt_version": prompt_version,
        } if transcript.variant == "cleaned" else None,
    }


def _preview(value: str | None, *, limit: int = 120) -> str | None:
    normalized = " ".join((value or "").split())
    if not normalized:
        return None
    return normalized if len(normalized) <= limit else f"{normalized[:limit]}..."


def _read_text(path: Any) -> str | None:
    try:
        value = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return None
    return value.strip() or None


def _work_state(
    work_id: int,
    asset_kinds: list[str],
    transcript_available: bool,
    note_available: bool,
    latest_jobs: dict[tuple[int, str], Job],
    note_state: NoteBundleState | None = None,
) -> dict[str, Any]:
    download = latest_jobs.get((work_id, "download_work"))
    transcription = latest_jobs.get((work_id, "transcribe_work"))
    note = latest_jobs.get((work_id, "generate_note"))
    has_audio = "audio" in asset_kinds

    failure: dict[str, str] | None = None
    if not has_audio and download and download.status == "failed":
        failure = {
            "stage": "audio_extraction",
            "reason": _failure_reason(download, "音频提取失败"),
        }
    elif not transcript_available and transcription and transcription.status == "failed":
        failure = {
            "stage": "transcription",
            "reason": _failure_reason(transcription, "ASR 转写失败"),
        }
    elif not note_available and note and note.status == "failed":
        failure = {
            "stage": "note_generation",
            "reason": _failure_reason(note, "笔记生成失败"),
        }

    if note_available:
        note_status = "succeeded"
    elif note and note.status in {"running", "pending", "failed"}:
        note_status = note.status
    else:
        note_status = "not_queued"

    extraction_failure = bool(
        failure and failure["stage"] in {"audio_extraction", "transcription"}
    )
    if extraction_failure:
        extraction_status = "failed"
    elif transcript_available:
        extraction_status = "succeeded"
    elif transcription and transcription.status in {"running", "pending"}:
        extraction_status = transcription.status
    elif has_audio:
        extraction_status = "not_queued"
    elif download and download.status in {"running", "pending"}:
        extraction_status = download.status
    else:
        extraction_status = "not_queued"
    return {
        "extraction_status": extraction_status,
        "note_status": note_status,
        "note_job_status": note.status if note else None,
        "failure": failure,
    }


def _note_contract_fields(
    note_state: NoteBundleState,
    work_state: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    coverage = note_state.coverage
    segment_coverage = _coverage_pair(
        coverage,
        "segment_disposition_count",
        "segment_total",
    )
    atom_coverage = _coverage_pair(
        coverage,
        "evidence_atom_included",
        "evidence_atom_total",
    )
    # Flat names are the stable API contract.  The nested aliases make the
    # shape convenient for clients while preserving older consumers.
    return {
        "note_quality_status": note_state.quality_status,
        "note_job_status": work_state.get("note_job_status"),
        "note_provider": note_state.provider,
        "note_model": note_state.model,
        "note_quality_gate": note_state.quality_gate,
        "note_segment_coverage": segment_coverage,
        "note_atom_coverage": atom_coverage,
        "note_coverage": coverage,
        "coverage": coverage,
        "note_generation": metadata,
        "note_generation_metadata": metadata,
        "generation_metadata": metadata,
    }


def _coverage_pair(
    coverage: dict[str, Any] | None,
    covered_key: str,
    total_key: str,
) -> dict[str, Any] | None:
    if not isinstance(coverage, dict):
        return None
    covered = coverage.get(covered_key)
    total = coverage.get(total_key)
    if not isinstance(covered, int) or isinstance(covered, bool):
        return None
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        return None
    return {
        "covered": max(0, covered),
        "total": total,
        "percent": round((covered / total * 100) if total else 0.0, 1),
    }


def _note_generation_metadata(
    note_dir: Path,
    note_state: NoteBundleState,
) -> dict[str, Any]:
    manifest = _read_json_object(note_dir / "generation_manifest.json")
    page = _read_json_object(note_dir / "page_metadata.json")
    audit = _read_json_object(note_dir / "generation_audit.json")
    metadata: dict[str, Any] = {}
    for source in (manifest, page, audit):
        if source:
            for key in (
                "generated_at",
                "started_at",
                "finished_at",
                "generator_version",
                "version",
                "prompt_version",
                "prompt_versions",
                "processing_audit_id",
                "attempts",
                "status",
            ):
                if key in source and key not in metadata:
                    metadata[key] = source[key]
    metadata.update(
        {
            "provider": note_state.provider,
            "model": note_state.model,
            "quality_gate": note_state.quality_gate,
        }
    )
    return metadata


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _failure_reason(job: Job, fallback: str) -> str:
    detail = " ".join((job.error_detail or "").split()).strip()
    code = " ".join((job.error_code or "").split()).strip()
    if not detail:
        return code or fallback
    if detail.endswith(":") and code:
        return f"{detail.rstrip(':').strip()}（{code}）"
    return detail
