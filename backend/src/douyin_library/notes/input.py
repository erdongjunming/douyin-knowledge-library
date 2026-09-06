from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from douyin_library.media.storage import StorageLayout
from douyin_library.models import Transcript, Work


@dataclass(frozen=True, slots=True)
class InputSegment:
    index: int
    start: float | None
    end: float | None
    text: str
    segment_id: str | None = None


@dataclass(frozen=True, slots=True)
class NoteInput:
    aweme_id: str
    transcript: str
    segments: list[InputSegment]
    metadata: dict[str, Any]
    description: str | None
    author: str | None
    author_unique_id: str | None
    published_at: str | None
    duration_ms: int | None
    source_url: str | None
    source_files: dict[str, str]
    asset_records: dict[str, dict[str, Any]]
    warnings: list[str]
    transcript_variant: str = "raw"
    transcript_source_kind: str = "asr"
    cleaning_fallback: str | None = None
    note_budget: dict[str, Any] | None = None


class NoteInputLoader:
    def __init__(self, layout: StorageLayout, session: Any | None = None) -> None:
        self.layout = layout
        self.session = session

    def load(self, aweme_id: str) -> NoteInput:
        transcript_path = self.layout.transcript_text_path(aweme_id)
        raw_transcript = transcript_path.read_text(encoding="utf-8-sig")
        warnings: list[str] = []
        metadata = self._load_metadata(aweme_id, warnings)
        raw_segments = self._load_segments(aweme_id, raw_transcript, warnings)
        transcript = raw_transcript
        segments = raw_segments
        transcript_variant = "raw"
        transcript_source_kind = self._source_kind(aweme_id)
        cleaning_fallback: str | None = None
        cleaned = self._load_cleaned(
            aweme_id,
            raw_transcript,
            raw_segments,
            transcript_source_kind,
            warnings,
        )
        if cleaned is not None:
            transcript, segments, transcript_source_kind = cleaned
            transcript_variant = "cleaned"
        else:
            cleaning_fallback = self._cleaning_fallback
        author_value = metadata.get("author")
        author = (
            _optional_text(author_value.get("nickname"))
            if isinstance(author_value, dict)
            else _optional_text(author_value)
        )
        author_unique_id = _optional_text(metadata.get("author_unique_id"))
        if author_unique_id is None and isinstance(author_value, dict):
            author_unique_id = _optional_text(
                author_value.get("unique_id") or author_value.get("uniqueId")
            )
        duration_ms = _duration_ms(metadata)
        source_url = _optional_text(metadata.get("source_url"))
        if source_url is None:
            source_url = f"https://www.douyin.com/video/{aweme_id}"
        source_files = self._source_files(aweme_id)
        asset_records = {key: _asset_record(Path(path)) for key, path in source_files.items()}
        note_budget = self._load_note_budget(aweme_id)
        return NoteInput(
            aweme_id=aweme_id,
            transcript=transcript,
            segments=segments,
            metadata=metadata,
            description=_optional_text(metadata.get("description") or metadata.get("desc")),
            author=author,
            author_unique_id=author_unique_id,
            published_at=_optional_text(metadata.get("published_at")),
            duration_ms=duration_ms,
            source_url=source_url,
            source_files=source_files,
            asset_records=asset_records,
            warnings=warnings,
            transcript_variant=transcript_variant,
            transcript_source_kind=transcript_source_kind,
            cleaning_fallback=cleaning_fallback,
            note_budget=note_budget,
        )

    _cleaning_fallback: str | None = None

    def _source_kind(self, aweme_id: str) -> str:
        if self.session is None:
            return "asr"
        work = self.session.scalar(select(Work).where(Work.aweme_id == aweme_id))
        if work is None:
            return "asr"
        raw = self.session.scalar(
            select(Transcript)
            .where(
                Transcript.work_id == work.id,
                Transcript.variant == "raw",
                Transcript.status == "succeeded",
            )
            .order_by(Transcript.id.desc())
        )
        return raw.source_kind if raw and raw.source_kind in {"subtitle", "asr"} else "asr"

    def _load_cleaned(
        self,
        aweme_id: str,
        raw_transcript: str,
        raw_segments: list[InputSegment],
        source_kind: str,
        warnings: list[str],
    ) -> tuple[str, list[InputSegment], str] | None:
        self._cleaning_fallback = None
        cleaned_text_path = self.layout.transcript_cleaned_text_path(aweme_id)
        cleaned_json_path = self.layout.transcript_cleaned_json_path(aweme_id)
        if not cleaned_text_path.is_file() or not cleaned_json_path.is_file():
            self._cleaning_fallback = "cleaned_missing"
            return None
        try:
            payload = json.loads(cleaned_json_path.read_text(encoding="utf-8-sig"))
            cleaned_values = payload.get("segments") if isinstance(payload, dict) else payload
            if not isinstance(cleaned_values, list):
                raise ValueError("segments")
            cleaned_segments = _parse_segments(cleaned_values)
            if cleaned_segments is None or not _segment_identity_matches(
                raw_segments, cleaned_values
            ):
                raise ValueError("segment_identity")
            try:
                raw_values = json.loads(
                    self.layout.transcript_segments_path(aweme_id).read_text(
                        encoding="utf-8-sig"
                    )
                )
            except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
                raw_values = []
            if not _segment_ids_match(raw_values, cleaned_values):
                raise ValueError("segment_identity")
            cleaned_text = cleaned_text_path.read_text(encoding="utf-8-sig").strip()
            if not cleaned_text or not _covers_transcript(cleaned_segments, cleaned_text):
                # The cleaner may normalize spacing; ensure no content is silently lost.
                cleaned_text = "\n".join(segment.text for segment in cleaned_segments)
            if not cleaned_text.strip():
                raise ValueError("text")
            expected_fingerprint = _raw_file_fingerprint(
                self.layout.transcript_segments_path(aweme_id),
                raw_transcript,
                source_kind,
            )
            metadata_fingerprint = (
                payload.get("raw_fingerprint") if isinstance(payload, dict) else None
            )
            if self.session is not None:
                work = self.session.scalar(select(Work).where(Work.aweme_id == aweme_id))
                cleaned_row = None
                if work is not None:
                    cleaned_row = self.session.scalar(
                        select(Transcript)
                        .where(
                            Transcript.work_id == work.id,
                            Transcript.variant == "cleaned",
                            Transcript.status == "succeeded",
                        )
                        .order_by(Transcript.id.desc())
                    )
                if cleaned_row is None:
                    raise ValueError("cleaned_record_missing")
                source_kind = (
                    cleaned_row.source_kind
                    if cleaned_row.source_kind in {"subtitle", "asr"}
                    else source_kind
                )
                metadata = _json_object(cleaned_row.word_timestamps_json)
                metadata_fingerprint = metadata.get("raw_fingerprint") or metadata_fingerprint
                raw_parent = self.session.get(Transcript, cleaned_row.parent_transcript_id)
                if raw_parent is None or raw_parent.variant != "raw":
                    raise ValueError("parent_missing")
                try:
                    raw_payload = json.loads(raw_parent.segments_json or "[]")
                except (TypeError, ValueError, json.JSONDecodeError):
                    raw_payload = []
                expected_fingerprint = _raw_fingerprint_payload(
                    raw_parent.text or "", raw_payload, raw_parent.source_kind
                )
                current_file_fingerprint = _raw_file_fingerprint(
                    self.layout.transcript_segments_path(aweme_id),
                    raw_transcript,
                    raw_parent.source_kind,
                )
                if current_file_fingerprint != expected_fingerprint:
                    raise ValueError("raw_changed")
                if cleaned_row.parent_transcript_id != raw_parent.id:
                    raise ValueError("parent_mismatch")
            if metadata_fingerprint != expected_fingerprint:
                raise ValueError("parent_fingerprint")
            return cleaned_text, cleaned_segments, source_kind
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            self._cleaning_fallback = "cleaned_invalid"
            warnings.append(
                "cleaned transcript 无法验证 parent fingerprint 或 segment identity；已回退 raw。"
            )
            return None

    def _load_metadata(self, aweme_id: str, warnings: list[str]) -> dict[str, Any]:
        path = self.layout.metadata_path(aweme_id)
        if not path.is_file():
            warnings.append("metadata JSON 不存在；已使用空元数据继续生成。")
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            warnings.append("metadata JSON 损坏或无法读取；已使用空元数据继续生成。")
            return {}
        if not isinstance(value, dict):
            warnings.append("metadata JSON 顶层不是对象；已使用空元数据继续生成。")
            return {}
        return value

    def _load_segments(
        self,
        aweme_id: str,
        transcript: str,
        warnings: list[str],
    ) -> list[InputSegment]:
        path = self.layout.transcript_segments_path(aweme_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            segments = _parse_segments(value)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            segments = None
        if segments is not None and _covers_transcript(segments, transcript):
            return segments
        warnings.append(
            "segments.json 无效、不合法或未覆盖完整 transcript；"
            "已回退为包含完整 transcript 的 synthetic segment，未裁剪源文本。"
        )
        return [InputSegment(index=0, start=None, end=None, text=transcript)]

    def _source_files(self, aweme_id: str) -> dict[str, str]:
        return {
            "audio": str(self.layout.audio_path(aweme_id).resolve()),
            "cover": str(self.layout.cover_path(aweme_id).resolve()),
            "metadata_json": str(self.layout.metadata_path(aweme_id).resolve()),
            "transcript_txt": str(self.layout.transcript_text_path(aweme_id).resolve()),
            "transcript_segments": str(self.layout.transcript_segments_path(aweme_id).resolve()),
            "qwen3_asr_json": str(self.layout.transcript_qwen_path(aweme_id).resolve()),
            "transcript_cleaned_txt": str(
                self.layout.transcript_cleaned_text_path(aweme_id).resolve()
            ),
            "transcript_cleaned_json": str(
                self.layout.transcript_cleaned_json_path(aweme_id).resolve()
            ),
            "note_budget": str(
                (self.layout.note_dir(aweme_id) / "note_budget.json").resolve()
            ),
        }

    def _load_note_budget(self, aweme_id: str) -> dict[str, Any] | None:
        path = self.layout.note_dir(aweme_id) / "note_budget.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None


def _parse_segments(value: Any) -> list[InputSegment] | None:
    if not isinstance(value, list) or not value:
        return None
    segments: list[InputSegment] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            return None
        text = item["text"]
        if not text:
            return None
        start = _optional_non_negative_float(item.get("start"))
        end = _optional_non_negative_float(item.get("end"))
        if item.get("start") is not None and start is None:
            return None
        if item.get("end") is not None and end is None:
            return None
        if start is not None and end is not None and end < start:
            return None
        segments.append(
            InputSegment(
                index=index,
                start=start,
                end=end,
                text=text,
                segment_id=_optional_text(item.get("id")),
            )
        )
    return segments


def _covers_transcript(segments: list[InputSegment], transcript: str) -> bool:
    """Check lossless segment coverage while tolerating block-boundary whitespace.

    ASR exporters commonly put a newline between fixed-duration chunks in
    ``transcript.txt`` but omit it from each segment's text.  Compare each
    segment in order and permit whitespace only immediately before a segment
    boundary.  Differences inside a segment (for example ``not able`` versus
    ``notable``) still invalidate the segmented representation.
    """

    normalized_transcript = _normalize_transcript_text(transcript)
    cursor = 0
    for segment in segments:
        while cursor < len(normalized_transcript) and normalized_transcript[cursor].isspace():
            cursor += 1
        segment_text = _normalize_segment_text(segment.text)
        if not segment_text or not normalized_transcript.startswith(segment_text, cursor):
            return False
        cursor += len(segment_text)
    return not normalized_transcript[cursor:].strip()


def _normalize_transcript_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.removeprefix("\ufeff").strip()


def _normalize_segment_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.removeprefix("\ufeff").strip()


def _optional_non_negative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _duration_ms(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("duration_ms")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    for container_name in ("audio_probe", "source_probe"):
        container = metadata.get(container_name)
        if not isinstance(container, dict):
            continue
        format_data = container.get("format")
        if not isinstance(format_data, dict):
            continue
        try:
            duration = float(format_data["duration"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration >= 0:
            return int(duration * 1000)
    return None


def _asset_record(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
    }


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _segments_from_json(value: str | None) -> list[InputSegment]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    segments = _parse_segments(parsed)
    return segments or []


def _segment_identity_matches(raw: list[InputSegment], cleaned: list[Any]) -> bool:
    if len(raw) != len(cleaned):
        return False
    for index, value in enumerate(cleaned):
        if not isinstance(value, dict):
            return False
        # IDs are optional in legacy segment files; position and timing remain
        # stable identity fields in that case.
        raw_id = str(index)
        cleaned_id = str(value.get("id", index))
        if cleaned_id != raw_id and value.get("id") is not None:
            # Existing InputSegment intentionally omits IDs, so accept IDs
            # that are stable but not positional (e.g. ``segment-1``).
            if index == 0 and cleaned_id:
                continue
            if not cleaned_id:
                return False
        start = _optional_non_negative_float(value.get("start"))
        end = _optional_non_negative_float(value.get("end"))
        if raw[index].start != start or raw[index].end != end:
            return False
    return True


def _segment_ids_match(raw: Any, cleaned: Any) -> bool:
    if not isinstance(raw, list) or len(raw) != len(cleaned):
        return False
    for raw_item, cleaned_item in zip(raw, cleaned, strict=True):
        if not isinstance(raw_item, dict) or not isinstance(cleaned_item, dict):
            return False
        raw_id = raw_item.get("id")
        cleaned_id = cleaned_item.get("id")
        if raw_id is not None and str(raw_id) != str(cleaned_id):
            return False
        for key in ("start", "end"):
            if _optional_non_negative_float(raw_item.get(key)) != _optional_non_negative_float(
                cleaned_item.get(key)
            ):
                return False
    return True


def _raw_fingerprint(
    transcript: str,
    segments: list[InputSegment],
    source_kind: str,
) -> str:
    payload = json.dumps(
        {
            "text": transcript,
            "segments": [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in segments
            ],
            "source_kind": source_kind,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raw_fingerprint_payload(transcript: str, segments: Any, source_kind: str) -> str:
    payload = json.dumps(
        {"text": transcript, "segments": segments, "source_kind": source_kind},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raw_file_fingerprint(path: Path, transcript: str, source_kind: str) -> str:
    try:
        segments = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        segments = []
    # Kept separate from segment validation so a malformed file simply fails
    # the cleaned variant check and falls back to the raw transcript.
    return _raw_fingerprint_payload(transcript, segments, source_kind)


__all__ = ["InputSegment", "NoteInput", "NoteInputLoader"]
