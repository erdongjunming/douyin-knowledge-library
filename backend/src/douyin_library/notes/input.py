from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from douyin_library.media.storage import StorageLayout

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class InputSegment:
    index: int
    start: float | None
    end: float | None
    text: str


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


class NoteInputLoader:
    def __init__(self, layout: StorageLayout) -> None:
        self.layout = layout

    def load(self, aweme_id: str) -> NoteInput:
        transcript_path = self.layout.transcript_text_path(aweme_id)
        transcript = transcript_path.read_text(encoding="utf-8-sig")
        warnings: list[str] = []
        metadata = self._load_metadata(aweme_id, warnings)
        segments = self._load_segments(aweme_id, transcript, warnings)
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
        )

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
        }


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
        segments.append(InputSegment(index=index, start=start, end=end, text=text))
    return segments


def _covers_transcript(segments: list[InputSegment], transcript: str) -> bool:
    joined = "".join(segment.text for segment in segments)
    return _WHITESPACE.sub("", joined) == _WHITESPACE.sub("", transcript)


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


__all__ = ["InputSegment", "NoteInput", "NoteInputLoader"]
