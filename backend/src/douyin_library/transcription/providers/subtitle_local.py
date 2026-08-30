from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from douyin_library.media.storage import StorageLayout
from douyin_library.models import Work
from douyin_library.transcription.contracts import TranscriptDraft, TranscriptSegment


@dataclass(frozen=True, slots=True)
class SubtitleQuality:
    text_chars: int
    segment_count: int
    timestamp_valid: bool
    placeholder_ratio: float
    usable: bool
    reason: str


_TIMING = re.compile(r"^\s*(\S+)\s+-->\s+(\S+)(?:\s+.*)?$")
_TAG = re.compile(r"<[^>]*>|\{\\[^}]*\}")
_PLACEHOLDER = re.compile(
    r"^(?:\[?(?:music|song|applause|laughter|noise|silence|音乐|歌声|掌声|笑声|音效|无声)\]?)$",
    re.I,
)


def _clean_text(value: str) -> str:
    value = html.unescape(value.replace("\\N", "\n").replace("\\n", "\n"))
    value = _TAG.sub("", value)
    return "\n".join(part.strip() for part in value.splitlines() if part.strip()).strip()


def normalize_subtitle_text(raw_text: str, format_hint: str | None = None) -> str:
    return "\n".join(segment.text for segment in parse_subtitle(raw_text, format_hint))


def _seconds(value: str) -> float | None:
    value = value.replace(",", ".")
    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            result = float(minutes) * 60 + float(seconds)
            return result if math.isfinite(result) else None
        if len(parts) == 3:
            hours, minutes, seconds = parts
            result = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            return result if math.isfinite(result) else None
    except ValueError:
        return None
    return None


def parse_subtitle(raw_text: str, format_hint: str | None = None) -> tuple[TranscriptSegment, ...]:
    hint = (format_hint or "").lower()
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if hint in {".ass", ".ssa", "ass", "ssa"} or any(
        line.startswith("Dialogue:") for line in lines
    ):
        output: list[TranscriptSegment] = []
        for line in lines:
            if not line.startswith("Dialogue:"):
                continue
            fields = line[len("Dialogue:") :].lstrip().split(",", 9)
            if len(fields) < 10:
                output.append(TranscriptSegment(f"segment-{len(output) + 1}", -1, -1, ""))
                continue
            start, end = _seconds(fields[1].strip()), _seconds(fields[2].strip())
            if start is None or end is None:
                start, end = -1, -1
            output.append(
                TranscriptSegment(f"segment-{len(output) + 1}", start, end, _clean_text(fields[9]))
            )
        return _dedupe(output)

    output = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper() == "WEBVTT" or line.startswith("NOTE"):
            i += 1
            continue
        timing = _TIMING.match(line)
        if timing or "-->" in line:
            start = _seconds(timing.group(1)) if timing else -1
            end = _seconds(timing.group(2)) if timing else -1
            if start is None or end is None:
                start, end = -1, -1
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i])
                i += 1
            output.append(
                TranscriptSegment(
                    f"segment-{len(output) + 1}", start, end, _clean_text("\n".join(text_lines))
                )
            )
            continue
        if hint in {".srt", "srt", ".vtt", "vtt"} and (
            line.isdigit() or line.lower().startswith("language:")
        ):
            i += 1
            continue
        if hint in {".txt", ".md", "txt", "md"}:
            text = _clean_text(line)
            if text:
                output.append(TranscriptSegment(f"segment-{len(output) + 1}", None, None, text))
        i += 1
    return _dedupe(output)


def _dedupe(segments: list[TranscriptSegment]) -> tuple[TranscriptSegment, ...]:
    result = []
    previous = None
    for segment in segments:
        if segment.text and segment.text != previous:
            result.append(
                TranscriptSegment(
                    f"segment-{len(result) + 1}", segment.start, segment.end, segment.text
                )
            )
            previous = segment.text
    return tuple(result)


def score_subtitle(
    segments: tuple[TranscriptSegment, ...], *, duration_ms: int | None = None
) -> SubtitleQuality:
    texts = [s.text.strip() for s in segments if s.text.strip()]
    chars = len("".join(texts))
    timestamped = [s for s in segments if s.start is not None or s.end is not None]
    has_timestamps = bool(timestamped)
    has_untimed = any(s.start is None and s.end is None for s in segments)
    timestamp_valid = all(
        s.start is not None
        and s.end is not None
        and math.isfinite(s.start)
        and math.isfinite(s.end)
        and s.start >= 0
        and s.end >= s.start
        for s in timestamped
    )
    if has_timestamps and has_untimed:
        timestamp_valid = False
    previous_start = None
    for segment in timestamped:
        if (
            previous_start is not None
            and segment.start is not None
            and segment.start < previous_start
        ):
            timestamp_valid = False
            break
        previous_start = segment.start
    placeholders = sum(bool(_PLACEHOLDER.fullmatch(text)) for text in texts)
    ratio = placeholders / len(texts) if texts else 0.0
    if not texts:
        reason = "empty_text"
    elif not segments:
        reason = "no_segments"
    elif has_timestamps and has_untimed:
        reason = "mixed_timestamp"
    elif not timestamp_valid:
        reason = "invalid_timestamp"
    elif ratio >= 0.5:
        reason = "placeholder_or_noise"
    elif chars < (
        1
        if duration_ms is not None and duration_ms <= 10_000
        else max(2, (duration_ms or 0) // 30_000)
    ):
        reason = "text_too_sparse_for_duration"
    else:
        reason = "usable"
    return SubtitleQuality(chars, len(segments), timestamp_valid, ratio, reason == "usable", reason)


class LocalSubtitleProvider:
    def __init__(self, settings: Any, layout: StorageLayout | None = None) -> None:
        self.settings = settings
        default_layout = layout or StorageLayout(Path(settings.storage_root))
        configured_root = getattr(settings, "transcription_subtitle_root", None)
        if layout is not None or configured_root is None:
            self.layout = default_layout
        else:
            self.layout = default_layout
            self.layout.subtitle = Path(configured_root)

    def discover(self, work: Work) -> TranscriptDraft | None:
        for path in self.layout.subtitle_candidate_paths(work.aweme_id):
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError):
                continue
            segments = parse_subtitle(raw, path.suffix)
            if score_subtitle(segments, duration_ms=work.duration_ms).usable:
                return TranscriptDraft(
                    segments, normalize_subtitle_text(raw, path.suffix), provider="subtitle_local"
                )
        return None


__all__ = [
    "LocalSubtitleProvider",
    "SubtitleQuality",
    "normalize_subtitle_text",
    "parse_subtitle",
    "score_subtitle",
]
