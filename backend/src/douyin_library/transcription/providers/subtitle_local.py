from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Iterable
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Work
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
    r"^[\[\(（【]?\s*(?:music|song|applause|laughter|noise|silence|"
    r"sound effect|字幕|音乐|歌声|掌声|笑声|音效|无声)\s*[\]\)）】]?$",
    re.I,
)
_NOISE_ONLY = re.compile(r"^[.…·~～\-_=♪♫♬]+$")


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
            minutes, seconds = float(parts[0]), float(parts[1])
            if minutes < 0 or seconds < 0 or seconds >= 60:
                return None
            result = minutes * 60 + seconds
            return result if math.isfinite(result) else None
        if len(parts) == 3:
            hours, minutes, seconds = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
            )
            if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
                return None
            result = hours * 3600 + minutes * 60 + seconds
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
            dialogue = line.strip()
            if not dialogue.startswith("Dialogue:"):
                continue
            fields = dialogue[len("Dialogue:") :].lstrip().split(",", 9)
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
    placeholders = sum(
        bool(_PLACEHOLDER.fullmatch(text) or _NOISE_ONLY.fullmatch(text)) for text in texts
    )
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
    def __init__(
        self,
        settings: Any,
        layout: StorageLayout | None = None,
        *,
        session: Session | None = None,
        asset_paths: Callable[[Work], Iterable[Path]] | None = None,
    ) -> None:
        self.settings = settings
        self.session = session
        self.asset_paths = asset_paths
        default_layout = layout or StorageLayout(Path(settings.storage_root))
        configured_root = getattr(settings, "transcription_subtitle_root", None)
        default_root = Path(settings.storage_root) / "subtitle"
        has_explicit_root = configured_root is not None and Path(configured_root) != default_root
        self.layout = copy(default_layout) if layout is not None else default_layout
        if has_explicit_root:
            self.layout.subtitle = Path(configured_root)

    def discover(self, work: Work) -> TranscriptDraft | None:
        candidates = list(self.layout.subtitle_candidate_paths(work.aweme_id))
        if self.session is not None and getattr(work, "id", None) is not None:
            assets = self.session.scalars(
                select(Asset).where(Asset.work_id == work.id, Asset.kind == "subtitle")
            ).all()
            candidates.extend(Path(asset.path) for asset in assets if asset.path)
        if self.asset_paths is not None:
            candidates.extend(Path(path) for path in self.asset_paths(work))

        seen: set[Path] = set()
        for path in candidates:
            path = Path(path)
            if path in seen:
                continue
            seen.add(path)
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
