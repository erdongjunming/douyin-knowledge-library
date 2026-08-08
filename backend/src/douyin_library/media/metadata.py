from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any


def can_delete_source_video(
    *,
    source_duration: float | None,
    audio_duration: float | None,
    audio_bytes: int,
    sha256: str,
    verified: bool,
) -> bool:
    if not verified or audio_bytes <= 0 or len(sha256) != 64:
        return False
    if source_duration is None or audio_duration is None:
        return False
    tolerance = max(0.5, source_duration * 0.02)
    return abs(source_duration - audio_duration) <= tolerance


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_audio_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    for stream in probe.get("streams", []):
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    return None


def duration_seconds(probe: dict[str, Any]) -> float | None:
    raw = (probe.get("format") or {}).get("duration")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
