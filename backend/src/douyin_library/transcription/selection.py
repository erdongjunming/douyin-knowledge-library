from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any

from douyin_library.models import Work
from douyin_library.transcription.contracts import TranscriptDraft
from douyin_library.transcription.providers.subtitle_local import score_subtitle


async def select_transcript_source(
    subtitle_provider: Any,
    asr_factory: Any,
    work: Work,
    audio_path: Path,
    *,
    duration_ms: int | None = None,
) -> TranscriptDraft:
    subtitle = subtitle_provider.discover(work)
    if (
        subtitle is not None
        and score_subtitle(
            subtitle.segments,
            duration_ms=duration_ms if duration_ms is not None else work.duration_ms,
        ).usable
    ):
        return replace(subtitle, provider=subtitle.provider or "subtitle_local")
    factory_args = (audio_path,) if len(inspect.signature(asr_factory).parameters) else ()
    provider = asr_factory(*factory_args)
    return await provider.transcribe(audio_path)


__all__ = ["select_transcript_source"]
