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
    provider = _build_asr_provider(asr_factory, audio_path)
    return await provider.transcribe(audio_path)


def _build_asr_provider(factory: Any, audio_path: Path) -> Any:
    """Call factories with their declared audio-path calling convention."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(audio_path=audio_path)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        return factory()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return factory(audio_path=audio_path)
    keyword_only = [
        parameter for parameter in parameters if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    if len(keyword_only) == 1:
        return factory(**{keyword_only[0].name: audio_path})
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return factory(audio_path)
    try:
        signature.bind(audio_path)
    except TypeError:
        signature.bind(audio_path=audio_path)
        return factory(audio_path=audio_path)
    return factory(audio_path)


__all__ = ["select_transcript_source"]
