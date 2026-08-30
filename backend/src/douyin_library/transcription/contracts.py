from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from douyin_library.models import Work


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    segment_id: str
    start: float | None
    end: float | None
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptDraft:
    segments: tuple[TranscriptSegment, ...]
    text: str
    language: str | None = None
    provider: str | None = None


@dataclass(frozen=True, slots=True)
class CleanedTranscript:
    segments: tuple[TranscriptSegment, ...]
    text: str
    language: str | None = None


@dataclass(frozen=True, slots=True)
class CleanerResult:
    transcript: CleanedTranscript
    usable: bool = False
    text_chars: int = 0
    segment_count: int = 0
    timestamp_valid: bool = False
    placeholder_ratio: float = 0.0


class TranscriptSourceProvider(Protocol):
    def discover(self, work: Work) -> TranscriptDraft | None: ...


class ASRProvider(Protocol):
    async def transcribe(
        self,
        audio_path: Path,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> TranscriptDraft: ...


class TranscriptPostprocessor(Protocol):
    async def clean(
        self,
        transcript: TranscriptDraft,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CleanedTranscript: ...


__all__ = [
    "ASRProvider",
    "CleanerResult",
    "CleanedTranscript",
    "TranscriptDraft",
    "TranscriptPostprocessor",
    "TranscriptSegment",
    "TranscriptSourceProvider",
]
