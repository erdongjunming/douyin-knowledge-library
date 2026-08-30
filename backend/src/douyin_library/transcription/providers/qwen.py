"""Qwen3-ASR provider adapter.

The subprocess runner remains responsible for FFmpeg conversion and process
management.  The service owns attempt paths, validation, publication, and
fencing; this adapter only supplies the configured runner behind the provider
boundary.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from douyin_library.config import Settings
from douyin_library.transcription.service import (
    QwenAsrSubprocess,
    SubprocessResult,
)


class QwenAsrProvider:
    """Configured Qwen subprocess exposed through the transcription protocol."""

    def __init__(self, runner: QwenAsrSubprocess) -> None:
        self.runner = runner

    @classmethod
    def from_settings(cls, settings: Settings) -> QwenAsrProvider:
        return cls(QwenAsrSubprocess.from_settings(settings))

    async def run(
        self,
        *,
        audio_path: Path,
        result_path: Path,
        cancel_event: asyncio.Event | None = None,
    ) -> SubprocessResult:
        """Run Qwen using the service-provided attempt-specific output path."""

        return await self.runner.run(
            audio_path=audio_path,
            result_path=result_path,
            cancel_event=cancel_event,
        )


__all__ = ["QwenAsrProvider"]
