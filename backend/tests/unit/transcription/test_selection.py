from pathlib import Path

import pytest

from douyin_library.models import Work
from douyin_library.transcription.contracts import TranscriptDraft, TranscriptSegment
from douyin_library.transcription.selection import select_transcript_source


class Subtitle:
    def __init__(self, draft):
        self.draft = draft
        self.calls = 0

    def discover(self, work):
        self.calls += 1
        return self.draft


class ASR:
    def __init__(self):
        self.calls = 0

    async def transcribe(self, audio_path):
        self.calls += 1
        return TranscriptDraft((TranscriptSegment("a", 0, 1, "asr"),), "asr", provider="asr")


@pytest.mark.asyncio
async def test_usable_subtitle_is_selected_without_asr() -> None:
    subtitle = Subtitle(TranscriptDraft((TranscriptSegment("s", 0, 1, "真实字幕"),), "真实字幕"))
    asr = ASR()
    result = await select_transcript_source(
        subtitle, lambda: asr, Work(aweme_id="1"), Path("a.m4a")
    )
    assert result.provider == "subtitle_local"
    assert asr.calls == 0


@pytest.mark.asyncio
async def test_unusable_subtitle_calls_asr_factory() -> None:
    subtitle = Subtitle(TranscriptDraft((TranscriptSegment("s", 0, 1, "[音乐]"),), "[音乐]"))
    asr = ASR()
    result = await select_transcript_source(
        subtitle, lambda audio_path: asr, Work(aweme_id="1"), Path("a.m4a")
    )
    assert result.provider == "asr"
    assert asr.calls == 1
