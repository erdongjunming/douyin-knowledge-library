from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from douyin_library.media.storage import StorageLayout
from douyin_library.transcription.contracts import (
    ASRProvider,
    CleanerResult,
    CleanedTranscript,
    TranscriptDraft,
    TranscriptPostprocessor,
    TranscriptSegment,
    TranscriptSourceProvider,
)


def test_transcript_value_objects_are_frozen_slots_dataclasses() -> None:
    segment = TranscriptSegment("segment-1", 0.0, 1.2, "Hello")
    draft = TranscriptDraft(segments=(segment,), text="Hello")
    cleaned = CleanedTranscript(segments=(segment,), text="Hello")
    result = CleanerResult(transcript=cleaned, usable=True)

    for value in (segment, draft, cleaned, result):
        params = value.__dataclass_params__
        assert is_dataclass(value)
        assert params.frozen is True
        assert hasattr(value, "__slots__")
        with pytest.raises(FrozenInstanceError):
            setattr(value, next(iter(value.__slots__)), "changed")


def test_provider_contracts_expose_expected_async_and_sync_methods() -> None:
    assert "discover" in TranscriptSourceProvider.__dict__
    assert "transcribe" in ASRProvider.__dict__
    assert "clean" in TranscriptPostprocessor.__dict__
    assert inspect.iscoroutinefunction(ASRProvider.transcribe)
    assert inspect.iscoroutinefunction(TranscriptPostprocessor.clean)


def test_storage_layout_exposes_subtitle_and_cleaned_transcript_paths(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(tmp_path)

    assert layout.subtitle == tmp_path / "subtitle"
    assert layout.subtitle_dir("abc") == tmp_path / "subtitle" / "abc"
    assert layout.subtitle_candidate_paths("abc") == tuple(
        tmp_path / "subtitle" / "abc" / f"subtitle{extension}"
        for extension in (".srt", ".vtt", ".ass", ".ssa", ".txt")
    )
    assert layout.transcript_cleaned_text_path("abc") == (
        tmp_path / "transcript" / "abc" / "transcript.cleaned.md"
    )
    assert layout.transcript_cleaned_json_path("abc") == (
        tmp_path / "transcript" / "abc" / "transcript.cleaned.json"
    )

    layout.initialize()
    assert layout.subtitle.is_dir()
