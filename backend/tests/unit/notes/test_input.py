from __future__ import annotations

import json
from pathlib import Path

from douyin_library.media.storage import StorageLayout
from douyin_library.notes.input import NoteInputLoader


def _seed_source(layout: StorageLayout, aweme_id: str, transcript: str) -> None:
    layout.initialize()
    layout.transcript_dir(aweme_id).mkdir(parents=True)
    layout.transcript_text_path(aweme_id).write_text(transcript, encoding="utf-8")
    layout.audio_path(aweme_id).write_bytes(b"audio-source")
    layout.cover_path(aweme_id).write_bytes(b"cover-source")


def test_load_preserves_complete_transcript_segments_metadata_and_assets(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(tmp_path / "library")
    transcript = "第一步记录问题。第二步整理案例。第三步复述验证。"
    _seed_source(layout, "731", transcript)
    segments = [
        {"start": 0.0, "end": 2.0, "text": "第一步记录问题。"},
        {"start": 2.0, "end": 4.0, "text": "第二步整理案例。"},
        {"start": 4.0, "end": 6.0, "text": "第三步复述验证。"},
    ]
    layout.transcript_segments_path("731").write_text(
        json.dumps(segments, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "description": "三步表达法",
        "author": "知识作者",
        "author_unique_id": "teacher-731",
        "published_at": "2026-08-28T08:00:00+08:00",
        "duration_ms": 6000,
        "source_url": "https://www.douyin.com/video/731",
    }
    layout.metadata_path("731").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    before = {
        path: path.read_bytes()
        for path in (
            layout.transcript_text_path("731"),
            layout.transcript_segments_path("731"),
            layout.metadata_path("731"),
            layout.audio_path("731"),
            layout.cover_path("731"),
        )
    }

    note_input = NoteInputLoader(layout).load("731")

    assert note_input.aweme_id == "731"
    assert note_input.transcript == transcript
    assert [segment.index for segment in note_input.segments] == [0, 1, 2]
    assert "".join(segment.text for segment in note_input.segments) == transcript
    assert note_input.metadata == metadata
    assert note_input.description == "三步表达法"
    assert note_input.author == "知识作者"
    assert note_input.author_unique_id == "teacher-731"
    assert note_input.published_at == "2026-08-28T08:00:00+08:00"
    assert note_input.duration_ms == 6000
    assert note_input.source_url == "https://www.douyin.com/video/731"
    assert note_input.source_files["transcript_txt"].endswith("transcript.txt")
    assert note_input.asset_records["audio"]["bytes"] == len(b"audio-source")
    assert note_input.warnings == []
    assert all(path.read_bytes() == content for path, content in before.items())


def test_invalid_segments_fall_back_to_one_complete_synthetic_segment(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(tmp_path / "library")
    transcript = "完整正文绝对不能被静默裁剪，哪怕分段文件没有覆盖最后这句话。"
    _seed_source(layout, "732", transcript)
    layout.transcript_segments_path("732").write_text(
        json.dumps([{"start": 0, "end": 1, "text": "完整正文"}], ensure_ascii=False),
        encoding="utf-8",
    )
    layout.metadata_path("732").write_text("{broken", encoding="utf-8")

    note_input = NoteInputLoader(layout).load("732")

    assert note_input.transcript == transcript
    assert len(note_input.segments) == 1
    assert note_input.segments[0].index == 0
    assert note_input.segments[0].text == transcript
    assert note_input.metadata == {}
    assert any(
        "segments.json" in warning and "完整 transcript" in warning
        for warning in note_input.warnings
    )
    assert any("metadata" in warning for warning in note_input.warnings)


def test_non_finite_probe_duration_is_ignored_without_breaking_load(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / "library")
    transcript = "完整转写。"
    _seed_source(layout, "733", transcript)
    layout.transcript_segments_path("733").write_text(
        json.dumps([{"start": 0, "end": 1, "text": transcript}], ensure_ascii=False),
        encoding="utf-8",
    )
    layout.metadata_path("733").write_text(
        json.dumps({"audio_probe": {"format": {"duration": float("inf")}}}),
        encoding="utf-8",
    )

    note_input = NoteInputLoader(layout).load("733")

    assert note_input.duration_ms is None
