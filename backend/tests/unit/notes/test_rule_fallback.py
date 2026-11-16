from __future__ import annotations

from douyin_library.notes.generator import generate_rule_note
from douyin_library.notes.input import InputSegment, NoteInput

NOW = "2026-08-28T10:00:00+08:00"


def _segments() -> list[InputSegment]:
    return [
        InputSegment(
            index=index,
            start=float(index),
            end=float(index + 1),
            text=f"源分段 {index:02d} 的完整忠实内容。",
        )
        for index in range(15)
    ]


def test_rule_fallback_accepts_note_input_and_contains_every_segment_in_source_order() -> None:
    segments = _segments()
    note_input = NoteInput(
        aweme_id="fallback-731",
        transcript="".join(segment.text for segment in segments),
        segments=segments,
        metadata={},
        description="完整 fallback",
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=15_000,
        source_url="https://www.douyin.com/video/fallback-731",
        source_files={},
        asset_records={},
        warnings=[],
    )

    bundle = generate_rule_note(note_input, source_files={}, generated_at=NOW)

    assert [item["segment_index"] for item in bundle.parsed["timeline"]] == list(range(15))
    assert [item["content"] for item in bundle.parsed["timeline"]] == [
        segment.text for segment in segments
    ]
    assert bundle.parsed["quality_gate"] == "fallback"
    assert bundle.note_score["quality_gate"] == "fallback"
    assert bundle.note_budget["quality_gate"] == "fallback"
    assert bundle.coverage_report["quality_gate"] == "fallback"
    assert bundle.parsed["semantic_complete"] is False
    assert bundle.parsed["input_sufficient"] is False
    assert bundle.parsed["section_drafts"] == []
    assert bundle.parsed["evidence_map"]["section_ids"]
    assert bundle.page_metadata["semantic_complete"] is False
    assert bundle.note_budget["semantic_complete"] is False
    assert "未通过语义完整性审查" in bundle.learning_note
    assert "semantic coverage" not in bundle.learning_note.casefold()


def test_rule_fallback_keeps_existing_dict_call_compatible() -> None:
    segments = _segments()
    bundle = generate_rule_note(
        {
            "aweme_id": "legacy-dict",
            "transcript": "".join(segment.text for segment in segments),
            "segments": [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in segments
            ],
            "description": "旧调用",
            "duration_seconds": 15,
            "source_files": {},
            "asset_records": {},
        },
        source_files={},
        generated_at=NOW,
    )

    assert len(bundle.parsed["timeline"]) == 15
    assert "## 核心观点" in bundle.learning_note
    assert "## 可执行行动" in bundle.learning_note


def test_rule_fallback_details_preserve_every_sentence_in_each_segment() -> None:
    segment_text = "第一句保留原文。第二句包含限定条件。第三句是具体例子。"
    segment = InputSegment(
        index=0,
        start=0.0,
        end=6.0,
        text=segment_text,
    )
    note_input = NoteInput(
        aweme_id="fallback-detail-731",
        transcript=segment_text,
        segments=[segment],
        metadata={},
        description="完整详细内容",
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=6_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )

    bundle = generate_rule_note(note_input, source_files={}, generated_at=NOW)

    assert bundle.parsed["details"] == [
        {"start": 0.0, "end": 6.0, "text": segment_text}
    ]
