from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from douyin_library.media.storage import StorageLayout
from douyin_library.notes.input import NoteInputLoader
from douyin_library.notes.pipeline import SemanticNotePipeline
from douyin_library.notes.render import evidence_map_from_generation, render_semantic_note


def _seed_layout(layout: StorageLayout, aweme_id: str) -> None:
    layout.initialize()
    layout.transcript_dir(aweme_id).mkdir(parents=True, exist_ok=True)
    layout.transcript_text_path(aweme_id).write_text(
        "第一步先看案例，条件是预算低于100元。第二步记录姜学长的原话：\"先做小实验\"。",
        encoding="utf-8",
    )
    layout.transcript_segments_path(aweme_id).write_text(
        json.dumps(
            [
                {"id": "s1", "start": 0, "end": 4, "text": "第一步先看案例，条件是预算低于100元。"},
                {
                    "id": "s2",
                    "start": 4,
                    "end": 8,
                    "text": "第二步记录姜学长的原话：\"先做小实验\"。",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_loader_prefers_cleaned_only_when_parent_fingerprint_and_segments_match(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    layout = StorageLayout(root)
    _seed_layout(layout, "evidence-1")
    layout.transcript_cleaned_text_path("evidence-1").write_text("清洗后的内容", encoding="utf-8")
    layout.transcript_cleaned_json_path("evidence-1").write_text(
        json.dumps(
            {
                "raw_fingerprint": "wrong",
                "segments": [{"id": "s1", "start": 0, "end": 4, "text": "清洗后的内容"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    note_input = NoteInputLoader(layout).load("evidence-1")
    assert note_input.transcript_variant == "raw"
    assert note_input.cleaning_fallback


def test_evidence_map_retains_traceable_atom_details() -> None:
    from test_pipeline import ScriptedProvider, _note_input

    note_input = _note_input()
    generation = asyncio.run(
        SemanticNotePipeline(ScriptedProvider()).generate(note_input, asyncio.Event())
    )
    evidence = evidence_map_from_generation(note_input, generation)
    assert evidence["items"]
    assert all(item["source_segment"] >= 0 for item in evidence["items"])
    assert all("start" in item and "end" in item for item in evidence["items"])
    assert {item["kind"] for item in evidence["items"]} >= {"example", "condition", "step"}
    assert all("asr_uncertainty" in item for item in evidence["items"])


def test_jiang_fixture_keeps_examples_conditions_steps_and_historical_note() -> None:
    from test_pipeline import ScriptedProvider, _note_input

    source = _note_input()
    texts = [
        "姜学长分享表达方法。",
        "第一步先记录问题。",
        "第二步整理案例，预算100元。",
        "第三步复述验证。",
        "案例：姜学长用小实验验证结论。",
        "条件：只有每周复盘才适用，ASR 人名可能误听。",
        "关注我领取模板。",
    ]
    segments = [
        replace(segment, text=texts[index])
        for index, segment in enumerate(source.segments)
    ]
    fixture = replace(source, transcript="".join(texts), segments=segments)
    generation = asyncio.run(
        SemanticNotePipeline(ScriptedProvider()).generate(fixture, asyncio.Event())
    )
    evidence = evidence_map_from_generation(fixture, generation)
    rendered = render_semantic_note(
        fixture, generation, generated_at="2026-08-31T00:00:00Z"
    ).learning_note
    assert any(
        item["kind"] == "example" and "案例" in item["source_excerpt"]
        for item in evidence["items"]
    )
    assert any(
        item["kind"] == "condition" and "每周复盘" in item["statement"]
        for item in evidence["items"]
    )
    assert sum(item["kind"] == "step" for item in evidence["items"]) == 3
    assert "100元" in rendered
    assert "姜学长" in rendered
    assert "每周复盘" in rendered
    assert "第一步" in rendered


def test_coverage_rejects_omitted_latin_named_entities() -> None:
    from test_coverage import _complete_extraction, _review, _synthesis

    extraction = _complete_extraction()
    named = extraction.atoms[0].model_copy(
        update={
            "id": "named",
            "kind": "claim",
            "statement": "Alice and OpenAI tested this",
            "source_excerpt": "Alice and OpenAI tested this",
        }
    )
    extraction = extraction.model_copy(
        update={
            "atoms": [*extraction.atoms, named],
            "dispositions": [
                extraction.dispositions[0].model_copy(
                    update={"atom_ids": [*extraction.dispositions[0].atom_ids, "named"]}
                ),
                *extraction.dispositions[1:],
            ],
        }
    )
    from douyin_library.notes.coverage import CoverageValidator

    omitted = _synthesis(
        [atom.id for atom in extraction.atoms if atom.id != "named"], extraction=extraction
    )
    assert CoverageValidator().validate(
        segment_count=4,
        extraction=extraction,
        synthesis=omitted,
        review=_review(range(4)),
        semantic_generator="test",
    ).quality_gate == "failed"

    preserved = _synthesis([atom.id for atom in extraction.atoms], extraction=extraction)
    preserved.themes[0].content = "Alice and OpenAI tested this"
    assert CoverageValidator().validate(
        segment_count=4,
        extraction=extraction,
        synthesis=preserved,
        review=_review(range(4)),
        semantic_generator="test",
    ).quality_gate == "passed"
