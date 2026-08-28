from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from test_pipeline import ScriptedProvider, _note_input

from douyin_library.notes.pipeline import SemanticNotePipeline
from douyin_library.notes.render import render_semantic_note


@pytest.mark.asyncio
async def test_render_emits_complete_evidence_linked_bundle() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )

    bundle = render_semantic_note(
        note_input,
        generation,
        generated_at="2026-08-28T10:00:00+08:00",
    )

    for heading in (
        "## 视频基本信息",
        "## 一句话总览",
        "## 完整结构",
        "## 主题复盘",
        "## 方法与步骤",
        "## 案例与反例",
        "## 前提、边界与风险",
        "## 关键概念",
        "## 可复用金句或表达",
        "## 我的理解与启发",
        "## 可执行行动",
        "## 完整逐段时间线",
        "## 推广与行动号召",
        "## 归档标签",
        "## 证据与 ASR 局限",
    ):
        assert heading in bundle.learning_note
    assert "质量状态：passed" in bundle.learning_note
    assert "[atom:" in bundle.learning_note
    assert "00:10" in bundle.learning_note
    assert "segment 0" in bundle.learning_note
    assert "segment 6" in bundle.learning_note
    assert bundle.learning_note.index("segment 0") < bundle.learning_note.index("segment 6")
    assert "关注我并领取完整模板" in bundle.learning_note
    assert "[ASR 原文表达]" in bundle.learning_note
    assert bundle.brief_json["summary"] == generation.synthesis.overview
    assert bundle.coverage_report["quality_gate"] == "passed"
    assert bundle.note_budget["transcript_chars"] == len(note_input.transcript)
    assert bundle.note_budget["segment_count"] == len(note_input.segments)
    assert bundle.note_budget["atom_count"] == len(generation.extraction.atoms)
    assert bundle.note_budget["input_truncated"] is False
    assert bundle.note_budget["batch_count"] == len(generation.batch_boundaries)
    assert bundle.note_budget["batch_boundaries"]
    assert {
        item["stage"] for item in bundle.note_budget["stage_batches"]
    } >= {"extract", "synthesize", "review", "repair"}
    assert bundle.note_budget["context_status"] == "within_budget"
    assert bundle.note_budget["provider"] == "scripted"
    assert bundle.note_budget["model"] == "complete-v1"
    assert bundle.note_budget["usage"]["input_tokens"] == generation.usage.input_tokens
    assert bundle.note_budget["repair_count"] == 1
    assert bundle.note_score["completeness"] == 100
    assert bundle.note_score["quality_gate"] == "passed"
    assert bundle.note_score["scoring_method"] == "coverage-report-v1"
    assert bundle.page_metadata["source_url"] == note_input.source_url
    assert bundle.asset_manifest["copies_original_assets"] is False


@pytest.mark.asyncio
async def test_failed_coverage_cannot_receive_perfect_completeness() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    failed_generation = replace(
        generation,
        coverage=generation.coverage.model_copy(
            update={"quality_gate": "failed", "repairable": False}
        ),
    )

    bundle = render_semantic_note(
        note_input,
        failed_generation,
        generated_at="2026-08-28T10:00:00+08:00",
    )

    assert bundle.note_score["quality_gate"] == "failed"
    assert bundle.note_score["completeness"] < 100


@pytest.mark.asyncio
async def test_total_score_penalizes_poor_organization_even_when_coverage_passes() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    poor_synthesis = generation.synthesis.model_copy(
        update={
            "structure": [],
            "concepts": [],
            "quotes": [],
            "implications": [],
            "actions": [],
            "tags": [],
        }
    )
    poor_generation = replace(generation, synthesis=poor_synthesis)

    bundle = render_semantic_note(
        note_input,
        poor_generation,
        generated_at="2026-08-28T10:00:00+08:00",
    )

    assert bundle.note_score["completeness"] == 100
    assert bundle.note_score["clarity"] < 100
    assert bundle.note_score["usefulness"] == 0
    assert bundle.note_score["score"] < 100
