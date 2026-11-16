from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from test_pipeline import ScriptedProvider, _note_input

from douyin_library.notes.contracts import SectionClaim, SectionDraft
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
        "## 术语表",
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
    assert bundle.note_budget["recommended_detail"]["level"] == "full"
    assert "step" in bundle.note_budget["recommended_detail"]["evidence_types"]
    assert bundle.note_budget["execution"]["mode"] == "local"
    assert bundle.note_budget["execution"]["is_local"] is True
    assert {
        item["stage"] for item in bundle.note_budget["stage_batches"]
    } >= {"extract", "synthesize", "review", "repair"}
    assert bundle.note_budget["context_status"] == "within_budget"
    assert bundle.note_budget["provider"] == "scripted"
    assert bundle.note_budget["model"] == "complete-v1"
    assert bundle.note_budget["usage"]["input_tokens"] == generation.usage.input_tokens
    assert bundle.note_budget["repair_count"] == 1
    assert bundle.note_score["completeness"] == 100
    assert bundle.note_score["quality_gate"] == "passed_with_warnings"
    assert "methods_steps_suspicious_missing" in bundle.note_score["warnings"]
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
async def test_failed_candidate_fills_missing_timeline_from_extraction() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    missing_index = 5
    disposition = next(
        item
        for item in generation.extraction.dispositions
        if item.segment_index == missing_index
    )
    failed_generation = replace(
        generation,
        synthesis=generation.synthesis.model_copy(
            update={
                "timeline": [
                    entry
                    for entry in generation.synthesis.timeline
                    if entry.segment_index != missing_index
                ]
            }
        ),
        coverage=generation.coverage.model_copy(
            update={"quality_gate": "failed", "repairable": False}
        ),
    )

    bundle = render_semantic_note(
        note_input,
        failed_generation,
        generated_at="2026-08-28T10:00:00+08:00",
    )

    rendered = bundle.parsed["timeline"][missing_index]
    assert rendered["disposition"] == disposition.disposition
    assert rendered["atom_ids"] == disposition.atom_ids


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


@pytest.mark.asyncio
async def test_render_includes_input_warnings_without_name_error() -> None:
    note_input = _note_input()
    note_input = replace(note_input, warnings=["segments.json 无效"])
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )

    bundle = render_semantic_note(
        note_input,
        generation,
        generated_at="2026-08-28T10:00:00+08:00",
    )

    assert "输入警告：segments.json 无效" in bundle.learning_note


@pytest.mark.asyncio
async def test_render_teaching_sections_keep_claim_evidence_and_missing_categories() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atom = generation.extraction.atoms[0]
    timestamp = atom.start if atom.start is not None else 0.0
    claim = SectionClaim(
        text="记录真实问题",
        atom_ids=[atom.id],
        timestamps=[timestamp],
    )
    draft = SectionDraft(
        section_id="methods_steps",
        title="方法与步骤",
        content="记录真实问题，再按主题整理。",
        claims=[claim],
        steps=[claim],
        examples=[],
        results=[],
        conditions=[],
        risks=[],
        actions=[claim],
        included_atom_ids=[atom.id],
        unprovided_items=[],
        generated_interpretations=["生成理解：步骤需要结合原文条件复用。"],
    )
    empty = SectionDraft(
        section_id="examples_results",
        title="案例与结果",
        content="视频未提供这一类信息。",
        included_atom_ids=[],
        unprovided_items=["example", "result"],
    )
    teaching_generation = replace(
        generation,
        section_drafts=(draft, empty),
    )

    bundle = render_semantic_note(
        note_input,
        teaching_generation,
        generated_at="2026-08-28T10:00:00+08:00",
    )

    body = bundle.learning_note
    labels = ("#### 观点", "#### 原文依据", "#### 解释", "#### 步骤", "#### 风险或行动")
    assert all(label in body for label in labels)
    assert [body.index(label) for label in labels] == sorted(body.index(label) for label in labels)
    assert f"[atom:{atom.id}]" in body
    assert "00:00" in body
    assert "生成理解：步骤需要结合原文条件复用。" in body
    assert "视频未提供这一类信息。" in body
    assert bundle.parsed["section_drafts"]
    assert bundle.parsed["semantic_complete"] is generation.coverage.semantic_complete
    assert "methods_steps" in bundle.parsed["evidence_map"]["section_ids"]
    assert "examples_results" in bundle.parsed["evidence_map"]["section_ids"]


@pytest.mark.asyncio
async def test_render_quality_and_budget_flags_fail_closed_and_keep_full_claim_map() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atom = generation.extraction.atoms[0]
    claim = SectionClaim(text="观点", atom_ids=[atom.id], timestamps=[0.0])
    draft = SectionDraft(
        section_id="overview",
        title="总览",
        content="泛化解释",
        claims=[claim],
        included_atom_ids=[atom.id],
        generated_interpretations=["生成理解：不要重复标签"],
    )
    failed = replace(
        generation,
        section_drafts=(draft,),
        coverage=generation.coverage.model_copy(
            update={
                "quality_gate": "failed",
                "semantic_complete": True,
                "depth_passed": True,
                "input_sufficient": True,
            }
        ),
    )

    bundle = render_semantic_note(
        note_input,
        failed,
        generated_at="2026-08-28T10:00:00+08:00",
    )

    assert bundle.parsed["semantic_complete"] is False
    assert bundle.note_budget["semantic_complete"] is False
    assert bundle.note_budget["teaching_outline"]
    assert bundle.note_budget["section_drafts"][0]["claims"][0]["text"] == "观点"
    assert bundle.parsed["evidence_map"]["section_drafts"][0]["claims"][0]["timestamps"] == [0.0]
    assert "生成理解：生成理解：" not in bundle.learning_note


@pytest.mark.asyncio
async def test_render_does_not_invent_timestamp_for_atom_without_source_time() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atom = generation.extraction.atoms[0].model_copy(update={"start": None, "end": None})
    extraction = generation.extraction.model_copy(
        update={"atoms": [atom, *generation.extraction.atoms[1:]]}
    )
    draft = SectionDraft(
        section_id="overview",
        title="总览",
        content="模型泛化句",
        included_atom_ids=[atom.id],
    )
    bundle = render_semantic_note(
        note_input,
        replace(generation, extraction=extraction, section_drafts=(draft,)),
        generated_at="2026-08-28T10:00:00+08:00",
    )

    assert "[atom:" + atom.id + "] 时间未知" in bundle.learning_note
    assert "生成理解：模型泛化句" in bundle.learning_note
    assert "[atom:" + atom.id + "] 00:00" not in bundle.learning_note
