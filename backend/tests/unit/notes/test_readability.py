from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_pipeline import ScriptedProvider, _note_input

from douyin_library.notes.contracts import SectionClaim, SectionDraft
from douyin_library.notes.pipeline import SemanticNotePipeline
from douyin_library.notes.prompts import SYNTHESIZE_SYSTEM_PROMPT
from douyin_library.notes.render import render_semantic_note


@pytest.mark.parametrize(
    "source, missing, expected",
    [
        ("第一步。第二步。然后。最后。", True, True),
        ("首先。接着。比如。例如。", True, True),
        ("第一步。第二步。最后。", True, False),
        ("首先。接着。之后。最后。", False, False),
    ],
)
def test_methods_missing_gate_counts_sequence_signals(source, missing, expected) -> None:
    from douyin_library.notes.coverage import _methods_section_suspicious
    from douyin_library.notes.pipeline import _empty_teaching_outline

    outline = _empty_teaching_outline()
    methods = next(section for section in outline.sections if section.section_id == "methods_steps")
    methods.atom_ids = [] if missing else ["a1"]
    assert _methods_section_suspicious(outline, source) is expected


@pytest.mark.asyncio
async def test_methods_warning_preserves_evidence_gate_and_bundle_consistency() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    note_input = replace(note_input, transcript="首先。接着。之后。最后。")
    bundle = render_semantic_note(note_input, generation, generated_at="2026-11-16")
    assert generation.coverage.quality_gate == "passed"
    assert bundle.coverage_report["quality_gate"] == "passed"
    for artifact in (
        bundle.parsed,
        bundle.note_budget,
        bundle.note_score,
        bundle.brief_json,
        bundle.page_metadata,
        bundle.asset_manifest,
    ):
        assert artifact["quality_gate"] == "passed_with_warnings"
    assert "质量状态：passed_with_warnings" in bundle.learning_note
    assert "methods_steps_suspicious_missing" in bundle.note_score["warnings"]


@pytest.mark.parametrize(
    "missing_count, points, chars, atoms, warning",
    [
        (2, ["术语", "记录问题", "切换场景", "检查结果", "使用工具"], 900, 1, None),
        (3, ["记录问题"], 900, 1, "readability_empty_sections"),
        (0, ["术语", "名词", "记录问题", "切换场景", "检查结果"], 900, 1, "readability_fragments"),
        (0, ["这是一个完整的长标题", "进行整理"], 901, 1, "readability_bloat"),
        (0, [], 1800, 2, None),
        (0, [], 1, 0, "readability_bloat"),
    ],
)
def test_readability_thresholds(missing_count, points, chars, atoms, warning) -> None:
    from douyin_library.notes.coverage import readability_report

    prefix = "视频未提供这一类信息。" * missing_count
    report = readability_report(prefix.ljust(chars, "x"), points, atoms)
    assert report["warnings"] == ([warning] if warning else [])
    assert report["score"] == (67 if warning else 100)


@pytest.mark.asyncio
async def test_readability_warning_does_not_mask_failed_evidence() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    draft = SectionDraft(
        section_id="core_points",
        title="核心观点",
        content="概念",
        claims=[
            SectionClaim(
                text="概念", atom_ids=[generation.extraction.atoms[0].id], timestamps=[0.0]
            )
        ],
        included_atom_ids=[generation.extraction.atoms[0].id],
    )
    for gate in ("passed", "failed"):
        candidate = replace(
            generation,
            section_drafts=(draft,),
            coverage=generation.coverage.model_copy(update={"quality_gate": gate}),
        )
        bundle = render_semantic_note(note_input, candidate, generated_at="2026-11-16")
        expected = "passed_with_warnings" if gate == "passed" else "failed"
        assert bundle.note_score["quality_gate"] == expected
        assert bundle.note_score["readability"] < 100
        assert (
            f"质量状态：{expected}（可读性 {bundle.note_score['readability']} 分）"
            in bundle.learning_note
        )
        assert bundle.note_score["note_chars"] == len(bundle.learning_note)


@pytest.mark.asyncio
async def test_readability_header_converges_at_character_threshold(monkeypatch) -> None:
    from douyin_library.notes import render

    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    bundle = render_semantic_note(note_input, generation, generated_at="2026-11-16")
    padding = (
        900 * len({atom.id for atom in generation.extraction.atoms}) + 1 - len(bundle.learning_note)
    )
    generation = replace(
        generation,
        synthesis=generation.synthesis.model_copy(
            update={"overview": generation.synthesis.overview + "x" * padding}
        ),
    )
    report = render.readability_report
    calls = 0

    def bounded_report(*args):
        nonlocal calls
        calls += 1
        assert calls <= 4, "quality header oscillates at the bloat threshold"
        return report(*args)

    monkeypatch.setattr(render, "readability_report", bounded_report)
    bundle = render_semantic_note(note_input, generation, generated_at="2026-11-16")
    assert bundle.note_score["readability_metrics"]["chars_per_atom"] > 900
    assert "readability_bloat" in bundle.note_score["warnings"]


@pytest.mark.asyncio
async def test_deduplication_preserves_inline_literals_headings_and_overlapping_atoms() -> None:
    from douyin_library.notes.render import _deduplicate_excerpts

    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atoms = [
        generation.extraction.atoms[0].model_copy(
            update={"source_excerpt": "100元", "kind": "number"}
        ),
        generation.extraction.atoms[1].model_copy(update={"source_excerpt": "只用100元完成测试。"}),
    ]
    generation = replace(
        generation, extraction=generation.extraction.model_copy(update={"atoms": atoms})
    )
    lines = [
        "# 100元实验",
        "## 一句话总览",
        "只用100元完成测试。",
        "- 案例：用100元测试页面。",
        "- 只用100元完成测试。",
        "## 完整逐段时间线",
        "只用100元完成测试。",
        "另一次也用100元。",
    ]
    rendered = "\n".join(
        _deduplicate_excerpts(
            lines,
            generation,
            [
                {"content": "只用100元完成测试。"},
                {"content": "另一次也用100元。"},
            ],
        )
    )
    assert "# 100元实验" in rendered
    assert "- 案例：用100元测试页面。" in rendered
    assert rendered.count("只用100元完成测试。") == 2


@pytest.mark.asyncio
async def test_deduplication_handles_generated_interpretation_sentence_lists() -> None:
    from douyin_library.notes.render import _deduplicate_excerpts

    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atom = generation.extraction.atoms[0]
    lines = [
        atom.source_excerpt,
        f"- 生成理解：{atom.source_excerpt}；还有后续内容。",
        f"- 原文：{atom.source_excerpt} [atom:{atom.id}] 00:00",
    ]
    result = "\n".join(_deduplicate_excerpts(lines, generation, [{"content": atom.source_excerpt}]))
    assert result.count(atom.source_excerpt) == 1
    assert "还有后续内容。" in result


def test_reference_only_rows_compact_without_losing_ids_or_timestamps() -> None:
    from douyin_library.notes.render import _compact_reference_groups

    lines = [
        "#### 原文依据",
        "",
        "- 原文：[atom:a] [atom:a] 00:00-00:05",
        "- 原文：[atom:b] [atom:b] 00:05-00:10",
        "#### 步骤",
        "",
        "- 记录问题 [atom:a]",
    ]
    result = _compact_reference_groups(lines)
    assert len(result) == len(lines) - 1
    assert result[2].count("[atom:a]") == 1
    assert "[atom:b] 00:05-00:10" in result[2]
    assert result[-1] == lines[-1]


def test_upgrade_job_accepts_warning_only_when_semantically_complete() -> None:
    from types import SimpleNamespace

    from douyin_library.jobs.handlers import JobOutcomeError, validate_handler_outcome
    from douyin_library.notes.service import GenerationResult

    job = SimpleNamespace(kind="generate_note", payload_json='{"legacy_only": true}')
    outcome = GenerationResult(
        aweme_id="731",
        status="generated",
        output_dir="notes/731",
        provider="ollama",
        quality_gate="passed_with_warnings",
        semantic_complete=True,
    )
    validate_handler_outcome(job, outcome)
    with pytest.raises(JobOutcomeError, match="semantic_incomplete"):
        validate_handler_outcome(job, replace(outcome, semantic_complete=False))


@pytest.mark.asyncio
async def test_warning_bundle_publishes_with_audit_and_stays_semantic(tmp_path: Path) -> None:
    from sqlalchemy.orm import Session
    from test_service_semantic import _seed, _TeachingProvider

    from douyin_library.config import Settings
    from douyin_library.notes.service import NoteGenerationService
    from douyin_library.notes.state import classify_note_bundle

    (engine, _), layout = _seed(tmp_path / "library")
    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=layout.root),
            provider_factory=lambda name: _TeachingProvider(),
        )
        result = await service.generate_one(
            "731", provider_policy="local", cancel_event=asyncio.Event()
        )
    assert result.status == "generated"
    assert result.quality_gate == "passed_with_warnings"
    audit = json.loads(
        (layout.note_dir("731") / "generation_audit.json").read_text(encoding="utf-8")
    )
    assert audit["quality_gate"] == "passed_with_warnings"
    assert "readability_empty_sections" in audit["warnings"]
    assert audit["coverage"]["quality_gate"] == "passed"
    assert classify_note_bundle(layout.note_dir("731")).quality_status == "semantic_complete"
    engine.dispose()


@pytest.mark.asyncio
async def test_glossary_collects_concepts_and_repeated_terms_without_stopwords() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atoms = [
        atom.model_copy(update={"source_excerpt": text})
        for atom, text in zip(
            generation.extraction.atoms,
            ["舒尔特方格可以练习专注。", "使用舒尔特方格，然后进行复盘。", "这个这个然后然后"],
            strict=False,
        )
    ]
    generation = replace(
        generation, extraction=generation.extraction.model_copy(update={"atoms": atoms})
    )
    bundle = render_semantic_note(note_input, generation, generated_at="2026-11-16")
    glossary = bundle.learning_note.split("## 术语表\n", 1)[1].split("\n## ", 1)[0]
    assert "舒尔特方格" in glossary
    assert generation.synthesis.concepts[0] in glossary
    assert "这个" not in glossary
    assert "然后" not in glossary


@pytest.mark.asyncio
async def test_asr_doubts_include_marked_atoms_and_unintelligible_segments() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atom = generation.extraction.atoms[0].model_copy(update={"asr_uncertainty": ["专名未确认"]})
    disposition = generation.extraction.dispositions[1].model_copy(
        update={"disposition": "unintelligible_asr", "atom_ids": []}
    )
    generation = replace(
        generation,
        extraction=generation.extraction.model_copy(
            update={
                "atoms": [atom, *generation.extraction.atoms[1:]],
                "dispositions": [
                    generation.extraction.dispositions[0],
                    disposition,
                    *generation.extraction.dispositions[2:],
                ],
            }
        ),
    )
    bundle = render_semantic_note(note_input, generation, generated_at="2026-11-16")
    doubts = bundle.learning_note.split("## 转写疑点清单\n", 1)[1].split("\n## ", 1)[0]
    assert f"00:00-00:10 · {atom.source_excerpt}" in doubts
    assert f"00:10-00:20 · {note_input.segments[1].text}" in doubts
    assert "专名未确认" in doubts
    assert bundle.learning_note.count(atom.source_excerpt) <= 2
    assert "- 未确认内容：" not in bundle.learning_note


def test_synthesis_uses_timeline_for_full_evidence_and_short_theme_summaries() -> None:
    assert (
        "每个纳入 atom 必须出现在源顺序时间线中（时间线是全量事实的载体）。"
        in SYNTHESIZE_SYSTEM_PROMPT
    )
    assert (
        "主题结构只做归类与提炼：每个主题项给出 atom_ids、timestamps 和一句不超过 30 字的概括，"
        in SYNTHESIZE_SYSTEM_PROMPT
    )
    assert "不得在主题结构中重复 atom 的完整原文。" in SYNTHESIZE_SYSTEM_PROMPT


def test_outline_and_section_require_methods_when_the_source_contains_them() -> None:
    from douyin_library.notes.prompts import (
        OUTLINE_PROMPT_VERSION,
        OUTLINE_SYSTEM_PROMPT,
        SECTION_PROMPT_VERSION,
        SECTION_SYSTEM_PROMPT,
    )

    assert "有序流程、做法、'怎么做'类内容必须分入 methods_steps；" in OUTLINE_SYSTEM_PROMPT
    assert "只有当源材料完全没有此类内容时才允许使用缺失文案。" in OUTLINE_SYSTEM_PROMPT
    assert (
        "unprovided_items 仅允许在源材料确实没有对应类别时使用；"
        "源材料含有序步骤或做法时不得声明缺失。" in SECTION_SYSTEM_PROMPT
    )
    assert OUTLINE_PROMPT_VERSION == "teaching-outline-v2"
    assert SECTION_PROMPT_VERSION == "teaching-section-v2"


@pytest.mark.asyncio
async def test_render_limits_full_excerpts_and_combines_theme_review() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    bundle = render_semantic_note(note_input, generation, generated_at="2026-11-16")

    for atom in generation.extraction.atoms:
        assert bundle.learning_note.count(atom.source_excerpt) <= 2, atom.id
        assert f"[atom:{atom.id}]" in bundle.learning_note
    review = bundle.learning_note.split("## 主题复盘\n", 1)[1].split("\n## ", 1)[0]
    assert len([line for line in review.splitlines() if line.startswith("- ")]) > 1
    assert "## 案例与反例\n" not in bundle.learning_note
    assert "## 推广与行动号召\n" in bundle.learning_note
    for segment in note_input.segments:
        assert segment.text in bundle.learning_note
    assert bundle.learning_note.count("视频未提供这一类信息") <= 2


@pytest.mark.asyncio
async def test_render_methods_includes_ordered_steps_without_empty_subgroups() -> None:
    note_input = _note_input()
    generation = await SemanticNotePipeline(ScriptedProvider()).generate(
        note_input, asyncio.Event()
    )
    atom = generation.extraction.atoms[0]
    draft = SectionDraft(
        section_id="methods_steps",
        title="方法与步骤",
        content="有序操作",
        steps=[SectionClaim(text="先记录具体问题", atom_ids=[atom.id], timestamps=[0.0])],
        included_atom_ids=[atom.id],
    )
    bundle = render_semantic_note(
        note_input, replace(generation, section_drafts=(draft,)), generated_at="2026-11-16"
    )
    methods = bundle.learning_note.split("### 方法与步骤\n", 1)[1].split("\n### ", 1)[0]
    assert "先记录具体问题" in methods
    assert "视频未提供这一类信息" not in methods
