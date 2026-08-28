from __future__ import annotations

import json
import math
from typing import Any

from douyin_library.notes.generator import NoteBundle
from douyin_library.notes.input import NoteInput
from douyin_library.notes.pipeline import SemanticGeneration


def render_semantic_note(
    note_input: NoteInput,
    generation: SemanticGeneration,
    *,
    generated_at: str,
) -> NoteBundle:
    synthesis = generation.synthesis
    coverage = generation.coverage.model_dump(mode="json", by_alias=True)
    timeline_by_index = {entry.segment_index: entry for entry in synthesis.timeline}
    timeline = [
        {
            "segment_index": segment.index,
            "start": segment.start,
            "end": segment.end,
            "disposition": (
                timeline_by_index[segment.index].disposition
                if segment.index in timeline_by_index
                else "unintelligible_asr"
            ),
            "atom_ids": (
                list(timeline_by_index[segment.index].atom_ids)
                if segment.index in timeline_by_index
                else []
            ),
            "content": segment.text,
        }
        for segment in note_input.segments
    ]
    parsed = synthesis.model_dump(mode="json")
    parsed.update(
        {
            "aweme_id": note_input.aweme_id,
            "quality_gate": generation.coverage.quality_gate,
            "timeline": timeline,
            "coverage_report": coverage,
        }
    )
    learning_note = _render_learning_note(note_input, generation, timeline)
    note_budget = _note_budget(note_input, generation, generated_at)
    note_score = _note_score(
        note_input.aweme_id, generation, learning_note, generated_at
    )
    brief_json = {
        "schema": "douyin-ai-brief-v2",
        "aweme_id": note_input.aweme_id,
        "title": synthesis.title,
        "summary": synthesis.overview,
        "key_points": [section.content for section in synthesis.themes],
        "concepts": list(synthesis.concepts),
        "quotes": list(synthesis.quotes),
        "topics": list(synthesis.tags),
        "tags": list(synthesis.tags),
        "source_files": dict(note_input.source_files),
        "generated_at": generated_at,
        "generation_method": "semantic-evidence-linked",
        "provider": generation.provider,
        "model": generation.model,
        "prompt_versions": dict(generation.prompt_versions),
        "quality_gate": generation.coverage.quality_gate,
    }
    page_metadata = {
        "schema": "douyin-note-page-metadata-v2",
        "aweme_id": note_input.aweme_id,
        "title": synthesis.title,
        "description": note_input.description,
        "author": note_input.author,
        "author_unique_id": note_input.author_unique_id,
        "published_at": note_input.published_at,
        "duration_ms": note_input.duration_ms,
        "source_url": note_input.source_url,
        "source_files": dict(note_input.source_files),
        "provider": generation.provider,
        "model": generation.model,
        "quality_gate": generation.coverage.quality_gate,
        "coverage": coverage,
        "generated_at": generated_at,
    }
    asset_manifest = {
        "schema": "douyin-note-asset-manifest-v2",
        "aweme_id": note_input.aweme_id,
        "assets": dict(note_input.asset_records),
        "source_files": dict(note_input.source_files),
        "generated_files": [
            "learning_note.md",
            "douyin_ai_brief.md",
            "douyin_ai_brief.json",
            "note_budget.json",
            "note_score.json",
            "page_metadata.json",
            "coverage_report.json",
            "generation_audit.json",
            "assets/asset_manifest.json",
        ],
        "copies_original_assets": False,
        "quality_gate": generation.coverage.quality_gate,
        "generated_at": generated_at,
    }
    return NoteBundle(
        parsed=parsed,
        raw_output=json.dumps(parsed, ensure_ascii=False, indent=2),
        learning_note=learning_note,
        brief_markdown=_render_brief(brief_json),
        brief_json=brief_json,
        note_budget=note_budget,
        note_score=note_score,
        page_metadata=page_metadata,
        asset_manifest=asset_manifest,
        coverage_report=coverage,
    )


def _render_learning_note(
    note_input: NoteInput,
    generation: SemanticGeneration,
    timeline: list[dict[str, Any]],
) -> str:
    synthesis = generation.synthesis
    lines = [
        f"# {synthesis.title}",
        "",
        (
            f"> 质量状态：{generation.coverage.quality_gate}；"
            f"生成器：{generation.provider}/{generation.model}。"
        ),
        "",
        "## 视频基本信息",
        "",
        f"- 视频 ID：`{note_input.aweme_id}`",
        f"- 作者：{note_input.author or '未知'}",
        f"- 作者唯一 ID：{note_input.author_unique_id or '未知'}",
        f"- 发布时间：{note_input.published_at or '未知'}",
        f"- 原始链接：{note_input.source_url or '未知'}",
        f"- 时长：{_duration_label(note_input.duration_ms)}",
        "",
        "## 一句话总览",
        "",
        synthesis.overview,
        "",
        "## 完整结构",
        "",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(synthesis.structure, 1))
    _append_sections(lines, "主题复盘", synthesis.themes)
    _append_sections(lines, "方法与步骤", synthesis.methods_and_steps)
    _append_sections(lines, "案例与反例", synthesis.examples_and_counterexamples)
    _append_sections(lines, "前提、边界与风险", synthesis.boundaries_and_risks)
    lines.extend(["", "## 关键概念", ""])
    lines.extend(f"- {item}" for item in synthesis.concepts)
    lines.extend(["", "## 可复用金句或表达", ""])
    lines.extend(f"> [ASR 原文表达] {item}" for item in synthesis.quotes)
    lines.extend(["", "## 我的理解与启发", ""])
    lines.extend(f"- {item}" for item in synthesis.implications)
    lines.extend(["", "## 可执行行动", ""])
    lines.extend(f"- [ ] {item}" for item in synthesis.actions)
    lines.extend(["", "## 完整逐段时间线", ""])
    for item in timeline:
        atom_refs = _atom_refs(item["atom_ids"])
        lines.append(
            f"{item['segment_index'] + 1}. **segment {item['segment_index']} "
            f"{_time_range(item['start'], item['end'])}** "
            f"`{item['disposition']}` {atom_refs} {item['content']}".rstrip()
        )
    _append_sections(lines, "推广与行动号召", synthesis.promotions)
    lines.extend(["", "## 归档标签", ""])
    lines.append(" ".join(f"#{tag.replace(' ', '-')}" for tag in synthesis.tags))
    lines.extend(
        [
            "",
            "## 证据与 ASR 局限",
            "",
            "- 内容以完整 ASR transcript 和逐段转写为事实主干，atom 引用用于追溯。",
            "- ASR 可能错听人名、术语、方言、数字或背景音乐覆盖的内容。",
            "- 未读取画面内嵌文字、贴纸、商品信息或纯视觉操作细节。",
            "- “我的理解与启发”是生成理解，不等同于作者观点。",
        ]
    )
    lines.extend(f"- 输入警告：{warning}" for warning in note_input.warnings)
    return "\n".join(lines).rstrip() + "\n"


def _append_sections(lines: list[str], heading: str, sections: list[Any]) -> None:
    lines.extend(["", f"## {heading}", ""])
    if not sections:
        lines.append("- 源材料未提供这一类内容。")
        return
    for section in sections:
        timestamps = ", ".join(_timestamp(value) for value in section.timestamps)
        context = " ".join(value for value in (timestamps, _atom_refs(section.atom_ids)) if value)
        lines.append(f"- **{section.heading}** {context} {section.content}".rstrip())


def _note_budget(
    note_input: NoteInput,
    generation: SemanticGeneration,
    generated_at: str,
) -> dict[str, Any]:
    transcript_chars = len(note_input.transcript)
    segment_chars = sum(len(segment.text) for segment in note_input.segments)
    metadata_chars = len(json.dumps(note_input.metadata, ensure_ascii=False))
    complete_input_chars = transcript_chars + segment_chars + metadata_chars
    return {
        "schema": "douyin-note-budget-v2",
        "aweme_id": note_input.aweme_id,
        "input_sources": dict(note_input.source_files),
        "transcript_chars": transcript_chars,
        "segment_chars": segment_chars,
        "metadata_chars": metadata_chars,
        "complete_input_chars": complete_input_chars,
        "estimated_input_tokens": math.ceil(complete_input_chars / 1.5),
        "segment_count": len(note_input.segments),
        "atom_count": len(generation.extraction.atoms),
        "context_tokens": generation.context_tokens,
        "batch_boundaries": list(generation.batch_boundaries),
        "batch_count": len(generation.batch_boundaries),
        "stage_batches": [
            {
                "stage": record.stage,
                "batch_boundary": record.batch_boundary,
                "context_budget": record.payload.get("context_budget"),
            }
            for record in generation.stage_records
        ],
        "stage_call_count": len(generation.stage_records),
        "input_truncated": False,
        "provider": generation.provider,
        "model": generation.model,
        "prompt_versions": dict(generation.prompt_versions),
        "usage": generation.usage.model_dump(mode="json"),
        "api_cost": generation.usage.api_cost,
        "cost_status": generation.usage.cost_status,
        "repair_count": generation.repair_count,
        "context_status": generation.context_status,
        "context_failures": list(generation.context_failures),
        "quality_gate": generation.coverage.quality_gate,
        "generated_at": generated_at,
    }


def _note_score(
    aweme_id: str,
    generation: SemanticGeneration,
    markdown: str,
    generated_at: str,
) -> dict[str, Any]:
    report = generation.coverage
    components = {
        "segment_disposition": _disposition_score(
            report.segment_total,
            report.segment_disposition_count,
            report.missing_segment_indexes,
        ),
        "evidence_included": _ratio(report.evidence_atom_included, report.evidence_atom_total),
        "evidence_traceable": _ratio(report.traceable_atom_count, report.evidence_atom_total),
        "independent_source_review": _ratio(
            report.source_segments_independently_reviewed, report.segment_total
        ),
        "ordered_steps": 100 if report.steps_verified else 0,
        "examples": 100 if report.examples_verified else 0,
        "conditions": 100 if report.conditions_verified else 0,
        "promotion": 100 if report.promotional_content_preserved else 0,
        "review_findings_clear": (
            100
            if not report.source_omission_candidates and not report.unsupported_draft_claims
            else 0
        ),
        "semantic_quality_gate": 100 if report.quality_gate == "passed" else 0,
    }
    completeness = round(sum(components.values()) / len(components))
    clarity_components = {
        "structure_present": 100 if generation.synthesis.structure else 0,
        "themes_present": 100 if generation.synthesis.themes else 0,
        "themes_timestamped": (
            100
            if generation.synthesis.themes
            and all(section.timestamps for section in generation.synthesis.themes)
            else 0
        ),
        "timeline_organized": (
            100
            if len(generation.synthesis.timeline) == report.segment_total
            else 0
        ),
    }
    usefulness_components = {
        "concepts": 100 if generation.synthesis.concepts else 0,
        "quotes": 100 if generation.synthesis.quotes else 0,
        "implications": 100 if generation.synthesis.implications else 0,
        "actions": 100 if generation.synthesis.actions else 0,
    }
    clarity = round(sum(clarity_components.values()) / len(clarity_components))
    usefulness = round(
        sum(usefulness_components.values()) / len(usefulness_components)
    )
    score = round(completeness * 0.6 + clarity * 0.2 + usefulness * 0.2, 1)
    problems = []
    if report.quality_gate != "passed":
        problems.append("语义完整性审查未通过，当前结果只能作为失败候选。")
    if report.missing_atom_ids:
        problems.append("存在未纳入的 evidence atom。")
    if report.source_omission_candidates:
        problems.append("独立原文审查仍发现来源遗漏。")
    if clarity < 100:
        problems.append("结构组织或时间戳标注不完整。")
    if usefulness < 100:
        problems.append("概念、原文表达、生成理解或行动建议不足。")
    return {
        "schema": "douyin-note-score-v2",
        "aweme_id": aweme_id,
        "completeness": completeness,
        "coverage_components": components,
        "clarity": clarity,
        "clarity_components": clarity_components,
        "usefulness": usefulness,
        "usefulness_components": usefulness_components,
        "quality_gate": report.quality_gate,
        "problems": problems,
        "score": score,
        "note_chars": len(markdown),
        "scoring_method": "coverage-report-v1",
        "generated_at": generated_at,
    }


def _render_brief(brief: dict[str, Any]) -> str:
    lines = [
        f"# {brief['title']}",
        "",
        f"> 质量状态：{brief['quality_gate']}；{brief['provider']}/{brief['model']}。",
        "",
        brief["summary"],
        "",
        "## 要点",
        "",
    ]
    lines.extend(f"- {item}" for item in brief["key_points"])
    lines.extend(["", "## 标签", "", " ".join(f"#{tag}" for tag in brief["tags"]), ""])
    return "\n".join(lines)


def _ratio(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 100
    return round(max(0, min(numerator, denominator)) / denominator * 100)


def _disposition_score(total: int, count: int, missing: list[int]) -> int:
    if total == 0:
        return 100 if count == 0 else 0
    if count != total:
        return 0
    return _ratio(total - len(missing), total)


def _duration_label(value: int | None) -> str:
    if value is None or value <= 0:
        return "未知"
    seconds = value // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _timestamp(value: float) -> str:
    seconds = int(value)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _time_range(start: float | None, end: float | None) -> str:
    left = _timestamp(start) if start is not None else "未知"
    right = _timestamp(end) if end is not None else "未知"
    return f"{left}-{right}"


def _atom_refs(atom_ids: list[str]) -> str:
    return " ".join(f"[atom:{atom_id}]" for atom_id in atom_ids)


__all__ = ["render_semantic_note"]
