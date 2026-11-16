from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from douyin_library.notes.contracts import SectionClaim, SectionDraft
from douyin_library.notes.coverage import _methods_section_suspicious, readability_report
from douyin_library.notes.generator import NoteBundle
from douyin_library.notes.input import NoteInput
from douyin_library.notes.pipeline import SemanticGeneration


def _visual_dependency_risk(note_input: NoteInput) -> str:
    values: list[Any] = [note_input.metadata, note_input.note_budget]
    for value in values:
        if not isinstance(value, dict):
            continue
        dependency = value.get("visual_dependency", value)
        if isinstance(dependency, dict) and dependency.get("risk") in {
            "low",
            "medium",
            "high",
        }:
            return str(dependency["risk"])
    return "unknown"


def _coverage_views(
    note_input: NoteInput,
    generation: SemanticGeneration,
) -> dict[str, Any]:
    return {
        "coverage_scope": "transcript",
        "text_coverage": {
            "quality_gate": generation.coverage.quality_gate,
            "input_sufficient": bool(generation.input_sufficient),
            "semantic_complete": bool(generation.coverage.semantic_complete),
        },
        "visual_coverage": {
            "status": "not_reviewed",
            "dependency_risk": _visual_dependency_risk(note_input),
            "limitations": ["现有材料未提供画面核验结果。"],
        },
    }


def render_semantic_note(
    note_input: NoteInput,
    generation: SemanticGeneration,
    *,
    generated_at: str,
) -> NoteBundle:
    synthesis = generation.synthesis
    coverage = generation.coverage.model_dump(mode="json", by_alias=True)
    coverage_views = _coverage_views(note_input, generation)
    timeline_by_index = {entry.segment_index: entry for entry in synthesis.timeline}
    disposition_by_index = {
        item.segment_index: item for item in generation.extraction.dispositions
    }
    timeline = [
        {
            "segment_index": segment.index,
            "start": segment.start,
            "end": segment.end,
            "disposition": (
                timeline_by_index[segment.index].disposition
                if segment.index in timeline_by_index
                else _required_disposition(disposition_by_index, segment.index).disposition
            ),
            "atom_ids": (
                list(timeline_by_index[segment.index].atom_ids)
                if segment.index in timeline_by_index
                else list(
                    _required_disposition(
                        disposition_by_index,
                        segment.index,
                    ).atom_ids
                )
            ),
            "content": segment.text,
        }
        for segment in note_input.segments
    ]
    parsed = synthesis.model_dump(mode="json")
    evidence_map = evidence_map_from_generation(note_input, generation)
    teaching_outline = generation.outline.model_dump(mode="json")
    section_drafts = [
        draft.model_dump(mode="json") for draft in generation.section_drafts
    ]
    depth_review = (
        generation.depth_review.model_dump(mode="json")
        if generation.depth_review is not None
        else None
    )
    semantic_complete = bool(
        generation.coverage.semantic_complete
        and generation.coverage.quality_gate == "passed"
        and generation.coverage.depth_passed
        and generation.coverage.input_sufficient
        and generation.input_sufficient
    )
    coverage.update(
        {
            "teaching_outline": teaching_outline,
            "section_drafts": section_drafts,
            "depth_review": depth_review,
            "semantic_complete": semantic_complete,
            "input_sufficient": bool(generation.input_sufficient),
            **coverage_views,
        }
    )
    parsed.update(
        {
            "aweme_id": note_input.aweme_id,
            "quality_gate": generation.coverage.quality_gate,
            "timeline": timeline,
            "coverage_report": coverage,
            "evidence_map": evidence_map,
            "teaching_outline": teaching_outline,
            "section_drafts": section_drafts,
            "depth_review": depth_review,
            "semantic_complete": semantic_complete,
            "input_sufficient": bool(generation.input_sufficient),
            **coverage_views,
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
        "transcript_variant": note_input.transcript_variant,
        "transcript_source_kind": note_input.transcript_source_kind,
        "cleaning_fallback": note_input.cleaning_fallback,
        "generated_at": generated_at,
        "generation_method": "semantic-evidence-linked",
        "provider": generation.provider,
        "model": generation.model,
        "prompt_versions": dict(generation.prompt_versions),
        "quality_gate": generation.coverage.quality_gate,
        "semantic_complete": semantic_complete,
        "input_sufficient": bool(generation.input_sufficient),
        "teaching_outline": teaching_outline,
        "section_drafts": section_drafts,
        "depth_review": depth_review,
        **coverage_views,
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
        "transcript_variant": note_input.transcript_variant,
        "transcript_source_kind": note_input.transcript_source_kind,
        "cleaning_fallback": note_input.cleaning_fallback,
        "provider": generation.provider,
        "model": generation.model,
        "quality_gate": generation.coverage.quality_gate,
        "coverage": coverage,
        "semantic_complete": semantic_complete,
        "input_sufficient": bool(generation.input_sufficient),
        "teaching_outline": teaching_outline,
        "section_drafts": section_drafts,
        "depth_review": depth_review,
        "generated_at": generated_at,
        **coverage_views,
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
            "evidence_map.json",
            "assets/asset_manifest.json",
        ],
        "copies_original_assets": False,
        "quality_gate": generation.coverage.quality_gate,
        "semantic_complete": semantic_complete,
        "input_sufficient": bool(generation.input_sufficient),
        "generated_at": generated_at,
        **coverage_views,
    }
    methods_warnings = []
    if _methods_section_suspicious(generation.outline, note_input.transcript):
        methods_warnings.append("methods_steps_suspicious_missing")
    core_points = []
    atoms = {atom.id: atom for atom in generation.extraction.atoms}
    for draft in generation.section_drafts:
        if draft.section_id == "core_points":
            core_points.extend(claim.text for claim in draft.claims)
            if not any(
                (
                    draft.claims,
                    draft.steps,
                    draft.examples,
                    draft.results,
                    draft.conditions,
                    draft.risks,
                    draft.actions,
                )
            ):
                core_points.extend(
                    atoms[atom_id].statement
                    for atom_id in draft.included_atom_ids
                    if atom_id in atoms
                )
    # Reserve three characters for the score (padding at line end) so changing
    # 100 to 67 cannot oscillate across the exact character-budget threshold.
    for _ in range(4):
        readability = readability_report(learning_note, core_points, len(atoms))
        warnings = [*methods_warnings, *readability["warnings"]]
        quality_gate = generation.coverage.quality_gate
        if quality_gate == "passed" and warnings:
            quality_gate = "passed_with_warnings"
        updated = re.sub(
            r"^> 质量状态：[^\n]*",
            f"> 质量状态：{quality_gate}（可读性 {readability['score']} 分）；"
            f"生成器：{generation.provider}/{generation.model}。"
            + " "
            * (3 - len(str(readability["score"]))),
            learning_note,
            count=1,
            flags=re.MULTILINE,
        )
        if updated == learning_note:
            break
        learning_note = updated
    else:
        raise ValueError("readability_quality_header_did_not_converge")
    for artifact in (parsed, brief_json, note_budget, note_score, page_metadata, asset_manifest):
        artifact["quality_gate"] = quality_gate
    note_score["warnings"] = warnings
    note_score["readability"] = readability["score"]
    note_score["readability_components"] = readability["components"]
    note_score["readability_metrics"] = {
        key: readability[key] for key in ("empty_sections", "fragment_ratio", "chars_per_atom")
    }
    note_score["note_chars"] = len(learning_note)
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


def evidence_map_from_generation(
    note_input: NoteInput,
    generation: SemanticGeneration,
) -> dict[str, Any]:
    """Project merged extraction atoms into a durable, traceable evidence map.

    The semantic pipeline already performs overlapping chunk extraction and
    deterministic merge by atom ID. This projection keeps that canonical result
    lossless while adding explicit segment IDs and timestamp pairs for readers.
    """
    segments = {segment.index: segment for segment in note_input.segments}
    items: list[dict[str, Any]] = []
    for atom in sorted(generation.extraction.atoms, key=lambda item: item.id):
        segment = segments.get(atom.source_segment)
        start = atom.start if atom.start is not None else (segment.start if segment else None)
        end = atom.end if atom.end is not None else (segment.end if segment else None)
        items.append(
            {
                "id": atom.id,
                "kind": atom.kind,
                "topic": atom.topic,
                "statement": atom.statement,
                "source_segment": atom.source_segment,
                "segment_id": (
                    segment.segment_id
                    if segment and segment.segment_id
                    else f"segment-{atom.source_segment}"
                ),
                "start": start,
                "end": end,
                "timestamps": [value for value in (start, end) if value is not None],
                "source_excerpt": atom.source_excerpt,
                "relations": list(atom.relations),
                "sequence": atom.sequence,
                "is_promotional": atom.is_promotional,
                "asr_uncertainty": list(atom.asr_uncertainty),
            }
        )
    return {
        "schema": "douyin-note-evidence-map-v1",
        "aweme_id": note_input.aweme_id,
        "transcript_variant": note_input.transcript_variant,
        "transcript_source_kind": note_input.transcript_source_kind,
        "cleaning_fallback": note_input.cleaning_fallback,
        "items": items,
        "segment_count": len(note_input.segments),
        "section_ids": [section.section_id for section in generation.outline.sections],
        "sections": [
            {
                "section_id": section.section_id,
                "title": section.title,
                "order": section.order,
                "atom_ids": list(section.atom_ids),
                "timestamps": list(section.timestamps),
            }
            for section in generation.outline.sections
        ],
        "section_drafts": [
            draft.model_dump(mode="json") for draft in generation.section_drafts
        ],
        "dispositions": [
            item.model_dump(mode="json") for item in generation.extraction.dispositions
        ],
    }


def _required_disposition(dispositions: dict[int, Any], segment_index: int) -> Any:
    try:
        return dispositions[segment_index]
    except KeyError as exc:
        raise ValueError(
            f"missing extraction disposition for segment {segment_index}"
        ) from exc


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
        f"- 转写变体：{note_input.transcript_variant} ({note_input.transcript_source_kind})",
        f"- 清洗回退：{note_input.cleaning_fallback or '无'}",
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
    if generation.section_drafts:
        _append_teaching_sections(lines, generation)
    _append_theme_review(lines, generation)
    _append_glossary(lines, generation)
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
            f"- 画面覆盖：未核验（依赖风险：{_visual_dependency_risk(note_input)}）。",
            "- 画面相关判断仅能写为未核验；现有材料未提供画面核验结果。",
            "- “我的理解与启发”是生成理解，不等同于作者观点。",
        ]
    )
    lines.extend(f"- 输入警告：{warning}" for warning in note_input.warnings)
    _append_asr_doubts(lines, note_input, generation)
    lines = _compact_reference_groups(_deduplicate_excerpts(lines, generation, timeline))
    return "\n".join(lines).rstrip() + "\n"


def _append_theme_review(lines: list[str], generation: SemanticGeneration) -> None:
    synthesis = generation.synthesis
    sections = [
        *synthesis.themes,
        *synthesis.methods_and_steps,
        *synthesis.examples_and_counterexamples,
        *synthesis.boundaries_and_risks,
    ]
    lines.extend(["", "## 主题复盘", ""])
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for section in sections:
        summary = re.split(r"[。！？\n]", section.content, maxsplit=1)[0].strip()
        if len(summary) > 30:
            summary = section.heading[:30]
        key = (summary, tuple(section.atom_ids))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {_atom_refs(section.atom_ids)} {summary}")


def _deduplicate_excerpts(
    lines: list[str],
    generation: SemanticGeneration,
    timeline: list[dict[str, Any]],
) -> list[str]:
    # Reserve the complete raw timeline; repeated body quotations become references.
    timeline_text = "\n".join(item["content"] for item in timeline)
    doubt_start = next(
        (index for index, line in enumerate(lines) if line == "## 转写疑点清单"), len(lines)
    )
    reserved_text = timeline_text + "\n" + "\n".join(lines[doubt_start:])
    timeline_indexes: set[int] = set()
    promotion_indexes: set[int] = set()
    heading = ""
    for index, line in enumerate(lines):
        if line.startswith("## "):
            heading = line
        if heading == "## 完整逐段时间线":
            timeline_indexes.add(index)
        elif heading == "## 推广与行动号召":
            promotion_indexes.add(index)
    body_order = sorted(
        range(len(lines)), key=lambda index: (index not in promotion_indexes, index)
    )
    excerpts: dict[str, list[str]] = {}
    for atom in generation.extraction.atoms:
        excerpts.setdefault(atom.source_excerpt, []).append(atom.id)
    for excerpt in sorted(excerpts, key=len, reverse=True):
        remaining = max(0, min(1, 2 - reserved_text.count(excerpt)))
        for index in body_order:
            line = lines[index]
            if index in timeline_indexes or index >= doubt_start or line.startswith("#"):
                continue

            def replace_excerpt(
                match: re.Match[str], line: str = line, excerpt: str = excerpt
            ) -> str:
                nonlocal remaining
                # Inline literals (numbers, names, or overlapping atom fragments)
                # are part of their containing claim, not duplicate quotations.
                before, after = line[: match.start()], line[match.end() :]
                before = re.sub(r"\[atom:[^\]]+\]", "", before)
                after = re.sub(r"\[atom:[^\]]+\]|\d{2,}:\d{2}(?:-\d{2,}:\d{2})?", "", after)
                prefix = re.sub(r"^(?:- |\d+\. |> \[ASR 原文表达\] )", "", before)
                prefix = re.sub(r"^(?:原文：|生成理解：)", "", prefix)
                standalone = not prefix.strip() and not after.strip(" ,；。")
                sentence = excerpt.endswith(("。", "！", "？", ".", "!", "?"))
                if not standalone and not (
                    sentence and (not prefix.strip() or before[-1] in '。！？； "“')
                ):
                    return match.group()
                if remaining:
                    remaining -= 1
                    return match.group()
                return _atom_refs(excerpts[excerpt])

            lines[index] = re.sub(re.escape(excerpt), replace_excerpt, line)
    return lines


def _compact_reference_groups(lines: list[str]) -> list[str]:
    result: list[str] = []
    previous_reference = False
    for line in lines:
        rest = re.sub(r"\[atom:[^\]]+\]|\d{2,}:\d{2}(?:-\d{2,}:\d{2})?", "", line)
        rest = re.sub(r"^- (?:原文：|生成理解：)?", "", rest)
        reference_only = "[atom:" in line and not rest.strip(" ,；。")
        if reference_only:
            line = re.sub(r"(\[atom:[^\]]+\])(?: \1)+", r"\1", line)
            line = re.sub(r"^- (?:原文：|生成理解：)?", "- ", line)
            if previous_reference:
                result[-1] += "；" + line.removeprefix("- ")
            else:
                result.append(line)
        else:
            result.append(line)
        previous_reference = reference_only
    return result


_TERM_STOPWORDS = (
    "这个",
    "那个",
    "这些",
    "那些",
    "我们",
    "你们",
    "他们",
    "自己",
    "什么",
    "怎么",
    "因为",
    "所以",
    "但是",
    "然后",
    "接着",
    "最后",
    "首先",
    "之后",
    "比如",
    "例如",
    "可以",
    "能够",
    "需要",
    "应该",
    "进行",
    "使用",
    "通过",
    "就是",
    "一个",
    "一下",
    "已经",
    "没有",
    "还有",
    "以及",
    "或者",
    "如果",
    "这样",
    "那样",
    "时候",
    "其实",
    "的",
    "了",
    "是",
    "在",
    "把",
    "和",
    "与",
    "就",
    "都",
    "也",
    "很",
    "要",
    "去",
)
_TERM_BOUNDARIES = re.compile("|".join(map(re.escape, _TERM_STOPWORDS)))


def _append_glossary(lines: list[str], generation: SemanticGeneration) -> None:
    counts: Counter[str] = Counter()
    # Function words delimit candidates; prefer maximal recurring 2-6 character
    # phrases over their overlapping fragments. This is a lexical heuristic, not POS tagging.
    for atom in generation.extraction.atoms:
        for phrase in re.findall(
            r"[\u4e00-\u9fff]+", _TERM_BOUNDARIES.sub(" ", atom.source_excerpt)
        ):
            for size in range(2, min(6, len(phrase)) + 1):
                counts.update(
                    phrase[index : index + size] for index in range(len(phrase) - size + 1)
                )
    terms = list(dict.fromkeys(generation.synthesis.concepts))
    candidates = sorted(
        (term for term, count in counts.items() if count >= 2),
        key=lambda term: (-len(term), -counts[term], term),
    )
    for term in candidates:
        if not any(term in existing for existing in terms):
            terms.append(term)
    lines.extend(["", "## 术语表", ""])
    lines.extend("- " + "；".join(terms[index:index + 8]) for index in range(0, len(terms), 8))


def _append_asr_doubts(
    lines: list[str], note_input: NoteInput, generation: SemanticGeneration
) -> None:
    unintelligible = {
        item.segment_index
        for item in generation.extraction.dispositions
        if item.disposition == "unintelligible_asr"
    }
    entries: dict[tuple[int, str], dict[str, Any]] = {}
    segments = {segment.index: segment for segment in note_input.segments}
    for atom in generation.extraction.atoms:
        if not atom.asr_uncertainty and atom.source_segment not in unintelligible:
            continue
        segment = segments.get(atom.source_segment)
        entry = entries.setdefault(
            (atom.source_segment, atom.source_excerpt),
            {
                "start": atom.start
                if atom.start is not None
                else (segment.start if segment else None),
                "end": atom.end if atom.end is not None else (segment.end if segment else None),
                "atom_ids": [],
                "markers": [],
            },
        )
        entry["atom_ids"].append(atom.id)
        entry["markers"].extend(atom.asr_uncertainty)
    for index in sorted(unintelligible):
        if any(key[0] == index for key in entries) or index not in segments:
            continue
        segment = segments[index]
        entries[(index, segment.text)] = {
            "start": segment.start,
            "end": segment.end,
            "atom_ids": [],
            "markers": [],
        }
    lines.extend(["", "## 转写疑点清单", ""])
    if not entries:
        lines.append("- 无已标记的转写疑点。")
    for (_, excerpt), entry in sorted(entries.items()):
        context = _atom_refs(entry["atom_ids"])
        markers = "；".join(dict.fromkeys(entry["markers"]))
        if markers:
            context += f"（未确认：{markers}）"
        lines.append(
            f"- {_time_range(entry['start'], entry['end'])} · {excerpt} {context}".rstrip()
        )


_TEACHING_SECTION_HEADINGS = {
    "overview": "主题观点",
    "core_points": "核心观点",
    "methods_steps": "方法与步骤",
    "examples_results": "案例与结果",
    "boundaries_risks": "前提、边界与风险",
    "actions": "行动建议",
    "timeline": "逐段证据",
    "evidence_limits": "证据与局限",
}


def _append_teaching_sections(
    lines: list[str], generation: SemanticGeneration
) -> None:
    atoms_by_id = {atom.id: atom for atom in generation.extraction.atoms}
    drafts_by_id = {draft.section_id: draft for draft in generation.section_drafts}
    lines.extend(["", "## 教学型正文", ""])
    for outline_section in sorted(generation.outline.sections, key=lambda item: item.order):
        section_heading = _TEACHING_SECTION_HEADINGS.get(
            outline_section.section_id,
            outline_section.title,
        )
        lines.extend(
            [
                "",
                f"### {section_heading}",
                "",
            ]
        )
        draft = drafts_by_id.get(outline_section.section_id)
        if draft is None:
            lines.append("视频未提供这一类信息。")
            continue
        _render_section_draft(lines, draft, atoms_by_id)


def _render_section_draft(
    lines: list[str],
    draft: SectionDraft,
    atoms_by_id: dict[str, Any],
) -> None:
    """Render a section as source-backed teaching blocks.

    Structured claims are always printed with their atom references and
    timestamps.  Text in ``generated_interpretations`` is deliberately
    labelled so it cannot be mistaken for an author quotation.
    """

    claims = list(draft.claims)
    structured_claims = [
        *draft.claims, *draft.steps, *draft.examples, *draft.results,
        *draft.conditions, *draft.risks, *draft.actions,
    ]
    if not structured_claims and not draft.included_atom_ids:
        lines.append("视频未提供这一类信息。")
        return
    if not structured_claims and draft.included_atom_ids:
        claims = [
            SectionClaim(
                text=atoms_by_id[atom_id].statement,
                atom_ids=[atom_id],
                timestamps=[
                    value
                    for value in (
                        atoms_by_id[atom_id].start,
                        atoms_by_id[atom_id].end,
                    )
                    if value is not None
                ]
                ,
            )
            for atom_id in draft.included_atom_ids
            if atom_id in atoms_by_id
            and _atom_timestamps(atoms_by_id[atom_id])
        ]

    _render_claim_group(lines, "观点", claims, atoms_by_id)
    source_ids = _claim_atom_ids(
        [
            *draft.claims,
            *draft.steps,
            *draft.examples,
            *draft.results,
            *draft.conditions,
            *draft.risks,
            *draft.actions,
        ]
    )
    source_ids.update(draft.included_atom_ids)
    source_lines: list[str] = []
    for atom_id in sorted(
        source_ids,
        key=lambda value: _atom_sort_key(atoms_by_id.get(value), value),
    ):
        atom = atoms_by_id.get(atom_id)
        if atom is None:
            continue
        timestamp = _atom_time_range(atom) or "时间未知"
        source_lines.append(
            f"原文：{atom.source_excerpt} {_atom_refs([atom.id])} {timestamp}".rstrip()
        )
    _render_text_group(lines, "原文依据", source_lines)

    interpretation_lines = [
        _generated_interpretation(item) for item in draft.generated_interpretations
    ]
    has_evidence = bool(source_ids)
    if (
        not interpretation_lines
        and has_evidence
        and draft.content
        and draft.content != "视频未提供这一类信息。"
    ):
        interpretation_lines = [f"生成理解：{draft.content}"]
    _render_text_group(lines, "解释", interpretation_lines)

    _render_claim_group(lines, "步骤", draft.steps, atoms_by_id)

    _render_claim_group(
        lines,
        "例子/结果",
        [*draft.examples, *draft.results],
        atoms_by_id,
    )
    _render_claim_group(
        lines,
        "适用条件/限制",
        draft.conditions,
        atoms_by_id,
    )
    _render_claim_group(
        lines,
        "风险或行动",
        [*draft.risks, *draft.actions],
        atoms_by_id,
    )


def _render_claim_group(
    lines: list[str],
    label: str,
    claims: list[SectionClaim],
    atoms_by_id: dict[str, Any],
) -> None:
    if not claims:
        return
    lines.extend([f"#### {label}", ""])
    for claim in claims:
        timestamps = [
            value
            for atom_id in claim.atom_ids
            for value in _atom_timestamps(atoms_by_id.get(atom_id))
        ]
        time_text = (
            ", ".join(_timestamp(value) for value in timestamps) or "时间未知"
        )
        context = " ".join(
            value for value in (_atom_refs(list(claim.atom_ids)), time_text) if value
        )
        lines.append(f"- {claim.text} {context}".rstrip())


def _render_text_group(lines: list[str], label: str, values: list[str]) -> None:
    if not values:
        return
    lines.extend([f"#### {label}", ""])
    lines.extend(f"- {value}" for value in values)


def _claim_atom_ids(claims: list[SectionClaim]) -> set[str]:
    return {atom_id for claim in claims for atom_id in claim.atom_ids}


def _generated_interpretation(value: str) -> str:
    text = value.strip()
    if text.startswith("生成理解："):
        text = text.removeprefix("生成理解：").strip()
    return f"生成理解：{text}"


def _atom_timestamps(atom: Any) -> list[float]:
    if atom is None:
        return []
    return [value for value in (atom.start, atom.end) if value is not None]


def _atom_time_range(atom: Any) -> str:
    if atom is None:
        return ""
    values = _atom_timestamps(atom)
    if not values:
        return ""
    if len(values) == 1:
        return _timestamp(values[0])
    return _time_range(values[0], values[-1])


def _atom_sort_key(atom: Any, atom_id: str) -> tuple[float, float, float, str]:
    if atom is None:
        return (float("inf"), float("inf"), float("inf"), atom_id)
    return (
        float(atom.source_segment),
        float(atom.start) if atom.start is not None else float("inf"),
        float(atom.sequence) if atom.sequence is not None else float("inf"),
        atom_id,
    )


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
    evidence_types = sorted({atom.kind for atom in generation.extraction.atoms})
    recommended_detail = _recommended_detail(evidence_types)
    execution_mode = _execution_mode(generation)
    teaching_outline = generation.outline.model_dump(mode="json")
    section_drafts = [
        draft.model_dump(mode="json") for draft in generation.section_drafts
    ]
    depth_review = (
        generation.depth_review.model_dump(mode="json")
        if generation.depth_review is not None
        else None
    )
    semantic_complete = bool(
        generation.coverage.semantic_complete
        and generation.coverage.quality_gate == "passed"
        and generation.coverage.depth_passed
        and generation.coverage.input_sufficient
        and generation.input_sufficient
    )
    result = {
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
        "recommended_detail": {
            "level": recommended_detail["level"],
            "evidence_types": evidence_types,
            "reason": recommended_detail["reason"],
        },
        "execution": {
            "mode": execution_mode,
            "provider": generation.provider,
            "model": generation.model,
            "is_local": generation.is_local,
            "is_external": execution_mode == "external",
        },
        "prompt_versions": dict(generation.prompt_versions),
        "usage": generation.usage.model_dump(mode="json"),
        "api_cost": generation.usage.api_cost,
        "cost_status": generation.usage.cost_status,
        "repair_count": generation.repair_count,
        "context_status": generation.context_status,
        "context_failures": list(generation.context_failures),
        "quality_gate": generation.coverage.quality_gate,
        "teaching_outline": teaching_outline,
        "section_drafts": section_drafts,
        "depth_review": depth_review,
        "semantic_complete": semantic_complete,
        "input_sufficient": bool(generation.input_sufficient),
        "generated_at": generated_at,
    }
    result.update(_coverage_views(note_input, generation))
    return result


def _recommended_detail(evidence_types: list[str]) -> dict[str, str]:
    """Choose detail from the evidence shape, not transcript length."""
    high_fidelity_types = {
        "step",
        "example",
        "counterexample",
        "condition",
        "warning",
        "number",
        "recommendation",
        "promotion",
    }
    if not evidence_types:
        return {
            "level": "minimal",
            "reason": "未提取到证据 atom，只能保留基础时间线。",
        }
    if high_fidelity_types.intersection(evidence_types):
        return {
            "level": "full",
            "reason": "包含步骤、案例、条件、风险或行动号召等需要完整保留的证据类型。",
        }
    return {
        "level": "standard",
        "reason": "当前证据以一般观点为主，保留主题与时间线即可复盘。",
    }


def _execution_mode(generation: SemanticGeneration) -> str:
    if generation.is_local:
        return "local"
    if generation.provider.strip().casefold() in {"rule", "local-rule"}:
        return "rule"
    return "external"


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


__all__ = ["_render_section_draft", "evidence_map_from_generation", "render_semantic_note"]
