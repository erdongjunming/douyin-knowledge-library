from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from douyin_library.notes.input import NoteInput

MODEL_NAME = "local-rule-v1"
PROMPT_VERSION = "knowledge-note-v1"
GENERATOR_VERSION = "1.0"
SYSTEM_PROMPT = """你是抖音知识资料库的学习笔记整理器。
只依据给定的页面元数据、ASR 转写和分段转写生成中文笔记，不补写视频未提供的事实。
输出必须包含标题、摘要、核心观点、详细梳理、关键概念、原文表达、理解启发、行动建议和主题标签。
引用来自 ASR 时保留原意并提示可能存在识别误差；证据不足或内容依赖画面时明确写出局限。
结构化结果必须可追溯到 source_files，并标记生成模型、提示词版本和证据等级。"""

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|[\r\n]+")
_SPACE = re.compile(r"\s+")
_HASHTAG = re.compile(r"#([^#\s]+)")
_LATIN_TERM = re.compile(r"\b[A-Za-z][A-Za-z0-9+._-]{1,24}\b")
_CHINESE = re.compile(r"[\u4e00-\u9fff]+")
_CTA = re.compile(r"关注|点赞|收藏|转发|评论区|下期|私信|点个赞")
_LOW_VALUE = re.compile(r"今天.*分享|本期|这期|上期|视频.*结束|大家好|欢迎.*交流")
_KEY_MARKERS = (
    "首先",
    "其次",
    "关键",
    "核心",
    "本质",
    "因此",
    "如果",
    "需要",
    "建议",
    "注意",
    "不要",
    "方法",
    "步骤",
    "原则",
    "可以",
    "通过",
    "意味着",
    "原因",
    "从而",
    "帮助",
    "防止",
    "保持",
    "练习",
    "训练",
    "放松",
)
_CONCEPT_SUFFIX = re.compile(
    r"[\u4e00-\u9fff]{0,8}(?:系统|能力|方法|原则|策略|思维|模型|机制|训练|肌肉|体态|"
    r"前倾|拉伸|表达|知识|习惯|成本|风险|需求|资源|流程|工具|动作|效果|结构)"
)
_CONCEPT_LEADING = re.compile(
    r"^(?:今天|本期|这期|上期|大家|我们|你们|他们|自己|一个|这个|那个|通过|关于|"
    r"进行|需要|可以|首先|其次|接下来|主要|很多|分享|一套|使用|选择)+"
)
_CONCEPT_CONNECTOR = re.compile(r"(?:如果|通过|因为|所以|以及|或者|的|了|把|让|使|从|对|而|并)")
_GENERIC_CONCEPTS = {
    "动作",
    "效果",
    "方法",
    "能力",
    "系统",
    "训练",
    "肌肉",
    "体态",
    "结构",
    "工具",
    "知识",
    "资源",
    "风险",
    "成本",
    "需求",
    "流程",
    "习惯",
}


@dataclass(frozen=True, slots=True)
class NoteBundle:
    parsed: dict[str, Any]
    raw_output: str
    learning_note: str
    brief_markdown: str
    brief_json: dict[str, Any]
    note_budget: dict[str, Any]
    note_score: dict[str, Any]
    page_metadata: dict[str, Any]
    asset_manifest: dict[str, Any]
    coverage_report: dict[str, Any] = field(default_factory=dict)


def generate_rule_note(
    note_input: dict[str, Any] | NoteInput,
    *,
    source_files: dict[str, str] | None = None,
    generated_at: str,
) -> NoteBundle:
    note_input = _note_input_dict(note_input)
    if source_files is None:
        source_files = dict(note_input.get("source_files") or {})
    note_input["source_files"] = source_files
    transcript = _clean_text(str(note_input.get("transcript") or ""))
    description = _clean_text(str(note_input.get("description") or ""))
    sentences = _sentences(transcript)
    key_points = _select_key_points(sentences)
    summary = _summary(key_points, sentences)
    title = _title(description, sentences, str(note_input["aweme_id"]))
    concepts = _concepts(description, transcript)
    quotes = _quotes(sentences)
    topics = _topics(description, concepts)
    details = _details(note_input.get("segments"), key_points)
    timeline = _timeline(note_input.get("segments"), str(note_input.get("transcript") or ""))
    actions = _actions(key_points)
    insights = _insights(key_points, concepts)
    evidence = _evidence(note_input, transcript)
    tags = _dedupe([*topics, "抖音学习笔记", "ASR转写"])
    coverage_report = _fallback_coverage(note_input, timeline)
    parsed = {
        "aweme_id": note_input["aweme_id"],
        "title": title,
        "summary": summary,
        "key_points": key_points,
        "details": details,
        "timeline": timeline,
        "concepts": concepts,
        "quotes": quotes,
        "insights": insights,
        "actions": actions,
        "topics": topics,
        "tags": tags,
        "evidence": evidence,
        "quality_gate": "fallback",
        "coverage_report": coverage_report,
    }
    raw_output = json.dumps(parsed, ensure_ascii=False, indent=2)
    page_metadata = _page_metadata(note_input, source_files, title, generated_at)
    note_budget = _note_budget(note_input, transcript, generated_at)
    learning_note = _render_learning_note(parsed, page_metadata)
    note_score = _note_score(
        parsed, note_budget, learning_note, coverage_report, generated_at
    )
    brief_json = {
        "aweme_id": note_input["aweme_id"],
        "title": title,
        "summary": summary,
        "key_points": key_points,
        "concepts": concepts,
        "quotes": quotes,
        "topics": topics,
        "tags": tags,
        "source_files": source_files,
        "generated_at": generated_at,
        "generation_method": "local-rule-based",
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "evidence_level": evidence["level"],
        "quality_gate": "fallback",
    }
    asset_manifest = _asset_manifest(note_input, source_files, generated_at)
    return NoteBundle(
        parsed=parsed,
        raw_output=raw_output,
        learning_note=learning_note,
        brief_markdown=_render_brief(brief_json),
        brief_json=brief_json,
        note_budget=note_budget,
        note_score=note_score,
        page_metadata=page_metadata,
        asset_manifest=asset_manifest,
        coverage_report=coverage_report,
    )


def _note_input_dict(note_input: dict[str, Any] | NoteInput) -> dict[str, Any]:
    if isinstance(note_input, NoteInput):
        value = asdict(note_input)
        value["duration_seconds"] = (
            note_input.duration_ms / 1000 if note_input.duration_ms else None
        )
        return value
    return dict(note_input)


def _clean_text(value: str) -> str:
    return _SPACE.sub(" ", value.replace("\ufeff", " ")).strip()


def _sentences(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in _SENTENCE_SPLIT.split(text):
        sentence = item.strip(" ，,。；;\t")
        if not sentence:
            continue
        if len(sentence) > 260:
            pieces = [part.strip() for part in re.split(r"[，,]", sentence) if part.strip()]
        else:
            pieces = [sentence]
        for piece in pieces:
            normalized = re.sub(r"\W", "", piece)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(piece)
    return values


def _sentence_score(sentence: str) -> float:
    length = len(sentence)
    score = 2.0 if 18 <= length <= 120 else 1.0
    score += sum(0.7 for marker in _KEY_MARKERS if marker in sentence)
    score += 0.8 if any(char.isdigit() for char in sentence) else 0.0
    score -= 2.0 if _CTA.search(sentence) else 0.0
    score -= 3.0 if _LOW_VALUE.search(sentence) else 0.0
    return score


def _select_key_points(sentences: list[str], limit: int = 6) -> list[str]:
    meaningful = [
        (index, sentence)
        for index, sentence in enumerate(sentences)
        if len(sentence) >= 8
    ]
    if not meaningful:
        return sentences[:1] or ["转写内容不足，暂时无法提取稳定观点。"]
    selected = sorted(
        meaningful,
        key=lambda item: (_sentence_score(item[1]), -item[0]),
        reverse=True,
    )[:limit]
    return [sentence for _, sentence in sorted(selected)]


def _summary(key_points: list[str], sentences: list[str]) -> str:
    source = key_points or sentences
    summary = "；".join(source[:3]).strip("；")
    if len(summary) > 420:
        summary = summary[:417].rstrip("，,；;") + "..."
    return summary or "转写内容过短，无法形成可靠摘要。"


def _title(description: str, sentences: list[str], aweme_id: str) -> str:
    without_tags = _HASHTAG.sub("", description).strip(" ，,。-_")
    candidate = without_tags.splitlines()[0].strip() if without_tags else ""
    if not candidate and sentences:
        candidate = sentences[0]
    if not candidate:
        return f"抖音视频 {aweme_id} 学习笔记"
    return candidate[:58].rstrip("，,。；;") + ("..." if len(candidate) > 58 else "")


def _concepts(description: str, transcript: str, limit: int = 10) -> list[str]:
    hashtags = [tag.strip("，,。；;") for tag in _HASHTAG.findall(description)]
    latin = [term for term in _LATIN_TERM.findall(f"{description} {transcript}")]
    if hashtags:
        return _dedupe([*hashtags, *latin])[:limit]
    phrases: list[str] = []
    for clause in re.split(r"[，,。！？!?；;：:]", transcript):
        for match in _CONCEPT_SUFFIX.findall(clause):
            phrase = _CONCEPT_CONNECTOR.split(match)[-1]
            phrase = _CONCEPT_LEADING.sub("", phrase).strip()
            if 2 <= len(phrase) <= 10 and phrase not in _GENERIC_CONCEPTS:
                phrases.append(phrase)
    concepts = _dedupe([*latin, *phrases])[:6]
    if concepts:
        return concepts
    fallback = [
        clause.strip()
        for clause in re.split(r"[，,。！？!?；;：:]", transcript)
        if 2 <= len(clause.strip()) <= 12 and not _CTA.search(clause)
    ]
    return _dedupe(fallback)[:limit] or ["主题待人工标注"]


def _quotes(sentences: list[str], limit: int = 5) -> list[str]:
    candidates = [
        sentence
        for sentence in sentences
        if 12 <= len(sentence) <= 110 and not _CTA.search(sentence)
    ]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            _sentence_score(item[1])
            + (1.0 if re.search(r"不是|而是|只有|不要只|想象|就像", item[1]) else 0.0),
            -item[0],
        ),
        reverse=True,
    )[:limit]
    return [sentence for _, sentence in sorted(ranked)] or ["原始转写过短，暂无可复用表达。"]


def _topics(description: str, concepts: list[str]) -> list[str]:
    hashtags = [tag.strip("，,。；;") for tag in _HASHTAG.findall(description)]
    return _dedupe([*hashtags, *concepts])[:8]


def _details(segments: Any, key_points: list[str]) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        return [
            {"start": None, "end": None, "text": point}
            for point in key_points
        ]
    details: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_sentences = _sentences(str(segment.get("text") or ""))
        if not segment_sentences:
            continue
        ranked = sorted(
            enumerate(segment_sentences),
            key=lambda item: (_sentence_score(item[1]), -item[0]),
            reverse=True,
        )[:2]
        text = "；".join(sentence for _, sentence in sorted(ranked))
        details.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
            }
        )
    return details or [{"start": None, "end": None, "text": point} for point in key_points]


def _timeline(segments: Any, transcript: str) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        return [
            {
                "segment_index": 0,
                "start": None,
                "end": None,
                "disposition": "included" if transcript else "unintelligible_asr",
                "atom_ids": [],
                "content": transcript,
            }
        ]
    result: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            text = str(segment)
            start = None
            end = None
        else:
            text = str(segment.get("text") or "")
            start = segment.get("start")
            end = segment.get("end")
        result.append(
            {
                "segment_index": index,
                "start": start,
                "end": end,
                "disposition": (
                    "promotion"
                    if _CTA.search(text)
                    else "included"
                    if text
                    else "unintelligible_asr"
                ),
                "atom_ids": [],
                "content": text,
            }
        )
    return result


def _fallback_coverage(
    note_input: dict[str, Any], timeline: list[dict[str, Any]]
) -> dict[str, Any]:
    segment_total = len(note_input.get("segments") or []) or 1
    return {
        "schema": "douyin-note-coverage-v1",
        "segment_total": segment_total,
        "segment_disposition_count": len(timeline),
        "missing_segment_indexes": [],
        "evidence_atom_total": 0,
        "evidence_atom_included": 0,
        "missing_atom_ids": [],
        "traceable_atom_count": 0,
        "source_segments_independently_reviewed": 0,
        "source_omission_candidates": [],
        "unsupported_draft_claims": [],
        "steps_verified": False,
        "examples_verified": False,
        "conditions_verified": False,
        "promotional_content_preserved": False,
        "semantic_generator": MODEL_NAME,
        "quality_gate": "fallback",
        "repairable": False,
    }


def _actions(key_points: list[str]) -> list[str]:
    instructional = [
        point
        for point in key_points
        if re.search(r"建议|需要|保持|不要|可以|首先|接下来|注意|使用", point)
    ]
    actions = [f"按转写中的操作要求实践：{point}" for point in instructional[:2]]
    if not actions and key_points:
        actions.append(f"把核心观点写成自己的判断标准：{key_points[0]}")
    experiment = next((point for point in key_points if point not in instructional[:2]), None)
    if experiment is not None:
        actions.append(f"围绕这一要点设计一次小实验：{experiment}")
    actions.extend(
        [
            "回到转写原文，为重要结论补充上下文，并标出可能的 ASR 错词。",
            "执行后记录结果、适用条件和失败原因，再决定是否沉淀为长期方法。",
        ]
    )
    return actions[:5]


def _insights(key_points: list[str], concepts: list[str]) -> list[str]:
    focus = "、".join(concepts[:3])
    first = key_points[0] if key_points else "材料本身"
    return [
        f"这条材料的可迁移价值，主要在于用“{focus}”重新组织这一判断：{first}",
        "复用时应保留视频给出的前提和边界，不把单条口述直接扩大为普遍事实。",
    ]


def _evidence(note_input: dict[str, Any], transcript: str) -> dict[str, Any]:
    duration = float(note_input.get("duration_seconds") or 0)
    density = round(len(transcript) / duration, 2) if duration else None
    risk = "low"
    if len(transcript) < 80 or (density is not None and density < 1):
        risk = "high"
    elif len(transcript) < 300 or (density is not None and density < 3):
        risk = "medium"
    return {
        "level": "E3-ASR",
        "basis": "本地 ASR 完整转写、分段转写和页面元数据",
        "visual_dependency_risk": risk,
        "transcript_density_chars_per_second": density,
        "limitations": [
            "ASR 可能错听人名、术语、方言和背景音乐覆盖的内容。",
            "未读取视频画面内嵌文字、贴纸、商品信息或操作细节。",
        ],
    }


def _page_metadata(
    note_input: dict[str, Any],
    source_files: dict[str, str],
    title: str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "aweme_id": note_input["aweme_id"],
        "title": title,
        "description": note_input.get("description"),
        "author": note_input.get("author"),
        "author_unique_id": note_input.get("author_unique_id"),
        "published_at": note_input.get("published_at"),
        "duration_ms": note_input.get("duration_ms"),
        "source_url": note_input.get("source_url"),
        "evidence_level": "E3-ASR",
        "source_files": source_files,
        "quality_gate": "fallback",
        "generated_at": generated_at,
    }


def _note_budget(
    note_input: dict[str, Any], transcript: str, generated_at: str
) -> dict[str, Any]:
    chars = len(transcript)
    if chars < 100:
        note_min, note_max, granularity = 250, 600, "minimal"
    elif chars < 1000:
        note_min, note_max, granularity = 600, 1200, "compact"
    elif chars < 4000:
        note_min, note_max, granularity = 1000, 2200, "standard"
    else:
        note_min, note_max, granularity = 1600, 3600, "detailed"
    evidence = _evidence(note_input, transcript)
    return {
        "schema": "douyin-note-budget-v1",
        "aweme_id": note_input["aweme_id"],
        "input_sources": note_input.get("source_files", {}),
        "transcript_chars": chars,
        "segment_count": len(note_input.get("segments") or []),
        "input_truncated": False,
        "truncation_limit": None,
        "estimated_input_tokens": math.ceil(chars / 1.5),
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "provider": "rule",
        "quality_gate": "fallback",
        "repair_count": 0,
        "recommended_note_chars_min": note_min,
        "recommended_note_chars_max": note_max,
        "granularity": granularity,
        "writing_guidance": "以转写为事实主干；信息少时不扩写，信息多时按时间段组织。",
        "visual_dependency": {
            "risk": evidence["visual_dependency_risk"],
            "needs_visual_review": evidence["visual_dependency_risk"] in {"medium", "high"},
            "reason": "音频转写不能覆盖画面内嵌文字和纯视觉信息。",
        },
        "generated_at": generated_at,
    }


def _asset_manifest(
    note_input: dict[str, Any], source_files: dict[str, str], generated_at: str
) -> dict[str, Any]:
    assets = note_input.get("asset_records", {})
    return {
        "schema": "douyin-note-asset-manifest-v1",
        "aweme_id": note_input["aweme_id"],
        "evidence_level": "E3-ASR",
        "assets": assets,
        "source_files": source_files,
        "generated_files": [
            "learning_note.md",
            "douyin_ai_brief.md",
            "douyin_ai_brief.json",
            "note_budget.json",
            "note_score.json",
            "page_metadata.json",
            "generation_audit.json",
            "assets/asset_manifest.json",
        ],
        "copies_original_assets": False,
        "quality_gate": "fallback",
        "generated_at": generated_at,
    }


def _note_score(
    parsed: dict[str, Any],
    budget: dict[str, Any],
    markdown: str,
    coverage_report: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    problems = [
        "内容由本地规则生成，未通过语义完整性审查，只能作为降级时间线使用。"
    ]
    transcript_chars = int(budget["transcript_chars"])
    if transcript_chars < 80:
        problems.append("转写文本过短，无法形成完整知识结构。")
    if budget["visual_dependency"]["needs_visual_review"]:
        problems.append("可能依赖画面信息，当前笔记只覆盖音频转写。")
    if not parsed["concepts"] or parsed["concepts"] == ["主题待人工标注"]:
        problems.append("关键概念需要人工补标。")
    note_chars = len(markdown)
    if note_chars > int(budget["recommended_note_chars_max"] * 1.3):
        problems.append("笔记长度超过建议预算，需要人工判断是否进一步压缩。")
    completeness = 0
    clarity = min(85, 55 + len(parsed["concepts"]) * 2)
    usefulness = min(85, 55 + len(parsed["actions"]) * 5)
    evidence_level = 75 if transcript_chars >= 300 else 55 if transcript_chars >= 80 else 30
    score = round(
        completeness * 0.3 + clarity * 0.2 + usefulness * 0.25 + evidence_level * 0.25,
        1,
    )
    return {
        "schema": "douyin-note-score-v1",
        "aweme_id": parsed["aweme_id"],
        "completeness": completeness,
        "coverage_components": {
            "independent_source_review": 0,
            "evidence_atoms": 0,
            "semantic_quality_gate": 0,
        },
        "quality_gate": coverage_report["quality_gate"],
        "clarity": clarity,
        "usefulness": usefulness,
        "evidence_level": evidence_level,
        "problems": problems,
        "score": score,
        "note_chars": note_chars,
        "scoring_method": "local-rule-v1",
        "generated_at": generated_at,
    }


def _render_learning_note(parsed: dict[str, Any], metadata: dict[str, Any]) -> str:
    evidence = parsed["evidence"]
    lines = [
        f"# {parsed['title']}",
        "",
        "> 注意：本笔记为规则降级结果，未通过语义完整性审查；请以完整逐段时间线和 ASR 原文为准。",
        "",
        f"> 证据等级：{evidence['level']}。本文基于本地 ASR 转写整理，未覆盖完整画面信息。",
        "",
        "## 视频基本信息",
        "",
        f"- 视频 ID：`{parsed['aweme_id']}`",
        f"- 作者：{metadata.get('author') or '未知'}",
        f"- 发布时间：{metadata.get('published_at') or '未知'}",
        f"- 原始链接：{metadata.get('source_url') or '未知'}",
        f"- 时长：{_duration_label(metadata.get('duration_ms'))}",
        "",
        "## 核心观点",
        "",
    ]
    lines.extend(f"- {point}" for point in parsed["key_points"])
    lines.extend(["", "## 详细内容梳理", ""])
    for index, detail in enumerate(parsed["details"], 1):
        timestamp = _timestamp(detail.get("start"))
        prefix = f"{index}. **{timestamp}** " if timestamp else f"{index}. "
        lines.append(prefix + detail["text"])
    lines.extend(["", "## 关键概念", ""])
    lines.extend(
        f"- **{concept}**：来自标题、标签或转写中的高频表达。"
        for concept in parsed["concepts"]
    )
    lines.extend(["", "## 可复用金句或表达", ""])
    lines.append("> 以下表达来自 ASR 转写，引用前应回听音频核对错词。")
    lines.append("")
    lines.extend(f"> {quote}" for quote in parsed["quotes"])
    lines.extend(["", "## 我的理解 / 启发", ""])
    lines.extend(f"- {item}" for item in parsed["insights"])
    lines.extend(["", "## 可执行行动", ""])
    lines.extend(f"- [ ] {item}" for item in parsed["actions"])
    lines.extend(["", "## 完整逐段时间线", ""])
    for item in parsed["timeline"]:
        timestamp = _timestamp(item.get("start"))
        prefix = (
            f"{item['segment_index'] + 1}. **{timestamp}** "
            if timestamp
            else f"{item['segment_index'] + 1}. "
        )
        lines.append(
            prefix
            + f"`{item['disposition']}` segment {item['segment_index']} "
            + item["content"]
        )
    lines.extend(["", "## 适合归档的主题标签", ""])
    lines.append(" ".join(f"#{tag.replace(' ', '-')}" for tag in parsed["tags"]))
    lines.extend(["", "## 证据范围与局限", ""])
    lines.extend(f"- {item}" for item in evidence["limitations"])
    if evidence["visual_dependency_risk"] in {"medium", "high"}:
        lines.append("- 转写密度偏低或视频可能依赖画面，详细复核应补关键帧/OCR。")
    return "\n".join(lines).rstrip() + "\n"


def _render_brief(brief: dict[str, Any]) -> str:
    lines = [
        f"# {brief['title']}",
        "",
        f"> 本地规则快读，质量状态：{brief['quality_gate']}；未通过语义完整性审查。",
        "",
        brief["summary"],
        "",
        "## 要点",
        "",
    ]
    lines.extend(f"- {point}" for point in brief["key_points"][:5])
    lines.extend(["", "## 标签", "", " ".join(f"#{tag}" for tag in brief["tags"]), ""])
    return "\n".join(lines)


def _duration_label(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "未知"
    seconds = int(value / 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _timestamp(value: Any) -> str:
    if not isinstance(value, (int, float)) or value < 0:
        return ""
    seconds = int(value)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip().lstrip("#")
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result
