from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.contracts import (
    CoverageReview,
    DepthReview,
    DraftClaim,
    EvidenceAtom,
    ExtractionResult,
    ProviderUsage,
    RepairResult,
    SectionDraft,
    SynthesisResult,
    TeachingOutline,
    TeachingOutlineSection,
)
from douyin_library.notes.depth import DepthValidator
from douyin_library.notes.input import InputSegment, NoteInput
from douyin_library.notes.pipeline import (
    ProviderContextBudgetError,
    SegmentBatch,
    SemanticNotePipeline,
    _canonical_source_extraction,
    _claim_review_payload,
    _claims_for_source_batch,
    _depth_repair_slice,
    _depth_review_payload,
    _draft_claims,
    _fallback_source_review_batches,
    _input_is_sufficient,
    _merge_extractions,
    _merge_reviews,
    _normalize_outline,
    _normalize_provider_depth_review,
    _normalize_section,
    _project_synthesis,
    _provider_request_budget,
    _remove_noop_disposition_corrections,
    _repair_slice,
    _slice_synthesis,
    _source_fallback_outline,
    _source_fallback_section_drafts,
    _source_review_payload,
    _split_segment_batch,
    _synthesis_payload,
    _teaching_synthesis,
    _validate_depth_review_refs,
    _validate_extraction_response,
    build_segment_batches,
)
from douyin_library.notes.providers.base import NoteProviderError, ProviderResponse


def _note_input() -> NoteInput:
    texts = [
        "这套方法用于把零散知识变成表达系统。",
        "第一步，记录真实问题。",
        "第二步，按主题整理案例。",
        "第三步，通过公开复述验证理解。",
        "例如小林连续复述七天，结果把准备时间从两小时降到半小时。",
        "只有每周复盘才适用；注意不要删除失败记录，否则无法回滚。",
        "关注我并领取完整模板，下期继续讲。",
    ]
    segments = [
        InputSegment(index=index, start=float(index * 10), end=float((index + 1) * 10), text=text)
        for index, text in enumerate(texts)
    ]
    return NoteInput(
        aweme_id="semantic-731",
        transcript="".join(texts),
        segments=segments,
        metadata={},
        description="三步表达系统",
        author="知识作者",
        author_unique_id="teacher",
        published_at="2026-08-28T08:00:00+08:00",
        duration_ms=70_000,
        source_url="https://www.douyin.com/video/semantic-731",
        source_files={},
        asset_records={},
        warnings=[],
    )


def test_batches_split_one_oversized_segment_without_losing_text() -> None:
    note_input = _long_note_input()
    text = note_input.transcript

    batches = build_segment_batches(note_input, context_tokens=100)

    assert len(batches) > 1
    assert "".join(segment.text for batch in batches for segment in batch.segments) == text
    assert all(batch.estimated_tokens <= 50 for batch in batches)
    assert all(
        segment.start == 0.0 and segment.end == 60.0
        for batch in batches
        for segment in batch.segments
    )


def test_batches_assign_absolute_source_offsets_for_repeated_segments() -> None:
    source = "重复。\n\n重复。"
    segments = [
        InputSegment(index=0, start=0.0, end=1.0, text="重复。"),
        InputSegment(index=1, start=1.0, end=2.0, text="重复。"),
    ]
    note_input = NoteInput(
        aweme_id="absolute-source-offsets",
        transcript=source,
        segments=segments,
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=2_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )

    batches = build_segment_batches(
        note_input,
        context_tokens=65_536,
        input_fraction=0.35,
    )

    flattened = [segment for batch in batches for segment in batch.segments]
    assert [(segment.source_offset, segment.text) for segment in flattened] == [
        (0, "重复。"),
        (5, "重复。"),
    ]
    assert all(
        source[segment.source_offset : segment.source_offset + len(segment.text)] == segment.text
        for segment in flattened
    )


def _long_note_input() -> NoteInput:
    text = "超长完整内容" * 3_000
    return NoteInput(
        aweme_id="long",
        transcript=text,
        segments=[InputSegment(index=0, start=0.0, end=60.0, text=text)],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=60_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=["synthetic segment"],
    )


def test_extraction_validation_rejects_truncated_source_excerpt() -> None:
    note_input = NoteInput(
        aweme_id="extract-validation",
        transcript="完整源片段，包含后续功能和边界。",
        segments=[
            InputSegment(
                index=0,
                start=0.0,
                end=1.0,
                text="完整源片段，包含后续功能和边界。",
            )
        ],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=1_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    batch = build_segment_batches(note_input, context_tokens=32)[0]
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "A-1",
                    "kind": "claim",
                    "topic": "主题",
                    "statement": "完整源片段",
                    "source_segment": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "source_excerpt": "完整源片段",
                    "relations": [],
                    "sequence": None,
                    "is_promotional": False,
                    "asr_uncertainty": [],
                }
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["A-1"],
                    "reason": None,
                }
            ],
        },
        strict=True,
    )

    with pytest.raises(ValueError, match="source coverage"):
        _validate_extraction_response(batch, extraction)


def test_extraction_validation_rejects_empty_included_disposition() -> None:
    note_input = _note_input()
    batch = build_segment_batches(note_input, context_tokens=32)[0]
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [],
            "dispositions": [
                {
                    "segment_index": batch.segments[0].index,
                    "disposition": "included",
                    "atom_ids": [],
                    "reason": None,
                }
            ],
        },
        strict=True,
    )

    with pytest.raises(ValueError, match="atom_ids"):
        _validate_extraction_response(batch, extraction)


def test_extraction_validation_rejects_atoms_excluded_by_unintelligible_disposition() -> None:
    segment = InputSegment(index=0, start=0.0, end=1.0, text="来源包含一个可提取事实。")
    batch = SegmentBatch(0, 0, 0, (segment,), len(segment.text), 8)
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "excluded-atom",
                    "kind": "claim",
                    "topic": "事实",
                    "statement": segment.text,
                    "source_segment": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "source_excerpt": segment.text,
                    "relations": [],
                    "sequence": None,
                    "is_promotional": False,
                    "asr_uncertainty": [],
                }
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "unintelligible_asr",
                    "atom_ids": [],
                    "reason": "模型错误地忽略了已有 atom",
                }
            ],
        },
        strict=True,
    )

    with pytest.raises(ValueError, match="atom mapping"):
        _validate_extraction_response(batch, extraction)


def test_canonical_extraction_is_lossless_and_strictly_valid() -> None:
    source = "第一步，记录问题。例如连续复述七天；注意不要删除记录。"
    batch = SegmentBatch(
        batch_index=2,
        start_segment=0,
        end_segment=0,
        segments=(InputSegment(index=0, start=0.0, end=8.0, text=source),),
        input_chars=len(source),
        estimated_tokens=20,
    )

    extraction = _canonical_source_extraction(batch)

    _validate_extraction_response(batch, extraction)
    assert extraction.dispositions[0].atom_ids
    assert all(atom.source_excerpt in source for atom in extraction.atoms)
    assert any(atom.kind == "step" for atom in extraction.atoms)
    assert any(atom.kind == "example" and atom.relations for atom in extraction.atoms)
    assert any(atom.kind == "warning" for atom in extraction.atoms)


def test_synthesis_projection_deduplicates_identical_atom_text() -> None:
    text = "完整证据" * 100
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "A-long",
                    "kind": "claim",
                    "topic": "主题",
                    "statement": text,
                    "source_segment": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "source_excerpt": text,
                    "relations": [],
                    "sequence": None,
                    "is_promotional": False,
                    "asr_uncertainty": [],
                }
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["A-long"],
                    "reason": None,
                }
            ],
        },
        strict=True,
    )

    payload = _synthesis_payload(_long_note_input(), extraction, 0, 1)
    projected_atom = payload["extraction"]["atoms"][0]

    assert projected_atom["source_excerpt"] == text
    assert projected_atom["source_excerpt_is_statement"] is True
    assert "statement" not in projected_atom


def _atom_for_segment(segment: Mapping[str, Any]) -> list[dict[str, Any]]:
    index = segment["index"]
    text = segment["text"]
    common = {
        "topic": "表达系统",
        "source_segment": index,
        "start": segment["start"],
        "end": segment["end"],
        "source_excerpt": text,
        "relations": [],
        "sequence": None,
        "is_promotional": False,
        "asr_uncertainty": [],
    }
    if index in {1, 2, 3}:
        return [
            {**common, "id": f"step-{index}", "kind": "step", "statement": text, "sequence": index}
        ]
    if index == 4:
        claim = {**common, "id": "case-result", "kind": "claim", "statement": "复述降低准备时间"}
        example = {
            **common,
            "id": "case-example",
            "kind": "example",
            "statement": text,
            "relations": ["case-result"],
        }
        return [claim, example]
    if index == 5:
        claim = {
            **common,
            "id": "condition-target",
            "kind": "claim",
            "statement": "方法需要持续复盘",
        }
        condition = {
            **common,
            "id": "condition",
            "kind": "condition",
            "statement": "只有每周复盘才适用",
            "relations": ["condition-target"],
        }
        warning = {
            **common,
            "id": "warning",
            "kind": "warning",
            "statement": "不要删除失败记录，否则无法回滚",
        }
        return [claim, condition, warning]
    if index == 6:
        return [
            {
                **common,
                "id": "promotion",
                "kind": "promotion",
                "statement": text,
                "is_promotional": True,
            }
        ]
    return [{**common, "id": "claim", "kind": "claim", "statement": text}]


def _synthesis(extraction: Mapping[str, Any], *, omit_warning: bool) -> dict[str, Any]:
    atoms = extraction["atoms"]
    included = [atom for atom in atoms if not (omit_warning and atom["kind"] == "warning")]
    included_ids = [atom["id"] for atom in included]
    by_segment: dict[int, list[dict[str, Any]]] = {}
    for atom in included:
        by_segment.setdefault(atom["source_segment"], []).append(atom)

    def section(atom: Mapping[str, Any]) -> dict[str, Any]:
        statement = atom.get("statement")
        if statement is None and atom.get("source_excerpt_is_statement"):
            statement = atom["source_excerpt"]
        return {
            "heading": atom["kind"],
            "content": statement,
            "atom_ids": [atom["id"]],
            "timestamps": [atom["start"]],
        }

    return {
        "title": "三步表达系统",
        "overview": "记录、整理、复述，并保留条件、风险与推广信息。",
        "structure": ["三步方法", "案例结果", "边界风险", "推广"],
        "themes": [section(atom) for atom in included],
        "methods_and_steps": [section(atom) for atom in included if atom["kind"] == "step"],
        "examples_and_counterexamples": [
            section(atom) for atom in included if atom["kind"] == "example"
        ],
        "boundaries_and_risks": [
            section(atom) for atom in included if atom["kind"] in {"condition", "warning"}
        ],
        "concepts": ["表达系统", "复盘"],
        "quotes": ["不要删除失败记录，否则无法回滚"],
        "implications": ["生成理解：持续复述可以作为理解校验，但不是作者明示的普遍结论。"],
        "actions": ["按三个步骤完成一次复述并记录结果。"],
        "timeline": [
            {
                "segment_index": disposition["segment_index"],
                "start": float(disposition["segment_index"] * 10),
                "end": float((disposition["segment_index"] + 1) * 10),
                "disposition": disposition["disposition"],
                "atom_ids": [
                    atom["id"] for atom in by_segment.get(disposition["segment_index"], [])
                ],
                "content": "；".join(
                    (
                        atom.get("statement")
                        if atom.get("statement") is not None
                        else atom["source_excerpt"]
                    )
                    for atom in by_segment.get(disposition["segment_index"], [])
                )
                or "无内容",
            }
            for disposition in extraction["dispositions"]
        ],
        "promotions": [section(atom) for atom in included if atom["kind"] == "promotion"],
        "tags": ["表达系统", "复盘"],
        "included_atom_ids": included_ids,
    }


class ScriptedProvider:
    name = "scripted"
    model = "complete-v1"
    is_local = True
    context_tokens = 65_536

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.payloads: list[tuple[str, Mapping[str, Any]]] = []
        self.extraction: dict[str, Any] | None = None
        self.review_count = 0

    async def complete_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        json_schema: Mapping[str, Any],
        cancel_event: asyncio.Event,
    ) -> ProviderResponse:
        self.stages.append(stage)
        self.payloads.append((stage, payload))
        assert system_prompt
        assert json_schema
        if stage == "extract":
            atoms = [atom for segment in payload["segments"] for atom in _atom_for_segment(segment)]
            dispositions = [
                {
                    "segment_index": segment["index"],
                    "disposition": "promotion" if segment["index"] == 6 else "included",
                    "atom_ids": [atom["id"] for atom in _atom_for_segment(segment)],
                    "reason": None,
                }
                for segment in payload["segments"]
            ]
            parsed = {"atoms": atoms, "dispositions": dispositions}
        elif stage == "synthesize":
            self.extraction = dict(payload["extraction"])
            parsed = _synthesis(self.extraction, omit_warning=True)
        elif stage == "review":
            self.review_count += 1
            if payload.get("review_mode") == "draft_claims":
                claim_ids = [claim["id"] for claim in payload["draft_claims"]]
                parsed = {
                    "reviewed_claim_ids": claim_ids,
                    "supported_claim_ids": claim_ids,
                    "contradicted_claim_ids": [],
                }
                return ProviderResponse(
                    raw_output="scripted raw output",
                    parsed_json=parsed,
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            includes_warning_segment = any(
                segment["index"] == 5 for segment in payload["source_segments"]
            )
            parsed = {
                "reviewed_segment_indexes": [
                    segment["index"] for segment in payload["source_segments"]
                ],
                "source_omission_candidates": (
                    [
                        {
                            "segment_index": 5,
                            "source_excerpt": "注意不要删除失败记录，否则无法回滚。",
                            "missing_statement": "遗漏回滚警告",
                            "suggested_kind": "warning",
                        }
                    ]
                    if payload.get("review_pass") == 1 and includes_warning_segment
                    else []
                ),
                "unsupported_draft_claims": [],
                "disposition_corrections": [],
            }
        elif stage == "repair":
            self.extraction = dict(payload["extraction"])
            parsed = {
                "extraction_patch": {"atoms": [], "dispositions": []},
                "synthesis_patch": _synthesis(
                    self.extraction,
                    omit_warning=False,
                ),
            }
        else:
            raise AssertionError(stage)
        return ProviderResponse(
            raw_output="scripted raw output",
            parsed_json=parsed,
            usage=ProviderUsage(
                input_tokens=10, output_tokens=5, api_cost=0.0, cost_status="zero_local"
            ),
            diagnostics={"scripted": True},
        )

    async def unload(self) -> None:
        return None


class TwoSectionDepthRepairProvider(ScriptedProvider):
    """Exercise repair orchestration with two independent teaching sections."""

    supports_teaching_pipeline = True

    def __init__(self, later_failure: str | None = None) -> None:
        super().__init__()
        self.later_failure = later_failure
        self.repair_calls = 0
        self.depth_reviews = 0

    async def complete_json(self, **kwargs: Any) -> ProviderResponse:
        stage = kwargs["stage"]
        payload = kwargs["payload"]
        self.stages.append(stage)
        self.payloads.append((stage, payload))

        if stage == "extract":
            atoms = [
                {
                    "id": f"unit-{segment['index']}",
                    "kind": "claim",
                    "topic": "repair unit",
                    "statement": segment["text"],
                    "source_segment": segment["index"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "source_excerpt": segment["text"],
                    "relations": [],
                    "sequence": None,
                    "is_promotional": False,
                    "asr_uncertainty": [],
                }
                for segment in payload["segments"]
            ]
            parsed: dict[str, Any] = {
                "atoms": atoms,
                "dispositions": [
                    {
                        "segment_index": atom["source_segment"],
                        "disposition": "included",
                        "atom_ids": [atom["id"]],
                        "reason": None,
                    }
                    for atom in atoms
                ],
            }
        elif stage == "outline":
            atom_ids = [atom["id"] for atom in payload["atom_catalog"]]
            assignments = {
                "overview": atom_ids[:1],
                "actions": atom_ids[1:],
            }
            parsed = {
                "sections": [
                    {
                        "section_id": section_id,
                        "title": section_id,
                        "order": order,
                        "atom_ids": assignments.get(section_id, []),
                        "required_evidence_types": [],
                        "missing_evidence_policy": "视频未提供这一类信息。",
                        "timestamps": [],
                    }
                    for order, section_id in enumerate(
                        (
                            "overview",
                            "core_points",
                            "methods_steps",
                            "examples_results",
                            "boundaries_risks",
                            "actions",
                            "timeline",
                            "evidence_limits",
                        ),
                        1,
                    )
                ],
                "included_atom_ids": atom_ids,
                "unassigned_atom_ids": [],
                "input_sufficient": True,
            }
        elif stage == "section":
            atoms = payload["atoms"]
            parsed = {
                "section_id": payload["section"]["section_id"],
                "title": payload["section"]["title"],
                "content": "；".join(atom["statement"] for atom in atoms)
                or "视频未提供这一类信息。",
                "claims": [],
                "steps": [],
                "examples": [],
                "results": [],
                "conditions": [],
                "risks": [],
                "actions": [],
                "included_atom_ids": [atom["id"] for atom in atoms],
                "unprovided_items": [],
                "generated_interpretations": [],
            }
        elif stage == "depth_review":
            self.depth_reviews += 1
            parsed = {
                "abstracted_atom_ids": [],
                "example_without_result_atom_ids": [],
                "condition_without_scope_atom_ids": [],
                "reordered_step_atom_ids": [],
                "lost_literal_atom_ids": [],
                "lost_warning_or_promotion_atom_ids": [],
                "lost_uncertainty_markers": [],
                "unsupported_claims": [],
                "repair_section_ids": (
                    ["overview", "actions"] if self.depth_reviews == 1 else []
                ),
                "input_sufficient": True,
            }
        elif stage == "review":
            if payload.get("review_mode") == "draft_claims":
                claim_ids = [claim["id"] for claim in payload["draft_claims"]]
                parsed = {
                    "reviewed_claim_ids": claim_ids,
                    "supported_claim_ids": claim_ids,
                    "contradicted_claim_ids": [],
                }
            else:
                parsed = {
                    "reviewed_segment_indexes": [
                        segment["index"] for segment in payload["source_segments"]
                    ],
                    "source_omission_candidates": [],
                    "unsupported_draft_claims": [],
                    "disposition_corrections": [],
                }
        elif stage == "repair":
            self.repair_calls += 1
            if self.repair_calls == 2 and self.later_failure == "context":
                raise ProviderContextBudgetError(
                    "repair",
                    {"stage": "repair", "code": "scripted_later_unit_context_error"},
                )
            if self.repair_calls == 2 and self.later_failure == "provider":
                raise NoteProviderError("ollama", "repair", "scripted_later_unit_provider_error")
            repaired_sections = deepcopy(payload["section_drafts"])
            if self.repair_calls == 2 and self.later_failure == "unknown_atom":
                repaired_sections[0]["included_atom_ids"].append("unknown-repair-atom")
            else:
                for section in repaired_sections:
                    section["content"] = f"已修复：{section['content']}"
            parsed = {
                "extraction_patch": {"atoms": [], "dispositions": []},
                "synthesis_patch": _repair_unit_draft(payload),
                "section_drafts": repaired_sections,
            }
        else:
            raise AssertionError(stage)

        return ProviderResponse(
            raw_output="two-section-repair",
            parsed_json=parsed,
            usage=ProviderUsage(
                input_tokens=10,
                output_tokens=5,
                api_cost=0.0,
                cost_status="zero_local",
            ),
            diagnostics={"scripted": True},
        )


class TransitiveRelationRepairProvider(TwoSectionDepthRepairProvider):
    """Expose a repair relation chain across two otherwise disjoint sections."""

    async def complete_json(self, **kwargs: Any) -> ProviderResponse:
        response = await super().complete_json(**kwargs)
        stage = kwargs["stage"]
        payload = kwargs["payload"]
        if stage == "extract":
            atoms = [
                {
                    "id": f"unit-{segment['index']}",
                    "kind": "claim",
                    "topic": "relation chain",
                    "statement": segment["text"],
                    "source_segment": segment["index"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "source_excerpt": segment["text"],
                    "relations": (
                        [f"unit-{segment['index'] + 1}"]
                        if segment["index"] < 2
                        else []
                    ),
                    "sequence": None,
                    "is_promotional": False,
                    "asr_uncertainty": [],
                }
                for segment in payload["segments"]
            ]
            parsed = {
                "atoms": atoms,
                "dispositions": [
                    {
                        "segment_index": atom["source_segment"],
                        "disposition": "included",
                        "atom_ids": [atom["id"]],
                        "reason": None,
                    }
                    for atom in atoms
                ],
            }
            return replace(response, parsed_json=parsed)
        if stage == "outline":
            parsed = deepcopy(response.parsed_json)
            assignments = {
                "overview": ["B000-A001"],
                "core_points": ["B000-A002"],
                "actions": ["B000-A003"],
            }
            for section in parsed["sections"]:
                section["atom_ids"] = assignments.get(section["section_id"], [])
            parsed["included_atom_ids"] = ["B000-A001", "B000-A002", "B000-A003"]
            return replace(response, parsed_json=parsed)
        return response


def _repair_unit_draft(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a valid repair patch scoped to the evidence supplied to one unit."""

    patch = deepcopy(payload["draft"])
    atom_ids = {atom["id"] for atom in payload["extraction"]["atoms"]}
    segment_indexes = {
        atom["source_segment"] for atom in payload["extraction"]["atoms"]
    }
    for field_name in (
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "promotions",
    ):
        patch[field_name] = [
            {
                **section,
                "atom_ids": [
                    atom_id for atom_id in section["atom_ids"] if atom_id in atom_ids
                ],
            }
            for section in patch[field_name]
            if set(section["atom_ids"]) & atom_ids
        ]
    patch["timeline"] = [
        {
            **entry,
            "atom_ids": [atom_id for atom_id in entry["atom_ids"] if atom_id in atom_ids],
        }
        for entry in patch["timeline"]
        if entry["segment_index"] in segment_indexes
    ]
    patch["included_atom_ids"] = [
        atom_id for atom_id in patch["included_atom_ids"] if atom_id in atom_ids
    ]
    return patch


def _two_section_repair_input() -> NoteInput:
    texts = [
        "第一项独立证据。补充说明它用于核验前提与结果。",
        "第二项独立证据。补充说明它用于核验边界与后果。",
    ]
    return NoteInput(
        aweme_id="two-repair-units",
        transcript="".join(texts),
        segments=[
            InputSegment(index=index, start=float(index), end=float(index + 1), text=text)
            for index, text in enumerate(texts)
        ],
        metadata={},
        description="两个独立修复章节",
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=2_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )


def _transitive_relation_repair_input() -> NoteInput:
    texts = [
        "第一项链式证据。补充记录初始条件与观察结果。",
        "第二项链式证据。补充记录中间条件与观察结果。",
        "第三项链式证据。补充记录最终条件与观察结果。",
    ]
    return NoteInput(
        aweme_id="transitive-repair-units",
        transcript="".join(texts),
        segments=[
            InputSegment(index=index, start=float(index), end=float(index + 1), text=text)
            for index, text in enumerate(texts)
        ],
        metadata={},
        description="传递关系修复章节",
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=3_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )


@pytest.mark.asyncio
async def test_teaching_repair_sends_each_independent_section_as_a_local_unit() -> None:
    provider = TwoSectionDepthRepairProvider()

    result = await SemanticNotePipeline(provider).generate(
        _two_section_repair_input(),
        asyncio.Event(),
    )

    repair_payloads = [payload for stage, payload in provider.payloads if stage == "repair"]

    assert provider.repair_calls == 2
    assert [payload["repair_section_ids"] for payload in repair_payloads] == [
        ["overview"],
        ["actions"],
    ]
    assert [
        [section["section_id"] for section in payload["section_drafts"]]
        for payload in repair_payloads
    ] == [["overview"], ["actions"]]
    assert [
        {atom["id"] for atom in payload["extraction"]["atoms"]}
        for payload in repair_payloads
    ] == [{"B000-unit-0"}, {"B000-unit-1"}]
    assert result.coverage.quality_gate == "passed"
    assert result.coverage.semantic_complete is True


@pytest.mark.asyncio
async def test_teaching_repair_projects_coverage_to_each_local_unit() -> None:
    provider = TwoSectionDepthRepairProvider()

    await SemanticNotePipeline(provider).generate(
        _two_section_repair_input(),
        asyncio.Event(),
    )

    repair_payloads = [payload for stage, payload in provider.payloads if stage == "repair"]

    for payload in repair_payloads:
        coverage = payload["coverage_report"]
        local_atom_ids = {atom["id"] for atom in payload["extraction"]["atoms"]}
        local_segment_indexes = {
            segment["index"] for segment in payload["source_segments"]
        }

        assert coverage["segment_total"] == len(local_segment_indexes)
        assert coverage["evidence_atom_total"] == len(local_atom_ids)
        assert set(coverage["missing_atom_ids"]) <= local_atom_ids
        assert {
            item["segment_index"] for item in coverage["source_omission_candidates"]
        } <= local_segment_indexes
        assert coverage["depth_review"] is None


@pytest.mark.asyncio
async def test_teaching_repair_transitively_joins_sections_with_related_atoms() -> None:
    provider = TransitiveRelationRepairProvider()

    result = await SemanticNotePipeline(provider).generate(
        _transitive_relation_repair_input(),
        asyncio.Event(),
    )

    repair_payloads = [payload for stage, payload in provider.payloads if stage == "repair"]

    assert provider.repair_calls == 1
    assert repair_payloads[0]["repair_section_ids"] == ["overview", "actions"]
    assert {
        atom["id"] for atom in repair_payloads[0]["extraction"]["atoms"]
    } == {"B000-unit-0", "B000-unit-1", "B000-unit-2"}
    assert result.coverage.quality_gate == "passed"


@pytest.mark.asyncio
async def test_teaching_repair_keeps_earlier_synthesis_only_patch_after_section_patch() -> None:
    class MixedRepairPatchProvider(TwoSectionDepthRepairProvider):
        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            response = await super().complete_json(**kwargs)
            if kwargs["stage"] != "repair" or self.repair_calls != 1:
                return response

            parsed = deepcopy(response.parsed_json)
            parsed["section_drafts"] = []
            parsed["synthesis_patch"]["actions"] = ["第一项独立证据。"]
            return replace(response, parsed_json=parsed)

    provider = MixedRepairPatchProvider()

    result = await SemanticNotePipeline(provider).generate(
        _two_section_repair_input(),
        asyncio.Event(),
    )

    assert provider.repair_calls == 2
    assert "第一项独立证据。" in result.synthesis.actions
    assert result.coverage.quality_gate == "passed"
    assert result.coverage.semantic_complete is True


@pytest.mark.asyncio
@pytest.mark.parametrize("later_failure", ["unknown_atom", "context", "provider"])
async def test_later_repair_unit_failure_returns_an_unpublished_failed_candidate(
    later_failure: str,
) -> None:
    provider = TwoSectionDepthRepairProvider(later_failure=later_failure)

    result = await SemanticNotePipeline(provider).generate(
        _two_section_repair_input(),
        asyncio.Event(),
    )

    assert provider.repair_calls == 2
    assert result.context_status == "failed"
    assert result.coverage.quality_gate == "failed"
    assert result.coverage.semantic_complete is False
    assert result.repair_count == 0
    assert result.section_drafts == ()
    if later_failure == "provider":
        assert result.context_failures[-1] == {
            "stage": "repair",
            "code": "provider_error",
            "provider": "ollama",
            "provider_stage": "repair",
            "provider_code": "scripted_later_unit_provider_error",
            "repair_section_ids": ["actions"],
        }


@pytest.mark.asyncio
async def test_long_input_is_layered_without_any_over_budget_provider_call() -> None:
    class LongContextProvider(ScriptedProvider):
        context_tokens = 32_768

    provider = LongContextProvider()
    note_input = _long_note_input()

    result = await SemanticNotePipeline(provider).generate(note_input, asyncio.Event())

    # The synthetic fallback segment is readable but cannot claim semantic
    # completeness because the original segment coverage was unavailable.
    assert result.coverage.quality_gate == "failed"
    assert result.coverage.input_sufficient is False
    assert result.context_status == "within_budget"
    assert provider.stages.count("extract") > 1
    assert provider.stages.count("synthesize") > 1
    assert provider.stages.count("review") > 1
    assert (
        "".join(
            segment["text"]
            for stage, payload in provider.payloads
            if stage == "extract"
            for segment in payload["segments"]
        )
        == note_input.transcript
    )
    assert (
        "".join(
            segment["text"]
            for stage, payload in provider.payloads
            if stage == "review" and payload.get("review_mode") == "source_coverage"
            for segment in payload["source_segments"]
        )
        == note_input.transcript
    )
    assert all(
        payload["context_budget"]["within_budget"] is True for _, payload in provider.payloads
    )


@pytest.mark.asyncio
async def test_non_fallback_claim_review_routes_claims_to_evidence_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each claim is reviewed alongside its evidence, not every source batch."""

    import douyin_library.notes.pipeline as pipeline_module

    note_input = _note_input()
    source_batches = [
        SegmentBatch(
            0,
            0,
            3,
            tuple(note_input.segments[:4]),
            sum(len(segment.text) for segment in note_input.segments[:4]),
            100,
        ),
        SegmentBatch(
            1,
            4,
            6,
            tuple(note_input.segments[4:]),
            sum(len(segment.text) for segment in note_input.segments[4:]),
            100,
        ),
    ]
    extraction_payload = {
        "atoms": [
            atom
            for segment in note_input.segments
            for atom in _atom_for_segment(
                {
                    "index": segment.index,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
        ],
        "dispositions": [
            {
                "segment_index": segment.index,
                "disposition": "promotion" if segment.index == 6 else "included",
                "atom_ids": [
                    atom["id"]
                    for atom in _atom_for_segment(
                        {
                            "index": segment.index,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                        }
                    )
                ],
                "reason": None,
            }
            for segment in note_input.segments
        ],
    }
    extraction = ExtractionResult.model_validate(extraction_payload, strict=True)
    synthesis = SynthesisResult.model_validate(
        _synthesis(extraction_payload, omit_warning=False),
        strict=True,
    )
    claims = _draft_claims(synthesis)
    expected = [
        [
            claim.id
            for claim in _claims_for_source_batch(claims, extraction, batch)
        ]
        for batch in source_batches
    ]
    provider = ScriptedProvider()
    monkeypatch.setattr(
        pipeline_module,
        "build_segment_batches",
        lambda *_args, **_kwargs: source_batches,
    )

    await SemanticNotePipeline(provider)._review(
        note_input,
        extraction,
        synthesis,
        1,
        asyncio.Event(),
        None,
        [],
    )

    actual = [
        [claim["id"] for claim in payload["draft_claims"]]
        for stage, payload in provider.payloads
        if stage == "review" and payload.get("review_mode") == "draft_claims"
    ]
    assert actual == expected


@pytest.mark.asyncio
async def test_pipeline_extracts_all_batches_reviews_raw_source_and_repairs_once() -> None:
    note_input = _note_input()
    provider = ScriptedProvider()

    result = await SemanticNotePipeline(provider).generate(note_input, asyncio.Event())

    extraction_payloads = [payload for stage, payload in provider.payloads if stage == "extract"]
    assert extraction_payloads
    assert extraction_payloads[0]["prior_evidence_atoms"] == []
    if len(extraction_payloads) > 1:
        assert extraction_payloads[1]["prior_evidence_catalog"]
        assert all(
            atom["id"].startswith("B000-")
            for atom in extraction_payloads[1]["prior_evidence_catalog"]
        )
    assert [
        segment["text"] for payload in extraction_payloads for segment in payload["segments"]
    ] == [segment.text for segment in note_input.segments]
    review_payloads = [payload for stage, payload in provider.payloads if stage == "review"]
    assert {payload["review_pass"] for payload in review_payloads} == {1, 2}
    for review_pass in (1, 2):
        assert (
            "".join(
                segment["text"]
                for payload in review_payloads
                if payload["review_pass"] == review_pass
                and payload.get("review_mode") == "source_coverage"
                for segment in payload["source_segments"]
            )
            == note_input.transcript
        )
    assert provider.stages.count("repair") == 1
    repair_index = provider.stages.index("repair")
    assert set(provider.stages[repair_index + 1 :]) == {"review"}
    assert result.repair_count == 1
    assert result.coverage.quality_gate == "passed"
    assert result.coverage.source_segments_independently_reviewed == len(note_input.segments)
    assert len({atom.id for atom in result.extraction.atoms}) == len(result.extraction.atoms)
    theme_text = "\n".join(section.content for section in result.synthesis.themes)
    timeline_text = "\n".join(entry.content for entry in result.synthesis.timeline)
    for expected in (
        "第一步",
        "第二步",
        "第三步",
        "复述七天",
        "只有每周复盘",
        "不要删除失败记录",
        "关注我",
    ):
        assert expected in theme_text
        assert expected in timeline_text
    assert [entry.segment_index for entry in result.synthesis.timeline] == list(range(7))
    assert result.usage.input_tokens == 10 * len(provider.stages)
    assert len(result.stage_records) == len(provider.stages)
    assert len(result.batch_boundaries) == len(extraction_payloads)
    assert all(
        payload["context_budget"]["within_budget"] is True for _, payload in provider.payloads
    )


@pytest.mark.asyncio
async def test_legacy_provider_exposes_fixed_teaching_outline_placeholders() -> None:
    result = await SemanticNotePipeline(ScriptedProvider()).generate(_note_input(), asyncio.Event())

    assert [section.section_id for section in result.outline.sections] == [
        "overview",
        "core_points",
        "methods_steps",
        "examples_results",
        "boundaries_risks",
        "actions",
        "timeline",
        "evidence_limits",
    ]
    assert result.section_drafts == ()
    assert result.section_attempts == {}


@pytest.mark.asyncio
async def test_teaching_provider_synthesizes_each_fixed_section_independently() -> None:
    class TeachingProvider(ScriptedProvider):
        supports_teaching_pipeline = True

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            stage = kwargs["stage"]
            payload = kwargs["payload"]
            if stage not in {"outline", "section"}:
                if stage == "depth_review":
                    self.stages.append(stage)
                    self.payloads.append((stage, payload))
                    return ProviderResponse(
                        raw_output="depth",
                        parsed_json={
                            "abstracted_atom_ids": [],
                            "example_without_result_atom_ids": [],
                            "condition_without_scope_atom_ids": [],
                            "reordered_step_atom_ids": [],
                            "lost_literal_atom_ids": [],
                            "lost_warning_or_promotion_atom_ids": [],
                            "lost_uncertainty_markers": [],
                            "unsupported_claims": [],
                            "repair_section_ids": [],
                            "input_sufficient": True,
                        },
                        usage=ProviderUsage(
                            input_tokens=10,
                            output_tokens=5,
                            api_cost=0.0,
                            cost_status="zero_local",
                        ),
                        diagnostics={"scripted": True},
                    )
                return await super().complete_json(**kwargs)
            self.stages.append(stage)
            self.payloads.append((stage, payload))
            if stage == "outline":
                atom_ids = [atom["id"] for atom in payload["atom_catalog"]]
                section_ids = [
                    "overview",
                    "core_points",
                    "methods_steps",
                    "examples_results",
                    "boundaries_risks",
                    "actions",
                    "timeline",
                    "evidence_limits",
                ]
                parsed = {
                    "sections": [
                        {
                            "section_id": section_id,
                            "title": section_id,
                            "order": order,
                            "atom_ids": atom_ids if order == 1 else [],
                            "required_evidence_types": [],
                            "missing_evidence_policy": "视频未提供这一类信息。",
                            "timestamps": [],
                        }
                        for order, section_id in enumerate(section_ids, 1)
                    ],
                    "included_atom_ids": atom_ids,
                    "unassigned_atom_ids": [],
                    "input_sufficient": True,
                }
            else:
                section = payload["section"]
                atom_ids = [atom["id"] for atom in payload["atoms"]]
                parsed = {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "content": "；".join(atom["statement"] for atom in payload["atoms"])
                    or "视频未提供这一类信息。",
                    "claims": [],
                    "steps": [],
                    "examples": [],
                    "results": [],
                    "conditions": [],
                    "risks": [],
                    "actions": [],
                    "included_atom_ids": atom_ids,
                    "unprovided_items": [],
                    "generated_interpretations": [],
                }
            return ProviderResponse(
                raw_output="teaching",
                parsed_json=parsed,
                usage=ProviderUsage(
                    input_tokens=10,
                    output_tokens=5,
                    api_cost=0.0,
                    cost_status="zero_local",
                ),
                diagnostics={"scripted": True},
            )

    provider = TeachingProvider()
    result = await SemanticNotePipeline(provider).generate(_note_input(), asyncio.Event())

    assert provider.stages.count("outline") == 1
    assert provider.stages.count("section") == 8
    assert [draft.section_id for draft in result.section_drafts] == [
        section.section_id for section in result.outline.sections
    ]
    assert result.section_attempts == {section.section_id: 1 for section in result.outline.sections}
    assert set(result.synthesis.included_atom_ids) == {atom.id for atom in result.extraction.atoms}


@pytest.mark.asyncio
async def test_teaching_repair_keeps_legacy_synthesis_patch_without_section_drafts() -> None:
    class LegacyTeachingRepairProvider(ScriptedProvider):
        supports_teaching_pipeline = True

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            stage = kwargs["stage"]
            payload = kwargs["payload"]
            if stage == "outline":
                atom_ids = [atom["id"] for atom in payload["atom_catalog"]]
                section_ids = [
                    "overview",
                    "core_points",
                    "methods_steps",
                    "examples_results",
                    "boundaries_risks",
                    "actions",
                    "timeline",
                    "evidence_limits",
                ]
                parsed = {
                    "sections": [
                        {
                            "section_id": section_id,
                            "title": section_id,
                            "order": order,
                            "atom_ids": atom_ids if order == 1 else [],
                            "required_evidence_types": [],
                            "missing_evidence_policy": "视频未提供这一类信息。",
                            "timestamps": [],
                        }
                        for order, section_id in enumerate(section_ids, 1)
                    ],
                    "included_atom_ids": atom_ids,
                    "unassigned_atom_ids": [],
                    "input_sufficient": True,
                }
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                return ProviderResponse(
                    raw_output="legacy teaching outline",
                    parsed_json=parsed,
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            if stage == "section":
                section = payload["section"]
                atom_ids = [atom["id"] for atom in payload["atoms"]]
                parsed = {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "content": "；".join(atom["statement"] for atom in payload["atoms"])
                    or "视频未提供这一类信息。",
                    "claims": [],
                    "steps": [],
                    "examples": [],
                    "results": [],
                    "conditions": [],
                    "risks": [],
                    "actions": [],
                    "included_atom_ids": atom_ids,
                    "unprovided_items": [],
                    "generated_interpretations": [],
                }
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                return ProviderResponse(
                    raw_output="legacy teaching section",
                    parsed_json=parsed,
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            if stage == "depth_review":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                return ProviderResponse(
                    raw_output="legacy teaching depth review",
                    parsed_json={
                        "abstracted_atom_ids": [],
                        "example_without_result_atom_ids": [],
                        "condition_without_scope_atom_ids": [],
                        "reordered_step_atom_ids": [],
                        "lost_literal_atom_ids": [],
                        "lost_warning_or_promotion_atom_ids": [],
                        "lost_uncertainty_markers": [],
                        "unsupported_claims": [],
                        "repair_section_ids": [],
                        "input_sufficient": True,
                    },
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            if stage == "repair":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                repaired = dict(payload["draft"])
                repaired["actions"] = ["legacy repaired action"]
                return ProviderResponse(
                    raw_output="legacy teaching repair",
                    parsed_json={
                        "extraction_patch": {
                            "atoms": [],
                            "dispositions": [],
                        },
                        "synthesis_patch": repaired,
                    },
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            return await super().complete_json(**kwargs)

    result = await SemanticNotePipeline(LegacyTeachingRepairProvider()).generate(
        _note_input(), asyncio.Event()
    )

    assert result.repair_count == 1
    assert result.synthesis.actions == ["legacy repaired action"]


def test_input_sufficiency_rejects_short_repeated_filler_transcript() -> None:
    repeated = replace(
        _note_input(),
        transcript="嗯。" * 4,
        segments=[
            InputSegment(
                index=index,
                start=float(index * 60),
                end=float((index + 1) * 60),
                text="嗯。",
            )
            for index in range(4)
        ],
    )

    assert _input_is_sufficient(repeated) is False


def test_input_sufficiency_rejects_long_repeated_filler_across_segments() -> None:
    repeated = replace(
        _note_input(),
        transcript="嗯。" * 24,
        segments=[
            InputSegment(
                index=index,
                start=float(index),
                end=float(index + 1),
                text="嗯。",
            )
            for index in range(24)
        ],
    )

    assert _input_is_sufficient(repeated) is False


@pytest.mark.asyncio
async def test_teaching_pipeline_cannot_override_local_sparse_input_gate() -> None:
    repeated = replace(
        _note_input(),
        transcript="嗯。" * 4,
        segments=[
            InputSegment(
                index=index,
                start=float(index * 60),
                end=float((index + 1) * 60),
                text="嗯。",
            )
            for index in range(4)
        ],
    )

    result = await SemanticNotePipeline(ScriptedProvider()).generate(repeated, asyncio.Event())

    assert result.input_sufficient is False
    assert result.coverage.quality_gate == "failed"
    assert result.coverage.semantic_complete is False


def test_input_sufficiency_rejects_sparse_inputs_but_not_visual_dependency_alone() -> None:
    assert _input_is_sufficient(_note_input()) is True
    acknowledgement_only = replace(
        _note_input(),
        transcript="嗯。\n嗯。\n嗯。\n嗯。",
        segments=[
            InputSegment(index=index, start=float(index), end=float(index + 1), text="嗯。")
            for index in range(4)
        ],
    )
    assert _input_is_sufficient(acknowledgement_only) is False
    sparse = replace(_note_input(), warnings=["synthetic segment"])
    assert _input_is_sufficient(sparse) is False
    synthetic_variant = replace(_note_input(), transcript_variant="synthetic")
    assert _input_is_sufficient(synthetic_variant) is False
    visual = replace(
        _note_input(),
        metadata={"visual_dependency": {"risk": "high"}},
    )
    assert _input_is_sufficient(visual) is True
    budget = replace(
        _note_input(),
        note_budget={"visual_dependency": {"risk": "medium"}},
    )
    assert _input_is_sufficient(budget) is True


def test_request_fit_reuses_the_schema_for_repeated_budget_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LocalProvider:
        name = "ollama"
        model = "qwen3:4b"
        is_local = True
        context_tokens = 65_536

    calls = 0
    original = CoverageReview.model_json_schema

    def counting_schema(cls: type[CoverageReview]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(
        CoverageReview,
        "model_json_schema",
        classmethod(counting_schema),
    )
    pipeline = SemanticNotePipeline(LocalProvider())

    assert pipeline._request_fits("review", {}, CoverageReview) is True
    assert pipeline._request_fits("review", {}, CoverageReview) is True

    assert calls <= 1


def test_fallback_source_position_is_not_exposed_in_provider_schema() -> None:
    atom_schema = ExtractionResult.model_json_schema()["$defs"]["EvidenceAtom"]

    assert "source_start" not in atom_schema["properties"]


@pytest.mark.asyncio
async def test_extraction_fallback_applies_to_remaining_batches_after_two_invalid_responses() -> (
    None
):
    class StopAfterFallback(RuntimeError):
        pass

    class InvalidMultiBatchProvider:
        name = "ollama"
        model = "qwen3:4b"
        is_local = True
        context_tokens = 65_536
        supports_teaching_pipeline = True

        def __init__(self) -> None:
            self.extract_calls = 0
            self.stages: list[str] = []
            self.source_review_payloads: list[Mapping[str, Any]] = []

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            self.stages.append(kwargs["stage"])
            if kwargs["stage"] == "review":
                payload = kwargs["payload"]
                if payload["review_mode"] == "source_coverage":
                    self.source_review_payloads.append(payload)
                    return ProviderResponse(
                        raw_output="source coverage reviewed",
                        parsed_json={
                            "reviewed_segment_indexes": sorted(
                                {segment["index"] for segment in payload["source_segments"]}
                            ),
                            "source_omission_candidates": [],
                            "unsupported_draft_claims": [],
                            "disposition_corrections": [],
                        },
                        usage=ProviderUsage(
                            input_tokens=10,
                            output_tokens=5,
                            api_cost=0.0,
                            cost_status="zero_local",
                        ),
                        diagnostics={"scripted": True},
                    )
                raise StopAfterFallback
            assert kwargs["stage"] == "extract"
            self.extract_calls += 1
            return ProviderResponse(
                raw_output="invalid extraction",
                parsed_json={
                    "atoms": [],
                    "dispositions": [
                        {
                            "segment_index": segment["index"],
                            "disposition": "included",
                            "atom_ids": [],
                            "reason": None,
                        }
                        for segment in kwargs["payload"]["segments"]
                    ],
                },
                usage=ProviderUsage(
                    input_tokens=10,
                    output_tokens=5,
                    api_cost=0.0,
                    cost_status="zero_local",
                ),
                diagnostics={"scripted": True},
            )

        async def unload(self) -> None:
            return None

    text_a = "甲" * 20_000
    text_b = "乙" * 20_000
    note_input = NoteInput(
        aweme_id="multi-batch-fallback",
        transcript=text_a + text_b,
        segments=[
            InputSegment(index=0, start=0.0, end=60.0, text=text_a),
            InputSegment(index=1, start=60.0, end=120.0, text=text_b),
        ],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=120_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    provider = InvalidMultiBatchProvider()

    with pytest.raises(StopAfterFallback):
        await SemanticNotePipeline(provider).generate(note_input, asyncio.Event())

    assert provider.extract_calls == 2
    assert 2 <= len(provider.source_review_payloads) <= 80
    review_atom_ids = [
        {atom["id"] for atom in payload["extraction"]["atoms"]}
        for payload in provider.source_review_payloads
    ]
    assert all(review_atom_ids)
    assert all(
        all(len(segment["text"]) <= 512 for segment in payload["source_segments"])
        for payload in provider.source_review_payloads
    )
    assert all(
        len(payload["extraction"]["atoms"]) == len(payload["source_segments"])
        for payload in provider.source_review_payloads
    )
    assert all(
        all(disposition["atom_ids"] for disposition in payload["extraction"]["dispositions"])
        for payload in provider.source_review_payloads
    )
    assert all(
        all(entry["atom_ids"] for entry in payload["draft"]["timeline"])
        for payload in provider.source_review_payloads
    )
    assert sum(map(len, review_atom_ids)) == len(set().union(*review_atom_ids))
    assert len(set().union(*review_atom_ids)) == 80
    assert "outline" not in provider.stages
    assert "section" not in provider.stages
    assert "depth_review" not in provider.stages


@pytest.mark.asyncio
async def test_later_extraction_failure_rebuilds_all_batches_as_source_fallback() -> None:
    class StopAfterFallback(RuntimeError):
        pass

    class ValidThenInvalidProvider:
        name = "ollama"
        model = "qwen3:4b"
        is_local = True
        context_tokens = 65_536
        supports_teaching_pipeline = True

        def __init__(self) -> None:
            self.extract_calls = 0
            self.stages: list[str] = []
            self.first_review_payload: Mapping[str, Any] | None = None

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            stage = kwargs["stage"]
            payload = kwargs["payload"]
            self.stages.append(stage)
            if stage == "review":
                self.first_review_payload = payload
                raise StopAfterFallback
            assert stage == "extract"
            self.extract_calls += 1
            if self.extract_calls == 1:
                segment = payload["segments"][0]
                parsed_json = {
                    "atoms": [
                        {
                            "id": "provider-first",
                            "kind": "claim",
                            "topic": "provider",
                            "statement": segment["text"],
                            "source_segment": segment["index"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "source_excerpt": segment["text"],
                        }
                    ],
                    "dispositions": [
                        {
                            "segment_index": segment["index"],
                            "disposition": "included",
                            "atom_ids": ["provider-first"],
                            "reason": None,
                        }
                    ],
                }
            else:
                parsed_json = {
                    "atoms": [],
                    "dispositions": [
                        {
                            "segment_index": segment["index"],
                            "disposition": "included",
                            "atom_ids": [],
                            "reason": None,
                        }
                        for segment in payload["segments"]
                    ],
                }
            return ProviderResponse(
                raw_output="scripted extraction",
                parsed_json=parsed_json,
                usage=ProviderUsage(
                    input_tokens=10,
                    output_tokens=5,
                    api_cost=0.0,
                    cost_status="zero_local",
                ),
                diagnostics={"scripted": True},
            )

        async def unload(self) -> None:
            return None

    first_text = "A" * 18_000
    second_text = "B" * 18_000
    note_input = NoteInput(
        aweme_id="later-batch-fallback",
        transcript=first_text + second_text,
        segments=[
            InputSegment(index=0, start=0.0, end=60.0, text=first_text),
            InputSegment(index=1, start=60.0, end=120.0, text=second_text),
        ],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=120_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    assert (
        len(
            build_segment_batches(
                note_input,
                ValidThenInvalidProvider.context_tokens,
                input_fraction=0.35,
            )
        )
        == 2
    )
    provider = ValidThenInvalidProvider()

    with pytest.raises(StopAfterFallback):
        await SemanticNotePipeline(provider).generate(note_input, asyncio.Event())

    assert provider.extract_calls == 3
    assert "outline" not in provider.stages
    assert "section" not in provider.stages
    assert "depth_review" not in provider.stages
    assert provider.first_review_payload is not None
    assert {
        item["reason"] for item in provider.first_review_payload["extraction"]["dispositions"]
    } == {"lossless source-preserving local fallback"}


def test_source_fallback_teaching_structure_passes_deterministic_depth_review() -> None:
    source = "这是方法。第一步，记录问题。例如连续复述七天；注意不要删除记录。"
    batch = SegmentBatch(
        batch_index=0,
        start_segment=0,
        end_segment=0,
        segments=(InputSegment(index=0, start=0.0, end=8.0, text=source),),
        input_chars=len(source),
        estimated_tokens=20,
    )
    extraction = _merge_extractions([(batch, _canonical_source_extraction(batch))])

    outline = _source_fallback_outline(extraction)
    drafts = _source_fallback_section_drafts(extraction, outline)
    review = DepthValidator().validate(extraction.atoms, outline, list(drafts))

    by_id = {draft.section_id: draft for draft in drafts}
    assert by_id["methods_steps"].steps
    assert by_id["examples_results"].examples
    assert by_id["examples_results"].results
    assert by_id["boundaries_risks"].risks
    anchor_ids = {atom.id for atom in extraction.atoms if atom.kind in {"claim", "step"}}
    for atom in extraction.atoms:
        if atom.kind in {"example", "condition", "warning"}:
            assert len(atom.relations) == 1
            assert atom.relations[0] in anchor_ids
            assert atom.relations[0] != atom.id
    assert review.repair_section_ids == []
    assert review.abstracted_atom_ids == []
    assert review.example_without_result_atom_ids == []
    assert review.condition_without_scope_atom_ids == []


def test_canonical_fallback_leaves_relations_empty_without_distinct_anchor() -> None:
    source = "例如只复述一次；如果没有记录；注意不要删除。"
    batch = SegmentBatch(
        batch_index=0,
        start_segment=0,
        end_segment=0,
        segments=(InputSegment(index=0, start=0.0, end=3.0, text=source),),
        input_chars=len(source),
        estimated_tokens=20,
    )

    extraction = _canonical_source_extraction(batch)

    assert {atom.kind for atom in extraction.atoms} == {
        "example",
        "condition",
        "warning",
    }
    assert all(atom.relations == [] for atom in extraction.atoms)


def test_source_fallback_claim_review_routes_only_relevant_atoms_per_batch() -> None:
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "A",
                    "kind": "claim",
                    "topic": "一",
                    "statement": "第一段",
                    "source_segment": 0,
                    "source_excerpt": "第一段",
                },
                {
                    "id": "B",
                    "kind": "claim",
                    "topic": "二",
                    "statement": "第二段",
                    "source_segment": 1,
                    "source_excerpt": "第二段",
                },
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["A"],
                    "reason": "lossless source-preserving local fallback",
                },
                {
                    "segment_index": 1,
                    "disposition": "included",
                    "atom_ids": ["B"],
                    "reason": "lossless source-preserving local fallback",
                },
            ],
        },
        strict=True,
    )
    claims = [
        {"id": "C-title", "field": "title", "text": "标题", "atom_ids": []},
        {"id": "C-a", "field": "themes", "text": "第一段", "atom_ids": ["A"]},
        {"id": "C-b", "field": "themes", "text": "第二段", "atom_ids": ["B"]},
    ]
    first = SegmentBatch(0, 0, 0, (InputSegment(index=0, start=0.0, end=1.0, text="第一段"),), 3, 2)
    second = SegmentBatch(
        1, 1, 1, (InputSegment(index=1, start=1.0, end=2.0, text="第二段"),), 3, 2
    )

    assert [item["id"] for item in _claims_for_source_batch(claims, extraction, first)] == [
        "C-title",
        "C-a",
    ]
    assert [item["id"] for item in _claims_for_source_batch(claims, extraction, second)] == ["C-b"]


def test_source_fallback_claim_routing_distinguishes_fragments_of_one_segment() -> None:
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "A",
                    "kind": "claim",
                    "topic": "一",
                    "statement": "前半段事实",
                    "source_segment": 0,
                    "source_excerpt": "前半段事实",
                },
                {
                    "id": "B",
                    "kind": "claim",
                    "topic": "二",
                    "statement": "后半段事实",
                    "source_segment": 0,
                    "source_excerpt": "后半段事实",
                },
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["A", "B"],
                    "reason": "lossless source-preserving local fallback",
                }
            ],
        },
        strict=True,
    )
    claims = [
        {"id": "C-title", "field": "title", "text": "标题", "atom_ids": []},
        {
            "id": "C-overview",
            "field": "overview",
            "text": "跨来源概述",
            "atom_ids": [],
        },
        {"id": "C-a", "field": "themes", "text": "前半段事实", "atom_ids": ["A"]},
        {"id": "C-b", "field": "themes", "text": "后半段事实", "atom_ids": ["B"]},
    ]
    first = SegmentBatch(
        0,
        0,
        0,
        (InputSegment(index=0, start=0.0, end=1.0, text="前半段事实"),),
        5,
        3,
    )
    second = SegmentBatch(
        1,
        0,
        0,
        (InputSegment(index=0, start=0.0, end=1.0, text="后半段事实"),),
        5,
        3,
    )

    assert [item["id"] for item in _claims_for_source_batch(claims, extraction, first)] == [
        "C-title",
        "C-overview",
        "C-a",
    ]
    assert [item["id"] for item in _claims_for_source_batch(claims, extraction, second)] == ["C-b"]


def test_source_review_payload_distinguishes_fragments_of_one_segment() -> None:
    note_input = NoteInput(
        aweme_id="fragment-source-review",
        transcript="前半段事实后半段事实",
        segments=[
            InputSegment(
                index=0,
                start=0.0,
                end=2.0,
                text="前半段事实后半段事实",
            )
        ],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=2_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "A",
                    "kind": "claim",
                    "topic": "一",
                    "statement": "前半段事实",
                    "source_segment": 0,
                    "source_excerpt": "前半段事实",
                },
                {
                    "id": "B",
                    "kind": "claim",
                    "topic": "二",
                    "statement": "后半段事实",
                    "source_segment": 0,
                    "source_excerpt": "后半段事实",
                },
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["A", "B"],
                    "reason": "lossless source-preserving local fallback",
                }
            ],
        },
        strict=True,
    )
    outline = _source_fallback_outline(extraction)
    drafts = _source_fallback_section_drafts(extraction, outline)
    synthesis = _teaching_synthesis(note_input, extraction, outline, drafts)
    batches = [
        SegmentBatch(
            0,
            0,
            0,
            (InputSegment(0, 0.0, 2.0, "前半段事实"),),
            5,
            3,
        ),
        SegmentBatch(
            1,
            0,
            0,
            (InputSegment(0, 0.0, 2.0, "后半段事实"),),
            5,
            3,
        ),
    ]

    payloads = [
        _source_review_payload(note_input, extraction, synthesis, batch, 1) for batch in batches
    ]
    payload_atom_ids = [
        {atom["id"] for atom in payload["extraction"]["atoms"]} for payload in payloads
    ]
    timeline_atom_ids = [set(payload["draft"]["timeline"][0]["atom_ids"]) for payload in payloads]
    section_atom_ids = [
        {
            atom_id
            for field_name in (
                "themes",
                "methods_and_steps",
                "examples_and_counterexamples",
                "boundaries_and_risks",
                "promotions",
            )
            for section in payload["draft"][field_name]
            for atom_id in section["atom_ids"]
        }
        for payload in payloads
    ]
    claims = [
        DraftClaim.model_validate(
            {
                "id": "C-title",
                "field": "title",
                "text": "标题",
                "atom_ids": [],
            },
            strict=True,
        ),
        DraftClaim.model_validate(
            {
                "id": "C-A",
                "field": "themes",
                "text": "前半段事实",
                "atom_ids": ["A"],
            },
            strict=True,
        ),
        DraftClaim.model_validate(
            {
                "id": "C-B",
                "field": "themes",
                "text": "后半段事实",
                "atom_ids": ["B"],
            },
            strict=True,
        ),
    ]
    claim_payloads = [
        _claim_review_payload(
            note_input,
            extraction,
            batch,
            1,
            _claims_for_source_batch(claims, extraction, batch),
        )
        for batch in batches
    ]

    assert payload_atom_ids == [{"A"}, {"B"}]
    assert payload_atom_ids[0].isdisjoint(payload_atom_ids[1])
    assert set().union(*payload_atom_ids) == {"A", "B"}
    assert timeline_atom_ids == [{"A"}, {"B"}]
    assert section_atom_ids == [{"A"}, {"B"}]
    assert [set(payload["draft"]["included_atom_ids"]) for payload in payloads] == [
        {"A"},
        {"B"},
    ]
    assert [
        {atom["id"] for atom in payload["extraction"]["atoms"]} for payload in claim_payloads
    ] == [{"A"}, {"B"}]
    assert [[claim["id"] for claim in payload["draft_claims"]] for payload in claim_payloads] == [
        ["C-title", "C-A"],
        ["C-B"],
    ]


def test_source_review_routes_repeated_fallback_atoms_by_fragment_position() -> None:
    source = "甲" * 1_024
    original = InputSegment(index=0, start=0.0, end=2.0, text=source)
    original_batch = SegmentBatch(0, 0, 0, (original,), len(source), 683)
    extraction = _canonical_source_extraction(original_batch)
    outline = _source_fallback_outline(extraction)
    synthesis = _teaching_synthesis(
        NoteInput(
            aweme_id="repeated-fallback-fragments",
            transcript=source,
            segments=[original],
            metadata={},
            description=None,
            author=None,
            author_unique_id=None,
            published_at=None,
            duration_ms=2_000,
            source_url=None,
            source_files={},
            asset_records={},
            warnings=[],
        ),
        extraction,
        outline,
        _source_fallback_section_drafts(extraction, outline),
    )
    fragments = _split_segment_batch(original_batch)
    atom_ids = [atom.id for atom in extraction.atoms]
    note_input = NoteInput(
        aweme_id="repeated-fallback-fragments",
        transcript=source,
        segments=[original],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=2_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    claims = [
        DraftClaim.model_validate(
            {"id": "C-title", "field": "title", "text": "标题", "atom_ids": []},
            strict=True,
        ),
        DraftClaim.model_validate(
            {
                "id": "C-first",
                "field": "themes",
                "text": source[:512],
                "atom_ids": [atom_ids[0]],
            },
            strict=True,
        ),
        DraftClaim.model_validate(
            {
                "id": "C-second",
                "field": "themes",
                "text": source[512:],
                "atom_ids": [atom_ids[1]],
            },
            strict=True,
        ),
    ]

    payloads = [
        _source_review_payload(note_input, extraction, synthesis, fragment, 1)
        for fragment in fragments
    ]

    assert [{atom["id"] for atom in payload["extraction"]["atoms"]} for payload in payloads] == [
        {atom_ids[0]},
        {atom_ids[1]},
    ]
    assert [
        [claim.id for claim in _claims_for_source_batch(claims, extraction, fragment)]
        for fragment in fragments
    ] == [["C-title", "C-first"], ["C-second"]]


def test_sliced_fallback_timeline_compacts_to_selected_atom_text() -> None:
    source = "甲" * 1_024
    original = InputSegment(index=0, start=0.0, end=2.0, text=source)
    batch = SegmentBatch(0, 0, 0, (original,), len(source), 683)
    extraction = _canonical_source_extraction(batch)
    note_input = NoteInput(
        aweme_id="compact-fallback-timeline",
        transcript=source,
        segments=[original],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=2_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    outline = _source_fallback_outline(extraction)
    synthesis = _teaching_synthesis(
        note_input,
        extraction,
        outline,
        _source_fallback_section_drafts(extraction, outline),
    )

    sliced = _slice_synthesis(
        synthesis,
        {0},
        {extraction.atoms[0].id},
        extraction,
    )
    projected = _project_synthesis(sliced, extraction, compact_content=True)

    assert len(sliced.timeline) == 1
    assert sliced.timeline[0].content == source[:512]
    assert projected["timeline"][0]["content_from_atom_ids"] == [extraction.atoms[0].id]
    assert "content" not in projected["timeline"][0]


def test_fallback_review_batches_preserve_original_source_whitespace() -> None:
    source = "  第一条。 \n 第二条。  "
    original = InputSegment(index=0, start=0.0, end=2.0, text=source)
    extraction = _canonical_source_extraction(SegmentBatch(0, 0, 0, (original,), len(source), 20))
    note_input = NoteInput(
        aweme_id="preserve-fallback-whitespace",
        transcript=source,
        segments=[original],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=2_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )

    batches = _fallback_source_review_batches(
        note_input,
        extraction,
        context_tokens=65_536,
        input_fraction=0.2,
    )

    assert "".join(segment.text for batch in batches for segment in batch.segments) == source


def test_fallback_review_batches_preserve_transcript_whitespace_between_segments() -> None:
    source = "第一条。\n\n第二条。"
    segments = [
        InputSegment(index=0, start=0.0, end=1.0, text="第一条。"),
        InputSegment(index=1, start=1.0, end=2.0, text="第二条。"),
    ]
    note_input = NoteInput(
        aweme_id="preserve-fallback-boundary-whitespace",
        transcript=source,
        segments=segments,
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=2_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    batch = build_segment_batches(note_input, context_tokens=65_536)[0]
    extraction = _canonical_source_extraction(batch)

    batches = _fallback_source_review_batches(
        note_input,
        extraction,
        context_tokens=65_536,
        input_fraction=0.2,
    )

    assert batches is not None
    assert "".join(segment.text for batch in batches for segment in batch.segments) == source


def test_fallback_review_batches_fail_closed_for_whitespace_only_segment() -> None:
    segments = [
        InputSegment(index=0, start=0.0, end=1.0, text="第一条。"),
        InputSegment(index=1, start=1.0, end=2.0, text="   "),
        InputSegment(index=2, start=2.0, end=3.0, text="第二条。"),
    ]
    source = "".join(segment.text for segment in segments)
    note_input = NoteInput(
        aweme_id="fallback-whitespace-segment",
        transcript=source,
        segments=segments,
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=3_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    extraction = _canonical_source_extraction(
        SegmentBatch(0, 0, 2, tuple(segments), len(source), 20)
    )

    assert (
        _fallback_source_review_batches(
            note_input,
            extraction,
            context_tokens=65_536,
            input_fraction=0.2,
        )
        == []
    )


@pytest.mark.asyncio
async def test_review_stops_after_irreducible_preflight_budget_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "来源事实。"
    segment = InputSegment(index=0, start=0.0, end=1.0, text=source)
    note_input = NoteInput(
        aweme_id="review-budget-failure",
        transcript=source,
        segments=[segment],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=1_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    batch = build_segment_batches(note_input, context_tokens=65_536)[0]
    extraction = _merge_extractions([(batch, _canonical_source_extraction(batch))])
    outline = _source_fallback_outline(extraction)
    synthesis = _teaching_synthesis(
        note_input,
        extraction,
        outline,
        _source_fallback_section_drafts(extraction, outline),
    )

    class NoProviderCalls:
        name = "ollama"
        model = "qwen3:4b"
        is_local = True
        context_tokens = 65_536

        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            raise AssertionError("an irreducible preflight failure must not call the provider")

        async def unload(self) -> None:
            return None

    provider = NoProviderCalls()
    pipeline = SemanticNotePipeline(provider)
    monkeypatch.setattr(pipeline, "_request_fits", lambda *args: False)
    context_failures: list[dict[str, Any]] = []

    review = await pipeline._review(
        note_input,
        extraction,
        synthesis,
        1,
        asyncio.Event(),
        None,
        [],
        context_failures,
    )

    assert provider.calls == 0
    assert review.reviewed_segment_indexes == []
    assert len(context_failures) == 1
    assert context_failures[0]["stage"] == "review"


@pytest.mark.asyncio
async def test_generation_fails_closed_when_segments_skip_nonwhitespace_source() -> None:
    source = "已覆盖的事实。未覆盖的事实。"
    note_input = NoteInput(
        aweme_id="source-provenance-failure",
        transcript=source,
        segments=[InputSegment(index=0, start=0.0, end=1.0, text="已覆盖的事实。")],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=1_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )

    class NoProviderCalls:
        name = "ollama"
        model = "qwen3:4b"
        is_local = True
        context_tokens = 65_536

        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            raise AssertionError("unmapped non-whitespace source must not reach provider")

        async def unload(self) -> None:
            return None

    provider = NoProviderCalls()
    result = await SemanticNotePipeline(provider).generate(note_input, asyncio.Event())

    assert provider.calls == 0
    assert result.context_status == "failed"
    assert result.context_failures[0]["code"] == "source_provenance_unverified"


def test_fallback_review_batches_keep_atoms_intact_across_initial_split() -> None:
    source = "甲" * 34_407
    original = InputSegment(index=0, start=0.0, end=120.0, text=source)
    note_input = NoteInput(
        aweme_id="fallback-initial-split",
        transcript=source,
        segments=[original],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=120_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    initial_batches = build_segment_batches(
        note_input,
        context_tokens=65_536,
        input_fraction=0.35,
    )
    assert [
        (segment.source_offset, len(segment.text))
        for batch in initial_batches
        for segment in batch.segments
    ] == [(0, 34_405), (34_405, 2)]
    extraction = _merge_extractions(
        [(batch, _canonical_source_extraction(batch)) for batch in initial_batches]
    )
    outline = _source_fallback_outline(extraction)
    synthesis = _teaching_synthesis(
        note_input,
        extraction,
        outline,
        _source_fallback_section_drafts(extraction, outline),
    )

    review_batches = _fallback_source_review_batches(
        note_input,
        extraction,
        context_tokens=65_536,
        input_fraction=0.2,
    )
    payloads = [
        _source_review_payload(note_input, extraction, synthesis, batch, 1)
        for batch in review_batches
    ]

    assert "".join(segment.text for batch in review_batches for segment in batch.segments) == source
    payload_atom_ids = [
        {atom["id"] for atom in payload["extraction"]["atoms"]} for payload in payloads
    ]
    assert all(payload_atom_ids)
    assert set().union(*payload_atom_ids) == {atom.id for atom in extraction.atoms}
    assert sum(map(len, payload_atom_ids)) == len(extraction.atoms)
    assert all(
        all(disposition["atom_ids"] for disposition in payload["extraction"]["dispositions"])
        for payload in payloads
    )
    assert all(
        all(entry["atom_ids"] for entry in payload["draft"]["timeline"]) for payload in payloads
    )


@pytest.mark.asyncio
async def test_source_fallback_review_ignores_source_text_false_positive_when_claim_is_supported(
) -> None:
    segment = InputSegment(
        index=0,
        start=0.0,
        end=1.0,
        text="来源只陈述这一条事实。",
    )
    note_input = NoteInput(
        aweme_id="fallback-review",
        transcript=segment.text,
        segments=[segment],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=1_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    batch = SegmentBatch(0, 0, 0, (segment,), len(segment.text), 8)
    extraction = _canonical_source_extraction(batch)
    outline = _source_fallback_outline(extraction)
    synthesis = _teaching_synthesis(
        note_input,
        extraction,
        outline,
        _source_fallback_section_drafts(extraction, outline),
    )

    class FallbackReviewProvider:
        name = "fallback-review"
        model = "fallback-review-v1"
        is_local = True
        context_tokens = 65_536

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            payload = kwargs["payload"]
            if payload["review_mode"] == "source_coverage":
                parsed = {
                    "reviewed_segment_indexes": [0],
                    "source_omission_candidates": [
                        {
                            "segment_index": 0,
                            "source_excerpt": segment.text,
                            "missing_statement": segment.text,
                            "suggested_kind": "claim",
                        }
                    ],
                    "unsupported_draft_claims": [
                        *(atom["id"] for atom in payload["extraction"]["atoms"]),
                        payload["source_segments"][0]["text"],
                        "概述",
                        "核心要点",
                        "方法步骤",
                        "示例与结果",
                        "边界与风险",
                        "行动建议",
                        "时间线",
                        "证据限制",
                        "视频未提供这一类信息。",
                    ],
                    "disposition_corrections": [],
                }
            else:
                claim_ids = [claim["id"] for claim in payload["draft_claims"]]
                parsed = {
                    "reviewed_claim_ids": claim_ids,
                    "supported_claim_ids": claim_ids,
                    "contradicted_claim_ids": [],
                }
            return ProviderResponse(
                raw_output="fallback review",
                parsed_json=parsed,
                usage=ProviderUsage(
                    input_tokens=1,
                    output_tokens=1,
                    api_cost=0.0,
                    cost_status="zero_local",
                ),
                diagnostics={"scripted": True},
            )

    review = await SemanticNotePipeline(FallbackReviewProvider())._review(
        note_input,
        extraction,
        synthesis,
        1,
        asyncio.Event(),
        None,
        [],
    )

    assert review.unsupported_draft_claims == []
    assert review.source_omission_candidates == []


@pytest.mark.asyncio
async def test_source_fallback_review_fails_closed_when_atom_provenance_is_unverified() -> None:
    segment = InputSegment(
        index=0,
        start=0.0,
        end=1.0,
        text="来源只陈述这一条事实。",
    )
    note_input = NoteInput(
        aweme_id="fallback-provenance-unverified",
        transcript=segment.text,
        segments=[segment],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=1_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    batch = SegmentBatch(0, 0, 0, (segment,), len(segment.text), 8)
    extraction = _canonical_source_extraction(batch)
    extraction = extraction.model_copy(
        update={
            "atoms": [extraction.atoms[0].model_copy(update={"source_start": len(segment.text)})]
        }
    )
    outline = _source_fallback_outline(extraction)
    synthesis = _teaching_synthesis(
        note_input,
        extraction,
        outline,
        _source_fallback_section_drafts(extraction, outline),
    )

    class NoProviderCalls:
        name = "ollama"
        model = "qwen3:4b"
        is_local = True
        context_tokens = 65_536

        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            raise AssertionError("unverified fallback provenance must not reach provider")

    provider = NoProviderCalls()
    context_failures: list[dict[str, Any]] = []
    review = await SemanticNotePipeline(provider)._review(
        note_input,
        extraction,
        synthesis,
        1,
        asyncio.Event(),
        None,
        [],
        context_failures,
    )

    assert provider.calls == 0
    assert review.reviewed_segment_indexes == []
    assert context_failures[0]["code"] == "fallback_source_provenance_unverified"


def test_draft_claims_infer_atom_ids_from_evidence_sections() -> None:
    synthesis = SynthesisResult.model_validate(
        {
            "title": "学习笔记",
            "overview": "第一段；执行第二步；原话",
            "structure": [],
            "themes": [
                {
                    "heading": "要点",
                    "content": "第一段",
                    "atom_ids": ["A"],
                },
                {
                    "heading": "行动",
                    "content": "执行第二步",
                    "atom_ids": ["B"],
                },
                {
                    "heading": "引用",
                    "content": "原话",
                    "atom_ids": ["C"],
                },
                {
                    "heading": "概念名",
                    "content": "概念定义",
                    "atom_ids": ["D"],
                },
            ],
            "methods_and_steps": [],
            "examples_and_counterexamples": [],
            "boundaries_and_risks": [],
            "concepts": ["概念名"],
            "quotes": ["原话"],
            "implications": [],
            "actions": ["执行第二步"],
            "timeline": [],
            "promotions": [],
            "tags": [],
            "included_atom_ids": ["A", "B", "C", "D"],
        },
        strict=True,
    )

    claims = _draft_claims(synthesis)
    by_field = {claim.field: claim for claim in claims if claim.field != "themes"}

    assert by_field["overview"].atom_ids == ["A", "B", "C"]
    assert by_field["quotes"].atom_ids == ["C"]
    assert by_field["actions"].atom_ids == ["B"]
    assert by_field["concepts"].atom_ids == ["D"]


def test_normalize_section_keeps_all_outline_atoms_when_provider_omits_one() -> None:
    atom_a = {
        "id": "A",
        "kind": "claim",
        "topic": "topic",
        "statement": "A",
        "source_segment": 0,
        "source_excerpt": "A",
    }
    atom_b = {**atom_a, "id": "B", "statement": "B", "source_excerpt": "B"}
    outline_section = TeachingOutlineSection(
        section_id="core_points",
        title="Core",
        order=1,
        atom_ids=["A", "B"],
        required_evidence_types=["claim"],
        missing_evidence_policy="视频未提供这一类信息。",
    )
    draft = SectionDraft(
        section_id="core_points",
        title="Core",
        content="A",
        included_atom_ids=["A"],
    )

    normalized = _normalize_section(
        draft,
        outline_section,
        {"A": atom_a, "B": atom_b},
    )

    assert normalized.included_atom_ids == ["A", "B"]


def test_normalize_section_replaces_template_only_content_with_source_evidence() -> None:
    atom = {
        "id": "A",
        "kind": "claim",
        "topic": "topic",
        "statement": "这是源材料中的完整事实。",
        "source_segment": 0,
        "source_excerpt": "这是源材料中的完整事实。",
    }
    outline_section = TeachingOutlineSection(
        section_id="core_points",
        title="核心要点",
        order=1,
        atom_ids=["A"],
        required_evidence_types=["claim"],
        missing_evidence_policy="视频未提供这一类信息。",
    )
    draft = SectionDraft(
        section_id="core_points",
        title="核心要点",
        content="核心要点",
        included_atom_ids=["A"],
        generated_interpretations=["视频原文：这是源材料中的完整事实。"],
    )

    normalized = _normalize_section(draft, outline_section, {"A": atom})

    assert normalized.content == "这是源材料中的完整事实。"
    assert normalized.generated_interpretations == []


def test_normalize_empty_section_uses_missing_evidence_policy() -> None:
    outline_section = TeachingOutlineSection(
        section_id="methods_steps",
        title="方法步骤",
        order=3,
        atom_ids=[],
        required_evidence_types=["step"],
        missing_evidence_policy="视频未提供这一类信息。",
    )
    draft = SectionDraft(
        section_id="methods_steps",
        title="方法步骤",
        content="方法步骤",
        included_atom_ids=[],
        generated_interpretations=["视频未提供这一类信息。"],
    )

    normalized = _normalize_section(draft, outline_section, {})

    assert normalized.content == "视频未提供这一类信息。"
    assert normalized.generated_interpretations == []


def test_normalize_evidenced_section_replaces_false_missing_placeholder() -> None:
    atom = {
        "id": "A",
        "kind": "step",
        "topic": "步骤",
        "statement": "当你不再执着时，就迈出了第一步。",
        "source_segment": 0,
        "source_excerpt": "当你不再执着时，就迈出了第一步。",
    }
    outline_section = TeachingOutlineSection(
        section_id="methods_steps",
        title="方法步骤",
        order=3,
        atom_ids=["A"],
        required_evidence_types=["step"],
        missing_evidence_policy="视频未提供这一类信息。",
    )
    draft = SectionDraft(
        section_id="methods_steps",
        title="方法步骤",
        content="视频未提供这一类信息。",
        included_atom_ids=["A"],
    )

    normalized = _normalize_section(draft, outline_section, {"A": atom})

    assert normalized.content == "当你不再执着时，就迈出了第一步。"


def test_fallback_outline_uses_atom_kinds_instead_of_template_assignments() -> None:
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "B000-A001",
                    "kind": "claim",
                    "topic": "视频原文",
                    "statement": "这是源材料中的完整事实。",
                    "source_segment": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "source_excerpt": "这是源材料中的完整事实。",
                    "relations": [],
                    "sequence": None,
                    "is_promotional": False,
                    "asr_uncertainty": [],
                }
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["B000-A001"],
                    "reason": "lossless source-preserving local fallback",
                }
            ],
        },
        strict=True,
    )
    sections = [
        TeachingOutlineSection(
            section_id=section_id,
            title=section_id,
            order=order,
            atom_ids=["B000-A001"],
            required_evidence_types=["claim"],
            missing_evidence_policy="视频未提供这一类信息。",
            timestamps=[0.0, 1.0],
        )
        for order, section_id in enumerate(
            (
                "overview",
                "core_points",
                "methods_steps",
                "examples_results",
                "boundaries_risks",
                "actions",
                "timeline",
                "evidence_limits",
            ),
            1,
        )
    ]

    normalized = _normalize_outline(
        TeachingOutline(
            sections=sections,
            included_atom_ids=["B000-A001"],
        ),
        extraction,
    )

    by_id = {section.section_id: section.atom_ids for section in normalized.sections}
    assert by_id["overview"] == ["B000-A001"]
    assert by_id["core_points"] == ["B000-A001"]
    assert by_id["timeline"] == ["B000-A001"]
    assert by_id["methods_steps"] == []
    assert by_id["examples_results"] == []
    assert (
        next(
            section.timestamps
            for section in normalized.sections
            if section.section_id == "methods_steps"
        )
        == []
    )


def test_teaching_synthesis_omits_empty_theme_sections_and_uses_atom_timestamps() -> None:
    note_input = NoteInput(
        aweme_id="short",
        transcript="这是源材料中的完整事实。",
        segments=[InputSegment(index=0, start=0.0, end=1.0, text="这是源材料中的完整事实。")],
        metadata={},
        description="短视频",
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=1_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    extraction = _canonical_source_extraction(build_segment_batches(note_input, 1_024)[0])
    atom_id = extraction.atoms[0].id
    outline = TeachingOutline(
        sections=[
            TeachingOutlineSection(
                section_id=section_id,
                title=section_id,
                order=order,
                atom_ids=[atom_id] if section_id in {"overview", "core_points", "timeline"} else [],
                required_evidence_types=["claim"],
                missing_evidence_policy="视频未提供这一类信息。",
            )
            for order, section_id in enumerate(
                (
                    "overview",
                    "core_points",
                    "methods_steps",
                    "examples_results",
                    "boundaries_risks",
                    "actions",
                    "timeline",
                    "evidence_limits",
                ),
                1,
            )
        ],
        included_atom_ids=[atom_id],
    )
    drafts = tuple(
        SectionDraft(
            section_id=section.section_id,
            title=section.title,
            content=(
                extraction.atoms[0].statement if section.atom_ids else "视频未提供这一类信息。"
            ),
            included_atom_ids=list(section.atom_ids),
            generated_interpretations=["视频未提供这一类信息。"],
        )
        for section in outline.sections
    )

    synthesis = _teaching_synthesis(note_input, extraction, outline, drafts)

    assert synthesis.themes[0].timestamps == [0.0]
    assert synthesis.methods_and_steps == []
    assert synthesis.examples_and_counterexamples == []
    assert synthesis.boundaries_and_risks == []
    assert synthesis.implications == []


def test_fallback_depth_review_discards_provider_structural_false_positives() -> None:
    note_input = NoteInput(
        aweme_id="short",
        transcript="原文事实。",
        segments=[InputSegment(index=0, start=0.0, end=1.0, text="原文事实。")],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=1_000,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    extraction = _canonical_source_extraction(build_segment_batches(note_input, 1_024)[0])
    atom_id = extraction.atoms[0].id
    provider_review = DepthReview(
        abstracted_atom_ids=[atom_id],
        example_without_result_atom_ids=[],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=[],
        lost_warning_or_promotion_atom_ids=[],
        lost_uncertainty_markers=[],
        unsupported_claims=["仍需失败的无来源主张"],
        repair_section_ids=["methods_steps"],
        input_sufficient=True,
    )

    normalized = _normalize_provider_depth_review(extraction, provider_review)

    assert normalized.abstracted_atom_ids == []
    assert normalized.repair_section_ids == []
    assert normalized.unsupported_claims == ["仍需失败的无来源主张"]


def test_identical_disposition_correction_is_ignored_but_real_change_remains() -> None:
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [
                {
                    "id": "A",
                    "kind": "claim",
                    "topic": "主题",
                    "statement": "事实",
                    "source_segment": 0,
                    "source_excerpt": "事实",
                }
            ],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["A"],
                    "reason": "internal fallback marker",
                }
            ],
        },
        strict=True,
    )
    no_op = CoverageReview(
        reviewed_segment_indexes=[0],
        source_omission_candidates=[],
        unsupported_draft_claims=[],
        disposition_corrections=[
            {"segment_index": 0, "disposition": "included", "atom_ids": ["A"]}
        ],
    )
    real_change = CoverageReview(
        reviewed_segment_indexes=[0],
        source_omission_candidates=[],
        unsupported_draft_claims=[],
        disposition_corrections=[
            {"segment_index": 0, "disposition": "promotion", "atom_ids": ["A"]}
        ],
    )

    assert _remove_noop_disposition_corrections(no_op, extraction).disposition_corrections == []
    empty_mapping = CoverageReview(
        reviewed_segment_indexes=[0],
        source_omission_candidates=[],
        unsupported_draft_claims=[],
        disposition_corrections=[{"segment_index": 0, "disposition": "included", "atom_ids": []}],
    )
    assert (
        _remove_noop_disposition_corrections(empty_mapping, extraction).disposition_corrections
        == []
    )
    assert (
        len(_remove_noop_disposition_corrections(real_change, extraction).disposition_corrections)
        == 1
    )


def test_depth_repair_slice_does_not_expand_empty_flagged_section_to_all_atoms() -> None:
    atoms = [
        EvidenceAtom(
            id="A",
            kind="claim",
            topic="one",
            statement="A",
            source_segment=0,
            source_excerpt="A",
        ),
        EvidenceAtom(
            id="B",
            kind="claim",
            topic="two",
            statement="B",
            source_segment=1,
            source_excerpt="B",
        ),
    ]
    extraction = ExtractionResult(
        atoms=atoms,
        dispositions=[
            {"segment_index": 0, "disposition": "included", "atom_ids": ["A"]},
            {"segment_index": 1, "disposition": "included", "atom_ids": ["B"]},
        ],
    )
    empty_section = SectionDraft(
        section_id="methods_steps",
        title="Methods",
        content="视频未提供这一类信息。",
        included_atom_ids=[],
    )
    review = DepthReview(
        abstracted_atom_ids=[],
        example_without_result_atom_ids=[],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=[],
        lost_warning_or_promotion_atom_ids=[],
        lost_uncertainty_markers=[],
        unsupported_claims=[],
        repair_section_ids=["methods_steps"],
        input_sufficient=True,
    )

    repaired, segments = _depth_repair_slice(
        extraction,
        (empty_section,),
        review,
    )

    assert repaired.atoms == []
    assert segments == set()


@pytest.mark.asyncio
async def test_repair_slice_scopes_unsupported_draft_claim_to_its_evidence() -> None:
    result = await SemanticNotePipeline(ScriptedProvider()).generate(
        _note_input(), asyncio.Event()
    )
    promotion = result.synthesis.promotions[0]
    review = result.review.model_copy(
        update={
            "unsupported_draft_claims": [
                f"{promotion.heading}: {promotion.content}",
            ]
        }
    )

    repaired, segments = _repair_slice(
        result.extraction,
        result.synthesis,
        review,
        result.coverage,
    )

    assert [atom.id for atom in repaired.atoms] == promotion.atom_ids
    assert segments == {6}


@pytest.mark.asyncio
async def test_repair_slice_falls_back_when_unsupported_claim_cannot_be_mapped() -> None:
    result = await SemanticNotePipeline(ScriptedProvider()).generate(
        _note_input(), asyncio.Event()
    )
    review = result.review.model_copy(
        update={"unsupported_draft_claims": ["无法映射到草稿的无来源主张"]}
    )
    coverage = result.coverage.model_copy(
        update={"missing_atom_ids": [result.extraction.atoms[0].id]}
    )

    repaired, segments = _repair_slice(
        result.extraction,
        result.synthesis,
        review,
        coverage,
    )

    assert [atom.id for atom in repaired.atoms] == [
        atom.id for atom in result.extraction.atoms
    ]
    assert segments == {segment.index for segment in _note_input().segments}


def test_depth_review_payload_keeps_cross_section_relations_in_context() -> None:
    note_input = _note_input()
    atoms = [
        EvidenceAtom(
            id="example",
            kind="example",
            topic="case",
            statement="例子",
            source_segment=0,
            source_excerpt="例子",
            relations=["result"],
        ),
        EvidenceAtom(
            id="result",
            kind="claim",
            topic="case",
            statement="结果",
            source_segment=1,
            source_excerpt="结果",
        ),
    ]
    extraction = ExtractionResult(
        atoms=atoms,
        dispositions=[
            {"segment_index": 0, "disposition": "included", "atom_ids": ["example"]},
            {"segment_index": 1, "disposition": "included", "atom_ids": ["result"]},
        ],
    )
    outline = TeachingOutline(
        sections=[
            TeachingOutlineSection(
                section_id="examples_results",
                title="Examples",
                order=1,
                atom_ids=["example"],
                required_evidence_types=["example", "result"],
                missing_evidence_policy="视频未提供这一类信息。",
            ),
            TeachingOutlineSection(
                section_id="core_points",
                title="Core",
                order=2,
                atom_ids=["result"],
                required_evidence_types=["claim"],
                missing_evidence_policy="视频未提供这一类信息。",
            ),
        ],
        included_atom_ids=["example", "result"],
    )
    section = SectionDraft(
        section_id="examples_results",
        title="Examples",
        content="例子",
        included_atom_ids=["example"],
    )
    context = SectionDraft(
        section_id="core_points",
        title="Core",
        content="结果",
        included_atom_ids=["result"],
    )

    payload = _depth_review_payload(
        note_input,
        extraction,
        outline,
        (section,),
        all_sections=(section, context),
    )

    assert payload["section_context"]
    assert {item["id"] for item in payload["atom_relation_context"]} == {"example", "result"}
    assert "result" in payload["atom_relation_context"][0]["relations"]


def test_depth_review_unknown_repair_section_is_fail_closed() -> None:
    review = DepthReview.model_construct(
        abstracted_atom_ids=[],
        example_without_result_atom_ids=[],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=[],
        lost_warning_or_promotion_atom_ids=[],
        lost_uncertainty_markers=[],
        unsupported_claims=[],
        repair_section_ids=["not-a-section"],
        input_sufficient=True,
        unknown_atom_ids=[],
        unknown_section_ids=[],
        non_repairable_findings=[],
    )

    checked = _validate_depth_review_refs(review, set(), section_ids={"overview"})

    assert checked.repair_section_ids == []
    assert checked.unknown_section_ids == ["not-a-section"]
    assert checked.non_repairable_findings


@pytest.mark.asyncio
async def test_teaching_pipeline_runs_depth_review_and_gates_incomplete_input() -> None:
    class DepthProvider(ScriptedProvider):
        supports_teaching_pipeline = True

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            if kwargs["stage"] == "outline":
                payload = kwargs["payload"]
                atom_ids = [atom["id"] for atom in payload["atom_catalog"]]
                section_ids = [
                    "overview",
                    "core_points",
                    "methods_steps",
                    "examples_results",
                    "boundaries_risks",
                    "actions",
                    "timeline",
                    "evidence_limits",
                ]
                self.stages.append("outline")
                self.payloads.append(("outline", payload))
                parsed = {
                    "sections": [
                        {
                            "section_id": section_id,
                            "title": section_id,
                            "order": order,
                            "atom_ids": atom_ids if order == 1 else [],
                            "required_evidence_types": [],
                            "missing_evidence_policy": "视频未提供这一类信息。",
                            "timestamps": [],
                        }
                        for order, section_id in enumerate(section_ids, 1)
                    ],
                    "included_atom_ids": atom_ids,
                    "unassigned_atom_ids": [],
                    "input_sufficient": True,
                }
            elif kwargs["stage"] == "section":
                payload = kwargs["payload"]
                section = payload["section"]
                atom_ids = [atom["id"] for atom in payload["atoms"]]
                self.stages.append("section")
                self.payloads.append(("section", payload))
                parsed = {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "content": "；".join(atom["statement"] for atom in payload["atoms"])
                    or "视频未提供这一类信息。",
                    "claims": [],
                    "steps": [],
                    "examples": [],
                    "results": [],
                    "conditions": [],
                    "risks": [],
                    "actions": [],
                    "included_atom_ids": atom_ids,
                    "unprovided_items": [],
                    "generated_interpretations": [],
                }
            elif kwargs["stage"] == "depth_review":
                self.stages.append("depth_review")
                self.payloads.append(("depth_review", kwargs["payload"]))
                parsed = {
                    "abstracted_atom_ids": [],
                    "example_without_result_atom_ids": [],
                    "condition_without_scope_atom_ids": [],
                    "reordered_step_atom_ids": [],
                    "lost_literal_atom_ids": [],
                    "lost_warning_or_promotion_atom_ids": [],
                    "lost_uncertainty_markers": [],
                    "unsupported_claims": [],
                    "repair_section_ids": [],
                    "input_sufficient": False,
                }
            else:
                return await super().complete_json(**kwargs)
            return ProviderResponse(
                raw_output="depth",
                parsed_json=parsed,
                usage=ProviderUsage(
                    input_tokens=10,
                    output_tokens=5,
                    api_cost=0.0,
                    cost_status="zero_local",
                ),
                diagnostics={"scripted": True},
            )

    provider = DepthProvider()
    sparse = replace(_note_input(), warnings=["synthetic segment"])
    result = await SemanticNotePipeline(provider).generate(sparse, asyncio.Event())

    assert provider.stages.count("depth_review") >= 1
    assert result.depth_review is not None
    assert result.input_sufficient is False
    assert result.coverage.quality_gate == "failed"
    assert result.coverage.semantic_complete is False


@pytest.mark.asyncio
async def test_teaching_coverage_repair_authorizes_only_affected_sections() -> None:
    class CoverageRepairTeachingProvider(ScriptedProvider):
        supports_teaching_pipeline = True

        def __init__(self) -> None:
            super().__init__()
            self.review_passes = 0

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            stage = kwargs["stage"]
            payload = kwargs["payload"]
            if stage == "extract":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                atoms = []
                dispositions = []
                for segment in payload["segments"]:
                    atom_id = f"claim-{segment['index']}"
                    atoms.append(
                        {
                            "id": atom_id,
                            "kind": "claim",
                            "topic": "topic",
                            "statement": segment["text"],
                            "source_segment": segment["index"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "source_excerpt": segment["text"],
                            "relations": [],
                            "sequence": None,
                            "is_promotional": False,
                            "asr_uncertainty": [],
                        }
                    )
                    dispositions.append(
                        {
                            "segment_index": segment["index"],
                            "disposition": "included",
                            "atom_ids": [atom_id],
                            "reason": None,
                        }
                    )
                parsed = {"atoms": atoms, "dispositions": dispositions}
            elif stage == "outline":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                atom_ids = [item["id"] for item in payload["atom_catalog"]]
                parsed = {
                    "sections": [
                        {
                            "section_id": section_id,
                            "title": section_id,
                            "order": order,
                            "atom_ids": atom_ids if order == 1 else [],
                            "required_evidence_types": [],
                            "missing_evidence_policy": "视频未提供这一类信息。",
                            "timestamps": [],
                        }
                        for order, section_id in enumerate(
                            (
                                "overview",
                                "core_points",
                                "methods_steps",
                                "examples_results",
                                "boundaries_risks",
                                "actions",
                                "timeline",
                                "evidence_limits",
                            ),
                            1,
                        )
                    ],
                    "included_atom_ids": atom_ids,
                    "unassigned_atom_ids": [],
                    "input_sufficient": True,
                }
            elif stage == "section":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                atoms = payload["atoms"]
                parsed = {
                    "section_id": payload["section"]["section_id"],
                    "title": payload["section"]["title"],
                    "content": "；".join(atom["statement"] for atom in atoms)
                    or "视频未提供这一类信息。",
                    "claims": [],
                    "steps": [],
                    "examples": [],
                    "results": [],
                    "conditions": [],
                    "risks": [],
                    "actions": [],
                    "included_atom_ids": [atom["id"] for atom in atoms],
                    "unprovided_items": [],
                    "generated_interpretations": [],
                }
            elif stage == "depth_review":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                parsed = {
                    "abstracted_atom_ids": [],
                    "example_without_result_atom_ids": [],
                    "condition_without_scope_atom_ids": [],
                    "reordered_step_atom_ids": [],
                    "lost_literal_atom_ids": [],
                    "lost_warning_or_promotion_atom_ids": [],
                    "lost_uncertainty_markers": [],
                    "unsupported_claims": [],
                    "repair_section_ids": [],
                    "input_sufficient": True,
                }
            elif stage == "review":
                self.review_passes += 1
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                if payload.get("review_mode") == "draft_claims":
                    claim_ids = [claim["id"] for claim in payload["draft_claims"]]
                    parsed = {
                        "reviewed_claim_ids": claim_ids,
                        "supported_claim_ids": claim_ids,
                        "contradicted_claim_ids": [],
                    }
                    return ProviderResponse(
                        raw_output="coverage-review-claims",
                        parsed_json=parsed,
                        usage=ProviderUsage(
                            input_tokens=10,
                            output_tokens=5,
                            api_cost=0.0,
                            cost_status="zero_local",
                        ),
                        diagnostics={"scripted": True},
                    )
                parsed = {
                    "reviewed_segment_indexes": [
                        segment["index"] for segment in payload["source_segments"]
                    ],
                    "source_omission_candidates": (
                        [
                            {
                                "segment_index": 0,
                                "source_excerpt": payload["source_segments"][0]["text"],
                                "missing_statement": "coverage omission",
                                "suggested_kind": "claim",
                            }
                        ]
                        if self.review_passes == 1
                        else []
                    ),
                    "unsupported_draft_claims": [],
                    "disposition_corrections": [],
                }
            elif stage == "repair":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                section_drafts = payload["section_drafts"]
                parsed = {
                    "extraction_patch": {"atoms": [], "dispositions": []},
                    "synthesis_patch": payload["draft"],
                    "section_drafts": section_drafts[:1],
                }
            else:
                raise AssertionError(stage)
            return ProviderResponse(
                raw_output="coverage-repair",
                parsed_json=parsed,
                usage=ProviderUsage(
                    input_tokens=10,
                    output_tokens=5,
                    api_cost=0.0,
                    cost_status="zero_local",
                ),
                diagnostics={"scripted": True},
            )

    provider = CoverageRepairTeachingProvider()
    result = await SemanticNotePipeline(provider).generate(_note_input(), asyncio.Event())

    assert result.repair_count == 1
    assert provider.stages.count("repair") == 1
    repair_payload = next(payload for stage, payload in provider.payloads if stage == "repair")
    assert repair_payload["repair_section_ids"]
    assert result.depth_review is not None


@pytest.mark.asyncio
async def test_complete_prior_catalog_links_early_claim_to_late_condition() -> None:
    segment_count = 20
    segments = [
        InputSegment(
            index=index,
            start=float(index),
            end=float(index + 1),
            text=f"segment-{index} " + "知" * 3_000,
        )
        for index in range(segment_count)
    ]
    note_input = NoteInput(
        aweme_id="catalog",
        transcript="".join(segment.text for segment in segments),
        segments=segments,
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=None,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )

    class CatalogCaptured(RuntimeError):
        pass

    class CatalogProvider(ScriptedProvider):
        context_tokens = 65_536

        def __init__(self) -> None:
            super().__init__()
            self.late_atom: dict[str, Any] | None = None

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            stage = kwargs["stage"]
            payload = kwargs["payload"]
            if stage == "extract":
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                catalog = payload.get("prior_evidence_catalog", [])
                atoms = []
                dispositions = []
                for segment in payload["segments"]:
                    is_late = segment["index"] == segment_count - 1
                    early = next(
                        (item for item in catalog if item["source_segment"] == 0),
                        None,
                    )
                    atom_id = f"a-{segment['index']}"
                    atoms.append(
                        {
                            "id": atom_id,
                            "kind": "condition" if is_late else "claim",
                            "topic": "catalog",
                            "statement": segment["text"],
                            "source_segment": segment["index"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "source_excerpt": segment["text"],
                            "relations": [early["id"]] if is_late and early else [],
                            "sequence": None,
                            "is_promotional": False,
                            "asr_uncertainty": [],
                        }
                    )
                    dispositions.append(
                        {
                            "segment_index": segment["index"],
                            "disposition": "included",
                            "atom_ids": [atom_id],
                            "reason": None,
                        }
                    )
                if any(segment["index"] == segment_count - 1 for segment in payload["segments"]):
                    self.late_atom = next(
                        atom for atom in atoms if atom["source_segment"] == segment_count - 1
                    )
                    raise CatalogCaptured
                return ProviderResponse(
                    raw_output="catalog extraction",
                    parsed_json={"atoms": atoms, "dispositions": dispositions},
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            response = await super().complete_json(**kwargs)
            if stage == "review" and payload.get("review_mode") == "source_coverage":
                parsed = dict(response.parsed_json)
                parsed["source_omission_candidates"] = []
                return ProviderResponse(
                    raw_output=response.raw_output,
                    parsed_json=parsed,
                    usage=response.usage,
                    diagnostics=response.diagnostics,
                )
            return response

    provider = CatalogProvider()

    with pytest.raises(CatalogCaptured):
        await SemanticNotePipeline(provider).generate(
            note_input,
            asyncio.Event(),
        )

    late_payload = next(
        payload
        for stage, payload in provider.payloads
        if stage == "extract"
        and any(segment["index"] == segment_count - 1 for segment in payload["segments"])
    )
    early = late_payload["prior_evidence_catalog"][0]
    assert early["id"] not in {atom["id"] for atom in late_payload["prior_evidence_atoms"]}
    assert late_payload["prior_evidence_catalog"][0] == {
        "id": early["id"],
        "kind": "claim",
        "topic": "catalog",
        "source_segment": 0,
    }
    assert provider.late_atom is not None
    assert provider.late_atom["relations"] == [early["id"]]


@pytest.mark.asyncio
async def test_repair_adds_traceable_related_atom_and_reviews_updated_extraction() -> None:
    class OmittedConditionProvider(ScriptedProvider):
        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            stage = kwargs["stage"]
            payload = kwargs["payload"]
            if stage == "repair":
                extraction = payload["extraction"]
                target_id = extraction["atoms"][0]["id"]
                repaired_atom = {
                    "id": "repair-condition",
                    "kind": "condition",
                    "topic": "表达系统",
                    "statement": "只有每周复盘才适用",
                    "source_segment": 5,
                    "start": 50.0,
                    "end": 60.0,
                    "source_excerpt": "只有每周复盘才适用",
                    "relations": [target_id],
                    "sequence": None,
                    "is_promotional": False,
                    "asr_uncertainty": [],
                }
                disposition = next(
                    dict(item) for item in extraction["dispositions"] if item["segment_index"] == 5
                )
                disposition["atom_ids"] = [
                    *disposition["atom_ids"],
                    repaired_atom["id"],
                ]
                combined = {
                    "atoms": [*extraction["atoms"], repaired_atom],
                    "dispositions": [
                        disposition if item["segment_index"] == 5 else dict(item)
                        for item in extraction["dispositions"]
                    ],
                }
                parsed = {
                    "extraction_patch": {
                        "atoms": [repaired_atom],
                        "dispositions": [disposition],
                    },
                    "synthesis_patch": _synthesis(combined, omit_warning=False),
                }
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                return ProviderResponse(
                    raw_output="repair with evidence",
                    parsed_json=parsed,
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )

            response = await super().complete_json(**kwargs)
            if stage == "extract" and any(segment["index"] == 5 for segment in payload["segments"]):
                parsed = {
                    "atoms": [
                        atom for atom in response.parsed_json["atoms"] if atom["id"] != "condition"
                    ],
                    "dispositions": [
                        {
                            **item,
                            "atom_ids": [
                                atom_id for atom_id in item["atom_ids"] if atom_id != "condition"
                            ],
                        }
                        for item in response.parsed_json["dispositions"]
                    ],
                }
                return ProviderResponse(
                    raw_output=response.raw_output,
                    parsed_json=parsed,
                    usage=response.usage,
                    diagnostics=response.diagnostics,
                )
            if (
                stage == "review"
                and payload.get("review_pass") == 1
                and payload.get("review_mode") == "source_coverage"
            ):
                parsed = dict(response.parsed_json)
                if any(segment["index"] == 5 for segment in payload["source_segments"]):
                    parsed["source_omission_candidates"] = [
                        {
                            "segment_index": 5,
                            "source_excerpt": "只有每周复盘才适用",
                            "missing_statement": "遗漏使用前提",
                            "suggested_kind": "condition",
                        }
                    ]
                return ProviderResponse(
                    raw_output=response.raw_output,
                    parsed_json=parsed,
                    usage=response.usage,
                    diagnostics=response.diagnostics,
                )
            if (
                stage == "review"
                and payload.get("review_pass") == 2
                and payload.get("review_mode") == "source_coverage"
                and any(segment["index"] == 5 for segment in payload["source_segments"])
            ):
                assert any(
                    atom["id"] == "repair-condition" for atom in payload["extraction"]["atoms"]
                )
            return response

    provider = OmittedConditionProvider()

    result = await SemanticNotePipeline(provider).generate(_note_input(), asyncio.Event())

    repaired = next(atom for atom in result.extraction.atoms if atom.id == "repair-condition")
    assert repaired.source_segment == 5
    assert repaired.source_excerpt == "只有每周复盘才适用"
    assert repaired.relations
    disposition = next(item for item in result.extraction.dispositions if item.segment_index == 5)
    assert repaired.id in disposition.atom_ids
    assert repaired.id in result.synthesis.included_atom_ids
    assert result.coverage.quality_gate == "passed"
    assert provider.stages.count("repair") == 1


@pytest.mark.asyncio
async def test_review_accounts_for_claims_in_every_final_draft_field() -> None:
    expected_fields = {
        "title",
        "overview",
        "structure",
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "concepts",
        "quotes",
        "implications",
        "actions",
        "timeline",
        "promotions",
        "tags",
    }

    class UnsupportedEverywhereProvider(ScriptedProvider):
        context_tokens = 65_536

        def __init__(self) -> None:
            super().__init__()
            self.seen_fields: dict[int, set[str]] = {1: set(), 2: set()}

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            stage = kwargs["stage"]
            payload = kwargs["payload"]
            if stage == "review" and payload.get("review_mode") == "draft_claims":
                claims = payload["draft_claims"]
                self.seen_fields[payload["review_pass"]].update(claim["field"] for claim in claims)
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                return ProviderResponse(
                    raw_output="unsupported claims",
                    parsed_json={
                        "reviewed_claim_ids": [claim["id"] for claim in claims],
                        "supported_claim_ids": [],
                        "contradicted_claim_ids": [claim["id"] for claim in claims],
                    },
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            if stage == "repair":
                draft = dict(payload["draft"])
                draft["title"] = "UNSUPPORTED title"
                draft["overview"] = "UNSUPPORTED overview"
                draft["structure"] = ["UNSUPPORTED structure"]
                for field in (
                    "themes",
                    "methods_and_steps",
                    "examples_and_counterexamples",
                    "boundaries_and_risks",
                    "promotions",
                ):
                    for section in draft[field]:
                        section["content"] = f"UNSUPPORTED {field}"
                for field in (
                    "concepts",
                    "quotes",
                    "implications",
                    "actions",
                    "tags",
                ):
                    draft[field] = [f"UNSUPPORTED {field}"]
                for entry in draft["timeline"]:
                    entry["content"] = "UNSUPPORTED timeline"
                self.stages.append(stage)
                self.payloads.append((stage, payload))
                return ProviderResponse(
                    raw_output="unchanged unsupported draft",
                    parsed_json={
                        "extraction_patch": {"atoms": [], "dispositions": []},
                        "synthesis_patch": draft,
                    },
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )

            response = await super().complete_json(**kwargs)
            if stage == "synthesize":
                draft = _synthesis(payload["extraction"], omit_warning=False)
                draft["title"] = "UNSUPPORTED title"
                draft["overview"] = "UNSUPPORTED overview"
                draft["structure"] = ["UNSUPPORTED structure"]
                for field in (
                    "themes",
                    "methods_and_steps",
                    "examples_and_counterexamples",
                    "boundaries_and_risks",
                    "promotions",
                ):
                    if draft[field]:
                        draft[field][0]["content"] = f"UNSUPPORTED {field}"
                for field in (
                    "concepts",
                    "quotes",
                    "implications",
                    "actions",
                    "tags",
                ):
                    draft[field] = [f"UNSUPPORTED {field}"]
                draft["timeline"][0]["content"] = "UNSUPPORTED timeline"
                return ProviderResponse(
                    raw_output=response.raw_output,
                    parsed_json=draft,
                    usage=response.usage,
                    diagnostics=response.diagnostics,
                )
            if stage == "review":
                parsed = dict(response.parsed_json)
                parsed["source_omission_candidates"] = []
                return ProviderResponse(
                    raw_output=response.raw_output,
                    parsed_json=parsed,
                    usage=response.usage,
                    diagnostics=response.diagnostics,
                )
            return response

    provider = UnsupportedEverywhereProvider()

    result = await SemanticNotePipeline(provider).generate(_note_input(), asyncio.Event())

    assert provider.seen_fields[1] == expected_fields
    assert provider.seen_fields[2] == expected_fields
    assert result.coverage.quality_gate == "failed"
    assert result.coverage.unsupported_draft_claims
    assert result.repair_count == 1


@pytest.mark.asyncio
async def test_pipeline_returns_failed_candidate_without_a_second_repair() -> None:
    class NeverFixedProvider(ScriptedProvider):
        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            response = await super().complete_json(**kwargs)
            if kwargs["stage"] == "repair" and self.extraction is not None:
                parsed = dict(response.parsed_json)
                parsed["synthesis_patch"] = _synthesis(
                    self.extraction,
                    omit_warning=True,
                )
                return ProviderResponse(
                    raw_output=response.raw_output,
                    parsed_json=parsed,
                    usage=response.usage,
                    diagnostics=response.diagnostics,
                )
            if (
                kwargs["stage"] == "review"
                and kwargs["payload"].get("review_pass") == 2
                and kwargs["payload"].get("review_mode") == "source_coverage"
                and any(segment["index"] == 5 for segment in kwargs["payload"]["source_segments"])
            ):
                parsed = dict(response.parsed_json)
                parsed["source_omission_candidates"] = [
                    {
                        "segment_index": 5,
                        "source_excerpt": "注意不要删除失败记录，否则无法回滚。",
                        "missing_statement": "仍遗漏回滚警告",
                        "suggested_kind": "warning",
                    }
                ]
                return ProviderResponse(
                    raw_output=response.raw_output,
                    parsed_json=parsed,
                    usage=response.usage,
                    diagnostics=response.diagnostics,
                )
            return response

    provider = NeverFixedProvider()

    result = await SemanticNotePipeline(provider).generate(_note_input(), asyncio.Event())

    assert result.coverage.quality_gate == "failed"
    assert result.repair_count == 1
    assert provider.stages.count("repair") == 1
    assert {
        payload["review_pass"] for stage, payload in provider.payloads if stage == "review"
    } == {1, 2}


@pytest.mark.asyncio
async def test_pipeline_revalidates_provider_json_with_strict_schema() -> None:
    class InvalidProvider(ScriptedProvider):
        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            response = await super().complete_json(**kwargs)
            parsed = dict(response.parsed_json)
            parsed["unexpected"] = True
            return ProviderResponse(
                raw_output=response.raw_output,
                parsed_json=parsed,
                usage=response.usage,
                diagnostics=response.diagnostics,
            )

    with pytest.raises(ValueError):
        await SemanticNotePipeline(InvalidProvider()).generate(_note_input(), asyncio.Event())


@pytest.mark.asyncio
async def test_pipeline_audits_provider_failure_before_reraising() -> None:
    class FailingProvider(ScriptedProvider):
        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            raise TimeoutError("provider timed out")

    class CollectingAudit:
        def __init__(self) -> None:
            self.records: list[Any] = []

        def record(self, record: Any) -> None:
            self.records.append(record)

    audit = CollectingAudit()

    with pytest.raises(TimeoutError, match="provider timed out"):
        await SemanticNotePipeline(FailingProvider()).generate(
            _note_input(), asyncio.Event(), audit
        )

    assert len(audit.records) == 1
    assert audit.records[0].stage == "extract"
    assert audit.records[0].payload["segments"]
    assert audit.records[0].diagnostics["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_final_review_cancellation_raises_and_preserves_audit() -> None:
    class CancelOnFinalReviewProvider(ScriptedProvider):
        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            response = await super().complete_json(**kwargs)
            payload = kwargs["payload"]
            if (
                kwargs["stage"] == "review"
                and payload.get("review_pass") == 2
                and any(segment["index"] == 6 for segment in payload["source_segments"])
            ):
                kwargs["cancel_event"].set()
            return response

    class CollectingAudit:
        def __init__(self) -> None:
            self.records: list[Any] = []

        def record(self, record: Any) -> None:
            self.records.append(record)

    audit = CollectingAudit()

    with pytest.raises(JobCancelledError, match="cancelled_by_user"):
        await SemanticNotePipeline(CancelOnFinalReviewProvider()).generate(
            _note_input(), asyncio.Event(), audit
        )

    assert audit.records[-1].stage == "review"
    assert audit.records[-1].payload["review_pass"] == 2
    assert any(segment["index"] == 6 for segment in audit.records[-1].payload["source_segments"])


def test_merge_renumbers_batch_local_segment_indexes_without_ambiguity() -> None:
    global_segments = tuple(_note_input().segments[1:3])
    batch = SegmentBatch(
        batch_index=1,
        start_segment=1,
        end_segment=2,
        segments=global_segments,
        input_chars=sum(len(segment.text) for segment in global_segments),
        estimated_tokens=20,
    )
    local_segments = [
        {
            "index": local_index,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for local_index, segment in enumerate(global_segments)
    ]
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [atom for segment in local_segments for atom in _atom_for_segment(segment)],
            "dispositions": [
                {
                    "segment_index": segment["index"],
                    "disposition": "included",
                    "atom_ids": [atom["id"] for atom in _atom_for_segment(segment)],
                    "reason": None,
                }
                for segment in local_segments
            ],
        },
        strict=True,
    )

    merged = _merge_extractions([(batch, extraction)])

    assert [item.segment_index for item in merged.dispositions] == [1, 2]
    assert sorted({atom.source_segment for atom in merged.atoms}) == [1, 2]


def test_merge_rewrites_a_unique_cross_batch_relation() -> None:
    segments = _note_input().segments[:2]
    first_batch = SegmentBatch(0, 0, 0, (segments[0],), 10, 10)
    second_batch = SegmentBatch(1, 1, 1, (segments[1],), 10, 10)
    first = ExtractionResult.model_validate(
        {
            "atoms": [_atom_for_segment(_segment_mapping(segments[0]))[0]],
            "dispositions": [
                {
                    "segment_index": 0,
                    "disposition": "included",
                    "atom_ids": ["claim"],
                    "reason": None,
                }
            ],
        },
        strict=True,
    )
    related = _atom_for_segment(_segment_mapping(segments[1]))[0]
    related.update(
        {
            "id": "related",
            "kind": "condition",
            "sequence": None,
            "relations": ["claim"],
        }
    )
    second = ExtractionResult.model_validate(
        {
            "atoms": [related],
            "dispositions": [
                {
                    "segment_index": 1,
                    "disposition": "included",
                    "atom_ids": ["related"],
                    "reason": None,
                }
            ],
        },
        strict=True,
    )

    merged = _merge_extractions([(first_batch, first), (second_batch, second)])

    assert merged.atoms[1].relations == [merged.atoms[0].id]


def test_merge_coalesces_fragment_dispositions_for_one_source_segment() -> None:
    original = _note_input().segments[0]
    first_batch = SegmentBatch(
        0,
        0,
        0,
        (InputSegment(0, original.start, original.end, "前半段"),),
        3,
        2,
    )
    second_batch = SegmentBatch(
        1,
        0,
        0,
        (InputSegment(0, original.start, original.end, "后半段"),),
        3,
        2,
    )

    def extraction(atom_id: str, excerpt: str) -> ExtractionResult:
        atom = _atom_for_segment({"index": 0, "start": 0.0, "end": 10.0, "text": excerpt})[0]
        atom["id"] = atom_id
        return ExtractionResult.model_validate(
            {
                "atoms": [atom],
                "dispositions": [
                    {
                        "segment_index": 0,
                        "disposition": "included",
                        "atom_ids": [atom_id],
                        "reason": None,
                    }
                ],
            },
            strict=True,
        )

    merged = _merge_extractions(
        [
            (first_batch, extraction("first", "前半段")),
            (second_batch, extraction("second", "后半段")),
        ]
    )

    assert len(merged.dispositions) == 1
    assert merged.dispositions[0].segment_index == 0
    assert merged.dispositions[0].atom_ids == [atom.id for atom in merged.atoms]


def test_merge_reviews_preserves_duplicate_indexes_from_one_response() -> None:
    review = CoverageReview.model_validate(
        {
            "reviewed_segment_indexes": [0, 0, 1],
            "source_omission_candidates": [],
            "unsupported_draft_claims": [],
            "disposition_corrections": [],
        },
        strict=True,
    )

    merged = _merge_reviews([review])

    assert merged.reviewed_segment_indexes == [0, 0, 1]


def test_merge_extractions_preserves_duplicate_dispositions_from_one_response() -> None:
    segment = _note_input().segments[0]
    batch = SegmentBatch(
        batch_index=0,
        start_segment=segment.index,
        end_segment=segment.index,
        segments=(segment,),
        input_chars=len(segment.text),
        estimated_tokens=10,
    )
    atom = _atom_for_segment(_segment_mapping(segment))[0]
    extraction = ExtractionResult.model_validate(
        {
            "atoms": [atom],
            "dispositions": [
                {
                    "segment_index": segment.index,
                    "disposition": "included",
                    "atom_ids": [atom["id"]],
                    "reason": None,
                },
                {
                    "segment_index": segment.index,
                    "disposition": "included",
                    "atom_ids": [atom["id"]],
                    "reason": None,
                },
            ],
        },
        strict=True,
    )

    merged = _merge_extractions([(batch, extraction)])

    assert [item.segment_index for item in merged.dispositions] == [0, 0]


def test_merge_extractions_preserves_duplicate_dispositions_from_later_batch() -> None:
    segment = _note_input().segments[0]
    first_batch = SegmentBatch(
        batch_index=0,
        start_segment=segment.index,
        end_segment=segment.index,
        segments=(segment,),
        input_chars=len(segment.text),
        estimated_tokens=10,
    )
    second_batch = SegmentBatch(
        batch_index=1,
        start_segment=segment.index,
        end_segment=segment.index,
        segments=(segment,),
        input_chars=len(segment.text),
        estimated_tokens=10,
    )

    def extraction(atom_id: str, duplicate: bool = False) -> ExtractionResult:
        atom = _atom_for_segment(_segment_mapping(segment))[0]
        atom["id"] = atom_id
        disposition = {
            "segment_index": segment.index,
            "disposition": "included",
            "atom_ids": [atom_id],
            "reason": None,
        }
        dispositions = [disposition, dict(disposition)] if duplicate else [disposition]
        return ExtractionResult.model_validate(
            {"atoms": [atom], "dispositions": dispositions},
            strict=True,
        )

    merged = _merge_extractions(
        [
            (first_batch, extraction("first")),
            (second_batch, extraction("second", duplicate=True)),
        ]
    )

    assert [item.segment_index for item in merged.dispositions] == [0, 0]
    assert merged.dispositions[0].atom_ids == ["B000-A001", "B001-A001"]
    assert merged.dispositions[1].atom_ids == ["B001-A001"]


@pytest.mark.asyncio
async def test_irreducible_repair_context_returns_failed_candidate_without_overflow_call() -> None:
    class TinyContextProvider(ScriptedProvider):
        context_tokens = 7_000

    provider = TinyContextProvider()

    result = await SemanticNotePipeline(provider).generate(_note_input(), asyncio.Event())

    assert result.context_status == "failed"
    assert result.context_failures[0]["stage"] == "repair"
    assert result.coverage.quality_gate == "failed"
    assert result.coverage.repairable is False
    assert result.repair_count == 0
    assert provider.stages.count("repair") == 0
    assert all(
        payload["context_budget"]["within_budget"] is True for _, payload in provider.payloads
    )


@pytest.mark.asyncio
async def test_irreducible_synthesis_budget_returns_failed_candidate() -> None:
    class OversizedSynthesisProvider(ScriptedProvider):
        context_tokens = 12_000

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            if kwargs["stage"] == "extract":
                self.stages.append("extract")
                self.payloads.append(("extract", kwargs["payload"]))
                segment = kwargs["payload"]["segments"][0]
                source_excerpt = "证据" * 5_000
                return ProviderResponse(
                    raw_output="oversized extraction",
                    parsed_json={
                        "atoms": [
                            {
                                "id": "oversized",
                                "kind": "claim",
                                "topic": "预算",
                                "statement": source_excerpt,
                                "source_segment": segment["index"],
                                "start": segment["start"],
                                "end": segment["end"],
                                "source_excerpt": source_excerpt,
                                "relations": [],
                                "sequence": None,
                                "is_promotional": False,
                                "asr_uncertainty": [],
                            }
                        ],
                        "dispositions": [
                            {
                                "segment_index": segment["index"],
                                "disposition": "included",
                                "atom_ids": ["oversized"],
                                "reason": None,
                            }
                        ],
                    },
                    usage=ProviderUsage(
                        input_tokens=10,
                        output_tokens=5,
                        api_cost=0.0,
                        cost_status="zero_local",
                    ),
                    diagnostics={"scripted": True},
                )
            raise AssertionError(f"unexpected provider call: {kwargs['stage']}")

    note_input = NoteInput(
        aweme_id="synthesis-budget",
        transcript="seed",
        segments=[InputSegment(index=0, start=0.0, end=1.0, text="seed")],
        metadata={},
        description=None,
        author=None,
        author_unique_id=None,
        published_at=None,
        duration_ms=None,
        source_url=None,
        source_files={},
        asset_records={},
        warnings=[],
    )
    provider = OversizedSynthesisProvider()

    result = await SemanticNotePipeline(provider).generate(
        note_input,
        asyncio.Event(),
    )

    assert result.context_status == "failed"
    assert result.context_failures[0]["stage"] == "synthesize"
    assert result.coverage.quality_gate == "failed"
    assert result.repair_count == 0
    assert provider.stages == ["extract"]


def _segment_mapping(segment: InputSegment) -> dict[str, Any]:
    return {
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "schema"),
    [
        ("extract", ExtractionResult),
        ("synthesize", SynthesisResult),
        ("review", CoverageReview),
        ("repair", RepairResult),
    ],
)
async def test_call_withholds_full_request_overflow_for_every_stage(
    stage: str,
    schema: type[Any],
) -> None:
    class TinyProvider(ScriptedProvider):
        context_tokens = 100

    class CollectingAudit:
        def __init__(self) -> None:
            self.records: list[Any] = []

        def record(self, record: Any) -> None:
            self.records.append(record)

    provider = TinyProvider()
    audit = CollectingAudit()
    records: list[Any] = []

    with pytest.raises(ProviderContextBudgetError):
        await SemanticNotePipeline(provider)._call(
            stage=stage,
            payload={
                "body": "x" * 2_000,
                "context_budget": {"within_budget": True},
            },
            schema=schema,
            cancel_event=asyncio.Event(),
            audit_sink=audit,
            records=records,
        )

    assert provider.stages == []
    assert len(audit.records) == 1
    failure = audit.records[0].payload["context_failure"]
    assert failure["stage"] == stage
    assert failure["method"] == "provider-request-v3-charclass"
    assert failure["within_budget"] is False


@pytest.mark.parametrize(
    ("stage", "schema", "minimum_reserve"),
    [
        ("extract", ExtractionResult, 1_024),
        ("synthesize", SynthesisResult, 1_024),
        ("review", CoverageReview, 768),
        ("repair", RepairResult, 2_048),
    ],
)
def test_request_budget_uses_stage_specific_payload_scaled_output_reserve(
    stage: str,
    schema: type[Any],
    minimum_reserve: int,
) -> None:
    small = _provider_request_budget(
        stage=stage,
        system_prompt="system",
        payload={"body": "x" * 10},
        json_schema=schema.model_json_schema(),
        context_tokens=100_000,
    )
    large = _provider_request_budget(
        stage=stage,
        system_prompt="system",
        payload={"body": "x" * 10_000},
        json_schema=schema.model_json_schema(),
        context_tokens=100_000,
    )

    assert small["reserved_output_tokens"] >= minimum_reserve
    assert large["reserved_output_tokens"] > small["reserved_output_tokens"]
    assert large["estimated_input_tokens"] > small["estimated_input_tokens"]
    assert large["estimated_total_tokens"] == (
        large["estimated_input_tokens"] + large["reserved_output_tokens"]
    )
    assert large["context_tokens"] == 100_000


def test_request_budget_does_not_count_chinese_utf8_bytes_as_tokens() -> None:
    payload = {"source": "中文证据" * 1_000}
    serialized_bytes = len(
        __import__("json").dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )

    budget = _provider_request_budget(
        stage="review",
        system_prompt="请只依据原始中文证据回答。" * 100,
        payload=payload,
        json_schema={},
        context_tokens=100_000,
    )

    assert budget["payload_tokens"] < serialized_bytes
    assert budget["payload_tokens"] >= len(payload["source"])


def test_request_budget_reuses_static_prompt_and_schema_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import douyin_library.notes.pipeline as pipeline_module

    calls: list[str] = []
    original_estimate = pipeline_module._estimate_prompt_tokens
    pipeline_module._cached_static_prompt_tokens.cache_clear()

    def observe(value: str) -> int:
        calls.append(value)
        return original_estimate(value)

    monkeypatch.setattr(pipeline_module, "_estimate_prompt_tokens", observe)
    system_prompt = "只依据本地原始材料。"
    schema = {"type": "object", "properties": {"result": {"type": "string"}}}

    for body in ("第一份材料", "第二份材料"):
        _provider_request_budget(
            stage="review",
            system_prompt=system_prompt,
            payload={"body": body},
            json_schema=schema,
            context_tokens=100_000,
        )

    serialized_schema = __import__("json").dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert calls.count(system_prompt) == 1
    assert calls.count(serialized_schema) == 1
