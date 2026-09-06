from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.contracts import (
    CoverageReview,
    ExtractionResult,
    ProviderUsage,
    RepairResult,
    SynthesisResult,
)
from douyin_library.notes.input import InputSegment, NoteInput
from douyin_library.notes.pipeline import (
    ProviderContextBudgetError,
    SegmentBatch,
    SemanticNotePipeline,
    _merge_extractions,
    _merge_reviews,
    _provider_request_budget,
    _synthesis_payload,
    build_segment_batches,
)
from douyin_library.notes.providers.base import ProviderResponse


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
                    if payload.get("review_pass") == 1
                    and includes_warning_segment
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


@pytest.mark.asyncio
async def test_long_input_is_layered_without_any_over_budget_provider_call() -> None:
    class LongContextProvider(ScriptedProvider):
        context_tokens = 131_072

    provider = LongContextProvider()
    note_input = _long_note_input()

    result = await SemanticNotePipeline(provider).generate(
        note_input, asyncio.Event()
    )

    assert result.coverage.quality_gate == "passed"
    assert result.context_status == "within_budget"
    assert provider.stages.count("extract") > 1
    assert provider.stages.count("synthesize") > 1
    assert provider.stages.count("review") > 1
    assert "".join(
        segment["text"]
        for stage, payload in provider.payloads
        if stage == "extract"
        for segment in payload["segments"]
    ) == note_input.transcript
    assert "".join(
        segment["text"]
        for stage, payload in provider.payloads
        if stage == "review"
        and payload.get("review_mode") == "source_coverage"
        for segment in payload["source_segments"]
    ) == note_input.transcript
    assert all(
        payload["context_budget"]["within_budget"] is True
        for _, payload in provider.payloads
    )


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
        assert "".join(
            segment["text"]
            for payload in review_payloads
            if payload["review_pass"] == review_pass
            and payload.get("review_mode") == "source_coverage"
            for segment in payload["source_segments"]
        ) == note_input.transcript
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
        payload["context_budget"]["within_budget"] is True
        for _, payload in provider.payloads
    )


@pytest.mark.asyncio
async def test_legacy_provider_exposes_fixed_teaching_outline_placeholders() -> None:
    result = await SemanticNotePipeline(ScriptedProvider()).generate(
        _note_input(), asyncio.Event()
    )

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
                    "content": "；".join(
                        atom["statement"] for atom in payload["atoms"]
                    )
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
    result = await SemanticNotePipeline(provider).generate(
        _note_input(), asyncio.Event()
    )

    assert provider.stages.count("outline") == 1
    assert provider.stages.count("section") == 8
    assert [draft.section_id for draft in result.section_drafts] == [
        section.section_id for section in result.outline.sections
    ]
    assert result.section_attempts == {
        section.section_id: 1 for section in result.outline.sections
    }
    assert set(result.synthesis.included_atom_ids) == {
        atom.id for atom in result.extraction.atoms
    }


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
                        (
                            item
                            for item in catalog
                            if item["source_segment"] == 0
                        ),
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
                if any(
                    segment["index"] == segment_count - 1
                    for segment in payload["segments"]
                ):
                    self.late_atom = next(
                        atom
                        for atom in atoms
                        if atom["source_segment"] == segment_count - 1
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
        and any(
            segment["index"] == segment_count - 1
            for segment in payload["segments"]
        )
    )
    early = late_payload["prior_evidence_catalog"][0]
    assert early["id"] not in {
        atom["id"] for atom in late_payload["prior_evidence_atoms"]
    }
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
                    dict(item)
                    for item in extraction["dispositions"]
                    if item["segment_index"] == 5
                )
                disposition["atom_ids"] = [
                    *disposition["atom_ids"],
                    repaired_atom["id"],
                ]
                combined = {
                    "atoms": [*extraction["atoms"], repaired_atom],
                    "dispositions": [
                        disposition
                        if item["segment_index"] == 5
                        else dict(item)
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
            if stage == "extract" and any(
                segment["index"] == 5 for segment in payload["segments"]
            ):
                parsed = {
                    "atoms": [
                        atom
                        for atom in response.parsed_json["atoms"]
                        if atom["id"] != "condition"
                    ],
                    "dispositions": [
                        {
                            **item,
                            "atom_ids": [
                                atom_id
                                for atom_id in item["atom_ids"]
                                if atom_id != "condition"
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
                if any(
                    segment["index"] == 5 for segment in payload["source_segments"]
                ):
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
                and any(
                    segment["index"] == 5
                    for segment in payload["source_segments"]
                )
            ):
                assert any(
                    atom["id"] == "repair-condition"
                    for atom in payload["extraction"]["atoms"]
                )
            return response

    provider = OmittedConditionProvider()

    result = await SemanticNotePipeline(provider).generate(
        _note_input(), asyncio.Event()
    )

    repaired = next(
        atom for atom in result.extraction.atoms if atom.id == "repair-condition"
    )
    assert repaired.source_segment == 5
    assert repaired.source_excerpt == "只有每周复盘才适用"
    assert repaired.relations
    disposition = next(
        item for item in result.extraction.dispositions if item.segment_index == 5
    )
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
                self.seen_fields[payload["review_pass"]].update(
                    claim["field"] for claim in claims
                )
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

    result = await SemanticNotePipeline(provider).generate(
        _note_input(), asyncio.Event()
    )

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
                and any(
                    segment["index"] == 5
                    for segment in kwargs["payload"]["source_segments"]
                )
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
        payload["review_pass"]
        for stage, payload in provider.payloads
        if stage == "review"
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
                and any(
                    segment["index"] == 6
                    for segment in payload["source_segments"]
                )
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
    assert any(
        segment["index"] == 6
        for segment in audit.records[-1].payload["source_segments"]
    )


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
            "atoms": [
                atom for segment in local_segments for atom in _atom_for_segment(segment)
            ],
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
        atom = _atom_for_segment(
            {"index": 0, "start": 0.0, "end": 10.0, "text": excerpt}
        )[0]
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
        context_tokens = 12_000

    provider = TinyContextProvider()

    result = await SemanticNotePipeline(provider).generate(
        _note_input(), asyncio.Event()
    )

    assert result.context_status == "failed"
    assert result.context_failures[0]["stage"] == "repair"
    assert result.coverage.quality_gate == "failed"
    assert result.coverage.repairable is False
    assert result.repair_count == 0
    assert provider.stages.count("repair") == 0
    assert all(
        payload["context_budget"]["within_budget"] is True
        for _, payload in provider.payloads
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
    assert failure["method"] == "provider-request-v2"
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
        payload={"body": "x" * 2_000},
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
