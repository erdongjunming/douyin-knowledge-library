from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_pipeline import ScriptedProvider, _atom_for_segment, _note_input, _synthesis

from douyin_library.notes.contracts import DraftClaimReview, ExtractionResult, SynthesisResult
from douyin_library.notes.pipeline import (
    SemanticNotePipeline,
    _claim_review_unsupported_ids,
    _draft_claims,
)
from douyin_library.notes.prompts import REVIEW_PROMPT_VERSION, REVIEW_SYSTEM_PROMPT
from douyin_library.notes.providers.base import ProviderResponse


class ClaimReviewProvider(ScriptedProvider):
    def __init__(self, mode: str, unknown_field: str | None = None) -> None:
        super().__init__()
        self.mode = mode
        self.unknown_field = unknown_field

    async def complete_json(self, **kwargs: Any) -> ProviderResponse:
        response = await super().complete_json(**kwargs)
        payload = kwargs["payload"]
        if kwargs["stage"] != "review":
            return response
        parsed = dict(response.parsed_json)
        if payload.get("review_mode") != "draft_claims":
            parsed["source_omission_candidates"] = []
        else:
            claim_ids = [claim["id"] for claim in payload["draft_claims"]]
            if self.mode == "omitted":
                del parsed["reviewed_claim_ids"]
            elif self.mode == "empty":
                parsed["reviewed_claim_ids"] = []
            elif self.mode == "partial":
                parsed["reviewed_claim_ids"] = claim_ids[:1]
            elif self.mode == "conflicting":
                parsed["contradicted_claim_ids"] = claim_ids[:1]
            if self.unknown_field is not None:
                parsed[self.unknown_field] = [*parsed[self.unknown_field], "unknown-claim"]
        return ProviderResponse(
            raw_output=response.raw_output,
            parsed_json=parsed,
            usage=response.usage,
            diagnostics=response.diagnostics,
        )


def _review_input() -> tuple[ExtractionResult, SynthesisResult]:
    atoms = [
        atom
        for segment in _note_input().segments
        for atom in _atom_for_segment(
            {
                "index": segment.index,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
        )
    ]
    extraction = ExtractionResult.model_validate(
        {
            "atoms": atoms,
            "dispositions": [
                {
                    "segment_index": segment.index,
                    "disposition": "included",
                    "atom_ids": [
                        atom["id"] for atom in atoms if atom["source_segment"] == segment.index
                    ],
                    "reason": None,
                }
                for segment in _note_input().segments
            ],
        }
    )
    synthesis = SynthesisResult.model_validate(
        _synthesis(extraction.model_dump(mode="json"), omit_warning=False)
    )
    return extraction, synthesis


def test_omitted_reviewed_claim_ids_default_to_empty() -> None:
    review = DraftClaimReview.model_validate(
        {"supported_claim_ids": ["C-title-0000"], "contradicted_claim_ids": []}
    )

    assert review.reviewed_claim_ids == []


def test_reviewed_structural_metadata_without_atom_evidence_is_not_a_fact_failure() -> None:
    claims = [
        {"id": "C-structure-0000", "field": "structure", "text": "核心要点", "atom_ids": []},
        {"id": "C-themes-0001", "field": "themes", "text": "事实", "atom_ids": ["A"]},
    ]

    unsupported = _claim_review_unsupported_ids(
        claims,
        reviewed_ids={"C-structure-0000", "C-themes-0001"},
        supported_ids={"C-themes-0001"},
        contradicted_ids=set(),
    )

    assert unsupported == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["omitted", "empty", "partial"])
async def test_incomplete_review_only_credits_explicitly_reviewed_claims(mode: str) -> None:
    extraction, synthesis = _review_input()
    claims = _draft_claims(synthesis)

    review = await SemanticNotePipeline(ClaimReviewProvider(mode))._review(
        _note_input(), extraction, synthesis, 1, asyncio.Event(), None, []
    )

    credited = 2 if mode == "partial" else 0
    assert set(review.unsupported_draft_claims) == {claim.text for claim in claims[credited:]}


@pytest.mark.asyncio
@pytest.mark.parametrize("contradict_first", [False, True])
async def test_incomplete_review_retries_only_missing_claims_and_keeps_reviewed_verdicts(
    contradict_first: bool,
) -> None:
    class MissingLastClaimProvider(ScriptedProvider):
        batches: list[list[str]]

        def __init__(self) -> None:
            super().__init__()
            self.batches = []

        async def complete_json(self, **kwargs: Any) -> ProviderResponse:
            response = await super().complete_json(**kwargs)
            parsed = dict(response.parsed_json)
            payload = kwargs["payload"]
            if kwargs["stage"] == "review":
                if payload.get("review_mode") == "draft_claims":
                    ids = [claim["id"] for claim in payload["draft_claims"]]
                    self.batches.append(ids)
                    if len(self.batches) == 1:
                        parsed["reviewed_claim_ids"] = ids[:-1]
                        parsed["supported_claim_ids"] = ids[1:-1] if contradict_first else ids[:-1]
                        parsed["contradicted_claim_ids"] = ids[:1] if contradict_first else []
                else:
                    parsed["source_omission_candidates"] = []
            return ProviderResponse(
                raw_output=response.raw_output, parsed_json=parsed,
                usage=response.usage, diagnostics=response.diagnostics,
            )

    extraction, synthesis = _review_input()
    provider = MissingLastClaimProvider()
    review = await SemanticNotePipeline(provider)._review(
        _note_input(), extraction, synthesis, 1, asyncio.Event(), None, []
    )

    assert len(provider.batches) == 2
    assert provider.batches[1] == provider.batches[0][-1:]
    assert review.unsupported_draft_claims == (
        [_draft_claims(synthesis)[0].text] if contradict_first else []
    )


@pytest.mark.asyncio
async def test_omitted_review_bookkeeping_fails_quality_gate_without_aborting() -> None:
    result = await SemanticNotePipeline(ClaimReviewProvider("omitted")).generate(
        _note_input(), asyncio.Event()
    )

    assert result.coverage.quality_gate == "failed"
    assert result.coverage.semantic_complete is False
    assert result.coverage.unsupported_draft_claims
    assert result.repair_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["reviewed_claim_ids", "supported_claim_ids", "contradicted_claim_ids"]
)
async def test_unknown_ids_in_every_claim_review_list_are_rejected(field: str) -> None:
    extraction, synthesis = _review_input()

    with pytest.raises(ValueError, match="unknown claim"):
        await SemanticNotePipeline(ClaimReviewProvider("complete", field))._review(
            _note_input(), extraction, synthesis, 1, asyncio.Event(), None, []
        )


@pytest.mark.asyncio
async def test_conflicting_claim_verdicts_remain_rejected() -> None:
    extraction, synthesis = _review_input()

    with pytest.raises(ValueError, match="conflicting verdicts"):
        await SemanticNotePipeline(ClaimReviewProvider("conflicting"))._review(
            _note_input(), extraction, synthesis, 1, asyncio.Event(), None, []
        )


def test_review_prompt_explains_both_modes_and_claim_id_accounting() -> None:
    assert REVIEW_PROMPT_VERSION == "raw-source-review-v2"
    for phrase in (
        "source_coverage",
        "draft_claims",
        "reviewed_claim_ids",
        "supported_claim_ids",
        "contradicted_claim_ids",
        "每个 claim",
    ):
        assert phrase in REVIEW_SYSTEM_PROMPT
