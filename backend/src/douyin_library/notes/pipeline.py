from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.contracts import (
    CoverageReport,
    CoverageReview,
    DepthReview,
    DraftClaim,
    DraftClaimReview,
    ExtractionResult,
    ProviderUsage,
    RepairResult,
    SectionDraft,
    SynthesisResult,
    TeachingOutline,
    TeachingOutlineSection,
    ThemeSection,
    TimelineEntry,
)
from douyin_library.notes.coverage import CoverageValidator
from douyin_library.notes.input import InputSegment, NoteInput
from douyin_library.notes.prompts import PROMPT_VERSIONS, SYSTEM_PROMPTS
from douyin_library.notes.providers.base import NoteProvider, NoteStage


@dataclass(frozen=True, slots=True)
class SegmentBatch:
    batch_index: int
    start_segment: int
    end_segment: int
    segments: tuple[InputSegment, ...]
    input_chars: int
    estimated_tokens: int

    def boundary(self) -> dict[str, int]:
        return {
            "batch_index": self.batch_index,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "segment_count": len(self.segments),
            "input_chars": self.input_chars,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: NoteStage
    prompt_version: str
    system_prompt: str
    payload: dict[str, Any]
    raw_output: str
    parsed_json: dict[str, Any]
    usage: ProviderUsage
    diagnostics: dict[str, Any]
    elapsed_ms: float
    batch_boundary: dict[str, int] | None


def _empty_teaching_outline() -> TeachingOutline:
    return TeachingOutline(
        sections=[
            TeachingOutlineSection(
                section_id=section_id,
                title=section_id.replace("_", " "),
                order=index,
                atom_ids=[],
                required_evidence_types=[],
                missing_evidence_policy="视频未提供这一类信息。",
            )
            for index, section_id in enumerate(_TEACHING_SECTION_IDS, 1)
        ],
        included_atom_ids=[],
        unassigned_atom_ids=[],
        input_sufficient=False,
    )


class AuditSink(Protocol):
    def record(self, record: StageRecord) -> Any: ...


class ProviderContextBudgetError(RuntimeError):
    """Raised when a request cannot fit the provider context window.

    ``failure`` is kept on the exception so orchestration can turn a preflight
    or last-mile budget miss into an audited failed candidate without making a
    second provider request.  Direct callers still receive the exception.
    """

    def __init__(
        self,
        stage: NoteStage,
        failure: Mapping[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.failure = copy.deepcopy(dict(failure or {}))
        super().__init__(f"{stage}:provider_context_budget_exceeded")


@dataclass(frozen=True, slots=True)
class SemanticGeneration:
    extraction: ExtractionResult
    synthesis: SynthesisResult
    review: CoverageReview
    coverage: CoverageReport
    repair_count: int
    provider: str
    model: str
    context_tokens: int
    usage: ProviderUsage
    stage_records: tuple[StageRecord, ...]
    batch_boundaries: tuple[dict[str, int], ...]
    prompt_versions: dict[str, str]
    is_local: bool
    outline: TeachingOutline = field(default_factory=_empty_teaching_outline)
    section_drafts: tuple[SectionDraft, ...] = ()
    depth_review: DepthReview | None = None
    section_attempts: dict[str, int] = field(default_factory=dict)
    input_sufficient: bool = False
    context_status: str = "within_budget"
    context_failures: tuple[dict[str, Any], ...] = ()


def build_segment_batches(
    note_input: NoteInput,
    context_tokens: int,
    *,
    input_fraction: float = 0.5,
) -> list[SegmentBatch]:
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if not 0 < input_fraction <= 1:
        raise ValueError("input_fraction must be between zero and one")
    token_limit = max(1, math.floor(context_tokens * input_fraction))
    char_limit = max(1, math.floor(token_limit * 1.5))
    batches: list[SegmentBatch] = []
    current: list[InputSegment] = []
    current_chars = 0
    current_tokens = 0
    expanded_segments = [
        InputSegment(
            index=segment.index,
            start=segment.start,
            end=segment.end,
            text=segment.text[offset : offset + char_limit],
        )
        for segment in note_input.segments
        for offset in range(0, max(1, len(segment.text)), char_limit)
    ]
    for segment in expanded_segments:
        segment_chars = len(segment.text)
        segment_tokens = max(1, math.ceil(segment_chars / 1.5))
        if current and current_tokens + segment_tokens > token_limit:
            batches.append(_batch(len(batches), current, current_chars, current_tokens))
            current = []
            current_chars = 0
            current_tokens = 0
        current.append(segment)
        current_chars += segment_chars
        current_tokens += segment_tokens
    if current:
        batches.append(_batch(len(batches), current, current_chars, current_tokens))
    return batches


class SemanticNotePipeline:
    def __init__(
        self,
        provider: NoteProvider,
        validator: CoverageValidator | None = None,
    ) -> None:
        self.provider = provider
        self.validator = validator or CoverageValidator()

    def _request_fits[T: BaseModel](
        self,
        stage: NoteStage,
        payload: Mapping[str, Any],
        schema: type[T],
    ) -> bool:
        return bool(
            _provider_request_budget(
                stage=stage,
                system_prompt=SYSTEM_PROMPTS[stage],
                payload=payload,
                json_schema=schema.model_json_schema(),
                context_tokens=self.provider.context_tokens,
            )["within_budget"]
        )

    async def generate(
        self,
        note_input: NoteInput,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None = None,
    ) -> SemanticGeneration:
        records: list[StageRecord] = []
        context_failures: list[dict[str, Any]] = []
        pending_batches = build_segment_batches(
            note_input,
            self.provider.context_tokens,
            input_fraction=0.35,
        )
        batches: list[SegmentBatch] = []
        extractions: list[tuple[SegmentBatch, ExtractionResult]] = []
        while pending_batches:
            batch = _reindex_batch(pending_batches.pop(0), len(batches))
            all_prior_atoms = (
                _merge_extractions(extractions).atoms if extractions else []
            )
            prior_atoms, _ = _bounded_prior_context(
                all_prior_atoms,
                batch.estimated_tokens,
                self.provider.context_tokens,
            )
            prior_catalog = [
                {
                    "id": atom.id,
                    "kind": atom.kind,
                    "topic": atom.topic,
                    "source_segment": atom.source_segment,
                }
                for atom in all_prior_atoms
            ]
            payload = _extraction_payload(
                note_input,
                batch,
                prior_catalog,
                prior_atoms,
            )
            while prior_atoms and not self._request_fits(
                "extract", payload, ExtractionResult
            ):
                _check_cancellation(cancel_event)
                prior_atoms = prior_atoms[1:]
                payload = _extraction_payload(
                    note_input,
                    batch,
                    prior_catalog,
                    prior_atoms,
                )
            if not self._request_fits("extract", payload, ExtractionResult):
                split_batches = _split_segment_batch(batch)
                if split_batches:
                    pending_batches = [*split_batches, *pending_batches]
                    continue
                failure = _budget_failure(
                    "extract",
                    _provider_request_budget(
                        stage="extract",
                        system_prompt=SYSTEM_PROMPTS["extract"],
                        payload=payload,
                        json_schema=ExtractionResult.model_json_schema(),
                        context_tokens=self.provider.context_tokens,
                    ),
                    batch_boundary=batch.boundary(),
                )
                await _record_context_failure(
                    failure,
                    audit_sink,
                    records,
                    batch_boundary=batch.boundary(),
                )
                _remember_context_failure(context_failures, failure)
                return self._failed_generation(
                    note_input=note_input,
                    extraction=_merge_extractions(extractions)
                    if extractions
                    else None,
                    synthesis=None,
                    review=None,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                )
            try:
                result = await self._call(
                    stage="extract",
                    payload=payload,
                    schema=ExtractionResult,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                    batch_boundary=batch.boundary(),
                )
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(exc, "extract", payload)
                _remember_context_failure(context_failures, failure)
                return self._failed_generation(
                    note_input=note_input,
                    extraction=_merge_extractions(extractions)
                    if extractions
                    else None,
                    synthesis=None,
                    review=None,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                )
            extractions.append((batch, result))
            batches.append(batch)
        extraction = _merge_extractions(extractions)
        outline = _empty_teaching_outline()
        section_drafts: tuple[SectionDraft, ...] = ()
        section_attempts: dict[str, int] = {}
        input_sufficient = True
        teaching_enabled = bool(
            getattr(self.provider, "supports_teaching_pipeline", False)
        )
        if teaching_enabled:
            try:
                outline = await self._build_outline(
                    note_input,
                    extraction,
                    cancel_event,
                    audit_sink,
                    records,
                )
                section_drafts, section_attempts = await self._synthesize_sections(
                    note_input,
                    extraction,
                    outline,
                    cancel_event,
                    audit_sink,
                    records,
                )
                input_sufficient = outline.input_sufficient and all(
                    not draft.unprovided_items for draft in section_drafts
                )
                synthesis = _teaching_synthesis(
                    note_input,
                    extraction,
                    outline,
                    section_drafts,
                )
            except AssertionError:
                # Keep the legacy path available for partially upgraded providers.
                teaching_enabled = False
        pending_slices = [] if teaching_enabled else [
            item
            for item, _ in _build_extraction_slices(
                extraction,
                self.provider.context_tokens,
            )
        ]
        extraction_slices: list[ExtractionResult] = []
        synthesis_parts: list[SynthesisResult] = []
        while pending_slices:
            extraction_slice = pending_slices.pop(0)
            trial_payload = _synthesis_payload(
                note_input,
                extraction_slice,
                len(extraction_slices),
                1,
            )
            if not self._request_fits(
                "synthesize",
                trial_payload,
                SynthesisResult,
            ):
                split_slices = _split_extraction_slice(extraction_slice)
                if split_slices:
                    pending_slices = [*split_slices, *pending_slices]
                    continue
                failure = _budget_failure(
                    "synthesize",
                    _provider_request_budget(
                        stage="synthesize",
                        system_prompt=SYSTEM_PROMPTS["synthesize"],
                        payload=trial_payload,
                        json_schema=SynthesisResult.model_json_schema(),
                        context_tokens=self.provider.context_tokens,
                    ),
                )
                await _record_context_failure(
                    failure,
                    audit_sink,
                    records,
                )
                _remember_context_failure(context_failures, failure)
                return self._failed_generation(
                    note_input=note_input,
                    extraction=extraction,
                    synthesis=_merge_syntheses(synthesis_parts)
                    if synthesis_parts
                    else None,
                    review=None,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                )
            extraction_slices.append(extraction_slice)
        for slice_index, extraction_slice in enumerate(extraction_slices):
            try:
                synthesis_parts.append(
                    await self._call(
                        stage="synthesize",
                        payload=_synthesis_payload(
                            note_input,
                            extraction_slice,
                            slice_index,
                            len(extraction_slices),
                        ),
                        schema=SynthesisResult,
                        cancel_event=cancel_event,
                        audit_sink=audit_sink,
                        records=records,
                    )
                )
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(exc, "synthesize", _synthesis_payload(
                    note_input,
                    extraction_slice,
                    slice_index,
                    len(extraction_slices),
                ))
                _remember_context_failure(context_failures, failure)
                return self._failed_generation(
                    note_input=note_input,
                    extraction=extraction,
                    synthesis=_merge_syntheses(synthesis_parts)
                    if synthesis_parts
                    else None,
                    review=None,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                )
        if not teaching_enabled:
            synthesis = _merge_syntheses(synthesis_parts)
        review = await self._review(
            note_input,
            extraction,
            synthesis,
            1,
            cancel_event,
            audit_sink,
            records,
            context_failures,
        )
        _check_cancellation(cancel_event)
        coverage = self._validate(note_input, extraction, synthesis, review)
        if context_failures:
            return self._failed_generation(
                note_input=note_input,
                extraction=extraction,
                synthesis=synthesis,
                review=review,
                context_failures=context_failures,
                records=records,
                batches=batches,
                coverage=coverage,
            )
        repair_count = 0
        if coverage.quality_gate == "failed" and coverage.repairable:
            _check_cancellation(cancel_event)
            repair_extraction, relevant_segments = _repair_slice(
                extraction, synthesis, review, coverage
            )
            repair_payload = _repair_payload(
                note_input,
                repair_extraction,
                synthesis,
                extraction,
                review,
                coverage,
                relevant_segments,
                compact_content=False,
            )
            repair_budget = _provider_request_budget(
                stage="repair",
                system_prompt=SYSTEM_PROMPTS["repair"],
                payload=repair_payload,
                json_schema=RepairResult.model_json_schema(),
                context_tokens=self.provider.context_tokens,
            )
            if not repair_budget["within_budget"]:
                compact_payload = _repair_payload(
                    note_input,
                    repair_extraction,
                    synthesis,
                    extraction,
                    review,
                    coverage,
                    relevant_segments,
                    compact_content=True,
                )
                compact_budget = _provider_request_budget(
                    stage="repair",
                    system_prompt=SYSTEM_PROMPTS["repair"],
                    payload=compact_payload,
                    json_schema=RepairResult.model_json_schema(),
                    context_tokens=self.provider.context_tokens,
                )
                if compact_budget["within_budget"]:
                    repair_payload = compact_payload
                    repair_budget = compact_budget
                else:
                    failure = _budget_failure("repair", compact_budget)
                    context_failures.append(failure)
                    await _record_context_failure(
                        failure,
                        audit_sink,
                        records,
                    )
                    coverage = coverage.model_copy(
                        update={"quality_gate": "failed", "repairable": False}
                    )
                    return self._failed_generation(
                        note_input=note_input,
                        extraction=extraction,
                        synthesis=synthesis,
                        review=review,
                        context_failures=context_failures,
                        records=records,
                        batches=batches,
                        coverage=coverage,
                    )
            repair_payload["context_budget"] = repair_budget
            try:
                repair_result = await self._call(
                    stage="repair",
                    payload=repair_payload,
                    schema=RepairResult,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                )
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(exc, "repair", repair_payload)
                _remember_context_failure(context_failures, failure)
                coverage = coverage.model_copy(
                    update={"quality_gate": "failed", "repairable": False}
                )
                return self._failed_generation(
                    note_input=note_input,
                    extraction=extraction,
                    synthesis=synthesis,
                    review=review,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                    coverage=coverage,
                )
            try:
                updated_extraction = _merge_repair_extraction_patch(
                    extraction,
                    repair_result.extraction_patch,
                    relevant_segments,
                )
                updated_synthesis = _merge_repair_patch(
                    synthesis,
                    repair_result.synthesis_patch,
                    relevant_segments,
                    extraction=updated_extraction,
                )
            except ValueError:
                raise
            else:
                extraction, synthesis = updated_extraction, updated_synthesis
                repair_count = 1
                review = await self._review(
                    note_input,
                    extraction,
                    synthesis,
                    2,
                    cancel_event,
                    audit_sink,
                    records,
                    context_failures,
                )
                _check_cancellation(cancel_event)
                coverage = self._validate(
                    note_input, extraction, synthesis, review
                )
                if context_failures:
                    coverage = coverage.model_copy(
                        update={"quality_gate": "failed", "repairable": False}
                    )
        _check_cancellation(cancel_event)
        return SemanticGeneration(
            extraction=extraction,
            synthesis=synthesis,
            review=review,
            coverage=coverage,
            repair_count=repair_count,
            provider=self.provider.name,
            model=self.provider.model,
            context_tokens=self.provider.context_tokens,
            usage=_aggregate_usage(records),
            stage_records=tuple(records),
            batch_boundaries=tuple(batch.boundary() for batch in batches),
            prompt_versions=dict(PROMPT_VERSIONS),
            is_local=bool(self.provider.is_local),
            outline=outline,
            section_drafts=section_drafts,
            section_attempts=section_attempts,
            input_sufficient=input_sufficient,
            context_status=(
                "within_budget" if not context_failures else "failed"
            ),
            context_failures=tuple(context_failures),
        )

    async def _review(
        self,
        note_input: NoteInput,
        extraction: ExtractionResult,
        synthesis: SynthesisResult,
        review_pass: int,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None,
        records: list[StageRecord],
        context_failures: list[dict[str, Any]] | None = None,
    ) -> CoverageReview:
        reviews: list[CoverageReview] = []
        claims = _draft_claims(synthesis)
        fragment_segment_indexes: set[int] = set()
        pending_batches = build_segment_batches(
            note_input,
            self.provider.context_tokens,
            input_fraction=0.2,
        )
        source_batches: list[SegmentBatch] = []
        while pending_batches:
            batch = _reindex_batch(
                pending_batches.pop(0),
                len(source_batches),
            )
            source_payload = _source_review_payload(
                note_input,
                extraction,
                synthesis,
                batch,
                review_pass,
            )
            source_fits = self._request_fits(
                "review",
                source_payload,
                CoverageReview,
            )
            claims_fit = all(
                self._request_fits(
                    "review",
                    _claim_review_payload(
                        note_input,
                        extraction,
                        batch,
                        review_pass,
                        [claim],
                    ),
                    DraftClaimReview,
                )
                for claim in claims
            )
            if not source_fits or not claims_fit:
                split_batches = _split_segment_batch(batch)
                if split_batches:
                    pending_batches = [*split_batches, *pending_batches]
                    continue
                budget_payload = (
                    source_payload
                    if not source_fits
                    else _claim_review_payload(
                        note_input,
                        extraction,
                        batch,
                        review_pass,
                        [
                            claim
                            for claim in claims
                            if not self._request_fits(
                                "review",
                                _claim_review_payload(
                                    note_input,
                                    extraction,
                                    batch,
                                    review_pass,
                                    [claim],
                                ),
                                DraftClaimReview,
                            )
                        ][:1],
                    )
                )
                budget_schema: type[BaseModel] = (
                    CoverageReview if not source_fits else DraftClaimReview
                )
                failure = _budget_failure(
                    "review",
                    _provider_request_budget(
                        stage="review",
                        system_prompt=SYSTEM_PROMPTS["review"],
                        payload=budget_payload,
                        json_schema=budget_schema.model_json_schema(),
                        context_tokens=self.provider.context_tokens,
                    ),
                    batch_boundary=batch.boundary(),
                )
                if context_failures is None:
                    raise ProviderContextBudgetError("review", failure)
                _remember_context_failure(context_failures, failure)
                await _record_context_failure(
                    failure,
                    audit_sink,
                    records,
                    batch_boundary=batch.boundary(),
                )
            source_batches.append(batch)
            fragment_segment_indexes = _fragment_segment_indexes(source_batches)

        for batch in source_batches:
            try:
                reviews.append(
                    await self._call(
                        stage="review",
                        payload=_source_review_payload(
                            note_input,
                            extraction,
                            synthesis,
                            batch,
                            review_pass,
                        ),
                        schema=CoverageReview,
                        cancel_event=cancel_event,
                        audit_sink=audit_sink,
                        records=records,
                        batch_boundary=batch.boundary(),
                    )
                )
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(
                    exc,
                    "review",
                    _source_review_payload(
                        note_input,
                        extraction,
                        synthesis,
                        batch,
                        review_pass,
                    ),
                )
                if context_failures is None:
                    raise
                _remember_context_failure(context_failures, failure)
                return _merge_reviews(
                    reviews,
                    fragment_segment_indexes=fragment_segment_indexes,
                )
        merged = _merge_reviews(
            reviews,
            fragment_segment_indexes=fragment_segment_indexes,
        )
        if not claims:
            return merged

        supported_claim_ids: set[str] = set()
        contradicted_claim_ids: set[str] = set()
        for batch in source_batches:
            claim_batches: list[list[DraftClaim]] = []
            current_claims: list[DraftClaim] = []
            for claim in claims:
                trial_claims = [*current_claims, claim]
                trial_payload = _claim_review_payload(
                    note_input,
                    extraction,
                    batch,
                    review_pass,
                    trial_claims,
                )
                if not self._request_fits(
                    "review",
                    _claim_review_payload(
                        note_input,
                        extraction,
                        batch,
                        review_pass,
                        [claim],
                    ),
                    DraftClaimReview,
                ):
                    failure = _budget_failure(
                        "review",
                        _provider_request_budget(
                            stage="review",
                            system_prompt=SYSTEM_PROMPTS["review"],
                            payload=_claim_review_payload(
                                note_input,
                                extraction,
                                batch,
                                review_pass,
                                [claim],
                            ),
                            json_schema=DraftClaimReview.model_json_schema(),
                            context_tokens=self.provider.context_tokens,
                        ),
                        batch_boundary=batch.boundary(),
                    )
                    if context_failures is None:
                        raise ProviderContextBudgetError("review", failure)
                    _remember_context_failure(context_failures, failure)
                    await _record_context_failure(
                        failure,
                        audit_sink,
                        records,
                        batch_boundary=batch.boundary(),
                    )
                    return merged.model_copy(
                        update={
                            "unsupported_draft_claims": _dedupe_text(
                                [
                                    *merged.unsupported_draft_claims,
                                    *(item.text for item in claims),
                                ]
                            )
                        }
                    )
                if current_claims and not self._request_fits(
                    "review",
                    trial_payload,
                    DraftClaimReview,
                ):
                    claim_batches.append(current_claims)
                    current_claims = [claim]
                else:
                    current_claims = trial_claims
            if current_claims:
                claim_batches.append(current_claims)

            for claim_batch in claim_batches:
                expected_ids = {claim.id for claim in claim_batch}
                try:
                    claim_review = await self._call(
                        stage="review",
                        payload=_claim_review_payload(
                            note_input,
                            extraction,
                            batch,
                            review_pass,
                            claim_batch,
                        ),
                        schema=DraftClaimReview,
                        cancel_event=cancel_event,
                        audit_sink=audit_sink,
                        records=records,
                        batch_boundary=batch.boundary(),
                    )
                except ProviderContextBudgetError as exc:
                    failure = _exception_failure(
                        exc,
                        "review",
                        _claim_review_payload(
                            note_input,
                            extraction,
                            batch,
                            review_pass,
                            claim_batch,
                        ),
                    )
                    if context_failures is None:
                        raise
                    _remember_context_failure(context_failures, failure)
                    return merged.model_copy(
                        update={
                            "unsupported_draft_claims": _dedupe_text(
                                [
                                    *merged.unsupported_draft_claims,
                                    *(item.text for item in claims),
                                ]
                            )
                        }
                    )
                reviewed_ids = set(claim_review.reviewed_claim_ids)
                supported_ids = set(claim_review.supported_claim_ids)
                contradicted_ids = set(claim_review.contradicted_claim_ids)
                if reviewed_ids != expected_ids:
                    raise ValueError("draft claim review did not account for its batch")
                if not supported_ids <= expected_ids or not contradicted_ids <= expected_ids:
                    raise ValueError("draft claim review referenced an unknown claim")
                if supported_ids & contradicted_ids:
                    raise ValueError("draft claim review returned conflicting verdicts")
                supported_claim_ids.update(supported_ids)
                contradicted_claim_ids.update(contradicted_ids)

        claim_by_id = {claim.id: claim for claim in claims}
        unsupported_ids = contradicted_claim_ids | (
            set(claim_by_id) - supported_claim_ids
        )
        return merged.model_copy(
            update={
                "unsupported_draft_claims": _dedupe_text(
                    [
                        *merged.unsupported_draft_claims,
                        *(claim_by_id[claim_id].text for claim_id in sorted(unsupported_ids)),
                    ]
                )
            }
        )

    async def _build_outline(
        self,
        note_input: NoteInput,
        extraction: ExtractionResult,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None,
        records: list[StageRecord],
    ) -> TeachingOutline:
        catalog = [_atom_catalog_entry(atom) for atom in extraction.atoms]
        chunks: list[list[dict[str, Any]]] = [catalog]
        outlines: list[TeachingOutline] = []
        while chunks:
            chunk = chunks.pop(0)
            indexes = {item["source_segment"] for item in chunk}
            payload = {
                "prompt_version": PROMPT_VERSIONS["outline"],
                "aweme_id": note_input.aweme_id,
                "metadata": _metadata_payload(note_input),
                "atom_catalog": chunk,
                "source_segments": [
                    _segment_payload(s)
                    for s in note_input.segments
                    if s.index in indexes
                ],
                "input_sufficient": bool(note_input.transcript.strip()),
            }
            if not self._request_fits("outline", payload, TeachingOutline):
                if len(chunk) <= 1:
                    raise ProviderContextBudgetError("outline")
                midpoint = max(1, len(chunk) // 2)
                chunks = [chunk[:midpoint], chunk[midpoint:], *chunks]
                continue
            outlines.append(
                await self._call(
                    stage="outline",
                    payload=payload,
                    schema=TeachingOutline,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                )
            )
        merged = _merge_outlines(outlines)
        return _normalize_outline(merged, extraction)

    async def _synthesize_sections(
        self,
        note_input: NoteInput,
        extraction: ExtractionResult,
        outline: TeachingOutline,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None,
        records: list[StageRecord],
    ) -> tuple[tuple[SectionDraft, ...], dict[str, int]]:
        atom_by_id = {atom.id: atom for atom in extraction.atoms}
        drafts: list[SectionDraft] = []
        attempts: dict[str, int] = {}
        for section in sorted(outline.sections, key=lambda item: item.order):
            section_atoms = [
                atom_by_id[atom_id]
                for atom_id in section.atom_ids
                if atom_id in atom_by_id
            ]
            attempts[section.section_id] = 0
            chunks = [section_atoms]
            chunk_drafts: list[SectionDraft] = []
            failed = False
            while chunks:
                chunk = chunks.pop(0)
                payload = _section_payload(note_input, section, chunk)
                if not self._request_fits("section", payload, SectionDraft):
                    if len(chunk) <= 1:
                        failed = True
                        break
                    midpoint = max(1, len(chunk) // 2)
                    chunks = [chunk[:midpoint], chunk[midpoint:], *chunks]
                    continue
                result: SectionDraft | None = None
                for attempt in range(1, 3):
                    attempts[section.section_id] += 1
                    try:
                        result = await self._call(
                            stage="section",
                            payload=payload,
                            schema=SectionDraft,
                            cancel_event=cancel_event,
                            audit_sink=audit_sink,
                            records=records,
                        )
                        break
                    except (ValueError, ProviderContextBudgetError):
                        if attempt == 2:
                            failed = True
                if result is not None:
                    chunk_drafts.append(result)
                if failed:
                    break
            if failed or not chunk_drafts:
                result = SectionDraft(
                    section_id=section.section_id,
                    title=section.title,
                    content="视频未提供这一类信息。",
                    included_atom_ids=[],
                    unprovided_items=[section.section_id],
                )
                drafts.append(_normalize_section(result, section, atom_by_id))
            else:
                drafts.append(
                    _normalize_section(
                        _merge_section_drafts(chunk_drafts, section),
                        section,
                        atom_by_id,
                    )
                )
        return tuple(drafts), attempts

    async def _call[T: BaseModel](
        self,
        *,
        stage: NoteStage,
        payload: Mapping[str, Any],
        schema: type[T],
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None,
        records: list[StageRecord],
        batch_boundary: dict[str, int] | None = None,
    ) -> T:
        _check_cancellation(cancel_event)
        system_prompt = SYSTEM_PROMPTS[stage]
        json_schema = schema.model_json_schema()
        request_payload = copy.deepcopy(dict(payload))
        budget = _provider_request_budget(
            stage=stage,
            system_prompt=system_prompt,
            payload=request_payload,
            json_schema=json_schema,
            context_tokens=self.provider.context_tokens,
        )
        request_payload["context_budget"] = budget
        payload = request_payload
        if not budget["within_budget"]:
            failure = {
                "stage": stage,
                "code": "provider_context_budget_exceeded",
                **budget,
            }
            await _record_context_failure(failure, audit_sink, records)
            raise ProviderContextBudgetError(stage, failure)
        started = time.perf_counter()
        try:
            response = await self.provider.complete_json(
                stage=stage,
                system_prompt=system_prompt,
                payload=payload,
                json_schema=json_schema,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            record = StageRecord(
                stage=stage,
                prompt_version=PROMPT_VERSIONS[stage],
                system_prompt=system_prompt,
                payload=copy.deepcopy(dict(payload)),
                raw_output="",
                parsed_json={},
                usage=ProviderUsage(
                    input_tokens=None,
                    output_tokens=None,
                    api_cost=None,
                    cost_status="unknown",
                ),
                diagnostics={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                elapsed_ms=(time.perf_counter() - started) * 1000,
                batch_boundary=copy.deepcopy(batch_boundary),
            )
            records.append(record)
            await _emit_audit(audit_sink, record)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        diagnostics = copy.deepcopy(response.diagnostics)
        if cancel_event.is_set():
            diagnostics["cancelled_after_provider_return"] = True
            record = StageRecord(
                stage=stage,
                prompt_version=PROMPT_VERSIONS[stage],
                system_prompt=system_prompt,
                payload=copy.deepcopy(dict(payload)),
                raw_output=response.raw_output,
                parsed_json=copy.deepcopy(response.parsed_json),
                usage=response.usage,
                diagnostics=diagnostics,
                elapsed_ms=elapsed_ms,
                batch_boundary=copy.deepcopy(batch_boundary),
            )
            records.append(record)
            await _emit_audit(audit_sink, record)
            _check_cancellation(cancel_event)
        try:
            parsed_json = copy.deepcopy(response.parsed_json)
            if stage == "repair":
                parsed_json = _materialize_repair_response(
                    parsed_json,
                    request_payload,
                )
            parsed = schema.model_validate(parsed_json, strict=True)
        except ValueError as exc:
            diagnostics["schema_validation_error"] = str(exc)
            record = StageRecord(
                stage=stage,
                prompt_version=PROMPT_VERSIONS[stage],
                system_prompt=system_prompt,
                payload=copy.deepcopy(dict(payload)),
                raw_output=response.raw_output,
                parsed_json=copy.deepcopy(parsed_json),
                usage=response.usage,
                diagnostics=diagnostics,
                elapsed_ms=elapsed_ms,
                batch_boundary=copy.deepcopy(batch_boundary),
            )
            records.append(record)
            await _emit_audit(audit_sink, record)
            raise
        record = StageRecord(
            stage=stage,
            prompt_version=PROMPT_VERSIONS[stage],
            system_prompt=system_prompt,
            payload=copy.deepcopy(dict(payload)),
            raw_output=response.raw_output,
            parsed_json=parsed.model_dump(mode="json"),
            usage=response.usage,
            diagnostics=diagnostics,
            elapsed_ms=elapsed_ms,
            batch_boundary=copy.deepcopy(batch_boundary),
        )
        records.append(record)
        await _emit_audit(audit_sink, record)
        _check_cancellation(cancel_event)
        return parsed

    def _validate(
        self,
        note_input: NoteInput,
        extraction: ExtractionResult,
        synthesis: SynthesisResult,
        review: CoverageReview,
    ) -> CoverageReport:
        return self.validator.validate(
            segment_count=len(note_input.segments),
            extraction=extraction,
            synthesis=synthesis,
            review=review,
            semantic_generator=f"{self.provider.name}/{self.provider.model}",
        )

    def _failed_generation(
        self,
        *,
        note_input: NoteInput,
        extraction: ExtractionResult | None,
        synthesis: SynthesisResult | None,
        review: CoverageReview | None,
        context_failures: list[dict[str, Any]],
        records: list[StageRecord],
        batches: list[SegmentBatch],
        coverage: CoverageReport | None = None,
    ) -> SemanticGeneration:
        candidate_extraction = _complete_candidate_extraction(
            note_input,
            extraction,
        )
        candidate_synthesis = synthesis or _failed_synthesis(
            note_input,
            candidate_extraction,
            context_failures[-1] if context_failures else None,
        )
        candidate_review = review or CoverageReview(
            reviewed_segment_indexes=[],
            source_omission_candidates=[],
            unsupported_draft_claims=[],
            disposition_corrections=[],
        )
        if coverage is None:
            coverage = self._validate(
                note_input,
                candidate_extraction,
                candidate_synthesis,
                candidate_review,
            )
        coverage = coverage.model_copy(
            update={"quality_gate": "failed", "repairable": False}
        )
        return SemanticGeneration(
            extraction=candidate_extraction,
            synthesis=candidate_synthesis,
            review=candidate_review,
            coverage=coverage,
            repair_count=0,
            provider=self.provider.name,
            model=self.provider.model,
            context_tokens=self.provider.context_tokens,
            usage=_aggregate_usage(records),
            stage_records=tuple(records),
            batch_boundaries=tuple(batch.boundary() for batch in batches),
            prompt_versions=dict(PROMPT_VERSIONS),
            is_local=bool(self.provider.is_local),
            outline=_empty_teaching_outline(),
            section_drafts=(),
            section_attempts={},
            input_sufficient=False,
            context_status="failed",
            context_failures=tuple(context_failures),
        )


def _batch(
    batch_index: int,
    segments: list[InputSegment],
    input_chars: int,
    estimated_tokens: int,
) -> SegmentBatch:
    return SegmentBatch(
        batch_index=batch_index,
        start_segment=segments[0].index,
        end_segment=segments[-1].index,
        segments=tuple(segments),
        input_chars=input_chars,
        estimated_tokens=estimated_tokens,
    )


def _reindex_batch(batch: SegmentBatch, batch_index: int) -> SegmentBatch:
    return SegmentBatch(
        batch_index=batch_index,
        start_segment=batch.start_segment,
        end_segment=batch.end_segment,
        segments=batch.segments,
        input_chars=batch.input_chars,
        estimated_tokens=batch.estimated_tokens,
    )


def _split_segment_batch(batch: SegmentBatch) -> list[SegmentBatch]:
    segments = list(batch.segments)
    if len(segments) > 1:
        midpoint = len(segments) // 2
        groups = (segments[:midpoint], segments[midpoint:])
    else:
        segment = segments[0]
        if len(segment.text) <= 1:
            return []
        midpoint = len(segment.text) // 2
        groups = (
            [
                InputSegment(
                    index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text[:midpoint],
                )
            ],
            [
                InputSegment(
                    index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text[midpoint:],
                )
            ],
        )
    return [
        _batch(
            0,
            group,
            sum(len(segment.text) for segment in group),
            sum(max(1, math.ceil(len(segment.text) / 1.5)) for segment in group),
        )
        for group in groups
    ]


def _segment_payload(segment: InputSegment) -> dict[str, Any]:
    return {
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }


def _extraction_payload(
    note_input: NoteInput,
    batch: SegmentBatch,
    prior_catalog: list[dict[str, Any]],
    prior_atoms: list[Any],
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSIONS["extract"],
        "aweme_id": note_input.aweme_id,
        "batch": batch.boundary(),
        "segments": [_segment_payload(segment) for segment in batch.segments],
        "prior_evidence_catalog": prior_catalog,
        "prior_evidence_atoms": [_project_atom(atom) for atom in prior_atoms],
        "input_truncated": False,
    }


def _metadata_payload(note_input: NoteInput) -> dict[str, Any]:
    return {
        "description": note_input.description,
        "author": note_input.author,
        "author_unique_id": note_input.author_unique_id,
        "published_at": note_input.published_at,
        "duration_ms": note_input.duration_ms,
        "source_url": note_input.source_url,
        "warnings": list(note_input.warnings),
    }


def _project_extraction(
    extraction: ExtractionResult,
    *,
    include_source_excerpts: bool = True,
) -> dict[str, Any]:
    """Build a lossless, prompt-only view of extraction evidence.

    The persisted extraction keeps the strict full atom contract.  Prompt payloads
    may reference a repeated statement from its source excerpt, or the complete
    source segment during review, without serializing the same text twice.
    """
    atoms = [
        _project_atom(
            atom,
            include_source_excerpt=include_source_excerpts,
        )
        for atom in extraction.atoms
    ]
    return {
        "atoms": atoms,
        "dispositions": [
            item.model_dump(mode="json") for item in extraction.dispositions
        ],
    }


def _project_atom(
    atom: Any,
    *,
    include_source_excerpt: bool = True,
) -> dict[str, Any]:
    data = (
        atom.model_dump(mode="json")
        if isinstance(atom, BaseModel)
        else copy.deepcopy(dict(atom))
    )
    statement = data.get("statement")
    excerpt = data.get("source_excerpt")
    if statement == excerpt:
        data.pop("statement", None)
        data["source_excerpt_is_statement"] = True
    if not include_source_excerpt:
        data.pop("source_excerpt", None)
        if data.pop("source_excerpt_is_statement", False):
            data["statement_in_source_segments"] = True
        data["source_excerpt_in_source_segments"] = True
    return data


def _project_synthesis(
    synthesis: SynthesisResult,
    extraction: ExtractionResult,
    *,
    compact_content: bool = False,
) -> dict[str, Any]:
    data = copy.deepcopy(synthesis.model_dump(mode="json"))
    if not compact_content:
        return data

    atom_text = {
        atom.id: atom.source_excerpt
        if atom.statement == atom.source_excerpt
        else atom.statement
        for atom in extraction.atoms
    }

    def compact_item(item: dict[str, Any]) -> dict[str, Any]:
        atom_ids = [str(value) for value in item.get("atom_ids", [])]
        content = item.get("content")
        if not isinstance(content, str) or not atom_ids:
            return item
        referenced = [atom_text[atom_id] for atom_id in atom_ids if atom_id in atom_text]
        if len(referenced) == 1 and content == referenced[0]:
            item.pop("content", None)
            item["content_from_atom_ids"] = atom_ids
        elif referenced and content == "；".join(referenced):
            item.pop("content", None)
            item["content_from_atom_ids"] = atom_ids
        return item

    for field_name in (
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "promotions",
    ):
        data[field_name] = [
            compact_item(item) for item in data[field_name]
        ]
    data["timeline"] = [compact_item(item) for item in data["timeline"]]
    return data


def _synthesis_payload(
    note_input: NoteInput,
    extraction: ExtractionResult,
    slice_index: int,
    slice_count: int,
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSIONS["synthesize"],
        "aweme_id": note_input.aweme_id,
        "metadata": _metadata_payload(note_input),
        "slice": {"index": slice_index, "count": slice_count},
        "extraction": _project_extraction(extraction),
        "input_truncated": False,
    }


def _atom_catalog_entry(atom: Any) -> dict[str, Any]:
    return {
        "id": atom.id,
        "kind": atom.kind,
        "topic": atom.topic,
        "statement": atom.statement,
        "source_segment": atom.source_segment,
        "timestamps": [value for value in (atom.start, atom.end) if value is not None],
        "relations": list(atom.relations),
        "uncertainty": list(atom.asr_uncertainty),
    }


_TEACHING_SECTION_IDS = (
    "overview",
    "core_points",
    "methods_steps",
    "examples_results",
    "boundaries_risks",
    "actions",
    "timeline",
    "evidence_limits",
)


def _normalize_outline(
    outline: TeachingOutline,
    extraction: ExtractionResult,
) -> TeachingOutline:
    atom_ids = {atom.id for atom in extraction.atoms}
    by_id = {section.section_id: section for section in outline.sections}
    if (
        len(by_id) != len(outline.sections)
        or set(by_id) != set(_TEACHING_SECTION_IDS)
    ):
        raise ValueError("outline must contain the fixed teaching sections")
    assigned: set[str] = set()
    sections: list[TeachingOutlineSection] = []
    for order, section_id in enumerate(_TEACHING_SECTION_IDS, 1):
        section = by_id[section_id]
        ids = [
            atom_id
            for atom_id in section.atom_ids
            if atom_id in atom_ids and atom_id not in assigned
        ]
        assigned.update(ids)
        sections.append(
            section.model_copy(update={"order": order, "atom_ids": ids})
        )
    unassigned = [atom.id for atom in extraction.atoms if atom.id not in assigned]
    if unassigned:
        limit_index = _TEACHING_SECTION_IDS.index("evidence_limits")
        sections[limit_index] = sections[limit_index].model_copy(
            update={"atom_ids": [*sections[limit_index].atom_ids, *unassigned]}
        )
    return outline.model_copy(
        update={
            "sections": sections,
            "included_atom_ids": [atom.id for atom in extraction.atoms],
            "unassigned_atom_ids": [],
        }
    )


def _merge_outlines(values: list[TeachingOutline]) -> TeachingOutline:
    sections: dict[str, TeachingOutlineSection] = {}
    included: list[str] = []
    for outline in values:
        for section in outline.sections:
            if section.section_id not in sections:
                sections[section.section_id] = section
            else:
                current = sections[section.section_id]
                sections[section.section_id] = current.model_copy(
                    update={
                        "atom_ids": _dedupe_text(
                            [*current.atom_ids, *section.atom_ids]
                        ),
                        "timestamps": sorted(
                            {*current.timestamps, *section.timestamps}
                        ),
                    }
                )
        included.extend(outline.included_atom_ids)
    return TeachingOutline(
        sections=list(sections.values()),
        included_atom_ids=_dedupe_text(included),
        unassigned_atom_ids=_dedupe_text(
            atom_id for outline in values for atom_id in outline.unassigned_atom_ids
        ),
        input_sufficient=all(outline.input_sufficient for outline in values),
    )


def _section_payload(
    note_input: NoteInput,
    section: TeachingOutlineSection,
    atoms: list[Any],
) -> dict[str, Any]:
    indexes = {atom.source_segment for atom in atoms}
    return {
        "prompt_version": PROMPT_VERSIONS["section"],
        "aweme_id": note_input.aweme_id,
        "section": section.model_dump(mode="json"),
        "atoms": [_atom_catalog_entry(atom) for atom in atoms],
        "source_segments": [
            _segment_payload(segment)
            for segment in note_input.segments
            if segment.index in indexes
        ],
        "input_truncated": False,
    }


def _normalize_section(
    draft: SectionDraft,
    outline_section: TeachingOutlineSection,
    atom_by_id: Mapping[str, Any],
) -> SectionDraft:
    allowed = set(outline_section.atom_ids)
    included = [atom_id for atom_id in draft.included_atom_ids if atom_id in allowed]
    if not included:
        included = [atom_id for atom_id in outline_section.atom_ids if atom_id in atom_by_id]
    return draft.model_copy(
        update={
            "section_id": outline_section.section_id,
            "title": outline_section.title,
            "included_atom_ids": included,
        }
    )


def _merge_section_drafts(
    values: list[SectionDraft],
    section: TeachingOutlineSection,
) -> SectionDraft:
    if not values:
        return SectionDraft(
            section_id=section.section_id,
            title=section.title,
            content="视频未提供这一类信息。",
            included_atom_ids=[],
        )
    data = values[0].model_dump(mode="json")
    data["content"] = "\n".join(item.content for item in values)
    for field_name in (
        "claims",
        "steps",
        "examples",
        "results",
        "conditions",
        "risks",
        "actions",
        "unprovided_items",
        "generated_interpretations",
    ):
        data[field_name] = [
            item.model_dump(mode="json")
            if hasattr(item, "model_dump")
            else item
            for value in values
            for item in getattr(value, field_name)
        ]
    data["included_atom_ids"] = _dedupe_text(
        atom_id for value in values for atom_id in value.included_atom_ids
    )
    return SectionDraft.model_validate(data, strict=True)


def _teaching_synthesis(
    note_input: NoteInput,
    extraction: ExtractionResult,
    outline: TeachingOutline,
    drafts: tuple[SectionDraft, ...],
) -> SynthesisResult:
    by_id = {draft.section_id: draft for draft in drafts}
    atom_by_id = {atom.id: atom for atom in extraction.atoms}

    def theme(section_id: str) -> list[ThemeSection]:
        draft = by_id.get(section_id)
        if draft is None:
            return []
        return [
            ThemeSection(
                heading=draft.title,
                content=draft.content,
                atom_ids=list(draft.included_atom_ids),
                timestamps=sorted(
                    {
                        timestamp
                        for item in (
                            *draft.claims,
                            *draft.steps,
                            *draft.examples,
                            *draft.results,
                            *draft.conditions,
                            *draft.risks,
                            *draft.actions,
                        )
                        for timestamp in item.timestamps
                    }
                ),
            )
        ]

    timeline = [
        TimelineEntry(
            segment_index=disposition.segment_index,
            start=next(
                (s.start for s in note_input.segments if s.index == disposition.segment_index),
                None,
            ),
            end=next(
                (s.end for s in note_input.segments if s.index == disposition.segment_index),
                None,
            ),
            disposition=disposition.disposition,
            atom_ids=list(disposition.atom_ids),
            content=(
                "；".join(
                    atom_by_id[atom_id].statement
                    for atom_id in disposition.atom_ids
                    if atom_id in atom_by_id
                )
                or "无内容"
            ),
        )
        for disposition in sorted(extraction.dispositions, key=lambda item: item.segment_index)
    ]
    overview = by_id.get("overview")
    promotion_sections = [
        ThemeSection(
            heading=atom.topic,
            content=atom.statement,
            atom_ids=[atom.id],
            timestamps=[
                value for value in (atom.start, atom.end) if value is not None
            ],
        )
        for atom in extraction.atoms
        if atom.kind == "promotion" or atom.is_promotional
    ]
    return SynthesisResult(
        title=note_input.description or note_input.aweme_id,
        overview=overview.content if overview else "视频未提供这一类信息。",
        structure=[section.title for section in outline.sections],
        themes=theme("core_points"),
        methods_and_steps=theme("methods_steps"),
        examples_and_counterexamples=theme("examples_results"),
        boundaries_and_risks=theme("boundaries_risks"),
        concepts=_dedupe_text(atom.topic for atom in extraction.atoms if atom.kind == "definition"),
        quotes=_dedupe_text(atom.statement for atom in extraction.atoms if atom.kind == "quote"),
        implications=_dedupe_text(draft.content for draft in drafts),
        actions=_dedupe_text(item.text for draft in drafts for item in draft.actions),
        timeline=timeline,
        promotions=promotion_sections,
        tags=[],
        included_atom_ids=[atom.id for atom in extraction.atoms],
    )


def _source_review_payload(
    note_input: NoteInput,
    extraction: ExtractionResult,
    synthesis: SynthesisResult,
    batch: SegmentBatch,
    review_pass: int,
) -> dict[str, Any]:
    segment_indexes = {segment.index for segment in batch.segments}
    atom_ids = {
        atom_id
        for entry in synthesis.timeline
        if entry.segment_index in segment_indexes
        for atom_id in entry.atom_ids
    }
    review_atom_ids = {
        atom.id
        for atom in extraction.atoms
        if atom.source_segment in segment_indexes
    }
    return {
        "prompt_version": PROMPT_VERSIONS["review"],
        "aweme_id": note_input.aweme_id,
        "review_pass": review_pass,
        "review_mode": "source_coverage",
        "source_segments": [
            _segment_payload(segment) for segment in batch.segments
        ],
        "extraction": _project_extraction(
            _subset_extraction(
                extraction,
                review_atom_ids,
                segment_indexes,
            ),
            include_source_excerpts=False,
        ),
        "draft": _project_synthesis(
            _slice_synthesis(
                synthesis,
                segment_indexes,
                atom_ids,
            ),
            extraction,
            compact_content=True,
        ),
        "input_truncated": False,
    }


def _claim_review_payload(
    note_input: NoteInput,
    extraction: ExtractionResult,
    batch: SegmentBatch,
    review_pass: int,
    claims: list[DraftClaim],
) -> dict[str, Any]:
    segment_indexes = {segment.index for segment in batch.segments}
    review_atom_ids = {
        atom.id
        for atom in extraction.atoms
        if atom.source_segment in segment_indexes
    }
    return {
        "prompt_version": PROMPT_VERSIONS["review"],
        "aweme_id": note_input.aweme_id,
        "review_pass": review_pass,
        "review_mode": "draft_claims",
        "source_segments": [
            _segment_payload(segment) for segment in batch.segments
        ],
        "extraction": _project_extraction(
            _subset_extraction(
                extraction,
                review_atom_ids,
                segment_indexes,
            ),
            include_source_excerpts=False,
        ),
        "draft_claims": [claim.model_dump(mode="json") for claim in claims],
        "input_truncated": False,
    }


def _repair_payload(
    note_input: NoteInput,
    repair_extraction: ExtractionResult,
    synthesis: SynthesisResult,
    extraction: ExtractionResult,
    review: CoverageReview,
    coverage: CoverageReport,
    relevant_segments: set[int],
    *,
    compact_content: bool,
) -> dict[str, Any]:
    relevant_ids = {
        atom_id
        for entry in synthesis.timeline
        if entry.segment_index in relevant_segments
        for atom_id in entry.atom_ids
    }
    draft = synthesis
    if compact_content:
        draft = _slice_synthesis(synthesis, relevant_segments, relevant_ids)
    draft_payload = _project_synthesis(
        draft,
        extraction,
        compact_content=compact_content,
    )
    return {
        "prompt_version": PROMPT_VERSIONS["repair"],
        "aweme_id": note_input.aweme_id,
        "extraction": _project_extraction(repair_extraction),
        "draft": draft_payload,
        "review": _slice_review(
            review,
            relevant_segments,
        ).model_dump(mode="json"),
        "coverage_report": coverage.model_dump(
            mode="json",
            by_alias=True,
        ),
        "input_truncated": False,
    }


def _materialize_repair_response(
    parsed: dict[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand prompt-only repair markers before strict schema validation."""
    result = copy.deepcopy(parsed)
    extraction_payload = payload.get("extraction")
    extraction_atoms = (
        extraction_payload.get("atoms", [])
        if isinstance(extraction_payload, Mapping)
        else []
    )
    atom_text = {
        atom.get("id"): atom.get("statement") or atom.get("source_excerpt")
        for atom in extraction_atoms
        if isinstance(atom, Mapping)
        and isinstance(atom.get("id"), str)
        and isinstance(atom.get("statement") or atom.get("source_excerpt"), str)
    }

    def materialize_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        if "content" not in item and isinstance(
            item.get("content_from_atom_ids"), list
        ):
            values = [
                atom_text[atom_id]
                for atom_id in item["content_from_atom_ids"]
                if atom_id in atom_text
            ]
            if values:
                item["content"] = "；".join(values)
            item.pop("content_from_atom_ids", None)

    synthesis_patch = result.get("synthesis_patch")
    if isinstance(synthesis_patch, dict):
        for field_name in (
            "themes",
            "methods_and_steps",
            "examples_and_counterexamples",
            "boundaries_and_risks",
            "promotions",
        ):
            for item in synthesis_patch.get(field_name, []):
                materialize_item(item)
        for item in synthesis_patch.get("timeline", []):
            materialize_item(item)

    extraction_patch = result.get("extraction_patch")
    if isinstance(extraction_patch, dict):
        for atom in extraction_patch.get("atoms", []):
            if not isinstance(atom, dict):
                continue
            if "statement" not in atom and atom.pop(
                "source_excerpt_is_statement", False
            ):
                source_excerpt = atom.get("source_excerpt")
                if isinstance(source_excerpt, str):
                    atom["statement"] = source_excerpt
            atom.pop("statement_in_source_segments", None)
            atom.pop("source_excerpt_in_source_segments", None)
    return result
def _context_budget(
    estimated_tokens: int,
    context_tokens: int,
) -> dict[str, Any]:
    source_token_limit = max(1, math.floor(context_tokens * 0.5))
    return {
        "accounting": "source-evidence-v1",
        "estimated_tokens": estimated_tokens,
        "source_token_limit": source_token_limit,
        "context_tokens": context_tokens,
        "within_budget": estimated_tokens <= source_token_limit,
    }


def _budget_failure(
    stage: NoteStage,
    budget: Mapping[str, Any],
    *,
    batch_boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "stage": stage,
        "code": "provider_context_budget_exceeded",
        **dict(budget),
    }
    if batch_boundary is not None:
        failure["batch_boundary"] = copy.deepcopy(dict(batch_boundary))
    return failure


def _remember_context_failure(
    failures: list[dict[str, Any]],
    failure: Mapping[str, Any],
) -> None:
    key = (
        failure.get("stage"),
        failure.get("code"),
        failure.get("estimated_total_tokens"),
        failure.get("batch_boundary"),
    )
    if not any(
        (
            existing.get("stage"),
            existing.get("code"),
            existing.get("estimated_total_tokens"),
            existing.get("batch_boundary"),
        )
        == key
        for existing in failures
    ):
        failures.append(copy.deepcopy(dict(failure)))


def _exception_failure(
    error: ProviderContextBudgetError,
    stage: NoteStage,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if error.failure:
        return copy.deepcopy(error.failure)
    return {
        "stage": stage,
        "code": "provider_context_budget_exceeded",
        "payload_tokens": len(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "context_budget"
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
    }


def _complete_candidate_extraction(
    note_input: NoteInput,
    extraction: ExtractionResult | None,
) -> ExtractionResult:
    atoms = [
        atom.model_dump(mode="json")
        for atom in (extraction.atoms if extraction is not None else [])
    ]
    dispositions = [
        item.model_dump(mode="json")
        for item in (extraction.dispositions if extraction is not None else [])
    ]
    existing_indexes = {item["segment_index"] for item in dispositions}
    for segment in note_input.segments:
        if segment.index in existing_indexes:
            continue
        dispositions.append(
            {
                "segment_index": segment.index,
                "disposition": "unintelligible_asr",
                "atom_ids": [],
                "reason": "未完成语义提取；提供方上下文预算不足。",
            }
        )
    return ExtractionResult.model_validate(
        {
            "atoms": atoms,
            "dispositions": _coalesce_dispositions(dispositions),
        },
        strict=True,
    )


def _failed_synthesis(
    note_input: NoteInput,
    extraction: ExtractionResult,
    failure: Mapping[str, Any] | None,
) -> SynthesisResult:
    disposition_by_index = {
        item.segment_index: item for item in extraction.dispositions
    }
    included_ids = [atom.id for atom in extraction.atoms]
    timeline: list[dict[str, Any]] = []
    reason = (
        str(failure.get("stage")) if failure is not None else "unknown"
    )
    for segment in note_input.segments:
        disposition = disposition_by_index.get(segment.index)
        if disposition is None:
            disposition_name = "unintelligible_asr"
            atom_ids: list[str] = []
        else:
            disposition_name = disposition.disposition
            atom_ids = list(disposition.atom_ids)
        timeline.append(
            {
                "segment_index": segment.index,
                "start": segment.start,
                "end": segment.end,
                "disposition": disposition_name,
                "atom_ids": atom_ids,
                "content": segment.text,
            }
        )
    return SynthesisResult.model_validate(
        {
            "title": "语义生成未完成",
            "overview": (
                f"提供方上下文预算不足（{reason}），未完成语义合成与覆盖审查；"
                "以下时间线仅忠实保留原始分段。"
            ),
            "structure": [],
            "themes": [],
            "methods_and_steps": [],
            "examples_and_counterexamples": [],
            "boundaries_and_risks": [],
            "concepts": [],
            "quotes": [],
            "implications": [],
            "actions": [],
            "timeline": timeline,
            "promotions": [],
            "tags": [],
            "included_atom_ids": included_ids,
        },
        strict=True,
    )


def _provider_request_budget(
    *,
    stage: NoteStage,
    system_prompt: str,
    payload: Mapping[str, Any],
    json_schema: Mapping[str, Any],
    context_tokens: int,
) -> dict[str, int | str | bool]:
    payload_without_budget = {
        key: value for key, value in payload.items() if key != "context_budget"
    }
    system_prompt_tokens = len(system_prompt.encode("utf-8"))
    payload_tokens = len(
        json.dumps(
            payload_without_budget,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    json_schema_tokens = len(
        json.dumps(
            json_schema,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    budget_metadata_tokens = 512
    output_floor, output_ratio = {
        "extract": (1_024, 2.0),
        "outline": (1_024, 1.0),
        "synthesize": (1_024, 1.5),
        "section": (1_024, 1.5),
        "review": (768, 0.75),
        "repair": (2_048, 1.5),
    }[stage]
    reserved_output_tokens = max(
        output_floor,
        math.ceil(payload_tokens * output_ratio),
    )
    estimated_input_tokens = (
        system_prompt_tokens
        + payload_tokens
        + json_schema_tokens
        + budget_metadata_tokens
    )
    estimated_total_tokens = estimated_input_tokens + reserved_output_tokens
    return {
        "method": "provider-request-v2",
        "system_prompt_tokens": system_prompt_tokens,
        "payload_tokens": payload_tokens,
        "json_schema_tokens": json_schema_tokens,
        "budget_metadata_tokens": budget_metadata_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "estimated_tokens": estimated_total_tokens,
        "context_tokens": context_tokens,
        "within_budget": estimated_total_tokens <= context_tokens,
    }


def _bounded_prior_context(
    atoms: list[Any],
    batch_tokens: int,
    context_tokens: int,
) -> tuple[list[Any], int]:
    remaining = max(
        0,
        math.floor(context_tokens * 0.5) - batch_tokens,
    )
    selected: list[Any] = []
    used = 0
    for atom in reversed(atoms):
        atom_tokens = max(1, math.ceil(len(atom.source_excerpt) / 1.5))
        if used + atom_tokens > remaining:
            continue
        selected.append(atom)
        used += atom_tokens
    selected.reverse()
    return selected, used


def _build_extraction_slices(
    extraction: ExtractionResult,
    context_tokens: int,
) -> list[tuple[ExtractionResult, int]]:
    token_limit = max(1, math.floor(context_tokens * 0.5))
    atom_groups: list[list[str]] = []
    current_ids: list[str] = []
    current_tokens = 0
    for atom in sorted(
        extraction.atoms,
        key=lambda item: (item.source_segment, item.start or 0, item.id),
    ):
        atom_tokens = max(1, math.ceil(len(atom.source_excerpt) / 1.5))
        if current_ids and current_tokens + atom_tokens > token_limit:
            atom_groups.append(current_ids)
            current_ids = []
            current_tokens = 0
        current_ids.append(atom.id)
        current_tokens += atom_tokens
    if current_ids:
        atom_groups.append(current_ids)
    if not atom_groups:
        atom_groups = [[]]
    content_segments = {atom.source_segment for atom in extraction.atoms}
    noncontent_segments = {
        item.segment_index
        for item in extraction.dispositions
        if item.segment_index not in content_segments
    }
    results: list[tuple[ExtractionResult, int]] = []
    for index, atom_ids in enumerate(atom_groups):
        selected_ids = set(atom_ids)
        segment_indexes = {
            atom.source_segment
            for atom in extraction.atoms
            if atom.id in selected_ids
        }
        if index == 0:
            segment_indexes.update(noncontent_segments)
        subset = _subset_extraction(extraction, selected_ids, segment_indexes)
        results.append((subset, _extraction_source_tokens(subset)))
    return results


def _split_extraction_slice(
    extraction: ExtractionResult,
) -> list[ExtractionResult]:
    if len(extraction.atoms) <= 1:
        return []
    midpoint = len(extraction.atoms) // 2
    atom_groups = (
        extraction.atoms[:midpoint],
        extraction.atoms[midpoint:],
    )
    content_segments = {atom.source_segment for atom in extraction.atoms}
    noncontent_segments = {
        item.segment_index
        for item in extraction.dispositions
        if item.segment_index not in content_segments
    }
    result: list[ExtractionResult] = []
    for index, atoms in enumerate(atom_groups):
        atom_ids = {atom.id for atom in atoms}
        segment_indexes = {atom.source_segment for atom in atoms}
        if index == 0:
            segment_indexes.update(noncontent_segments)
        result.append(
            _subset_extraction(extraction, atom_ids, segment_indexes)
        )
    return result


def _subset_extraction(
    extraction: ExtractionResult,
    atom_ids: set[str],
    segment_indexes: set[int],
) -> ExtractionResult:
    atoms = [
        atom.model_dump(mode="json")
        for atom in extraction.atoms
        if atom.id in atom_ids
    ]
    dispositions: list[dict[str, Any]] = []
    for item in extraction.dispositions:
        selected = [atom_id for atom_id in item.atom_ids if atom_id in atom_ids]
        if item.segment_index not in segment_indexes and not selected:
            continue
        data = item.model_dump(mode="json")
        data["atom_ids"] = selected
        if not selected and data["disposition"] in {"included", "promotion"}:
            continue
        dispositions.append(data)
    return ExtractionResult.model_validate(
        {"atoms": atoms, "dispositions": dispositions}, strict=True
    )


def _extraction_source_tokens(extraction: ExtractionResult) -> int:
    return max(
        1,
        sum(
            max(1, math.ceil(len(atom.source_excerpt) / 1.5))
            for atom in extraction.atoms
        ),
    )


def _merge_syntheses(values: list[SynthesisResult]) -> SynthesisResult:
    if not values:
        raise ValueError("at least one synthesis result is required")
    section_fields = (
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "promotions",
    )
    merged: dict[str, Any] = {
        "title": values[0].title,
        "overview": "；".join(_dedupe_text(item.overview for item in values)),
        "structure": _dedupe_text(
            item for value in values for item in value.structure
        ),
        "concepts": _dedupe_text(
            item for value in values for item in value.concepts
        ),
        "quotes": _dedupe_text(
            item for value in values for item in value.quotes
        ),
        "implications": _dedupe_text(
            item for value in values for item in value.implications
        ),
        "actions": _dedupe_text(
            item for value in values for item in value.actions
        ),
        "tags": _dedupe_text(item for value in values for item in value.tags),
        "included_atom_ids": _dedupe_text(
            atom_id for value in values for atom_id in value.included_atom_ids
        ),
    }
    for field_name in section_fields:
        merged[field_name] = [
            section.model_dump(mode="json")
            for value in values
            for section in getattr(value, field_name)
        ]
    timeline: dict[int, list[dict[str, Any]]] = {}
    for value in values:
        for entry in value.timeline:
            timeline.setdefault(entry.segment_index, []).append(
                entry.model_dump(mode="json")
            )
    merged["timeline"] = [
        _merge_timeline_entries(entries) for _, entries in sorted(timeline.items())
    ]
    return SynthesisResult.model_validate(merged, strict=True)


def _merge_timeline_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    first = entries[0]
    atom_ids = _dedupe_text(
        atom_id for entry in entries for atom_id in entry["atom_ids"]
    )
    dispositions = [entry["disposition"] for entry in entries]
    if atom_ids:
        disposition = "included" if "included" in dispositions else "promotion"
    else:
        disposition = dispositions[0]
    return {
        "segment_index": first["segment_index"],
        "start": first["start"],
        "end": first["end"],
        "disposition": disposition,
        "atom_ids": atom_ids,
        "content": "；".join(
            _dedupe_text(entry["content"] for entry in entries)
        ),
    }


def _slice_synthesis(
    synthesis: SynthesisResult,
    segment_indexes: set[int],
    atom_ids: set[str],
) -> SynthesisResult:
    def sections(field_name: str) -> list[dict[str, Any]]:
        return [
            section.model_dump(mode="json")
            for section in getattr(synthesis, field_name)
            if set(section.atom_ids) & atom_ids
        ]

    timeline = [
        entry.model_dump(mode="json")
        for entry in synthesis.timeline
        if entry.segment_index in segment_indexes
    ]
    return SynthesisResult.model_validate(
        {
            "title": synthesis.title,
            "overview": "仅包含本批源分段的局部草稿，用于分层审查。",
            "structure": [entry["heading"] for entry in sections("themes")],
            "themes": sections("themes"),
            "methods_and_steps": sections("methods_and_steps"),
            "examples_and_counterexamples": sections(
                "examples_and_counterexamples"
            ),
            "boundaries_and_risks": sections("boundaries_and_risks"),
            "concepts": [],
            "quotes": [],
            "implications": [],
            "actions": [],
            "timeline": timeline,
            "promotions": sections("promotions"),
            "tags": [],
            "included_atom_ids": [
                atom_id
                for atom_id in synthesis.included_atom_ids
                if atom_id in atom_ids
            ],
        },
        strict=True,
    )


def _draft_claims(synthesis: SynthesisResult) -> list[DraftClaim]:
    claims: list[DraftClaim] = []

    def add(field: str, text: str, atom_ids: list[str]) -> None:
        claims.append(
            DraftClaim.model_validate(
                {
                    "id": f"C-{field}-{len(claims):04d}",
                    "field": field,
                    "text": text,
                    "atom_ids": atom_ids,
                },
                strict=True,
            )
        )

    add("title", synthesis.title, [])
    add("overview", synthesis.overview, [])
    for item in synthesis.structure:
        add("structure", item, [])
    for field_name in (
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "promotions",
    ):
        for section in getattr(synthesis, field_name):
            add(
                field_name,
                f"{section.heading}: {section.content}",
                list(section.atom_ids),
            )
    for field_name in (
        "concepts",
        "quotes",
        "implications",
        "actions",
        "tags",
    ):
        for item in getattr(synthesis, field_name):
            add(field_name, item, [])
    for entry in synthesis.timeline:
        add("timeline", entry.content, list(entry.atom_ids))
    return claims


def _claim_batches(
    claims: list[DraftClaim],
    context_tokens: int,
) -> list[tuple[list[DraftClaim], int]]:
    token_limit = max(1, math.floor(context_tokens * 0.25))
    result: list[tuple[list[DraftClaim], int]] = []
    current: list[DraftClaim] = []
    current_tokens = 0
    for claim in claims:
        claim_tokens = max(
            1,
            math.ceil(
                (len(claim.text) + sum(len(atom_id) for atom_id in claim.atom_ids))
                / 1.5
            ),
        )
        if current and current_tokens + claim_tokens > token_limit:
            result.append((current, current_tokens))
            current = []
            current_tokens = 0
        current.append(claim)
        current_tokens += claim_tokens
    if current:
        result.append((current, current_tokens))
    return result


def _merge_reviews(
    values: list[CoverageReview],
    *,
    fragment_segment_indexes: set[int] | None = None,
) -> CoverageReview:
    omission_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    correction_by_segment: dict[int, dict[str, Any]] = {}
    reviewed_indexes_by_segment: dict[int, list[int]] = {}
    internally_duplicated: set[int] = set()
    for value in values:
        counts = Counter(value.reviewed_segment_indexes)
        internally_duplicated.update(
            index for index, count in counts.items() if count > 1
        )
        for index in value.reviewed_segment_indexes:
            reviewed_indexes_by_segment.setdefault(index, []).append(index)
        for item in value.source_omission_candidates:
            omission_by_key[
                (item.segment_index, item.source_excerpt, item.missing_statement)
            ] = item.model_dump(mode="json")
        for item in value.disposition_corrections:
            correction_by_segment[item.segment_index] = item.model_dump(mode="json")
    return CoverageReview.model_validate(
        {
            "reviewed_segment_indexes": [
                index
                for index in sorted(reviewed_indexes_by_segment)
                for _ in range(
                    1
                    if (
                        fragment_segment_indexes is not None
                        and index in fragment_segment_indexes
                        and index not in internally_duplicated
                    )
                    else len(reviewed_indexes_by_segment[index])
                )
            ],
            "source_omission_candidates": list(omission_by_key.values()),
            "unsupported_draft_claims": _dedupe_text(
                claim
                for value in values
                for claim in value.unsupported_draft_claims
            ),
            "disposition_corrections": list(correction_by_segment.values()),
        },
        strict=True,
    )


def _fragment_segment_indexes(batches: list[SegmentBatch]) -> set[int]:
    counts = Counter(
        segment.index for batch in batches for segment in batch.segments
    )
    return {index for index, count in counts.items() if count > 1}


def _slice_review(
    review: CoverageReview,
    segment_indexes: set[int],
) -> CoverageReview:
    return CoverageReview.model_validate(
        {
            "reviewed_segment_indexes": [
                index
                for index in review.reviewed_segment_indexes
                if index in segment_indexes
            ],
            "source_omission_candidates": [
                item.model_dump(mode="json")
                for item in review.source_omission_candidates
                if item.segment_index in segment_indexes
            ],
            "unsupported_draft_claims": list(
                review.unsupported_draft_claims
            ),
            "disposition_corrections": [
                item.model_dump(mode="json")
                for item in review.disposition_corrections
                if item.segment_index in segment_indexes
            ],
        },
        strict=True,
    )


def _repair_slice(
    extraction: ExtractionResult,
    synthesis: SynthesisResult,
    review: CoverageReview,
    coverage: CoverageReport,
) -> tuple[ExtractionResult, set[int]]:
    atom_by_id = {atom.id: atom for atom in extraction.atoms}
    thematic_ids = {
        atom_id
        for field_name in (
            "themes",
            "methods_and_steps",
            "examples_and_counterexamples",
            "boundaries_and_risks",
            "promotions",
        )
        for section in getattr(synthesis, field_name)
        for atom_id in section.atom_ids
    }
    relevant_ids = set(coverage.missing_atom_ids)
    relevant_ids.update(set(synthesis.included_atom_ids) - thematic_ids)
    relevant_segments = {
        item.segment_index for item in review.source_omission_candidates
    }
    relevant_segments.update(coverage.missing_segment_indexes)
    timeline_indexes = {entry.segment_index for entry in synthesis.timeline}
    relevant_segments.update(
        item.segment_index
        for item in extraction.dispositions
        if item.segment_index not in timeline_indexes
    )
    if not coverage.promotional_content_preserved:
        relevant_ids.update(
            atom.id
            for atom in extraction.atoms
            if atom.kind == "promotion" or atom.is_promotional
        )
    if review.unsupported_draft_claims:
        relevant_ids.update(atom_by_id)
    relevant_ids.update(
        atom.id
        for atom in extraction.atoms
        if atom.source_segment in relevant_segments
    )
    related_ids: set[str] = set()
    for atom_id in tuple(relevant_ids):
        atom = atom_by_id.get(atom_id)
        if atom is not None:
            related_ids.update(
                relation_id
                for relation_id in atom.relations
                if relation_id in atom_by_id
            )
    relevant_ids.update(related_ids)
    if not relevant_ids and extraction.atoms:
        relevant_ids.update(atom_by_id)
    relevant_segments.update(
        atom_by_id[atom_id].source_segment
        for atom_id in relevant_ids
        if atom_id in atom_by_id
    )
    return (
        _subset_extraction(extraction, relevant_ids, relevant_segments),
        relevant_segments,
    )


def _merge_repair_extraction_patch(
    extraction: ExtractionResult,
    patch: ExtractionResult,
    relevant_segments: set[int],
) -> ExtractionResult:
    if any(atom.source_segment not in relevant_segments for atom in patch.atoms):
        raise ValueError("repair atom is outside the authorized source segments")
    if any(
        item.segment_index not in relevant_segments
        for item in patch.dispositions
    ):
        raise ValueError("repair disposition is outside the authorized source segments")

    atoms = {
        atom.id: atom.model_dump(mode="json") for atom in extraction.atoms
    }
    for atom in patch.atoms:
        existing = atoms.get(atom.id)
        if (
            existing is not None
            and existing["source_segment"] not in relevant_segments
        ):
            raise ValueError("repair atom ID collides outside the authorized segments")
        atoms[atom.id] = atom.model_dump(mode="json")

    allowed_ids = set(atoms)
    for atom in patch.atoms:
        if any(relation_id not in allowed_ids for relation_id in atom.relations):
            raise ValueError("repair atom references an unknown atom")

    dispositions = {
        item.segment_index: item.model_dump(mode="json")
        for item in extraction.dispositions
    }
    for item in patch.dispositions:
        if any(atom_id not in allowed_ids for atom_id in item.atom_ids):
            raise ValueError("repair disposition references an unknown atom")
        if any(
            atoms[atom_id]["source_segment"] != item.segment_index
            for atom_id in item.atom_ids
        ):
            raise ValueError("repair disposition references another segment")
        dispositions[item.segment_index] = item.model_dump(mode="json")

    return ExtractionResult.model_validate(
        {
            "atoms": list(atoms.values()),
            "dispositions": [
                dispositions[index] for index in sorted(dispositions)
            ],
        },
        strict=True,
    )


def _merge_repair_patch(
    synthesis: SynthesisResult,
    patch: SynthesisResult,
    relevant_segments: set[int],
    *,
    extraction: ExtractionResult | None = None,
) -> SynthesisResult:
    patch_timeline_indexes = [entry.segment_index for entry in patch.timeline]
    if any(index not in relevant_segments for index in patch_timeline_indexes):
        raise ValueError("repair timeline is outside the authorized source segments")
    if len(patch_timeline_indexes) != len(set(patch_timeline_indexes)):
        raise ValueError("repair timeline contains duplicate source segments")

    if extraction is not None:
        allowed_atom_ids = {atom.id for atom in extraction.atoms}
        patch_atom_ids = {
            atom_id
            for field_name in (
                "themes",
                "methods_and_steps",
                "examples_and_counterexamples",
                "boundaries_and_risks",
                "promotions",
            )
            for section in getattr(patch, field_name)
            for atom_id in section.atom_ids
        }
        patch_atom_ids.update(
            atom_id for entry in patch.timeline for atom_id in entry.atom_ids
        )
        patch_atom_ids.update(patch.included_atom_ids)
        if not patch_atom_ids <= allowed_atom_ids:
            raise ValueError("repair synthesis references an unknown atom")
        atom_by_id = {atom.id: atom for atom in extraction.atoms}
        for entry in patch.timeline:
            if any(
                atom_by_id[atom_id].source_segment != entry.segment_index
                for atom_id in entry.atom_ids
            ):
                raise ValueError("repair timeline references another segment")
        for field_name in (
            "themes",
            "methods_and_steps",
            "examples_and_counterexamples",
            "boundaries_and_risks",
            "promotions",
        ):
            for section in getattr(patch, field_name):
                if any(atom_id not in atom_by_id for atom_id in section.atom_ids):
                    raise ValueError("repair section references an unknown atom")

    relevant_ids = {
        atom_id
        for entry in synthesis.timeline
        if entry.segment_index in relevant_segments
        for atom_id in entry.atom_ids
    }
    section_fields = (
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "promotions",
    )
    data = synthesis.model_dump(mode="json")
    for field_name in section_fields:
        data[field_name] = [
            section.model_dump(mode="json")
            for section in getattr(synthesis, field_name)
            if not (set(section.atom_ids) & relevant_ids)
        ] + [
            section.model_dump(mode="json")
            for section in getattr(patch, field_name)
        ]
    data["timeline"] = [
        entry.model_dump(mode="json")
        for entry in synthesis.timeline
        if entry.segment_index not in relevant_segments
    ] + [entry.model_dump(mode="json") for entry in patch.timeline]
    data["timeline"].sort(key=lambda entry: entry["segment_index"])
    data["included_atom_ids"] = _dedupe_text(
        [
            atom_id
            for atom_id in synthesis.included_atom_ids
            if atom_id not in relevant_ids
        ]
        + list(patch.included_atom_ids)
    )
    for field_name in (
        "structure",
        "concepts",
        "quotes",
        "implications",
        "actions",
        "tags",
    ):
        data[field_name] = _dedupe_text(
            [*getattr(synthesis, field_name), *getattr(patch, field_name)]
        )
    return SynthesisResult.model_validate(data, strict=True)


def _dedupe_text(values: Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _merge_extractions(
    values: list[tuple[SegmentBatch, ExtractionResult]],
) -> ExtractionResult:
    merged_atoms: list[dict[str, Any]] = []
    merged_dispositions: list[dict[str, Any]] = []
    disposition_batch_indexes: list[int] = []
    prepared: list[
        tuple[SegmentBatch, ExtractionResult, dict[str, str], bool]
    ] = []
    id_occurrences: dict[str, list[str]] = {}
    for batch, extraction in values:
        global_indexes = {segment.index for segment in batch.segments}
        returned_indexes = [
            *(atom.source_segment for atom in extraction.atoms),
            *(item.segment_index for item in extraction.dispositions),
        ]
        uses_local_indexes = bool(returned_indexes) and all(
            0 <= value < len(batch.segments) for value in returned_indexes
        ) and any(value not in global_indexes for value in returned_indexes)
        local_ids = [atom.id for atom in extraction.atoms]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("extraction batch contains duplicate atom IDs")
        id_map = {
            atom.id: f"B{batch.batch_index:03d}-A{index:03d}"
            for index, atom in enumerate(extraction.atoms, 1)
        }
        for old_id, new_id in id_map.items():
            id_occurrences.setdefault(old_id, []).append(new_id)
        prepared.append((batch, extraction, id_map, uses_local_indexes))
    unique_global_ids = {
        old_id: new_ids[0]
        for old_id, new_ids in id_occurrences.items()
        if len(new_ids) == 1
    }
    for batch, extraction, id_map, uses_local_indexes in prepared:
        for atom in extraction.atoms:
            data = atom.model_dump(mode="json")
            data["id"] = id_map[atom.id]
            data["source_segment"] = _global_segment_index(
                atom.source_segment, batch, uses_local_indexes
            )
            data["relations"] = [
                id_map.get(value, unique_global_ids.get(value, value))
                for value in atom.relations
            ]
            merged_atoms.append(data)
        for disposition in extraction.dispositions:
            data = disposition.model_dump(mode="json")
            data["segment_index"] = _global_segment_index(
                disposition.segment_index, batch, uses_local_indexes
            )
            data["atom_ids"] = [id_map.get(value, value) for value in disposition.atom_ids]
            merged_dispositions.append(data)
            disposition_batch_indexes.append(batch.batch_index)
    sequence_by_topic: dict[str, int] = {}
    for atom in sorted(
        merged_atoms,
        key=lambda value: (
            value["source_segment"],
            value["start"] if value["start"] is not None else float("inf"),
            value["id"],
        ),
    ):
        if atom["kind"] != "step":
            continue
        topic = atom["topic"]
        sequence_by_topic[topic] = sequence_by_topic.get(topic, 0) + 1
        atom["sequence"] = sequence_by_topic[topic]
    merged_dispositions = _coalesce_dispositions(
        merged_dispositions,
        batch_indexes=disposition_batch_indexes,
    )
    return ExtractionResult.model_validate(
        {"atoms": merged_atoms, "dispositions": merged_dispositions}, strict=True
    )


def _coalesce_dispositions(
    values: list[dict[str, Any]],
    *,
    batch_indexes: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Merge only duplicate source fragments from different provider batches.

    A provider response is allowed to contain duplicate dispositions.  Those
    duplicates are evidence of the response itself and must remain visible so
    coverage validation can reject them.  When a source segment was split into
    multiple batches, however, the resulting dispositions are fragments of the
    same source and should be merged across batch boundaries only.
    """
    if batch_indexes is not None and len(batch_indexes) != len(values):
        raise ValueError("batch_indexes must align with dispositions")
    grouped: dict[int, list[tuple[int | None, dict[str, Any]]]] = {}
    for position, value in enumerate(values):
        batch_index = batch_indexes[position] if batch_indexes is not None else None
        grouped.setdefault(value["segment_index"], []).append((batch_index, value))

    result: list[dict[str, Any]] = []
    for segment_index, entries in sorted(grouped.items()):
        if batch_indexes is None:
            result.append(_combine_dispositions(segment_index, [item for _, item in entries]))
            continue

        # Keep duplicate dispositions emitted by every individual provider
        # response, while folding one representative from each distinct batch
        # into a single disposition for the source segment.
        by_batch: dict[int | None, list[dict[str, Any]]] = {}
        for batch_index, item in entries:
            by_batch.setdefault(batch_index, []).append(item)
        representatives = [items[0] for items in by_batch.values()]
        result.append(_combine_dispositions(segment_index, representatives))
        for items in by_batch.values():
            result.extend(items[1:])
    return result


def _combine_dispositions(
    segment_index: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    atom_ids = [atom_id for item in items for atom_id in item["atom_ids"]]
    kinds = [item["disposition"] for item in items]
    if atom_ids:
        disposition = "included" if "included" in kinds else "promotion"
    elif all(kind == "unintelligible_asr" for kind in kinds):
        disposition = "unintelligible_asr"
    else:
        disposition = kinds[0]
    reasons = list(
        dict.fromkeys(item["reason"] for item in items if item["reason"])
    )
    return {
        "segment_index": segment_index,
        "disposition": disposition,
        "atom_ids": atom_ids,
        "reason": "；".join(reasons) if reasons else None,
    }


def _global_segment_index(
    value: int,
    batch: SegmentBatch,
    uses_local_indexes: bool,
) -> int:
    if uses_local_indexes and 0 <= value < len(batch.segments):
        return batch.segments[value].index
    return value


def _aggregate_usage(records: list[StageRecord]) -> ProviderUsage:
    input_values = [record.usage.input_tokens for record in records]
    output_values = [record.usage.output_tokens for record in records]
    cost_values = [record.usage.api_cost for record in records]
    statuses = {record.usage.cost_status for record in records}
    if statuses == {"zero_local"}:
        cost_status = "zero_local"
    elif "unknown" in statuses or any(value is None for value in cost_values):
        cost_status = "unknown"
    elif "estimated_from_config" in statuses:
        cost_status = "estimated_from_config"
    else:
        cost_status = "reported"
    return ProviderUsage(
        input_tokens=_sum_optional(input_values),
        output_tokens=_sum_optional(output_values),
        api_cost=_sum_optional(cost_values),
        cost_status=cost_status,
    )


def _sum_optional(values: list[int | float | None]) -> int | float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


async def _emit_audit(audit_sink: AuditSink | None, record: StageRecord) -> None:
    if audit_sink is None:
        return
    result = audit_sink.record(record)
    if inspect.isawaitable(result):
        await result


async def _record_context_failure(
    failure: dict[str, Any],
    audit_sink: AuditSink | None,
    records: list[StageRecord],
    *,
    batch_boundary: Mapping[str, Any] | None = None,
) -> None:
    stage = failure["stage"]
    record = StageRecord(
        stage=stage,
        prompt_version=PROMPT_VERSIONS[stage],
        system_prompt=SYSTEM_PROMPTS[stage],
        payload={"context_failure": copy.deepcopy(failure)},
        raw_output="",
        parsed_json={},
        usage=ProviderUsage(
            input_tokens=None,
            output_tokens=None,
            api_cost=None,
            cost_status="unknown",
        ),
        diagnostics=copy.deepcopy(failure),
        elapsed_ms=0.0,
        batch_boundary=(
            copy.deepcopy(dict(batch_boundary))
            if batch_boundary is not None
            else copy.deepcopy(failure.get("batch_boundary"))
        ),
    )
    records.append(record)
    await _emit_audit(audit_sink, record)


def _check_cancellation(cancel_event: asyncio.Event) -> None:
    if cancel_event.is_set():
        raise JobCancelledError("cancelled_by_user")


__all__ = [
    "AuditSink",
    "ProviderContextBudgetError",
    "SegmentBatch",
    "SemanticGeneration",
    "SemanticNotePipeline",
    "StageRecord",
    "build_segment_batches",
]
