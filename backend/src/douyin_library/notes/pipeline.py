from __future__ import annotations

import asyncio
import copy
import inspect
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.contracts import (
    CoverageReport,
    CoverageReview,
    ExtractionResult,
    ProviderUsage,
    SynthesisResult,
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


class AuditSink(Protocol):
    def record(self, record: StageRecord) -> Any: ...


class ProviderContextBudgetError(RuntimeError):
    pass


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

    async def generate(
        self,
        note_input: NoteInput,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None = None,
    ) -> SemanticGeneration:
        records: list[StageRecord] = []
        context_failures: list[dict[str, Any]] = []
        batches = build_segment_batches(
            note_input,
            self.provider.context_tokens,
            input_fraction=0.35,
        )
        extractions: list[tuple[SegmentBatch, ExtractionResult]] = []
        for batch in batches:
            all_prior_atoms = (
                _merge_extractions(extractions).atoms if extractions else []
            )
            prior_atoms, prior_tokens = _bounded_prior_context(
                all_prior_atoms,
                batch.estimated_tokens,
                self.provider.context_tokens,
            )
            payload = {
                "prompt_version": PROMPT_VERSIONS["extract"],
                "aweme_id": note_input.aweme_id,
                "batch": batch.boundary(),
                "segments": [_segment_payload(segment) for segment in batch.segments],
                "prior_evidence_atoms": [
                    atom.model_dump(mode="json") for atom in prior_atoms
                ],
                "context_budget": _context_budget(
                    batch.estimated_tokens + prior_tokens,
                    self.provider.context_tokens,
                ),
                "input_truncated": False,
            }
            result = await self._call(
                stage="extract",
                payload=payload,
                schema=ExtractionResult,
                cancel_event=cancel_event,
                audit_sink=audit_sink,
                records=records,
                batch_boundary=batch.boundary(),
            )
            extractions.append((batch, result))
        extraction = _merge_extractions(extractions)
        extraction_slices = _build_extraction_slices(
            extraction, self.provider.context_tokens
        )
        synthesis_parts: list[SynthesisResult] = []
        for slice_index, (extraction_slice, estimated_tokens) in enumerate(
            extraction_slices
        ):
            synthesis_parts.append(
                await self._call(
                    stage="synthesize",
                    payload={
                        "prompt_version": PROMPT_VERSIONS["synthesize"],
                        "aweme_id": note_input.aweme_id,
                        "metadata": _metadata_payload(note_input),
                        "slice": {
                            "index": slice_index,
                            "count": len(extraction_slices),
                        },
                        "extraction": extraction_slice.model_dump(mode="json"),
                        "context_budget": _context_budget(
                            estimated_tokens, self.provider.context_tokens
                        ),
                        "input_truncated": False,
                    },
                    schema=SynthesisResult,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                )
            )
        synthesis = _merge_syntheses(synthesis_parts)
        review = await self._review(
            note_input, synthesis, 1, cancel_event, audit_sink, records
        )
        coverage = self._validate(note_input, extraction, synthesis, review)
        repair_count = 0
        if coverage.quality_gate == "failed" and coverage.repairable:
            repair_extraction, relevant_segments = _repair_slice(
                extraction, synthesis, review, coverage
            )
            repair_budget = _context_budget(
                _extraction_source_tokens(repair_extraction),
                self.provider.context_tokens,
            )
            if repair_budget["within_budget"]:
                repair_patch = await self._call(
                    stage="repair",
                    payload={
                        "prompt_version": PROMPT_VERSIONS["repair"],
                        "aweme_id": note_input.aweme_id,
                        "extraction": repair_extraction.model_dump(mode="json"),
                        "draft": _slice_synthesis(
                            synthesis,
                            relevant_segments,
                            {atom.id for atom in repair_extraction.atoms},
                        ).model_dump(mode="json"),
                        "review": _slice_review(
                            review, relevant_segments
                        ).model_dump(mode="json"),
                        "coverage_report": coverage.model_dump(
                            mode="json", by_alias=True
                        ),
                        "context_budget": repair_budget,
                        "input_truncated": False,
                    },
                    schema=SynthesisResult,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                )
                synthesis = _merge_repair_patch(
                    synthesis, repair_patch, relevant_segments
                )
                repair_count = 1
                review = await self._review(
                    note_input, synthesis, 2, cancel_event, audit_sink, records
                )
                coverage = self._validate(
                    note_input, extraction, synthesis, review
                )
            else:
                failure = {
                    "stage": "repair",
                    "code": "provider_context_budget_exceeded",
                    **repair_budget,
                }
                context_failures.append(failure)
                await _record_context_failure(failure, audit_sink, records)
                coverage = coverage.model_copy(
                    update={"quality_gate": "failed", "repairable": False}
                )
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
            context_status=(
                "within_budget" if not context_failures else "failed"
            ),
            context_failures=tuple(context_failures),
        )

    async def _review(
        self,
        note_input: NoteInput,
        synthesis: SynthesisResult,
        review_pass: int,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None,
        records: list[StageRecord],
    ) -> CoverageReview:
        reviews: list[CoverageReview] = []
        for batch in build_segment_batches(
            note_input, self.provider.context_tokens
        ):
            segment_indexes = {segment.index for segment in batch.segments}
            atom_ids = {
                atom_id
                for entry in synthesis.timeline
                if entry.segment_index in segment_indexes
                for atom_id in entry.atom_ids
            }
            reviews.append(
                await self._call(
                    stage="review",
                    payload={
                        "prompt_version": PROMPT_VERSIONS["review"],
                        "aweme_id": note_input.aweme_id,
                        "review_pass": review_pass,
                        "source_segments": [
                            _segment_payload(segment)
                            for segment in batch.segments
                        ],
                        "draft": _slice_synthesis(
                            synthesis, segment_indexes, atom_ids
                        ).model_dump(mode="json"),
                        "context_budget": _context_budget(
                            batch.estimated_tokens,
                            self.provider.context_tokens,
                        ),
                        "input_truncated": False,
                    },
                    schema=CoverageReview,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                    batch_boundary=batch.boundary(),
                )
            )
        return _merge_reviews(reviews)

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
        budget = payload.get("context_budget")
        if isinstance(budget, Mapping) and not budget.get("within_budget", False):
            failure = {
                "stage": stage,
                "code": "provider_context_budget_exceeded",
                **dict(budget),
            }
            await _record_context_failure(failure, audit_sink, records)
            raise ProviderContextBudgetError(
                f"{stage}:provider_context_budget_exceeded"
            )
        started = time.perf_counter()
        try:
            response = await self.provider.complete_json(
                stage=stage,
                system_prompt=SYSTEM_PROMPTS[stage],
                payload=payload,
                json_schema=schema.model_json_schema(),
                cancel_event=cancel_event,
            )
        except Exception as exc:
            record = StageRecord(
                stage=stage,
                prompt_version=PROMPT_VERSIONS[stage],
                system_prompt=SYSTEM_PROMPTS[stage],
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
        try:
            parsed = schema.model_validate(response.parsed_json, strict=True)
        except ValueError as exc:
            diagnostics["schema_validation_error"] = str(exc)
            record = StageRecord(
                stage=stage,
                prompt_version=PROMPT_VERSIONS[stage],
                system_prompt=SYSTEM_PROMPTS[stage],
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
            raise
        record = StageRecord(
            stage=stage,
            prompt_version=PROMPT_VERSIONS[stage],
            system_prompt=SYSTEM_PROMPTS[stage],
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


def _segment_payload(segment: InputSegment) -> dict[str, Any]:
    return {
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
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


def _merge_reviews(values: list[CoverageReview]) -> CoverageReview:
    omission_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    correction_by_segment: dict[int, dict[str, Any]] = {}
    for value in values:
        for item in value.source_omission_candidates:
            omission_by_key[
                (item.segment_index, item.source_excerpt, item.missing_statement)
            ] = item.model_dump(mode="json")
        for item in value.disposition_corrections:
            correction_by_segment[item.segment_index] = item.model_dump(mode="json")
    return CoverageReview.model_validate(
        {
            "reviewed_segment_indexes": sorted(
                {
                    index
                    for value in values
                    for index in value.reviewed_segment_indexes
                }
            ),
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


def _merge_repair_patch(
    synthesis: SynthesisResult,
    patch: SynthesisResult,
    relevant_segments: set[int],
) -> SynthesisResult:
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
    merged_dispositions = _coalesce_dispositions(merged_dispositions)
    return ExtractionResult.model_validate(
        {"atoms": merged_atoms, "dispositions": merged_dispositions}, strict=True
    )


def _coalesce_dispositions(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for value in values:
        grouped.setdefault(value["segment_index"], []).append(value)
    result: list[dict[str, Any]] = []
    for segment_index, items in sorted(grouped.items()):
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
        result.append(
            {
                "segment_index": segment_index,
                "disposition": disposition,
                "atom_ids": atom_ids,
                "reason": "；".join(reasons) if reasons else None,
            }
        )
    return result


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
        batch_boundary=None,
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
