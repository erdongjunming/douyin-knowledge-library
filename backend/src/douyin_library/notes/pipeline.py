from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import re
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
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
    SectionClaim,
    SectionDraft,
    SynthesisResult,
    TeachingOutline,
    TeachingOutlineSection,
    ThemeSection,
    TimelineEntry,
)
from douyin_library.notes.coverage import CoverageValidator
from douyin_library.notes.depth import DepthValidator
from douyin_library.notes.input import InputSegment, NoteInput
from douyin_library.notes.prompts import (
    EXTRACT_RETRY_PROMPT_VERSION,
    EXTRACT_RETRY_SYSTEM_PROMPT,
    PROMPT_VERSIONS,
    REVIEW_RETRY_PROMPT_VERSION,
    REVIEW_RETRY_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    TEACHING_REPAIR_PROMPT_VERSION,
    TEACHING_REPAIR_SYSTEM_PROMPT,
)
from douyin_library.notes.providers.base import NoteProvider, NoteProviderError, NoteStage


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


@dataclass(frozen=True, slots=True)
class RepairUnit:
    """One inseparable teaching-repair request.

    A unit owns whole teaching sections. Sections that share evidence, source
    segments, or an atom relation stay together so one patch cannot overwrite
    the context required by another patch.
    """

    section_ids: tuple[str, ...]
    extraction: ExtractionResult
    relevant_segments: frozenset[int]
    section_drafts: tuple[SectionDraft, ...]
    outline: TeachingOutline
    depth_review: DepthReview | None


class RepairUnitFailure(RuntimeError):
    """A repair unit failed before the round could safely be committed."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = copy.deepcopy(dict(failure))
        super().__init__(str(self.failure.get("code", "repair_unit_failed")))


@cache
def _cached_model_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Build each static provider schema once per process."""

    return schema.model_json_schema()


def _model_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return an isolated cached schema for provider and budget calls."""

    return copy.deepcopy(_cached_model_json_schema(schema))


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
    source_segments = _positioned_source_segments(note_input) or note_input.segments
    expanded_segments = [
        InputSegment(
            index=segment.index,
            start=segment.start,
            end=segment.end,
            text=segment.text[offset : offset + char_limit],
            segment_id=segment.segment_id,
            source_offset=segment.source_offset + offset,
        )
        for segment in source_segments
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


def _positioned_source_segments(note_input: NoteInput) -> list[InputSegment] | None:
    """Return raw segments with zero-based offsets proven against the transcript.

    ASR segment files often omit only the whitespace separating neighboring
    chunks.  A position is usable for provenance only when every segment can
    be located in order and every omitted span is whitespace.  Callers that do
    not require positional provenance may retain the original segments when
    this proof is unavailable; fallback publication will fail closed later.
    """

    cursor = 0
    positioned: list[InputSegment] = []
    for segment in note_input.segments:
        source_start = note_input.transcript.find(segment.text, cursor)
        if source_start < 0 or note_input.transcript[cursor:source_start].strip():
            return None
        positioned.append(
            InputSegment(
                index=segment.index,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                segment_id=segment.segment_id,
                source_offset=source_start,
            )
        )
        cursor = source_start + len(segment.text)
    if note_input.transcript[cursor:].strip():
        return None
    return positioned


def _fallback_source_review_batches(
    note_input: NoteInput,
    extraction: ExtractionResult,
    context_tokens: int,
    *,
    input_fraction: float,
) -> list[SegmentBatch] | None:
    """Batch canonical fallback evidence without cutting or rewriting source."""

    if not _is_source_preserving_fallback(extraction):
        return None
    source_material = _fallback_source_material(note_input)
    if source_material is None:
        return []
    positioned_atoms = [atom for atom in extraction.atoms if atom.source_start is not None]
    if len(positioned_atoms) != len(extraction.atoms):
        return []
    token_limit = max(1, math.floor(context_tokens * input_fraction))
    segments: list[InputSegment] = []
    assigned_atom_count = 0
    for source_segment, source, fragment_offset in source_material:
        source_start = source_segment.source_offset
        source_end = source_start + len(source_segment.text)
        source_atoms = sorted(
            (
                atom
                for atom in positioned_atoms
                if atom.source_segment == source_segment.index
                and source_start <= (atom.source_start or 0)
                and (atom.source_start or 0) + len(atom.source_excerpt) <= source_end
            ),
            key=lambda atom: (atom.source_start or 0, atom.id),
        )
        if not source_atoms:
            return []
        cursor = 0
        segment_fragments: list[InputSegment] = []
        for atom in source_atoms:
            atom_start = (atom.source_start or 0) - fragment_offset
            atom_end = atom_start + len(atom.source_excerpt)
            if atom_start < cursor or source[atom_start:atom_end] != atom.source_excerpt:
                return []
            # Begin each fragment immediately after the prior atom.  This keeps
            # inter-atom whitespace with the following atom and never removes
            # it from the review source.
            segment_fragments.append(
                InputSegment(
                    index=source_segment.index,
                    start=source_segment.start,
                    end=source_segment.end,
                    text=source[cursor:atom_end],
                    segment_id=source_segment.segment_id,
                    source_offset=fragment_offset + cursor,
                )
            )
            cursor = atom_end
        assigned_atom_count += len(source_atoms)
        if segment_fragments:
            # Preserve trailing whitespace (or other source text) in the final
            # atom-safe fragment instead of reconstructing input from excerpts.
            final_fragment = segment_fragments[-1]
            segment_fragments[-1] = InputSegment(
                index=final_fragment.index,
                start=final_fragment.start,
                end=final_fragment.end,
                text=final_fragment.text + source[cursor:],
                segment_id=final_fragment.segment_id,
                source_offset=final_fragment.source_offset,
            )
        segments.extend(segment_fragments)
    if assigned_atom_count != len(positioned_atoms):
        return []
    batches: list[SegmentBatch] = []
    current: list[InputSegment] = []
    current_chars = 0
    current_tokens = 0
    for segment in segments:
        segment_tokens = max(1, math.ceil(len(segment.text) / 1.5))
        if current and current_tokens + segment_tokens > token_limit:
            batches.append(_batch(len(batches), current, current_chars, current_tokens))
            current = []
            current_chars = 0
            current_tokens = 0
        current.append(segment)
        current_chars += len(segment.text)
        current_tokens += segment_tokens
    if current:
        batches.append(_batch(len(batches), current, current_chars, current_tokens))
    return batches


def _fallback_source_material(
    note_input: NoteInput,
) -> list[tuple[InputSegment, str, int]] | None:
    """Map every canonical segment to its exact raw-transcript material.

    ASR segment files may omit whitespace between adjacent chunks.  A fallback
    source review must retain those raw characters while every coordinate stays
    a zero-based absolute index into the original transcript.  If the ordered
    mapping cannot be proven without skipping non-whitespace, leave publication
    closed.
    """

    source_segments = _positioned_source_segments(note_input)
    if not source_segments:
        return None
    cursor = 0
    material: list[tuple[InputSegment, str, int]] = []
    for source_segment in source_segments:
        source_start = source_segment.source_offset
        source_end = source_start + len(source_segment.text)
        material.append(
            (
                source_segment,
                note_input.transcript[cursor:source_end],
                cursor,
            )
        )
        cursor = source_end
    if note_input.transcript[cursor:].strip() or not material:
        return None
    last_segment, last_text, last_offset = material[-1]
    material[-1] = (
        last_segment,
        last_text + note_input.transcript[cursor:],
        last_offset,
    )
    return material


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
                json_schema=_model_json_schema(schema),
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
        if _positioned_source_segments(note_input) is None:
            failure = {
                "stage": "extract",
                "code": "source_provenance_unverified",
                "message": ("source segments do not losslessly cover the raw transcript"),
            }
            _remember_context_failure(context_failures, failure)
            await _record_context_failure(failure, audit_sink, records)
            return self._failed_generation(
                note_input=note_input,
                extraction=None,
                synthesis=None,
                review=None,
                context_failures=context_failures,
                records=records,
                batches=[],
            )
        pending_batches = build_segment_batches(
            note_input,
            self.provider.context_tokens,
            input_fraction=0.35,
        )
        batches: list[SegmentBatch] = []
        extractions: list[tuple[SegmentBatch, ExtractionResult]] = []
        source_fallback_mode = False
        while pending_batches:
            batch = _reindex_batch(pending_batches.pop(0), len(batches))
            if source_fallback_mode:
                extractions.append((batch, _canonical_source_extraction(batch)))
                batches.append(batch)
                continue
            all_prior_atoms = _merge_extractions(extractions).atoms if extractions else []
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
            while True:
                payload = _extraction_payload(
                    note_input,
                    batch,
                    prior_catalog,
                    prior_atoms,
                )
                if self._request_fits("extract", payload, ExtractionResult):
                    break
                _check_cancellation(cancel_event)
                if prior_atoms:
                    # Keep the compact catalog available for relation lookup,
                    # but drop the verbose prior atom bodies first.
                    prior_atoms = prior_atoms[1:]
                    continue
                if len(prior_catalog) > 1:
                    prior_catalog = _shrink_atom_catalog(prior_catalog)
                    continue
                break
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
                        json_schema=_model_json_schema(ExtractionResult),
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
                    extraction=_merge_extractions(extractions) if extractions else None,
                    synthesis=None,
                    review=None,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                )
            try:
                result: ExtractionResult | None = None
                validation_error: ValueError | None = None
                extraction_attempts = (
                    2 if getattr(self.provider, "supports_teaching_pipeline", False) else 1
                )
                for extraction_attempt in range(extraction_attempts):
                    request_payload = dict(payload)
                    if validation_error is not None:
                        request_payload["validation_feedback"] = str(validation_error)
                    result = await self._call(
                        stage="extract",
                        payload=request_payload,
                        schema=ExtractionResult,
                        cancel_event=cancel_event,
                        audit_sink=audit_sink,
                        records=records,
                        batch_boundary=batch.boundary(),
                        system_prompt_override=(
                            EXTRACT_RETRY_SYSTEM_PROMPT if extraction_attempt else None
                        ),
                        prompt_version_override=(
                            EXTRACT_RETRY_PROMPT_VERSION if extraction_attempt else None
                        ),
                    )
                    if extraction_attempts == 1:
                        break
                    try:
                        _validate_extraction_response(batch, result)
                        validation_error = None
                        break
                    except ValueError as exc:
                        validation_error = exc
                        if extraction_attempt == extraction_attempts - 1:
                            raise
                if result is None:
                    raise ValueError("extraction response was empty")
            except ValueError:
                if not validation_error or not getattr(
                    self.provider, "supports_teaching_pipeline", False
                ):
                    raise
                # Keep the local semantic pipeline usable when a small model
                # returns schema-shaped but lossy extraction twice. The
                # fallback is built only from exact source clauses; all later
                # synthesis, source review and depth gates still run.
                # Once any batch needs the exact-source fallback, rebuild the
                # earlier batches with the same deterministic representation.
                # Mixing provider atoms with fallback atoms would make the
                # teaching projection and per-batch review inconsistent.
                extractions = [
                    (prior_batch, _canonical_source_extraction(prior_batch))
                    for prior_batch, _ in extractions
                ]
                result = _canonical_source_extraction(batch)
                source_fallback_mode = True
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(exc, "extract", payload)
                _remember_context_failure(context_failures, failure)
                return self._failed_generation(
                    note_input=note_input,
                    extraction=_merge_extractions(extractions) if extractions else None,
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
        # Input sufficiency is a local gate for every provider path, including
        # legacy semantic providers that do not expose teaching sections.
        input_sufficient = _input_is_sufficient(note_input)
        teaching_enabled = bool(getattr(self.provider, "supports_teaching_pipeline", False))
        if teaching_enabled:
            if _is_source_preserving_fallback(extraction):
                outline = _source_fallback_outline(extraction)
                section_drafts = _source_fallback_section_drafts(
                    extraction,
                    outline,
                )
                section_attempts = {section.section_id: 0 for section in outline.sections}
                input_sufficient = _input_is_sufficient(note_input)
                synthesis = _teaching_synthesis(
                    note_input,
                    extraction,
                    outline,
                    section_drafts,
                )
            else:
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
                    # ``unprovided_items`` normally records honest evidence gaps
                    # (for example, a video with no counterexample).  Only the
                    # section-id marker is an orchestration failure emitted by our
                    # bounded section retry path.
                    section_failed = any(
                        draft.section_id in draft.unprovided_items for draft in section_drafts
                    )
                    input_sufficient = outline.input_sufficient and not section_failed
                    synthesis = _teaching_synthesis(
                        note_input,
                        extraction,
                        outline,
                        section_drafts,
                    )
                except AssertionError:
                    # Keep the legacy path available for partially upgraded providers.
                    teaching_enabled = False
        pending_slices = (
            []
            if teaching_enabled
            else [
                item
                for item, _ in _build_extraction_slices(
                    extraction,
                    self.provider.context_tokens,
                )
            ]
        )
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
                        json_schema=_model_json_schema(SynthesisResult),
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
                    synthesis=_merge_syntheses(synthesis_parts) if synthesis_parts else None,
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
                failure = _exception_failure(
                    exc,
                    "synthesize",
                    _synthesis_payload(
                        note_input,
                        extraction_slice,
                        slice_index,
                        len(extraction_slices),
                    ),
                )
                _remember_context_failure(context_failures, failure)
                return self._failed_generation(
                    note_input=note_input,
                    extraction=extraction,
                    synthesis=_merge_syntheses(synthesis_parts) if synthesis_parts else None,
                    review=None,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                )
        if not teaching_enabled:
            synthesis = _merge_syntheses(synthesis_parts)
        depth_review: DepthReview | None = None
        if teaching_enabled:
            try:
                depth_review = await self._review_depth(
                    note_input,
                    extraction,
                    outline,
                    section_drafts,
                    cancel_event,
                    audit_sink,
                    records,
                    input_sufficient=_input_is_sufficient(note_input) and input_sufficient,
                )
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(exc, "depth_review", {})
                _remember_context_failure(context_failures, failure)
                return self._failed_generation(
                    note_input=note_input,
                    extraction=extraction,
                    synthesis=synthesis,
                    review=None,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                )
            # Provider depth output cannot override local evidence sufficiency.
            # Keep the gate fail-closed when transcript/segment metadata is
            # sparse or visual evidence is explicitly required.
            input_sufficient = (
                input_sufficient
                and _input_is_sufficient(note_input)
                and depth_review.input_sufficient
            )
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
        coverage = self._validate(
            note_input,
            extraction,
            synthesis,
            review,
            input_sufficient=input_sufficient,
            depth_review=depth_review,
        )
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
        depth_repairable = bool(
            teaching_enabled
            and depth_review is not None
            and depth_review.repair_section_ids
            and not depth_review.non_repairable_findings
            and not depth_review.unknown_atom_ids
            and not depth_review.unknown_section_ids
        )
        if (
            coverage.quality_gate == "failed"
            and (coverage.repairable or depth_repairable)
            and teaching_enabled
        ):
            # Successful unit patches remain private until the complete repair
            # round succeeds. A historical upgrade must never expose a partial
            # teaching note after a later unit fails.
            original_extraction = extraction
            original_synthesis = synthesis
            original_review = review
            original_coverage = coverage
            try:
                extraction, synthesis, section_drafts = await self._repair_teaching_units(
                    note_input=note_input,
                    extraction=extraction,
                    synthesis=synthesis,
                    review=review,
                    coverage=coverage,
                    outline=outline,
                    section_drafts=section_drafts,
                    depth_review=depth_review,
                    depth_repairable=depth_repairable,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                )
            except RepairUnitFailure as exc:
                _remember_context_failure(context_failures, exc.failure)
                coverage = original_coverage.model_copy(
                    update={"quality_gate": "failed", "repairable": False}
                )
                return self._failed_generation(
                    note_input=note_input,
                    extraction=original_extraction,
                    synthesis=original_synthesis,
                    review=original_review,
                    context_failures=context_failures,
                    records=records,
                    batches=batches,
                    coverage=coverage,
                )
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
            try:
                depth_review = await self._review_depth(
                    note_input,
                    extraction,
                    outline,
                    section_drafts,
                    cancel_event,
                    audit_sink,
                    records,
                    input_sufficient=_input_is_sufficient(note_input) and input_sufficient,
                )
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(exc, "depth_review", {})
                _remember_context_failure(context_failures, failure)
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
            input_sufficient = input_sufficient and depth_review.input_sufficient
            coverage = self._validate(
                note_input,
                extraction,
                synthesis,
                review,
                input_sufficient=input_sufficient,
                depth_review=depth_review,
            )
            if context_failures:
                coverage = coverage.model_copy(
                    update={"quality_gate": "failed", "repairable": False}
                )
        elif coverage.quality_gate == "failed" and (coverage.repairable or depth_repairable):
            _check_cancellation(cancel_event)
            if depth_repairable and depth_review is not None:
                repair_extraction, relevant_segments = _depth_repair_slice(
                    extraction,
                    section_drafts,
                    depth_review,
                    outline=outline,
                )
            else:
                repair_extraction, relevant_segments = _repair_slice(
                    extraction, synthesis, review, coverage
                )
            teaching_repair_section_ids = (
                list(depth_review.repair_section_ids)
                if depth_repairable and depth_review is not None
                else _coverage_repair_section_ids(
                    outline,
                    section_drafts,
                    repair_extraction,
                    relevant_segments,
                )
            )
            repair_sections = (
                tuple(
                    draft
                    for draft in section_drafts
                    if draft.section_id in teaching_repair_section_ids
                )
                if teaching_enabled
                else section_drafts
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
                outline=outline if teaching_enabled else None,
                section_drafts=repair_sections if teaching_enabled else (),
                depth_review=depth_review if teaching_enabled else None,
                repair_section_ids=(teaching_repair_section_ids if teaching_enabled else []),
            )
            repair_system_prompt = (
                TEACHING_REPAIR_SYSTEM_PROMPT if teaching_enabled else SYSTEM_PROMPTS["repair"]
            )
            if teaching_enabled:
                repair_payload["prompt_version"] = TEACHING_REPAIR_PROMPT_VERSION
            repair_budget = _provider_request_budget(
                stage="repair",
                system_prompt=repair_system_prompt,
                payload=repair_payload,
                json_schema=_model_json_schema(RepairResult),
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
                    outline=outline if teaching_enabled else None,
                    section_drafts=repair_sections if teaching_enabled else (),
                    depth_review=depth_review if teaching_enabled else None,
                    repair_section_ids=(teaching_repair_section_ids if teaching_enabled else []),
                )
                compact_budget = _provider_request_budget(
                    stage="repair",
                    system_prompt=repair_system_prompt,
                    payload=compact_payload,
                    json_schema=_model_json_schema(RepairResult),
                    context_tokens=self.provider.context_tokens,
                )
                if compact_budget["within_budget"]:
                    repair_payload = compact_payload
                    if teaching_enabled:
                        repair_payload["prompt_version"] = TEACHING_REPAIR_PROMPT_VERSION
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
                    system_prompt_override=(
                        TEACHING_REPAIR_SYSTEM_PROMPT if teaching_enabled else None
                    ),
                    prompt_version_override=(
                        TEACHING_REPAIR_PROMPT_VERSION if teaching_enabled else None
                    ),
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
                if teaching_enabled and repair_result.section_drafts:
                    outline_by_id = {section.section_id: section for section in outline.sections}
                    normalized_repairs: dict[str, SectionDraft] = {}
                    for repaired_draft in repair_result.section_drafts:
                        if (
                            depth_review is None
                            or repaired_draft.section_id not in set(teaching_repair_section_ids)
                            or repaired_draft.section_id not in outline_by_id
                        ):
                            raise ValueError("teaching repair section is not authorized")
                        normalized_repairs[repaired_draft.section_id] = _normalize_section(
                            repaired_draft,
                            outline_by_id[repaired_draft.section_id],
                            {atom.id: atom for atom in extraction.atoms},
                        )
                    updated_by_id = {
                        draft.section_id: draft for draft in normalized_repairs.values()
                    }
                    section_drafts = tuple(
                        updated_by_id.get(draft.section_id, draft) for draft in section_drafts
                    )
                    if depth_repairable or teaching_repair_section_ids:
                        synthesis = _teaching_synthesis(
                            note_input,
                            extraction,
                            outline,
                            section_drafts,
                        )
                elif teaching_enabled and depth_repairable:
                    # A depth repair without a returned section patch cannot
                    # claim that the affected teaching section was repaired.
                    input_sufficient = False
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
                if teaching_enabled:
                    try:
                        depth_review = await self._review_depth(
                            note_input,
                            extraction,
                            outline,
                            section_drafts,
                            cancel_event,
                            audit_sink,
                            records,
                            input_sufficient=_input_is_sufficient(note_input) and input_sufficient,
                        )
                    except ProviderContextBudgetError as exc:
                        failure = _exception_failure(exc, "depth_review", {})
                        _remember_context_failure(context_failures, failure)
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
                    input_sufficient = input_sufficient and depth_review.input_sufficient
                coverage = self._validate(
                    note_input,
                    extraction,
                    synthesis,
                    review,
                    input_sufficient=input_sufficient,
                    depth_review=depth_review,
                )
                if context_failures:
                    coverage = coverage.model_copy(
                        update={"quality_gate": "failed", "repairable": False}
                    )
        _check_cancellation(cancel_event)
        prompt_versions = dict(PROMPT_VERSIONS)
        if teaching_enabled:
            prompt_versions["repair"] = TEACHING_REPAIR_PROMPT_VERSION
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
            prompt_versions=prompt_versions,
            is_local=bool(self.provider.is_local),
            outline=outline,
            section_drafts=section_drafts,
            section_attempts=section_attempts,
            depth_review=depth_review,
            input_sufficient=input_sufficient,
            context_status=("within_budget" if not context_failures else "failed"),
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
        fallback_batches = _fallback_source_review_batches(
            note_input,
            extraction,
            self.provider.context_tokens,
            input_fraction=0.2,
        )
        if fallback_batches is None:
            pending_batches = build_segment_batches(
                note_input,
                self.provider.context_tokens,
                input_fraction=0.2,
            )
        elif not fallback_batches:
            failure = {
                "stage": "review",
                "code": "fallback_source_provenance_unverified",
                "message": (
                    "canonical fallback evidence could not be mapped "
                    "losslessly to the raw source transcript"
                ),
            }
            if context_failures is None:
                raise ProviderContextBudgetError("review", failure)
            _remember_context_failure(context_failures, failure)
            await _record_context_failure(failure, audit_sink, records)
            return _merge_reviews(reviews)
        else:
            pending_batches = fallback_batches
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
            batch_claims = _claims_for_source_batch(claims, extraction, batch)
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
                for claim in batch_claims
            )
            if not source_fits or not claims_fit:
                split_batches = (
                    []
                    if _is_source_preserving_fallback(extraction) and len(batch.segments) == 1
                    else _split_segment_batch(batch)
                )
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
                            for claim in batch_claims
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
                        json_schema=_model_json_schema(budget_schema),
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
                if not source_fits:
                    return _merge_reviews(reviews)
            source_batches.append(batch)
            fragment_segment_indexes = _fragment_segment_indexes(source_batches)

        for batch in source_batches:
            try:
                source_payload = _source_review_payload(
                    note_input, extraction, synthesis, batch, review_pass
                )
                source_review: CoverageReview | None = None
                feedback: str | None = None
                for attempt in range(2):
                    request_payload = dict(source_payload)
                    if feedback:
                        request_payload["validation_feedback"] = feedback
                    candidate = await self._call(
                        stage="review",
                        payload=request_payload,
                        schema=CoverageReview,
                        cancel_event=cancel_event,
                        audit_sink=audit_sink,
                        records=records,
                        batch_boundary=batch.boundary(),
                        system_prompt_override=(REVIEW_RETRY_SYSTEM_PROMPT if attempt else None),
                        prompt_version_override=(REVIEW_RETRY_PROMPT_VERSION if attempt else None),
                    )
                    try:
                        _validate_source_review(candidate, batch)
                        source_review = candidate
                        break
                    except ValueError as exc:
                        feedback = str(exc)
                        if attempt == 1:
                            raise
                if source_review is None:
                    raise ValueError("source review response was empty")
                source_review = _normalize_source_fallback_review(
                    source_review,
                    extraction,
                    aweme_id=note_input.aweme_id,
                )
                reviews.append(source_review)
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
        merged = _remove_noop_disposition_corrections(merged, extraction)
        merged = _normalize_source_fallback_unsupported_claims(
            extraction,
            claims,
            merged,
        )
        if not claims:
            return merged

        supported_claim_ids: set[str] = set()
        contradicted_claim_ids: set[str] = set()
        reviewed_claim_ids: set[str] = set()
        for batch in source_batches:
            batch_claims = _claims_for_source_batch(claims, extraction, batch)
            claim_batches: list[list[DraftClaim]] = []
            current_claims: list[DraftClaim] = []
            for claim in batch_claims:
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
                            json_schema=_model_json_schema(DraftClaimReview),
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
                    claim_payload = _claim_review_payload(
                        note_input, extraction, batch, review_pass, claim_batch
                    )
                    claim_review: DraftClaimReview | None = None
                    partial_review: DraftClaimReview | None = None
                    feedback: str | None = None
                    for attempt in range(2):
                        request_payload = dict(claim_payload)
                        if feedback:
                            request_payload["validation_feedback"] = feedback
                        candidate = await self._call(
                            stage="review",
                            payload=request_payload,
                            schema=DraftClaimReview,
                            cancel_event=cancel_event,
                            audit_sink=audit_sink,
                            records=records,
                            batch_boundary=batch.boundary(),
                            system_prompt_override=(
                                REVIEW_RETRY_SYSTEM_PROMPT if attempt else None
                            ),
                            prompt_version_override=(
                                REVIEW_RETRY_PROMPT_VERSION if attempt else None
                            ),
                        )
                        try:
                            _validate_claim_review(candidate, expected_ids)
                            if set(candidate.reviewed_claim_ids) != expected_ids:
                                feedback = (
                                    "reviewed_claim_ids 不完整：必须逐项返回本批全部 claim.id；"
                                    "来源未提及的 claim 也必须列入 reviewed_claim_ids，不能省略。"
                                )
                                if attempt == 0:
                                    # Keep only explicit, valid verdicts. The
                                    # retry independently reviews the missing
                                    # claims against their original evidence,
                                    # instead of asking the model to repeat a
                                    # long list it has already omitted from.
                                    reviewed = set(candidate.reviewed_claim_ids)
                                    partial_review = DraftClaimReview(
                                        reviewed_claim_ids=sorted(reviewed),
                                        supported_claim_ids=sorted(
                                            set(candidate.supported_claim_ids) & reviewed
                                        ),
                                        contradicted_claim_ids=sorted(
                                            set(candidate.contradicted_claim_ids) & reviewed
                                        ),
                                    )
                                    expected_ids -= reviewed
                                    claim_payload = _claim_review_payload(
                                        note_input, extraction, batch, review_pass,
                                        [claim for claim in claim_batch
                                         if claim.id in expected_ids],
                                    )
                                    continue
                            claim_review = candidate
                            break
                        except ValueError as exc:
                            feedback = str(exc)
                            if attempt == 1:
                                raise
                    if claim_review is None:
                        raise ValueError("draft claim review response was empty")
                    if partial_review is not None:
                        claim_review = DraftClaimReview(
                            reviewed_claim_ids=[
                                *partial_review.reviewed_claim_ids,
                                *claim_review.reviewed_claim_ids,
                            ],
                            supported_claim_ids=[
                                *partial_review.supported_claim_ids,
                                *claim_review.supported_claim_ids,
                            ],
                            contradicted_claim_ids=[
                                *partial_review.contradicted_claim_ids,
                                *claim_review.contradicted_claim_ids,
                            ],
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
                reviewed_claim_ids.update(reviewed_ids)
                supported_claim_ids.update(reviewed_ids & supported_ids)
                contradicted_claim_ids.update(contradicted_ids)

        claim_by_id = {claim.id: claim for claim in claims}
        unsupported_ids = _claim_review_unsupported_ids(
            claims,
            reviewed_ids=reviewed_claim_ids,
            supported_ids=supported_claim_ids,
            contradicted_ids=contradicted_claim_ids,
        )
        if _is_source_preserving_fallback(extraction):
            unsupported_ids = {
                claim_id
                for claim_id in unsupported_ids
                if claim_id in contradicted_claim_ids or claim_id not in reviewed_claim_ids
            }
        unsupported_draft_claims = list(merged.unsupported_draft_claims)
        if _is_source_preserving_fallback(extraction):
            supported_texts = {
                claim_by_id[claim_id].text
                for claim_id in supported_claim_ids - contradicted_claim_ids
            }
            unsupported_draft_claims = [
                value for value in unsupported_draft_claims if value not in supported_texts
            ]
        return merged.model_copy(
            update={
                "unsupported_draft_claims": _dedupe_text(
                    [
                        *unsupported_draft_claims,
                        *(claim_by_id[claim_id].text for claim_id in sorted(unsupported_ids)),
                    ]
                )
            }
        )

    async def _review_depth(
        self,
        note_input: NoteInput,
        extraction: ExtractionResult,
        outline: TeachingOutline,
        sections: tuple[SectionDraft, ...],
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None,
        records: list[StageRecord],
        *,
        input_sufficient: bool,
    ) -> DepthReview:
        if _is_source_preserving_fallback(extraction):
            deterministic = DepthValidator().validate(
                atoms=extraction.atoms,
                outline=outline,
                sections=list(sections),
            )
            return deterministic.model_copy(update={"input_sufficient": input_sufficient})
        groups: list[tuple[SectionDraft, ...]] = [sections]
        provider_reviews: list[DepthReview] = []
        while groups:
            group = groups.pop(0)
            payload = _depth_review_payload(
                note_input,
                extraction,
                outline,
                group,
                all_sections=sections,
            )
            if not self._request_fits("depth_review", payload, DepthReview):
                if len(group) > 1:
                    midpoint = max(1, len(group) // 2)
                    groups = [group[:midpoint], group[midpoint:], *groups]
                    continue
                section_chunks = _split_depth_section(group[0], outline)
                if section_chunks:
                    groups = [*section_chunks, *groups]
                    continue
                compact = _depth_review_payload(
                    note_input,
                    extraction,
                    outline,
                    group,
                    all_sections=sections,
                    compact=True,
                    context_tokens=self.provider.context_tokens,
                )
                if self._request_fits("depth_review", compact, DepthReview):
                    payload = compact
                else:
                    fragment_payloads = _depth_fragment_payloads(
                        note_input,
                        extraction,
                        outline,
                        group,
                        all_sections=sections,
                        context_tokens=self.provider.context_tokens,
                    )
                    if not fragment_payloads or any(
                        not self._request_fits("depth_review", item, DepthReview)
                        for item in fragment_payloads
                    ):
                        raise ProviderContextBudgetError("depth_review")
                    for fragment_payload in fragment_payloads:
                        provider_reviews.append(
                            await self._call(
                                stage="depth_review",
                                payload=fragment_payload,
                                schema=DepthReview,
                                cancel_event=cancel_event,
                                audit_sink=audit_sink,
                                records=records,
                            )
                        )
                    continue
            provider_reviews.append(
                await self._call(
                    stage="depth_review",
                    payload=payload,
                    schema=DepthReview,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                )
            )
        provider_review = _merge_depth_reviews(
            provider_reviews,
            input_sufficient=input_sufficient,
        )
        provider_review = _validate_depth_review_refs(
            provider_review,
            {atom.id for atom in extraction.atoms},
            section_ids={section.section_id for section in outline.sections},
        )
        provider_review = _normalize_provider_depth_review(
            extraction,
            provider_review,
        )
        deterministic = DepthValidator().validate(
            atoms=extraction.atoms,
            outline=outline,
            sections=list(sections),
            review=provider_review,
        )
        return deterministic.model_copy(update={"input_sufficient": input_sufficient})

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
                    _segment_payload(s) for s in note_input.segments if s.index in indexes
                ],
                "input_sufficient": bool(note_input.transcript.strip()),
            }
            if not self._request_fits("outline", payload, TeachingOutline):
                # Never shorten an atom or its source segment to fit. Split
                # the evidence set and let each request carry complete text.
                if len(chunk) <= 1:
                    raise ProviderContextBudgetError("outline")
                midpoint = max(1, len(chunk) // 2)
                chunks = [chunk[:midpoint], chunk[midpoint:], *chunks]
                continue
            outline = await self._call(
                stage="outline",
                payload=payload,
                schema=TeachingOutline,
                cancel_event=cancel_event,
                audit_sink=audit_sink,
                records=records,
            )
            _validate_outline_response(outline)
            outlines.append(outline)
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
                atom_by_id[atom_id] for atom_id in section.atom_ids if atom_id in atom_by_id
            ]
            attempts[section.section_id] = 0
            chunks = [section_atoms]
            chunk_drafts: list[SectionDraft] = []
            failed = False
            while chunks:
                chunk = chunks.pop(0)
                payload = _section_payload(note_input, section, chunk)
                if not self._request_fits("section", payload, SectionDraft):
                    # A compact payload would silently discard facts. Keep
                    # complete atom/source text and split the atom set.
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
                        raw_result = await self._call(
                            stage="section",
                            payload=payload,
                            schema=SectionDraft,
                            cancel_event=cancel_event,
                            audit_sink=audit_sink,
                            records=records,
                        )
                        # Normalize inside the retry boundary so an otherwise
                        # valid JSON response with unauthorized atom references
                        # is treated as a failed section attempt, not as a
                        # pipeline-wide crash.
                        result = _normalize_section(
                            raw_result,
                            section,
                            atom_by_id,
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
                if chunk_drafts:
                    # Preserve successful atom chunks when one oversized or
                    # malformed chunk fails.  The explicit marker keeps the
                    # section readable while the quality gate can reject it.
                    partial = _merge_section_drafts(chunk_drafts, section)
                    partial = partial.model_copy(
                        update={
                            "unprovided_items": [
                                *partial.unprovided_items,
                                section.section_id,
                            ]
                        }
                    )
                    drafts.append(_normalize_section(partial, section, atom_by_id))
                else:
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

    async def _repair_teaching_units(
        self,
        *,
        note_input: NoteInput,
        extraction: ExtractionResult,
        synthesis: SynthesisResult,
        review: CoverageReview,
        coverage: CoverageReport,
        outline: TeachingOutline,
        section_drafts: tuple[SectionDraft, ...],
        depth_review: DepthReview | None,
        depth_repairable: bool,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink | None,
        records: list[StageRecord],
    ) -> tuple[ExtractionResult, SynthesisResult, tuple[SectionDraft, ...]]:
        """Run one bounded repair request per independent teaching section.

        Repair patches are accumulated only in memory.  The caller publishes a
        result after this method returns and performs one global review, so a
        later unit failure can never expose an earlier partial patch.
        """

        if depth_repairable and depth_review is not None:
            repair_extraction, all_segments = _depth_repair_slice(
                extraction,
                section_drafts,
                depth_review,
                outline=outline,
            )
            section_ids = list(depth_review.repair_section_ids)
        else:
            repair_extraction, all_segments = _repair_slice(
                extraction,
                synthesis,
                review,
                coverage,
            )
            section_ids = _coverage_repair_section_ids(
                outline,
                section_drafts,
                repair_extraction,
                all_segments,
            )

        outline_by_id = {section.section_id: section for section in outline.sections}
        draft_by_id = {draft.section_id: draft for draft in section_drafts}
        atom_by_id = {atom.id: atom for atom in extraction.atoms}
        atoms_by_segment: dict[int, set[str]] = {}
        evidence_neighbors = {atom_id: set() for atom_id in atom_by_id}
        for atom in extraction.atoms:
            atoms_by_segment.setdefault(atom.source_segment, set()).add(atom.id)
            for relation_id in atom.relations:
                if relation_id not in atom_by_id:
                    continue
                evidence_neighbors[atom.id].add(relation_id)
                evidence_neighbors[relation_id].add(atom.id)
        for segment_atom_ids in atoms_by_segment.values():
            for atom_id in segment_atom_ids:
                evidence_neighbors[atom_id].update(segment_atom_ids)

        def evidence_closure(
            atom_ids: set[str],
            segment_indexes: set[int],
        ) -> tuple[set[str], set[int]]:
            """Return all atoms and source segments connected to this evidence."""

            pending = [
                *atom_ids,
                *(
                    atom_id
                    for segment_index in segment_indexes
                    for atom_id in atoms_by_segment.get(segment_index, ())
                ),
            ]
            expanded_atom_ids: set[str] = set()
            expanded_segments = set(segment_indexes)
            while pending:
                atom_id = pending.pop()
                if atom_id in expanded_atom_ids or atom_id not in atom_by_id:
                    continue
                expanded_atom_ids.add(atom_id)
                atom = atom_by_id[atom_id]
                expanded_segments.add(atom.source_segment)
                pending.extend(evidence_neighbors[atom_id] - expanded_atom_ids)
            return expanded_atom_ids, expanded_segments

        allowed_sections = [
            section_id
            for section_id in section_ids
            if section_id in outline_by_id
        ]
        if not allowed_sections:
            raise RepairUnitFailure(
                {
                    "stage": "repair",
                    "code": "repair_unit_no_authorized_sections",
                }
            )

        # Build one ownership record per section, then join records that share
        # source evidence or atom relations.  A unit always owns complete
        # section ids; no section is split across provider calls.
        ownership: dict[str, tuple[set[str], set[int]]] = {}
        for section_id in allowed_sections:
            atom_ids = set(outline_by_id[section_id].atom_ids)
            draft = draft_by_id.get(section_id)
            if draft is not None:
                atom_ids.update(draft.included_atom_ids)
            if depth_review is not None:
                for field_name in (
                    "abstracted_atom_ids",
                    "example_without_result_atom_ids",
                    "condition_without_scope_atom_ids",
                    "reordered_step_atom_ids",
                    "lost_literal_atom_ids",
                    "lost_warning_or_promotion_atom_ids",
                ):
                    atom_ids.update(
                        atom_id
                        for atom_id in getattr(depth_review, field_name)
                        if atom_id in atom_by_id and atom_id in atom_ids
                    )
            atom_ids = {atom_id for atom_id in atom_ids if atom_id in atom_by_id}
            segments = {
                atom_by_id[atom_id].source_segment
                for atom_id in atom_ids
            }
            if not segments:
                # A section with no assigned atom can still be authorized by a
                # source omission or a disposition correction.
                segments = {
                    segment_index
                    for segment_index in all_segments
                    if any(
                        atom.source_segment == segment_index
                        and atom.id in set(outline_by_id[section_id].atom_ids)
                        for atom in extraction.atoms
                    )
                }
            ownership[section_id] = evidence_closure(atom_ids, segments)

        parent = {section_id: section_id for section_id in allowed_sections}

        def find(section_id: str) -> str:
            while parent[section_id] != section_id:
                parent[section_id] = parent[parent[section_id]]
                section_id = parent[section_id]
            return section_id

        def join(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for index, left in enumerate(allowed_sections):
            left_atoms, left_segments = ownership[left]
            for right in allowed_sections[index + 1 :]:
                right_atoms, right_segments = ownership[right]
                if left_segments & right_segments:
                    join(left, right)
                    continue
                if any(
                    relation_id in right_atoms
                    for atom_id in left_atoms
                    for relation_id in atom_by_id[atom_id].relations
                ) or any(
                    relation_id in left_atoms
                    for atom_id in right_atoms
                    for relation_id in atom_by_id[atom_id].relations
                ):
                    join(left, right)

        grouped: dict[str, list[str]] = {}
        for section_id in allowed_sections:
            grouped.setdefault(find(section_id), []).append(section_id)

        units: list[RepairUnit] = []
        for grouped_ids in grouped.values():
            unit_atom_ids: set[str] = set()
            unit_segments: set[int] = set()
            for section_id in grouped_ids:
                section_atoms, section_segments = ownership[section_id]
                unit_atom_ids.update(section_atoms)
                unit_segments.update(section_segments)
            unit_atom_ids, unit_segments = evidence_closure(unit_atom_ids, unit_segments)
            unit_segments.update(
                atom.source_segment
                for atom in repair_extraction.atoms
                if atom.id in unit_atom_ids
            )
            if not unit_segments:
                unit_segments.update(all_segments)
            local_extraction = _subset_extraction(
                extraction,
                unit_atom_ids,
                unit_segments,
            )
            local_sections = tuple(
                draft_by_id[section_id]
                for section_id in grouped_ids
                if section_id in draft_by_id
            )
            local_outline_sections = [outline_by_id[section_id] for section_id in grouped_ids]
            local_atom_ids = {atom.id for atom in local_extraction.atoms}
            local_outline = TeachingOutline(
                sections=local_outline_sections,
                included_atom_ids=[
                    atom_id
                    for atom_id in outline.included_atom_ids
                    if atom_id in local_atom_ids
                ],
                unassigned_atom_ids=[],
                input_sufficient=outline.input_sufficient,
            )
            local_depth = None
            if depth_review is not None:
                depth_data = depth_review.model_dump(mode="json")
                depth_atom_fields = (
                    "abstracted_atom_ids",
                    "example_without_result_atom_ids",
                    "condition_without_scope_atom_ids",
                    "reordered_step_atom_ids",
                    "lost_literal_atom_ids",
                    "lost_warning_or_promotion_atom_ids",
                )
                for field_name in depth_atom_fields:
                    depth_data[field_name] = [
                        atom_id
                        for atom_id in depth_data[field_name]
                        if atom_id in local_atom_ids
                    ]
                depth_data["repair_section_ids"] = list(grouped_ids)
                depth_data["unknown_atom_ids"] = []
                depth_data["unknown_section_ids"] = []
                local_depth = DepthReview.model_validate(depth_data, strict=True)
            units.append(
                RepairUnit(
                    section_ids=tuple(grouped_ids),
                    extraction=local_extraction,
                    relevant_segments=frozenset(unit_segments),
                    section_drafts=local_sections,
                    outline=local_outline,
                    depth_review=local_depth,
                )
            )

        working_extraction = extraction
        working_synthesis = synthesis
        working_drafts = {draft.section_id: draft for draft in section_drafts}
        repaired_section_drafts = False
        synthesis_only_patches: list[tuple[SynthesisResult, set[int]]] = []
        for unit in units:
            if cancel_event.is_set():
                raise RepairUnitFailure(
                    {
                        "stage": "repair",
                        "code": "repair_unit_cancelled",
                        "repair_section_ids": list(unit.section_ids),
                    }
                )
            local_synthesis = _slice_synthesis(
                working_synthesis,
                set(unit.relevant_segments),
                {atom.id for atom in unit.extraction.atoms},
                working_extraction,
            )
            local_review = _slice_review(review, set(unit.relevant_segments))
            payload = _repair_payload(
                note_input,
                unit.extraction,
                local_synthesis,
                working_extraction,
                local_review,
                coverage,
                set(unit.relevant_segments),
                compact_content=True,
                outline=unit.outline,
                section_drafts=unit.section_drafts,
                depth_review=unit.depth_review,
                repair_section_ids=list(unit.section_ids),
            )
            payload["prompt_version"] = TEACHING_REPAIR_PROMPT_VERSION
            # Give each local request stable, source-oriented atom ids.  The
            # merged extraction uses deterministic ``B###-A###`` ids, while a
            # local unit should remain readable and independent of atoms that
            # are outside the request.  Keep a reverse map so returned patches
            # can be merged back into the canonical extraction safely.
            local_to_global: dict[str, str] = {}
            global_to_local: dict[str, str] = {}
            candidate_counts: Counter[str] = Counter(
                f"{atom.id.split('-', 1)[0]}-unit-{atom.source_segment}"
                for atom in unit.extraction.atoms
            )
            for atom in unit.extraction.atoms:
                candidate = f"{atom.id.split('-', 1)[0]}-unit-{atom.source_segment}"
                if candidate_counts[candidate] != 1:
                    continue
                local_to_global[candidate] = atom.id
                global_to_local[atom.id] = candidate

            def remap(value: Any, mapping: Mapping[str, str]) -> Any:
                if isinstance(value, str):
                    return mapping.get(value, value)
                if isinstance(value, list):
                    return [remap(item, mapping) for item in value]
                if isinstance(value, dict):
                    return {key: remap(item, mapping) for key, item in value.items()}
                return value

            if global_to_local:
                payload = remap(payload, global_to_local)
            budget = _provider_request_budget(
                stage="repair",
                system_prompt=TEACHING_REPAIR_SYSTEM_PROMPT,
                payload=payload,
                json_schema=_model_json_schema(RepairResult),
                context_tokens=self.provider.context_tokens,
            )
            if not budget["within_budget"]:
                failure = _budget_failure("repair", budget)
                failure["repair_section_ids"] = list(unit.section_ids)
                await _record_context_failure(failure, audit_sink, records)
                raise RepairUnitFailure(failure)
            payload["context_budget"] = budget
            try:
                repair_result = await self._call(
                    stage="repair",
                    payload=payload,
                    schema=RepairResult,
                    cancel_event=cancel_event,
                    audit_sink=audit_sink,
                    records=records,
                    system_prompt_override=TEACHING_REPAIR_SYSTEM_PROMPT,
                    prompt_version_override=TEACHING_REPAIR_PROMPT_VERSION,
                )
                if local_to_global:
                    repair_result = RepairResult.model_validate(
                        remap(repair_result.model_dump(mode="json"), local_to_global),
                        strict=True,
                    )
                updated_extraction = _merge_repair_extraction_patch(
                    working_extraction,
                    repair_result.extraction_patch,
                    set(unit.relevant_segments),
                )
                updated_synthesis = _merge_repair_patch(
                    working_synthesis,
                    repair_result.synthesis_patch,
                    set(unit.relevant_segments),
                    extraction=updated_extraction,
                )
                outline_by_id_global = {
                    section.section_id: section for section in outline.sections
                }
                for repaired_draft in repair_result.section_drafts:
                    if repaired_draft.section_id not in unit.section_ids:
                        raise ValueError("teaching repair section is not authorized")
                    normalized = _normalize_section(
                        repaired_draft,
                        outline_by_id_global[repaired_draft.section_id],
                        {atom.id: atom for atom in updated_extraction.atoms},
                    )
                    working_drafts[normalized.section_id] = normalized
                    repaired_section_drafts = True
                if not repair_result.section_drafts:
                    synthesis_only_patches.append(
                        (repair_result.synthesis_patch, set(unit.relevant_segments))
                    )
            except JobCancelledError as exc:
                raise RepairUnitFailure(
                    {
                        "stage": "repair",
                        "code": "repair_unit_cancelled",
                        "message": str(exc),
                        "repair_section_ids": list(unit.section_ids),
                    }
                ) from exc
            except NoteProviderError as exc:
                raise RepairUnitFailure(
                    {
                        "stage": "repair",
                        "code": "provider_error",
                        "provider": exc.provider,
                        "provider_stage": exc.stage,
                        "provider_code": exc.code,
                        "repair_section_ids": list(unit.section_ids),
                    }
                ) from exc
            except ProviderContextBudgetError as exc:
                failure = _exception_failure(exc, "repair", payload)
                failure["repair_section_ids"] = list(unit.section_ids)
                await _record_context_failure(failure, audit_sink, records)
                raise RepairUnitFailure(failure) from exc
            except (TypeError, ValueError, KeyError) as exc:
                raise RepairUnitFailure(
                    {
                        "stage": "repair",
                        "code": "repair_unit_merge_failed",
                        "message": str(exc),
                        "repair_section_ids": list(unit.section_ids),
                    }
                ) from exc
            working_extraction, working_synthesis = updated_extraction, updated_synthesis

        if repaired_section_drafts:
            # In the teaching pipeline the rendered synthesis is derived from
            # the section drafts.  Rebuild it once after every private unit
            # merge so the global review sees each repaired chapter.
            working_synthesis = _teaching_synthesis(
                note_input,
                working_extraction,
                outline,
                tuple(
                    working_drafts.get(draft.section_id, draft)
                    for draft in section_drafts
                ),
            )
            for patch, patch_segments in synthesis_only_patches:
                working_synthesis = _merge_repair_patch(
                    working_synthesis,
                    patch,
                    patch_segments,
                    extraction=working_extraction,
                )

        return (
            working_extraction,
            working_synthesis,
            tuple(
                working_drafts.get(draft.section_id, draft)
                for draft in section_drafts
            ),
        )

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
        system_prompt_override: str | None = None,
        prompt_version_override: str | None = None,
    ) -> T:
        _check_cancellation(cancel_event)
        system_prompt = system_prompt_override or SYSTEM_PROMPTS[stage]
        prompt_version = prompt_version_override or PROMPT_VERSIONS[stage]
        json_schema = _model_json_schema(schema)
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
                prompt_version=prompt_version,
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
                prompt_version=prompt_version,
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
                prompt_version=prompt_version,
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
            prompt_version=prompt_version,
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
        *,
        depth_review: DepthReview | None = None,
        input_sufficient: bool = True,
    ) -> CoverageReport:
        return self.validator.validate(
            segment_count=len(note_input.segments),
            extraction=extraction,
            synthesis=synthesis,
            review=review,
            semantic_generator=f"{self.provider.name}/{self.provider.model}",
            depth_review=depth_review,
            input_sufficient=input_sufficient,
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
                input_sufficient=False,
            )
        coverage = coverage.model_copy(
            update={
                "quality_gate": "failed",
                "repairable": False,
                "depth_passed": False,
                "input_sufficient": False,
                "semantic_complete": False,
            }
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
                    segment_id=segment.segment_id,
                    source_offset=segment.source_offset,
                )
            ],
            [
                InputSegment(
                    index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text[midpoint:],
                    segment_id=segment.segment_id,
                    source_offset=segment.source_offset + midpoint,
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


def _input_is_sufficient(note_input: NoteInput) -> bool:
    """Return whether the available transcript can support a complete note."""

    if not note_input.transcript.strip() or not note_input.segments:
        return False
    # A few acknowledgement syllables can be structurally valid ASR input, but
    # cannot support a complete learning note. Count Unicode alphanumerics so
    # line breaks and repeated punctuation cannot make a sparse transcript pass.
    if sum(character.isalnum() for character in note_input.transcript) < 12:
        return False
    # A transcription can exceed the character threshold by repeating one
    # acknowledgement across many ASR chunks.  Require enough variation in
    # the segment text to support a learning note; otherwise no provider is
    # called and the existing note stays in place.
    normalized_segments = [
        "".join(character for character in segment.text.casefold() if character.isalnum())
        for segment in note_input.segments
    ]
    non_empty_segments = [text for text in normalized_segments if text]
    if not non_empty_segments or (
        len(non_empty_segments) > 1 and len(set(non_empty_segments)) == 1
    ):
        return False
    if note_input.transcript_variant not in {"raw", "cleaned"}:
        return False
    warning_text = " ".join(note_input.warnings).casefold()
    sparse_markers = (
        "synthetic segment",
        "segments.json",
        "未覆盖完整 transcript",
        "转写过少",
        "转写稀疏",
        "sparse transcript",
    )
    if any(marker.casefold() in warning_text for marker in sparse_markers):
        return False

    # Visual dependency is reported independently by rendering. It can limit
    # what the note claims about the video, but it does not make complete text
    # evidence insufficient by itself.
    return True


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
        "dispositions": [item.model_dump(mode="json") for item in extraction.dispositions],
    }


def _project_atom(
    atom: Any,
    *,
    include_source_excerpt: bool = True,
) -> dict[str, Any]:
    data = (
        atom.model_dump(mode="json") if isinstance(atom, BaseModel) else copy.deepcopy(dict(atom))
    )
    # ``source_start`` is local routing metadata, not semantic evidence the
    # model should be asked to infer or repeat.
    data.pop("source_start", None)
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
        atom.id: atom.source_excerpt if atom.statement == atom.source_excerpt else atom.statement
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
        data[field_name] = [compact_item(item) for item in data[field_name]]
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


def _shrink_atom_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce catalog overhead while retaining the oldest and newest links."""

    if len(catalog) <= 1:
        return catalog
    keep = max(1, len(catalog) // 2)
    if keep == 1:
        return [catalog[0]]
    return [catalog[0], *catalog[-(keep - 1) :]]


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


def _validate_outline_response(outline: TeachingOutline) -> None:
    section_ids = [section.section_id for section in outline.sections]
    if (
        len(section_ids) != len(_TEACHING_SECTION_IDS)
        or len(set(section_ids)) != len(section_ids)
        or set(section_ids) != set(_TEACHING_SECTION_IDS)
    ):
        raise ValueError("outline must contain each fixed teaching section exactly once")


def _validate_extraction_response(
    batch: SegmentBatch,
    extraction: ExtractionResult,
) -> None:
    """Reject structurally valid JSON that loses source evidence.

    Ollama can satisfy the Pydantic schema while returning a shortened title,
    an excerpt with a timestamp appended, or an included disposition without
    any atoms. Those responses poison every later teaching stage, so retry the
    extraction while the complete source segment is still available.
    """

    segments_by_index: dict[int, list[InputSegment]] = {}
    for segment in batch.segments:
        segments_by_index.setdefault(segment.index, []).append(segment)
    expected_segments = set(segments_by_index)
    disposition_by_segment: dict[int, list[Any]] = {}
    for disposition in extraction.dispositions:
        disposition_by_segment.setdefault(disposition.segment_index, []).append(disposition)
    if set(disposition_by_segment) != expected_segments:
        raise ValueError("extraction source coverage has missing segment dispositions")
    if any(
        len(items) != len(segments_by_index[index])
        for index, items in disposition_by_segment.items()
        if len(segments_by_index[index]) == 1
    ):
        raise ValueError("extraction source coverage has duplicate segment dispositions")

    atom_by_id: dict[str, Any] = {}
    for atom in extraction.atoms:
        if atom.id in atom_by_id:
            raise ValueError(f"extraction atom_ids duplicate: {atom.id}")
        if atom.source_segment not in expected_segments:
            raise ValueError("extraction atom source_segment is outside the batch")
        atom_by_id[atom.id] = atom

    for disposition in extraction.dispositions:
        if disposition.disposition in {"included", "promotion"} and not disposition.atom_ids:
            raise ValueError("extraction disposition requires atom_ids")
        if disposition.disposition not in {"included", "promotion"} and disposition.atom_ids:
            raise ValueError("extraction disposition has unauthorized atom_ids")
        for atom_id in disposition.atom_ids:
            atom = atom_by_id.get(atom_id)
            if atom is None or atom.source_segment != disposition.segment_index:
                raise ValueError("extraction disposition references an invalid atom_id")

    # Every extracted atom must be accounted for by exactly one disposition of
    # its source segment.  Without this, a schema-shaped response can attach
    # meaningful atoms to segments marked ``unintelligible_asr`` with empty
    # atom lists.  The later source-coverage loop deliberately skips those
    # dispositions, allowing a contradictory extraction to poison the
    # teaching stages instead of triggering the exact-source fallback.
    for segment_index in expected_segments:
        expected_atom_ids = {
            atom.id for atom in extraction.atoms if atom.source_segment == segment_index
        }
        mapped_atom_ids = [
            atom_id
            for disposition in disposition_by_segment[segment_index]
            for atom_id in disposition.atom_ids
        ]
        if len(mapped_atom_ids) != len(set(mapped_atom_ids)):
            raise ValueError("extraction atom mapping duplicates a source atom")
        if set(mapped_atom_ids) != expected_atom_ids:
            raise ValueError("extraction atom mapping does not match its source segment")

    for segment_index, source_segments in segments_by_index.items():
        dispositions = disposition_by_segment[segment_index]
        # Duplicate indexes are lossless fragments of one original segment.
        # Their dispositions are merged later, so validate excerpt membership
        # without requiring one fragment to cover another fragment's text.
        if len(source_segments) > 1:
            source_texts = [segment.text for segment in source_segments]
            for atom in extraction.atoms:
                if atom.source_segment == segment_index and not any(
                    atom.source_excerpt in source for source in source_texts
                ):
                    raise ValueError("extraction source coverage contains a non-source excerpt")
            continue
        if dispositions[0].disposition not in {"included", "promotion"}:
            continue
        source = source_segments[0].text
        if not source.strip():
            continue
        intervals: list[tuple[int, int]] = []
        for atom in extraction.atoms:
            if atom.source_segment != segment_index:
                continue
            excerpt = atom.source_excerpt
            starts: list[int] = []
            cursor = 0
            while excerpt:
                start = source.find(excerpt, cursor)
                if start < 0:
                    break
                starts.append(start)
                cursor = start + 1
            if not starts:
                raise ValueError("extraction source coverage contains a non-source excerpt")
            intervals.extend((start, start + len(excerpt)) for start in starts)
        covered = [False] * len(source)
        for start, end in intervals:
            for offset in range(start, end):
                if not source[offset].isspace():
                    covered[offset] = True
        if any(not covered[offset] for offset, char in enumerate(source) if not char.isspace()):
            raise ValueError("extraction source coverage is incomplete")


def _canonical_source_extraction(batch: SegmentBatch) -> ExtractionResult:
    """Preserve exact source evidence after two invalid local extractions."""

    atoms: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    for segment_number, segment in enumerate(batch.segments, 1):
        source = segment.text
        if not source.strip():
            dispositions.append(
                {
                    "segment_index": segment.index,
                    "disposition": "unintelligible_asr",
                    "atom_ids": [],
                    "reason": "source segment is empty",
                }
            )
            continue
        # Keep punctuation with each clause so the source coverage validator
        # can prove that no character was discarded or rewritten. Long
        # unpunctuated ASR spans must also be split before review: the review
        # layer can further divide source segments for its own context budget,
        # and an atom larger than a review fragment cannot be evidenced there.
        clauses = _source_preserving_fallback_clause_spans(source)
        segment_atom_ids: list[str] = []
        for clause_number, (clause, clause_start) in enumerate(clauses, 1):
            text = clause.strip()
            if not text:
                continue
            leading_whitespace = len(clause) - len(clause.lstrip())
            atom_id = f"fallback-{batch.batch_index:03d}-{segment_number:03d}-{clause_number:03d}"
            kind = "claim"
            sequence = None
            promotional = bool(re.search(r"关注|点赞|收藏|领取|下期|私信|购买|优惠", text))
            if promotional:
                kind = "promotion"
            elif re.search(r"第[一二三四五六七八九十0-9]+步|首先|其次|最后|步骤", text):
                kind = "step"
                sequence = (
                    sum(
                        atom.get("sequence") is not None
                        for atom in atoms
                        if atom.get("kind") == "step"
                    )
                    + 1
                )
            elif re.search(r"例如|比如|案例", text):
                kind = "example"
            elif re.search(r"只有|如果|前提|条件|除非", text):
                kind = "condition"
            elif re.search(r"注意|不要|风险|否则", text):
                kind = "warning"
            atom = {
                "id": atom_id,
                "kind": kind,
                "topic": "视频原文",
                "statement": text,
                "source_segment": segment.index,
                "start": segment.start,
                "end": segment.end,
                "source_excerpt": text,
                "source_start": segment.source_offset + clause_start + leading_whitespace,
                "relations": [],
                "sequence": sequence,
                "is_promotional": promotional,
                "asr_uncertainty": [],
            }
            atoms.append(atom)
            segment_atom_ids.append(atom_id)
        # Example, condition, and warning atoms must have a relation for the strict
        # coverage validator. Link them to the first factual atom in the same
        # source segment without inventing a new claim.
        anchor = next(
            (
                item["id"]
                for item in atoms
                if item["source_segment"] == segment.index and item["kind"] in {"claim", "step"}
            ),
            None,
        )
        if anchor is not None:
            for item in atoms:
                if item["source_segment"] == segment.index and item["kind"] in {
                    "example",
                    "condition",
                    "warning",
                }:
                    item["relations"] = [anchor]
        dispositions.append(
            {
                "segment_index": segment.index,
                "disposition": "promotion"
                if any(
                    item["source_segment"] == segment.index and item["kind"] == "promotion"
                    for item in atoms
                )
                else "included",
                "atom_ids": segment_atom_ids,
                "reason": "lossless source-preserving local fallback",
            }
        )
    return ExtractionResult.model_validate(
        {"atoms": atoms, "dispositions": dispositions}, strict=True
    )


def _source_preserving_fallback_clauses(source: str) -> list[str]:
    """Split source verbatim into review-sized evidence atoms."""

    return [text for text, _ in _source_preserving_fallback_clause_spans(source)]


def _source_preserving_fallback_clause_spans(source: str) -> list[tuple[str, int]]:
    """Return lossless fallback clauses with their source character starts."""

    max_chars = 512
    clauses: list[tuple[str, int]] = []
    cursor = 0
    for clause in re.split(r"(?<=[。！？!?；;])", source):
        clause_start = cursor
        cursor += len(clause)
        if not clause.strip():
            continue
        clauses.extend(
            (clause[offset : offset + max_chars], clause_start + offset)
            for offset in range(0, len(clause), max_chars)
            if clause[offset : offset + max_chars].strip()
        )
    return clauses or [(source, 0)]


def _validate_source_review(review: CoverageReview, batch: SegmentBatch) -> None:
    expected = {segment.index for segment in batch.segments}
    actual = review.reviewed_segment_indexes
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("source review must cover each source segment exactly once")


def _validate_claim_review(
    review: DraftClaimReview,
    expected_ids: set[str],
) -> None:
    reviewed = set(review.reviewed_claim_ids)
    supported = set(review.supported_claim_ids)
    contradicted = set(review.contradicted_claim_ids)
    if (
        not reviewed <= expected_ids
        or not supported <= expected_ids
        or not contradicted <= expected_ids
    ):
        raise ValueError("draft claim review referenced an unknown claim")
    if supported & contradicted:
        raise ValueError("draft claim review returned conflicting verdicts")


def _normalize_outline(
    outline: TeachingOutline,
    extraction: ExtractionResult,
) -> TeachingOutline:
    atom_ids = {atom.id for atom in extraction.atoms}
    by_id = {section.section_id: section for section in outline.sections}
    if (
        len(outline.sections) != len(_TEACHING_SECTION_IDS)
        or len(by_id) != len(outline.sections)
        or set(by_id) != set(_TEACHING_SECTION_IDS)
    ):
        raise ValueError("outline must contain the fixed teaching sections")

    referenced_ids = {atom_id for section in outline.sections for atom_id in section.atom_ids}
    referenced_ids.update(outline.included_atom_ids)
    referenced_ids.update(outline.unassigned_atom_ids)
    unknown_ids = sorted(referenced_ids - atom_ids)
    if unknown_ids:
        raise ValueError(f"outline references unknown atoms: {unknown_ids}")

    source_order = {atom.id: index for index, atom in enumerate(extraction.atoms)}
    source_atoms = {atom.id: atom for atom in extraction.atoms}

    def ordered_unique(values: list[str]) -> list[str]:
        return sorted(
            dict.fromkeys(values),
            key=lambda atom_id: source_order[atom_id],
        )

    assigned: set[str] = set()
    sections: list[TeachingOutlineSection] = []
    source_fallback = _is_source_preserving_fallback(extraction)
    fallback_ids_by_section = {
        "overview": [atom.id for atom in extraction.atoms],
        "core_points": [
            atom.id
            for atom in extraction.atoms
            if atom.kind != "promotion" and not atom.is_promotional
        ],
        "methods_steps": [atom.id for atom in extraction.atoms if atom.kind == "step"],
        "examples_results": [
            atom.id for atom in extraction.atoms if atom.kind in {"example", "counterexample"}
        ],
        "boundaries_risks": [
            atom.id for atom in extraction.atoms if atom.kind in {"condition", "warning"}
        ],
        "actions": [atom.id for atom in extraction.atoms if atom.kind == "recommendation"],
        "timeline": [atom.id for atom in extraction.atoms],
        "evidence_limits": [atom.id for atom in extraction.atoms if atom.asr_uncertainty],
    }
    for order, section_id in enumerate(_TEACHING_SECTION_IDS, 1):
        section = by_id[section_id]
        # An atom may legitimately support more than one teaching section.
        # Only duplicate references inside the same section are collapsed.
        ids = ordered_unique(
            fallback_ids_by_section[section_id] if source_fallback else section.atom_ids
        )
        assigned.update(ids)
        timestamps = section.timestamps
        if source_fallback:
            timestamps = sorted(
                {
                    value
                    for atom_id in ids
                    for value in (
                        source_atoms[atom_id].start,
                        source_atoms[atom_id].end,
                    )
                    if value is not None
                }
            )
        sections.append(
            section.model_copy(update={"order": order, "atom_ids": ids, "timestamps": timestamps})
        )
    unassigned = [atom.id for atom in extraction.atoms if atom.id not in assigned]
    if unassigned:
        limit_index = _TEACHING_SECTION_IDS.index("evidence_limits")
        sections[limit_index] = sections[limit_index].model_copy(
            update={"atom_ids": ordered_unique([*sections[limit_index].atom_ids, *unassigned])}
        )
    return outline.model_copy(
        update={
            "sections": sections,
            "included_atom_ids": [atom.id for atom in extraction.atoms],
            "unassigned_atom_ids": [],
        }
    )


def _source_fallback_outline(extraction: ExtractionResult) -> TeachingOutline:
    atom_by_id = {atom.id: atom for atom in extraction.atoms}
    source_order = {atom.id: index for index, atom in enumerate(extraction.atoms)}

    def with_relations(atom_ids: list[str]) -> list[str]:
        expanded = list(atom_ids)
        for atom_id in atom_ids:
            atom = atom_by_id[atom_id]
            expanded.extend(relation for relation in atom.relations if relation in atom_by_id)
        return sorted(dict.fromkeys(expanded), key=source_order.__getitem__)

    non_promotional = [
        atom.id for atom in extraction.atoms if atom.kind != "promotion" and not atom.is_promotional
    ]
    overview_ids = with_relations(non_promotional[:3])
    assignments = {
        "overview": overview_ids,
        "core_points": with_relations(non_promotional),
        "methods_steps": with_relations(
            [atom.id for atom in extraction.atoms if atom.kind == "step"]
        ),
        "examples_results": with_relations(
            [atom.id for atom in extraction.atoms if atom.kind in {"example", "counterexample"}]
        ),
        "boundaries_risks": with_relations(
            [atom.id for atom in extraction.atoms if atom.kind in {"condition", "warning"}]
        ),
        "actions": with_relations(
            [atom.id for atom in extraction.atoms if atom.kind == "recommendation"]
        ),
        "timeline": [atom.id for atom in extraction.atoms],
        "evidence_limits": with_relations(
            [atom.id for atom in extraction.atoms if atom.asr_uncertainty]
        ),
    }
    titles = {
        "overview": "概述",
        "core_points": "核心要点",
        "methods_steps": "方法步骤",
        "examples_results": "示例与结果",
        "boundaries_risks": "边界与风险",
        "actions": "行动建议",
        "timeline": "时间线",
        "evidence_limits": "证据限制",
    }
    evidence_types: dict[str, list[str]] = {
        "overview": ["claim"],
        "core_points": ["claim", "reason"],
        "methods_steps": ["step"],
        "examples_results": ["example", "result"],
        "boundaries_risks": ["condition", "risk"],
        "actions": ["action"],
        "timeline": ["claim"],
        "evidence_limits": ["uncertainty"],
    }
    sections = []
    for order, section_id in enumerate(_TEACHING_SECTION_IDS, 1):
        atom_ids = assignments[section_id]
        sections.append(
            TeachingOutlineSection(
                section_id=section_id,
                title=titles[section_id],
                order=order,
                atom_ids=atom_ids,
                required_evidence_types=evidence_types[section_id],
                missing_evidence_policy="视频未提供这一类信息。",
                timestamps=sorted(
                    {
                        value
                        for atom_id in atom_ids
                        for value in (
                            atom_by_id[atom_id].start,
                            atom_by_id[atom_id].end,
                        )
                        if value is not None
                    }
                ),
            )
        )
    return TeachingOutline(
        sections=sections,
        included_atom_ids=[atom.id for atom in extraction.atoms],
        unassigned_atom_ids=[],
        input_sufficient=bool(extraction.atoms),
    )


def _source_fallback_section_drafts(
    extraction: ExtractionResult,
    outline: TeachingOutline,
) -> tuple[SectionDraft, ...]:
    atom_by_id = {atom.id: atom for atom in extraction.atoms}

    def claim(atom: Any) -> SectionClaim:
        timestamps = [value for value in (atom.start, atom.end) if value is not None]
        if not timestamps:
            timestamps = [float(atom.source_segment)]
        return SectionClaim(
            text=atom.statement,
            atom_ids=[atom.id],
            timestamps=timestamps,
        )

    drafts = []
    for section in sorted(outline.sections, key=lambda item: item.order):
        atoms = [atom_by_id[atom_id] for atom_id in section.atom_ids]
        result_ids = {
            relation
            for atom in atoms
            if atom.kind in {"example", "counterexample"}
            for relation in atom.relations
            if relation in atom_by_id
        }
        relation_anchor_ids = {
            relation
            for atom in atoms
            if atom.kind in {"condition", "warning"}
            for relation in atom.relations
            if relation in atom_by_id
        }
        claim_atoms = [
            atom
            for atom in atoms
            if atom.kind in {"claim", "reason", "definition", "number", "quote"}
            or atom.id in relation_anchor_ids
        ]
        drafts.append(
            SectionDraft(
                section_id=section.section_id,
                title=section.title,
                content=(
                    "；".join(atom.statement for atom in atoms)
                    if atoms
                    else section.missing_evidence_policy
                ),
                claims=[claim(atom) for atom in claim_atoms],
                steps=[claim(atom) for atom in atoms if atom.kind == "step"],
                examples=[
                    claim(atom) for atom in atoms if atom.kind in {"example", "counterexample"}
                ],
                results=[
                    claim(atom_by_id[atom_id])
                    for atom_id in section.atom_ids
                    if atom_id in result_ids
                ],
                conditions=[claim(atom) for atom in atoms if atom.kind == "condition"],
                risks=[claim(atom) for atom in atoms if atom.kind == "warning"],
                actions=[claim(atom) for atom in atoms if atom.kind == "recommendation"],
                included_atom_ids=list(section.atom_ids),
                unprovided_items=[],
                generated_interpretations=[],
            )
        )
    return tuple(drafts)


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
                        "atom_ids": _dedupe_text([*current.atom_ids, *section.atom_ids]),
                        "timestamps": sorted({*current.timestamps, *section.timestamps}),
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
    source_segments = [
        _segment_payload(segment) for segment in note_input.segments if segment.index in indexes
    ]
    atom_catalog = [_atom_catalog_entry(atom) for atom in atoms]
    return {
        "prompt_version": PROMPT_VERSIONS["section"],
        "aweme_id": note_input.aweme_id,
        "section": section.model_dump(mode="json"),
        "atoms": atom_catalog,
        "source_segments": source_segments,
        "input_truncated": False,
    }


def _depth_review_payload(
    note_input: NoteInput,
    extraction: ExtractionResult,
    outline: TeachingOutline,
    sections: tuple[SectionDraft, ...],
    *,
    all_sections: tuple[SectionDraft, ...] | None = None,
    compact: bool = False,
    context_tokens: int = 65_536,
    source_segments_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    context_sections = all_sections or sections
    outline_by_id = {section.section_id: section for section in outline.sections}
    section_atom_ids = {atom_id for section in sections for atom_id in section.included_atom_ids}
    for section in sections:
        if section.included_atom_ids:
            continue
        outline_section = outline_by_id.get(section.section_id)
        if outline_section is not None:
            section_atom_ids.update(outline_section.atom_ids)
    atom_by_id = {atom.id: atom for atom in extraction.atoms}
    relation_ids = set(section_atom_ids)
    for atom_id in tuple(section_atom_ids):
        atom = atom_by_id.get(atom_id)
        if atom is None:
            continue
        relation_ids.update(atom.relations)
    target_indexes = {
        atom_by_id[atom_id].source_segment for atom_id in section_atom_ids if atom_id in atom_by_id
    }
    relation_indexes = {atom.source_segment for atom in extraction.atoms if atom.id in relation_ids}
    source_segments = source_segments_override or [
        _segment_payload(segment)
        for segment in note_input.segments
        if segment.index in target_indexes
    ]
    related_source_segments = [
        _segment_payload(segment)
        for segment in note_input.segments
        if segment.index in relation_indexes - target_indexes
    ]
    section_context = [_section_depth_context(item) for item in context_sections]
    atom_relation_context = [
        _atom_depth_context(atom, compact=compact)
        for atom in extraction.atoms
        if atom.id in relation_ids or atom.id in section_atom_ids
    ]
    outline_payload = (
        _compact_depth_outline(outline, set(relation_ids))
        if compact
        else outline.model_dump(mode="json")
    )
    return {
        "prompt_version": PROMPT_VERSIONS["depth_review"],
        "aweme_id": note_input.aweme_id,
        "raw_segments": source_segments,
        "related_source_segments": related_source_segments,
        "outline": outline_payload,
        "section_drafts": [section.model_dump(mode="json") for section in sections],
        "section_context": section_context,
        "atom_relation_context": atom_relation_context,
        "extraction": _project_extraction(
            _subset_extraction(extraction, section_atom_ids, target_indexes),
            include_source_excerpts=False if compact else True,
        ),
        "input_sufficient": bool(note_input.transcript.strip()),
        "input_compacted": compact,
        "input_truncated": False,
    }


def _section_depth_context(section: SectionDraft) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "title": section.title,
        "included_atom_ids": list(section.included_atom_ids),
        "claim_atom_ids": [
            list(claim.atom_ids)
            for field_name in (
                "claims",
                "steps",
                "examples",
                "results",
                "conditions",
                "risks",
                "actions",
            )
            for claim in getattr(section, field_name)
        ],
    }


def _compact_depth_outline(
    outline: TeachingOutline,
    relevant_atom_ids: set[str],
) -> dict[str, Any]:
    sections = []
    for section in outline.sections:
        sections.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "order": section.order,
                "atom_ids": [
                    atom_id for atom_id in section.atom_ids if atom_id in relevant_atom_ids
                ],
                "required_evidence_types": list(section.required_evidence_types),
                "timestamps": list(section.timestamps),
            }
        )
    return {
        "sections": sections,
        "included_atom_ids": [
            atom_id for atom_id in outline.included_atom_ids if atom_id in relevant_atom_ids
        ],
        "unassigned_atom_ids": [],
        "input_sufficient": outline.input_sufficient,
    }


def _atom_depth_context(atom: Any, *, compact: bool) -> dict[str, Any]:
    value = {
        "id": atom.id,
        "kind": atom.kind,
        "source_segment": atom.source_segment,
        "relations": list(atom.relations),
        "sequence": atom.sequence,
        "is_promotional": atom.is_promotional,
        "uncertainty": list(atom.asr_uncertainty),
    }
    if not compact:
        value["statement"] = atom.statement
        value["source_excerpt"] = atom.source_excerpt
    return value


def _split_depth_section(
    section: SectionDraft,
    outline: TeachingOutline,
) -> list[tuple[SectionDraft, ...]]:
    outline_section = next(
        (item for item in outline.sections if item.section_id == section.section_id),
        None,
    )
    atom_ids = list(section.included_atom_ids)
    if not atom_ids and outline_section is not None:
        atom_ids = list(outline_section.atom_ids)
    if len(atom_ids) <= 1:
        return []
    midpoint = max(1, len(atom_ids) // 2)
    return [
        (_project_depth_section(section, set(atom_ids[:midpoint])),),
        (_project_depth_section(section, set(atom_ids[midpoint:])),),
    ]


def _project_depth_section(section: SectionDraft, atom_ids: set[str]) -> SectionDraft:
    fields = (
        "claims",
        "steps",
        "examples",
        "results",
        "conditions",
        "risks",
        "actions",
    )
    updates: dict[str, Any] = {
        "included_atom_ids": list(
            dict.fromkeys(
                item for item in [*section.included_atom_ids, *sorted(atom_ids)] if item in atom_ids
            )
        ),
        "generated_interpretations": [],
    }
    for field_name in fields:
        updates[field_name] = [
            claim for claim in getattr(section, field_name) if set(claim.atom_ids) & atom_ids
        ]
    texts = [claim.text for field_name in fields for claim in updates[field_name]]
    updates["content"] = "\n".join(dict.fromkeys(texts)) or "视频未提供这一类信息。"
    return section.model_copy(update=updates)


def _depth_fragment_payloads(
    note_input: NoteInput,
    extraction: ExtractionResult,
    outline: TeachingOutline,
    sections: tuple[SectionDraft, ...],
    *,
    all_sections: tuple[SectionDraft, ...],
    context_tokens: int,
    fragment_chars: int | None = None,
) -> list[dict[str, Any]]:
    outline_by_id = {section.section_id: section for section in outline.sections}
    fragment_atom_ids: set[str] = set()
    for section in sections:
        outline_section = outline_by_id.get(section.section_id)
        fragment_atom_ids.update(
            section.included_atom_ids
            or (outline_section.atom_ids if outline_section is not None else [])
        )
    indexes = {atom.source_segment for atom in extraction.atoms if atom.id in fragment_atom_ids}
    if not indexes:
        return []
    max_chars = max(32, fragment_chars or int(context_tokens * 0.15))
    compact_sections = tuple(
        _project_depth_section(
            section,
            set(
                section.included_atom_ids or outline_by_id.get(section.section_id).atom_ids
                if outline_by_id.get(section.section_id) is not None
                else section.included_atom_ids
            ),
        )
        for section in sections
    )
    fragments: list[dict[str, Any]] = []
    for segment in note_input.segments:
        if segment.index not in indexes:
            continue
        text = segment.text
        count = max(1, math.ceil(len(text) / max_chars))
        for index in range(count):
            start = index * max_chars
            fragment = _segment_payload(segment)
            fragment["text"] = text[start : start + max_chars]
            fragment["fragment_index"] = index
            fragment["fragment_count"] = count
            fragments.append(fragment)
    payloads = [
        _depth_review_payload(
            note_input,
            extraction,
            outline,
            compact_sections,
            all_sections=all_sections,
            compact=True,
            context_tokens=context_tokens,
            source_segments_override=[fragment],
        )
        for fragment in fragments
    ]
    budget_ok = all(
        bool(
            _provider_request_budget(
                stage="depth_review",
                system_prompt=SYSTEM_PROMPTS["depth_review"],
                payload=payload,
                json_schema=_model_json_schema(DepthReview),
                context_tokens=context_tokens,
            )["within_budget"]
        )
        for payload in payloads
    )
    if budget_ok:
        return payloads
    if max_chars <= 32:
        return []
    return _depth_fragment_payloads(
        note_input,
        extraction,
        outline,
        sections,
        all_sections=all_sections,
        context_tokens=context_tokens,
        fragment_chars=max(32, max_chars // 2),
    )


def _merge_depth_reviews(
    values: list[DepthReview],
    *,
    input_sufficient: bool,
) -> DepthReview:
    if not values:
        return DepthReview(
            abstracted_atom_ids=[],
            example_without_result_atom_ids=[],
            condition_without_scope_atom_ids=[],
            reordered_step_atom_ids=[],
            lost_literal_atom_ids=[],
            lost_warning_or_promotion_atom_ids=[],
            lost_uncertainty_markers=[],
            unsupported_claims=[],
            repair_section_ids=[],
            input_sufficient=input_sufficient,
        )
    list_fields = (
        "abstracted_atom_ids",
        "example_without_result_atom_ids",
        "condition_without_scope_atom_ids",
        "reordered_step_atom_ids",
        "lost_literal_atom_ids",
        "lost_warning_or_promotion_atom_ids",
        "lost_uncertainty_markers",
        "unsupported_claims",
        "repair_section_ids",
        "unknown_atom_ids",
        "unknown_section_ids",
        "non_repairable_findings",
    )
    data = {
        field_name: _dedupe_text(item for value in values for item in getattr(value, field_name))
        for field_name in list_fields
    }
    data["input_sufficient"] = input_sufficient and all(value.input_sufficient for value in values)
    return DepthReview.model_validate(data, strict=True)


def _is_source_preserving_fallback(extraction: ExtractionResult) -> bool:
    return bool(extraction.atoms and extraction.dispositions) and all(
        disposition.reason
        in {
            "lossless source-preserving local fallback",
            "source segment is empty",
        }
        for disposition in extraction.dispositions
    )


def _normalize_source_fallback_review(
    review: CoverageReview,
    extraction: ExtractionResult,
    *,
    aweme_id: str | None = None,
) -> CoverageReview:
    """Remove deterministic fallback/layout labels from provider findings.

    Source-preserving fallback atom IDs and the fixed teaching layout are
    created locally from the source. A model may report those labels or fixed
    missing-evidence text as unsupported claims even though they are directly
    traceable. Keep any remaining free-form finding for the normal quality
    gate to reject.
    """

    if not _is_source_preserving_fallback(extraction):
        return review
    # The canonical fallback deterministically carries every proven source
    # clause into the overview and its source-segment timeline.  Its positional
    # provenance is verified before any review request is sent.  A provider
    # therefore cannot discover a source omission on this path; local models
    # have instead reported the verbatim timeline entries themselves as
    # omissions (for example, every clause of a long REIT transcript).  Keep
    # the structural review for malformed responses, but let the deterministic
    # coverage validator be authoritative for source omission here.
    # Atom IDs are implementation identifiers, while an atom's statement may
    # be the exact source-derived text of a real draft claim.  Do not discard
    # that source text here: the later claim-aware normalization keeps it only
    # when it exactly matches a traceable DraftClaim.
    atom_labels = {atom.id for atom in extraction.atoms}
    layout_labels = set(_TEACHING_SECTION_IDS)
    layout_labels.update(
        {
            "概述",
            "核心要点",
            "方法步骤",
            "示例与结果",
            "边界与风险",
            "行动建议",
            "时间线",
            "证据限制",
        }
    )
    layout_labels.add("视频未提供这一类信息。")
    layout_labels.add("视频原文")
    if aweme_id:
        layout_labels.add(aweme_id)

    def is_structural(value: str) -> bool:
        text = value.strip()
        if not text:
            return True
        if text in atom_labels or text in layout_labels:
            return True
        if any(text.startswith(f"{label}:") for label in layout_labels):
            return True
        return False

    return review.model_copy(
        update={
            "source_omission_candidates": [],
            "unsupported_draft_claims": [
                value for value in review.unsupported_draft_claims if not is_structural(value)
            ]
        }
    )


def _normalize_provider_depth_review(
    extraction: ExtractionResult,
    review: DepthReview,
) -> DepthReview:
    """Let deterministic validation recalculate structural fallback findings."""

    if not _is_source_preserving_fallback(extraction):
        return review
    return review.model_copy(
        update={
            "abstracted_atom_ids": [],
            "example_without_result_atom_ids": [],
            "condition_without_scope_atom_ids": [],
            "reordered_step_atom_ids": [],
            "lost_literal_atom_ids": [],
            "lost_warning_or_promotion_atom_ids": [],
            "lost_uncertainty_markers": [],
            "repair_section_ids": [],
        }
    )


def _validate_depth_review_refs(
    review: DepthReview,
    atom_ids: set[str],
    *,
    section_ids: set[str] | None = None,
) -> DepthReview:
    atom_fields = (
        "abstracted_atom_ids",
        "example_without_result_atom_ids",
        "condition_without_scope_atom_ids",
        "reordered_step_atom_ids",
        "lost_literal_atom_ids",
        "lost_warning_or_promotion_atom_ids",
    )
    unknown = _dedupe_text(
        atom_id
        for field_name in atom_fields
        for atom_id in getattr(review, field_name)
        if atom_id not in atom_ids
    )
    allowed_sections = set(_TEACHING_SECTION_IDS)
    if section_ids is not None:
        allowed_sections &= section_ids
    unknown_sections = _dedupe_text(
        section_id for section_id in review.repair_section_ids if section_id not in allowed_sections
    )
    valid_repair_sections = [
        section_id for section_id in review.repair_section_ids if section_id in allowed_sections
    ]
    if not unknown and not unknown_sections:
        return review
    return review.model_copy(
        update={
            "unknown_atom_ids": _dedupe_text([*review.unknown_atom_ids, *unknown]),
            "non_repairable_findings": _dedupe_text(
                [
                    *review.non_repairable_findings,
                    *(f"unknown depth atom: {atom_id}" for atom_id in unknown),
                    *(f"unknown depth section: {section_id}" for section_id in unknown_sections),
                ]
            ),
            "repair_section_ids": valid_repair_sections,
            "unknown_section_ids": _dedupe_text([*review.unknown_section_ids, *unknown_sections]),
        }
    )


def _normalize_section(
    draft: SectionDraft,
    outline_section: TeachingOutlineSection,
    atom_by_id: Mapping[str, Any],
) -> SectionDraft:
    if draft.section_id != outline_section.section_id:
        raise ValueError(
            f"section draft id mismatch: {draft.section_id} != {outline_section.section_id}"
        )
    allowed = set(outline_section.atom_ids)
    referenced = set(draft.included_atom_ids)
    for field_name in (
        "claims",
        "steps",
        "examples",
        "results",
        "conditions",
        "risks",
        "actions",
    ):
        referenced.update(
            atom_id for claim in getattr(draft, field_name) for atom_id in claim.atom_ids
        )
    unknown = sorted(
        atom_id for atom_id in referenced if atom_id not in atom_by_id or atom_id not in allowed
    )
    if unknown:
        raise ValueError(f"section references unauthorized atoms: {unknown}")

    source_order = {atom_id: index for index, atom_id in enumerate(atom_by_id)}
    included = sorted(
        dict.fromkeys([*draft.included_atom_ids, *outline_section.atom_ids]),
        key=lambda atom_id: source_order[atom_id],
    )
    assigned_atoms = [atom_by_id[atom_id] for atom_id in included]

    def atom_value(atom: Any, name: str) -> Any:
        return atom.get(name) if isinstance(atom, Mapping) else getattr(atom, name)

    searchable = "\n".join(
        [
            draft.content,
            *(
                item.text
                for field_name in (
                    "claims",
                    "steps",
                    "examples",
                    "results",
                    "conditions",
                    "risks",
                    "actions",
                )
                for item in getattr(draft, field_name)
            ),
        ]
    )
    missing_statements = [
        str(atom_value(atom, "statement"))
        for atom in assigned_atoms
        if str(atom_value(atom, "statement")) not in searchable
        and str(atom_value(atom, "source_excerpt")) not in searchable
    ]
    content = draft.content
    generated_interpretations = list(draft.generated_interpretations)
    source_statements = [str(atom_value(atom, "statement")) for atom in assigned_atoms]

    def normalized_text(value: str) -> str:
        return re.sub(r"\s+", "", value).strip()

    source_echoes = {
        *(normalized_text(value) for value in source_statements),
        normalized_text("".join(source_statements)),
    }
    generated_interpretations = [
        value
        for value in generated_interpretations
        if normalized_text(value.removeprefix("视频原文：").removeprefix("原文："))
        not in source_echoes
    ]
    structured_items = [
        item
        for field_name in (
            "claims",
            "steps",
            "examples",
            "results",
            "conditions",
            "risks",
            "actions",
        )
        for item in getattr(draft, field_name)
    ]
    if not included and not structured_items:
        content = outline_section.missing_evidence_policy
        generated_interpretations = []
    if missing_statements:
        normalized_content = re.sub(r"\s+", " ", content).strip().casefold()
        template_values = {
            outline_section.title.strip().casefold(),
            outline_section.section_id.casefold(),
            outline_section.section_id.replace("_", " ").casefold(),
            outline_section.missing_evidence_policy.strip().casefold(),
            "claim",
            "step",
            "example",
            "result",
            "condition",
            "risk",
            "action",
            "核心要点",
            "方法步骤",
            "示例与结果",
            "边界与风险",
            "时间线",
            "视频原文",
        }
        exact_source = "\n".join(dict.fromkeys(source_statements))
        if normalized_content in template_values:
            content = exact_source
        else:
            content = f"{content.rstrip()}\n\n原文证据：\n{exact_source}"
    return draft.model_copy(
        update={
            "section_id": outline_section.section_id,
            "title": outline_section.title,
            "included_atom_ids": included,
            "content": content,
            "generated_interpretations": generated_interpretations,
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
    content_parts = list(dict.fromkeys(item.content for item in values))
    data["content"] = "\n".join(content_parts)
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
        if field_name in {"unprovided_items", "generated_interpretations"}:
            data[field_name] = list(
                dict.fromkeys(item for value in values for item in getattr(value, field_name))
            )
            continue
        seen: set[tuple[Any, ...]] = set()
        merged_items: list[dict[str, Any]] = []
        for value in values:
            for item in getattr(value, field_name):
                item_data = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                key = (
                    item_data.get("text"),
                    tuple(item_data.get("atom_ids", [])),
                    tuple(item_data.get("timestamps", [])),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_items.append(item_data)
        data[field_name] = merged_items
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
    if _is_source_preserving_fallback(extraction):
        return _source_fallback_synthesis(note_input, extraction, outline)
    by_id = {draft.section_id: draft for draft in drafts}
    atom_by_id = {atom.id: atom for atom in extraction.atoms}

    def theme(section_id: str) -> list[ThemeSection]:
        draft = by_id.get(section_id)
        if draft is None or not draft.included_atom_ids:
            return []
        return [
            ThemeSection(
                heading=draft.title,
                content=draft.content,
                atom_ids=list(draft.included_atom_ids),
                timestamps=sorted(
                    {
                        atom.start
                        for atom_id in draft.included_atom_ids
                        if (atom := atom_by_id.get(atom_id)) is not None and atom.start is not None
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
    source_fallback = _is_source_preserving_fallback(extraction)
    promotion_sections = [
        ThemeSection(
            heading=atom.topic,
            content=atom.statement,
            atom_ids=[atom.id],
            timestamps=[value for value in (atom.start, atom.end) if value is not None],
        )
        for atom in extraction.atoms
        if atom.kind == "promotion" or atom.is_promotional
    ]
    return SynthesisResult(
        title=note_input.description or note_input.aweme_id,
        overview=overview.content if overview else "视频未提供这一类信息。",
        structure=[section.title for section in outline.sections],
        # The outline may legitimately assign evidence to the overview,
        # actions, timeline, or evidence-limits chapter instead of
        # ``core_points``.  Each populated teaching chapter must remain in the
        # traceable thematic projection; otherwise valid evidence can never
        # satisfy the existing coverage gate merely because of its chapter.
        themes=[
            *theme("overview"),
            *theme("core_points"),
            *theme("actions"),
            *theme("timeline"),
            *theme("evidence_limits"),
        ],
        methods_and_steps=theme("methods_steps"),
        examples_and_counterexamples=theme("examples_results"),
        boundaries_and_risks=theme("boundaries_risks"),
        concepts=_dedupe_text(atom.topic for atom in extraction.atoms if atom.kind == "definition"),
        quotes=_dedupe_text(atom.statement for atom in extraction.atoms if atom.kind == "quote"),
        implications=(
            []
            if source_fallback
            else _dedupe_text(
                value if value.startswith("生成理解：") else f"生成理解：{value}"
                for draft in drafts
                for value in draft.generated_interpretations
                if value != "视频未提供这一类信息。"
            )
        ),
        actions=_dedupe_text(item.text for draft in drafts for item in draft.actions),
        timeline=timeline,
        promotions=promotion_sections,
        tags=[],
        included_atom_ids=[atom.id for atom in extraction.atoms],
    )


def _source_fallback_synthesis(
    note_input: NoteInput,
    extraction: ExtractionResult,
    outline: TeachingOutline,
) -> SynthesisResult:
    atom_by_id = {atom.id: atom for atom in extraction.atoms}
    outline_by_id = {section.section_id: section for section in outline.sections}

    def theme(atom: Any) -> ThemeSection:
        return ThemeSection(
            heading=atom.topic,
            content=atom.statement,
            atom_ids=[atom.id],
            timestamps=[atom.start] if atom.start is not None else [],
        )

    def atoms_for(section_id: str) -> list[Any]:
        section = outline_by_id[section_id]
        return [atom_by_id[atom_id] for atom_id in section.atom_ids]

    overview_atoms = atoms_for("overview")
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
        for disposition in sorted(
            extraction.dispositions,
            key=lambda item: item.segment_index,
        )
    ]
    return SynthesisResult(
        title=note_input.description or note_input.aweme_id,
        overview=("；".join(atom.statement for atom in overview_atoms) or "视频未提供这一类信息。"),
        structure=[section.title for section in outline.sections],
        themes=[theme(atom) for atom in atoms_for("core_points")],
        methods_and_steps=[
            theme(atom) for atom in atoms_for("methods_steps") if atom.kind == "step"
        ],
        examples_and_counterexamples=[
            theme(atom)
            for atom in atoms_for("examples_results")
            if atom.kind in {"example", "counterexample"}
        ],
        boundaries_and_risks=[
            theme(atom)
            for atom in atoms_for("boundaries_risks")
            if atom.kind in {"condition", "warning"}
        ],
        concepts=_dedupe_text(atom.topic for atom in extraction.atoms if atom.kind == "definition"),
        quotes=_dedupe_text(atom.statement for atom in extraction.atoms if atom.kind == "quote"),
        implications=[],
        actions=_dedupe_text(
            atom.statement for atom in extraction.atoms if atom.kind == "recommendation"
        ),
        timeline=timeline,
        promotions=[
            theme(atom)
            for atom in extraction.atoms
            if atom.kind == "promotion" or atom.is_promotional
        ],
        tags=_dedupe_text(atom.topic for atom in extraction.atoms),
        included_atom_ids=[atom.id for atom in extraction.atoms],
    )


def _atom_ids_for_source_batch(
    extraction: ExtractionResult,
    batch: SegmentBatch,
) -> set[str]:
    """Return the evidence atoms wholly represented by a review fragment.

    Provider-produced atoms predate positional provenance, so their source
    excerpts retain the former membership check.  Canonical fallback atoms do
    carry a source position; that coordinate is required whenever repeated
    text makes a substring match ambiguous.
    """

    selected: set[str] = set()
    use_positions = _is_source_preserving_fallback(extraction)
    for atom in extraction.atoms:
        if not atom.source_excerpt.strip():
            continue
        for segment in batch.segments:
            if segment.index != atom.source_segment:
                continue
            if not use_positions or atom.source_start is None:
                if atom.source_excerpt in segment.text:
                    selected.add(atom.id)
                continue
            fragment_start = segment.source_offset
            fragment_end = fragment_start + len(segment.text)
            atom_end = atom.source_start + len(atom.source_excerpt)
            if fragment_start <= atom.source_start and atom_end <= fragment_end:
                selected.add(atom.id)
    return selected


def _source_review_payload(
    note_input: NoteInput,
    extraction: ExtractionResult,
    synthesis: SynthesisResult,
    batch: SegmentBatch,
    review_pass: int,
) -> dict[str, Any]:
    segment_indexes = {segment.index for segment in batch.segments}
    review_atom_ids = _atom_ids_for_source_batch(extraction, batch)
    atom_ids = {
        atom_id
        for entry in synthesis.timeline
        if entry.segment_index in segment_indexes
        for atom_id in entry.atom_ids
        if atom_id in review_atom_ids
    }
    return {
        "prompt_version": PROMPT_VERSIONS["review"],
        "aweme_id": note_input.aweme_id,
        "review_pass": review_pass,
        "review_mode": "source_coverage",
        "source_segments": [_segment_payload(segment) for segment in batch.segments],
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
                extraction,
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
    review_atom_ids = _atom_ids_for_source_batch(extraction, batch)
    return {
        "prompt_version": PROMPT_VERSIONS["review"],
        "aweme_id": note_input.aweme_id,
        "review_pass": review_pass,
        "review_mode": "draft_claims",
        "source_segments": [_segment_payload(segment) for segment in batch.segments],
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
    outline: TeachingOutline | None = None,
    section_drafts: tuple[SectionDraft, ...] = (),
    depth_review: DepthReview | None = None,
    repair_section_ids: list[str] | None = None,
) -> dict[str, Any]:
    relevant_ids = {
        atom_id
        for entry in synthesis.timeline
        if entry.segment_index in relevant_segments
        for atom_id in entry.atom_ids
    }
    draft = synthesis
    if compact_content:
        draft = _slice_synthesis(
            synthesis,
            relevant_segments,
            relevant_ids,
            extraction,
        )
    draft_payload = _project_synthesis(
        draft,
        extraction,
        compact_content=compact_content,
    )
    return {
        "prompt_version": PROMPT_VERSIONS["repair"],
        "aweme_id": note_input.aweme_id,
        "source_segments": [
            _segment_payload(segment)
            for segment in note_input.segments
            if segment.index in relevant_segments
        ],
        "extraction": _project_extraction(repair_extraction),
        "draft": draft_payload,
        "review": _slice_review(
            review,
            relevant_segments,
        ).model_dump(mode="json"),
        "coverage_report": _slice_coverage_report(
            coverage,
            repair_extraction,
            relevant_segments,
        ),
        "outline": outline.model_dump(mode="json") if outline else None,
        "section_drafts": [section.model_dump(mode="json") for section in section_drafts],
        "depth_review": (depth_review.model_dump(mode="json") if depth_review else None),
        "repair_section_ids": list(repair_section_ids or []),
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
        extraction_payload.get("atoms", []) if isinstance(extraction_payload, Mapping) else []
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
        if "content" not in item and isinstance(item.get("content_from_atom_ids"), list):
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
            if "statement" not in atom and atom.pop("source_excerpt_is_statement", False):
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
                {key: value for key, value in payload.items() if key != "context_budget"},
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
    disposition_by_index = {item.segment_index: item for item in extraction.dispositions}
    included_ids = [atom.id for atom in extraction.atoms]
    timeline: list[dict[str, Any]] = []
    reason = str(failure.get("stage")) if failure is not None else "unknown"
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
    # The provider reports a token context window, not a byte window.  UTF-8
    # bytes over-count Chinese evidence by roughly 3x, causing local-only
    # repairs to be rejected before Ollama sees them.  The exact Qwen
    # tokenizer is intentionally not a runtime dependency, so use a
    # conservative character-class estimate with a safety margin instead.
    system_prompt_tokens = _cached_static_prompt_tokens(system_prompt)
    payload_tokens = _estimate_prompt_tokens(
        json.dumps(
            payload_without_budget,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    json_schema_tokens = _cached_static_prompt_tokens(
        json.dumps(
            json_schema,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    budget_metadata_tokens = 512
    output_floor, output_ratio = {
        "extract": (1_024, 2.0),
        "outline": (1_024, 1.0),
        "synthesize": (1_024, 1.5),
        "section": (1_024, 1.5),
        "review": (768, 0.75),
        "depth_review": (1_024, 1.0),
        "repair": (2_048, 1.5),
    }[stage]
    reserved_output_tokens = max(
        output_floor,
        math.ceil(payload_tokens * output_ratio),
    )
    estimated_input_tokens = (
        system_prompt_tokens + payload_tokens + json_schema_tokens + budget_metadata_tokens
    )
    estimated_total_tokens = estimated_input_tokens + reserved_output_tokens
    return {
        "method": "provider-request-v3-charclass",
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


def _estimate_prompt_tokens(value: str) -> int:
    """Conservatively estimate Qwen-style prompt tokens without cloud calls.

    CJK characters usually map close to one token, while ASCII prose and JSON
    keys pack several characters into a token.  Weighting punctuation more
    heavily and applying a 15% margin keeps the estimate on the safe side
    without treating UTF-8's three bytes per Han character as three tokens.
    """

    units = 0.0
    for character in value:
        codepoint = ord(character)
        if character.isspace():
            units += 0.15
        elif (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x3040 <= codepoint <= 0x30FF
            or 0xAC00 <= codepoint <= 0xD7AF
        ):
            units += 1.0
        elif character.isascii() and character.isalnum():
            units += 0.35
        elif character.isascii():
            units += 0.5
        else:
            units += 0.9
    return max(1, math.ceil(units * 1.15))


@lru_cache(maxsize=64)
def _cached_static_prompt_tokens(value: str) -> int:
    """Estimate recurring system-prompt and schema tokens once per process."""

    return _estimate_prompt_tokens(value)


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
            atom.source_segment for atom in extraction.atoms if atom.id in selected_ids
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
        result.append(_subset_extraction(extraction, atom_ids, segment_indexes))
    return result


def _subset_extraction(
    extraction: ExtractionResult,
    atom_ids: set[str],
    segment_indexes: set[int],
) -> ExtractionResult:
    atoms = [atom.model_dump(mode="json") for atom in extraction.atoms if atom.id in atom_ids]
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
        sum(max(1, math.ceil(len(atom.source_excerpt) / 1.5)) for atom in extraction.atoms),
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
        "structure": _dedupe_text(item for value in values for item in value.structure),
        "concepts": _dedupe_text(item for value in values for item in value.concepts),
        "quotes": _dedupe_text(item for value in values for item in value.quotes),
        "implications": _dedupe_text(item for value in values for item in value.implications),
        "actions": _dedupe_text(item for value in values for item in value.actions),
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
            timeline.setdefault(entry.segment_index, []).append(entry.model_dump(mode="json"))
    merged["timeline"] = [
        _merge_timeline_entries(entries) for _, entries in sorted(timeline.items())
    ]
    return SynthesisResult.model_validate(merged, strict=True)


def _merge_timeline_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    first = entries[0]
    atom_ids = _dedupe_text(atom_id for entry in entries for atom_id in entry["atom_ids"])
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
        "content": "；".join(_dedupe_text(entry["content"] for entry in entries)),
    }


def _slice_synthesis(
    synthesis: SynthesisResult,
    segment_indexes: set[int],
    atom_ids: set[str],
    extraction: ExtractionResult,
) -> SynthesisResult:
    atom_text = {
        atom.id: (atom.source_excerpt if atom.statement == atom.source_excerpt else atom.statement)
        for atom in extraction.atoms
    }

    def sections(field_name: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for section in getattr(synthesis, field_name):
            selected_ids = [atom_id for atom_id in section.atom_ids if atom_id in atom_ids]
            if not selected_ids:
                continue
            data = section.model_dump(mode="json")
            data["atom_ids"] = selected_ids
            result.append(data)
        return result

    timeline = []
    for entry in synthesis.timeline:
        if entry.segment_index not in segment_indexes:
            continue
        selected_ids = [atom_id for atom_id in entry.atom_ids if atom_id in atom_ids]
        if not selected_ids:
            continue
        data = entry.model_dump(mode="json")
        data["atom_ids"] = selected_ids
        data["content"] = "；".join(
            atom_text[atom_id] for atom_id in selected_ids if atom_id in atom_text
        )
        timeline.append(data)
    return SynthesisResult.model_validate(
        {
            "title": synthesis.title,
            "overview": "仅包含本批源分段的局部草稿，用于分层审查。",
            "structure": [entry["heading"] for entry in sections("themes")],
            "themes": sections("themes"),
            "methods_and_steps": sections("methods_and_steps"),
            "examples_and_counterexamples": sections("examples_and_counterexamples"),
            "boundaries_and_risks": sections("boundaries_and_risks"),
            "concepts": [],
            "quotes": [],
            "implications": [],
            "actions": [],
            "timeline": timeline,
            "promotions": sections("promotions"),
            "tags": [],
            "included_atom_ids": [
                atom_id for atom_id in synthesis.included_atom_ids if atom_id in atom_ids
            ],
        },
        strict=True,
    )


def _claims_for_source_batch[T: DraftClaim | Mapping[str, Any]](
    claims: list[T],
    extraction: ExtractionResult,
    batch: SegmentBatch,
) -> list[T]:
    relevant_atom_ids = _atom_ids_for_source_batch(extraction, batch)
    has_positional_fallback = any(atom.source_start is not None for atom in extraction.atoms)
    selected: list[T] = []
    for claim in claims:
        if isinstance(claim, Mapping):
            atom_ids = set(claim.get("atom_ids", []))
        else:
            atom_ids = set(claim.atom_ids)
        if atom_ids & relevant_atom_ids:
            selected.append(claim)
        elif (
            not atom_ids
            and (
                any(segment.index == 0 and segment.source_offset == 0 for segment in batch.segments)
                if has_positional_fallback
                else batch.batch_index == 0
            )
        ):
            selected.append(claim)
    return selected


def _draft_claims(synthesis: SynthesisResult) -> list[DraftClaim]:
    claims: list[DraftClaim] = []
    evidence_sections = [
        section
        for field_name in (
            "themes",
            "methods_and_steps",
            "examples_and_counterexamples",
            "boundaries_and_risks",
            "promotions",
        )
        for section in getattr(synthesis, field_name)
    ]

    def inferred_atom_ids(
        text: str,
        *,
        aggregate_content: bool = False,
        match_heading: bool = False,
    ) -> list[str]:
        normalized = text.strip()
        if not normalized or normalized == "视频未提供这一类信息。":
            return []
        matched: set[str] = set()
        for section in evidence_sections:
            content = section.content.strip()
            content_matches = content == normalized or (
                aggregate_content and content != "视频未提供这一类信息。" and content in normalized
            )
            if content_matches or (match_heading and section.heading.strip() == normalized):
                matched.update(section.atom_ids)
        return [atom_id for atom_id in synthesis.included_atom_ids if atom_id in matched]

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
    add(
        "overview",
        synthesis.overview,
        inferred_atom_ids(synthesis.overview, aggregate_content=True),
    )
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
            atom_ids = (
                inferred_atom_ids(item, match_heading=True)
                if field_name == "concepts"
                else inferred_atom_ids(item)
                if field_name in {"quotes", "actions"}
                else []
            )
            add(field_name, item, atom_ids)
    for entry in synthesis.timeline:
        add("timeline", entry.content, list(entry.atom_ids))
    return claims


def _claim_review_unsupported_ids(
    claims: list[DraftClaim | Mapping[str, Any]],
    *,
    reviewed_ids: set[str],
    supported_ids: set[str],
    contradicted_ids: set[str],
) -> set[str]:
    """Classify claim review results without treating layout labels as facts."""

    unsupported: set[str] = set()
    for claim in claims:
        claim_id = str(claim.get("id") if isinstance(claim, Mapping) else claim.id)
        field_name = str(claim.get("field") if isinstance(claim, Mapping) else claim.field)
        text = str(claim.get("text") if isinstance(claim, Mapping) else claim.text)
        atom_ids = list(claim.get("atom_ids", []) if isinstance(claim, Mapping) else claim.atom_ids)
        if claim_id in contradicted_ids or claim_id not in reviewed_ids:
            unsupported.add(claim_id)
            continue
        if claim_id in supported_ids:
            continue
        editorial = field_name in {"title", "structure", "tags"} and not atom_ids
        labeled_interpretation = field_name == "implications" and text.strip().startswith(
            "生成理解："
        )
        if not editorial and not labeled_interpretation:
            unsupported.add(claim_id)
    return unsupported


def _normalize_source_fallback_unsupported_claims(
    extraction: ExtractionResult,
    claims: list[DraftClaim],
    review: CoverageReview,
) -> CoverageReview:
    """Keep only reviewable fallback draft text from source-coverage feedback."""

    if not _is_source_preserving_fallback(extraction):
        return review
    atom_ids = {atom.id for atom in extraction.atoms}
    reviewable_text = {claim.text for claim in claims if claim.atom_ids}
    return review.model_copy(
        update={
            "unsupported_draft_claims": [
                value
                for value in review.unsupported_draft_claims
                if value not in atom_ids and value in reviewable_text
            ]
        }
    )


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
            math.ceil((len(claim.text) + sum(len(atom_id) for atom_id in claim.atom_ids)) / 1.5),
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
        internally_duplicated.update(index for index, count in counts.items() if count > 1)
        for index in value.reviewed_segment_indexes:
            reviewed_indexes_by_segment.setdefault(index, []).append(index)
        for item in value.source_omission_candidates:
            omission_by_key[(item.segment_index, item.source_excerpt, item.missing_statement)] = (
                item.model_dump(mode="json")
            )
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
                claim for value in values for claim in value.unsupported_draft_claims
            ),
            "disposition_corrections": list(correction_by_segment.values()),
        },
        strict=True,
    )


def _remove_noop_disposition_corrections(
    review: CoverageReview,
    extraction: ExtractionResult,
) -> CoverageReview:
    current = {disposition.segment_index: disposition for disposition in extraction.dispositions}
    meaningful = []
    for correction in review.disposition_corrections:
        existing = current.get(correction.segment_index)
        if (
            existing is not None
            and correction.disposition == existing.disposition
            and (
                not correction.atom_ids
                or Counter(correction.atom_ids) == Counter(existing.atom_ids)
            )
        ):
            continue
        meaningful.append(correction)
    return review.model_copy(update={"disposition_corrections": meaningful})


def _fragment_segment_indexes(batches: list[SegmentBatch]) -> set[int]:
    counts = Counter(segment.index for batch in batches for segment in batch.segments)
    return {index for index, count in counts.items() if count > 1}


def _slice_review(
    review: CoverageReview,
    segment_indexes: set[int],
) -> CoverageReview:
    return CoverageReview.model_validate(
        {
            "reviewed_segment_indexes": [
                index for index in review.reviewed_segment_indexes if index in segment_indexes
            ],
            "source_omission_candidates": [
                item.model_dump(mode="json")
                for item in review.source_omission_candidates
                if item.segment_index in segment_indexes
            ],
            "unsupported_draft_claims": list(review.unsupported_draft_claims),
            "disposition_corrections": [
                item.model_dump(mode="json")
                for item in review.disposition_corrections
                if item.segment_index in segment_indexes
            ],
        },
        strict=True,
    )


def _slice_coverage_report(
    coverage: CoverageReport,
    extraction: ExtractionResult,
    segment_indexes: set[int],
) -> dict[str, Any]:
    """Project coverage instructions to evidence owned by one repair unit."""

    atom_ids = {atom.id for atom in extraction.atoms}
    data = coverage.model_dump(mode="json", by_alias=True)
    missing_atom_ids = [
        atom_id for atom_id in coverage.missing_atom_ids if atom_id in atom_ids
    ]
    data.update(
        {
            "segment_total": len(segment_indexes),
            "segment_disposition_count": sum(
                item.segment_index in segment_indexes
                for item in extraction.dispositions
            ),
            "missing_segment_indexes": [
                index
                for index in coverage.missing_segment_indexes
                if index in segment_indexes
            ],
            "evidence_atom_total": len(atom_ids),
            "evidence_atom_included": max(0, len(atom_ids) - len(missing_atom_ids)),
            "missing_atom_ids": missing_atom_ids,
            "traceable_atom_count": max(0, len(atom_ids) - len(missing_atom_ids)),
            "source_segments_independently_reviewed": min(
                coverage.source_segments_independently_reviewed,
                len(segment_indexes),
            ),
            "source_omission_candidates": [
                item.model_dump(mode="json")
                for item in coverage.source_omission_candidates
                if item.segment_index in segment_indexes
            ],
            # Draft claims and the depth result are scoped by their dedicated
            # local payload fields.  Keeping the global copies here makes a
            # local repair request grow with unrelated sections.
            "unsupported_draft_claims": [],
            "depth_review": None,
        }
    )
    return data


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
    relevant_segments = {item.segment_index for item in review.source_omission_candidates}
    relevant_segments.update(coverage.missing_segment_indexes)
    timeline_indexes = {entry.segment_index for entry in synthesis.timeline}
    relevant_segments.update(
        item.segment_index
        for item in extraction.dispositions
        if item.segment_index not in timeline_indexes
    )
    if not coverage.promotional_content_preserved:
        relevant_ids.update(
            atom.id for atom in extraction.atoms if atom.kind == "promotion" or atom.is_promotional
        )
    if review.unsupported_draft_claims:
        unsupported_claims = set(review.unsupported_draft_claims)
        claims_by_text: dict[str, list[DraftClaim]] = {}
        for claim in _draft_claims(synthesis):
            claims_by_text.setdefault(claim.text, []).append(claim)
        matching_claims = [
            claim
            for text in unsupported_claims
            for claim in claims_by_text.get(text, [])
        ]
        if (
            len({claim.text for claim in matching_claims}) != len(unsupported_claims)
            or any(not claim.atom_ids for claim in matching_claims)
        ):
            relevant_ids.update(atom_by_id)
        else:
            relevant_ids.update(
                atom_id for claim in matching_claims for atom_id in claim.atom_ids
            )
    relevant_ids.update(
        atom.id for atom in extraction.atoms if atom.source_segment in relevant_segments
    )
    related_ids: set[str] = set()
    for atom_id in tuple(relevant_ids):
        atom = atom_by_id.get(atom_id)
        if atom is not None:
            related_ids.update(
                relation_id for relation_id in atom.relations if relation_id in atom_by_id
            )
    relevant_ids.update(related_ids)
    if not relevant_ids and extraction.atoms:
        relevant_ids.update(atom_by_id)
    relevant_segments.update(
        atom_by_id[atom_id].source_segment for atom_id in relevant_ids if atom_id in atom_by_id
    )
    return (
        _subset_extraction(extraction, relevant_ids, relevant_segments),
        relevant_segments,
    )


def _depth_repair_slice(
    extraction: ExtractionResult,
    sections: tuple[SectionDraft, ...],
    depth_review: DepthReview,
    *,
    outline: TeachingOutline | None = None,
) -> tuple[ExtractionResult, set[int]]:
    flagged = set(depth_review.repair_section_ids)
    atom_ids = {
        atom_id
        for section in sections
        if section.section_id in flagged
        for atom_id in section.included_atom_ids
    }
    if outline is not None:
        atom_ids.update(
            atom_id
            for section in outline.sections
            if section.section_id in flagged
            for atom_id in section.atom_ids
        )
    atom_ids.update(depth_review.abstracted_atom_ids)
    atom_ids.update(depth_review.example_without_result_atom_ids)
    atom_ids.update(depth_review.condition_without_scope_atom_ids)
    atom_ids.update(depth_review.reordered_step_atom_ids)
    atom_ids.update(depth_review.lost_literal_atom_ids)
    atom_ids.update(
        atom_id for atom_id in depth_review.lost_warning_or_promotion_atom_ids if atom_id
    )
    atom_by_id = {atom.id: atom for atom in extraction.atoms}
    uncertainty_markers = set(depth_review.lost_uncertainty_markers)
    atom_ids.update(
        atom.id
        for atom in extraction.atoms
        if uncertainty_markers.intersection(atom.asr_uncertainty)
    )
    atom_ids = {atom_id for atom_id in atom_ids if atom_id in atom_by_id}
    relevant_segments = {atom_by_id[atom_id].source_segment for atom_id in atom_ids}
    for atom_id in tuple(atom_ids):
        relevant_segments.update(
            relation.source_segment
            for relation in atom_by_id.values()
            if relation.id in atom_by_id[atom_id].relations
        )
    return _subset_extraction(extraction, atom_ids, relevant_segments), relevant_segments


def _coverage_repair_section_ids(
    outline: TeachingOutline,
    sections: tuple[SectionDraft, ...],
    repair_extraction: ExtractionResult,
    relevant_segments: set[int],
) -> list[str]:
    """Return the teaching sections authorized by a coverage-only repair."""

    repair_atom_ids = {atom.id for atom in repair_extraction.atoms}
    repair_section_ids: list[str] = []
    for section in sections:
        if set(section.included_atom_ids) & repair_atom_ids:
            repair_section_ids.append(section.section_id)
            continue
        outline_section = next(
            (
                candidate
                for candidate in outline.sections
                if candidate.section_id == section.section_id
            ),
            None,
        )
        if outline_section is None:
            continue
        if set(outline_section.atom_ids) & repair_atom_ids:
            repair_section_ids.append(section.section_id)
            continue
        if any(
            atom.source_segment in relevant_segments
            for atom in repair_extraction.atoms
            if atom.id in outline_section.atom_ids
        ):
            repair_section_ids.append(section.section_id)
    if repair_section_ids:
        return list(dict.fromkeys(repair_section_ids))
    # A source omission may introduce a brand-new atom, so no existing draft
    # can identify its eventual section. Keep the bounded repair target at the
    # evidence-limits chapter rather than authorizing every chapter.
    return [
        section.section_id
        for section in outline.sections
        if section.section_id == "evidence_limits"
    ]


def _merge_repair_extraction_patch(
    extraction: ExtractionResult,
    patch: ExtractionResult,
    relevant_segments: set[int],
) -> ExtractionResult:
    if any(atom.source_segment not in relevant_segments for atom in patch.atoms):
        raise ValueError("repair atom is outside the authorized source segments")
    if any(item.segment_index not in relevant_segments for item in patch.dispositions):
        raise ValueError("repair disposition is outside the authorized source segments")

    atoms = {atom.id: atom.model_dump(mode="json") for atom in extraction.atoms}
    for atom in patch.atoms:
        existing = atoms.get(atom.id)
        if existing is not None and existing["source_segment"] not in relevant_segments:
            raise ValueError("repair atom ID collides outside the authorized segments")
        atoms[atom.id] = atom.model_dump(mode="json")

    allowed_ids = set(atoms)
    for atom in patch.atoms:
        if any(relation_id not in allowed_ids for relation_id in atom.relations):
            raise ValueError("repair atom references an unknown atom")

    dispositions = {
        item.segment_index: item.model_dump(mode="json") for item in extraction.dispositions
    }
    for item in patch.dispositions:
        if any(atom_id not in allowed_ids for atom_id in item.atom_ids):
            raise ValueError("repair disposition references an unknown atom")
        if any(atoms[atom_id]["source_segment"] != item.segment_index for atom_id in item.atom_ids):
            raise ValueError("repair disposition references another segment")
        dispositions[item.segment_index] = item.model_dump(mode="json")

    return ExtractionResult.model_validate(
        {
            "atoms": list(atoms.values()),
            "dispositions": [dispositions[index] for index in sorted(dispositions)],
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
        patch_atom_ids.update(atom_id for entry in patch.timeline for atom_id in entry.atom_ids)
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
        ] + [section.model_dump(mode="json") for section in getattr(patch, field_name)]
    data["timeline"] = [
        entry.model_dump(mode="json")
        for entry in synthesis.timeline
        if entry.segment_index not in relevant_segments
    ] + [entry.model_dump(mode="json") for entry in patch.timeline]
    data["timeline"].sort(key=lambda entry: entry["segment_index"])
    data["included_atom_ids"] = _dedupe_text(
        [atom_id for atom_id in synthesis.included_atom_ids if atom_id not in relevant_ids]
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
    prepared: list[tuple[SegmentBatch, ExtractionResult, dict[str, str], bool]] = []
    id_occurrences: dict[str, list[str]] = {}
    for batch, extraction in values:
        global_indexes = {segment.index for segment in batch.segments}
        returned_indexes = [
            *(atom.source_segment for atom in extraction.atoms),
            *(item.segment_index for item in extraction.dispositions),
        ]
        uses_local_indexes = (
            bool(returned_indexes)
            and all(0 <= value < len(batch.segments) for value in returned_indexes)
            and any(value not in global_indexes for value in returned_indexes)
        )
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
        old_id: new_ids[0] for old_id, new_ids in id_occurrences.items() if len(new_ids) == 1
    }
    for batch, extraction, id_map, uses_local_indexes in prepared:
        for atom in extraction.atoms:
            data = atom.model_dump(mode="json")
            data["id"] = id_map[atom.id]
            data["source_segment"] = _global_segment_index(
                atom.source_segment, batch, uses_local_indexes
            )
            data["relations"] = [
                id_map.get(value, unique_global_ids.get(value, value)) for value in atom.relations
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
    reasons = list(dict.fromkeys(item["reason"] for item in items if item["reason"]))
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
