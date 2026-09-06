from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

AtomKind = Literal[
    "claim",
    "reason",
    "definition",
    "step",
    "example",
    "counterexample",
    "condition",
    "warning",
    "number",
    "quote",
    "recommendation",
    "promotion",
]
DispositionKind = Literal[
    "included",
    "duplicate",
    "transition",
    "promotion",
    "unintelligible_asr",
]
DraftClaimField = Literal[
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
]
QualityGate = Literal["passed", "failed", "fallback"]
TeachingSectionId = Literal[
    "overview",
    "core_points",
    "methods_steps",
    "examples_results",
    "boundaries_risks",
    "actions",
    "timeline",
    "evidence_limits",
]
EvidenceRequirement = Literal[
    "claim",
    "reason",
    "step",
    "example",
    "result",
    "condition",
    "risk",
    "number",
    "named_entity",
    "uncertainty",
    "promotion",
    "action",
]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_STRICT_MODEL_CONFIG = ConfigDict(
    allow_inf_nan=False,
    extra="forbid",
    strict=True,
)


class EvidenceAtom(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    id: NonEmptyString
    kind: AtomKind
    topic: NonEmptyString
    statement: NonEmptyString
    source_segment: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    source_excerpt: NonEmptyString
    relations: list[NonEmptyString] = Field(default_factory=list)
    sequence: int | None = Field(default=None, ge=1)
    is_promotional: bool = False
    asr_uncertainty: list[NonEmptyString] = Field(default_factory=list)


class SegmentDisposition(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    segment_index: int = Field(ge=0)
    disposition: DispositionKind
    atom_ids: list[NonEmptyString] = Field(default_factory=list)
    reason: NonEmptyString | None = None


class ExtractionResult(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    atoms: list[EvidenceAtom]
    dispositions: list[SegmentDisposition]


class ThemeSection(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    heading: NonEmptyString
    content: NonEmptyString
    atom_ids: list[NonEmptyString]
    timestamps: list[NonNegativeFloat] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    segment_index: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    disposition: DispositionKind
    atom_ids: list[NonEmptyString]
    content: NonEmptyString


class SynthesisResult(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    title: NonEmptyString
    overview: NonEmptyString
    structure: list[NonEmptyString]
    themes: list[ThemeSection]
    methods_and_steps: list[ThemeSection]
    examples_and_counterexamples: list[ThemeSection]
    boundaries_and_risks: list[ThemeSection]
    concepts: list[NonEmptyString]
    quotes: list[NonEmptyString]
    implications: list[NonEmptyString]
    actions: list[NonEmptyString]
    timeline: list[TimelineEntry]
    promotions: list[ThemeSection]
    tags: list[NonEmptyString]
    included_atom_ids: list[NonEmptyString]


class RepairResult(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    extraction_patch: ExtractionResult
    synthesis_patch: SynthesisResult


class DraftClaim(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    id: NonEmptyString
    field: DraftClaimField
    text: NonEmptyString
    atom_ids: list[NonEmptyString]


class DraftClaimReview(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    reviewed_claim_ids: list[NonEmptyString]
    supported_claim_ids: list[NonEmptyString]
    contradicted_claim_ids: list[NonEmptyString]


class SourceOmissionCandidate(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    segment_index: int = Field(ge=0)
    source_excerpt: NonEmptyString
    missing_statement: NonEmptyString
    suggested_kind: AtomKind


class CoverageReview(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    reviewed_segment_indexes: list[NonNegativeInt]
    source_omission_candidates: list[SourceOmissionCandidate]
    unsupported_draft_claims: list[NonEmptyString]
    disposition_corrections: list[SegmentDisposition]


class ProviderUsage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    api_cost: float | None = Field(default=None, ge=0)
    cost_status: Literal[
        "zero_local",
        "reported",
        "estimated_from_config",
        "unknown",
    ]


class CoverageReport(BaseModel):
    model_config = ConfigDict(
        **_STRICT_MODEL_CONFIG,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    schema_name: Literal["douyin-note-coverage-v1"] = Field(
        default="douyin-note-coverage-v1",
        alias="schema",
    )
    segment_total: int = Field(ge=0)
    segment_disposition_count: int = Field(ge=0)
    missing_segment_indexes: list[NonNegativeInt]
    evidence_atom_total: int = Field(ge=0)
    evidence_atom_included: int = Field(ge=0)
    missing_atom_ids: list[NonEmptyString]
    traceable_atom_count: int = Field(ge=0)
    source_segments_independently_reviewed: int = Field(ge=0)
    source_omission_candidates: list[SourceOmissionCandidate]
    unsupported_draft_claims: list[NonEmptyString]
    steps_verified: bool
    examples_verified: bool
    conditions_verified: bool
    numbers_verified: bool = True
    named_entities_verified: bool = True
    uncertainty_markers_verified: bool = True
    promotional_content_preserved: bool
    semantic_generator: NonEmptyString
    quality_gate: QualityGate
    repairable: bool
    depth_passed: bool = False
    input_sufficient: bool = False
    depth_review: dict[str, Any] | None = None
    semantic_complete: bool = False


class TeachingOutlineSection(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    section_id: TeachingSectionId
    title: NonEmptyString
    order: int = Field(ge=1)
    atom_ids: list[NonEmptyString]
    required_evidence_types: list[EvidenceRequirement]
    missing_evidence_policy: Literal["视频未提供这一类信息。"]
    timestamps: list[NonNegativeFloat] = Field(default_factory=list)


class TeachingOutline(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    sections: list[TeachingOutlineSection]
    included_atom_ids: list[NonEmptyString]
    unassigned_atom_ids: list[NonEmptyString] = Field(default_factory=list)
    input_sufficient: bool = True


class SectionClaim(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    text: NonEmptyString
    atom_ids: list[NonEmptyString] = Field(min_length=1)
    timestamps: list[NonNegativeFloat] = Field(min_length=1)


class SectionDraft(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    section_id: TeachingSectionId
    title: NonEmptyString
    content: NonEmptyString
    claims: list[SectionClaim] = Field(default_factory=list)
    steps: list[SectionClaim] = Field(default_factory=list)
    examples: list[SectionClaim] = Field(default_factory=list)
    results: list[SectionClaim] = Field(default_factory=list)
    conditions: list[SectionClaim] = Field(default_factory=list)
    risks: list[SectionClaim] = Field(default_factory=list)
    actions: list[SectionClaim] = Field(default_factory=list)
    included_atom_ids: list[NonEmptyString]
    unprovided_items: list[NonEmptyString] = Field(default_factory=list)
    generated_interpretations: list[NonEmptyString] = Field(default_factory=list)


class DepthReview(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    abstracted_atom_ids: list[NonEmptyString]
    example_without_result_atom_ids: list[NonEmptyString]
    condition_without_scope_atom_ids: list[NonEmptyString]
    reordered_step_atom_ids: list[NonEmptyString]
    lost_literal_atom_ids: list[NonEmptyString]
    lost_warning_or_promotion_atom_ids: list[NonEmptyString]
    lost_uncertainty_markers: list[NonEmptyString]
    unsupported_claims: list[NonEmptyString]
    repair_section_ids: list[TeachingSectionId]
    input_sufficient: bool
    unknown_atom_ids: list[NonEmptyString] = Field(default_factory=list)
    unknown_section_ids: list[NonEmptyString] = Field(default_factory=list)
    non_repairable_findings: list[NonEmptyString] = Field(default_factory=list)


__all__ = [
    "AtomKind",
    "CoverageReport",
    "CoverageReview",
    "DepthReview",
    "DispositionKind",
    "DraftClaim",
    "DraftClaimField",
    "DraftClaimReview",
    "EvidenceAtom",
    "ExtractionResult",
    "ProviderUsage",
    "QualityGate",
    "RepairResult",
    "SegmentDisposition",
    "SectionClaim",
    "SectionDraft",
    "SourceOmissionCandidate",
    "SynthesisResult",
    "TeachingOutline",
    "TeachingOutlineSection",
    "TeachingSectionId",
    "EvidenceRequirement",
    "ThemeSection",
    "TimelineEntry",
]
