from __future__ import annotations

from typing import Annotated, Literal

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


__all__ = [
    "AtomKind",
    "CoverageReport",
    "CoverageReview",
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
    "SourceOmissionCandidate",
    "SynthesisResult",
    "ThemeSection",
    "TimelineEntry",
]
