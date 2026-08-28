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
QualityGate = Literal["passed", "failed", "fallback"]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class EvidenceAtom(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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
    model_config = ConfigDict(extra="forbid", strict=True)

    segment_index: int = Field(ge=0)
    disposition: DispositionKind
    atom_ids: list[NonEmptyString] = Field(default_factory=list)
    reason: NonEmptyString | None = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    atoms: list[EvidenceAtom]
    dispositions: list[SegmentDisposition]


class ThemeSection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    heading: NonEmptyString
    content: NonEmptyString
    atom_ids: list[NonEmptyString]
    timestamps: list[NonNegativeFloat] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    segment_index: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    disposition: DispositionKind
    atom_ids: list[NonEmptyString]
    content: NonEmptyString


class SynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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


class SourceOmissionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    segment_index: int = Field(ge=0)
    source_excerpt: NonEmptyString
    missing_statement: NonEmptyString
    suggested_kind: AtomKind


class CoverageReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewed_segment_indexes: list[NonNegativeInt]
    source_omission_candidates: list[SourceOmissionCandidate]
    unsupported_draft_claims: list[NonEmptyString]
    disposition_corrections: list[SegmentDisposition]


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        strict=True,
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
    promotional_content_preserved: bool
    semantic_generator: NonEmptyString
    quality_gate: QualityGate
    repairable: bool


__all__ = [
    "AtomKind",
    "CoverageReport",
    "CoverageReview",
    "DispositionKind",
    "EvidenceAtom",
    "ExtractionResult",
    "ProviderUsage",
    "QualityGate",
    "SegmentDisposition",
    "SourceOmissionCandidate",
    "SynthesisResult",
    "ThemeSection",
    "TimelineEntry",
]
