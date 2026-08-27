from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class EvidenceAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: AtomKind
    topic: str
    statement: str
    source_segment: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    source_excerpt: str
    relations: list[str] = Field(default_factory=list)
    sequence: int | None = Field(default=None, ge=1)
    is_promotional: bool = False
    asr_uncertainty: list[str] = Field(default_factory=list)


class SegmentDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_index: int = Field(ge=0)
    disposition: DispositionKind
    atom_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atoms: list[EvidenceAtom]
    dispositions: list[SegmentDisposition]


class ThemeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    content: str
    atom_ids: list[str]
    timestamps: list[NonNegativeFloat] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_index: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    disposition: DispositionKind
    atom_ids: list[str]
    content: str


class SynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    overview: str
    structure: list[str]
    themes: list[ThemeSection]
    methods_and_steps: list[ThemeSection]
    examples_and_counterexamples: list[ThemeSection]
    boundaries_and_risks: list[ThemeSection]
    concepts: list[str]
    quotes: list[str]
    implications: list[str]
    actions: list[str]
    timeline: list[TimelineEntry]
    promotions: list[ThemeSection]
    tags: list[str]
    included_atom_ids: list[str]


class SourceOmissionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_index: int = Field(ge=0)
    source_excerpt: str
    missing_statement: str
    suggested_kind: AtomKind


class CoverageReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed_segment_indexes: list[NonNegativeInt]
    source_omission_candidates: list[SourceOmissionCandidate]
    unsupported_draft_claims: list[str]
    disposition_corrections: list[SegmentDisposition]


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    missing_atom_ids: list[str]
    traceable_atom_count: int = Field(ge=0)
    source_segments_independently_reviewed: int = Field(ge=0)
    source_omission_candidates: list[SourceOmissionCandidate]
    unsupported_draft_claims: list[str]
    steps_verified: bool
    examples_verified: bool
    conditions_verified: bool
    promotional_content_preserved: bool
    semantic_generator: str
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
