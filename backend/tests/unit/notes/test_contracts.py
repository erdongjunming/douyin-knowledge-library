from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from douyin_library.notes.contracts import (
    CoverageReport,
    CoverageReview,
    EvidenceAtom,
    ExtractionResult,
    ProviderUsage,
    SegmentDisposition,
    SourceOmissionCandidate,
    SynthesisResult,
    ThemeSection,
    TimelineEntry,
)


def _atom(**overrides: object) -> EvidenceAtom:
    values: dict[str, object] = {
        "id": "A-0001",
        "kind": "claim",
        "topic": "topic",
        "statement": "A source-backed statement.",
        "source_segment": 0,
        "source_excerpt": "source excerpt",
    }
    values.update(overrides)
    return EvidenceAtom(**values)


def _synthesis() -> SynthesisResult:
    return SynthesisResult(
        title="Title",
        overview="Overview",
        structure=["Opening"],
        themes=[],
        methods_and_steps=[],
        examples_and_counterexamples=[],
        boundaries_and_risks=[],
        concepts=[],
        quotes=[],
        implications=[],
        actions=[],
        timeline=[],
        promotions=[],
        tags=[],
        included_atom_ids=["A-0001"],
    )


def _report(**overrides: object) -> CoverageReport:
    values: dict[str, object] = {
        "segment_total": 1,
        "segment_disposition_count": 1,
        "missing_segment_indexes": [],
        "evidence_atom_total": 1,
        "evidence_atom_included": 1,
        "missing_atom_ids": [],
        "traceable_atom_count": 1,
        "source_segments_independently_reviewed": 1,
        "source_omission_candidates": [],
        "unsupported_draft_claims": [],
        "steps_verified": True,
        "examples_verified": True,
        "conditions_verified": True,
        "promotional_content_preserved": True,
        "semantic_generator": "ollama/qwen3:8b",
        "quality_gate": "passed",
        "repairable": False,
    }
    values.update(overrides)
    return CoverageReport(**values)


@pytest.mark.parametrize(
    "instance",
    [
        _atom(),
        SegmentDisposition(segment_index=0, disposition="included", atom_ids=["A-0001"]),
        ExtractionResult(
            atoms=[_atom()],
            dispositions=[
                SegmentDisposition(
                    segment_index=0,
                    disposition="included",
                    atom_ids=["A-0001"],
                )
            ],
        ),
        ThemeSection(
            heading="Heading",
            content="Content",
            atom_ids=["A-0001"],
            timestamps=[0.0],
        ),
        TimelineEntry(
            segment_index=0,
            disposition="included",
            atom_ids=["A-0001"],
            content="Content",
        ),
        _synthesis(),
        SourceOmissionCandidate(
            segment_index=0,
            source_excerpt="source",
            missing_statement="missing",
            suggested_kind="claim",
        ),
        CoverageReview(
            reviewed_segment_indexes=[0],
            source_omission_candidates=[],
            unsupported_draft_claims=[],
            disposition_corrections=[],
        ),
        ProviderUsage(cost_status="zero_local"),
        _report(),
    ],
    ids=lambda value: type(value).__name__,
)
def test_all_contracts_reject_extra_fields(instance: object) -> None:
    model_type = type(instance)
    data = instance.model_dump()
    data["unexpected"] = True

    with pytest.raises(ValidationError):
        model_type.model_validate(data)


@pytest.mark.parametrize(
    ("model_type", "values"),
    [
        (EvidenceAtom, {**_atom().model_dump(), "source_segment": -1}),
        (EvidenceAtom, {**_atom().model_dump(), "start": -0.1}),
        (EvidenceAtom, {**_atom().model_dump(), "end": -0.1}),
        (EvidenceAtom, {**_atom().model_dump(), "sequence": 0}),
        (
            SegmentDisposition,
            {"segment_index": -1, "disposition": "included", "atom_ids": []},
        ),
        (
            TimelineEntry,
            {
                "segment_index": -1,
                "disposition": "included",
                "atom_ids": [],
                "content": "content",
            },
        ),
        (
            SourceOmissionCandidate,
            {
                "segment_index": -1,
                "source_excerpt": "source",
                "missing_statement": "missing",
                "suggested_kind": "claim",
            },
        ),
        (ProviderUsage, {"input_tokens": -1, "cost_status": "unknown"}),
        (ProviderUsage, {"output_tokens": -1, "cost_status": "unknown"}),
        (ProviderUsage, {"api_cost": -0.01, "cost_status": "reported"}),
        (
            ThemeSection,
            {
                "heading": "Heading",
                "content": "Content",
                "atom_ids": [],
                "timestamps": [-0.1],
            },
        ),
        (
            CoverageReview,
            {
                "reviewed_segment_indexes": [-1],
                "source_omission_candidates": [],
                "unsupported_draft_claims": [],
                "disposition_corrections": [],
            },
        ),
        (CoverageReport, {**_report().model_dump(), "missing_segment_indexes": [-1]}),
    ],
)
def test_contracts_reject_negative_indexes_tokens_and_costs(
    model_type: type[object], values: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model_type(**values)


def test_coverage_report_serializes_the_fixed_schema_alias() -> None:
    report = _report()

    serialized = report.model_dump(mode="json")
    serialized_json = json.loads(report.model_dump_json())

    assert serialized["schema"] == "douyin-note-coverage-v1"
    assert "schema_name" not in serialized
    assert serialized_json["schema"] == "douyin-note-coverage-v1"
    assert "schema_name" not in serialized_json
    assert CoverageReport.model_validate(serialized) == report
    with pytest.raises(ValidationError):
        CoverageReport.model_validate({**serialized, "schema": "another-schema"})
