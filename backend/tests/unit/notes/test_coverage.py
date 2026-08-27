from __future__ import annotations

from collections.abc import Iterable

import pytest

from douyin_library.notes.contracts import (
    CoverageReview,
    EvidenceAtom,
    ExtractionResult,
    SegmentDisposition,
    SourceOmissionCandidate,
    SynthesisResult,
)
from douyin_library.notes.coverage import CoverageValidator


def _atom(atom_id: str, source_segment: int, **overrides: object) -> EvidenceAtom:
    values: dict[str, object] = {
        "id": atom_id,
        "kind": "claim",
        "topic": "topic",
        "statement": f"Statement for {atom_id}",
        "source_segment": source_segment,
        "source_excerpt": f"Source for {atom_id}",
    }
    values.update(overrides)
    return EvidenceAtom(**values)


def _disposition(
    segment_index: int,
    atom_ids: Iterable[str],
    disposition: str = "included",
) -> SegmentDisposition:
    return SegmentDisposition(
        segment_index=segment_index,
        disposition=disposition,
        atom_ids=list(atom_ids),
    )


def _synthesis(included_atom_ids: Iterable[str]) -> SynthesisResult:
    return SynthesisResult(
        title="Title",
        overview="Overview",
        structure=[],
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
        included_atom_ids=list(included_atom_ids),
    )


def _review(indexes: Iterable[int]) -> CoverageReview:
    return CoverageReview(
        reviewed_segment_indexes=list(indexes),
        source_omission_candidates=[],
        unsupported_draft_claims=[],
        disposition_corrections=[],
    )


def _complete_extraction() -> ExtractionResult:
    atoms = [
        _atom("A-step-1", 0, kind="step", topic="method", sequence=1),
        _atom("A-step-2", 0, kind="step", topic="method", sequence=2),
        _atom("A-claim", 1),
        _atom("A-example", 1, kind="example", relations=["A-claim"]),
        _atom("A-condition", 2, kind="condition", relations=["A-claim"]),
        _atom("A-promo", 3, kind="promotion", is_promotional=True),
    ]
    return ExtractionResult(
        atoms=atoms,
        dispositions=[
            _disposition(0, ["A-step-1", "A-step-2"]),
            _disposition(1, ["A-claim", "A-example"]),
            _disposition(2, ["A-condition"]),
            _disposition(3, ["A-promo"], "promotion"),
        ],
    )


def _validate(
    extraction: ExtractionResult,
    *,
    segment_count: int = 4,
    synthesis: SynthesisResult | None = None,
    review: CoverageReview | None = None,
):
    return CoverageValidator().validate(
        segment_count=segment_count,
        extraction=extraction,
        synthesis=synthesis or _synthesis(atom.id for atom in extraction.atoms),
        review=review or _review(range(segment_count)),
        semantic_generator="ollama/qwen3:8b",
    )


def test_complete_evidence_passes_every_gate() -> None:
    report = _validate(_complete_extraction())

    assert report.quality_gate == "passed"
    assert report.segment_total == 4
    assert report.segment_disposition_count == 4
    assert report.missing_segment_indexes == []
    assert report.evidence_atom_total == 6
    assert report.evidence_atom_included == 6
    assert report.missing_atom_ids == []
    assert report.traceable_atom_count == 6
    assert report.source_segments_independently_reviewed == 4
    assert report.steps_verified is True
    assert report.examples_verified is True
    assert report.conditions_verified is True
    assert report.promotional_content_preserved is True
    assert report.semantic_generator == "ollama/qwen3:8b"
    assert report.repairable is False


def test_missing_segment_disposition_is_a_nonrepairable_failure() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0)],
        dispositions=[_disposition(0, ["A-1"])],
    )

    report = _validate(extraction, segment_count=2)

    assert report.quality_gate == "failed"
    assert report.missing_segment_indexes == [1]
    assert report.repairable is False


def test_duplicate_segment_disposition_is_a_nonrepairable_failure() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0)],
        dispositions=[_disposition(0, ["A-1"]), _disposition(0, ["A-1"])],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.segment_disposition_count == 2
    assert report.missing_segment_indexes == []
    assert report.repairable is False


def test_unknown_disposition_atom_reference_is_nonrepairable() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0)],
        dispositions=[_disposition(0, ["A-unknown"])],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_disposition_atom_source_mismatch_is_nonrepairable() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 1)],
        dispositions=[_disposition(0, ["A-1"]), _disposition(1, [])],
    )

    report = _validate(extraction, segment_count=2)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_atom_missing_from_synthesis_is_reported_as_repairable() -> None:
    extraction = _complete_extraction()

    report = _validate(
        extraction,
        synthesis=_synthesis(atom.id for atom in extraction.atoms if atom.id != "A-example"),
    )

    assert report.quality_gate == "failed"
    assert report.missing_atom_ids == ["A-example"]
    assert report.evidence_atom_included == 5
    assert report.repairable is True


def test_missing_synthesis_atom_with_bad_traceability_is_nonrepairable() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0, source_excerpt="")],
        dispositions=[_disposition(0, ["A-1"])],
    )

    report = _validate(
        extraction,
        segment_count=1,
        synthesis=_synthesis([]),
    )

    assert report.quality_gate == "failed"
    assert report.missing_atom_ids == ["A-1"]
    assert report.repairable is False


def test_unknown_synthesis_atom_is_nonrepairable() -> None:
    extraction = _complete_extraction()
    included = [atom.id for atom in extraction.atoms]

    report = _validate(extraction, synthesis=_synthesis([*included, "A-unknown"]))

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_untraceable_included_atom_is_nonrepairable() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0, source_excerpt="   ")],
        dispositions=[_disposition(0, ["A-1"])],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.traceable_atom_count == 0
    assert report.repairable is False


@pytest.mark.parametrize("reviewed_indexes", [[0], [0, 1, 1]])
def test_raw_source_review_requires_each_segment_exactly_once(
    reviewed_indexes: list[int],
) -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0), _atom("A-2", 1)],
        dispositions=[_disposition(0, ["A-1"]), _disposition(1, ["A-2"])],
    )

    report = _validate(
        extraction,
        segment_count=2,
        review=_review(reviewed_indexes),
    )

    assert report.quality_gate == "failed"
    assert report.repairable is True


@pytest.mark.parametrize("review_problem", ["omission", "unsupported"])
def test_review_findings_fail_as_repairable(review_problem: str) -> None:
    review = _review(range(4))
    if review_problem == "omission":
        review.source_omission_candidates.append(
            SourceOmissionCandidate(
                segment_index=2,
                source_excerpt="omitted source",
                missing_statement="omitted statement",
                suggested_kind="warning",
            )
        )
    else:
        review.unsupported_draft_claims.append("Invented draft claim")

    report = _validate(_complete_extraction(), review=review)

    assert report.quality_gate == "failed"
    assert report.repairable is True


def test_step_sequence_gap_fails_verification() -> None:
    extraction = _complete_extraction()
    extraction.atoms[1].sequence = 3

    report = _validate(extraction)

    assert report.quality_gate == "failed"
    assert report.steps_verified is False
    assert report.repairable is False


@pytest.mark.parametrize(
    ("atom_id", "verified_field"),
    [("A-example", "examples_verified"), ("A-condition", "conditions_verified")],
)
def test_relationless_examples_and_conditions_fail_verification(
    atom_id: str,
    verified_field: str,
) -> None:
    extraction = _complete_extraction()
    next(atom for atom in extraction.atoms if atom.id == atom_id).relations = []

    report = _validate(extraction)

    assert report.quality_gate == "failed"
    assert getattr(report, verified_field) is False
    assert report.repairable is False


def test_dropped_promotion_fails_as_repairable() -> None:
    extraction = _complete_extraction()

    report = _validate(
        extraction,
        synthesis=_synthesis(atom.id for atom in extraction.atoms if atom.id != "A-promo"),
    )

    assert report.quality_gate == "failed"
    assert report.promotional_content_preserved is False
    assert report.repairable is True


def test_zero_segments_with_empty_results_passes() -> None:
    extraction = ExtractionResult(atoms=[], dispositions=[])

    report = _validate(extraction, segment_count=0, review=_review([]))

    assert report.quality_gate == "passed"
    assert report.segment_total == 0
    assert report.evidence_atom_total == 0
    assert report.traceable_atom_count == 0
    assert report.source_segments_independently_reviewed == 0
    assert report.repairable is False
