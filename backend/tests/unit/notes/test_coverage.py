from __future__ import annotations

from collections.abc import Iterable

import pytest

from douyin_library.notes.contracts import (
    CoverageReview,
    DepthReview,
    EvidenceAtom,
    ExtractionResult,
    SegmentDisposition,
    SourceOmissionCandidate,
    SynthesisResult,
    ThemeSection,
    TimelineEntry,
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


def _synthesis(
    included_atom_ids: Iterable[str],
    *,
    extraction: ExtractionResult | None = None,
) -> SynthesisResult:
    atom_ids = list(included_atom_ids)
    included = set(atom_ids)
    return SynthesisResult(
        title="Title",
        overview="Overview",
        structure=[],
        themes=(
            [
                ThemeSection(
                    heading="Complete evidence",
                    content="Complete evidence body",
                    atom_ids=atom_ids,
                )
            ]
            if atom_ids
            else []
        ),
        methods_and_steps=[],
        examples_and_counterexamples=[],
        boundaries_and_risks=[],
        concepts=[],
        quotes=[],
        implications=[],
        actions=[],
        timeline=(
            [
                TimelineEntry(
                    segment_index=disposition.segment_index,
                    disposition=disposition.disposition,
                    atom_ids=[
                        atom_id
                        for atom_id in disposition.atom_ids
                        if atom_id in included
                    ],
                    content=f"Segment {disposition.segment_index}",
                )
                for disposition in extraction.dispositions
            ]
            if extraction is not None
            else []
        ),
        promotions=[],
        tags=[],
        included_atom_ids=atom_ids,
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
        synthesis=(
            synthesis
            if synthesis is not None
            else _synthesis(
                (atom.id for atom in extraction.atoms),
                extraction=extraction,
            )
        ),
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


def test_depth_failure_is_not_semantically_complete() -> None:
    report = _validate(_complete_extraction())
    depth_review = DepthReview(
        abstracted_atom_ids=[],
        example_without_result_atom_ids=["A-example"],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=[],
        lost_warning_or_promotion_atom_ids=[],
        lost_uncertainty_markers=[],
        unsupported_claims=[],
        repair_section_ids=["examples_results"],
        input_sufficient=True,
    )
    extraction = _complete_extraction()
    checked = CoverageValidator().validate(
        segment_count=4,
        extraction=extraction,
        synthesis=_synthesis((atom.id for atom in extraction.atoms), extraction=extraction),
        review=_review(range(4)),
        semantic_generator="ollama/qwen3:8b",
        depth_review=depth_review,
    )

    assert report.quality_gate == "passed"
    assert checked.quality_gate == "failed"
    assert getattr(checked, "semantic_complete", False) is False


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
        synthesis=_synthesis(
            (atom.id for atom in extraction.atoms if atom.id != "A-example"),
            extraction=extraction,
        ),
    )

    assert report.quality_gate == "failed"
    assert report.missing_atom_ids == ["A-example"]
    assert report.evidence_atom_included == 5
    assert report.repairable is True


def test_missing_synthesis_atom_with_bad_traceability_is_nonrepairable() -> None:
    atom = _atom("A-1", 0)
    atom.source_excerpt = ""
    extraction = ExtractionResult(
        atoms=[atom],
        dispositions=[_disposition(0, ["A-1"])],
    )

    report = _validate(
        extraction,
        segment_count=1,
        synthesis=_synthesis([], extraction=extraction),
    )

    assert report.quality_gate == "failed"
    assert report.missing_atom_ids == ["A-1"]
    assert report.repairable is False


def test_unknown_synthesis_atom_is_nonrepairable() -> None:
    extraction = _complete_extraction()
    included = [atom.id for atom in extraction.atoms]

    report = _validate(
        extraction,
        synthesis=_synthesis([*included, "A-unknown"], extraction=extraction),
    )

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_untraceable_included_atom_is_nonrepairable() -> None:
    atom = _atom("A-1", 0)
    atom.source_excerpt = "   "
    extraction = ExtractionResult(
        atoms=[atom],
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


def test_step_without_sequence_fails_verification() -> None:
    extraction = _complete_extraction()
    extraction.atoms[0].sequence = None
    extraction.atoms[1].sequence = 1

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


def test_example_relation_to_unknown_atom_fails_verification() -> None:
    extraction = _complete_extraction()
    example = next(atom for atom in extraction.atoms if atom.id == "A-example")
    example.relations = ["A-unknown"]

    report = _validate(extraction)

    assert report.quality_gate == "failed"
    assert report.examples_verified is False
    assert report.repairable is False


def test_dropped_promotion_fails_as_repairable() -> None:
    extraction = _complete_extraction()

    report = _validate(
        extraction,
        synthesis=_synthesis(
            (atom.id for atom in extraction.atoms if atom.id != "A-promo"),
            extraction=extraction,
        ),
    )

    assert report.quality_gate == "failed"
    assert report.promotional_content_preserved is False
    assert report.repairable is True


def test_is_promotional_non_promotion_atom_cannot_be_dropped() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0, is_promotional=True)],
        dispositions=[_disposition(0, ["A-1"])],
    )

    report = _validate(
        extraction,
        segment_count=1,
        synthesis=_synthesis([], extraction=extraction),
    )

    assert report.quality_gate == "failed"
    assert report.promotional_content_preserved is False
    assert report.repairable is True


def test_promotion_with_wrong_disposition_is_nonrepairable() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0, kind="promotion")],
        dispositions=[_disposition(0, ["A-1"], "duplicate")],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.promotional_content_preserved is False
    assert report.repairable is False


@pytest.mark.parametrize("disposition_kind", ["included", "promotion"])
def test_content_dispositions_require_an_atom(disposition_kind: str) -> None:
    extraction = ExtractionResult(
        atoms=[],
        dispositions=[_disposition(0, [], disposition_kind)],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.repairable is False


@pytest.mark.parametrize(
    "disposition_kind",
    ["duplicate", "transition", "unintelligible_asr"],
)
def test_non_content_dispositions_may_have_no_atoms(disposition_kind: str) -> None:
    extraction = ExtractionResult(
        atoms=[],
        dispositions=[_disposition(0, [], disposition_kind)],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "passed"


@pytest.mark.parametrize(
    "section_field",
    [
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "promotions",
    ],
)
def test_unknown_theme_section_atom_reference_is_nonrepairable(
    section_field: str,
) -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    getattr(synthesis, section_field).append(
        ThemeSection(
            heading="Bad reference",
            content="Bad reference body",
            atom_ids=["A-unknown"],
        )
    )

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_unknown_timeline_atom_reference_is_nonrepairable() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.timeline[0].atom_ids.append("A-unknown")

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_body_reference_missing_from_included_ids_is_nonrepairable() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms if atom.id != "A-example"),
        extraction=extraction,
    )
    synthesis.themes.append(
        ThemeSection(
            heading="Non-included reference",
            content="Non-included reference body",
            atom_ids=["A-example"],
        )
    )

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_complete_timeline_does_not_replace_thematic_coverage() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.themes = []

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is True


def test_atom_only_referenced_in_timeline_is_a_repairable_omission() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.themes[0].atom_ids.remove("A-example")

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is True


def test_each_included_atom_with_a_thematic_reference_passes() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "passed"


def test_theme_requires_timestamps_for_each_referenced_known_start() -> None:
    extraction = _complete_extraction()
    extraction.atoms[0].start = 1.5
    extraction.atoms[1].start = 3.0
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.themes[0].timestamps = [1.5]

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is True


def test_theme_does_not_fabricate_a_timestamp_for_unknown_atom_start() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0, start=None)],
        dispositions=[_disposition(0, ["A-1"])],
    )
    synthesis = _synthesis(["A-1"], extraction=extraction)
    synthesis.themes[0].timestamps = [0.0]

    report = _validate(extraction, segment_count=1, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is True


def test_evidence_specific_theme_groups_may_be_empty() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )

    assert synthesis.methods_and_steps == []
    assert synthesis.examples_and_counterexamples == []
    assert synthesis.boundaries_and_risks == []
    assert synthesis.promotions == []
    assert _validate(extraction, synthesis=synthesis).quality_gate == "passed"


def test_zero_segments_with_empty_results_passes() -> None:
    extraction = ExtractionResult(atoms=[], dispositions=[])

    report = _validate(extraction, segment_count=0, review=_review([]))

    assert report.quality_gate == "passed"
    assert report.segment_total == 0
    assert report.evidence_atom_total == 0
    assert report.traceable_atom_count == 0
    assert report.source_segments_independently_reviewed == 0
    assert report.repairable is False


def test_timeline_requires_every_segment() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.timeline.pop(1)

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"


def test_timeline_rejects_duplicate_segment() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.timeline.insert(1, synthesis.timeline[0].model_copy(deep=True))

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_timeline_rejects_out_of_range_segment() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.timeline.append(
        TimelineEntry(
            segment_index=4,
            disposition="duplicate",
            atom_ids=[],
            content="Out of range",
        )
    )

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_timeline_rejects_reordered_segments() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.timeline[0], synthesis.timeline[1] = (
        synthesis.timeline[1],
        synthesis.timeline[0],
    )

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_timeline_disposition_must_match_extraction() -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.timeline[0].disposition = "promotion"

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


@pytest.mark.parametrize(
    "timeline_atom_ids",
    [["A-step-1"], ["A-step-1", "A-step-1"]],
)
def test_timeline_atom_mapping_must_match_without_duplicates(
    timeline_atom_ids: list[str],
) -> None:
    extraction = _complete_extraction()
    synthesis = _synthesis(
        (atom.id for atom in extraction.atoms),
        extraction=extraction,
    )
    synthesis.timeline[0].atom_ids = timeline_atom_ids

    report = _validate(extraction, synthesis=synthesis)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_zero_segments_rejects_nonempty_timeline() -> None:
    extraction = ExtractionResult(atoms=[], dispositions=[])
    synthesis = _synthesis([], extraction=extraction)
    synthesis.timeline.append(
        TimelineEntry(
            segment_index=0,
            disposition="duplicate",
            atom_ids=[],
            content="Unexpected segment",
        )
    )

    report = _validate(
        extraction,
        segment_count=0,
        synthesis=synthesis,
        review=_review([]),
    )

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_atom_must_be_mapped_by_a_disposition() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0)],
        dispositions=[_disposition(0, [], "transition")],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_atom_cannot_be_mapped_twice_by_dispositions() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0)],
        dispositions=[_disposition(0, ["A-1", "A-1"])],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.repairable is False


def test_atom_mapping_requires_content_disposition() -> None:
    extraction = ExtractionResult(
        atoms=[_atom("A-1", 0)],
        dispositions=[_disposition(0, ["A-1"], "transition")],
    )

    report = _validate(extraction, segment_count=1)

    assert report.quality_gate == "failed"
    assert report.repairable is False
