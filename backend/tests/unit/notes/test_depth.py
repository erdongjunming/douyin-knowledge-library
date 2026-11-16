from douyin_library.notes.contracts import (
    DepthReview,
    EvidenceAtom,
    SectionClaim,
    SectionDraft,
    TeachingOutline,
    TeachingOutlineSection,
)
from douyin_library.notes.depth import DepthValidator


def _atom(atom_id: str, kind: str = "claim", **overrides: object) -> EvidenceAtom:
    values = {
        "id": atom_id,
        "kind": kind,
        "topic": "method",
        "statement": f"Statement {atom_id}",
        "source_segment": 0,
        "source_excerpt": f"Excerpt {atom_id}",
    }
    values.update(overrides)
    return EvidenceAtom(**values)


def _claim(atom_id: str, text: str) -> SectionClaim:
    return SectionClaim(text=text, atom_ids=[atom_id], timestamps=[0.0])


def _fixture():
    atoms = [
        _atom("claim"),
        _atom("step-1", "step", sequence=1),
        _atom("step-2", "step", sequence=2),
        _atom("result"),
        _atom("example", "example", relations=["result"]),
        _atom("condition", "condition", relations=["claim"]),
        _atom("warning", "warning", relations=["claim"]),
        _atom("promo", "promotion", is_promotional=True),
        _atom("uncertain", "claim", asr_uncertainty=["听不清"]),
    ]
    outline = TeachingOutline(
        sections=[
            TeachingOutlineSection(
                section_id="methods_steps",
                title="Method",
                order=1,
                atom_ids=[atom.id for atom in atoms],
                required_evidence_types=["claim", "step", "example", "condition"],
                missing_evidence_policy="视频未提供这一类信息。",
            )
        ],
        included_atom_ids=[atom.id for atom in atoms],
    )
    sections = [
        SectionDraft(
            section_id="methods_steps",
            title="Method",
            content=(
                "Claim. Step one. Step two. Example result. Condition scope. "
                "Warning. promo 听不清 Excerpt claim Excerpt step-1 Excerpt step-2 "
                "Excerpt result Excerpt example Excerpt condition Excerpt warning Excerpt promo "
                "Excerpt uncertain"
            ),
            claims=[_claim("claim", "Claim")],
            steps=[_claim("step-1", "Step one"), _claim("step-2", "Step two")],
            examples=[_claim("example", "Example")],
            results=[_claim("result", "Result")],
            conditions=[_claim("condition", "Condition")],
            risks=[_claim("warning", "Warning")],
            included_atom_ids=[atom.id for atom in atoms],
        )
    ]
    return atoms, outline, sections


def test_complete_fixture_passes_deterministically() -> None:
    atoms, outline, sections = _fixture()

    review = DepthValidator().validate(atoms, outline, sections)

    assert review == DepthReview(
        abstracted_atom_ids=[],
        example_without_result_atom_ids=[],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=[],
        lost_warning_or_promotion_atom_ids=[],
        lost_uncertainty_markers=[],
        unsupported_claims=[],
        repair_section_ids=[],
        input_sufficient=True,
    )


def test_missing_result_and_condition_scope_are_findings() -> None:
    atoms, outline, sections = _fixture()
    sections[0].results = []
    sections[0].conditions = [_claim("condition", "unrelated condition")]
    next(atom for atom in atoms if atom.id == "condition").relations = []

    review = DepthValidator().validate(atoms, outline, sections)

    assert review.example_without_result_atom_ids == ["example"]
    assert review.condition_without_scope_atom_ids == ["condition"]


def test_example_cannot_satisfy_its_own_result_requirement() -> None:
    atoms, outline, sections = _fixture()
    example = next(atom for atom in atoms if atom.id == "example")
    example.relations = ["example"]
    sections[0].results = [_claim("example", "Result")]

    review = DepthValidator().validate(atoms, outline, sections)

    assert review.example_without_result_atom_ids == ["example"]


def test_risk_without_same_chapter_scope_is_a_finding() -> None:
    atoms, outline, sections = _fixture()
    next(atom for atom in atoms if atom.id == "warning").relations = []
    sections[0].risks = [_claim("warning", "Unscoped risk")]

    review = DepthValidator().validate(atoms, outline, sections)

    assert "warning" in review.condition_without_scope_atom_ids


def test_structured_example_requires_result_even_for_claim_atom() -> None:
    atoms, outline, sections = _fixture()
    next(atom for atom in atoms if atom.id == "example").kind = "claim"
    sections[0].results = []

    review = DepthValidator().validate(atoms, outline, sections)

    assert "example" in review.example_without_result_atom_ids


def test_outline_section_atom_outside_included_is_nonrepairable() -> None:
    atoms, outline, sections = _fixture()
    outline.included_atom_ids.remove("claim")

    review = DepthValidator().validate(atoms, outline, sections)

    assert review.unknown_atom_ids == ["claim"]
    assert review.repair_section_ids == []
    assert review.non_repairable_findings


def test_depth_review_serialization_is_stable_across_repeated_runs() -> None:
    atoms, outline, sections = _fixture()
    outline.included_atom_ids.append("unknown-b")
    outline.included_atom_ids.append("unknown-a")
    sections[0].included_atom_ids.extend(["unknown-b", "unknown-a"])
    validator = DepthValidator()

    serialized = [validator.validate(atoms, outline, sections).model_dump_json() for _ in range(5)]

    assert len(set(serialized)) == 1


def test_unknown_references_and_unsupported_claims_are_reported() -> None:
    atoms, outline, sections = _fixture()
    outline.included_atom_ids.append("unknown")
    sections[0].included_atom_ids.append("unknown")
    review = DepthReview(
        abstracted_atom_ids=[],
        example_without_result_atom_ids=[],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=[],
        lost_warning_or_promotion_atom_ids=[],
        lost_uncertainty_markers=[],
        unsupported_claims=["unsupported"],
        repair_section_ids=[],
        input_sufficient=True,
    )

    result = DepthValidator().validate(atoms, outline, sections, review=review)

    assert result.unsupported_claims == ["unsupported"]
    assert result.repair_section_ids == []
    assert result.unknown_atom_ids == ["unknown"]
    assert result.non_repairable_findings


def test_unsupported_claim_is_nonrepairable_even_when_section_is_flagged() -> None:
    atoms, outline, sections = _fixture()
    review = DepthReview(
        abstracted_atom_ids=[],
        example_without_result_atom_ids=[],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=[],
        lost_warning_or_promotion_atom_ids=[],
        lost_uncertainty_markers=[],
        unsupported_claims=["unsupported"],
        repair_section_ids=["methods_steps"],
        input_sufficient=True,
    )

    result = DepthValidator().validate(atoms, outline, sections, review=review)

    assert result.repair_section_ids == ["methods_steps"]
    assert result.non_repairable_findings


def test_lost_literal_and_reordered_steps_are_reported() -> None:
    atoms, outline, sections = _fixture()
    sections[0].content = "Step two. Step one. Claim. Example result. Condition scope."
    sections[0].content = sections[0].content.replace("听不清", "")

    review = DepthValidator().validate(atoms, outline, sections)

    assert "听不清" in review.lost_uncertainty_markers
    assert set(review.reordered_step_atom_ids) == {"step-1", "step-2"}
