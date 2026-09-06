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
        _atom("example", "example", relations=["claim"]),
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
                "Warning. promo 听不清 Excerpt warning Excerpt promo"
            ),
            claims=[_claim("claim", "Claim")],
            steps=[_claim("step-1", "Step one"), _claim("step-2", "Step two")],
            examples=[_claim("example", "Example")],
            results=[_claim("example", "Result")],
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
    atoms[4].relations = []

    review = DepthValidator().validate(atoms, outline, sections)

    assert review.example_without_result_atom_ids == ["example"]
    assert review.condition_without_scope_atom_ids == ["condition"]


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
    assert result.repair_section_ids == ["methods_steps"]


def test_lost_literal_and_reordered_steps_are_reported() -> None:
    atoms, outline, sections = _fixture()
    sections[0].content = "Step two. Step one. Claim. Example result. Condition scope."
    sections[0].content = sections[0].content.replace("听不清", "")

    review = DepthValidator().validate(atoms, outline, sections)

    assert "听不清" in review.lost_uncertainty_markers
    assert set(review.reordered_step_atom_ids) == {"step-1", "step-2"}
