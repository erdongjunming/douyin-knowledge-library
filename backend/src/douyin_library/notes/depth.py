from __future__ import annotations

import re
from collections.abc import Iterable

from douyin_library.notes.contracts import (
    DepthReview,
    EvidenceAtom,
    SectionDraft,
    TeachingOutline,
)


class DepthValidator:
    """Run deterministic, structure-preserving checks over a teaching draft."""

    def validate(
        self,
        atoms: list[EvidenceAtom],
        outline: TeachingOutline,
        sections: list[SectionDraft],
        review: DepthReview | None = None,
    ) -> DepthReview:
        atom_by_id = {atom.id: atom for atom in atoms}
        outline_by_id = {section.section_id: section for section in outline.sections}
        section_by_id = {section.section_id: section for section in sections}
        content = {section.section_id: self._section_text(section) for section in sections}
        section_atoms = {section.section_id: set(section.included_atom_ids) for section in sections}
        repair_sections: list[str] = []

        def mark(section_id: str) -> None:
            if section_id in outline_by_id and section_id not in repair_sections:
                repair_sections.append(section_id)

        unknown_ids = set(outline.included_atom_ids) - atom_by_id.keys()
        for outline_section in outline.sections:
            unknown_ids.update(set(outline_section.atom_ids) - atom_by_id.keys())
            unknown_ids.update(set(outline_section.atom_ids) - set(outline.included_atom_ids))
        for section in sections:
            unknown_ids.update(set(section.included_atom_ids) - atom_by_id.keys())
            outline_entry = outline_by_id.get(section.section_id)
            outline_ids = set(outline_entry.atom_ids) if outline_entry else set()
            unknown_ids.update(set(section.included_atom_ids) - outline_ids)
            if section.section_id not in outline_by_id:
                continue
            if unknown_ids:
                mark(section.section_id)

        abstracted: list[str] = []
        for atom in atoms:
            if not atom.statement.strip() or not atom.source_excerpt.strip():
                if atom.id not in set().union(*(section_atoms.values()), set()):
                    continue
                section_id = self._section_for_atom(atom.id, section_atoms)
                unprovided = self._unprovided(section_by_id[section_id]) if section_id else set()
                if section_id and not unprovided.intersection(
                    {atom.id, atom.statement, atom.source_excerpt}
                ):
                    abstracted.append(atom.id)
                    mark(section_id)

        examples_missing: list[str] = []
        conditions_missing: list[str] = []
        reordered_steps: list[str] = []
        lost_literals: list[str] = []
        lost_warning_promo: list[str] = []
        lost_uncertainty: list[str] = []
        for section in sections:
            text = content[section.section_id]
            claims = [*section.claims, *section.steps]
            claim_ids = {atom_id for claim in claims for atom_id in claim.atom_ids}
            result_ids = {atom_id for result in section.results for atom_id in result.atom_ids}
            for claim in section.examples:
                atom = next((atom_by_id.get(atom_id) for atom_id in claim.atom_ids), None)
                if atom and not result_ids.intersection({atom.id, *atom.relations}):
                    examples_missing.extend(claim.atom_ids)
                    mark(section.section_id)
            for claim in [*section.conditions, *section.risks]:
                related = set(claim.atom_ids)
                related.update(
                    relation
                    for atom_id in claim.atom_ids
                    if (atom := atom_by_id.get(atom_id)) is not None
                    for relation in atom.relations
                )
                if not related.intersection(claim_ids):
                    conditions_missing.extend(claim.atom_ids)
                    mark(section.section_id)
            for atom in atoms:
                if atom.kind == "warning" and atom.id in section_atoms[section.section_id]:
                    if not set(atom.relations).intersection(claim_ids):
                        conditions_missing.append(atom.id)
                        mark(section.section_id)
            steps_in_section = [
                atom
                for atom in atoms
                if atom.id in section_atoms[section.section_id] and atom.kind == "step"
            ]
            sequences = sorted(
                atom.sequence for atom in steps_in_section if atom.sequence is not None
            )
            if sequences and sequences != list(range(1, max(sequences) + 1)):
                reordered_steps.extend(atom.id for atom in steps_in_section)
                mark(section.section_id)
            for atom in atoms:
                if atom.id not in section_atoms[section.section_id]:
                    continue
                if atom.kind == "step" and atom.sequence is not None:
                    ordered = [
                        claim_atom for claim in section.steps for claim_atom in claim.atom_ids
                    ]
                    expected = sorted(
                        (
                            a
                            for a in atoms
                            if a.id in section_atoms[section.section_id] and a.kind == "step"
                        ),
                        key=lambda item: item.sequence or 0,
                    )
                    if ordered and [item.id for item in expected] != ordered:
                        reordered_steps.extend(item.id for item in expected)
                        mark(section.section_id)
                    claim_text_by_atom = {
                        atom_id: claim.text for claim in section.steps for atom_id in claim.atom_ids
                    }
                    positions = [
                        text.find(claim_text_by_atom.get(atom.id, atom.statement))
                        for atom in expected
                    ]
                    if positions != sorted(positions) or any(
                        position < 0 for position in positions
                    ):
                        reordered_steps.extend(item.id for item in expected)
                        mark(section.section_id)
                literals = [atom.statement, atom.source_excerpt, *atom.asr_uncertainty]
                if atom.kind == "number" or self._looks_like_named_entity(atom.statement):
                    if not any(value and value in text for value in literals[:2]):
                        lost_literals.append(atom.id)
                        mark(section.section_id)
                if atom.kind in {"warning", "promotion"} or atom.is_promotional:
                    if not any(value and value in text for value in literals[:2]):
                        lost_warning_promo.append(atom.id)
                        mark(section.section_id)
                if atom.asr_uncertainty and not all(
                    marker in text for marker in atom.asr_uncertainty
                ):
                    lost_uncertainty.extend(atom.asr_uncertainty)
                    mark(section.section_id)

        inherited = review or DepthReview(
            abstracted_atom_ids=[],
            example_without_result_atom_ids=[],
            condition_without_scope_atom_ids=[],
            reordered_step_atom_ids=[],
            lost_literal_atom_ids=[],
            lost_warning_or_promotion_atom_ids=[],
            lost_uncertainty_markers=[],
            unsupported_claims=[],
            repair_section_ids=[],
            input_sufficient=outline.input_sufficient,
        )
        return DepthReview(
            abstracted_atom_ids=self._unique([*inherited.abstracted_atom_ids, *abstracted]),
            example_without_result_atom_ids=self._unique(
                [*inherited.example_without_result_atom_ids, *examples_missing]
            ),
            condition_without_scope_atom_ids=self._unique(
                [*inherited.condition_without_scope_atom_ids, *conditions_missing]
            ),
            reordered_step_atom_ids=self._unique(
                [*inherited.reordered_step_atom_ids, *reordered_steps]
            ),
            lost_literal_atom_ids=self._unique([*inherited.lost_literal_atom_ids, *lost_literals]),
            lost_warning_or_promotion_atom_ids=self._unique(
                [*inherited.lost_warning_or_promotion_atom_ids, *lost_warning_promo]
            ),
            lost_uncertainty_markers=self._unique(
                [*inherited.lost_uncertainty_markers, *lost_uncertainty]
            ),
            unsupported_claims=self._unique(inherited.unsupported_claims),
            repair_section_ids=self._unique([*inherited.repair_section_ids, *repair_sections]),
            input_sufficient=outline.input_sufficient,
        )

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _section_text(section: SectionDraft) -> str:
        values = [section.content, *section.unprovided_items, *section.generated_interpretations]
        values.extend(
            claim.text
            for field in (
                section.claims,
                section.steps,
                section.examples,
                section.results,
                section.conditions,
                section.risks,
                section.actions,
            )
            for claim in field
        )
        return "\n".join(values)

    @staticmethod
    def _section_for_atom(atom_id: str, sections: dict[str, set[str]]) -> str | None:
        return next((section_id for section_id, ids in sections.items() if atom_id in ids), None)

    @staticmethod
    def _unprovided(section: SectionDraft) -> set[str]:
        return set(section.unprovided_items)

    @staticmethod
    def _looks_like_named_entity(statement: str) -> bool:
        if re.search(r"(?:姜学长|老师|先生|女士)", statement):
            return True
        tokens = re.findall(r"[A-Za-z]+", statement)
        return any(
            token.isascii()
            and token[:1].isupper()
            and token[1:].isalpha()
            and token not in {"Statement", "Source", "Excerpt"}
            for token in tokens[1:]
        )


__all__ = ["DepthValidator"]
