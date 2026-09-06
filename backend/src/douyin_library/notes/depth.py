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

    _STRUCTURED_FIELDS = (
        "claims",
        "steps",
        "examples",
        "results",
        "conditions",
        "risks",
        "actions",
    )

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
        body = {section.section_id: self._section_text(section) for section in sections}
        outline_ids = set(outline.included_atom_ids)
        outline_ids.update(atom_id for section in outline.sections for atom_id in section.atom_ids)
        assigned = {section.section_id: set(section.atom_ids) for section in outline.sections}
        assigned.update(
            {section.section_id: set(section.included_atom_ids) for section in sections}
        )
        unknown_atoms: list[str] = []
        unknown_sections: list[str] = []
        non_repairable: list[str] = []

        for atom_id in outline_ids:
            if atom_id not in atom_by_id:
                unknown_atoms.append(atom_id)
        for section in sections:
            if section.section_id not in outline_by_id:
                unknown_sections.append(section.section_id)
                non_repairable.append(f"unknown section: {section.section_id}")
                continue
            allowed = set(outline_by_id[section.section_id].atom_ids)
            for atom_id in section.included_atom_ids:
                if atom_id not in atom_by_id or atom_id not in allowed:
                    unknown_atoms.append(atom_id)
            for field_name in self._STRUCTURED_FIELDS:
                for claim in getattr(section, field_name):
                    for atom_id in claim.atom_ids:
                        if atom_id not in atom_by_id or atom_id not in allowed:
                            unknown_atoms.append(atom_id)

        unknown_atoms = self._unique(unknown_atoms)
        non_repairable.extend(f"unknown atom reference: {atom_id}" for atom_id in unknown_atoms)

        abstracted: list[str] = []
        for section_id, atom_ids in assigned.items():
            section = section_by_id.get(section_id)
            if section is None:
                continue
            unprovided = set(section.unprovided_items)
            searchable = body[section_id]
            for atom_id in atom_ids:
                atom = atom_by_id.get(atom_id)
                if atom is None:
                    continue
                if (
                    atom.statement not in searchable
                    and atom.source_excerpt not in searchable
                    and not ({atom.id, atom.statement, atom.source_excerpt} & unprovided)
                ):
                    abstracted.append(atom.id)

        examples_missing: list[str] = []
        conditions_missing: list[str] = []
        reordered_steps: list[str] = []
        lost_literals: list[str] = []
        lost_warning_promo: list[str] = []
        lost_uncertainty: list[str] = []
        repair_sections: list[str] = []

        def repair(section_id: str) -> None:
            if section_id in outline_by_id and section_id not in repair_sections:
                repair_sections.append(section_id)

        for section_id, atom_ids in assigned.items():
            section = section_by_id.get(section_id)
            if section is None:
                continue
            text = body[section_id]
            claims = [claim for field in (section.claims, section.steps) for claim in field]
            claim_ids = {atom_id for claim in claims for atom_id in claim.atom_ids}
            result_ids = {atom_id for claim in section.results for atom_id in claim.atom_ids}
            steps = [atom for atom in atoms if atom.id in atom_ids and atom.kind == "step"]
            sequences = sorted(atom.sequence for atom in steps if atom.sequence is not None)
            if sequences and sequences != list(range(1, max(sequences) + 1)):
                reordered_steps.extend(atom.id for atom in steps)
                repair(section_id)
            for atom in atoms:
                if atom.id not in atom_ids:
                    continue
                if atom.kind in {"example", "counterexample"}:
                    has_result = bool(set(atom.relations) & result_ids) or atom.id in result_ids
                    if not atom.relations or not has_result:
                        examples_missing.append(atom.id)
                        repair(section_id)
                if atom.kind in {"condition", "warning"}:
                    if not set(atom.relations) & claim_ids:
                        conditions_missing.append(atom.id)
                        repair(section_id)
                if atom.kind == "number" or self._looks_like_named_entity(atom.statement):
                    if atom.statement not in text and atom.source_excerpt not in text:
                        lost_literals.append(atom.id)
                        repair(section_id)
                if atom.kind == "promotion" or atom.is_promotional:
                    if atom.statement not in text and atom.source_excerpt not in text:
                        lost_warning_promo.append(atom.id)
                        repair(section_id)
                if atom.asr_uncertainty:
                    for marker in atom.asr_uncertainty:
                        if marker not in text:
                            lost_uncertainty.append(marker)
                            repair(section_id)
            expected = sorted(steps, key=lambda atom: atom.sequence or 0)
            structured_order = [atom_id for claim in section.steps for atom_id in claim.atom_ids]
            if structured_order and [atom.id for atom in expected] != structured_order:
                reordered_steps.extend(atom.id for atom in expected)
                repair(section_id)
            positions = [
                text.find(
                    next(
                        (claim.text for claim in section.steps if atom.id in claim.atom_ids),
                        atom.statement,
                    )
                )
                for atom in expected
            ]
            if any(position < 0 for position in positions) or positions != sorted(positions):
                reordered_steps.extend(atom.id for atom in expected)
                repair(section_id)

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
            unknown_atom_ids=self._unique([*inherited.unknown_atom_ids, *unknown_atoms]),
            unknown_section_ids=self._unique([*inherited.unknown_section_ids, *unknown_sections]),
            non_repairable_findings=self._unique(
                [*inherited.non_repairable_findings, *non_repairable]
            ),
        )

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _section_text(section: SectionDraft) -> str:
        values = [section.content, *section.generated_interpretations]
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
    def _looks_like_named_entity(statement: str) -> bool:
        if re.search(r"(?:姜学长|老师|先生|女士)", statement):
            return True
        tokens = re.findall(r"[A-Za-z]+", statement)
        return any(token[:1].isupper() and token[1:].isalpha() for token in tokens[1:])


__all__ = ["DepthValidator"]
