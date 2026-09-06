from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from douyin_library.notes.contracts import (
    CoverageReport,
    CoverageReview,
    DepthReview,
    EvidenceAtom,
    ExtractionResult,
    SegmentDisposition,
    SynthesisResult,
)

_COMMON_SENTENCE_OPENERS = {
    "A",
    "An",
    "And",
    "First",
    "I",
    "It",
    "Next",
    "The",
    "This",
    "That",
    "Today",
    "We",
    "Complete",
    "Segment",
    "Source",
    "Statement",
}


def _looks_like_latin_name(token: str) -> bool:
    token = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token)
    if not token or not token.isascii() or not token.isalpha():
        return False
    if token.isupper() and len(token) >= 2:
        return True
    if any(char.isupper() for char in token[1:]):
        return True
    return token[0].isupper() and token not in _COMMON_SENTENCE_OPENERS


class CoverageValidator:
    def validate(
        self,
        *,
        segment_count: int,
        extraction: ExtractionResult,
        synthesis: SynthesisResult,
        review: CoverageReview,
        semantic_generator: str,
        depth_review: DepthReview | None = None,
        input_sufficient: bool = True,
    ) -> CoverageReport:
        if segment_count < 0:
            raise ValueError("segment_count must be non-negative")

        expected_segments = set(range(segment_count))
        disposition_counts = Counter(
            disposition.segment_index for disposition in extraction.dispositions
        )
        missing_segment_indexes = sorted(expected_segments - disposition_counts.keys())
        disposition_coverage_valid = set(disposition_counts) == expected_segments and all(
            count == 1 for count in disposition_counts.values()
        )

        atom_counts = Counter(atom.id for atom in extraction.atoms)
        atom_by_id = {atom.id: atom for atom in extraction.atoms}
        duplicate_atom_ids = any(count != 1 for count in atom_counts.values())
        disposition_references_valid = self._disposition_references_are_valid(
            extraction.dispositions,
            atom_by_id,
            expected_segments,
        )

        included_ids = set(synthesis.included_atom_ids)
        missing_atom_ids = [atom.id for atom in extraction.atoms if atom.id not in included_ids]
        unknown_synthesis_references = any(
            atom_id not in atom_by_id for atom_id in synthesis.included_atom_ids
        )
        included_atoms = [atom for atom in extraction.atoms if atom.id in included_ids]
        traceable_atom_count = sum(
            self._is_traceable(atom, segment_count) for atom in included_atoms
        )
        all_atoms_traceable = all(
            self._is_traceable(atom, segment_count) for atom in extraction.atoms
        )
        thematic_atom_ids = self._thematic_atom_ids(synthesis)
        thematic_id_set = set(thematic_atom_ids)
        unknown_thematic_references = any(
            atom_id not in atom_by_id for atom_id in thematic_atom_ids
        )
        non_included_thematic_references = any(
            atom_id not in included_ids for atom_id in thematic_atom_ids
        )
        included_atoms_missing_from_themes = included_ids - thematic_id_set
        theme_timestamps_valid = self._theme_timestamps_are_valid(
            synthesis,
            atom_by_id,
        )
        timeline_foundational_valid, timeline_complete = self._timeline_status(
            synthesis,
            extraction.dispositions,
            segment_count,
            atom_by_id,
            included_ids,
        )

        review_counts = Counter(review.reviewed_segment_indexes)
        review_coverage_valid = set(review_counts) == expected_segments and all(
            count == 1 for count in review_counts.values()
        )
        independently_reviewed = len(expected_segments & review_counts.keys())

        steps_verified = self._steps_are_contiguous(extraction.atoms)
        examples_verified = self._relations_are_valid(
            extraction.atoms,
            atom_by_id,
            kinds={"example", "counterexample"},
        )
        conditions_verified = self._relations_are_valid(
            extraction.atoms,
            atom_by_id,
            kinds={"condition"},
        )
        quotations_verified = self._quotations_are_preserved(extraction.atoms, synthesis)
        numbers_verified = self._text_categories_are_preserved(
            extraction.atoms, synthesis, category="number"
        )
        named_entities_verified = self._named_entities_are_preserved(extraction.atoms, synthesis)
        uncertainty_markers_verified = self._uncertainty_markers_are_preserved(
            extraction.atoms, synthesis
        )
        promotion_inclusion_valid, promotion_dispositions_valid = self._promotions_are_preserved(
            extraction.atoms,
            extraction.dispositions,
            included_ids,
        )

        foundational_error = any(
            (
                not disposition_coverage_valid,
                duplicate_atom_ids,
                not disposition_references_valid,
                unknown_synthesis_references,
                unknown_thematic_references,
                non_included_thematic_references,
                not timeline_foundational_valid,
                not all_atoms_traceable,
                not steps_verified,
                not examples_verified,
                not conditions_verified,
                not quotations_verified,
                not numbers_verified,
                not named_entities_verified,
                not uncertainty_markers_verified,
                not promotion_dispositions_valid,
                bool(review.disposition_corrections),
            )
        )
        repair_issue = any(
            (
                bool(missing_atom_ids),
                not review_coverage_valid,
                bool(review.source_omission_candidates),
                bool(review.unsupported_draft_claims),
                not promotion_inclusion_valid,
                bool(included_atoms_missing_from_themes),
                not theme_timestamps_valid,
                not timeline_complete,
            )
        )
        passed = not foundational_error and not repair_issue

        depth_passed = depth_review is not None and not any(
            (
                depth_review.abstracted_atom_ids,
                depth_review.example_without_result_atom_ids,
                depth_review.condition_without_scope_atom_ids,
                depth_review.reordered_step_atom_ids,
                depth_review.lost_literal_atom_ids,
                depth_review.lost_warning_or_promotion_atom_ids,
                depth_review.lost_uncertainty_markers,
                depth_review.unsupported_claims,
                depth_review.repair_section_ids,
            )
        )
        semantic_complete = bool(depth_passed and passed and input_sufficient)
        report_values = dict(
            segment_total=segment_count,
            segment_disposition_count=len(extraction.dispositions),
            missing_segment_indexes=missing_segment_indexes,
            evidence_atom_total=len(extraction.atoms),
            evidence_atom_included=len(included_atoms),
            missing_atom_ids=missing_atom_ids,
            traceable_atom_count=traceable_atom_count,
            source_segments_independently_reviewed=independently_reviewed,
            source_omission_candidates=review.source_omission_candidates,
            unsupported_draft_claims=review.unsupported_draft_claims,
            steps_verified=steps_verified,
            examples_verified=examples_verified,
            conditions_verified=conditions_verified,
            numbers_verified=numbers_verified,
            named_entities_verified=named_entities_verified,
            uncertainty_markers_verified=uncertainty_markers_verified,
            promotional_content_preserved=(
                promotion_inclusion_valid and promotion_dispositions_valid
            ),
            semantic_generator=semantic_generator,
            quality_gate=(
                "passed" if passed and (depth_review is None or semantic_complete) else "failed"
            ),
            repairable=not passed and not foundational_error and repair_issue,
            depth_passed=depth_passed,
            input_sufficient=input_sufficient,
            depth_review=depth_review,
            semantic_complete=semantic_complete,
        )
        # Keep callers compatible with a pre-Task1A contract during rolling upgrades.
        supported = getattr(CoverageReport, "model_fields", {})
        return CoverageReport(
            **{key: value for key, value in report_values.items() if key in supported}
        )

    @staticmethod
    def _disposition_references_are_valid(
        dispositions: list[SegmentDisposition],
        atom_by_id: dict[str, EvidenceAtom],
        expected_segments: set[int],
    ) -> bool:
        mapped_atom_counts: Counter[str] = Counter()
        for disposition in dispositions:
            if disposition.segment_index not in expected_segments:
                return False
            if disposition.disposition in {"included", "promotion"} and not disposition.atom_ids:
                return False
            if disposition.atom_ids and disposition.disposition not in {"included", "promotion"}:
                return False
            for atom_id in disposition.atom_ids:
                atom = atom_by_id.get(atom_id)
                if atom is None or atom.source_segment != disposition.segment_index:
                    return False
            mapped_atom_counts.update(disposition.atom_ids)
        return mapped_atom_counts == Counter(atom_by_id.keys())

    @staticmethod
    def _is_traceable(atom: EvidenceAtom, segment_count: int) -> bool:
        return bool(atom.source_excerpt.strip()) and 0 <= atom.source_segment < segment_count

    @staticmethod
    def _steps_are_contiguous(atoms: list[EvidenceAtom]) -> bool:
        sequences_by_topic: dict[str, list[int]] = defaultdict(list)
        for atom in atoms:
            if atom.kind != "step":
                continue
            if atom.sequence is None:
                return False
            sequences_by_topic[atom.topic].append(atom.sequence)
        return all(
            sorted(sequences) == list(range(1, max(sequences) + 1))
            for sequences in sequences_by_topic.values()
        )

    @staticmethod
    def _relations_are_valid(
        atoms: list[EvidenceAtom],
        atom_by_id: dict[str, EvidenceAtom],
        *,
        kinds: set[str],
    ) -> bool:
        relevant_atoms = [atom for atom in atoms if atom.kind in kinds]
        return all(
            atom.relations and all(relation_id in atom_by_id for relation_id in atom.relations)
            for atom in relevant_atoms
        )

    @staticmethod
    def _promotions_are_preserved(
        atoms: list[EvidenceAtom],
        dispositions: list[SegmentDisposition],
        included_ids: set[str],
    ) -> tuple[bool, bool]:
        promotional_atoms = [
            atom for atom in atoms if atom.kind == "promotion" or atom.is_promotional
        ]
        dispositions_by_segment: dict[int, list[SegmentDisposition]] = defaultdict(list)
        for disposition in dispositions:
            dispositions_by_segment[disposition.segment_index].append(disposition)

        inclusion_valid = all(atom.id in included_ids for atom in promotional_atoms)
        disposition_valid = all(
            any(
                disposition.disposition in {"promotion", "included"}
                and atom.id in disposition.atom_ids
                for disposition in dispositions_by_segment.get(atom.source_segment, [])
            )
            for atom in promotional_atoms
        )
        return inclusion_valid, disposition_valid

    @staticmethod
    def _quotations_are_preserved(atoms: list[EvidenceAtom], synthesis: SynthesisResult) -> bool:
        quote_atoms = [atom for atom in atoms if atom.kind == "quote"]
        if not quote_atoms:
            return True
        # A quote atom must survive in the dedicated quote section. Matching
        # by source text tolerates harmless punctuation around the quote.
        rendered = " ".join(synthesis.quotes)
        return all(
            atom.source_excerpt in rendered or atom.statement in rendered for atom in quote_atoms
        )

    @staticmethod
    def _synthesis_text(synthesis: SynthesisResult) -> str:
        return json.dumps(synthesis.model_dump(mode="json"), ensure_ascii=False)

    @classmethod
    def _text_categories_are_preserved(
        cls, atoms: list[EvidenceAtom], synthesis: SynthesisResult, *, category: str
    ) -> bool:
        relevant = [atom for atom in atoms if atom.kind == category]
        if not relevant:
            return True
        text = cls._synthesis_text(synthesis)
        return all(atom.source_excerpt in text or atom.statement in text for atom in relevant)

    @classmethod
    def _named_entities_are_preserved(
        cls, atoms: list[EvidenceAtom], synthesis: SynthesisResult
    ) -> bool:
        # AtomKind deliberately stays compact; identify likely proper names by
        # explicit name markers or conservative capitalized Latin tokens.
        candidates = [
            atom
            for atom in atoms
            if re.search(r"(?:姜学长|老师|先生|女士)", atom.statement)
            or any(_looks_like_latin_name(token) for token in atom.statement.split())
        ]
        if not candidates:
            return True
        text = cls._synthesis_text(synthesis)
        return all(atom.source_excerpt in text or atom.statement in text for atom in candidates)

    @classmethod
    def _uncertainty_markers_are_preserved(
        cls, atoms: list[EvidenceAtom], synthesis: SynthesisResult
    ) -> bool:
        markers = [marker for atom in atoms for marker in atom.asr_uncertainty]
        if not markers:
            return True
        text = cls._synthesis_text(synthesis)
        return all(marker in text for marker in markers)

    @staticmethod
    def _thematic_atom_ids(synthesis: SynthesisResult) -> list[str]:
        section_groups = (
            synthesis.themes,
            synthesis.methods_and_steps,
            synthesis.examples_and_counterexamples,
            synthesis.boundaries_and_risks,
            synthesis.promotions,
        )
        return [
            atom_id
            for sections in section_groups
            for section in sections
            for atom_id in section.atom_ids
        ]

    @staticmethod
    def _theme_timestamps_are_valid(
        synthesis: SynthesisResult,
        atom_by_id: dict[str, EvidenceAtom],
    ) -> bool:
        section_groups = (
            synthesis.themes,
            synthesis.methods_and_steps,
            synthesis.examples_and_counterexamples,
            synthesis.boundaries_and_risks,
            synthesis.promotions,
        )
        for sections in section_groups:
            for section in sections:
                expected = Counter(
                    atom.start
                    for atom_id in section.atom_ids
                    if (atom := atom_by_id.get(atom_id)) is not None and atom.start is not None
                )
                if Counter(section.timestamps) != expected:
                    return False
        return True

    @staticmethod
    def _timeline_status(
        synthesis: SynthesisResult,
        dispositions: list[SegmentDisposition],
        segment_count: int,
        atom_by_id: dict[str, EvidenceAtom],
        included_ids: set[str],
    ) -> tuple[bool, bool]:
        expected_order = list(range(segment_count))
        timeline_indexes = [entry.segment_index for entry in synthesis.timeline]
        structure_complete = timeline_indexes == expected_order
        missing_only = (
            len(timeline_indexes) < segment_count
            and len(timeline_indexes) == len(set(timeline_indexes))
            and set(timeline_indexes).issubset(expected_order)
            and timeline_indexes == sorted(timeline_indexes)
        )
        structure_foundational_valid = structure_complete or missing_only

        disposition_by_segment = {
            disposition.segment_index: disposition for disposition in dispositions
        }
        mapping_foundational_valid = True
        mapping_complete = True
        for entry in synthesis.timeline:
            disposition = disposition_by_segment.get(entry.segment_index)
            if disposition is None or entry.disposition != disposition.disposition:
                mapping_foundational_valid = False
                continue

            timeline_atom_counts = Counter(entry.atom_ids)
            if any(count != 1 for count in timeline_atom_counts.values()):
                mapping_foundational_valid = False
                continue
            if any(
                atom_id not in atom_by_id or atom_id not in included_ids
                for atom_id in entry.atom_ids
            ):
                mapping_foundational_valid = False
                continue

            disposition_atom_counts = Counter(disposition.atom_ids)
            if timeline_atom_counts == disposition_atom_counts:
                continue
            included_disposition_counts = Counter(
                atom_id for atom_id in disposition.atom_ids if atom_id in included_ids
            )
            if (
                timeline_atom_counts == included_disposition_counts
                and included_disposition_counts != disposition_atom_counts
            ):
                mapping_complete = False
                continue
            mapping_foundational_valid = False

        foundational_valid = structure_foundational_valid and mapping_foundational_valid
        complete = structure_complete and mapping_complete and mapping_foundational_valid
        return foundational_valid, complete


__all__ = ["CoverageValidator"]
