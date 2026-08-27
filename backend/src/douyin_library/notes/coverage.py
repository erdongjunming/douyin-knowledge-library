from __future__ import annotations

from collections import Counter, defaultdict

from douyin_library.notes.contracts import (
    CoverageReport,
    CoverageReview,
    EvidenceAtom,
    ExtractionResult,
    SegmentDisposition,
    SynthesisResult,
)


class CoverageValidator:
    def validate(
        self,
        *,
        segment_count: int,
        extraction: ExtractionResult,
        synthesis: SynthesisResult,
        review: CoverageReview,
        semantic_generator: str,
    ) -> CoverageReport:
        if segment_count < 0:
            raise ValueError("segment_count must be non-negative")

        expected_segments = set(range(segment_count))
        disposition_counts = Counter(
            disposition.segment_index for disposition in extraction.dispositions
        )
        missing_segment_indexes = sorted(expected_segments - disposition_counts.keys())
        disposition_coverage_valid = (
            set(disposition_counts) == expected_segments
            and all(count == 1 for count in disposition_counts.values())
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
        missing_atom_ids = [
            atom.id for atom in extraction.atoms if atom.id not in included_ids
        ]
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

        review_counts = Counter(review.reviewed_segment_indexes)
        review_coverage_valid = (
            set(review_counts) == expected_segments
            and all(count == 1 for count in review_counts.values())
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
        promotion_inclusion_valid, promotion_dispositions_valid = (
            self._promotions_are_preserved(
                extraction.atoms,
                extraction.dispositions,
                included_ids,
            )
        )

        foundational_error = any(
            (
                not disposition_coverage_valid,
                duplicate_atom_ids,
                not disposition_references_valid,
                unknown_synthesis_references,
                not all_atoms_traceable,
                not steps_verified,
                not examples_verified,
                not conditions_verified,
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
            )
        )
        passed = not foundational_error and not repair_issue

        return CoverageReport(
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
            promotional_content_preserved=(
                promotion_inclusion_valid and promotion_dispositions_valid
            ),
            semantic_generator=semantic_generator,
            quality_gate="passed" if passed else "failed",
            repairable=not passed and not foundational_error and repair_issue,
        )

    @staticmethod
    def _disposition_references_are_valid(
        dispositions: list[SegmentDisposition],
        atom_by_id: dict[str, EvidenceAtom],
        expected_segments: set[int],
    ) -> bool:
        for disposition in dispositions:
            if disposition.segment_index not in expected_segments:
                return False
            for atom_id in disposition.atom_ids:
                atom = atom_by_id.get(atom_id)
                if atom is None or atom.source_segment != disposition.segment_index:
                    return False
        return True

    @staticmethod
    def _is_traceable(atom: EvidenceAtom, segment_count: int) -> bool:
        return bool(atom.source_excerpt.strip()) and 0 <= atom.source_segment < segment_count

    @staticmethod
    def _steps_are_contiguous(atoms: list[EvidenceAtom]) -> bool:
        sequences_by_topic: dict[str, list[int]] = defaultdict(list)
        for atom in atoms:
            if atom.kind == "step" and atom.sequence is not None:
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
            atom.relations
            and all(relation_id in atom_by_id for relation_id in atom.relations)
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


__all__ = ["CoverageValidator"]
