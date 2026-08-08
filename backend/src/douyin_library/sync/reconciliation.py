from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.models import Source, WorkSource


@dataclass(frozen=True, slots=True)
class ReconciliationOutcome:
    deactivated_count: int


class SourceReconciler:
    def __init__(self, session: Session) -> None:
        self.session = session

    def finish(
        self,
        *,
        run_status: str,
        seen_by_source: dict[int, set[int]],
        source_kinds: set[str],
    ) -> ReconciliationOutcome:
        if run_status != "complete":
            return ReconciliationOutcome(deactivated_count=0)

        deactivated_count = 0
        source_ids = self.session.scalars(
            select(Source.id).where(Source.kind.in_(source_kinds))
        ).all()
        for source_id in source_ids:
            seen_work_ids = seen_by_source.get(source_id, set())
            relations = self.session.scalars(
                select(WorkSource).where(WorkSource.source_id == source_id)
            )
            for relation in relations:
                if relation.active and relation.work_id not in seen_work_ids:
                    relation.active = False
                    deactivated_count += 1
        return ReconciliationOutcome(deactivated_count=deactivated_count)
