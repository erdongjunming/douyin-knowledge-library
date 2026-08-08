from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.models import Source, Work, WorkSource
from douyin_library.sync.reconciliation import SourceReconciler


def test_incomplete_sync_does_not_deactivate_unseen_sources(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        source = Source(kind="favorite_default", external_id="default")
        old = Work(aweme_id="old")
        new = Work(aweme_id="new")
        session.add_all([source, old, new])
        session.flush()
        session.add_all(
            [
                WorkSource(work_id=old.id, source_id=source.id),
                WorkSource(work_id=new.id, source_id=source.id),
            ]
        )
        session.commit()

        outcome = SourceReconciler(session).finish(
            run_status="incomplete",
            seen_by_source={source.id: {new.id}},
            source_kinds={"favorite_default", "favorite_folder"},
        )

        assert outcome.deactivated_count == 0
        assert session.scalars(select(WorkSource.active).order_by(WorkSource.id)).all() == [
            True,
            True,
        ]
