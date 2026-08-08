from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from douyin_library.db import create_schema, upsert_work_with_source
from douyin_library.models import Source, Transcript, Work


def test_same_aweme_from_creator_and_favorite_is_one_work(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        creator_work = upsert_work_with_source(
            session,
            aweme_id="731",
            source_kind="creator_post",
            source_external_id="creator:alice",
        )
        favorite_work = upsert_work_with_source(
            session,
            aweme_id="731",
            source_kind="favorite_default",
            source_external_id="favorites:default",
        )

        assert creator_work.id == favorite_work.id
        assert session.scalar(select(Work).where(Work.aweme_id == "731")) is not None
        assert len(session.scalars(select(Source)).all()) == 2


def test_only_successful_transcripts_are_unique_per_engine_and_model(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        work = Work(aweme_id="731")
        session.add(work)
        session.flush()
        session.add_all(
            [
                Transcript(
                    work_id=work.id,
                    engine="faster-whisper",
                    model="small",
                    status="failed",
                ),
                Transcript(
                    work_id=work.id,
                    engine="faster-whisper",
                    model="small",
                    status="pending",
                ),
            ]
        )
        session.commit()

        session.add_all(
            [
                Transcript(
                    work_id=work.id,
                    engine="faster-whisper",
                    model="small",
                    status="succeeded",
                ),
                Transcript(
                    work_id=work.id,
                    engine="faster-whisper",
                    model="small",
                    status="succeeded",
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            session.commit()
