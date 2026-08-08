from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from douyin_library.db import create_schema, rebuild_work_fts
from douyin_library.models import Analysis, Transcript, Work


def test_fts_tracks_content_and_can_be_rebuilt(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        work = Work(aweme_id="731", description="关于学习方法的口述")
        session.add(work)
        session.flush()
        session.add(
            Transcript(
                work_id=work.id,
                engine="faster-whisper",
                model="small",
                text="理解认知负荷有助于安排复习",
                status="succeeded",
            )
        )
        session.add(
            Analysis(
                work_id=work.id,
                provider="ollama",
                model="qwen",
                prompt_version="v1",
                summary="用间隔重复巩固知识",
                keywords_json='["复习"]',
                topics_json='["认知科学"]',
                status="succeeded",
            )
        )
        session.commit()

    with engine.connect() as connection:
        indexed_row = connection.execute(
            text(
                "SELECT transcript, summary, keywords, topics "
                "FROM work_fts WHERE aweme_id = '731'"
            )
        ).one()
        assert indexed_row == (
            "理解认知负荷有助于安排复习",
            "用间隔重复巩固知识",
            '["复习"]',
            '["认知科学"]',
        )
        indexed = connection.execute(
            text("SELECT aweme_id FROM work_fts WHERE work_fts MATCH :query"),
            {"query": "认知负"},
        ).scalar_one()
        connection.execute(text("DELETE FROM work_fts"))
        connection.commit()

    assert indexed == "731"
    assert rebuild_work_fts(engine) == 1

    with engine.connect() as connection:
        rebuilt = connection.execute(
            text("SELECT aweme_id FROM work_fts WHERE work_fts MATCH :query"),
            {"query": "间隔重"},
        ).scalar_one()

    assert rebuilt == "731"
