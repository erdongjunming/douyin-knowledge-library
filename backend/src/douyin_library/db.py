from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from douyin_library.models import Base, Source, Work, WorkSource

FTS_TABLE_SQL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS work_fts USING fts5("
    "aweme_id, description, transcript, summary, keywords, topics, tokenize='trigram')"
)

FTS_DOCUMENT_SQL = """
INSERT OR REPLACE INTO work_fts(
    rowid, aweme_id, description, transcript, summary, keywords, topics
)
SELECT
    work.id,
    work.aweme_id,
    COALESCE(work.description, ''),
    COALESCE((
        SELECT group_concat(transcript.text, ' ')
        FROM transcript
        WHERE transcript.work_id = work.id
          AND transcript.status = 'succeeded'
    ), ''),
    COALESCE((
        SELECT group_concat(analysis.summary, ' ')
        FROM analysis
        WHERE analysis.work_id = work.id
          AND analysis.status = 'succeeded'
    ), ''),
    COALESCE((
        SELECT group_concat(analysis.keywords_json, ' ')
        FROM analysis
        WHERE analysis.work_id = work.id
          AND analysis.status = 'succeeded'
    ), ''),
    COALESCE((
        SELECT group_concat(analysis.topics_json, ' ')
        FROM analysis
        WHERE analysis.work_id = work.id
          AND analysis.status = 'succeeded'
    ), '')
FROM work
"""


def create_fts_schema(connection: Connection) -> None:
    connection.exec_driver_sql(FTS_TABLE_SQL)
    refresh_new_work = FTS_DOCUMENT_SQL + " WHERE work.id = NEW.id;"
    refresh_new_owner = FTS_DOCUMENT_SQL + " WHERE work.id = NEW.work_id;"
    refresh_old_owner = FTS_DOCUMENT_SQL + " WHERE work.id = OLD.work_id;"
    triggers = {
        "work_fts_work_insert": (
            "AFTER INSERT ON work BEGIN " + refresh_new_work + " END"
        ),
        "work_fts_work_update": (
            "AFTER UPDATE OF aweme_id, description ON work BEGIN "
            + refresh_new_work
            + " END"
        ),
        "work_fts_work_delete": (
            "AFTER DELETE ON work BEGIN DELETE FROM work_fts WHERE rowid = OLD.id; END"
        ),
        "work_fts_transcript_insert": (
            "AFTER INSERT ON transcript BEGIN " + refresh_new_owner + " END"
        ),
        "work_fts_transcript_update": (
            "AFTER UPDATE ON transcript BEGIN "
            + refresh_old_owner
            + refresh_new_owner
            + " END"
        ),
        "work_fts_transcript_delete": (
            "AFTER DELETE ON transcript BEGIN " + refresh_old_owner + " END"
        ),
        "work_fts_analysis_insert": (
            "AFTER INSERT ON analysis BEGIN " + refresh_new_owner + " END"
        ),
        "work_fts_analysis_update": (
            "AFTER UPDATE ON analysis BEGIN "
            + refresh_old_owner
            + refresh_new_owner
            + " END"
        ),
        "work_fts_analysis_delete": (
            "AFTER DELETE ON analysis BEGIN " + refresh_old_owner + " END"
        ),
    }
    for name, body in triggers.items():
        connection.exec_driver_sql(f"CREATE TRIGGER IF NOT EXISTS {name} {body}")


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        create_fts_schema(connection)


def rebuild_work_fts(engine: Engine) -> int:
    with engine.begin() as connection:
        create_fts_schema(connection)
        connection.exec_driver_sql("DELETE FROM work_fts")
        connection.exec_driver_sql(FTS_DOCUMENT_SQL)
        count = connection.exec_driver_sql("SELECT count(*) FROM work_fts").scalar_one()
    return int(count)


def upsert_work_with_source(
    session: Session,
    *,
    aweme_id: str,
    source_kind: str,
    source_external_id: str,
) -> Work:
    work = session.scalar(select(Work).where(Work.aweme_id == aweme_id))
    if work is None:
        work = Work(aweme_id=aweme_id)
        session.add(work)
        session.flush()

    source = session.scalar(
        select(Source).where(
            Source.kind == source_kind,
            Source.external_id == source_external_id,
        )
    )
    if source is None:
        source = Source(kind=source_kind, external_id=source_external_id)
        session.add(source)
        session.flush()

    relation = session.scalar(
        select(WorkSource).where(
            WorkSource.work_id == work.id,
            WorkSource.source_id == source.id,
        )
    )
    if relation is None:
        session.add(WorkSource(work=work, source=source))

    session.commit()
    return work
