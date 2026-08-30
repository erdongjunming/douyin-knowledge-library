from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Transcript, Work


def _alembic_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


def test_migration_backfills_legacy_transcript_without_changing_text(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{database_path}")
    legacy_text = "旧行原文：逐字保留。\n第二行也不能变化。"
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE work ("
            "id INTEGER PRIMARY KEY, aweme_id VARCHAR(40) NOT NULL, description TEXT)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE analysis ("
            "id INTEGER PRIMARY KEY, work_id INTEGER, summary TEXT, "
            "keywords_json TEXT, topics_json TEXT, status VARCHAR(24))"
        )
        connection.exec_driver_sql(
            "CREATE TABLE transcript ("
            "id INTEGER PRIMARY KEY, work_id INTEGER NOT NULL, engine VARCHAR(80) NOT NULL, "
            "model VARCHAR(120) NOT NULL, language VARCHAR(20), text TEXT, "
            "segments_json TEXT, word_timestamps_json TEXT, status VARCHAR(24) NOT NULL, "
            "error TEXT, FOREIGN KEY(work_id) REFERENCES work(id))"
        )
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX uq_transcript_succeeded "
            "ON transcript(work_id, engine, model) WHERE status = 'succeeded'"
        )
        connection.execute(text("INSERT INTO work(id, aweme_id) VALUES (1, 'legacy')"))
        connection.execute(
            text(
                "INSERT INTO transcript("
                "id, work_id, engine, model, text, status"
                ") VALUES (1, 1, 'qwen3-asr', '0.6b', :body, 'succeeded')"
            ),
            {"body": legacy_text},
        )
    engine.dispose()

    config = _alembic_config(database_path)
    command.stamp(config, "0005_resource_lease")
    command.upgrade(config, "head")

    migrated = create_engine(f"sqlite:///{database_path}")
    columns = {column["name"]: column for column in inspect(migrated).get_columns("transcript")}
    assert columns["source_kind"]["nullable"] is False
    assert columns["variant"]["nullable"] is False
    assert columns["parent_transcript_id"]["nullable"] is True
    with migrated.connect() as connection:
        row = connection.execute(
            text(
                "SELECT text, source_kind, variant, parent_transcript_id "
                "FROM transcript WHERE id = 1"
            )
        ).one()
    assert row == (legacy_text, "asr", "raw", None)

    command.downgrade(config, "0005_resource_lease")
    downgraded = create_engine(f"sqlite:///{database_path}")
    assert "variant" not in {
        column["name"] for column in inspect(downgraded).get_columns("transcript")
    }
    with downgraded.begin() as connection:
        connection.execute(
            text("UPDATE transcript SET text = 'downgraded text' WHERE id = 1")
        )
    with downgraded.connect() as connection:
        assert connection.execute(
            text("SELECT transcript FROM work_fts WHERE rowid = 1")
        ).scalar_one() == "downgraded text"


def test_raw_sources_and_cleaned_parent_persist_as_separate_variants(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'variants.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="731")
        session.add(work)
        session.flush()
        subtitle = Transcript(
            work_id=work.id,
            engine="local-subtitle",
            model="embedded",
            source_kind="subtitle",
            variant="raw",
            text="字幕原文",
            status="succeeded",
        )
        asr = Transcript(
            work_id=work.id,
            engine="qwen3-asr",
            model="0.6b",
            source_kind="asr",
            variant="raw",
            text="ASR 原文",
            status="succeeded",
        )
        session.add_all((subtitle, asr))
        session.flush()
        cleaned = Transcript(
            work_id=work.id,
            engine="qwen3-asr",
            model="0.6b",
            source_kind="asr",
            variant="cleaned",
            parent_transcript_id=asr.id,
            text="清洗文本",
            status="succeeded",
        )
        session.add(cleaned)
        session.commit()

        rows = session.scalars(select(Transcript).order_by(Transcript.id)).all()
        assert [(row.source_kind, row.variant) for row in rows] == [
            ("subtitle", "raw"),
            ("asr", "raw"),
            ("asr", "cleaned"),
        ]
        assert cleaned.parent_transcript_id == asr.id

        session.add(
            Transcript(
                work_id=work.id,
                engine="qwen3-asr",
                model="0.6b",
                source_kind="asr",
                variant="raw",
                text="duplicate",
                status="succeeded",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_fts_prefers_valid_cleaned_and_falls_back_to_raw_on_update(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'fts.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="732")
        session.add(work)
        session.flush()
        raw = Transcript(
            work_id=work.id,
            engine="qwen3-asr",
            model="0.6b",
            source_kind="asr",
            variant="raw",
            text="原始检索文本",
            status="succeeded",
        )
        session.add(raw)
        session.commit()

        assert session.execute(
            text("SELECT transcript FROM work_fts WHERE rowid = :id"), {"id": work.id}
        ).scalar_one() == "原始检索文本"

        cleaned = Transcript(
            work_id=work.id,
            engine="ollama",
            model="qwen3:4b",
            source_kind="asr",
            variant="cleaned",
            parent_transcript_id=raw.id,
            text="清洗后的检索文本",
            status="succeeded",
        )
        session.add(cleaned)
        session.commit()
        assert session.execute(
            text("SELECT transcript FROM work_fts WHERE rowid = :id"), {"id": work.id}
        ).scalar_one() == "清洗后的检索文本"

        cleaned.status = "failed"
        session.commit()
        assert session.execute(
            text("SELECT transcript FROM work_fts WHERE rowid = :id"), {"id": work.id}
        ).scalar_one() == "原始检索文本"
        assert session.scalar(select(Transcript.text).where(Transcript.id == raw.id)) == (
            "原始检索文本"
        )


def test_cleaned_paths_do_not_modify_raw_transcript_files(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    raw_text = b"\xef\xbb\xbfraw transcript\r\nsecond line\r\n"
    raw_segments = b'[{"id":1,"text":"raw"}]\r\n'
    raw_text_path = layout.transcript_text_path("733")
    raw_segments_path = layout.transcript_segments_path("733")
    raw_text_path.parent.mkdir(parents=True)
    raw_text_path.write_bytes(raw_text)
    raw_segments_path.write_bytes(raw_segments)

    cleaned_text = layout.transcript_cleaned_text_path("733")
    cleaned_json = layout.transcript_cleaned_json_path("733")
    staged_text = cleaned_text.with_name(f".{cleaned_text.name}.tmp")
    staged_json = cleaned_json.with_name(f".{cleaned_json.name}.tmp")
    staged_text.write_bytes(b"cleaned markdown\n")
    staged_json.write_text(json.dumps({"parent_transcript_id": 1}), encoding="utf-8")
    os.replace(staged_text, cleaned_text)
    os.replace(staged_json, cleaned_json)

    assert raw_text_path.read_bytes() == raw_text
    assert raw_segments_path.read_bytes() == raw_segments
