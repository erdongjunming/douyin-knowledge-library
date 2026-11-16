from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError


def test_initial_migration_creates_core_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "migrated.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")

    command.upgrade(config, "head")
    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path}")
    tables = set(inspect(engine).get_table_names())
    assert {
        "creator",
        "work",
        "source",
        "work_source",
        "favorite_folder",
        "sync_run",
        "asset",
        "transcript",
        "analysis",
        "job",
        "processing_audit",
        "setting",
        "work_fts",
    }.issubset(tables)

    with engine.connect() as connection:
        triggers = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            ).scalars()
        )

    assert "work_fts_transcript_insert" in triggers
    assert "work_fts_analysis_update" in triggers


def test_resource_lease_migration_is_idempotent_and_indexed(tmp_path: Path) -> None:
    database_path = tmp_path / "lease-migrated.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")

    command.upgrade(config, "0004_job_heartbeat")
    command.upgrade(config, "head")
    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path}")
    inspector = inspect(engine)
    assert "resource_lease" in inspector.get_table_names()
    assert {
        "resource_name",
        "lease_token",
        "owner_worker_id",
        "owner_job_id",
        "acquired_at",
        "heartbeat_at",
        "expires_at",
    } == {column["name"] for column in inspector.get_columns("resource_lease")}
    indexes = {index["name"] for index in inspector.get_indexes("resource_lease")}
    assert {
        "ix_resource_lease_owner_worker_id",
        "ix_resource_lease_owner_job_id",
        "ix_resource_lease_heartbeat_at",
        "ix_resource_lease_expires_at",
    }.issubset(indexes)

    with engine.connect() as connection:
        unique_indexes = {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA index_list(resource_lease)")
            if row[2]
        }
    assert unique_indexes


def test_note_queue_migration_backfills_and_preserves_duplicate_history(tmp_path: Path) -> None:
    database_path = tmp_path / "note-lane-migrated.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    command.upgrade(config, "0006_transcript_variants")
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        # 0001 uses current metadata on a fresh install; simulate the persisted
        # pre-lane shape by dropping only the new indexes/columns.
        connection.exec_driver_sql("DROP INDEX IF EXISTS uq_job_running_legacy")
        connection.exec_driver_sql("DROP INDEX IF EXISTS uq_job_active_note_dedupe")
        connection.exec_driver_sql("DROP INDEX IF EXISTS ix_job_queue_lane")
        connection.exec_driver_sql("DROP INDEX IF EXISTS ix_job_dedupe_key")
        connection.exec_driver_sql("ALTER TABLE job DROP COLUMN queue_lane")
        connection.exec_driver_sql("ALTER TABLE job DROP COLUMN dedupe_key")
        connection.exec_driver_sql("ALTER TABLE job DROP COLUMN claim_fence")
        connection.exec_driver_sql(
            "INSERT INTO work (id, aweme_id, media_type, availability) "
            "VALUES (1, 'old-aweme', 'video', 'available')"
        )
        connection.exec_driver_sql(
            "INSERT INTO job (id, kind, work_id, payload_json, status, stage, "
            "progress, attempts, created_at) VALUES "
            "(1, 'generate_note', 1, '{}', 'pending', 'discover', 0, 0, CURRENT_TIMESTAMP), "
            "(2, 'generate_note', 1, '{\"legacy_only\": true}', "
            "'pending', 'discover', 0, 0, CURRENT_TIMESTAMP), "
            "(3, 'download_work', 1, '{}', 'pending', 'discover', 0, 0, CURRENT_TIMESTAMP)"
        )
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT id, queue_lane, dedupe_key, claim_fence, status, error_code "
                "FROM job ORDER BY id"
            )
        ).mappings().all()
        assert len(rows) == 3
        assert rows[0]["queue_lane"] == "new_note"
        assert rows[0]["dedupe_key"] == "note:old-aweme"
        assert rows[1]["queue_lane"] == "legacy_upgrade"
        assert rows[1]["dedupe_key"] == "note:old-aweme"
        assert rows[1]["status"] == "cancelled"
        assert rows[1]["claim_fence"] == 1
        assert rows[1]["error_code"] == "dedupe_migration_conflict"
        assert rows[2]["queue_lane"] == "pipeline"
        with pytest.raises(IntegrityError):
            connection.exec_driver_sql(
                "INSERT INTO job (kind, work_id, status, stage, progress, attempts, "
                "created_at, queue_lane, dedupe_key, claim_fence) "
                "VALUES ('generate_note', 1, 'pending', 'discover', 0, 0, CURRENT_TIMESTAMP, "
                "'legacy_upgrade', 'note:old-aweme', 0)"
            )
