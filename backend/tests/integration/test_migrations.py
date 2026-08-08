from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


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
