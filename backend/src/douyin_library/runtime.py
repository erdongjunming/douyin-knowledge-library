from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from douyin_library.config import Settings
from douyin_library.maintenance import create_database_backup

BACKEND_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Runtime:
    database_path: Path
    engine: Engine


def database_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def initialize_runtime(settings: Settings) -> Runtime:
    data_dir = settings.storage_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    database_path = data_dir / "library.db"

    alembic_config = Config(str(BACKEND_ROOT / "alembic.ini"))
    alembic_config.set_main_option(
        "script_location",
        str(BACKEND_ROOT / "migrations"),
    )
    alembic_config.set_main_option("sqlalchemy.url", database_url(database_path))

    backup_interval = timedelta(hours=settings.database_backup_interval_hours)
    if _migration_required(database_path):
        # A pending migration can alter or drop columns.  Force a fresh,
        # verified online backup immediately before running Alembic so the
        # operator always has a rollback point, even when the normal interval
        # would otherwise suppress the snapshot.
        backup_interval = timedelta(0)
    create_database_backup(
        database_path,
        backup_root=data_dir / "backups",
        audit_root=settings.storage_root / "audit",
        min_interval=backup_interval,
        retention=settings.database_backup_retention,
    )
    command.upgrade(alembic_config, "head")

    engine = create_engine(
        database_url(database_path),
        connect_args={"timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(connection: object, record: object) -> None:
        del record
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

    return Runtime(
        database_path=database_path,
        engine=engine,
    )


def _migration_required(database_path: Path) -> bool:
    """Return whether the existing database is behind the migration head."""

    path = database_path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        return False
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url(path))
    script = ScriptDirectory.from_config(config)
    heads = set(script.get_heads())
    if not heads:
        return False
    engine = create_engine(database_url(path), connect_args={"timeout": 5})
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current = set(context.get_current_heads())
    except Exception:
        # A non-empty file without a readable Alembic version is treated as a
        # migration candidate.  Forcing a backup is safer than mutating it
        # without a rollback point; Alembic will report the original error.
        return True
    finally:
        engine.dispose()
    return current != heads
