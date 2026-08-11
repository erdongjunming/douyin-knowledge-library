from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from douyin_library.config import Settings

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
    command.upgrade(alembic_config, "head")

    return Runtime(
        database_path=database_path,
        engine=create_engine(database_url(database_path)),
    )
