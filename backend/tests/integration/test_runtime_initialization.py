from pathlib import Path

from sqlalchemy import inspect

from douyin_library.config import Settings
from douyin_library.runtime import initialize_runtime


def test_runtime_initializes_database_on_first_start(tmp_path: Path) -> None:
    settings = Settings(storage_root=tmp_path / "library")

    runtime = initialize_runtime(settings)

    tables = set(inspect(runtime.engine).get_table_names())
    assert runtime.database_path.is_file()
    assert "work" in tables
    assert "alembic_version" in tables
