from pathlib import Path

from sqlalchemy import create_engine

from douyin_library.cli import main
from douyin_library.db import create_schema


def test_rebuild_index_command_accepts_explicit_database(
    tmp_path: Path, capsys
) -> None:
    database_path = tmp_path / "library.db"
    create_schema(create_engine(f"sqlite:///{database_path}"))

    exit_code = main(["rebuild-index", "--database", str(database_path)])

    assert exit_code == 0
    assert "Rebuilt full-text index for 0 works" in capsys.readouterr().out
