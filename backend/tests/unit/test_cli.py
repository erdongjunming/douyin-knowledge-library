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


def test_transcribe_one_command_runs_exact_aweme(monkeypatch, capsys) -> None:
    calls: list[str] = []

    def fake_transcribe(aweme_id: str) -> tuple[int, str]:
        calls.append(aweme_id)
        return 42, "succeeded"

    monkeypatch.setattr("douyin_library.cli.run_transcribe_one", fake_transcribe)

    exit_code = main(["transcribe-one", "--aweme-id", "7670752284527185193"])

    assert exit_code == 0
    assert calls == ["7670752284527185193"]
    assert "job 42 succeeded" in capsys.readouterr().out
