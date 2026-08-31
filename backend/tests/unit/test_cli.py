from pathlib import Path

import pytest
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


@pytest.mark.asyncio
async def test_cli_one_shot_uses_job_worker_execution_path() -> None:
    calls: list[tuple[int, bool]] = []

    class FakeWorker:
        async def run_job(self, job_id: int, *, ignore_schedule: bool = False) -> None:
            calls.append((job_id, ignore_schedule))

    class FakeRuntime:
        worker = FakeWorker()

        # The CLI must not call this direct handler path: it has no heartbeat.
        handlers = object()

    from douyin_library.cli import _run_job

    await _run_job(FakeRuntime(), 73)

    assert calls == [(73, True)]


def test_generate_notes_command_passes_incremental_options(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_generate(**kwargs) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "total_transcripts": 10,
            "completed_before": 2,
            "eligible": 8,
            "selected": 3,
            "generated": 0,
            "skipped": 2,
            "failed": 0,
            "would_generate": 3,
            "dry_run": True,
            "items": [],
        }

    monkeypatch.setattr("douyin_library.cli.run_generate_notes", fake_generate)

    exit_code = main(["generate-notes", "--limit", "3", "--dry-run"])

    assert exit_code == 0
    assert calls == [
        {"limit": 3, "force": False, "dry_run": True, "aweme_ids": None}
    ]
    assert '"would_generate": 3' in capsys.readouterr().out


def test_generate_notes_command_passes_provider_and_legacy_controls(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_generate(**kwargs) -> dict[str, object]:
        calls.append(kwargs)
        return {"failed": 0, "dry_run": True}

    monkeypatch.setattr("douyin_library.cli.run_generate_notes", fake_generate)

    assert main([
        "generate-notes",
        "--aweme-id", "731",
        "--aweme-id", "732",
        "--provider", "ollama",
        "--legacy-only",
        "--dry-run",
    ]) == 0
    assert calls == [{
        "limit": None,
        "force": False,
        "dry_run": True,
        "aweme_ids": {"731", "732"},
        "provider_policy": "local",
        "legacy_only": True,
    }]


def test_generate_notes_rejects_force_legacy_only(monkeypatch) -> None:
    monkeypatch.setattr("douyin_library.cli.run_generate_notes", lambda **kwargs: {})
    import pytest

    with pytest.raises(SystemExit):
        main(["generate-notes", "--force", "--legacy-only"])


def test_clean_transcripts_dry_run_reports_selected_ids_without_writes(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_clean(**kwargs) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "selected": 2,
            "items": [
                {"aweme_id": "731", "status": "would_clean"},
                {"aweme_id": "732", "status": "would_clean"},
            ],
            "dry_run": True,
            "failed": 0,
        }

    monkeypatch.setattr("douyin_library.cli.run_clean_transcripts", fake_clean)

    assert main([
        "clean-transcripts",
        "--aweme-id", "731",
        "--aweme-id", "732",
        "--limit", "3",
        "--provider", "local",
        "--model", "fake-model",
        "--dry-run",
    ]) == 0
    assert calls == [{
        "aweme_ids": {"731", "732"},
        "limit": 3,
        "dry_run": True,
        "provider": "ollama",
        "model": "fake-model",
    }]
    output = capsys.readouterr().out
    assert '"would_clean"' in output
    assert '"731"' in output and '"732"' in output
