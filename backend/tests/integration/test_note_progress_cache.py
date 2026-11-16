from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.api.routes import notes
from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.db import create_schema
from douyin_library.jobs.queue import JobQueue
from douyin_library.main import app
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Job


@pytest.mark.asyncio
async def test_inventory_cache_bypasses_refresh_expires_and_observes_completed_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "library"
    layout = StorageLayout(storage)
    layout.initialize()
    layout.transcript_dir("731").mkdir(parents=True)
    layout.transcript_text_path("731").write_text("原始转写", encoding="utf-8")
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    clock = [100.0]
    monkeypatch.setattr(notes, "monotonic", lambda: clock[0], raising=False)
    scans = 0
    snapshot = notes._quality_details_snapshot

    def counted_snapshot(service):
        nonlocal scans
        scans += 1
        return snapshot(service)

    monkeypatch.setattr(notes, "_quality_details_snapshot", counted_snapshot)
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                before = (await client.get("/api/notes/progress")).json()
                assert before["completed"] == 0
                await client.get("/api/notes/progress")
                assert scans == 1

                layout.note_dir("731").mkdir(parents=True)
                layout.note_dir("731").joinpath("learning_note.md").write_text(
                    "# Existing note", encoding="utf-8",
                )
                refreshed = (await client.get("/api/notes/progress?refresh=true")).json()
                assert refreshed["completed"] == 1
                assert scans == 2
                await client.get("/api/notes/progress")
                assert scans == 2

                clock[0] += 60
                await client.get("/api/notes/progress")
                assert scans == 3

                job = queue.create(kind="generate_note", payload={"aweme_id": "731"})
                await client.get("/api/notes/progress")
                layout.note_dir("731").joinpath("learning_note.md").unlink()
                job.status = "succeeded"
                session.commit()
                completed = (await client.get("/api/notes/progress")).json()
                assert completed["completed"] == 0
                assert scans == 5

                await client.post("/api/notes/generate", json={"dry_run": True})
                await client.get("/api/notes/progress")
                assert scans == 6

                other_root = tmp_path / "other"
                monkeypatch.setenv("DKL_STORAGE_ROOT", str(other_root))
                other = (await client.get("/api/notes/progress")).json()
                assert other["total"] == 0
                assert scans == 7

                monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
                other_engine = create_engine(f"sqlite:///{tmp_path / 'other.db'}")
                create_schema(other_engine)
                with Session(other_engine) as other_session:
                    app.dependency_overrides[get_job_queue] = lambda: JobQueue(other_session)
                    await client.get("/api/notes/progress")
                    assert scans == 8
                other_engine.dispose()
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("aweme_ids", [None, ["731"]])
async def test_failed_upgrade_reports_blocked_count_instead_of_active_duplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aweme_ids: list[str] | None,
) -> None:
    storage = tmp_path / "library"
    layout = StorageLayout(storage)
    layout.initialize()
    layout.transcript_dir("731").mkdir(parents=True)
    layout.transcript_text_path("731").write_text("原始转写", encoding="utf-8")
    layout.note_dir("731").mkdir(parents=True)
    layout.note_dir("731").joinpath("learning_note.md").write_text("# Old note")
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(
            kind="generate_note", payload={"aweme_id": "731", "legacy_only": True},
        )
        job.status = "failed"
        session.commit()
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                response = await client.post(
                    "/api/notes/generate", json={"legacy_only": True, "aweme_ids": aweme_ids},
                )
            assert response.status_code == 202
            assert response.json()["queued"] == 0
            assert response.json()["already_queued"] == 0
            assert response.json()["blocked_failed"] == 1
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_only", [False, True])
async def test_generate_api_dry_run_preserves_all_files_and_does_not_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_only: bool,
) -> None:
    storage = tmp_path / "library"
    layout = StorageLayout(storage)
    layout.initialize()
    layout.transcript_dir("731").mkdir(parents=True)
    layout.transcript_text_path("731").write_text("原始转写", encoding="utf-8")
    layout.note_dir("731").mkdir(parents=True)
    layout.note_dir("731").joinpath("learning_note.md").write_text("# Old note")
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            before = {path: path.read_bytes() for path in storage.rglob("*") if path.is_file()}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                response = await client.post("/api/notes/generate", json={
                    "dry_run": True, "legacy_only": legacy_only, "aweme_ids": ["731"],
                })
            assert response.status_code == 200
            assert response.json()["queued"] == 0
            assert session.scalars(select(Job)).all() == []
            assert {
                path: path.read_bytes() for path in storage.rglob("*") if path.is_file()
            } == before
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
    engine.dispose()
