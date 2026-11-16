from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.db import create_schema
from douyin_library.jobs.queue import JobQueue
from douyin_library.main import app


@pytest.mark.asyncio
async def test_jobs_api_exposes_lane_labels_and_lane_counts(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        queue.create(kind="generate_note", payload={"aweme_id": "new-1"})
        queue.create(
            kind="generate_note",
            payload={"aweme_id": "old-1", "legacy_only": True},
        )
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/jobs")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert response.status_code == 200
    body = response.json()
    assert {item["queue_lane"] for item in body["items"]} == {"new_note", "legacy_upgrade"}
    assert {item["lane_label"] for item in body["items"]} == {"新笔记录入", "旧笔记升级"}
    assert body["lane_counts"]["new_note"]["pending"] == 1
    assert body["lane_counts"]["legacy_upgrade"]["pending"] == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_jobs_filters_find_old_paused_tasks_and_paginate(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        old = queue.create(
            kind="generate_note", payload={"aweme_id": "old-paused", "legacy_only": True},
        )
        queue.pause(old.id)
        for index in range(105):
            queue.create(kind="generate_note", payload={"aweme_id": f"new-{index}"})
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                recent = (await client.get("/api/jobs?limit=20")).json()
                second = (await client.get("/api/jobs?limit=20&offset=20")).json()
                paused = (await client.get(
                    "/api/jobs?status=paused&queue_lane=legacy_upgrade",
                )).json()
                resumed = await client.post("/api/jobs/bulk-resume", json={"job_ids": [old.id]})
                queue.pause(old.id)
                cancelled = await client.post(f"/api/jobs/{old.id}/cancel")
                invalid = await client.get("/api/jobs?limit=1000&status=invalid")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
        assert len(recent["items"]) == 20
        assert recent["total"] == 106 and recent["has_more"] is True
        assert not ({job["id"] for job in recent["items"]} &
                    {job["id"] for job in second["items"]})
        assert [job["id"] for job in paused["items"]] == [old.id]
        assert paused["total"] == 1 and paused["has_more"] is False
        assert paused["lane_counts"]["new_note"]["pending"] == 105
        assert resumed.json()["items"][0]["status"] == "pending"
        assert cancelled.json()["status"] == "cancelled"
        assert invalid.status_code == 422
    engine.dispose()


@pytest.mark.asyncio
async def test_failed_legacy_retry_is_explicit_and_fenced(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(
            kind="generate_note", payload={"aweme_id": "failed-old", "legacy_only": True},
        )
        queue.claim(job.id, worker_id="old-worker")
        old_fence = job.claim_fence
        queue.fail(job.id, error_code="note_generation_failed", error_detail="quality_gate_failed")
        assert queue.get(job.id).status == "failed"
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                listed = (await client.get("/api/jobs?status=failed")).json()
                retried = await client.post(f"/api/jobs/{job.id}/retry")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
        assert listed["items"][0]["error_detail"] == "quality_gate_failed"
        assert retried.status_code == 200 and retried.json()["status"] == "pending"
        assert queue.get(job.id).claim_fence > old_fence
        assert not queue.heartbeat(job.id, worker_id="old-worker", claim_fence=old_fence)
    engine.dispose()


@pytest.mark.asyncio
async def test_refresh_reports_recovery_and_bounded_legacy_refill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from douyin_library.api.routes import jobs as jobs_routes

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        queue.create(kind="generate_note", payload={"aweme_id": "old-1", "legacy_only": True})
        monkeypatch.setattr(
            jobs_routes, "_refresh_legacy_upgrade",
            lambda _: {"queued": 2, "reason": "new_queue_empty"},
        )
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/jobs/refresh")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert response.status_code == 200
    body = response.json()
    assert body["legacy_refill"] == {"queued": 2, "reason": "new_queue_empty"}
    assert "recovered_by_lane" in body
    engine.dispose()


@pytest.mark.asyncio
async def test_bulk_pause_returns_per_item_results_for_skipped_jobs(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="generate_note")
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/jobs/bulk-pause",
                    json={"job_ids": [job.id, 999999]},
                )
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert response.status_code == 200
    body = response.json()
    assert body["paused"] == 1
    assert body["skipped"] == 1
    assert {item["id"] for item in body["results"]} == {job.id, 999999}
    assert next(item for item in body["results"] if item["id"] == 999999)["status"] == "skipped"
    engine.dispose()


@pytest.mark.asyncio
async def test_legacy_generate_enables_scheduler_but_waits_for_new_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from douyin_library.jobs.legacy_upgrade import legacy_upgrade_enabled
    from douyin_library.media.storage import StorageLayout

    layout = StorageLayout(tmp_path / "storage")
    layout.initialize()
    layout.transcript_dir("old-1").mkdir(parents=True)
    layout.transcript_text_path("old-1").write_text("old transcript", encoding="utf-8")
    layout.note_dir("old-1").mkdir(parents=True)
    (layout.note_dir("old-1") / "learning_note.md").write_text("# old note", encoding="utf-8")
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(layout.root))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        queue.create(kind="generate_note", payload={"aweme_id": "new-1"})
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/notes/generate", json={"legacy_only": True})
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
        assert legacy_upgrade_enabled(session) is True

    assert response.status_code == 202
    assert response.json()["queued"] == 0
    assert response.json()["legacy_scheduler"]["reason"] == "new_note_runnable"
    engine.dispose()
