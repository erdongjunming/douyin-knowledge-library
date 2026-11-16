from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.api.dependencies import get_job_control
from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.db import create_schema
from douyin_library.jobs.control import JobControl
from douyin_library.jobs.queue import JobQueue
from douyin_library.main import app
from douyin_library.models import Asset, Job, Transcript, Work


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["download_work", "transcribe_work", "clean_transcript_work"])
@pytest.mark.parametrize("identity", [{"work_id": 7}, {"payload": {"reference": "same"}}])
async def test_retry_rejects_duplicate_pipeline_jobs(
    tmp_path: Path, kind: str, identity: dict,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        failed = queue.create(kind=kind, **identity)
        queue.claim(failed.id)
        queue.transition(failed.id, "failed")
        active = queue.create(kind=kind, **identity)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                response = await client.post(f"/api/jobs/{failed.id}/retry")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "duplicate_active_job"
        assert queue.get(failed.id).status == "failed"
        assert queue.get(active.id).status == "pending"
    engine.dispose()


@pytest.mark.asyncio
async def test_retry_conflict_keeps_the_existing_note_job(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        failed = queue.create(kind="generate_note", payload={"aweme_id": "same"})
        queue.claim(failed.id)
        queue.transition(failed.id, "failed")
        active = queue.create(kind="generate_note", payload={"aweme_id": "same"})
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                response = await client.post(f"/api/jobs/{failed.id}/retry")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
        assert response.status_code == 409
        assert response.json()["detail"] == "duplicate_active_job"
        assert queue.get(failed.id).status == "failed"
        assert queue.get(active.id).status == "pending"
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["download_work", "transcribe_work", "generate_note"])
async def test_explicit_retry_reopens_terminal_failure(tmp_path: Path, kind: str) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind=kind)
        claimed = queue.claim(job.id, worker_id="old-worker")
        old_fence = claimed.claim_fence
        queue.transition(job.id, "failed")
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                response = await client.post(f"/api/jobs/{job.id}/retry")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
        assert response.status_code == 200, response.text
        retried = queue.get(job.id)
        assert retried.status == "pending"
        assert retried.finished_at is None
        assert retried.worker_id is None
        assert retried.heartbeat_at is None
        assert retried.claim_fence > old_fence
        queue.claim(job.id, worker_id="replacement")
        assert not queue.heartbeat(job.id, worker_id="old-worker", claim_fence=old_fence)
    engine.dispose()


@pytest.mark.asyncio
async def test_jobs_api_can_create_and_list_persistent_job(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                created = await client.post("/api/jobs", json={"kind": "sync_all"})
                listed = await client.get("/api/jobs")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert created.status_code == 200
    assert created.json()["status"] == "pending"
    assert listed.status_code == 200
    assert listed.json()["items"][0]["kind"] == "sync_all"


@pytest.mark.asyncio
async def test_jobs_api_rejects_retired_analyze_work_kind(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/jobs",
                    json={"kind": "analyze_work"},
                )
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
            engine.dispose()

    assert response.status_code == 409
    assert response.json()["detail"] == "unsupported_job_kind:analyze_work"


@pytest.mark.asyncio
async def test_jobs_api_cancels_running_worker_signal(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    control = JobControl()

    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="transcribe_work")
        queue.claim_next(worker_id="asr-test")
        event = control.register(job.id)
        app.dependency_overrides[get_job_queue] = lambda: queue
        app.dependency_overrides[get_job_control] = lambda: control
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(f"/api/jobs/{job.id}/cancel")
        finally:
            control.unregister(job.id)
            app.dependency_overrides.pop(get_job_queue, None)
            app.dependency_overrides.pop(get_job_control, None)

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        assert event.is_set()
        assert queue.get(job.id).status == "cancelled"


@pytest.mark.asyncio
async def test_jobs_api_cancels_durable_job_without_embedded_workers(tmp_path: Path) -> None:
    """API-only deployments still cancel jobs owned by a separate worker process."""

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    previous_runtime = getattr(app.state, "runtime", None)
    previous_control = getattr(app.state, "job_control", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    app.state.job_control = None
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="transcribe_work")
        queue.claim(job.id, worker_id="asr-process-1")
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(f"/api/jobs/{job.id}/cancel")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
    try:
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        with Session(engine) as check_session:
            assert JobQueue(check_session).get(job.id).status == "cancelled"
    finally:
        app.state.runtime = previous_runtime
        app.state.job_control = previous_control
        engine.dispose()


@pytest.mark.asyncio
async def test_jobs_api_refresh_recovers_stale_running_jobs(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="generate_note")
        queue.claim(job.id, worker_id="stopped-upgrade")
        stale = queue.get(job.id)
        stale.heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
        session.commit()
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/jobs/refresh")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert response.status_code == 200
    assert response.json()["recovered"] == 1
    assert response.json()["items"][0]["status"] == "pending"
    engine.dispose()


@pytest.mark.asyncio
async def test_jobs_api_bulk_pause_selected_jobs(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        selected = queue.create(kind="generate_note")
        other = queue.create(kind="generate_note", work_id=2)
        selected_id = selected.id
        other_id = other.id
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/jobs/bulk-pause", json={"job_ids": [selected_id]}
                )
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert response.status_code == 200
    assert response.json()["paused"] == 1
    assert response.json()["items"][0]["id"] == selected_id
    assert response.json()["items"][0]["status"] == "paused"
    with Session(engine) as check_session:
        assert JobQueue(check_session).get(other_id).status == "pending"
    engine.dispose()


@pytest.mark.asyncio
async def test_runtime_engine_serves_jobs_and_works(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(tmp_path / "missing-storage"))
    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            jobs = await client.get("/api/jobs")
            works = await client.get("/api/works")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert jobs.status_code == 200
    assert jobs.json()["items"] == []
    assert jobs.json()["total"] == 0
    assert works.status_code == 200
    assert works.json() == {"items": []}


@pytest.mark.asyncio
async def test_jobs_api_hides_a_historical_error_after_retry_succeeds(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="download_work")
        queue.claim_next()
        queue.fail(job.id, error_code="network_error", error_detail="offline")
        retried = queue.get(job.id)
        retried.next_retry_at = None
        session.commit()
        queue.claim_next()
        queue.transition(job.id, "succeeded")
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                listed = await client.get("/api/jobs")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

        audited = queue.get(job.id)
        assert audited.error_code == "network_error"
        assert listed.json()["items"][0]["error_code"] is None


@pytest.mark.asyncio
async def test_jobs_progress_summarizes_unique_audio_work_states(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        works = [Work(aweme_id=f"aweme-{index}") for index in range(1, 6)]
        session.add_all(works)
        session.flush()
        session.add_all(
            Asset(work_id=work.id, kind="audio", path=str(tmp_path / f"{work.id}.m4a"))
            for work in works[:4]
        )
        session.add(
            Transcript(
                work_id=works[0].id,
                engine="qwen3-asr",
                model="Qwen/Qwen3-ASR-0.6B",
                status="succeeded",
                text="done",
            )
        )
        session.add_all(
            [
                Job(kind="transcribe_work", work_id=works[0].id, status="failed"),
                Job(kind="transcribe_work", work_id=works[1].id, status="running"),
                Job(kind="transcribe_work", work_id=works[2].id, status="pending"),
                Job(kind="transcribe_work", work_id=works[3].id, status="failed"),
            ]
        )
        session.commit()

        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/jobs/progress")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert response.status_code == 200
    assert response.json() == {
        "works": {"collected": 5, "audio_ready": 4},
        "audio_extraction": {
            "total": 5,
            "completed": 4,
            "running": 0,
            "pending": 0,
            "failed": 0,
            "not_queued": 1,
            "percent": 80.0,
        },
        "transcription": {
            "total": 4,
            "completed": 1,
            "running": 1,
            "pending": 1,
            "failed": 1,
            "not_queued": 0,
            "percent": 25.0,
        },
    }


@pytest.mark.asyncio
async def test_notes_api_queues_missing_transcripts_and_reports_progress(
    tmp_path: Path, monkeypatch
) -> None:
    storage = tmp_path / "library"
    transcript_dir = storage / "transcript" / "note-731"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.txt").write_text(
        "这是可用于生成笔记的完整转写。", encoding="utf-8"
    )
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        work = Work(aweme_id="note-731", description="知识笔记测试")
        session.add(work)
        session.commit()
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                before = await client.get("/api/notes/progress")
                queued = await client.post("/api/notes/generate", json={})
                after = await client.get("/api/notes/progress")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert before.json()["not_queued"] == 1
    assert queued.status_code == 202
    assert queued.json()["queued"] == 1
    assert after.json()["pending"] == 1
    assert after.json()["recent_aweme_id"] == "note-731"
