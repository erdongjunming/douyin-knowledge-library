import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import douyin_library.jobs.queue as queue_module
from douyin_library.db import create_schema
from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.worker import JobWorker


@pytest.mark.asyncio
async def test_fast_restart_recovers_later_stale_job_while_pool_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    now = datetime.now(UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(queue_module, "datetime", Clock)
    with Session(engine) as session:
        queue = JobQueue(session)
        abandoned = queue.create(kind="download_work")
        queue.claim(abandoned.id, worker_id="crashed-worker")
        fresh = queue.create(kind="download_work")
        queue.claim(fresh.id, worker_id="other-live-worker")
        assert queue.recover_stale_jobs(stale_after=timedelta(seconds=30)) == 0
        worker = JobWorker(queue, JobHandlers(queue), pause_requested=lambda: True)
        worker.stale_after_seconds = 30
        worker.recovery_interval_seconds = 0.01
        task = asyncio.create_task(worker.run(poll_interval=0.005))
        try:
            await asyncio.sleep(0.025)
            now += timedelta(seconds=31)
            queue.heartbeat(fresh.id, worker_id="other-live-worker")
            deadline = asyncio.get_running_loop().time() + 0.3
            while True:
                session.expire_all()
                if queue.get(abandoned.id).status == "pending":
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("worker never recovered a job which became stale after restart")
                await asyncio.sleep(0.005)
            assert queue.get(abandoned.id).error_code == "worker_stale_recovered"
            assert queue.get(fresh.id).status == "running"
            assert queue.get(fresh.id).worker_id == "other-live-worker"
        finally:
            worker.stop()
            await asyncio.wait_for(task, timeout=1)
    engine.dispose()


def test_runtime_passes_configured_stale_threshold(tmp_path: Path) -> None:
    from douyin_library.config import Settings
    from douyin_library.jobs.runtime import build_worker_runtime

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    runtime = build_worker_runtime(
        engine, Settings(storage_root=tmp_path, job_stale_seconds=47),
        transcriber_factory=lambda _: object(),
    )
    try:
        assert runtime.worker.stale_after_seconds == 47
        assert 0 < runtime.worker.recovery_interval_seconds <= 30
    finally:
        runtime.close()
        engine.dispose()
