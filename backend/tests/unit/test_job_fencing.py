from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.control import JobCancelledError
from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.worker import JobWorker
from douyin_library.media.storage import StorageLayout
from douyin_library.notes.service import NoteGenerationService
from douyin_library.transcription.service import TranscriptionService


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_late_handler_cannot_finish_a_reclaimed_job(
    tmp_path: Path, outcome: str
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    started = asyncio.Event()
    finish = asyncio.Event()
    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def old_handler(job, cancel_event):
            del job, cancel_event
            started.set()
            await finish.wait()
            if outcome == "error":
                raise RuntimeError("late failure")
            if outcome == "cancel":
                raise JobCancelledError("old claim lost")

        handlers.register("generate_note", old_handler)
        job = queue.create(kind="generate_note")
        job_id = job.id
        task = asyncio.create_task(handlers.run_one(worker_id="same-process"))
        await asyncio.wait_for(started.wait(), timeout=1)
        with Session(engine) as other_session:
            other = JobQueue(other_session)
            assert other.release(job_id, worker_id="same-process")
            replacement = other.claim(job_id, worker_id="same-process")
            replacement_fence = getattr(replacement, "claim_fence", 0)
        finish.set()
        assert await asyncio.wait_for(task, timeout=1)
        session.expire_all()
        current = queue.get(job_id)
        assert current.status == "running"
        assert current.error_code is None
        assert getattr(current, "claim_fence", 0) == replacement_fence
    engine.dispose()


@pytest.mark.asyncio
async def test_worker_shutdown_cannot_release_a_newer_claim(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    started = asyncio.Event()
    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def waiting(job, cancel_event):
            del job, cancel_event
            started.set()
            await asyncio.Event().wait()

        handlers.register("generate_note", waiting)
        job = queue.create(kind="generate_note")
        job_id = job.id
        worker = JobWorker(queue, handlers, heartbeat_interval=10)
        task = asyncio.create_task(worker.run_one())
        await asyncio.wait_for(started.wait(), timeout=1)
        with Session(engine) as other_session:
            other = JobQueue(other_session)
            assert other.release(job_id, worker_id=worker.worker_id)
            other.claim(job_id, worker_id=worker.worker_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        session.expire_all()
        assert queue.get(job_id).status == "running"
    engine.dispose()


def test_note_publication_rejects_owner_changed_after_orm_refresh(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="generate_note")
        job = queue.claim(job.id, worker_id="first-worker")
        job_id = job.id
        original_fence = getattr(job, "claim_fence", None)
        service = NoteGenerationService(
            session, StorageLayout(tmp_path), worker_id="first-worker"
        )
        assert service._owns_generation(
            job,
            None,
            expected_worker_id="first-worker",
            expected_fence=original_fence,
        )
        with Session(engine) as other_session:
            other = JobQueue(other_session)
            assert other.release(job_id, worker_id="first-worker")
            other.claim(job_id, worker_id="second-worker")
        session.refresh(job)
        assert not service._owns_generation(
            job,
            None,
            expected_worker_id="first-worker",
            expected_fence=original_fence,
        )
    engine.dispose()


@pytest.mark.asyncio
async def test_worker_replenishes_after_pause_check_and_before_claim(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    calls: list[str] = []
    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def handler(job, cancel_event):
            del job, cancel_event
            calls.append("run")

        def refill() -> None:
            calls.append("refill")
            queue.create(kind="download_work")

        handlers.register("download_work", handler)
        worker = JobWorker(
            queue,
            handlers,
            idle_callback=refill,
            pause_requested=lambda: True,
        )
        assert await worker.run_one() is False
        assert calls == []
        worker.pause_requested = lambda: False
        assert await worker.run_one() is True
        assert calls == ["refill", "run"]
    engine.dispose()


@pytest.mark.asyncio
async def test_replenishment_failure_keeps_existing_pipeline_runnable(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def handler(job, cancel_event):
            del job, cancel_event

        def broken_refill() -> None:
            raise RuntimeError("scheduler unavailable")

        handlers.register("download_work", handler)
        job = queue.create(kind="download_work")
        worker = JobWorker(queue, handlers, idle_callback=broken_refill)
        assert await worker.run_one() is True
        assert queue.get(job.id).status == "succeeded"
    engine.dispose()


@pytest.mark.asyncio
async def test_transcription_commit_rejects_same_worker_newer_fence(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="transcribe_work")
        job = queue.claim(job.id, worker_id="asr-worker")
        job_id = job.id
        transcriber = TranscriptionService(
            session,
            StorageLayout(tmp_path),
            Settings(storage_root=tmp_path),
            object(),
            worker_id="asr-worker",
        )
        with Session(engine) as other_session:
            other = JobQueue(other_session)
            assert other.release(job_id, worker_id="asr-worker", claim_fence=job.claim_fence)
            replacement = other.claim(job_id, worker_id="asr-worker")
            assert replacement.claim_fence != job.claim_fence
        with pytest.raises(JobCancelledError, match="cancelled_by_user|job_claim_lost"):
            await transcriber._commit_if_owned(job)
        session.expire_all()
        assert queue.get(job_id).status == "running"
    engine.dispose()


def test_note_publication_checks_db_ownership_without_orm_refresh(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="generate_note")
        job = queue.claim(job.id, worker_id="first-worker")
        job_id = job.id
        original_fence = getattr(job, "claim_fence", None)
        service = NoteGenerationService(
            session, StorageLayout(tmp_path), worker_id="first-worker"
        )
        assert service._owns_generation(
            job,
            None,
            expected_worker_id="first-worker",
            expected_fence=original_fence,
        )
        with Session(engine) as other_session:
            other = JobQueue(other_session)
            assert other.release(job_id, worker_id="first-worker")
            other.claim(job_id, worker_id="first-worker")
        assert not service._owns_generation(
            job,
            None,
            expected_worker_id="first-worker",
            expected_fence=original_fence,
        )
    engine.dispose()
