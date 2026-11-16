import asyncio
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.jobs.control import JobCancelledError, JobControl
from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.worker import JobWorker


@pytest.mark.asyncio
async def test_running_job_stays_cancelled_when_handler_exits(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    control = JobControl()
    started = asyncio.Event()
    with Session(engine) as worker_session:
        queue = JobQueue(worker_session)
        handlers = JobHandlers(queue, control)

        async def wait_for_cancel(job, cancel_event: asyncio.Event) -> None:
            del job
            started.set()
            await cancel_event.wait()
            raise JobCancelledError("cancelled_by_user")

        handlers.register("transcribe_work", wait_for_cancel)
        job = queue.create(kind="transcribe_work")
        task = asyncio.create_task(handlers.run_one(worker_id="asr-test"))
        await asyncio.wait_for(started.wait(), timeout=1)

        with Session(engine) as api_session:
            JobQueue(api_session).cancel(job.id)
        assert control.request_cancel(job.id) is True
        assert await asyncio.wait_for(task, timeout=1) is True

        worker_session.expire_all()
        cancelled = queue.get(job.id)
        assert cancelled.status == "cancelled"
        assert cancelled.error_code == "cancelled_by_user"


def test_worker_kind_filter_keeps_asr_separate(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = JobQueue(session)
        download = queue.create(kind="download_work")
        transcribe = queue.create(kind="transcribe_work")

        claimed_general = queue.claim_next(
            kinds={"download_work"},
            worker_id="general-1",
        )
        claimed_asr = queue.claim_next(
            kinds={"transcribe_work"},
            worker_id="asr-1",
        )

        assert claimed_general is not None and claimed_general.id == download.id
        assert claimed_asr is not None and claimed_asr.id == transcribe.id
        assert claimed_general.worker_id == "general-1"
        assert claimed_asr.worker_id == "asr-1"


@pytest.mark.asyncio
async def test_targeted_worker_claims_with_identity_and_observes_remote_cancel(
    tmp_path: Path,
) -> None:
    """A one-shot worker must retain the same durable cancellation contract."""

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    control = JobControl()
    started = asyncio.Event()
    observed_worker_ids: list[str | None] = []

    with Session(engine) as worker_session:
        queue = JobQueue(worker_session)
        handlers = JobHandlers(queue, control)

        async def wait_for_cancel(job, cancel_event: asyncio.Event) -> None:
            observed_worker_ids.append(job.worker_id)
            started.set()
            await cancel_event.wait()
            raise JobCancelledError("cancelled_by_user")

        handlers.register("transcribe_work", wait_for_cancel)
        job = queue.create(kind="transcribe_work")
        worker = JobWorker(
            queue,
            handlers,
            kinds={"transcribe_work"},
            heartbeat_interval=0.01,
            worker_name="asr-one-shot",
        )
        task = asyncio.create_task(worker.run_job(job.id, ignore_schedule=True))
        await asyncio.wait_for(started.wait(), timeout=1)

        with Session(engine) as api_session:
            JobQueue(api_session).cancel(job.id)

        await asyncio.wait_for(task, timeout=1)
        worker_session.expire_all()
        current = queue.get(job.id)
        assert current.status == "cancelled"
        assert observed_worker_ids
        assert observed_worker_ids[0]
        assert observed_worker_ids[0].startswith("asr-one-shot-")
