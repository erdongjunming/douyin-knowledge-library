from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.runtime import ASR_JOB_KINDS, GENERAL_JOB_KINDS
from douyin_library.jobs.worker import JobWorker
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, Work
from douyin_library.transcription.service import SubprocessResult, TranscriptionService


def test_worker_pool_parser_exposes_separate_general_and_asr_pools() -> None:
    from douyin_library.worker_main import build_parser

    parser = build_parser()
    general = parser.parse_args(["--pool", "general", "--concurrency", "2"])
    asr = parser.parse_args(["--pool", "asr", "--concurrency", "1"])

    assert general.pool == "general"
    assert general.concurrency == 2
    assert asr.pool == "asr"
    assert asr.concurrency == 1
    assert GENERAL_JOB_KINDS.isdisjoint(ASR_JOB_KINDS)
    assert "transcribe_work" in ASR_JOB_KINDS
    assert "clean_transcript_work" in GENERAL_JOB_KINDS
    assert "clean_transcript_work" not in ASR_JOB_KINDS


@pytest.mark.asyncio
async def test_worker_heartbeat_observes_remote_cancel(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        job = JobQueue(session).create(kind="transcribe_work")
        claimed = JobQueue(session).claim(job.id, worker_id="asr-test")
        worker = JobWorker(
            JobQueue(session),
            handlers=type("Handlers", (), {})(),
            kinds=ASR_JOB_KINDS,
            heartbeat_interval=0.02,
            worker_name="asr-test",
        )
        event = asyncio.Event()
        with Session(engine) as api_session:
            JobQueue(api_session).cancel(claimed.id)
        task = asyncio.create_task(worker._heartbeat(claimed.id, event))
        await asyncio.wait_for(event.wait(), timeout=1)
        await asyncio.wait_for(task, timeout=1)
    engine.dispose()


@pytest.mark.asyncio
async def test_worker_heartbeat_cancels_when_lease_renewal_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A lost GPU lease must stop the job's cooperative work promptly."""

    import douyin_library.jobs.worker as worker_module

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    renew_calls = 0

    class EmptyLeaseStore:
        def __init__(self, bind: object) -> None:
            del bind

        def for_owner(self, *args: object, **kwargs: object) -> list[object]:
            del args, kwargs
            return [object()]

        def renew_for_owner(self, *args: object, **kwargs: object) -> int:
            nonlocal renew_calls
            del args, kwargs
            renew_calls += 1
            return 0

    monkeypatch.setattr(worker_module, "ResourceLeaseStore", EmptyLeaseStore)
    caplog.set_level(logging.ERROR, logger="douyin_library.jobs.worker")
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="transcribe_work")
        worker = JobWorker(
            queue,
            handlers=type("Handlers", (), {})(),
            kinds=ASR_JOB_KINDS,
            heartbeat_interval=0.01,
            worker_name="asr-test",
        )
        claimed = queue.claim(job.id, worker_id=worker.worker_id)
        event = asyncio.Event()
        task = asyncio.create_task(worker._heartbeat(claimed.id, event))
        try:
            await asyncio.wait_for(event.wait(), timeout=1)
            await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        assert renew_calls == 1
        assert event.is_set()
        assert any(
            "lease" in record.getMessage().lower() and "renew" in record.getMessage().lower()
            for record in caplog.records
        )
    engine.dispose()


@pytest.mark.asyncio
async def test_worker_heartbeat_records_lease_renewal_exception_and_cancels(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A lease backend outage must be observable and signal cancellation."""

    import douyin_library.jobs.worker as worker_module

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    class FailingLeaseStore:
        def __init__(self, bind: object) -> None:
            del bind

        def for_owner(self, *args: object, **kwargs: object) -> list[object]:
            del args, kwargs
            return [object()]

        def renew_for_owner(self, *args: object, **kwargs: object) -> int:
            del args, kwargs
            raise RuntimeError("lease_backend_down")

    monkeypatch.setattr(worker_module, "ResourceLeaseStore", FailingLeaseStore)
    caplog.set_level(logging.ERROR, logger="douyin_library.jobs.worker")
    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="transcribe_work")
        worker = JobWorker(
            queue,
            handlers=type("Handlers", (), {})(),
            kinds=ASR_JOB_KINDS,
            heartbeat_interval=0.01,
            worker_name="asr-test",
        )
        claimed = queue.claim(job.id, worker_id=worker.worker_id)
        event = asyncio.Event()
        task = asyncio.create_task(worker._heartbeat(claimed.id, event))
        try:
            await asyncio.wait_for(event.wait(), timeout=1)
            await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        assert event.is_set()
        assert "lease_backend_down" in caplog.text
    engine.dispose()


@pytest.mark.asyncio
async def test_worker_pool_uses_configured_stale_job_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import douyin_library.worker_main as worker_main

    settings = Settings(
        storage_root=tmp_path,
        job_concurrency=1,
        transcription_concurrency=1,
        job_stale_seconds=47,
    )
    recovered_with: list[timedelta] = []
    worker_started = asyncio.Event()
    worker_stopped = asyncio.Event()

    class FakeQueue:
        def recover_stale_jobs(self, **kwargs: object) -> int:
            recovered_with.append(kwargs["stale_after"])
            return 0

    class FakeWorker:
        async def run(self, *, poll_interval: float) -> None:
            del poll_interval
            worker_started.set()
            await worker_stopped.wait()

        def stop(self) -> None:
            worker_stopped.set()

    class FakeWorkerRuntime:
        def __init__(self) -> None:
            self.queue = FakeQueue()
            self.worker = FakeWorker()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeEngine:
        def __init__(self) -> None:
            self.disposed = False

        def dispose(self) -> None:
            self.disposed = True

    runtime = SimpleNamespace(engine=FakeEngine())
    worker_runtime = FakeWorkerRuntime()
    monkeypatch.setattr(worker_main, "initialize_runtime", lambda current: runtime)
    monkeypatch.setattr(
        worker_main,
        "build_worker_runtime",
        lambda *args, **kwargs: worker_runtime,
    )
    monkeypatch.setattr(worker_main, "enqueue_missing_notes", lambda *args: 0)

    real_event = asyncio.Event
    captured_events: list[asyncio.Event] = []

    def capture_event() -> asyncio.Event:
        event = real_event()
        captured_events.append(event)
        return event

    monkeypatch.setattr(worker_main.asyncio, "Event", capture_event)
    task = asyncio.create_task(
        worker_main.run_pool(
            "general",
            concurrency=1,
            poll_interval=0.01,
            settings=settings,
        )
    )
    try:
        await asyncio.wait_for(worker_started.wait(), timeout=1)
        assert recovered_with == [timedelta(seconds=settings.job_stale_seconds)]
        assert captured_events
        captured_events[0].set()
        await asyncio.wait_for(task, timeout=1)
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert worker_runtime.closed is True
    assert runtime.engine.disposed is True


@pytest.mark.asyncio
async def test_worker_pool_propagates_worker_crash_and_stops_siblings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A crashed worker must wake and tear down the whole pool promptly."""

    import douyin_library.worker_main as worker_main

    class FakeEngine:
        def __init__(self) -> None:
            self.disposed = False

        def dispose(self) -> None:
            self.disposed = True

    class FakeQueue:
        def recover_stale_jobs(self, **kwargs: object) -> int:
            del kwargs
            return 0

    runtimes: list[FakeWorkerRuntime] = []

    class FakeWorker:
        def __init__(self, *, crash: bool) -> None:
            self.crash = crash
            self.stop_calls = 0
            self.cancelled = asyncio.Event()

        async def run(self, *, poll_interval: float) -> None:
            del poll_interval
            if self.crash:
                raise RuntimeError("worker_boom")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

        def stop(self) -> None:
            self.stop_calls += 1

    class FakeWorkerRuntime:
        def __init__(self, *, crash: bool) -> None:
            self.queue = FakeQueue()
            self.worker = FakeWorker(crash=crash)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    runtime = SimpleNamespace(engine=FakeEngine())
    monkeypatch.setattr(worker_main, "initialize_runtime", lambda current: runtime)

    def build_runtime(*args: object, **kwargs: object) -> FakeWorkerRuntime:
        del args
        worker_name = str(kwargs["worker_name"])
        item = FakeWorkerRuntime(crash=worker_name.endswith("-1"))
        runtimes.append(item)
        return item

    monkeypatch.setattr(worker_main, "build_worker_runtime", build_runtime)
    monkeypatch.setattr(worker_main, "enqueue_missing_notes", lambda *args: 0)

    pool_task = asyncio.create_task(
        worker_main.run_pool(
            "general",
            concurrency=2,
            poll_interval=0.01,
            settings=Settings(storage_root=tmp_path),
        )
    )
    try:
        with pytest.raises(RuntimeError, match="worker_boom"):
            await asyncio.wait_for(asyncio.shield(pool_task), timeout=1)
    except TimeoutError:
        pool_task.cancel()
        with suppress(asyncio.CancelledError, RuntimeError):
            await pool_task
        pytest.fail("worker pool stayed alive after a worker crashed")

    assert len(runtimes) == 2
    assert all(item.worker.stop_calls == 1 for item in runtimes)
    assert runtimes[1].worker.cancelled.is_set()
    assert all(item.closed for item in runtimes)
    assert runtime.engine.disposed is True


@pytest.mark.asyncio
async def test_transcription_waits_for_gpu_lease_before_starting_subprocess(
    tmp_path: Path,
) -> None:
    from douyin_library.jobs.leases import ResourceLeaseStore

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "lease-1.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    with Session(engine) as session:
        work = Work(aweme_id="lease-1")
        session.add(work)
        session.flush()
        session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        job = Job(kind="transcribe_work", work_id=work.id, status="running", worker_id="asr")
        session.add(job)
        session.commit()
        work_id, job_id = work.id, job.id

    blocker_store = ResourceLeaseStore(engine)
    blocker = blocker_store.try_acquire("gpu:0", "other", 999, ttl=60)
    assert blocker is not None
    calls = 0

    class FakeTranscriber:
        async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
            nonlocal calls
            calls += 1
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text('{"text":"ok","segments":[]}', encoding="utf-8")
            return SubprocessResult(0, "", "")

    with Session(engine) as session:
        work = session.get(Work, work_id)
        job = session.get(Job, job_id)
        assert work is not None and job is not None
        service = TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage, gpu_lease_poll_seconds=0.02),
            FakeTranscriber(),
            lease_store=ResourceLeaseStore(engine),
            worker_id="asr",
        )
        task = asyncio.create_task(service.process(job, work))
        await asyncio.sleep(0.08)
        assert calls == 0
        assert session.get(Job, job_id).stage == "waiting_gpu"
        assert blocker_store.release(blocker) is True
        await asyncio.wait_for(task, timeout=1)
        assert calls == 1
        assert ResourceLeaseStore(engine).for_owner("asr", job_id) == []
    engine.dispose()
