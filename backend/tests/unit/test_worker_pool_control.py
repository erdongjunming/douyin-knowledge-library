from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import douyin_library.worker_main as worker_main
from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.worker import JobWorker


@pytest.mark.asyncio
async def test_paused_polling_worker_never_claims() -> None:
    calls = []
    queue = SimpleNamespace(claim_next=lambda **kwargs: calls.append(kwargs))
    worker = JobWorker(queue, SimpleNamespace())
    worker.pause_requested = lambda: True

    assert await worker.run_one() is False
    assert calls == []


def test_control_cli_is_durable_and_does_not_initialize_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(worker_main, "Settings", lambda: Settings(storage_root=tmp_path))
    monkeypatch.setattr(
        worker_main, "initialize_runtime", lambda *_: pytest.fail("control initialized database"),
    )
    assert worker_main.main(["--pool", "asr", "--pause"]) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["paused"] is True
    assert paused["status"] == "pause_requested"
    assert worker_main.main(["--pool", "asr", "--status"]) == 0
    assert json.loads(capsys.readouterr().out)["paused"] is True
    assert worker_main.main(["--pool", "general", "--status"]) == 0
    assert json.loads(capsys.readouterr().out)["paused"] is False
    assert worker_main.main(["--pool", "asr", "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["paused"] is False
    assert worker_main.PoolControl(tmp_path, "asr").is_paused() is False
    assert not (tmp_path / "data" / "library.db").exists()


@pytest.mark.parametrize("bad_content", ["{", "[]", "null", '{"paused":"false"}'])
def test_unreadable_control_keeps_pool_paused(tmp_path: Path, bad_content: str) -> None:
    control = worker_main.PoolControl(tmp_path, "asr")
    control.path.parent.mkdir(parents=True)
    control.path.write_text(bad_content, encoding="utf-8")
    assert control.is_paused() is True
    assert control.status()["status"] == "control_error"
    control.set_paused(False)
    assert control.is_paused() is False


def test_failed_resume_keeps_existing_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    control = worker_main.PoolControl(tmp_path, "asr")
    control.set_paused(True)
    monkeypatch.setattr(worker_main, "Settings", lambda: Settings(storage_root=tmp_path))

    def fail_replace(*args):
        raise OSError("synthetic_write_failure")

    monkeypatch.setattr(worker_main.os, "replace", fail_replace)
    assert worker_main.main(["--pool", "asr", "--resume"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "control_error"
    assert control.is_paused() is True


@pytest.mark.asyncio
async def test_pause_finishes_active_job_and_heartbeats_then_resume_claims_next(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    control = worker_main.PoolControl(tmp_path, "asr")
    entered = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()
    heartbeats = []
    try:
        with Session(engine) as session:
            queue = JobQueue(session)
            first = queue.create(kind="transcribe_work")
            second = queue.create(kind="transcribe_work")
            handlers = JobHandlers(queue)

            async def handle(job, cancel_event):
                if job.id == first.id:
                    entered.set()
                    await finish.wait()
                    assert not cancel_event.is_set()
                else:
                    completed.set()

            handlers.register("transcribe_work", handle)
            worker = JobWorker(queue, handlers, kinds={"transcribe_work"})
            worker.pause_requested = control.is_paused

            async def heartbeat(job_id, cancel_event, claim_fence=None):
                del claim_fence
                while True:
                    heartbeats.append(job_id)
                    await asyncio.sleep(0.005)

            worker._heartbeat = heartbeat
            task = asyncio.create_task(worker.run(poll_interval=0.005))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                control.set_paused(True)
                await asyncio.sleep(0.025)
                assert len(heartbeats) > 1
                assert queue.get(first.id).status == "running"
                assert queue.get(second.id).status == "pending"
                finish.set()
                await asyncio.sleep(0.025)
                assert queue.get(first.id).status == "succeeded"
                assert queue.get(second.id).status == "pending"
                control.set_paused(False)
                await asyncio.wait_for(completed.wait(), timeout=1)
                assert queue.get(second.id).status == "succeeded"
            finally:
                worker.stop()
                finish.set()
                await asyncio.wait_for(task, timeout=1)
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_existing_pause_is_installed_before_pool_starts_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_main.PoolControl(tmp_path, "asr").set_paused(True)
    started = asyncio.Event()
    checks = []

    class Worker:
        async def run(self, **kwargs):
            checks.append(self.pause_requested())
            started.set()
            await asyncio.Event().wait()

        def stop(self):
            pass

    runtime = SimpleNamespace(engine=SimpleNamespace(dispose=lambda: None))
    built = SimpleNamespace(
        queue=SimpleNamespace(recover_stale_jobs=lambda **kwargs: 0),
        worker=Worker(), close=lambda: None,
    )
    monkeypatch.setattr(worker_main, "initialize_runtime", lambda *_: runtime)
    monkeypatch.setattr(worker_main, "build_worker_runtime", lambda *args, **kwargs: built)
    monkeypatch.setattr(worker_main, "enqueue_missing_transcriptions", lambda *_: 0)
    task = asyncio.create_task(
        worker_main.run_pool("asr", settings=Settings(storage_root=tmp_path))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert checks == [True]
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
