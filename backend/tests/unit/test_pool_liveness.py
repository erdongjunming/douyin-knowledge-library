import asyncio
import json
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import douyin_library.worker_main as worker_main
from douyin_library.config import Settings
from douyin_library.jobs.liveness import PoolHeartbeat, pool_liveness


def test_pool_heartbeat_checks_process_and_timestamp(tmp_path: Path) -> None:
    heartbeat = PoolHeartbeat(tmp_path, "general")
    heartbeat.publish()
    assert pool_liveness(tmp_path, "general")["status"] == "alive"
    payload = json.loads(heartbeat.path.read_text(encoding="utf-8"))
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    payload["pid"] = child.pid
    heartbeat.path.write_text(json.dumps(payload), encoding="utf-8")
    assert pool_liveness(tmp_path, "general")["status"] == "unavailable"
    heartbeat.publish()
    payload = json.loads(heartbeat.path.read_text(encoding="utf-8"))
    payload["updated_at"] = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    heartbeat.path.write_text(json.dumps(payload), encoding="utf-8")
    assert pool_liveness(tmp_path, "general")["status"] == "unavailable"


@pytest.mark.asyncio
async def test_pool_heartbeat_failure_closes_worker_runtime(monkeypatch, tmp_path: Path) -> None:
    closed = []
    worker_cancelled = asyncio.Event()

    class Worker:
        async def run(self, **kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                worker_cancelled.set()
                raise

        def stop(self):
            pass

    async def broken_heartbeat(self):
        raise OSError("heartbeat_write_failed")

    runtime = SimpleNamespace(engine=SimpleNamespace(dispose=lambda: closed.append("engine")))
    built = SimpleNamespace(
        queue=SimpleNamespace(recover_stale_jobs=lambda **kwargs: 0),
        worker=Worker(), close=lambda: closed.append("worker"),
    )
    monkeypatch.setattr(worker_main, "initialize_runtime", lambda _: runtime)
    monkeypatch.setattr(worker_main, "build_worker_runtime", lambda *args, **kwargs: built)
    monkeypatch.setattr(worker_main, "enqueue_missing_transcriptions", lambda _: 0)
    monkeypatch.setattr(PoolHeartbeat, "run", broken_heartbeat)
    task = asyncio.create_task(worker_main.run_pool(
        "asr", settings=Settings(storage_root=tmp_path, transcription_concurrency=1),
    ))
    try:
        with pytest.raises(OSError, match="heartbeat_write_failed"):
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
        assert worker_cancelled.is_set()
        assert closed == ["worker", "engine"]
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, OSError):
            await task
