import asyncio
from pathlib import Path

import pytest
from sqlalchemy import inspect

import douyin_library.main as main_module
from douyin_library.config import Settings
from douyin_library.main import app
from douyin_library.runtime import initialize_runtime


def test_runtime_initializes_database_on_first_start(tmp_path: Path) -> None:
    settings = Settings(storage_root=tmp_path / "library")

    runtime = initialize_runtime(settings)

    tables = set(inspect(runtime.engine).get_table_names())
    assert runtime.database_path.is_file()
    assert "work" in tables
    assert "alembic_version" in tables


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_worker(monkeypatch) -> None:
    class FakeEngine:
        disposed = False

        def dispose(self) -> None:
            self.disposed = True

    class FakeWorker:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()

        async def run(self) -> None:
            self.started.set()
            await self.stopped.wait()

        def stop(self) -> None:
            self.stopped.set()

    class FakeWorkerRuntime:
        def __init__(self) -> None:
            self.worker = FakeWorker()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    engine = FakeEngine()
    runtime = type("FakeRuntime", (), {"engine": engine})()
    worker_runtime = FakeWorkerRuntime()
    monkeypatch.setattr(main_module, "initialize_runtime", lambda settings: runtime)
    monkeypatch.setattr(
        main_module,
        "build_worker_runtime",
        lambda runtime_engine, settings: worker_runtime,
        raising=False,
    )

    async with main_module.lifespan(app):
        await asyncio.wait_for(worker_runtime.worker.started.wait(), timeout=0.1)
        assert app.state.worker_runtime is worker_runtime

    assert worker_runtime.worker.stopped.is_set()
    assert worker_runtime.closed is True
    assert engine.disposed is True
