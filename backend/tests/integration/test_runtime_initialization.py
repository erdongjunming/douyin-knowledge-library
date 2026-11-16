import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, inspect

import douyin_library.main as main_module
from douyin_library.config import Settings
from douyin_library.db import create_schema
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
async def test_lifespan_starts_and_stops_worker(monkeypatch, tmp_path: Path) -> None:
    class FakeEngine:
        disposed = False

        def dispose(self) -> None:
            self.disposed = True

    class FakeWorker:
        def __init__(self, worker_id: str) -> None:
            self.worker_id = worker_id
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()

        async def run(self) -> None:
            self.started.set()
            await self.stopped.wait()

        def stop(self) -> None:
            self.stopped.set()

    class FakeWorkerRuntime:
        def __init__(self, worker_id: str) -> None:
            self.worker = FakeWorker(worker_id)
            self.queue = type(
                "FakeQueue",
                (),
                {"recover_stale_jobs": lambda self, **kwargs: 0},
            )()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    engine = FakeEngine()
    runtime = type(
        "FakeRuntime",
        (),
        {"engine": engine, "database_path": tmp_path / "library.db"},
    )()
    worker_runtimes = [FakeWorkerRuntime("general-test"), FakeWorkerRuntime("asr-test")]
    built_runtimes = list(worker_runtimes)
    backfilled: list[tuple[str, object]] = []
    monkeypatch.setattr(main_module, "initialize_runtime", lambda settings: runtime)
    real_settings = Settings(
        storage_root=tmp_path,
        job_concurrency=1,
        transcription_concurrency=1,
        embedded_workers=True,
    )
    monkeypatch.setattr(main_module, "Settings", lambda: real_settings)
    monkeypatch.setattr(
        main_module,
        "build_worker_runtime",
        lambda runtime_engine, settings, **kwargs: built_runtimes.pop(0),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "enqueue_missing_transcriptions",
        lambda queue: backfilled.append(("transcriptions", queue)),
    )
    monkeypatch.setattr(
        main_module,
        "enqueue_missing_notes",
        lambda queue, settings: backfilled.append(("notes", queue)),
    )

    async with main_module.lifespan(app):
        await asyncio.wait_for(
            asyncio.gather(*(item.worker.started.wait() for item in worker_runtimes)),
            timeout=0.1,
        )
        assert app.state.worker_runtime is worker_runtimes[0]
        assert backfilled == [
            ("transcriptions", worker_runtimes[0].queue),
            ("notes", worker_runtimes[0].queue),
        ]

    assert all(item.worker.stopped.is_set() for item in worker_runtimes)
    assert all(item.closed for item in worker_runtimes)
    assert engine.disposed is True


@pytest.mark.asyncio
async def test_lifespan_observes_embedded_worker_crash_and_stops_siblings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.disposed = False

        def dispose(self) -> None:
            self.disposed = True

    class FakeQueue:
        def recover_stale_jobs(self, **kwargs: object) -> int:
            del kwargs
            return 0

    class FakeWorker:
        def __init__(self, *, crash: bool) -> None:
            self.crash = crash
            self.worker_id = "embedded-test"
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.stop_calls = 0

        async def run(self) -> None:
            self.started.set()
            if self.crash:
                raise RuntimeError("embedded_worker_boom")
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

    engine = FakeEngine()
    runtime = SimpleNamespace(
        engine=engine,
        database_path=tmp_path / "library.db",
    )
    worker_runtimes = [
        FakeWorkerRuntime(crash=True),
        FakeWorkerRuntime(crash=False),
    ]
    built_runtimes = list(worker_runtimes)
    settings = Settings(
        storage_root=tmp_path,
        job_concurrency=1,
        transcription_concurrency=1,
        embedded_workers=True,
    )
    monkeypatch.setattr(main_module, "initialize_runtime", lambda current: runtime)
    monkeypatch.setattr(main_module, "Settings", lambda: settings)
    monkeypatch.setattr(
        main_module,
        "build_worker_runtime",
        lambda runtime_engine, current_settings, **kwargs: built_runtimes.pop(0),
    )
    monkeypatch.setattr(main_module, "enqueue_missing_transcriptions", lambda queue: None)
    monkeypatch.setattr(main_module, "enqueue_missing_notes", lambda queue, current: None)

    async with main_module.lifespan(app):
        await asyncio.wait_for(worker_runtimes[0].worker.started.wait(), timeout=1)
        await asyncio.wait_for(worker_runtimes[1].worker.started.wait(), timeout=1)
        deadline = asyncio.get_running_loop().time() + 1
        while getattr(app.state, "worker_error", None) is None:
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("embedded worker crash was not observed")
            await asyncio.sleep(0.01)

        assert app.state.worker_error == "RuntimeError: embedded_worker_boom"
        # Error publication precedes sibling cleanup. Wait for the monitor's
        # completion instead of depending on the event loop's scheduling order.
        await asyncio.wait_for(asyncio.shield(app.state.worker_monitor_task), timeout=1)
        assert worker_runtimes[1].worker.cancelled.is_set()
        assert app.state.worker_monitor_task.done()
        assert app.state.worker_monitor_task.exception() is None

    assert all(item.closed for item in worker_runtimes)
    assert engine.disposed is True


@pytest.mark.asyncio
async def test_lifespan_defaults_to_api_only_and_backfills_jobs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    runtime = type(
        "FakeRuntime",
        (),
        {"engine": engine, "database_path": tmp_path / "library.db"},
    )()
    backfilled: list[str] = []
    monkeypatch.setattr(main_module, "initialize_runtime", lambda settings: runtime)
    settings = Settings(storage_root=tmp_path, embedded_workers=False)
    monkeypatch.setattr(main_module, "Settings", lambda: settings)
    monkeypatch.setattr(
        main_module,
        "enqueue_missing_transcriptions",
        lambda queue: backfilled.append("transcriptions"),
    )
    monkeypatch.setattr(
        main_module,
        "enqueue_missing_notes",
        lambda queue, settings: backfilled.append("notes"),
    )

    async with main_module.lifespan(app):
        assert app.state.worker_runtimes == []
        assert app.state.worker_task is None
        assert app.state.worker_runtime is None
        assert backfilled == ["transcriptions", "notes"]

    engine.dispose()


@pytest.mark.asyncio
async def test_lifespan_keeps_http_app_available_when_storage_fails(monkeypatch) -> None:
    def fail_to_initialize(settings: Settings):
        raise OSError("database is read-only")

    monkeypatch.setattr(main_module, "initialize_runtime", fail_to_initialize)

    async with main_module.lifespan(app):
        response = await main_module.health()
        assert response["status"] == "degraded"
        assert response["detail"] == "storage_unavailable"
        assert response["api"]["status"] == "alive"
        assert app.state.runtime is None
        assert app.state.worker_runtime is None


@pytest.mark.asyncio
async def test_backup_loop_continues_after_one_failure(monkeypatch) -> None:
    calls: list[object] = []

    def backup(*args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        calls.append("backup")
        if len(calls) == 1:
            raise RuntimeError("disk busy")
        return {"backup": "ok"}

    async def sleep(seconds: float) -> None:
        del seconds
        if len(calls) >= 2:
            raise StopAsyncIteration

    async def to_thread(function: object, *args: object, **kwargs: object) -> object:
        return function(*args, **kwargs)  # type: ignore[operator]

    monkeypatch.setattr(main_module, "create_database_backup", backup)
    monkeypatch.setattr(main_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(main_module.asyncio, "to_thread", to_thread)
    settings = Settings(
        storage_root=Path("."),
        database_backup_interval_hours=1,
    )

    with pytest.raises(StopAsyncIteration):
        await main_module._database_backup_loop(Path("library.db"), settings)
    assert len(calls) == 2


def test_initialize_runtime_forces_backup_before_pending_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(storage_root=tmp_path / "library")
    database_path = settings.storage_root / "data" / "library.db"
    database_path.parent.mkdir(parents=True)
    database_path.write_bytes(b"existing")
    backup_calls: list[timedelta] = []

    monkeypatch.setattr(
        "douyin_library.runtime.create_database_backup",
        lambda database, **kwargs: backup_calls.append(kwargs["min_interval"]),
    )
    monkeypatch.setattr("douyin_library.runtime._migration_required", lambda path: True)
    monkeypatch.setattr("douyin_library.runtime.command.upgrade", lambda *args, **kwargs: None)

    runtime = initialize_runtime(settings)
    try:
        assert backup_calls == [timedelta(0)]
    finally:
        runtime.engine.dispose()
