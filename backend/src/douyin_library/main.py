import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy.orm import Session

from douyin_library.api.routes.creators import router as creators_router
from douyin_library.api.routes.favorites import router as favorites_router
from douyin_library.api.routes.jobs import router as jobs_router
from douyin_library.api.routes.notes import router as notes_router
from douyin_library.api.routes.settings import router as settings_router
from douyin_library.api.routes.works import router as works_router
from douyin_library.config import Settings
from douyin_library.jobs.control import JobControl
from douyin_library.jobs.liveness import pool_liveness
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.runtime import (
    ASR_JOB_KINDS,
    GENERAL_JOB_KINDS,
    WorkerRuntime,
    build_worker_runtime,
    enqueue_missing_notes,
    enqueue_missing_transcriptions,
)
from douyin_library.maintenance import create_database_backup
from douyin_library.runtime import initialize_runtime
from douyin_library.worker_main import PoolControl, _monitor_worker_tasks, _shutdown_worker_tasks

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    runtime = None
    worker_runtime = None
    worker_runtimes = []
    worker_tasks: list[asyncio.Task[None]] = []
    worker_monitor_task: asyncio.Task[None] | None = None
    backup_task: asyncio.Task[None] | None = None
    job_control = JobControl()
    startup_error: str | None = None
    try:
        runtime = initialize_runtime(settings)
        if settings.embedded_workers:
            for index in range(settings.job_concurrency):
                worker_runtimes.append(
                    build_worker_runtime(
                        runtime.engine,
                        settings,
                        job_control=job_control,
                        worker_kinds=GENERAL_JOB_KINDS,
                        worker_name=f"general-{index + 1}",
                    )
                )
            for index in range(settings.transcription_concurrency):
                worker_runtimes.append(
                    build_worker_runtime(
                        runtime.engine,
                        settings,
                        job_control=job_control,
                        worker_kinds=ASR_JOB_KINDS,
                        worker_name=f"asr-{index + 1}",
                    )
                )
            worker_runtime = worker_runtimes[0] if worker_runtimes else None
            if worker_runtime is not None:
                worker_runtime.queue.recover_stale_jobs(
                    stale_after=timedelta(seconds=settings.job_stale_seconds)
                )
                enqueue_missing_transcriptions(worker_runtime.queue)
                enqueue_missing_notes(worker_runtime.queue, settings)
        else:
            # Separate worker processes claim the same durable queue.  The API
            # still performs startup reconciliation so a newly started worker
            # sees all missing transcription/note jobs immediately.
            with Session(runtime.engine) as session:
                queue = JobQueue(session)
                queue.recover_stale_jobs(
                    stale_after=timedelta(seconds=settings.job_stale_seconds)
                )
                enqueue_missing_transcriptions(queue)
                enqueue_missing_notes(queue, settings)
    except Exception as exc:  # Keep the HTTP app available when storage is temporarily unavailable.
        startup_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Unable to initialize the local runtime during startup")
        for item in worker_runtimes:
            item.close()
        worker_runtimes.clear()
        worker_runtime = None
        if runtime is not None:
            runtime.engine.dispose()
            runtime = None
    for item in worker_runtimes:
        worker_tasks.append(
            asyncio.create_task(
                item.worker.run(),
                name=f"douyin-library-{item.worker.worker_id}",
            )
        )
    app.state.worker_error = None
    if worker_tasks:
        worker_monitor_task = asyncio.create_task(
            _watch_embedded_workers(app, worker_runtimes, worker_tasks),
            name="douyin-library-worker-monitor",
        )
    if runtime is not None:
        backup_task = asyncio.create_task(
            _database_backup_loop(runtime.database_path, settings),
            name="douyin-library-database-backup",
        )
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.worker_runtime = worker_runtime
    app.state.worker_runtimes = worker_runtimes
    app.state.worker_tasks = worker_tasks
    app.state.worker_task = worker_tasks[0] if worker_tasks else None
    app.state.worker_monitor_task = worker_monitor_task
    app.state.job_control = job_control
    app.state.backup_task = backup_task
    app.state.startup_error = startup_error
    try:
        yield
    finally:
        if backup_task is not None:
            backup_task.cancel()
            with suppress(asyncio.CancelledError):
                await backup_task
        if worker_monitor_task is not None:
            worker_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_monitor_task
        if worker_runtimes and worker_tasks:
            await _shutdown_worker_tasks(
                worker_runtimes,
                worker_tasks,
                cancel_pending=False,
            )
            for item in worker_runtimes:
                item.close()
        if runtime is not None:
            runtime.engine.dispose()
        # Do not leak a previous startup state into an in-process restart or test.
        app.state.runtime = None
        app.state.settings = None
        app.state.worker_runtime = None
        app.state.worker_runtimes = []
        app.state.worker_tasks = []
        app.state.worker_task = None
        app.state.worker_monitor_task = None
        app.state.worker_error = None
        app.state.job_control = None
        app.state.backup_task = None
        app.state.startup_error = None


app = FastAPI(title="Douyin Knowledge Library", lifespan=lifespan)
app.include_router(settings_router)
app.include_router(creators_router)
app.include_router(favorites_router)
app.include_router(jobs_router)
app.include_router(notes_router)
app.include_router(works_router)


async def _watch_embedded_workers(
    app: FastAPI,
    runtimes: list[WorkerRuntime],
    tasks: list[asyncio.Task[None]],
) -> None:
    """Consume worker failures and shut down sibling workers in-process."""

    try:
        await _monitor_worker_tasks(tasks, asyncio.Event())
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        app.state.worker_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Embedded worker task failed; stopping the worker pool")
        try:
            await _shutdown_worker_tasks(
                runtimes,
                tasks,
                cancel_pending=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unable to finish embedded worker shutdown")


@app.get("/api/health")
async def health() -> dict:
    result = {"status": "ok", "api": {"status": "alive"}}
    settings = getattr(app.state, "settings", None)
    worker_error = getattr(app.state, "worker_error", None)
    workers = {"mode": "uninitialized", "status": "unknown"}
    if settings is not None and settings.embedded_workers:
        tasks = getattr(app.state, "worker_tasks", [])
        failed = any(task.done() for task in tasks)
        workers = {
            "mode": "embedded",
            "status": "failed" if failed else ("alive" if tasks else "unavailable"),
            "running_tasks": sum(not task.done() for task in tasks),
        }
        if workers["status"] != "alive":
            result.update(status="degraded", detail="worker_unavailable")
    elif settings is not None:
        pools = {}
        unavailable = False
        for pool in ("general", "asr"):
            evidence = pool_liveness(settings.storage_root, pool)
            control = PoolControl(settings.storage_root, pool).status()
            evidence["paused"] = control["paused"]
            pools[pool] = evidence
            # Pausing claims is intentional control state, independent of
            # whether a runtime heartbeat is present.
            if evidence["status"] != "alive" and not evidence["paused"]:
                unavailable = True
        workers = {
            "mode": "independent",
            "status": "alive" if all(p["status"] == "alive" for p in pools.values())
            else "unavailable",
            "pools": pools,
        }
        if unavailable:
            result.update(status="degraded", detail="worker_unavailable")
    if worker_error:
        workers["status"] = "failed"
        result.update(status="degraded", detail="worker_failed")
    result["workers"] = workers
    startup_error = getattr(app.state, "startup_error", None)
    if startup_error:
        # Do not expose the exception text or filesystem paths to the browser.
        result.update(status="degraded", detail="storage_unavailable")
    return result


async def _database_backup_loop(database_path: Path, settings: Settings) -> None:
    interval_seconds = settings.database_backup_interval_hours * 3600
    while True:
        await asyncio.sleep(min(interval_seconds, 3600))
        try:
            result = await asyncio.to_thread(
                create_database_backup,
                database_path,
                backup_root=settings.storage_root / "data" / "backups",
                audit_root=settings.storage_root / "audit",
                min_interval=timedelta(hours=settings.database_backup_interval_hours),
                retention=settings.database_backup_retention,
            )
            # A defensive check for test/adaptor implementations that return
            # an exception object instead of raising it.
            if isinstance(result, BaseException):
                raise result
        except asyncio.CancelledError:
            raise
        except Exception:
            # A transient lock/disk error must not silently kill the maintenance
            # task.  The next interval retries the operation.
            logger.exception("Automatic database backup failed")
