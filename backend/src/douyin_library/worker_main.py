"""Standalone worker-pool entry point.

The HTTP process owns the API and database bootstrap.  General and ASR pools
run this module in separate processes so a cancelled ASR child can be
terminated without taking the API down, and so each pool has an explicit
concurrency boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import uuid
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from douyin_library.config import Settings
from douyin_library.jobs.liveness import PoolHeartbeat
from douyin_library.jobs.runtime import (
    ASR_JOB_KINDS,
    GENERAL_JOB_KINDS,
    WorkerRuntime,
    build_worker_runtime,
    enqueue_missing_notes,
    enqueue_missing_transcriptions,
)
from douyin_library.runtime import initialize_runtime

_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_POOL_CONTROL_SCHEMA = "douyin-worker-pool-control-v1"


class PoolControl:
    """Durable claim control; an in-flight job keeps its normal lifecycle."""

    def __init__(self, root: Path, pool: str) -> None:
        worker_kinds(pool)
        self.pool = pool
        self.path = Path(root) / "runtime" / "worker-pools" / f"{pool}.json"

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pool": self.pool,
            "control_file": str(self.path),
            "paused": False,
            "status": "enabled",
            "updated_at": None,
        }
        try:
            value = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if (
                not isinstance(value, dict)
                or value.get("schema") != _POOL_CONTROL_SCHEMA
                or value.get("pool") != self.pool
                or not isinstance(value.get("paused"), bool)
            ):
                raise ValueError("invalid_worker_pool_control")
        except FileNotFoundError:
            return result
        except (OSError, UnicodeError, ValueError):
            result.update(paused=True, status="control_error", error="pool_control_unreadable")
            return result
        result.update(
            paused=value["paused"],
            status="pause_requested" if value["paused"] else "enabled",
            updated_at=value.get("updated_at"),
        )
        return result

    def is_paused(self) -> bool:
        return bool(self.status()["paused"])

    def set_paused(self, paused: bool) -> dict[str, Any]:
        payload = {
            "schema": _POOL_CONTROL_SCHEMA,
            "pool": self.pool,
            "paused": paused,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.status()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Douyin Knowledge Library worker pool")
    parser.add_argument(
        "--pool",
        choices=("general", "asr"),
        required=True,
        help="job family handled by this process",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="number of workers (defaults to the configured pool concurrency)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="seconds between empty-queue polls",
    )
    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--pause", dest="control", action="store_const", const="pause",
        help="persist a pause request; finish the current job before pausing claims",
    )
    control.add_argument(
        "--resume", dest="control", action="store_const", const="resume",
        help="allow the pool to claim queued jobs again",
    )
    control.add_argument(
        "--status", dest="control", action="store_const", const="status",
        help="print the durable pool control state as JSON without starting workers",
    )
    return parser


def worker_kinds(pool: str) -> set[str]:
    if pool == "general":
        return set(GENERAL_JOB_KINDS)
    if pool == "asr":
        return set(ASR_JOB_KINDS)
    raise ValueError("unknown_worker_pool")


async def run_pool(
    pool: str,
    *,
    concurrency: int | None = None,
    poll_interval: float | None = None,
    settings: Settings | None = None,
) -> None:
    current = settings or Settings()
    selected_concurrency = (
        concurrency
        if concurrency is not None
        else (
            current.job_concurrency
            if pool == "general"
            else current.transcription_concurrency
        )
    )
    if selected_concurrency < 1:
        raise ValueError("worker_concurrency_must_be_positive")
    selected_poll = (
        current.worker_poll_seconds if poll_interval is None else poll_interval
    )
    if selected_poll <= 0:
        raise ValueError("worker_poll_interval_must_be_positive")

    runtime = initialize_runtime(current)
    kinds = worker_kinds(pool)
    runtimes = [
        build_worker_runtime(
            runtime.engine,
            current,
            worker_kinds=kinds,
            worker_name=f"{pool}-{index + 1}",
        )
        for index in range(selected_concurrency)
    ]
    control = PoolControl(current.storage_root, pool)
    for item in runtimes:
        item.worker.pause_requested = control.is_paused
    stop_event = asyncio.Event()
    heartbeat = PoolHeartbeat(current.storage_root, pool)
    heartbeat_task: asyncio.Task[None] | None = None
    pool_failed = False
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(signum, stop_event.set)

    try:
        queue = runtimes[0].queue
        queue.recover_stale_jobs(
            stale_after=timedelta(seconds=current.job_stale_seconds)
        )
        if pool == "asr":
            enqueue_missing_transcriptions(queue)
        else:
            enqueue_missing_notes(queue, current)
        tasks = [
            asyncio.create_task(
                item.worker.run(poll_interval=selected_poll),
                name=f"douyin-{pool}-{index + 1}",
            )
            for index, item in enumerate(runtimes)
        ]
        heartbeat_task = asyncio.create_task(heartbeat.run(), name=f"douyin-{pool}-heartbeat")
        monitor_failed = False
        shutdown_error: BaseException | None = None
        try:
            await _monitor_worker_tasks([*tasks, heartbeat_task], stop_event)
        except BaseException:
            monitor_failed = True
            heartbeat_task.cancel()
            with suppress(OSError):
                heartbeat.publish("failed")
            raise
        finally:
            shutdown_error = await _shutdown_worker_tasks(
                runtimes,
                tasks,
                cancel_pending=monitor_failed,
            )
        if shutdown_error is not None:
            raise shutdown_error
    except BaseException as exc:
        pool_failed = not isinstance(exc, asyncio.CancelledError)
        raise
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            # The monitor already surfaced any heartbeat exception. Cleanup
            # must still close workers and the engine if publication failed.
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat_task
        with suppress(OSError):
            heartbeat.publish("failed" if pool_failed else "stopped")
        for item in runtimes:
            item.close()
        runtime.engine.dispose()


async def _monitor_worker_tasks(
    tasks: Sequence[asyncio.Task[None]],
    stop_event: asyncio.Event,
) -> None:
    """Wait for shutdown or surface the first worker that exits unexpectedly."""

    if not tasks:
        await stop_event.wait()
        return

    stop_task = asyncio.create_task(stop_event.wait(), name="douyin-worker-stop-waiter")
    pending: set[asyncio.Task[None]] = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(
                {*pending, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            completed_workers = [task for task in done if task is not stop_task]
            if stop_task in done:
                # A worker may finish cleanly in the same event-loop turn as
                # the shutdown signal.  Treat that as a normal stop, while
                # still surfacing a real worker exception.
                for task in completed_workers:
                    if task.cancelled():
                        continue
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                return
            if completed_workers:
                for task in completed_workers:
                    if task.cancelled():
                        raise RuntimeError("worker_task_cancelled")
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                raise RuntimeError("worker_task_exited_unexpectedly")
    finally:
        stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task


async def _shutdown_worker_tasks(
    runtimes: Sequence[WorkerRuntime],
    tasks: Sequence[asyncio.Task[None]],
    *,
    cancel_pending: bool,
) -> BaseException | None:
    """Stop workers and bound cleanup so a broken handler cannot hang shutdown."""

    for item in runtimes:
        with suppress(Exception):
            item.worker.stop()
    if cancel_pending:
        for task in tasks:
            if not task.done():
                task.cancel()
    if not tasks:
        return None

    results: list[object] = []
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        for task in tasks:
            if not task.done():
                task.cancel()
        with suppress(asyncio.CancelledError, TimeoutError):
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS,
            )
    return next(
        (
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ),
        None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.control is not None:
        control = PoolControl(Settings().storage_root, args.pool)
        try:
            result = (
                control.status()
                if args.control == "status"
                else control.set_paused(args.control == "pause")
            )
        except OSError as exc:
            result = {
                "pool": args.pool, "status": "control_error", "error": str(exc),
                "control_file": str(control.path),
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1 if result["status"] == "control_error" else 0
    try:
        asyncio.run(
            run_pool(
                args.pool,
                concurrency=args.concurrency,
                poll_interval=args.poll_interval,
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_pool", "worker_kinds"]
