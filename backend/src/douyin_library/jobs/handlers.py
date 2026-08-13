from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from douyin_library.jobs.queue import JobQueue

JobHandler = Callable[[Any], Awaitable[None]]


class JobHandlers:
    def __init__(self, queue: JobQueue) -> None:
        self.queue = queue
        self._handlers: dict[str, JobHandler] = {}

    def register(self, kind: str, handler: JobHandler) -> None:
        self._handlers[kind] = handler

    async def run_one(self) -> bool:
        job = self.queue.claim_next()
        if job is None:
            return False
        await self._run_claimed(job)
        return True

    async def run_job(self, job_id: int, *, ignore_schedule: bool = False) -> None:
        job = self.queue.claim(job_id, ignore_schedule=ignore_schedule)
        await self._run_claimed(job)

    async def _run_claimed(self, job: Any) -> None:
        handler = self._handlers.get(job.kind)
        if handler is None:
            self.queue.fail(
                job.id,
                error_code="handler_not_registered",
                error_detail=f"No handler registered for {job.kind}",
            )
            return
        try:
            await handler(job)
        except Exception as exc:  # noqa: BLE001
            self.queue.fail(
                job.id,
                error_code="handler_failed",
                error_detail=str(exc),
            )
        else:
            self.queue.transition(job.id, "succeeded")
