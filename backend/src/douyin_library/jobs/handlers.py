from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from douyin_library.jobs.control import JobCancelledError, JobControl
from douyin_library.jobs.queue import JobQueue

JobHandler = Callable[[Any, asyncio.Event], Awaitable[None]]


class JobHandlers:
    def __init__(self, queue: JobQueue, control: JobControl | None = None) -> None:
        self.queue = queue
        self.control = control or JobControl()
        self._handlers: dict[str, JobHandler] = {}

    def register(self, kind: str, handler: JobHandler) -> None:
        self._handlers[kind] = handler

    async def run_one(
        self,
        *,
        kinds: set[str] | None = None,
        worker_id: str | None = None,
    ) -> bool:
        job = self.queue.claim_next(kinds=kinds, worker_id=worker_id)
        if job is None:
            return False
        cancel_event = self.control.register(job.id)
        try:
            await self.run_claimed(job, cancel_event)
        finally:
            self.control.unregister(job.id)
        return True

    async def run_job(
        self,
        job_id: int,
        *,
        ignore_schedule: bool = False,
        worker_id: str | None = None,
    ) -> None:
        job = self.queue.claim(
            job_id,
            ignore_schedule=ignore_schedule,
            worker_id=worker_id,
        )
        cancel_event = self.control.register(job.id)
        try:
            await self.run_claimed(job, cancel_event)
        finally:
            self.control.unregister(job.id)

    async def run_claimed(self, job: Any, cancel_event: asyncio.Event) -> None:
        handler = self._handlers.get(job.kind)
        if handler is None:
            self.queue.fail(
                job.id,
                error_code="handler_not_registered",
                error_detail=f"No handler registered for {job.kind}",
            )
            return
        try:
            if cancel_event.is_set():
                raise JobCancelledError("cancelled_by_user")
            await handler(job, cancel_event)
        except JobCancelledError:
            self.queue.session.expire_all()
            current = self.queue.get(job.id)
            if current.status == "running":
                self.queue.cancel(job.id)
            return
        except Exception as exc:  # noqa: BLE001
            self.queue.session.expire_all()
            if self.queue.get(job.id).status == "cancelled":
                return
            self.queue.fail(
                job.id,
                error_code="handler_failed",
                error_detail=str(exc),
            )
        else:
            self.queue.session.expire_all()
            if self.queue.get(job.id).status == "running":
                self.queue.transition(job.id, "succeeded")
