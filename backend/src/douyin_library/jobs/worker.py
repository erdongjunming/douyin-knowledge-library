from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import suppress
from typing import Any

from sqlalchemy.orm import Session

from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.leases import ResourceLeaseStore
from douyin_library.jobs.queue import JobQueue

logger = logging.getLogger(__name__)


class JobWorker:
    def __init__(
        self,
        queue: JobQueue,
        handlers: JobHandlers,
        *,
        kinds: set[str] | None = None,
        heartbeat_interval: float = 10.0,
        worker_name: str = "worker",
        lease_ttl_seconds: float = 120.0,
    ) -> None:
        self.queue = queue
        self.handlers = handlers
        self.kinds = kinds
        self.heartbeat_interval = heartbeat_interval
        self.lease_ttl_seconds = lease_ttl_seconds
        self.worker_id = f"{worker_name}-{uuid.uuid4().hex[:12]}"
        self._stop = asyncio.Event()

    async def run(self, *, poll_interval: float = 1.0) -> None:
        while not self._stop.is_set():
            if not await self.run_one():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=poll_interval)
                except TimeoutError:
                    pass

    async def run_one(self) -> bool:
        """Claim and execute one job, returning ``False`` when the pool is idle."""

        job = self.queue.claim_next(kinds=self.kinds, worker_id=self.worker_id)
        if job is None:
            return False
        await self._run_claimed(job)
        return True

    async def run_job(self, job_id: int, *, ignore_schedule: bool = False) -> None:
        """Claim and execute one specific job with this worker's lease identity.

        This is used by one-shot commands such as ``transcribe-one``.  It must
        follow the same heartbeat/cancellation path as the polling worker;
        calling ``JobHandlers.run_job`` directly would leave the job without a
        durable owner or a cross-process cancellation watcher.
        """

        job = self.queue.claim(
            job_id,
            ignore_schedule=ignore_schedule,
            worker_id=self.worker_id,
        )
        await self._run_claimed(job)

    async def _run_claimed(self, job: Any) -> None:
        cancel_event = self.handlers.control.register(job.id)
        heartbeat_task = asyncio.create_task(
            self._heartbeat(job.id, cancel_event),
            name=f"job-heartbeat-{job.id}",
        )
        try:
            await self.handlers.run_claimed(job, cancel_event)
        except asyncio.CancelledError:
            # Cancelling the worker task is a shutdown path.  Requeue the job
            # only while this worker still owns it; the DB cancellation path
            # has already moved a remotely cancelled job to ``cancelled``.
            self.queue.release(job.id, worker_id=self.worker_id)
            raise
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            self.handlers.control.unregister(job.id)

    async def _heartbeat(
        self,
        job_id: int,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        bind = self.queue.session.get_bind()
        lease_store = ResourceLeaseStore(bind)
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                with Session(bind) as session:
                    queue = JobQueue(session)
                    if not queue.heartbeat(job_id, worker_id=self.worker_id):
                        if cancel_event is not None:
                            cancel_event.set()
                        logger.warning(
                            "Job heartbeat lost ownership; requesting cancellation "
                            "(job_id=%s, worker_id=%s)",
                            job_id,
                            self.worker_id,
                        )
                        return

                # General workers also handle jobs which never acquire a GPU
                # (for example sync and download jobs).  A zero renewal count
                # is only a lease failure when this job currently owns one.
                # The independent store query also avoids holding the job
                # heartbeat transaction while touching the lease table.
                leases_for_job = None
                list_leases = getattr(lease_store, "for_owner", None)
                if callable(list_leases):
                    result = list_leases(self.worker_id, job_id)
                    leases_for_job = list(result) if result is not None else None
                if leases_for_job is not None and not leases_for_job:
                    continue

                renewed = lease_store.renew_for_owner(
                    self.worker_id,
                    job_id,
                    ttl=self.lease_ttl_seconds,
                )
                if not renewed:
                    if cancel_event is not None:
                        cancel_event.set()
                    logger.error(
                        "Resource lease renewal returned no active lease; "
                        "requesting cancellation (job_id=%s, worker_id=%s)",
                        job_id,
                        self.worker_id,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                if cancel_event is not None:
                    cancel_event.set()
                logger.exception(
                    "Job heartbeat or resource lease renewal failed; "
                    "requesting cancellation (job_id=%s, worker_id=%s)",
                    job_id,
                    self.worker_id,
                )
                return

    def stop(self) -> None:
        self._stop.set()
