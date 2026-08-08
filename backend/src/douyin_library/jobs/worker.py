from __future__ import annotations

import asyncio
from datetime import timedelta

from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.queue import JobQueue


class JobWorker:
    def __init__(self, queue: JobQueue, handlers: JobHandlers) -> None:
        self.queue = queue
        self.handlers = handlers
        self._stop = asyncio.Event()

    async def run(self, *, poll_interval: float = 1.0) -> None:
        self.queue.recover_stale_jobs(stale_after=timedelta(minutes=15))
        while not self._stop.is_set():
            handled = await self.handlers.run_one()
            if not handled:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=poll_interval)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()
