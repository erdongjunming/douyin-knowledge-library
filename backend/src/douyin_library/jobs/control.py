from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass


class JobCancelledError(RuntimeError):
    pass


@dataclass(slots=True)
class _ActiveJob:
    event: asyncio.Event
    loop: asyncio.AbstractEventLoop


class JobControl:
    """Thread-safe bridge between HTTP cancellation and async workers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, _ActiveJob] = {}
        self._requested: set[int] = set()

    def register(self, job_id: int) -> asyncio.Event:
        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        with self._lock:
            requested = job_id in self._requested
            self._active[job_id] = _ActiveJob(event=event, loop=loop)
        if requested:
            event.set()
        return event

    def unregister(self, job_id: int, event: asyncio.Event | None = None) -> None:
        with self._lock:
            active = self._active.get(job_id)
            if active is None or event is None or active.event is event:
                self._active.pop(job_id, None)
                self._requested.discard(job_id)

    def request_cancel(self, job_id: int) -> bool:
        with self._lock:
            self._requested.add(job_id)
            active = self._active.get(job_id)
        if active is None:
            return False
        active.loop.call_soon_threadsafe(active.event.set)
        return True

    def is_active(self, job_id: int) -> bool:
        with self._lock:
            return job_id in self._active
