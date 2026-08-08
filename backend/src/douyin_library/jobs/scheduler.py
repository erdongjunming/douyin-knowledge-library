from __future__ import annotations

from dataclasses import dataclass

from apscheduler.schedulers.background import BackgroundScheduler

from douyin_library.jobs.queue import JobQueue


@dataclass(frozen=True, slots=True)
class SyncSchedule:
    cadence: str
    day_of_week: str | None = None
    day: int | None = None
    hour: int = 3
    minute: int = 0


class SyncScheduler:
    def __init__(self, queue: JobQueue) -> None:
        self.queue = queue
        self.scheduler = BackgroundScheduler()

    def add_sync_schedule(self, schedule: SyncSchedule) -> None:
        if schedule.cadence == "weekly":
            self.scheduler.add_job(
                self._enqueue_sync,
                "cron",
                day_of_week=schedule.day_of_week or "sun",
                hour=schedule.hour,
                minute=schedule.minute,
            )
        elif schedule.cadence == "monthly":
            self.scheduler.add_job(
                self._enqueue_sync,
                "cron",
                day=schedule.day or 1,
                hour=schedule.hour,
                minute=schedule.minute,
            )
        else:
            raise ValueError("unsupported_sync_cadence")

    def _enqueue_sync(self) -> None:
        self.queue.create(kind="sync_all")

    def start(self) -> None:
        self.scheduler.start()

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)
