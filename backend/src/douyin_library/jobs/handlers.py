from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from douyin_library.jobs.control import JobCancelledError, JobControl
from douyin_library.jobs.queue import JobQueue
from douyin_library.notes.service import GenerationResult


class JobOutcomeError(RuntimeError):
    """Raised when a handler returns an explicit unsuccessful outcome."""


@dataclass(frozen=True, slots=True)
class JobHandlerOutcome:
    """Optional explicit result contract for handlers with semantic outcomes.

    Existing handlers may continue to return ``None``.  A handler that returns
    a ``GenerationResult`` or wraps one in this type is validated before the
    queue is allowed to transition the job to ``succeeded``.
    """

    result: Any = None
    success: bool = True
    error_code: str | None = None
    error_detail: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.success


JobHandler = Callable[[Any, asyncio.Event], Awaitable[Any]]


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
            outcome = await handler(job, cancel_event)
            validate_handler_outcome(job, outcome)
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


def validate_handler_outcome(job: Any, outcome: Any) -> None:
    if outcome is None:
        return

    if isinstance(outcome, JobHandlerOutcome):
        if not outcome.success:
            detail = outcome.error_detail or "handler_outcome_failed"
            if outcome.error_code:
                detail = f"{outcome.error_code}:{detail}"
            raise JobOutcomeError(detail)
        outcome = outcome.result
        if outcome is None:
            raise JobOutcomeError("handler_outcome_missing_result")

    if not isinstance(outcome, GenerationResult):
        raise JobOutcomeError("handler_returned_unrecognized_outcome")
    if outcome.status != "generated":
        detail = outcome.error or f"generation_status:{outcome.status}"
        raise JobOutcomeError(f"generation_outcome_failed:{detail}")

    # Normal missing-note generation may still use the deterministic rule
    # fallback.  Existing-note upgrades are stricter: only a published,
    # semantic Ollama result is a successful job outcome.
    payload = getattr(job, "payload_json", None)
    if payload is None:
        payload = getattr(job, "payload", None)
    if isinstance(payload, dict):
        is_upgrade = payload.get("legacy_only") is True
    else:
        try:
            is_upgrade = json.loads(payload or "{}").get("legacy_only") is True
        except (TypeError, ValueError, AttributeError):
            is_upgrade = False
    if not is_upgrade:
        return
    if outcome.provider != "ollama":
        raise JobOutcomeError(
            f"generation_outcome_failed:upgrade_provider:{outcome.provider or 'missing'}"
        )
    if outcome.quality_gate != "passed":
        raise JobOutcomeError(
            "generation_outcome_failed:upgrade_quality_gate:"
            f"{outcome.quality_gate or 'missing'}"
        )


__all__ = [
    "JobHandlerOutcome",
    "JobHandlers",
    "JobOutcomeError",
    "validate_handler_outcome",
]
