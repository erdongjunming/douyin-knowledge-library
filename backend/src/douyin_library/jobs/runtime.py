from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue
from douyin_library.jobs.worker import JobWorker
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Job, Work
from douyin_library.security.cookie_store import CookieStore
from douyin_library.sync.creator_sync import CreatorSyncService
from douyin_library.sync.favorites_sync import FavoritesSyncService
from douyin_library.upstream.jiji_gateway import JijiGateway


class CookieReader(Protocol):
    def read(self) -> str | None: ...


class MediaProcessor(Protocol):
    async def process(self, work: Work) -> Any: ...


GatewayFactory = Callable[[str], Any]
MediaServiceFactory = Callable[[Session, Any, StorageLayout], MediaProcessor]


@dataclass(slots=True)
class WorkerRuntime:
    session: Session
    queue: JobQueue
    handlers: JobHandlers
    worker: JobWorker

    def close(self) -> None:
        self.session.close()


def build_worker_runtime(
    engine: Engine,
    settings: Settings,
    *,
    cookie_store: CookieReader | None = None,
    gateway_factory: GatewayFactory | None = None,
    media_service_factory: MediaServiceFactory | None = None,
) -> WorkerRuntime:
    session = Session(engine)
    queue = JobQueue(session)
    handlers = JobHandlers(queue)
    store = cookie_store or CookieStore()
    make_gateway = gateway_factory or JijiGateway.from_cookie_header
    make_media_service = media_service_factory or _build_media_service

    async def sync_creator(job: Job) -> None:
        payload = _job_payload(job)
        reference = payload.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("sync_creator_reference_missing")
        gateway = _gateway_for_job(store, make_gateway)
        try:
            result = await CreatorSyncService(session, gateway).sync(reference)
        finally:
            await _close_gateway(gateway)
        _enqueue_downloads(queue, result.new_work_ids)
        if not result.complete:
            raise RuntimeError("creator_sync_incomplete")

    async def sync_favorites(job: Job) -> None:
        gateway = _gateway_for_job(store, make_gateway)
        try:
            result = await FavoritesSyncService(session, gateway).sync_all()
        finally:
            await _close_gateway(gateway)
        _enqueue_downloads(queue, result.new_work_ids)
        if not result.complete:
            raise RuntimeError("favorites_sync_incomplete")

    async def download_work(job: Job) -> None:
        if job.work_id is None:
            raise ValueError("download_work_id_missing")
        work = session.get(Work, job.work_id)
        if work is None:
            raise ValueError(f"work_not_found:{job.work_id}")
        job.stage = "download"
        session.commit()
        gateway = _gateway_for_job(store, make_gateway)
        try:
            service = make_media_service(
                session,
                gateway,
                StorageLayout(settings.storage_root),
            )
            await service.process(work)
        finally:
            await _close_gateway(gateway)

    handlers.register("sync_creator", sync_creator)
    handlers.register("sync_favorites", sync_favorites)
    handlers.register("download_work", download_work)
    return WorkerRuntime(
        session=session,
        queue=queue,
        handlers=handlers,
        worker=JobWorker(queue, handlers),
    )


def _job_payload(job: Job) -> Mapping[str, object]:
    if not job.payload_json:
        return {}
    try:
        payload = json.loads(job.payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("job_payload_invalid_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("job_payload_not_object")
    return payload


def _gateway_for_job(store: CookieReader, factory: GatewayFactory) -> Any:
    cookie = store.read()
    if not cookie:
        raise RuntimeError("douyin_cookie_not_configured")
    return factory(cookie)


async def _close_gateway(gateway: Any) -> None:
    closer = getattr(gateway, "close", None)
    if not callable(closer):
        return
    result = closer()
    if inspect.isawaitable(result):
        await result


def _enqueue_downloads(queue: JobQueue, work_ids: tuple[int, ...]) -> None:
    for work_id in work_ids:
        try:
            queue.create(kind="download_work", work_id=work_id)
        except DuplicateActiveJobError:
            continue


def _build_media_service(
    session: Session,
    collector: Any,
    layout: StorageLayout,
) -> WorkMediaService:
    return WorkMediaService(session, collector, layout)
