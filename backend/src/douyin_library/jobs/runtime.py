from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import exists, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from douyin_library.audit import ProcessingAuditRecorder
from douyin_library.config import Settings
from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue
from douyin_library.jobs.worker import JobWorker
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, Transcript, Work
from douyin_library.security.cookie_store import CookieStore
from douyin_library.sync.creator_sync import CreatorSyncService
from douyin_library.sync.favorites_sync import FavoritesSyncService
from douyin_library.transcription.providers.qwen import QwenAsrProvider
from douyin_library.transcription.service import (
    Transcriber,
    TranscriptionService,
)
from douyin_library.upstream.jiji_gateway import EXPECTED_UPSTREAM_VERSION, JijiGateway


class CookieReader(Protocol):
    def read(self) -> str | None: ...


class MediaProcessor(Protocol):
    async def process(self, work: Work) -> Any: ...


GatewayFactory = Callable[[str], Any]
MediaServiceFactory = Callable[[Session, Any, StorageLayout], MediaProcessor]
TranscriberFactory = Callable[[Settings], Transcriber]


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
    transcriber_factory: TranscriberFactory | None = None,
) -> WorkerRuntime:
    session = Session(engine)
    queue = JobQueue(session)
    handlers = JobHandlers(queue)
    store = cookie_store or CookieStore()
    make_gateway = gateway_factory or JijiGateway.from_cookie_header
    make_media_service = media_service_factory or _build_media_service
    make_transcriber = transcriber_factory or QwenAsrProvider.from_settings

    async def sync_creator(job: Job) -> None:
        payload = _job_payload(job)
        reference = payload.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("sync_creator_reference_missing")
        recorder = ProcessingAuditRecorder(session)
        audit = recorder.start(
            operation="sync_creator",
            processor="jiji-gateway",
            version=EXPECTED_UPSTREAM_VERSION,
            job=job,
            request={"reference": reference},
        )
        try:
            gateway = _gateway_for_job(store, make_gateway)
            try:
                result = await CreatorSyncService(session, gateway).sync(reference)
            finally:
                await _close_gateway(gateway)
            response = {
                "sync_run_id": result.sync_run_id,
                "status": result.status,
                "creator": {
                    "id": result.creator.id,
                    "sec_uid": result.creator.sec_uid,
                    "nickname": result.creator.nickname,
                    "unique_id": result.creator.unique_id,
                },
                "new_work_count": result.new_work_count,
                "new_work_ids": list(result.new_work_ids),
                "discovered_count": result.discovered_count,
                "pages": list(result.audit_pages),
            }
            if result.complete:
                recorder.succeed(
                    audit,
                    response=response,
                    sync_run_id=result.sync_run_id,
                )
            else:
                recorder.fail(
                    audit,
                    response=response,
                    error=result.error_detail or "creator_sync_incomplete",
                    sync_run_id=result.sync_run_id,
                )
            _enqueue_downloads(queue, result.new_work_ids)
            if not result.complete:
                raise RuntimeError("creator_sync_incomplete")
        except Exception as exc:
            if audit.status == "running":
                recorder.fail(audit, error=str(exc))
            raise

    async def transcribe_work(job: Job) -> None:
        if job.work_id is None:
            raise ValueError("transcribe_work_id_missing")
        work = session.get(Work, job.work_id)
        if work is None:
            raise ValueError(f"work_not_found:{job.work_id}")
        job.stage = "transcribe"
        session.commit()
        await TranscriptionService(
            session,
            StorageLayout(settings.storage_root),
            settings,
            make_transcriber(settings),
        ).process(job, work)

    async def sync_favorites(job: Job) -> None:
        recorder = ProcessingAuditRecorder(session)
        audit = recorder.start(
            operation="sync_favorites",
            processor="jiji-gateway",
            version=EXPECTED_UPSTREAM_VERSION,
            job=job,
            request={},
        )
        try:
            gateway = _gateway_for_job(store, make_gateway)
            try:
                result = await FavoritesSyncService(session, gateway).sync_all()
            finally:
                await _close_gateway(gateway)
            response = {
                "sync_run_id": result.sync_run_id,
                "status": result.status,
                "default_count": result.default_count,
                "folder_count": result.folder_count,
                "unique_work_count": result.unique_work_count,
                "new_work_ids": list(result.new_work_ids),
                "pages": list(result.audit_pages),
            }
            if result.complete:
                recorder.succeed(
                    audit,
                    response=response,
                    sync_run_id=result.sync_run_id,
                )
            else:
                recorder.fail(
                    audit,
                    response=response,
                    error=result.error_detail or "favorites_sync_incomplete",
                    sync_run_id=result.sync_run_id,
                )
            _enqueue_downloads(queue, result.new_work_ids)
            if not result.complete:
                raise RuntimeError("favorites_sync_incomplete")
        except Exception as exc:
            if audit.status == "running":
                recorder.fail(audit, error=str(exc))
            raise

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
        _enqueue_work(queue, "transcribe_work", work.id)

    handlers.register("sync_creator", sync_creator)
    handlers.register("sync_favorites", sync_favorites)
    handlers.register("download_work", download_work)
    handlers.register("transcribe_work", transcribe_work)
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
        _enqueue_work(queue, "download_work", work_id)


def _enqueue_work(queue: JobQueue, kind: str, work_id: int) -> None:
    try:
        queue.create(kind=kind, work_id=work_id)
    except DuplicateActiveJobError:
        pass


def _build_media_service(
    session: Session,
    collector: Any,
    layout: StorageLayout,
) -> WorkMediaService:
    return WorkMediaService(session, collector, layout)


def enqueue_missing_transcriptions(queue: JobQueue) -> int:
    work_ids = queue.session.scalars(
        select(Asset.work_id)
        .where(
            Asset.kind == "audio",
            ~exists().where(
                Transcript.work_id == Asset.work_id,
                Transcript.status == "succeeded",
            ),
        )
        .order_by(Asset.work_id)
    ).all()
    created = 0
    for work_id in work_ids:
        try:
            queue.create(kind="transcribe_work", work_id=work_id)
            created += 1
        except DuplicateActiveJobError:
            continue
    return created
