from __future__ import annotations

import asyncio
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
from douyin_library.jobs.control import JobCancelledError, JobControl
from douyin_library.jobs.handlers import JobHandlers
from douyin_library.jobs.leases import ResourceLeaseStore
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue
from douyin_library.jobs.worker import JobWorker
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, Transcript, Work
from douyin_library.notes.service import NoteGenerationService
from douyin_library.security.cookie_store import CookieStore
from douyin_library.sync.creator_sync import CreatorSyncService
from douyin_library.sync.favorites_sync import FavoritesSyncService
from douyin_library.transcription.providers.ollama_cleaner import OllamaTranscriptCleaner
from douyin_library.transcription.providers.qwen import QwenAsrProvider
from douyin_library.transcription.providers.subtitle_local import LocalSubtitleProvider
from douyin_library.transcription.service import (
    Transcriber,
    TranscriptCleaningService,
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
CleanerFactory = Callable[[Settings], Any]
GENERAL_JOB_KINDS = {
    "sync_creator",
    "sync_favorites",
    "download_work",
    "clean_transcript_work",
    "generate_note",
}
ASR_JOB_KINDS = {"transcribe_work"}


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
    cleaner_factory: CleanerFactory | None = None,
    job_control: JobControl | None = None,
    worker_kinds: set[str] | None = None,
    worker_name: str = "worker",
) -> WorkerRuntime:
    session = Session(engine)
    queue = JobQueue(session)
    control = job_control or JobControl()
    handlers = JobHandlers(queue, control)
    lease_store = ResourceLeaseStore(engine)
    storage_layout = StorageLayout(settings.storage_root)
    store = cookie_store or CookieStore()
    make_gateway = gateway_factory or JijiGateway.from_cookie_header
    make_media_service = media_service_factory or _build_media_service
    make_transcriber = transcriber_factory or QwenAsrProvider.from_settings
    make_cleaner = cleaner_factory or (lambda current: OllamaTranscriptCleaner(current))
    subtitle_provider = LocalSubtitleProvider(settings, storage_layout, session=session)

    async def sync_creator(job: Job, cancel_event: asyncio.Event) -> None:
        _raise_if_cancelled(cancel_event)
        payload = _job_payload(job)
        reference = payload.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("sync_creator_reference_missing")
        recorder = ProcessingAuditRecorder(
            session,
            audit_root=settings.storage_root / "audit",
            inline_max_bytes=settings.audit_inline_max_bytes,
        )
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

    async def transcribe_work(job: Job, cancel_event: asyncio.Event) -> None:
        if job.work_id is None:
            raise ValueError("transcribe_work_id_missing")
        work = session.get(Work, job.work_id)
        if work is None:
            raise ValueError(f"work_not_found:{job.work_id}")
        job.stage = "transcribe"
        session.commit()
        await TranscriptionService(
            session,
            storage_layout,
            settings,
            make_transcriber(settings),
            subtitle_provider=subtitle_provider,
            lease_store=lease_store,
            worker_id=job.worker_id,
        ).process(job, work, cancel_event=cancel_event)
        _enqueue_generate_note(
            queue,
            storage_layout,
            aweme_id=work.aweme_id,
            work_id=work.id,
        )
        if settings.transcription_cleaning_enabled:
            _enqueue_clean_transcript(queue, work, settings)

    async def clean_transcript_work(job: Job, cancel_event: asyncio.Event) -> None:
        if job.work_id is None:
            raise ValueError("clean_transcript_work_id_missing")
        work = session.get(Work, job.work_id)
        if work is None:
            raise ValueError(f"work_not_found:{job.work_id}")
        job.stage = "clean"
        session.commit()
        cleaner = make_cleaner(settings)
        payload = _job_payload(job)
        provider = payload.get("provider", "auto")
        if provider == "local":
            provider = "ollama"
        if provider not in {"auto", "ollama"}:
            raise ValueError(f"unsupported_cleaner_provider:{provider}")
        await TranscriptCleaningService(
            session,
            storage_layout,
            settings,
            cleaner,
            lease_store=lease_store,
            worker_id=job.worker_id,
        ).process(
            job,
            work,
            cancel_event=cancel_event,
            raw_transcript_id=int(payload["raw_transcript_id"])
            if payload.get("raw_transcript_id") is not None
            else None,
            raw_fingerprint=str(payload["raw_fingerprint"])
            if payload.get("raw_fingerprint") is not None
            else None,
            cleaner_model=str(payload["model"]) if payload.get("model") is not None else None,
            prompt_version=str(payload["prompt_version"])
            if payload.get("prompt_version") is not None
            else None,
        )

    async def sync_favorites(job: Job, cancel_event: asyncio.Event) -> None:
        _raise_if_cancelled(cancel_event)
        recorder = ProcessingAuditRecorder(
            session,
            audit_root=settings.storage_root / "audit",
            inline_max_bytes=settings.audit_inline_max_bytes,
        )
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

    async def download_work(job: Job, cancel_event: asyncio.Event) -> None:
        _raise_if_cancelled(cancel_event)
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
        _raise_if_cancelled(cancel_event)
        _enqueue_work(queue, "transcribe_work", work.id)

    async def generate_note(job: Job, cancel_event: asyncio.Event) -> None:
        _raise_if_cancelled(cancel_event)
        payload = _job_payload(job)
        work = session.get(Work, job.work_id) if job.work_id is not None else None
        aweme_id = work.aweme_id if work else payload.get("aweme_id")
        if not isinstance(aweme_id, str) or not aweme_id:
            raise ValueError("generate_note_aweme_id_missing")
        force = payload.get("force", False)
        if not isinstance(force, bool):
            raise ValueError("generate_note_force_invalid")
        job.stage = "analyze"
        job.progress = 0.1
        session.commit()
        generation_result = NoteGenerationService(
            session,
            StorageLayout(settings.storage_root),
            settings,
            worker_id=job.worker_id,
            lease_store=lease_store,
        ).generate_one(
            aweme_id,
            force=force,
            job=job,
            cancel_event=cancel_event,
            provider_policy=payload.get("provider_policy"),
            legacy_only=bool(payload.get("legacy_only", False)),
        )
        if inspect.isawaitable(generation_result):
            await generation_result
        job.progress = 0.95
        session.commit()

    handlers.register("sync_creator", sync_creator)
    handlers.register("sync_favorites", sync_favorites)
    handlers.register("download_work", download_work)
    handlers.register("transcribe_work", transcribe_work)
    handlers.register("clean_transcript_work", clean_transcript_work)
    handlers.register("generate_note", generate_note)
    worker = JobWorker(
        queue,
        handlers,
        kinds=worker_kinds,
        heartbeat_interval=settings.job_heartbeat_seconds,
        worker_name=worker_name,
        lease_ttl_seconds=settings.gpu_lease_ttl_seconds,
    )
    return WorkerRuntime(
        session=session,
        queue=queue,
        handlers=handlers,
        worker=worker,
    )


def _raise_if_cancelled(cancel_event: asyncio.Event) -> None:
    if cancel_event.is_set():
        raise JobCancelledError("cancelled_by_user")


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


def enqueue_missing_notes(queue: JobQueue, settings: Settings) -> int:
    layout = StorageLayout(settings.storage_root)
    candidates = NoteGenerationService(queue.session, layout).discover()
    created = 0
    for candidate in candidates:
        if _enqueue_generate_note(
            queue,
            layout,
            aweme_id=candidate.aweme_id,
            work_id=candidate.work_id,
        ):
            created += 1
    return created


def _enqueue_generate_note(
    queue: JobQueue,
    layout: StorageLayout,
    *,
    aweme_id: str,
    work_id: int | None,
) -> bool:
    if (layout.note_dir(aweme_id) / "learning_note.md").is_file():
        return False
    try:
        queue.create(
            kind="generate_note",
            work_id=work_id,
            payload={"aweme_id": aweme_id, "force": False},
        )
    except DuplicateActiveJobError:
        return False
    return True


def _enqueue_clean_transcript(queue: JobQueue, work: Work, settings: Settings) -> bool:
    raw = queue.session.scalar(
        select(Transcript)
        .where(
            Transcript.work_id == work.id,
            Transcript.variant == "raw",
            Transcript.status == "succeeded",
        )
        .order_by(Transcript.id.desc())
    )
    if raw is None:
        return False
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "text": raw.text or "",
                "segments": json.loads(raw.segments_json or "[]"),
                "source_kind": raw.source_kind,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "raw_transcript_id": raw.id,
        "raw_fingerprint": fingerprint,
        "model": settings.transcription_cleaner_model,
        "prompt_version": settings.transcription_cleaner_prompt_version,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    existing = queue.session.scalar(
        select(Job).where(
            Job.kind == "clean_transcript_work",
            Job.work_id == work.id,
            Job.payload_json == payload_json,
        )
    )
    if existing is not None:
        return False
    try:
        queue.create(kind="clean_transcript_work", work_id=work.id, payload=payload)
    except DuplicateActiveJobError:
        return False
    return True
