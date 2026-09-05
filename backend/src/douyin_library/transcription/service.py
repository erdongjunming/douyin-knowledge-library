from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from douyin_library.audit import ProcessingAuditRecorder
from douyin_library.config import Settings, resolve_tool
from douyin_library.jobs.control import JobCancelledError
from douyin_library.jobs.leases import ResourceLeaseStore
from douyin_library.media.metadata import file_sha256
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, Transcript, Work
from douyin_library.notes.publisher import note_publication_lock
from douyin_library.transcription.contracts import (
    CleanerResult,
    TranscriptDraft,
    TranscriptSegment,
)

DEFAULT_QWEN_PYTHON = (
    Path.home() / ".cache" / "rimagination-notes" / "qwen3-asr-venv" / "Scripts" / "python.exe"
)
DEFAULT_QWEN_SCRIPT = Path.home() / ".codex" / "skills" / "dy-note" / "scripts" / "run_qwen_asr.py"


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str


class Transcriber(Protocol):
    async def run(
        self,
        *,
        audio_path: Path,
        result_path: Path,
        cancel_event: asyncio.Event | None = None,
    ) -> SubprocessResult: ...


class QwenAsrSubprocess:
    def __init__(
        self,
        *,
        python_path: Path,
        script_path: Path,
        ffmpeg_path: Path,
        model: str,
        language: str,
        chunk_seconds: float,
    ) -> None:
        self.python_path = python_path
        self.script_path = script_path
        self.ffmpeg_path = ffmpeg_path
        self.model = model
        self.language = language
        self.chunk_seconds = chunk_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> QwenAsrSubprocess:
        ffmpeg = resolve_tool(settings, "ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg_not_found")
        return cls(
            python_path=settings.qwen_python_path or DEFAULT_QWEN_PYTHON,
            script_path=settings.qwen_asr_script_path or DEFAULT_QWEN_SCRIPT,
            ffmpeg_path=ffmpeg,
            model=settings.transcription_model,
            language=settings.transcription_language,
            chunk_seconds=settings.qwen_chunk_seconds,
        )

    async def run(
        self,
        *,
        audio_path: Path,
        result_path: Path,
        cancel_event: asyncio.Event | None = None,
    ) -> SubprocessResult:
        if not self.python_path.is_file():
            raise RuntimeError(f"qwen_python_not_found:{self.python_path}")
        if not self.script_path.is_file():
            raise RuntimeError(f"qwen_script_not_found:{self.script_path}")
        if not self.ffmpeg_path.is_file():
            raise RuntimeError(f"ffmpeg_not_found:{self.ffmpeg_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="dkl_qwen_",
            suffix=".wav",
            dir=result_path.parent,
        )
        os.close(fd)
        wav_path = Path(temporary_name)
        try:
            ffmpeg_process = await asyncio.create_subprocess_exec(
                str(self.ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            ffmpeg_stdout, ffmpeg_stderr = await _communicate_or_cancel(
                ffmpeg_process,
                cancel_event,
            )
            if ffmpeg_process.returncode:
                detail = ffmpeg_stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg_asr_conversion_failed:{detail}")
            process = await asyncio.create_subprocess_exec(
                str(self.python_path),
                str(self.script_path),
                "--audio",
                str(wav_path),
                "--out",
                str(result_path),
                "--model",
                self.model,
                "--language",
                self.language,
                "--chunk-seconds",
                str(self.chunk_seconds),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            stdout, stderr = await _communicate_or_cancel(process, cancel_event)
        finally:
            wav_path.unlink(missing_ok=True)
        return SubprocessResult(
            returncode=int(process.returncode or 0),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr="\n".join(
                part
                for part in (
                    ffmpeg_stdout.decode("utf-8", errors="replace"),
                    ffmpeg_stderr.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                )
                if part
            ),
        )


class TranscriptionService:
    engine = "qwen3-asr"
    audit_version = "1"

    def __init__(
        self,
        session: Session,
        layout: StorageLayout,
        settings: Settings,
        transcriber: Transcriber,
        *,
        subtitle_provider: Any | None = None,
        lease_store: ResourceLeaseStore | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.session = session
        self.layout = layout
        self.settings = settings
        self.transcriber = transcriber
        self.subtitle_provider = subtitle_provider
        # A lease store uses short independent transactions, which is
        # important while another process (the general worker) owns the GPU.
        self.lease_store = lease_store or ResourceLeaseStore(session.get_bind())
        self.worker_id = worker_id or "asr-worker"

    async def process(
        self,
        job: Job,
        work: Work,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> Transcript:
        self.layout.initialize()
        existing_raw = self._existing_raw_asr(work)
        if existing_raw is not None:
            recorder = ProcessingAuditRecorder(
                self.session,
                audit_root=self.settings.storage_root / "audit",
                inline_max_bytes=self.settings.audit_inline_max_bytes,
            )
            audit = recorder.start(
                operation="transcribe_work",
                processor=self.engine,
                model=self.settings.transcription_model,
                version=self.audit_version,
                job=job,
                request={
                    "reused_existing": True,
                    "transcript_id": existing_raw.id,
                    "raw_fingerprint": _transcript_fingerprint(existing_raw),
                },
            )
            recorder.succeed(
                audit,
                response={"reused_existing": True, "transcript_id": existing_raw.id},
            )
            return existing_raw
        subtitle_provider = self.subtitle_provider
        if subtitle_provider is not None:
            subtitle = subtitle_provider.discover(work)
            if subtitle is not None:
                return await self._process_subtitle(
                    job,
                    work,
                    subtitle,
                    cancel_event=cancel_event,
                )
        audio = self.session.scalar(
            select(Asset).where(Asset.work_id == work.id, Asset.kind == "audio")
        )
        if audio is None:
            raise RuntimeError("audio_asset_missing")
        audio_path = Path(audio.path)
        audio_bytes, audio_sha256 = _validate_audio_asset(audio)

        recorder = ProcessingAuditRecorder(
            self.session,
            audit_root=self.settings.storage_root / "audit",
            inline_max_bytes=self.settings.audit_inline_max_bytes,
        )
        result_path = self._result_path(work)
        output_snapshot = _snapshot_files(
            (
                result_path,
                self.layout.transcript_text_path(work.aweme_id),
                self.layout.transcript_segments_path(work.aweme_id),
            )
        )
        # Never hand the stable output path to an external process.  A
        # subprocess can exit successfully without writing anything, in which
        # case reading the old stable file would silently publish stale ASR.
        # A per-attempt sibling also lets a failed/cancelled run preserve the
        # previously verified result.
        attempt_result_path = _attempt_result_path(result_path)
        request = {
            "audio_path": str(audio_path),
            "result_path": str(result_path),
            "audio_bytes": audio_bytes,
            "audio_sha256": audio_sha256,
            "model": self.settings.transcription_model,
            "language": self.settings.transcription_language,
            "chunk_seconds": self.settings.qwen_chunk_seconds,
        }
        audit = recorder.start(
            operation="transcribe_work",
            processor=self.engine,
            model=self.settings.transcription_model,
            version=self.audit_version,
            job=job,
            request=request,
            input_files=[audio_path],
        )
        run_result = SubprocessResult(returncode=-1, stdout="", stderr="")
        lease_handle = None
        published_snapshot: dict[Path, bytes | None] | None = None
        committed = False
        try:
            lease_handle = await self._acquire_gpu(job, cancel_event)
            _raise_if_cancelled(cancel_event)
            transcriber_parameters = inspect.signature(self.transcriber.run).parameters
            transcriber_kwargs: dict[str, Any] = {
                "audio_path": audio_path,
                "result_path": attempt_result_path,
            }
            if "cancel_event" in transcriber_parameters:
                transcriber_kwargs["cancel_event"] = cancel_event
            run_result = await self.transcriber.run(**transcriber_kwargs)
            _raise_if_cancelled(cancel_event)
            if run_result.returncode != 0:
                raise RuntimeError(f"qwen_asr_exit_code:{run_result.returncode}")
            if not attempt_result_path.is_file():
                raise RuntimeError("qwen_asr_result_missing")
            payload = json.loads(attempt_result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("qwen_asr_result_invalid")
            # Renew before adding pending ORM rows.  ResourceLeaseStore uses a
            # short independent transaction, so doing this after the session
            # starts a write transaction would contend with SQLite's writer
            # lock.  The ownership check and conditional job update below form
            # the final commit fence for cancellation/reassignment races.
            await self._renew_lease(lease_handle)
            await self._assert_lease_owned(lease_handle)
            with note_publication_lock(self.settings.storage_root, work.aweme_id):
                # Publish the validated attempt only after parsing it.  The
                # per-work lock covers publication through rollback/commit.
                os.replace(attempt_result_path, result_path)
                transcript = self._save_transcript(work, payload)
                output_paths = self._write_outputs(work, payload)
                published_snapshot = _snapshot_files((result_path, *output_paths))
                await self._commit_if_owned(job, lease_handle)
                committed = True
                recorder.succeed(
                    audit,
                    response=payload,
                    stdout=run_result.stdout,
                    stderr=run_result.stderr,
                    output_files=[result_path, *output_paths],
                )
                return transcript
        except JobCancelledError as exc:
            self.session.rollback()
            if not committed:
                with note_publication_lock(self.settings.storage_root, work.aweme_id):
                    if published_snapshot is None:
                        _restore_files(output_snapshot)
                    else:
                        _restore_files(output_snapshot, expected=published_snapshot)
            recorder.cancel(
                audit,
                error=str(exc),
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                output_files=[result_path] if result_path.exists() else [],
            )
            raise
        except Exception as exc:
            self.session.rollback()
            if not committed:
                with note_publication_lock(self.settings.storage_root, work.aweme_id):
                    if published_snapshot is None:
                        _restore_files(output_snapshot)
                    else:
                        _restore_files(output_snapshot, expected=published_snapshot)
            recorder.fail(
                audit,
                error=str(exc),
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                output_files=[result_path] if result_path.exists() else [],
            )
            raise
        finally:
            attempt_result_path.unlink(missing_ok=True)
            await self._release_gpu(lease_handle)

    def _existing_raw_asr(self, work: Work) -> Transcript | None:
        rows = self.session.scalars(
            select(Transcript)
            .where(
                Transcript.work_id == work.id,
                Transcript.source_kind == "asr",
                Transcript.variant == "raw",
                Transcript.status == "succeeded",
            )
            .order_by(Transcript.id.desc())
        ).all()
        if not rows:
            return None
        for row in rows:
            if _raw_transcript_files_match(self.layout, work.aweme_id, row):
                return row
        raise RuntimeError("raw_transcript_inconsistent")

    async def _process_subtitle(
        self, job: Job, work: Work, draft: Any, *, cancel_event: asyncio.Event | None
    ) -> Transcript:
        _raise_if_cancelled(cancel_event)
        existing = self.session.scalar(
            select(Transcript)
            .where(
                Transcript.work_id == work.id,
                Transcript.variant == "raw",
                Transcript.source_kind == "subtitle",
                Transcript.status == "succeeded",
            )
            .order_by(Transcript.id.desc())
        )
        if existing is not None:
            return existing
        recorder = ProcessingAuditRecorder(
            self.session,
            audit_root=self.settings.storage_root / "audit",
            inline_max_bytes=self.settings.audit_inline_max_bytes,
        )
        audit = recorder.start(
            operation="transcribe_work",
            processor="subtitle_local",
            model="subtitle",
            version="1",
            job=job,
            request={"source_kind": "subtitle"},
        )
        try:
            with note_publication_lock(self.settings.storage_root, work.aweme_id):
                text_path = self.layout.transcript_text_path(work.aweme_id)
                segments_path = self.layout.transcript_segments_path(work.aweme_id)
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(draft.text.strip(), encoding="utf-8")
                segments = [
                    {"id": s.segment_id, "start": s.start, "end": s.end, "text": s.text}
                    for s in draft.segments
                ]
                segments_path.write_text(
                    json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                transcript = Transcript(
                    work_id=work.id,
                    engine="subtitle-local",
                    model="subtitle",
                    source_kind="subtitle",
                    variant="raw",
                    language=draft.language,
                    text=draft.text.strip(),
                    segments_json=json.dumps(segments, ensure_ascii=False),
                    status="succeeded",
                )
                self.session.add(transcript)
                self.session.flush()
                self._upsert_asset(work.id, "transcript_text", text_path, "text/plain")
                self._upsert_asset(
                    work.id, "transcript_segments", segments_path, "application/json"
                )
                await self._commit_if_owned(job)
                recorder.succeed(
                    audit,
                    response={"source_kind": "subtitle"},
                    output_files=(text_path, segments_path),
                )
                return transcript
        except Exception as exc:
            self.session.rollback()
            recorder.fail(audit, error=str(exc))
            raise

    async def _renew_lease(self, lease_handle: object | None) -> None:
        if lease_handle is None:
            return
        renew = getattr(self.lease_store, "renew", None)
        if not callable(renew):
            return
        result = renew(
            lease_handle,
            ttl=float(getattr(self.settings, "gpu_lease_ttl_seconds", 120.0)),
        )
        if inspect.isawaitable(result):
            result = await result
        if not result:
            raise JobCancelledError("cancelled_by_user")

    async def _assert_lease_owned(self, lease_handle: object | None) -> None:
        """Abort publication when the fencing lease is no longer ours."""

        if lease_handle is None:
            return
        checker = getattr(self.lease_store, "is_owned", None)
        if not callable(checker):
            raise JobCancelledError("gpu_lease_lost")
        try:
            result = checker(lease_handle)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001
            raise JobCancelledError("gpu_lease_lost") from exc
        if not result:
            raise JobCancelledError("gpu_lease_lost")

    async def _commit_if_owned(
        self,
        job: Job,
        lease_handle: object | None = None,
    ) -> None:
        """Commit transcript rows only while job and lease ownership hold."""

        # The output writes above are synchronous and can outlive the lease
        # heartbeat.  Re-check immediately before the DB commit as a second
        # fence for a takeover that happens during that window.
        await self._assert_lease_owned(lease_handle)

        owner_worker_id = job.worker_id
        statement = update(Job).where(
            Job.id == job.id,
            Job.status == "running",
        )
        if owner_worker_id is None:
            statement = statement.where(Job.worker_id.is_(None))
        else:
            statement = statement.where(Job.worker_id == owner_worker_id)
        result = self.session.execute(statement.values(heartbeat_at=datetime.now(UTC)))
        if not result.rowcount:
            raise JobCancelledError("cancelled_by_user")
        self.session.commit()

    async def _acquire_gpu(
        self,
        job: Job,
        cancel_event: asyncio.Event | None,
    ) -> object:
        """Wait for the shared GPU lease without holding a DB transaction."""

        resource_name = getattr(self.settings, "gpu_resource_name", "gpu:0")
        lease_ttl = float(getattr(self.settings, "gpu_lease_ttl_seconds", 120.0))
        poll_seconds = max(
            0.01,
            float(getattr(self.settings, "gpu_lease_poll_seconds", 0.5)),
        )
        owner_worker_id = job.worker_id or self.worker_id
        while True:
            _raise_if_cancelled(cancel_event)
            self.session.expire(job)
            current = self.session.get(Job, job.id)
            if current is None or current.status != "running":
                raise JobCancelledError("cancelled_by_user")
            if current.stage != "waiting_gpu":
                current.stage = "waiting_gpu"
                self.session.commit()
                job = current
            handle = self.lease_store.try_acquire(
                resource_name,
                owner_worker_id,
                job.id,
                ttl=lease_ttl,
            )
            if handle is not None:
                current = self.session.get(Job, job.id)
                if current is None or current.status != "running":
                    self.lease_store.release(handle)
                    raise JobCancelledError("cancelled_by_user")
                current.stage = "transcribe"
                self.session.commit()
                return handle
            if cancel_event is None:
                await asyncio.sleep(poll_seconds)
            else:
                sleep_task = asyncio.create_task(asyncio.sleep(poll_seconds))
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    {sleep_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task
                if cancel_task in done and cancel_event.is_set():
                    raise JobCancelledError("cancelled_by_user")

    async def _release_gpu(self, handle: object | None) -> None:
        if handle is None:
            return
        try:
            result = self.lease_store.release(handle)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # The lease is fenced; a stale worker must never overwrite a new
            # owner.  The next expiry/recovery pass can safely reclaim it.
            return

    def _result_path(self, work: Work) -> Path:
        return self.layout.transcript_dir(work.aweme_id) / "qwen3_asr.json"

    def _save_transcript(self, work: Work, payload: dict[str, Any]) -> Transcript:
        text = str(payload.get("text") or "").strip()
        segments = payload.get("segments")
        if not text or not isinstance(segments, list):
            raise RuntimeError("qwen_asr_result_invalid")
        transcript = Transcript(
            work_id=work.id,
            engine=self.engine,
            model=str(payload.get("model") or self.settings.transcription_model),
            language=str(payload.get("language") or self.settings.transcription_language),
            text=text,
            segments_json=json.dumps(segments, ensure_ascii=False),
            status="succeeded",
        )
        self.session.add(transcript)
        self.session.flush()
        return transcript

    def _write_outputs(self, work: Work, payload: dict[str, Any]) -> list[Path]:
        text_path = self.layout.transcript_text_path(work.aweme_id)
        segments_path = self.layout.transcript_segments_path(work.aweme_id)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(str(payload["text"]).strip(), encoding="utf-8")
        segments_path.write_text(
            json.dumps(payload["segments"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._upsert_asset(work.id, "transcript_text", text_path, "text/plain")
        self._upsert_asset(
            work.id,
            "transcript_segments",
            segments_path,
            "application/json",
        )
        return [text_path, segments_path]

    def _upsert_asset(
        self,
        work_id: int,
        kind: str,
        path: Path,
        mime_type: str,
    ) -> None:
        asset = self.session.scalar(
            select(Asset).where(Asset.work_id == work_id, Asset.kind == kind)
        )
        if asset is None:
            asset = Asset(work_id=work_id, kind=kind)
            self.session.add(asset)
        asset.path = str(path)
        asset.mime_type = mime_type
        asset.bytes = path.stat().st_size
        asset.sha256 = file_sha256(path)


class TranscriptCleaningService:
    """Derive an immutable cleaned transcript from a successful raw transcript."""

    engine = "ollama-cleaner"
    audit_version = "1"

    def __init__(
        self,
        session: Session,
        layout: StorageLayout,
        settings: Settings,
        cleaner: Any,
        *,
        lease_store: ResourceLeaseStore | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.session = session
        self.layout = layout
        self.settings = settings
        self.cleaner = cleaner
        self.lease_store = lease_store or ResourceLeaseStore(session.get_bind())
        self.worker_id = worker_id or "general-worker"

    async def process(
        self,
        job: Job,
        work: Work,
        *,
        cancel_event: asyncio.Event | None = None,
        raw_transcript_id: int | None = None,
        raw_fingerprint: str | None = None,
        cleaner_model: str | None = None,
        prompt_version: str | None = None,
    ) -> Transcript:
        self.layout.initialize()
        raw_query = select(Transcript).where(
            Transcript.work_id == work.id,
            Transcript.variant == "raw",
            Transcript.status == "succeeded",
        )
        if raw_transcript_id is not None:
            raw_query = raw_query.where(Transcript.id == raw_transcript_id)
        else:
            raw_query = raw_query.order_by(Transcript.id.desc())
        raw = self.session.scalar(raw_query)
        if raw is None or not raw.text:
            raise RuntimeError("clean_transcript_stale_input")
        current_fingerprint = _transcript_fingerprint(raw)
        if raw_fingerprint is not None and raw_fingerprint != current_fingerprint:
            raise RuntimeError("clean_transcript_stale_input")
        raw_fingerprint = current_fingerprint
        model = str(getattr(self.cleaner, "model", self.settings.transcription_cleaner_model))
        current_prompt_version = str(
            getattr(
                self.cleaner, "prompt_version", self.settings.transcription_cleaner_prompt_version
            )
        )
        if cleaner_model is not None and cleaner_model != model:
            raise RuntimeError("clean_transcript_stale_input")
        if prompt_version is not None and prompt_version != current_prompt_version:
            raise RuntimeError("clean_transcript_stale_input")
        prompt_version = current_prompt_version
        existing = self.session.scalar(
            select(Transcript).where(
                Transcript.work_id == work.id,
                Transcript.variant == "cleaned",
                Transcript.status == "succeeded",
                Transcript.parent_transcript_id == raw.id,
                Transcript.engine == raw.engine,
                Transcript.model == raw.model,
                Transcript.word_timestamps_json
                == json.dumps(
                    {
                        "raw_fingerprint": raw_fingerprint,
                        "cleaner_model": model,
                        "prompt_version": prompt_version,
                    },
                    sort_keys=True,
                ),
            )
        )
        if existing is not None:
            return existing

        recorder = ProcessingAuditRecorder(
            self.session,
            audit_root=self.settings.storage_root / "audit",
            inline_max_bytes=self.settings.audit_inline_max_bytes,
        )
        audit = recorder.start(
            operation="clean_transcript_work",
            processor=str(getattr(self.cleaner, "name", self.engine)),
            model=model,
            version=prompt_version,
            job=job,
            request={
                "raw_transcript_id": raw.id,
                "raw_fingerprint": raw_fingerprint,
                "prompt_version": prompt_version,
            },
        )
        lease_handle = None
        output_paths = (
            self.layout.transcript_cleaned_text_path(work.aweme_id),
            self.layout.transcript_cleaned_json_path(work.aweme_id),
        )
        snapshot = _snapshot_files(output_paths)
        published_expected: dict[Path, bytes | None] | None = None
        staged_paths: tuple[Path, ...] = ()
        committed = False
        try:
            lease_handle = await self._acquire_gpu(job, cancel_event)
            _raise_if_cancelled(cancel_event)
            if getattr(self.cleaner, "_lease_checker", None) is None:
                try:
                    self.cleaner._lease_checker = lambda: self.lease_store.is_owned(lease_handle)
                except (AttributeError, TypeError):
                    pass
            draft = _draft_from_raw(raw)
            cleaner_kwargs: dict[str, Any] = {}
            with suppress(TypeError, ValueError):
                if "cancel_event" in inspect.signature(self.cleaner.clean).parameters:
                    cleaner_kwargs["cancel_event"] = cancel_event
            result = await self.cleaner.clean(draft, **cleaner_kwargs)
            cleaned = _validate_cleaner_result(result, draft)
            with note_publication_lock(self.settings.storage_root, work.aweme_id):
                await self._assert_lease_owned(lease_handle)
                staged = tuple(_attempt_result_path(path) for path in output_paths)
                staged_paths = staged
                output_paths[0].parent.mkdir(parents=True, exist_ok=True)
                staged[0].write_text(cleaned.text.strip(), encoding="utf-8")
                staged[1].write_text(
                    json.dumps(
                        [
                            {"id": s.segment_id, "start": s.start, "end": s.end, "text": s.text}
                            for s in cleaned.segments
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                for source, target in zip(staged, output_paths, strict=True):
                    os.replace(source, target)
                published_expected = _snapshot_files(output_paths)
                transcript = Transcript(
                    work_id=work.id,
                    engine=raw.engine,
                    model=raw.model,
                    source_kind=raw.source_kind,
                    variant="cleaned",
                    parent_transcript_id=raw.id,
                    language=cleaned.language or raw.language,
                    text=cleaned.text,
                    segments_json=json.dumps(
                        [
                            {"id": s.segment_id, "start": s.start, "end": s.end, "text": s.text}
                            for s in cleaned.segments
                        ],
                        ensure_ascii=False,
                    ),
                    word_timestamps_json=json.dumps(
                        {
                            "raw_fingerprint": raw_fingerprint,
                            "cleaner_model": model,
                            "prompt_version": prompt_version,
                        },
                        sort_keys=True,
                    ),
                    status="succeeded",
                )
                self.session.add(transcript)
                await self._commit_if_owned(job, lease_handle)
                committed = True
                recorder.succeed(
                    audit,
                    response={"raw_fingerprint": raw_fingerprint, "prompt_version": prompt_version},
                    output_files=output_paths,
                )
                return transcript
        except JobCancelledError as exc:
            self.session.rollback()
            if not committed:
                with note_publication_lock(self.settings.storage_root, work.aweme_id):
                    if published_expected is None:
                        _restore_files(snapshot)
                    else:
                        _restore_files(snapshot, expected=published_expected)
            recorder.cancel(audit, error=str(exc), output_files=output_paths)
            raise
        except Exception as exc:
            self.session.rollback()
            if not committed:
                with note_publication_lock(self.settings.storage_root, work.aweme_id):
                    if published_expected is None:
                        _restore_files(snapshot)
                    else:
                        _restore_files(snapshot, expected=published_expected)
            recorder.fail(audit, error=str(exc), output_files=output_paths)
            raise
        finally:
            for staged_path in staged_paths:
                staged_path.unlink(missing_ok=True)
            await self._release_gpu(lease_handle)

    async def _acquire_gpu(self, job: Job, cancel_event: asyncio.Event | None) -> object:
        resource = getattr(self.settings, "gpu_resource_name", "gpu:0")
        poll = max(0.01, float(getattr(self.settings, "gpu_lease_poll_seconds", 0.5)))
        ttl = float(getattr(self.settings, "gpu_lease_ttl_seconds", 120.0))
        owner = job.worker_id or self.worker_id
        while True:
            _raise_if_cancelled(cancel_event)
            current = self.session.get(Job, job.id)
            if current is None or current.status != "running":
                raise JobCancelledError("cancelled_by_user")
            if current.stage != "waiting_gpu":
                current.stage = "waiting_gpu"
                self.session.commit()
            handle = self.lease_store.try_acquire(resource, owner, job.id, ttl=ttl)
            if handle is not None:
                current.stage = "clean"
                self.session.commit()
                return handle
            await asyncio.sleep(poll)

    async def _assert_lease_owned(self, handle: object | None) -> None:
        if handle is not None and not self.lease_store.is_owned(handle):
            raise JobCancelledError("gpu_lease_lost")

    async def _commit_if_owned(self, job: Job, handle: object | None) -> None:
        await self._assert_lease_owned(handle)
        result = self.session.execute(
            update(Job)
            .where(Job.id == job.id, Job.status == "running", Job.worker_id == job.worker_id)
            .values(heartbeat_at=datetime.now(UTC))
        )
        if not result.rowcount:
            raise JobCancelledError("cancelled_by_user")
        self.session.commit()

    async def _release_gpu(self, handle: object | None) -> None:
        if handle is not None:
            with suppress(Exception):
                self.lease_store.release(handle)


def _draft_from_raw(raw: Transcript) -> TranscriptDraft:
    values = json.loads(raw.segments_json or "[]")
    if not isinstance(values, list):
        raise RuntimeError("raw_transcript_segments_invalid")
    segments: list[TranscriptSegment] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise RuntimeError("raw_transcript_segments_invalid")
        segments.append(
            TranscriptSegment(
                str(value.get("id") or index), value.get("start"), value.get("end"), value["text"]
            )
        )
    if not segments:
        raise RuntimeError("raw_transcript_segments_invalid")
    return TranscriptDraft(tuple(segments), raw.text or "", raw.language, raw.source_kind)


def _validate_cleaner_result(result: Any, draft: TranscriptDraft):
    cleaned = result.transcript if isinstance(result, CleanerResult) else result
    if isinstance(result, CleanerResult) and not result.usable:
        raise RuntimeError("cleaner_result_not_usable")
    if (
        not hasattr(cleaned, "segments")
        or not hasattr(cleaned, "text")
        or not cleaned.text.strip()
        or not cleaned.segments
        or len(cleaned.segments) != len(draft.segments)
    ):
        raise RuntimeError("cleaner_result_invalid")
    for original, edited in zip(draft.segments, cleaned.segments, strict=True):
        if (
            edited.segment_id != original.segment_id
            or edited.start != original.start
            or edited.end != original.end
            or not isinstance(edited.text, str)
            or not edited.text.strip()
        ):
            raise RuntimeError("cleaner_result_invalid")
    return cleaned


def _transcript_fingerprint(raw: Transcript) -> str:
    payload = json.dumps(
        {
            "text": raw.text or "",
            "segments": json.loads(raw.segments_json or "[]"),
            "source_kind": raw.source_kind,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_audio_asset(asset: Asset) -> tuple[int, str]:
    audio_path = Path(asset.path)
    if not audio_path.is_file():
        raise RuntimeError(f"audio_file_missing:{audio_path}")
    try:
        audio_bytes = audio_path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"audio_file_unreadable:{audio_path}") from exc
    if audio_bytes <= 0:
        raise RuntimeError("audio_asset_empty")
    try:
        audio_sha256 = file_sha256(audio_path)
    except OSError as exc:
        raise RuntimeError(f"audio_file_unreadable:{audio_path}") from exc
    if asset.bytes is not None and int(asset.bytes) != audio_bytes:
        raise RuntimeError("audio_asset_changed")
    if asset.sha256 and asset.sha256 != audio_sha256:
        raise RuntimeError("audio_asset_changed")
    return audio_bytes, audio_sha256


def _raw_transcript_files_match(
    layout: StorageLayout,
    aweme_id: str,
    raw: Transcript,
) -> bool:
    try:
        segments = json.loads(raw.segments_json or "[]")
        if not isinstance(segments, list) or not segments or not raw.text:
            return False
        if any(
            not isinstance(value, dict) or not isinstance(value.get("text"), str)
            for value in segments
        ):
            return False
        if layout.transcript_text_path(aweme_id).read_text(encoding="utf-8") != raw.text:
            return False
        file_segments = json.loads(
            layout.transcript_segments_path(aweme_id).read_text(encoding="utf-8")
        )
        if file_segments != segments:
            return False
        return layout.transcript_qwen_path(aweme_id).is_file()
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _snapshot_files(paths: tuple[Path, ...]) -> dict[Path, bytes | None]:
    """Capture generated transcript files so a cancelled run can roll back."""

    snapshot: dict[Path, bytes | None] = {}
    for path in dict.fromkeys(paths):
        snapshot[path] = path.read_bytes() if path.is_file() else None
    return snapshot


def _attempt_result_path(result_path: Path) -> Path:
    return result_path.with_name(f".{result_path.stem}.{uuid4().hex}{result_path.suffix}")


def _restore_files(
    snapshot: dict[Path, bytes | None],
    *,
    expected: dict[Path, bytes | None] | None = None,
) -> None:
    try:
        if expected is not None:
            current = {
                candidate: candidate.read_bytes() if candidate.is_file() else None
                for candidate in expected
            }
            if current != expected:
                return
        for restore_path, restore_content in snapshot.items():
            if restore_content is None:
                restore_path.unlink(missing_ok=True)
            else:
                restore_path.parent.mkdir(parents=True, exist_ok=True)
                restore_path.write_bytes(restore_content)
    except OSError:
        # Preserve the original processing error; the audit still records
        # the failed attempt and the next run can repair the files.
        return


async def _communicate_or_cancel(
    process: asyncio.subprocess.Process,
    cancel_event: asyncio.Event | None,
) -> tuple[bytes, bytes]:
    communication = asyncio.create_task(process.communicate())
    cancellation: asyncio.Task[bool] | None = None
    try:
        if cancel_event is None:
            return await communication
        cancellation = asyncio.create_task(cancel_event.wait())
        done, _ = await asyncio.wait(
            {communication, cancellation},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancellation in done and communication not in done and cancel_event.is_set():
            await _terminate_process(process, communication)
            raise JobCancelledError("cancelled_by_user")
        return await communication
    except asyncio.CancelledError:
        await _terminate_process(process, communication)
        raise
    finally:
        if cancellation is not None:
            cancellation.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation


def _raise_if_cancelled(cancel_event: asyncio.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError("cancelled_by_user")


async def _terminate_process(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
    try:
        return await asyncio.wait_for(asyncio.shield(communication), timeout=5.0)
    except TimeoutError:
        _terminate_process_tree(process)
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        return await communication


def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    """Ask Windows to terminate descendants when a worker owns a process tree."""

    if os.name != "nt":
        return
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    with suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
