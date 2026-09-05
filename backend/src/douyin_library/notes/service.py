from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from douyin_library.audit import ProcessingAuditRecorder
from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.jobs.leases import ResourceLeaseStore
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Analysis, Job, Work
from douyin_library.notes.generator import (
    GENERATOR_VERSION,
    MODEL_NAME,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    NoteBundle,
    generate_rule_note,
)
from douyin_library.notes.input import NoteInput, NoteInputLoader
from douyin_library.notes.pipeline import (
    SemanticGeneration,
    SemanticNotePipeline,
    StageRecord,
)
from douyin_library.notes.providers.base import NoteProvider
from douyin_library.notes.providers.external import ExternalNoteProvider
from douyin_library.notes.providers.ollama import OllamaNoteProvider
from douyin_library.notes.providers.selector import (
    ProviderName,
    ProviderPolicy,
    select_provider_names,
)
from douyin_library.notes.publisher import NoteBundlePublisher, publish_with_guard
from douyin_library.notes.render import render_semantic_note
from douyin_library.notes.state import classify_note_bundle

ProviderFactory = Callable[[ProviderName], NoteProvider]


@dataclass(frozen=True, slots=True)
class NoteCandidate:
    aweme_id: str
    transcript_path: Path
    note_dir: Path
    work_id: int | None
    completed: bool
    quality_status: str = "missing"


@dataclass(slots=True)
class GenerationResult:
    aweme_id: str
    status: str
    output_dir: str
    written_files: list[str] = field(default_factory=list)
    preserved_files: list[str] = field(default_factory=list)
    error: str | None = None
    provider: str | None = None
    model: str | None = None
    quality_gate: str | None = None


@dataclass(slots=True)
class BatchGenerationResult:
    total_transcripts: int
    completed_before: int
    eligible: int
    selected: int
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    would_generate: int = 0
    dry_run: bool = False
    items: list[dict[str, Any]] = field(default_factory=list)
    items_truncated: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _StageCollector:
    def __init__(self) -> None:
        self.records: list[StageRecord] = []

    def record(self, record: StageRecord) -> None:
        self.records.append(record)


class NoteGenerationService:
    """Run semantic note generation and publish only validated candidates.

    The public methods are dual-mode for compatibility with the original CLI:
    calls made outside an event loop execute synchronously; calls from workers
    return an awaitable and retain provider cancellation semantics.
    """

    def __init__(
        self,
        session: Session,
        layout: StorageLayout,
        settings: Settings | None = None,
        *,
        provider_factory: ProviderFactory | None = None,
        publisher: NoteBundlePublisher | None = None,
        lease_store: ResourceLeaseStore | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.session = session
        self.layout = layout
        self.settings = settings or Settings(storage_root=layout.root)
        self.provider_factory = provider_factory
        self.publisher = publisher or NoteBundlePublisher(layout.root)
        self.lease_store = lease_store or ResourceLeaseStore(session.get_bind())
        self.worker_id = worker_id or "note-service"

    def discover(
        self,
        *,
        force: bool = False,
        aweme_ids: set[str] | None = None,
        legacy_only: bool = False,
    ) -> list[NoteCandidate]:
        if force and legacy_only:
            raise ValueError("force_and_legacy_only_are_mutually_exclusive")
        if not self.layout.transcript.is_dir():
            return []
        transcript_paths = sorted(self.layout.transcript.glob("*/transcript.txt"))
        if aweme_ids is not None:
            transcript_paths = [
                path for path in transcript_paths if path.parent.name in aweme_ids
            ]
        ids = [path.parent.name for path in transcript_paths]
        works = (
            {
                work.aweme_id: work
                for work in self.session.scalars(
                    select(Work).where(Work.aweme_id.in_(ids))
                ).all()
            }
            if ids
            else {}
        )
        candidates: list[NoteCandidate] = []
        for transcript_path in transcript_paths:
            aweme_id = transcript_path.parent.name
            note_dir = self.layout.note_dir(aweme_id)
            state = classify_note_bundle(note_dir)
            completed = (note_dir / "learning_note.md").is_file()
            if legacy_only and not state.upgrade_eligible:
                continue
            if not legacy_only and completed and not force:
                continue
            work = works.get(aweme_id)
            candidates.append(
                NoteCandidate(
                    aweme_id=aweme_id,
                    transcript_path=transcript_path,
                    note_dir=note_dir,
                    work_id=work.id if work else None,
                    completed=completed,
                    quality_status=state.quality_status,
                )
            )
        return candidates

    def inventory(self, *, aweme_ids: set[str] | None = None) -> tuple[int, int]:
        if not self.layout.transcript.is_dir():
            return 0, 0
        paths = list(self.layout.transcript.glob("*/transcript.txt"))
        if aweme_ids is not None:
            paths = [path for path in paths if path.parent.name in aweme_ids]
        completed = sum(
            (self.layout.note_dir(path.parent.name) / "learning_note.md").is_file()
            for path in paths
        )
        return len(paths), completed

    def run_batch(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        dry_run: bool = False,
        aweme_ids: set[str] | None = None,
        provider_policy: ProviderPolicy | None = None,
        legacy_only: bool = False,
    ) -> BatchGenerationResult | Awaitable[BatchGenerationResult]:
        return _sync_or_coroutine(
            self._run_batch_async(
                limit=limit,
                force=force,
                dry_run=dry_run,
                aweme_ids=aweme_ids,
                provider_policy=provider_policy,
                legacy_only=legacy_only,
            )
        )

    async def _run_batch_async(
        self,
        *,
        limit: int | None,
        force: bool,
        dry_run: bool,
        aweme_ids: set[str] | None,
        provider_policy: ProviderPolicy | None,
        legacy_only: bool,
    ) -> BatchGenerationResult:
        total, completed = self.inventory(aweme_ids=aweme_ids)
        candidates = self.discover(
            force=force,
            aweme_ids=aweme_ids,
            legacy_only=legacy_only,
        )
        selected = candidates[:limit] if limit is not None else candidates
        result = BatchGenerationResult(
            total_transcripts=total,
            completed_before=completed,
            eligible=len(candidates),
            selected=len(selected),
            skipped=0 if force or legacy_only else completed,
            would_generate=len(selected) if dry_run else 0,
            dry_run=dry_run,
        )
        if dry_run:
            result.items = [
                {
                    "aweme_id": item.aweme_id,
                    "status": "would_generate",
                    "output_dir": str(item.note_dir),
                    "quality_status": item.quality_status,
                }
                for item in selected[:50]
            ]
            result.items_truncated = max(0, len(selected) - len(result.items))
            return result
        for candidate in selected:
            try:
                item = await self._generate_one_async(
                    candidate.aweme_id,
                    force=force,
                    job=None,
                    provider_policy=provider_policy,
                    legacy_only=legacy_only,
                    cancel_event=asyncio.Event(),
                )
            except JobCancelledError as exc:
                result.failed += 1
                _append_item(
                    result,
                    {
                        "aweme_id": candidate.aweme_id,
                        "status": "cancelled",
                        "output_dir": str(candidate.note_dir),
                        "error": str(exc),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                result.failed += 1
                _append_item(
                    result,
                    {
                        "aweme_id": candidate.aweme_id,
                        "status": "failed",
                        "output_dir": str(candidate.note_dir),
                        "error": str(exc),
                    },
                )
            else:
                if item.status == "generated":
                    result.generated += 1
                elif legacy_only:
                    # An upgrade candidate that remains preserved, skipped, or
                    # otherwise non-semantic is a failed upgrade attempt.  It
                    # must stay visible for retry instead of looking complete.
                    result.failed += 1
                else:
                    result.skipped += 1
                _append_item(result, asdict(item))
        return result

    def generate_one(
        self,
        aweme_id: str,
        *,
        force: bool = False,
        job: Job | None = None,
        provider_policy: ProviderPolicy | None = None,
        legacy_only: bool = False,
        cancel_event: asyncio.Event | None = None,
    ) -> GenerationResult | Awaitable[GenerationResult]:
        return _sync_or_coroutine(
            self._generate_one_async(
                aweme_id,
                force=force,
                job=job,
                provider_policy=provider_policy,
                legacy_only=legacy_only,
                cancel_event=cancel_event or asyncio.Event(),
            )
        )

    async def _generate_one_async(
        self,
        aweme_id: str,
        *,
        force: bool,
        job: Job | None,
        provider_policy: ProviderPolicy | None,
        legacy_only: bool,
        cancel_event: asyncio.Event,
    ) -> GenerationResult:
        if force and legacy_only:
            raise ValueError("force_and_legacy_only_are_mutually_exclusive")
        _raise_if_cancelled(cancel_event)
        note_dir = self.layout.note_dir(aweme_id)
        current_state = classify_note_bundle(note_dir)
        learning_note = note_dir / "learning_note.md"
        if legacy_only and not current_state.upgrade_eligible:
            return GenerationResult(aweme_id, "skipped", str(note_dir))
        if learning_note.is_file() and not force and not legacy_only:
            return GenerationResult(aweme_id, "skipped", str(note_dir))
        transcript_path = self.layout.transcript_text_path(aweme_id)
        if not transcript_path.is_file():
            raise FileNotFoundError(f"transcript_not_found:{aweme_id}")

        work = self.session.scalar(select(Work).where(Work.aweme_id == aweme_id))
        note_input = NoteInputLoader(self.layout, self.session).load(aweme_id)
        source_files = dict(note_input.source_files)
        # Existing-note upgrades are intentionally Ollama-only.  In
        # particular, do not let an old ``auto`` payload fall through to an
        # external provider or the deterministic rule generator.
        policy = "local" if legacy_only else (provider_policy or self.settings.note_provider_policy)
        provider_names = ["ollama"] if legacy_only else select_provider_names(self.settings, policy)
        recorder = ProcessingAuditRecorder(
            self.session,
            audit_root=self.layout.root / "audit",
            inline_max_bytes=self.settings.audit_inline_max_bytes,
        )
        audit = recorder.start(
            operation="generate_note",
            processor="note-generator",
            model=None,
            version="semantic-v2",
            job=job,
            work_id=work.id if work else None,
            request={
                "prompt": SYSTEM_PROMPT,
                "provider_policy": policy,
                "provider_chain": list(provider_names),
                "input": asdict(note_input),
            },
            input_files=[Path(path) for path in source_files.values()],
        )
        started_at = audit.started_at.isoformat()
        collector = _StageCollector()
        attempts: list[dict[str, Any]] = []
        selected_bundle: NoteBundle | None = None
        selected_generation: SemanticGeneration | None = None
        selected_provider = "rule"
        selected_model = MODEL_NAME
        selected_prompt_versions: dict[str, str] = {}
        selected_quality_gate = "fallback"
        lease_handle: Any | None = None
        selected_lease_handle: Any | None = None
        try:
            for provider_name in provider_names:
                _raise_if_cancelled(cancel_event)
                attempt: dict[str, Any] = {"provider": provider_name, "status": "started"}
                if provider_name == "rule":
                    try:
                        selected_bundle = generate_rule_note(
                            note_input,
                            source_files=source_files,
                            generated_at=datetime.now(UTC).isoformat(),
                        )
                        selected_provider = "rule"
                        selected_model = MODEL_NAME
                        selected_prompt_versions = {"fallback": PROMPT_VERSION}
                        selected_quality_gate = "fallback"
                        attempt.update({"status": "succeeded", "quality_gate": "fallback"})
                    except Exception as exc:  # noqa: BLE001
                        attempt.update({"status": "failed", "error": _safe_error(exc)})
                    attempts.append(attempt)
                    if selected_bundle is not None:
                        break
                    continue

                provider: NoteProvider | None = None
                try:
                    provider = await self._make_provider(provider_name)
                    if legacy_only and getattr(provider, "name", None) != "ollama":
                        raise RuntimeError("legacy_upgrade_provider_not_ollama")
                    lease_handle = await self._acquire_gpu_if_needed(
                        provider,
                        job=job,
                        cancel_event=cancel_event,
                    )
                    if job is not None:
                        job.stage = "analyze"
                        job.progress = 0.15
                        self.session.commit()
                    generation = await SemanticNotePipeline(provider).generate(
                        note_input,
                        cancel_event,
                        collector,
                    )
                    attempt.update(
                        {
                            "status": (
                                "succeeded"
                                if generation.coverage.quality_gate == "passed"
                                else "failed"
                            ),
                            "quality_gate": generation.coverage.quality_gate,
                            "coverage": generation.coverage.model_dump(mode="json", by_alias=True),
                            "repair_count": generation.repair_count,
                        }
                    )
                    if generation.coverage.quality_gate == "passed":
                        selected_generation = generation
                        selected_provider = provider.name
                        selected_model = provider.model
                        selected_prompt_versions = dict(generation.prompt_versions)
                        selected_quality_gate = generation.coverage.quality_gate
                        selected_bundle = render_semantic_note(
                            note_input,
                            generation,
                            generated_at=datetime.now(UTC).isoformat(),
                        )
                        # Keep the local lease through provider unload and the
                        # publisher's final ownership fence.  Failed attempts
                        # still release their lease in the usual finally path.
                        selected_lease_handle = lease_handle
                        lease_handle = None
                except JobCancelledError:
                    attempt["status"] = "cancelled"
                    raise
                except Exception as exc:  # noqa: BLE001
                    attempt.update(
                        {
                            "status": "failed" if provider is not None else "unavailable",
                            "error": _safe_error(exc),
                        }
                    )
                finally:
                    # Keep the provider's unload inside the lease window.  A
                    # provider may be unavailable before construction, hence
                    # the defensive local lookup.
                    if provider is not None:
                        with suppress(Exception):
                            await provider.unload()
                    await self._release_gpu(lease_handle)
                    lease_handle = None
                    attempts.append(attempt)
                if selected_bundle is not None:
                    break

            if selected_bundle is None:
                raise RuntimeError("note_generation_no_provider_available")

            audit.processor = f"{selected_provider}-note-generator"
            audit.model = selected_model
            audit.version = "semantic-v2" if selected_generation is not None else GENERATOR_VERSION
            response = {
                "provider": selected_provider,
                "model": selected_model,
                "quality_gate": selected_quality_gate,
                "attempts": attempts,
                "stage_records": [_stage_record_dict(item) for item in collector.records],
                "raw_output": selected_bundle.raw_output,
                "parsed_result": selected_bundle.parsed,
                "coverage": (
                    selected_generation.coverage.model_dump(mode="json", by_alias=True)
                    if selected_generation is not None
                    else selected_bundle.coverage_report
                ),
            }
            descriptor = self._generation_audit_descriptor(
                aweme_id=aweme_id,
                note_input=note_input,
                audit=audit,
                started_at=started_at,
                provider=selected_provider,
                model=selected_model,
                quality_gate=selected_quality_gate,
                attempts=attempts,
                generation=selected_generation,
                error=None,
            )
            outputs = _bundle_outputs(
                selected_bundle,
                descriptor,
                selected_provider,
                selected_model,
            )
            publish_kwargs: dict[str, Any] = {
                "force": force or legacy_only,
                # Bind the ownership snapshot at callback creation time.  The
                # surrounding async loop reuses and clears its locals in
                # ``finally`` blocks; a late-bound closure can otherwise lose
                # the lease token on Python 3.14.
                "ownership_check": (
                    lambda current_job=job, current_lease=selected_lease_handle: (
                        self._owns_generation(current_job, current_lease)
                    )
                ),
            }
            if legacy_only:
                publish_kwargs["publish_guard"] = (
                    lambda expected_state=current_state, current_dir=note_dir: (
                        _legacy_state_unchanged(current_dir, expected_state)
                    )
                )
            publish_result = publish_with_guard(
                self.publisher,
                self.layout.root,
                aweme_id,
                outputs,
                **publish_kwargs,
            )
            if publish_result.status == "published":
                # A generation is only successful once the validated bundle has
                # made it through publication.  This keeps a failed atomic
                # replace from leaving a misleading ``succeeded`` audit.
                recorder.succeed(audit, response=response)
                if work is not None:
                    self._persist_analysis(
                        work,
                        selected_bundle,
                        provider=selected_provider,
                        model=selected_model,
                        prompt_version=next(
                            iter(selected_prompt_versions.values()), PROMPT_VERSION
                        ),
                        coverage=(
                            selected_generation.coverage.model_dump(mode="json", by_alias=True)
                            if selected_generation is not None
                            else selected_bundle.coverage_report
                        ),
                    )
                if job is not None:
                    job.progress = 0.95
                    self.session.commit()
                return GenerationResult(
                    aweme_id=aweme_id,
                    status="generated",
                    output_dir=str(note_dir),
                    written_files=[
                        *[str(note_dir / name) for name in outputs],
                    ],
                    provider=selected_provider,
                    model=selected_model,
                    quality_gate=selected_quality_gate,
                )
            recorder.skip(
                audit,
                reason=f"publication_{publish_result.status}",
                response={**response, "publication_status": publish_result.status},
            )
            return GenerationResult(
                aweme_id=aweme_id,
                status=publish_result.status,
                output_dir=str(note_dir),
                provider=selected_provider,
                model=selected_model,
                quality_gate=selected_quality_gate,
            )
        except JobCancelledError as exc:
            await self._release_gpu(lease_handle)
            if audit.status == "running":
                recorder.cancel(audit, error=str(exc))
            self._write_attempt_audit(
                note_dir,
                aweme_id=aweme_id,
                note_input=note_input,
                started_at=started_at,
                audit=audit,
                provider=selected_provider,
                model=selected_model,
                attempts=attempts,
                error=str(exc),
                status="cancelled",
            )
            raise
        except Exception as exc:  # noqa: BLE001
            await self._release_gpu(lease_handle)
            if audit.status == "running":
                recorder.fail(
                    audit,
                    error=str(exc),
                    response={
                        "attempts": attempts,
                        "stage_records": [_stage_record_dict(item) for item in collector.records],
                    },
                )
            self._write_attempt_audit(
                note_dir,
                aweme_id=aweme_id,
                note_input=note_input,
                started_at=started_at,
                audit=audit,
                provider=selected_provider,
                model=selected_model,
                attempts=attempts,
                error=str(exc),
                status="failed",
            )
            if learning_note.is_file():
                return GenerationResult(
                    aweme_id=aweme_id,
                    status="preserved_existing",
                    output_dir=str(note_dir),
                    error=str(exc),
                )
            raise
        finally:
            await self._release_gpu(selected_lease_handle)

    async def _make_provider(self, provider_name: ProviderName) -> NoteProvider:
        if self.provider_factory is not None:
            value = self.provider_factory(provider_name)
            if inspect.isawaitable(value):
                value = await value
            return value
        if provider_name == "ollama":
            from douyin_library.notes.ollama_runtime import probe_ollama

            status = await probe_ollama(self.settings)
            if not status.ready:
                raise RuntimeError(f"ollama_not_ready:{status.service_status}")
            return OllamaNoteProvider(self.settings)
        if provider_name == "external":
            return ExternalNoteProvider(self.settings)
        raise ValueError(f"unknown_note_provider:{provider_name}")

    async def _acquire_gpu_if_needed(
        self,
        provider: NoteProvider,
        *,
        job: Job | None,
        cancel_event: asyncio.Event,
    ) -> Any | None:
        if not provider.is_local or self.lease_store is None:
            return None
        resource_name = getattr(self.settings, "gpu_resource_name", "gpu:0")
        ttl = float(getattr(self.settings, "gpu_lease_ttl_seconds", 120.0))
        interval = max(
            0.01,
            float(getattr(self.settings, "gpu_lease_poll_seconds", 0.5)),
        )
        owner_worker_id = (
            job.worker_id if job is not None and job.worker_id else self.worker_id
        )
        while True:
            _raise_if_cancelled(cancel_event)
            if job is not None:
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
                job.id if job is not None else None,
                ttl=ttl,
            )
            if inspect.isawaitable(handle):
                handle = await handle
            if handle is not None:
                if job is not None:
                    current = self.session.get(Job, job.id)
                    if current is None or current.status != "running":
                        self.lease_store.release(handle)
                        raise JobCancelledError("cancelled_by_user")
                    current.stage = "analyze"
                    self.session.commit()
                return handle
            if cancel_event is None:
                await asyncio.sleep(interval)
                continue
            sleep_task = asyncio.create_task(asyncio.sleep(interval))
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

    async def _release_gpu(self, handle: Any | None) -> None:
        if handle is None or self.lease_store is None:
            return
        release = getattr(self.lease_store, "release", None)
        if not callable(release):
            return
        result = release(handle)
        if inspect.isawaitable(result):
            await result

    def _owns_generation(self, job: Job | None, lease_handle: Any | None) -> bool:
        if job is not None and not _job_is_running(self.session, job.id):
            return False
        if lease_handle is None or self.lease_store is None:
            return True
        checker = getattr(self.lease_store, "is_owned", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(lease_handle))
        except Exception:
            return False

    def _persist_analysis(
        self,
        work: Work,
        bundle: NoteBundle,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        coverage: Mapping[str, Any],
    ) -> None:
        self.session.execute(
            update(Analysis)
            .where(Analysis.work_id == work.id, Analysis.status == "succeeded")
            .values(status="superseded")
        )
        self.session.add(
            Analysis(
                work_id=work.id,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                summary=bundle.parsed.get("summary") or bundle.parsed.get("overview"),
                keywords_json=json.dumps(bundle.parsed.get("tags", []), ensure_ascii=False),
                topics_json=json.dumps(
                    bundle.parsed.get("topics", bundle.parsed.get("tags", [])),
                    ensure_ascii=False,
                ),
                knowledge_points_json=json.dumps(
                    bundle.parsed.get("key_points", []), ensure_ascii=False
                ),
                evidence_json=json.dumps(dict(coverage), ensure_ascii=False),
                status="succeeded",
            )
        )
        self.session.commit()

    def _generation_audit_descriptor(
        self,
        *,
        aweme_id: str,
        note_input: NoteInput,
        audit: Any,
        started_at: str,
        provider: str,
        model: str,
        quality_gate: str,
        attempts: list[dict[str, Any]],
        generation: SemanticGeneration | None,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "schema": "douyin-note-generation-audit-v2",
            "aweme_id": aweme_id,
            "status": "succeeded" if error is None else "failed",
            "processor": f"{provider}-note-generator",
            "provider": provider,
            "model": model,
            "version": "semantic-v2" if generation is not None else GENERATOR_VERSION,
            "prompt_versions": (
                dict(generation.prompt_versions)
                if generation is not None
                else {"fallback": PROMPT_VERSION}
            ),
            "input": {
                "source_files": dict(note_input.source_files),
                "transcript_chars": len(note_input.transcript),
                "segment_count": len(note_input.segments),
                "warnings": list(note_input.warnings),
                "transcript_variant": note_input.transcript_variant,
                "transcript_source_kind": note_input.transcript_source_kind,
                "cleaning_fallback": note_input.cleaning_fallback,
            },
            "processing_audit_id": audit.id,
            "audit_references": {
                "request_json": "processing_audit.request_json",
                "response_json": "processing_audit.response_json",
            },
            "attempts": attempts,
            "coverage": (
                generation.coverage.model_dump(mode="json", by_alias=True)
                if generation is not None
                else None
            ),
            "quality_gate": quality_gate,
            "error": error,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
        }

    def _write_attempt_audit(
        self,
        note_dir: Path,
        *,
        aweme_id: str,
        note_input: NoteInput,
        started_at: str,
        audit: Any,
        provider: str,
        model: str,
        attempts: list[dict[str, Any]],
        error: str,
        status: str,
    ) -> None:
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            path = (
                note_dir
                / "attempts"
                / f"{timestamp}-{uuid4().hex[:8]}"
                / "generation_audit.json"
            )
            descriptor = self._generation_audit_descriptor(
                aweme_id=aweme_id,
                note_input=note_input,
                audit=audit,
                started_at=started_at,
                provider=provider,
                model=model,
                quality_gate="failed",
                attempts=attempts,
                generation=None,
                error=error,
            )
            descriptor["status"] = status
            _atomic_write(path, _json_text(descriptor))
        except OSError:
            pass


def _sync_or_coroutine[T](coroutine: Awaitable[T]) -> T | Awaitable[T]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    return coroutine


def _bundle_outputs(
    bundle: NoteBundle,
    generation_audit: Mapping[str, Any],
    provider: str,
    model: str,
) -> dict[str, str]:
    quality_gate = str(bundle.page_metadata.get("quality_gate") or "fallback")
    return {
        "learning_note.md": bundle.learning_note,
        "douyin_ai_brief.md": bundle.brief_markdown,
        "douyin_ai_brief.json": _json_text(bundle.brief_json),
        "note_budget.json": _json_text(bundle.note_budget),
        "note_score.json": _json_text(bundle.note_score),
        "page_metadata.json": _json_text(bundle.page_metadata),
        "coverage_report.json": _json_text(bundle.coverage_report),
        "evidence_map.json": _json_text(
            bundle.parsed.get(
                "evidence_map",
                {"schema": "douyin-note-evidence-map-v1", "items": []},
            )
        ),
        "generation_audit.json": _json_text(generation_audit),
        "assets/asset_manifest.json": _json_text(bundle.asset_manifest),
        "generation_manifest.json": _json_text(
            {
                "schema": "douyin-note-generation-manifest-v1",
                "provider": provider,
                "model": model,
                "quality_gate": quality_gate,
                "generator_version": (
                    "semantic-v2" if quality_gate == "passed" else GENERATOR_VERSION
                ),
                "generated_files": [
                    "learning_note.md",
                    "douyin_ai_brief.md",
                    "douyin_ai_brief.json",
                    "note_budget.json",
                    "note_score.json",
                    "page_metadata.json",
                    "coverage_report.json",
                    "generation_audit.json",
                    "evidence_map.json",
                    "assets/asset_manifest.json",
                ],
            }
        ),
    }


def _stage_record_dict(record: StageRecord) -> dict[str, Any]:
    return {
        "stage": record.stage,
        "prompt_version": record.prompt_version,
        "system_prompt": record.system_prompt,
        "payload": record.payload,
        "raw_output": record.raw_output,
        "parsed_json": record.parsed_json,
        "usage": record.usage.model_dump(mode="json"),
        "diagnostics": record.diagnostics,
        "elapsed_ms": record.elapsed_ms,
        "batch_boundary": record.batch_boundary,
    }


def _job_is_running(session: Session, job_id: int) -> bool:
    row = session.get(Job, job_id)
    return row is not None and row.status == "running"


def _raise_if_cancelled(cancel_event: asyncio.Event) -> None:
    if cancel_event.is_set():
        raise JobCancelledError("cancelled_by_user")


def _safe_error(error: BaseException) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}:{text[:240]}" if text else type(error).__name__


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _append_item(result: BatchGenerationResult, item: dict[str, Any]) -> None:
    if len(result.items) < 50:
        result.items.append(item)
    else:
        result.items_truncated += 1


def _legacy_state_unchanged(note_dir: Path, expected_state: Any) -> bool:
    """Keep a legacy-only upgrade from replacing a newer protected bundle."""

    current = classify_note_bundle(note_dir)
    if not current.upgrade_eligible or not expected_state.upgrade_eligible:
        return False
    if current.quality_status != expected_state.quality_status:
        return False
    return all(
        getattr(current, field) == getattr(expected_state, field)
        for field in ("provider", "model", "quality_gate", "generator_version")
    )


__all__ = [
    "BatchGenerationResult",
    "GenerationResult",
    "NoteCandidate",
    "NoteGenerationService",
]
