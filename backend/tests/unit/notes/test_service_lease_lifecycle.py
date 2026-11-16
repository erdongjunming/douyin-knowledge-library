from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.jobs.leases import ResourceLeaseStore
from douyin_library.models import Analysis, Job, ProcessingAudit
from douyin_library.notes.publisher import NoteBundlePublisher
from douyin_library.notes.service import NoteGenerationService
from tests.integration.test_teaching_note_regression import (
    JIANG_AWEME_ID,
    JiangTeachingProvider,
    _seed_jiang_history,
    _sha256,
)


class _ObservedLeases(ResourceLeaseStore):
    def __init__(self, engine):
        self.now = datetime.now(UTC)
        super().__init__(engine, clock=lambda: self.now)
        self.renewed = asyncio.Event()
        self.acquired = asyncio.Event()
        self.handle = None
        self.renewals = 0

    def try_acquire(self, *args, **kwargs):
        self.handle = super().try_acquire(*args, **kwargs)
        self.acquired.set()
        return self.handle

    def renew(self, handle, **kwargs):
        result = super().renew(handle, **kwargs)
        self.renewals += 1
        self.renewed.set()
        return result


def _settings(root: Path) -> Settings:
    return Settings(storage_root=root, note_provider_policy="local").model_copy(
        update={"job_heartbeat_seconds": 0.01}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_standalone_renews_through_long_generation_and_unload(
    tmp_path: Path, batch: bool,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    competitor = ResourceLeaseStore(engine, clock=lambda: leases.now)

    async def cross_initial_expiry():
        for _ in range(2):
            leases.renewed.clear()
            leases.now += timedelta(seconds=110)
            await asyncio.wait_for(leases.renewed.wait(), timeout=1)
            assert competitor.try_acquire("gpu:0", "competing-worker") is None

    class SlowProvider(JiangTeachingProvider):
        unloaded = False

        async def complete_json(self, **kwargs):
            if not self.stages:
                await cross_initial_expiry()
            return await super().complete_json(**kwargs)

        async def unload(self):
            await cross_initial_expiry()
            assert leases.is_owned(leases.handle)
            self.unloaded = True

    provider = SlowProvider()
    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: provider, lease_store=leases,
            )
            if batch:
                result = await service.run_batch(legacy_only=True)
                assert result.generated == 1
                assert result.failed == 0
            else:
                result = await service.generate_one(JIANG_AWEME_ID, legacy_only=True)
                assert result.status == "generated"
        assert provider.unloaded
        assert leases.renewals >= 4
        assert not leases.is_owned(leases.handle)
        for path, digest in tracked.items():
            if layout.note_dir(JIANG_AWEME_ID) not in path.parents:
                assert _sha256(path) == digest
    finally:
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("renewal_failure", ["stolen", "exception"])
async def test_lease_loss_cancels_generation_and_preserves_old_note(
    tmp_path: Path, renewal_failure: str,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    competitor = ResourceLeaseStore(engine, clock=lambda: leases.now)
    competitor_handle = None

    def fail_renewal(handle, **kwargs):
        nonlocal competitor_handle
        if renewal_failure == "exception":
            raise RuntimeError("synthetic_database_unavailable")
        leases.now += timedelta(seconds=121)
        competitor_handle = competitor.try_acquire("gpu:0", "new-owner")
        assert competitor_handle is not None
        return False

    leases.renew = fail_renewal

    class HangingProvider(JiangTeachingProvider):
        unloaded = False
        request_cancelled = False

        async def complete_json(self, **kwargs):
            try:
                await asyncio.sleep(2)
                return await super().complete_json(**kwargs)
            except asyncio.CancelledError:
                self.request_cancelled = True
                raise

        async def unload(self):
            self.unloaded = True

    provider = HangingProvider()
    cancel_event = asyncio.Event()
    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: provider, lease_store=leases,
            )
            with pytest.raises(JobCancelledError, match="resource_lease_lost"):
                await asyncio.wait_for(service.generate_one(
                    JIANG_AWEME_ID, legacy_only=True, cancel_event=cancel_event,
                ), timeout=0.5)
        assert cancel_event.is_set()
        assert provider.request_cancelled
        assert not provider.unloaded
        assert {path: _sha256(path) for path in tracked} == tracked
        if competitor_handle is not None:
            assert competitor.is_owned(competitor_handle)
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_cancelling_gpu_wait_does_not_unload_current_owner(tmp_path: Path) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    competitor = ResourceLeaseStore(engine)
    competitor_handle = competitor.try_acquire("gpu:0", "current-owner")

    class WaitingProvider(JiangTeachingProvider):
        unloaded = False

        async def unload(self):
            self.unloaded = True

    provider = WaitingProvider()
    cancel_event = asyncio.Event()
    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: provider, lease_store=leases,
            )
            task = asyncio.create_task(service.generate_one(
                JIANG_AWEME_ID, legacy_only=True, cancel_event=cancel_event,
            ))
            await asyncio.wait_for(leases.acquired.wait(), timeout=1)
            cancel_event.set()
            with pytest.raises(JobCancelledError):
                await task
        assert not provider.unloaded
        assert competitor.is_owned(competitor_handle)
        assert {path: _sha256(path) for path in tracked} == tracked
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_cancellation_at_acquisition_releases_without_unloading(tmp_path: Path) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    cancel_event = asyncio.Event()
    real_acquire = leases.try_acquire

    async def cancelling_acquire(*args, **kwargs):
        handle = real_acquire(*args, **kwargs)
        cancel_event.set()
        return handle

    leases.try_acquire = cancelling_acquire

    class CancelledProvider(JiangTeachingProvider):
        unloaded = False

        async def unload(self):
            self.unloaded = True

    provider = CancelledProvider()
    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: provider, lease_store=leases,
            )
            with pytest.raises(JobCancelledError):
                await service.generate_one(
                    JIANG_AWEME_ID, legacy_only=True, cancel_event=cancel_event,
                )
        assert not provider.unloaded
        assert provider.stages == []
        assert not leases.is_owned(leases.handle)
        assert {path: _sha256(path) for path in tracked} == tracked
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_task_cancelled_during_failure_unload_releases_lease(tmp_path: Path) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    unloading = asyncio.Event()

    class FailedProvider(JiangTeachingProvider):
        async def complete_json(self, **kwargs):
            raise RuntimeError("synthetic_provider_failure")

        async def unload(self):
            unloading.set()
            await asyncio.Event().wait()

    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: FailedProvider(), lease_store=leases,
            )
            task = asyncio.create_task(service.generate_one(JIANG_AWEME_ID, legacy_only=True))
            await asyncio.wait_for(unloading.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            audit = session.scalar(select(ProcessingAudit))
            assert audit.status == "cancelled"
            assert audit.finished_at is not None
        assert not leases.is_owned(leases.handle)
        assert not any(
            task.get_name() == f"note-lease-heartbeat-{JIANG_AWEME_ID}"
            for task in asyncio.all_tasks()
        )
        assert {path: _sha256(path) for path in tracked} == tracked
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_worker_managed_generation_does_not_start_standalone_renewal(tmp_path: Path) -> None:
    engine, layout, _ = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    event_loop_thread = threading.get_ident()

    class WorkerProvider(JiangTeachingProvider):
        async def complete_json(self, **kwargs):
            if not self.stages:
                await asyncio.sleep(0.03)
            return await super().complete_json(**kwargs)

    class WorkerPublisher(NoteBundlePublisher):
        def publish(self, *args, **kwargs):
            assert threading.get_ident() == event_loop_thread
            return super().publish(*args, **kwargs)

    try:
        with Session(engine) as session:
            job = Job(kind="generate_note", status="running", worker_id="test-worker")
            session.add(job)
            session.commit()
            result = await NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: WorkerProvider(), lease_store=leases,
                publisher=WorkerPublisher(layout.root),
            ).generate_one(JIANG_AWEME_ID, legacy_only=True, job=job)
        assert result.status == "generated"
        assert leases.renewals == 0
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_slow_publication_renews_lease_without_cross_thread_store_access(
    tmp_path: Path,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    competitor = ResourceLeaseStore(engine, clock=lambda: leases.now)
    loop = asyncio.get_running_loop()
    event_loop_thread = threading.get_ident()
    real_renew = leases.renew
    real_is_owned = leases.is_owned

    def observed_renew(handle, **kwargs):
        assert threading.get_ident() == event_loop_thread
        return real_renew(handle, **kwargs)

    def observed_is_owned(handle):
        assert threading.get_ident() == event_loop_thread
        return real_is_owned(handle)

    leases.renew = observed_renew
    leases.is_owned = observed_is_owned

    async def advance_clock_and_wait_for_renewal():
        leases.renewed.clear()
        leases.now += timedelta(seconds=110)
        await asyncio.wait_for(leases.renewed.wait(), timeout=5)
        assert competitor.try_acquire("gpu:0", "competing-publisher") is None

    class SlowPublisher(NoteBundlePublisher):
        def _replace_outputs(self, staging, note_dir, outputs):
            for _ in range(2):
                asyncio.run_coroutine_threadsafe(
                    advance_clock_and_wait_for_renewal(), loop
                ).result(timeout=5)
            super()._replace_outputs(staging, note_dir, outputs)

    try:
        with Session(engine) as session:
            result = await NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: JiangTeachingProvider(), lease_store=leases,
                publisher=SlowPublisher(layout.root),
            ).generate_one(JIANG_AWEME_ID, legacy_only=True)
        assert result.status == "generated"
        assert leases.renewals >= 2
        assert not leases.is_owned(leases.handle)
        old_note = layout.note_dir(JIANG_AWEME_ID) / "learning_note.md"
        assert any(
            _sha256(path) == tracked[old_note]
            for path in (old_note.parent / "versions").rglob("learning_note.md")
        )
    finally:
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["event", "task", "repeated_task", "lease_loss"])
async def test_publication_cancellation_waits_for_rollback_before_releasing_lease(
    tmp_path: Path, cancellation: str,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    replacing = asyncio.Event()
    finish_replacement = threading.Event()
    publisher_finished = threading.Event()
    cancel_event = asyncio.Event()
    competitor = ResourceLeaseStore(engine, clock=lambda: leases.now)
    competitor_handle = None
    loop = asyncio.get_running_loop()

    class WaitingPublisher(NoteBundlePublisher):
        def _replace_outputs(self, staging, note_dir, outputs):
            super()._replace_outputs(staging, note_dir, outputs)
            loop.call_soon_threadsafe(replacing.set)
            assert finish_replacement.wait(timeout=5)

        def publish(self, *args, **kwargs):
            try:
                return super().publish(*args, **kwargs)
            finally:
                publisher_finished.set()

    service_task = None
    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: JiangTeachingProvider(), lease_store=leases,
                publisher=WaitingPublisher(layout.root),
            )
            service_task = asyncio.create_task(service.generate_one(
                JIANG_AWEME_ID, legacy_only=True, cancel_event=cancel_event,
            ))
            await asyncio.wait_for(replacing.wait(), timeout=2)
            assert not publisher_finished.is_set(), "publication blocked the event loop"
            if cancellation == "event":
                cancel_event.set()
            elif cancellation == "lease_loss":
                leases.now += timedelta(seconds=121)
                competitor_handle = competitor.try_acquire("gpu:0", "new-owner")
                assert competitor_handle is not None
            else:
                service_task.cancel()
            await asyncio.wait_for(cancel_event.wait(), timeout=2)
            if cancellation == "repeated_task":
                service_task.cancel()
                await asyncio.sleep(0)
            assert not service_task.done()
            if cancellation == "lease_loss":
                assert competitor.is_owned(competitor_handle)
            else:
                assert leases.is_owned(leases.handle)
            finish_replacement.set()
            if cancellation in {"event", "lease_loss"}:
                expected_error = (
                    "resource_lease_lost" if cancellation == "lease_loss" else "cancelled_by_user"
                )
                with pytest.raises(JobCancelledError, match=expected_error):
                    await service_task
            else:
                with pytest.raises(asyncio.CancelledError):
                    await service_task
            audit = session.scalar(select(ProcessingAudit))
            assert audit.status == "cancelled"
            assert audit.finished_at is not None
        assert publisher_finished.is_set()
        assert not leases.is_owned(leases.handle)
        if competitor_handle is not None:
            assert competitor.is_owned(competitor_handle)
        assert {path: _sha256(path) for path in tracked} == tracked
        assert not any(
            task.get_name() == f"note-lease-heartbeat-{JIANG_AWEME_ID}"
            for task in asyncio.all_tasks()
        )
    finally:
        finish_replacement.set()
        if service_task is not None:
            with suppress(asyncio.CancelledError, Exception):
                await service_task
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["event", "task", "repeated_task"])
async def test_cancellation_after_publication_fence_records_committed_success(
    tmp_path: Path, cancellation: str,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    leases = _ObservedLeases(engine)
    cancel_event = asyncio.Event()
    fence_passed = asyncio.Event()
    commit_manifest = threading.Event()
    loop = asyncio.get_running_loop()
    publisher = NoteBundlePublisher(layout.root)
    real_replace = publisher.replace

    def wait_before_manifest_replace(source, target):
        if target.name == "generation_manifest.json":
            loop.call_soon_threadsafe(fence_passed.set)
            assert commit_manifest.wait(timeout=5)
        real_replace(source, target)

    publisher.replace = wait_before_manifest_replace
    service_task = None
    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, _settings(layout.root),
                provider_factory=lambda name: JiangTeachingProvider(), lease_store=leases,
                publisher=publisher,
            )
            service_task = asyncio.create_task(service.generate_one(
                JIANG_AWEME_ID, legacy_only=True, cancel_event=cancel_event,
            ))
            await asyncio.wait_for(fence_passed.wait(), timeout=2)
            if cancellation == "event":
                cancel_event.set()
            else:
                service_task.cancel()
            await asyncio.wait_for(cancel_event.wait(), timeout=2)
            if cancellation == "repeated_task":
                service_task.cancel()
                await asyncio.sleep(0)
            assert not service_task.done()
            assert leases.is_owned(leases.handle)
            commit_manifest.set()
            result = await service_task
            assert result.status == "generated"
            assert not service_task.cancelled()
            audit = session.scalar(select(ProcessingAudit))
            assert audit.status == "succeeded"
            assert audit.finished_at is not None
            analysis = session.scalar(select(Analysis))
            assert analysis.status == "succeeded"
            assert analysis.provider == "ollama"
        note_dir = layout.note_dir(JIANG_AWEME_ID)
        manifest = json.loads((note_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["semantic_complete"] is True
        assert any(
            _sha256(path) == tracked[note_dir / "learning_note.md"]
            for path in (note_dir / "versions").rglob("learning_note.md")
        )
        for path, digest in tracked.items():
            if note_dir not in path.parents:
                assert _sha256(path) == digest
        assert not leases.is_owned(leases.handle)
    finally:
        commit_manifest.set()
        if service_task is not None:
            with suppress(asyncio.CancelledError, Exception):
                await service_task
        engine.dispose()
