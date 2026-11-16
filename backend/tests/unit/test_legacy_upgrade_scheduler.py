from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.legacy_upgrade import (
    refresh_legacy_upgrade,
    set_legacy_upgrade_enabled,
)
from douyin_library.jobs.queue import JobQueue


@dataclass(frozen=True)
class _Candidate:
    aweme_id: str
    work_id: int | None = None


def _queue(tmp_path: Path) -> tuple[Session, JobQueue]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    session = Session(engine)
    return session, JobQueue(session)


def test_legacy_scheduler_is_disabled_until_explicitly_enabled(tmp_path, monkeypatch) -> None:
    session, queue = _queue(tmp_path)
    monkeypatch.setattr(
        "douyin_library.jobs.legacy_upgrade.NoteGenerationService.discover",
        lambda self, **kwargs: [_Candidate("old-1")],
    )
    try:
        result = refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))
        assert result["queued"] == 0
        assert result["reason"] == "disabled"
        set_legacy_upgrade_enabled(queue, True)
        result = refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))
        assert result["queued"] == 1
        assert result["lane"] == "legacy_upgrade"
    finally:
        session.close()


def test_scheduler_continues_after_completed_batch_without_restarting_failures(
    tmp_path, monkeypatch
):
    session, queue = _queue(tmp_path)
    monkeypatch.setattr(
        "douyin_library.jobs.legacy_upgrade.NoteGenerationService.discover",
        lambda self, **kwargs: [_Candidate(f"old-{index}") for index in range(5)],
    )
    try:
        set_legacy_upgrade_enabled(queue)
        assert refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))["queued"] == 3
        for _ in range(3):
            job = queue.claim_next(worker_id="worker")
            assert job is not None
            queue.fail(
                job.id,
                error_code="handler_failed",
                error_detail="legacy_upgrade_input_insufficient",
            )
        assert refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))["queued"] == 2
    finally:
        session.close()


def test_legacy_scheduler_waits_for_runnable_new_note_and_caps_batch(tmp_path, monkeypatch) -> None:
    session, queue = _queue(tmp_path)
    monkeypatch.setattr(
        "douyin_library.jobs.legacy_upgrade.NoteGenerationService.discover",
        lambda self, **kwargs: [_Candidate(f"old-{index}") for index in range(5)],
    )
    try:
        set_legacy_upgrade_enabled(queue, True)
        queue.create(kind="generate_note", payload={"aweme_id": "new-1"})
        result = refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))
        assert result["queued"] == 0
        assert result["reason"] == "new_note_runnable"
        queue.cancel(1)
        result = refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))
        assert result["queued"] == 3
        # Refresh must respect the outstanding batch, rather than append three
        # more on every click or worker idle poll.
        again = refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))
        assert again["queued"] == 0
        assert again["reason"] == "batch_active"
    finally:
        session.close()


def test_concurrent_refresh_keeps_one_three_item_batch(tmp_path, monkeypatch) -> None:
    session, queue = _queue(tmp_path)
    engine = session.get_bind()
    set_legacy_upgrade_enabled(queue)
    session.close()
    def discover(self, **kwargs):
        return [_Candidate(f"old-{index}") for index in range(6)]

    monkeypatch.setattr(
        "douyin_library.jobs.legacy_upgrade.NoteGenerationService.discover", discover,
    )

    def refresh():
        with Session(engine) as own_session:
            return refresh_legacy_upgrade(JobQueue(own_session), Settings(storage_root=tmp_path))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: refresh(), range(2)))
    assert sorted(result["queued"] for result in results) == [0, 3]


def test_scheduler_reads_enabled_state_fresh_across_sessions(tmp_path, monkeypatch) -> None:
    session, queue = _queue(tmp_path)
    set_legacy_upgrade_enabled(queue)
    monkeypatch.setattr(
        "douyin_library.jobs.legacy_upgrade.NoteGenerationService.discover",
        lambda self, **kwargs: [_Candidate("old-1")],
    )
    try:
        from douyin_library.jobs.legacy_upgrade import legacy_upgrade_enabled
        from douyin_library.models import Setting

        cached = session.get(Setting, "legacy_upgrade.enabled")
        assert cached is not None and legacy_upgrade_enabled(session)
        with Session(session.get_bind()) as other:
            set_legacy_upgrade_enabled(JobQueue(other), False)
        assert (
            refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))["reason"]
            == "disabled"
        )
    finally:
        session.close()


def test_idle_discovery_is_throttled_across_sessions_but_manual_refresh_bypasses(
    tmp_path, monkeypatch,
) -> None:
    session, queue = _queue(tmp_path)
    engine = session.get_bind()
    scans = []
    monkeypatch.setattr(
        "douyin_library.jobs.legacy_upgrade.NoteGenerationService.discover",
        lambda self, **kwargs: scans.append(1) or [],
    )
    settings = Settings(storage_root=tmp_path)
    try:
        set_legacy_upgrade_enabled(queue)
        refresh_legacy_upgrade(queue, settings, throttle=True)
        with Session(engine) as other:
            result = refresh_legacy_upgrade(JobQueue(other), settings, throttle=True)
        assert len(scans) == 1
        assert result["reason"] == "discovery_cooldown"
        refresh_legacy_upgrade(queue, settings)
        assert len(scans) == 2
    finally:
        session.close()


def test_failed_legacy_candidates_are_reported_as_blocked_not_already_queued(
    tmp_path, monkeypatch,
) -> None:
    session, queue = _queue(tmp_path)
    monkeypatch.setattr(
        "douyin_library.jobs.legacy_upgrade.NoteGenerationService.discover",
        lambda self, **kwargs: [_Candidate("old-failed")],
    )
    try:
        set_legacy_upgrade_enabled(queue)
        refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))
        job = queue.claim_next(worker_id="worker")
        queue.fail(job.id, error_code="handler_failed", error_detail="input_insufficient")
        result = refresh_legacy_upgrade(queue, Settings(storage_root=tmp_path))
        assert result["queued"] == 0
        assert result["skipped"] == 0
        assert result["blocked_failed"] == 1
    finally:
        session.close()


