from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.jobs.queue import (
    DuplicateActiveJobError,
    JobQueue,
    JobTransitionError,
)
from douyin_library.models import Work


def _engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    return engine


def test_note_jobs_are_assigned_distinct_lanes_and_canonical_key(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        work = Work(aweme_id="aweme-1")
        session.add(work)
        session.commit()
        work_id = work.id
        queue = JobQueue(session)

        normal = queue.create(kind="generate_note", work_id=work_id)

    with Session(engine) as session:
        queue = JobQueue(session)
        # The normal note is active, so the compatibility legacy request for
        # the same work is rejected across lanes.
        assert normal.queue_lane == "new_note"
        assert normal.dedupe_key == "note:aweme-1"
        with pytest.raises(DuplicateActiveJobError):
            queue.create(
                kind="generate_note",
                work_id=work_id,
                payload={"legacy_only": True},
            )


def test_legacy_payload_maps_to_legacy_lane(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        work = Work(aweme_id="aweme-2")
        session.add(work)
        session.commit()
        work_id = work.id
        job = JobQueue(session).create(
            kind="generate_note",
                work_id=work_id,
            payload={"legacy_only": True},
        )
        assert job.queue_lane == "legacy_upgrade"
        assert job.dedupe_key == "note:aweme-2"


def test_new_note_is_claimed_before_legacy_upgrade(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        first = Work(aweme_id="new")
        old = Work(aweme_id="old")
        session.add_all([first, old])
        session.commit()
        queue = JobQueue(session)
        legacy = queue.create(
            kind="generate_note",
            work_id=old.id,
            payload={"legacy_only": True},
        )
        new = queue.create(kind="generate_note", work_id=first.id)
        claimed = queue.claim_next(worker_id="worker-a")
        assert claimed is not None
        assert claimed.id == new.id
        assert queue.get(legacy.id).status == "pending"


def test_legacy_claim_capacity_is_one(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        first = Work(aweme_id="old-1")
        second = Work(aweme_id="old-2")
        session.add_all([first, second])
        session.commit()
        queue = JobQueue(session)
        one = queue.create(
            kind="generate_note",
            work_id=first.id,
            payload={"legacy_only": True},
        )
        two = queue.create(
            kind="generate_note",
            work_id=second.id,
            payload={"legacy_only": True},
        )
        claimed = queue.claim_next(worker_id="worker-a")
        assert claimed is not None
        assert claimed.id == one.id
        assert queue.claim_next(worker_id="worker-b") is None
        queue.transition(
            one.id,
            "succeeded",
            worker_id="worker-a",
            claim_fence=claimed.claim_fence,
        )
        assert queue.claim_next(worker_id="worker-b").id == two.id


def test_stale_recovery_fences_old_owner_and_cools_legacy_job(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        work = Work(aweme_id="old-stale")
        session.add(work)
        session.commit()
        queue = JobQueue(session)
        job = queue.create(
            kind="generate_note",
            work_id=work.id,
            payload={"legacy_only": True},
        )
        claimed = queue.claim(job.id, worker_id="worker-a")
        old_fence = claimed.claim_fence
        claimed.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()

        assert queue.recover_stale_jobs(stale_after=timedelta(minutes=1)) == 1
        recovered = queue.get(job.id)
        assert recovered.status == "pending"
        assert recovered.claim_fence > old_fence
        assert recovered.attempts == 1
        assert recovered.next_retry_at is not None
        assert not queue.heartbeat(job.id, worker_id="worker-a", claim_fence=old_fence)


def test_fenced_transition_rejects_late_worker(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        job = JobQueue(session).create(kind="transcribe_work")
        queue = JobQueue(session)
        claimed = queue.claim(job.id, worker_id="worker-a")
        with pytest.raises(JobTransitionError, match="ownership"):
            queue.transition(
                job.id,
                "succeeded",
                worker_id="worker-a",
                claim_fence=claimed.claim_fence - 1,
            )


def test_input_insufficient_legacy_failure_is_terminal(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        work = Work(aweme_id="old-input")
        session.add(work)
        session.commit()
        queue = JobQueue(session)
        job = queue.create(
            kind="generate_note",
            work_id=work.id,
            payload={"legacy_only": True},
        )
        claimed = queue.claim(job.id, worker_id="worker-a")
        failed = queue.fail(
            job.id,
            error_code="legacy_upgrade_input_insufficient",
            error_detail="no transcript text",
            worker_id="worker-a",
            claim_fence=claimed.claim_fence,
        )
        assert failed.status == "failed"
        assert failed.attempts == 1
        assert failed.next_retry_at is not None


def test_specific_legacy_claim_obeys_runnable_new_note_priority(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        queue = JobQueue(session)
        old = queue.create(
            kind="generate_note",
            payload={"aweme_id": "old-specific", "legacy_only": True},
        )
        new = queue.create(
            kind="generate_note",
            payload={"aweme_id": "new-specific"},
        )
        with pytest.raises(JobTransitionError):
            queue.claim(old.id, worker_id="worker-old")
        claimed = queue.claim_next(kinds={"generate_note"}, worker_id="worker-new")
        assert claimed is not None
        assert claimed.id == new.id


def test_workless_note_payloads_dedupe_across_policy_and_lane(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        queue = JobQueue(session)
        queue.create(
            kind="generate_note",
            payload={"aweme_id": "workless", "provider_policy": "local"},
        )
        with pytest.raises(DuplicateActiveJobError):
            queue.create(
                kind="generate_note",
                payload={"aweme_id": "workless", "legacy_only": True},
            )


def test_terminal_legacy_failure_is_not_recreated_but_new_note_can_retry(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        work = Work(aweme_id="failed-legacy")
        session.add(work)
        session.commit()
        queue = JobQueue(session)
        legacy = queue.create(
            kind="generate_note",
            work_id=work.id,
            payload={"legacy_only": True},
        )
        claimed = queue.claim(legacy.id, worker_id="worker-a")
        queue.fail(
            legacy.id,
            error_code="legacy_upgrade_input_insufficient",
            error_detail="missing transcript",
            worker_id="worker-a",
            claim_fence=claimed.claim_fence,
        )
        with pytest.raises(DuplicateActiveJobError, match="cooled"):
            queue.create(
                kind="generate_note",
                work_id=work.id,
                payload={"legacy_only": True},
            )
        normal = queue.create(kind="generate_note", work_id=work.id)
        assert normal.queue_lane == "new_note"

