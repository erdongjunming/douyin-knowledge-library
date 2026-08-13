import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue, JobTransitionError
from douyin_library.models import Work


def test_job_queue_persists_json_payload(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        job = JobQueue(session).create(
            kind="sync_creator",
            payload={"reference": "sec-731"},
        )

        assert json.loads(job.payload_json or "{}") == {"reference": "sec-731"}


def test_job_state_machine_and_active_deduplication(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        work = Work(aweme_id="731")
        session.add(work)
        session.commit()
        queue = JobQueue(session)
        job = queue.create(kind="process_work", work_id=work.id)

        with pytest.raises(DuplicateActiveJobError):
            queue.create(kind="process_work", work_id=work.id)
        with pytest.raises(JobTransitionError):
            queue.transition(job.id, "succeeded")

        running = queue.claim_next()
        assert running is not None
        assert running.status == "running"
        queue.transition(job.id, "succeeded")
        assert queue.get(job.id).status == "succeeded"


def test_download_failure_uses_retry_schedule(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="download_work")
        queue.claim_next()
        before = datetime.now(UTC)

        retried = queue.fail(job.id, error_code="network_error", error_detail="offline")

        assert retried.status == "pending"
        assert retried.attempts == 1
        assert retried.next_retry_at is not None
        retry_at = retried.next_retry_at.replace(tzinfo=UTC)
        assert before + timedelta(seconds=55) <= retry_at <= before + timedelta(seconds=65)


def test_successful_retry_preserves_last_failure_for_audit(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="download_work")
        queue.claim_next()
        queue.fail(job.id, error_code="network_error", error_detail="offline")
        retried = queue.get(job.id)
        retried.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()

        queue.claim_next()
        succeeded = queue.transition(job.id, "succeeded")

        assert succeeded.attempts == 1
        assert succeeded.error_code == "network_error"
        assert succeeded.error_detail == "offline"
        assert succeeded.next_retry_at is None


def test_transcription_is_claimed_before_older_download_backlog(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        first = Work(aweme_id="731")
        second = Work(aweme_id="732")
        session.add_all([first, second])
        session.commit()
        queue = JobQueue(session)
        download = queue.create(kind="download_work", work_id=first.id)
        transcribe = queue.create(kind="transcribe_work", work_id=second.id)

        claimed = queue.claim_next()

        assert claimed is not None
        assert claimed.id == transcribe.id
        assert queue.get(download.id).status == "pending"


def test_specific_pending_job_can_be_claimed_without_claiming_queue_head(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        first = queue.create(kind="transcribe_work")
        target = queue.create(kind="sync_creator", payload={"reference": "target"})

        claimed = queue.claim(target.id, ignore_schedule=True)

        assert claimed.id == target.id
        assert claimed.status == "running"
        assert queue.get(first.id).status == "pending"
