from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.jobs.queue import JobQueue
from douyin_library.models import ProcessingAudit


def test_crashed_running_job_is_requeued(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="process_work")
        queue.claim_next()
        audit = ProcessingAudit(
            job_id=job.id,
            operation="process_work",
            processor="test",
            request_json="{}",
            status="running",
            started_at=datetime.now(UTC) - timedelta(hours=1),
        )
        session.add(audit)
        job.started_at = datetime.now(UTC) - timedelta(hours=1)
        session.commit()

        recovered = queue.recover_stale_jobs(stale_after=timedelta(minutes=10))

        assert recovered == 1
        assert queue.get(job.id).status == "pending"
        assert queue.get(job.id).attempts == 0
        session.refresh(audit)
        assert audit.status == "failed"
        assert audit.error == "worker_stale_recovered"
        assert audit.finished_at is not None
