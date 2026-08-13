from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import create_engine, select

from douyin_library.config import Settings
from douyin_library.db import rebuild_work_fts
from douyin_library.jobs.queue import DuplicateActiveJobError
from douyin_library.jobs.runtime import build_worker_runtime
from douyin_library.models import Job, Work
from douyin_library.runtime import initialize_runtime


def _database_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m douyin_library.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebuild_parser = subparsers.add_parser("rebuild-index")
    rebuild_parser.add_argument("--database", type=Path)
    transcribe_parser = subparsers.add_parser("transcribe-one")
    transcribe_parser.add_argument("--aweme-id", required=True)
    args = parser.parse_args(argv)

    if args.command == "rebuild-index":
        database_path = args.database or Settings().storage_root / "data" / "library.db"
        if not database_path.is_file():
            parser.error(f"database does not exist: {database_path}")
        count = rebuild_work_fts(create_engine(_database_url(database_path)))
        print(f"Rebuilt full-text index for {count} works")
        return 0

    if args.command == "transcribe-one":
        job_id, status = run_transcribe_one(args.aweme_id)
        print(f"Transcription job {job_id} {status}")
        return 0 if status == "succeeded" else 1

    return 1


def run_transcribe_one(aweme_id: str) -> tuple[int, str]:
    settings = Settings()
    runtime = initialize_runtime(settings)
    worker_runtime = build_worker_runtime(runtime.engine, settings)
    try:
        work = worker_runtime.session.scalar(select(Work).where(Work.aweme_id == aweme_id))
        if work is None:
            raise RuntimeError(f"work_not_found:{aweme_id}")
        try:
            job = worker_runtime.queue.create(kind="transcribe_work", work_id=work.id)
        except DuplicateActiveJobError:
            job = worker_runtime.session.scalar(
                select(Job).where(
                    Job.kind == "transcribe_work",
                    Job.work_id == work.id,
                    Job.status.in_(("pending", "running")),
                )
            )
            if job is None:
                raise
            if job.status == "running":
                worker_runtime.queue.recover_stale_jobs()
        asyncio.run(_run_job(worker_runtime, job.id))
        status = worker_runtime.queue.get(job.id).status
        return job.id, status
    finally:
        worker_runtime.close()
        runtime.engine.dispose()


async def _run_job(worker_runtime: object, job_id: int) -> None:
    await worker_runtime.handlers.run_job(job_id, ignore_schedule=True)


if __name__ == "__main__":
    raise SystemExit(main())
