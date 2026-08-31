from __future__ import annotations

import argparse
import asyncio
import json
from datetime import timedelta
from pathlib import Path

from sqlalchemy import create_engine, exists, select
from sqlalchemy.orm import Session

from douyin_library.audit import compact_large_audit_payloads
from douyin_library.config import Settings
from douyin_library.db import rebuild_work_fts
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue
from douyin_library.jobs.runtime import ASR_JOB_KINDS, GENERAL_JOB_KINDS, build_worker_runtime
from douyin_library.maintenance import create_database_backup, vacuum_database
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Job, Transcript, Work
from douyin_library.notes.input import NoteInputLoader
from douyin_library.notes.providers.selector import ProviderPolicy
from douyin_library.notes.service import NoteGenerationService
from douyin_library.runtime import initialize_runtime
from douyin_library.transcription.service import _transcript_fingerprint


def _database_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m douyin_library.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebuild_parser = subparsers.add_parser("rebuild-index")
    rebuild_parser.add_argument("--database", type=Path)
    compact_parser = subparsers.add_parser("compact-audits")
    compact_parser.add_argument("--database", type=Path)
    compact_parser.add_argument("--audit-root", type=Path)
    compact_parser.add_argument("--threshold-bytes", type=int)
    compact_parser.add_argument("--vacuum", action="store_true")
    transcribe_parser = subparsers.add_parser("transcribe-one")
    transcribe_parser.add_argument("--aweme-id", required=True)
    clean_parser = subparsers.add_parser("clean-transcripts")
    clean_parser.add_argument("--aweme-id", action="append", dest="aweme_ids")
    clean_parser.add_argument("--limit", type=int)
    clean_parser.add_argument("--provider", choices=("auto", "local", "ollama"), default="auto")
    clean_parser.add_argument("--model")
    clean_parser.add_argument("--dry-run", action="store_true")
    notes_parser = subparsers.add_parser("generate-notes")
    scope = notes_parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int)
    scope.add_argument("--all", action="store_true")
    notes_parser.add_argument("--force", action="store_true")
    notes_parser.add_argument("--legacy-only", action="store_true")
    notes_parser.add_argument(
        "--provider",
        choices=("auto", "local", "ollama", "external", "rule"),
        help="provider selection policy",
    )
    notes_parser.add_argument("--dry-run", action="store_true")
    notes_parser.add_argument("--aweme-id", action="append", dest="aweme_ids")
    args = parser.parse_args(argv)

    if args.command == "rebuild-index":
        database_path = args.database or Settings().storage_root / "data" / "library.db"
        if not database_path.is_file():
            parser.error(f"database does not exist: {database_path}")
        count = rebuild_work_fts(create_engine(_database_url(database_path)))
        print(f"Rebuilt full-text index for {count} works")
        return 0

    if args.command == "compact-audits":
        settings = Settings()
        database_path = args.database or settings.storage_root / "data" / "library.db"
        if not database_path.is_file():
            parser.error(f"database does not exist: {database_path}")
        threshold = args.threshold_bytes or settings.audit_inline_max_bytes
        if threshold < 4096:
            parser.error("--threshold-bytes must be at least 4096")
        backup = create_database_backup(
            database_path,
            backup_root=database_path.parent / "backups",
            audit_root=args.audit_root or settings.storage_root / "audit",
            min_interval=timedelta(0),
            retention=settings.database_backup_retention,
        )
        engine = create_engine(_database_url(database_path))
        try:
            with Session(engine) as session:
                result = compact_large_audit_payloads(
                    session,
                    audit_root=args.audit_root or settings.storage_root / "audit",
                    inline_max_bytes=threshold,
                )
        finally:
            engine.dispose()
        if args.vacuum:
            vacuum_database(database_path)
        result["vacuumed"] = bool(args.vacuum)
        result["backup"] = backup["backup"] if backup else None
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "transcribe-one":
        job_id, status = run_transcribe_one(args.aweme_id)
        print(f"Transcription job {job_id} {status}")
        return 0 if status == "succeeded" else 1

    if args.command == "clean-transcripts":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be at least 1")
        result = run_clean_transcripts(
            aweme_ids=set(args.aweme_ids) if args.aweme_ids else None,
            limit=args.limit,
            dry_run=args.dry_run,
            provider="ollama" if args.provider == "local" else args.provider,
            model=args.model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("failed", 0) else 0

    if args.command == "generate-notes":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be at least 1")
        if args.force and args.legacy_only:
            parser.error("--force cannot be combined with --legacy-only")
        generate_kwargs = {
            "limit": args.limit,
            "force": args.force,
            "dry_run": args.dry_run,
            "aweme_ids": set(args.aweme_ids) if args.aweme_ids else None,
        }
        # Omit new default-valued arguments for old integrations that patch or
        # wrap this function with the original four-key call contract.
        if args.provider is not None:
            generate_kwargs["provider_policy"] = (
                "local" if args.provider == "ollama" else args.provider
            )
        if args.legacy_only:
            generate_kwargs["legacy_only"] = True
        result = run_generate_notes(**generate_kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["failed"] else 0

    return 1


def run_transcribe_one(aweme_id: str) -> tuple[int, str]:
    settings = Settings()
    runtime = initialize_runtime(settings)
    worker_runtime = build_worker_runtime(
        runtime.engine,
        settings,
        worker_kinds=ASR_JOB_KINDS,
        worker_name="asr-cli",
    )
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


def run_generate_notes(
    *,
    limit: int | None,
    force: bool,
    dry_run: bool,
    aweme_ids: set[str] | None = None,
    provider_policy: ProviderPolicy = "auto",
    legacy_only: bool = False,
) -> dict[str, object]:
    settings = Settings()
    runtime = initialize_runtime(settings)
    try:
        with Session(runtime.engine) as session:
            return NoteGenerationService(
                session,
                StorageLayout(settings.storage_root),
            ).run_batch(
                limit=limit,
                force=force,
                dry_run=dry_run,
                aweme_ids=aweme_ids,
                provider_policy=provider_policy,
                legacy_only=legacy_only,
            ).as_dict()
    finally:
        runtime.engine.dispose()


def run_clean_transcripts(
    *,
    aweme_ids: set[str] | None,
    limit: int | None,
    dry_run: bool,
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, object]:
    effective_provider = "ollama" if provider == "local" else provider
    if effective_provider not in {"auto", "ollama"}:
        return {
            "selected": 0,
            "queued": 0,
            "generated": 0,
            "failed": 1,
            "dry_run": dry_run,
            "provider": provider,
            "items": [],
            "error": f"unsupported_cleaner_provider:{provider}",
        }
    settings = Settings()
    if model:
        settings.transcription_cleaner_model = model.strip()
    database_path = settings.storage_root / "data" / "library.db"
    if not database_path.is_file():
        return {"selected": 0, "failed": 1, "items": [], "error": "database_not_found"}
    engine = create_engine(_database_url(database_path))
    try:
        with Session(engine) as session:
            statement = (
                select(Work)
                .where(
                    exists().where(
                        Transcript.work_id == Work.id,
                        Transcript.variant == "raw",
                        Transcript.status == "succeeded",
                    )
                )
                .order_by(Work.id)
            )
            if aweme_ids:
                statement = statement.where(Work.aweme_id.in_(tuple(sorted(aweme_ids))))
            candidates: list[Work] = []
            for work in session.scalars(statement).all():
                raw = session.scalar(
                    select(Transcript)
                    .where(
                        Transcript.work_id == work.id,
                        Transcript.variant == "raw",
                        Transcript.status == "succeeded",
                    )
                    .order_by(Transcript.id.desc())
                )
                if raw is None:
                    continue
                cleaned = session.scalar(
                    select(Transcript).where(
                        Transcript.work_id == work.id,
                        Transcript.variant == "cleaned",
                        Transcript.status == "succeeded",
                        Transcript.parent_transcript_id == raw.id,
                    )
                )
                valid_cleaned = _has_valid_cleaned_variant(
                    work.aweme_id,
                    cleaned,
                    settings=settings,
                    session=session,
                )
                if not valid_cleaned:
                    candidates.append(work)
            selected = candidates[:limit] if limit is not None else candidates
            items: list[dict[str, object]] = []
            if dry_run:
                return {
                    "selected": len(selected),
                    "queued": 0,
                    "generated": 0,
                    "failed": 0,
                    "dry_run": True,
                    "provider": provider,
                    "model": settings.transcription_cleaner_model,
                    "items": [
                        {"aweme_id": work.aweme_id, "status": "would_clean"}
                        for work in selected
                    ],
                }
            queue = JobQueue(session)
            jobs: list[int] = []
            for work in selected:
                raw = session.scalar(
                    select(Transcript)
                    .where(
                        Transcript.work_id == work.id,
                        Transcript.variant == "raw",
                        Transcript.status == "succeeded",
                    )
                    .order_by(Transcript.id.desc())
                )
                if raw is None:
                    continue
                try:
                    job = queue.create(
                        kind="clean_transcript_work",
                        work_id=work.id,
                        payload={
                            "raw_transcript_id": raw.id,
                            "raw_fingerprint": _transcript_fingerprint(raw),
                            "model": settings.transcription_cleaner_model,
                            "prompt_version": settings.transcription_cleaner_prompt_version,
                            "provider": effective_provider,
                        },
                    )
                except DuplicateActiveJobError:
                    items.append({"aweme_id": work.aweme_id, "status": "already_queued"})
                    continue
                jobs.append(job.id)
                items.append({"aweme_id": work.aweme_id, "job_id": job.id, "status": "queued"})
        if not jobs:
            return {
                "selected": len(selected),
                "queued": 0,
                "generated": 0,
                "failed": 0,
                "dry_run": False,
                "items": items,
            }
        runtime = build_worker_runtime(
            engine,
            settings,
            worker_kinds=GENERAL_JOB_KINDS,
            worker_name="clean-cli",
        )
        try:
            for job_id in jobs:
                asyncio.run(_run_job(runtime, job_id))
            generated = 0
            failed = len(jobs) - generated
            for item in items:
                job_id = item.get("job_id")
                if not isinstance(job_id, int):
                    continue
                current = runtime.queue.get(job_id)
                # A worker may leave a retryable job pending; the CLI reports
                # the outcome of this invocation as failed rather than leaking
                # an intermediate queue state to callers.
                item["status"] = "succeeded" if current.status == "succeeded" else "failed"
                if current.status == "succeeded":
                    generated += 1
                elif current.error_detail or current.error_code:
                    item["error"] = current.error_detail or current.error_code
                    item["reason"] = current.error_code or current.error_detail
            failed = len(jobs) - generated
        finally:
            runtime.close()
        return {
            "selected": len(selected),
            "queued": len(jobs),
            "generated": generated,
            "failed": failed,
            "dry_run": False,
            "provider": provider,
            "model": settings.transcription_cleaner_model,
            "items": items,
        }
    finally:
        engine.dispose()


def _has_valid_cleaned_variant(
    aweme_id: str,
    cleaned: Transcript | None,
    *,
    settings: Settings,
    session: Session,
) -> bool:
    if cleaned is None:
        return False
    try:
        loaded = NoteInputLoader(StorageLayout(settings.storage_root), session).load(aweme_id)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return loaded.transcript_variant == "cleaned"


async def _run_job(worker_runtime: object, job_id: int) -> None:
    await worker_runtime.worker.run_job(job_id, ignore_schedule=True)


if __name__ == "__main__":
    raise SystemExit(main())
