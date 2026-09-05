from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.jobs.handlers import JobHandlerOutcome, JobHandlers
from douyin_library.jobs.queue import JobQueue
from douyin_library.notes.service import GenerationResult


@pytest.mark.asyncio
async def test_generation_result_must_be_a_published_semantic_success(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def preserved(job, cancel_event):
            del job, cancel_event
            return GenerationResult(
                aweme_id="731",
                status="preserved_existing",
                output_dir=str(tmp_path),
                provider="ollama",
                quality_gate="passed",
            )

        handlers.register("generate_note", preserved)
        job = queue.create(kind="generate_note")

        assert await handlers.run_one() is True

        failed = queue.get(job.id)
        assert failed.status == "failed"
        assert failed.error_code == "handler_failed"
        assert "preserved_existing" in (failed.error_detail or "")

    engine.dispose()


@pytest.mark.asyncio
async def test_generation_result_with_passed_ollama_is_a_success(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def generated(job, cancel_event):
            del job, cancel_event
            return GenerationResult(
                aweme_id="731",
                status="generated",
                output_dir=str(tmp_path),
                provider="ollama",
                quality_gate="passed",
            )

        handlers.register("generate_note", generated)
        job = queue.create(kind="generate_note")

        assert await handlers.run_one() is True

        assert queue.get(job.id).status == "succeeded"

    engine.dispose()


@pytest.mark.asyncio
async def test_handler_without_outcome_remains_successful(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def ordinary(job, cancel_event):
            del job, cancel_event

        handlers.register("download_work", ordinary)
        job = queue.create(kind="download_work")

        assert await handlers.run_one() is True

        assert queue.get(job.id).status == "succeeded"

    engine.dispose()


@pytest.mark.asyncio
async def test_explicit_handler_outcome_validates_wrapped_generation_result(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def skipped(job, cancel_event):
            del job, cancel_event
            return JobHandlerOutcome(
                result=GenerationResult(
                    aweme_id="731",
                    status="skipped",
                    output_dir=str(tmp_path),
                )
            )

        handlers.register("generate_note", skipped)
        job = queue.create(kind="generate_note")

        assert await handlers.run_one() is True

        assert queue.get(job.id).status == "failed"

    engine.dispose()


@pytest.mark.asyncio
async def test_explicit_empty_outcome_is_not_a_successful_noop(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        handlers = JobHandlers(queue)

        async def noop(job, cancel_event):
            del job, cancel_event
            return JobHandlerOutcome()

        handlers.register("generate_note", noop)
        job = queue.create(kind="generate_note")

        assert await handlers.run_one() is True

        assert queue.get(job.id).status == "failed"

    engine.dispose()
