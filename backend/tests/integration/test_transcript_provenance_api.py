from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.main import app
from douyin_library.models import Job, Transcript, Work


@pytest.mark.asyncio
async def test_work_detail_exposes_cleaned_transcript_provenance_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "library"
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="prov-cleaned", description="清洗作品")
        session.add(work)
        session.flush()
        session.add_all(
            [
                Transcript(
                    work_id=work.id,
                    engine="qwen3-asr",
                    model="asr-model",
                    source_kind="asr",
                    variant="raw",
                    text="原始正文",
                    segments_json='[{"id":"s1","start":0,"end":1,"text":"原始正文"}]',
                    status="succeeded",
                ),
                Transcript(
                    work_id=work.id,
                    engine="qwen3-asr",
                    model="asr-model",
                    source_kind="asr",
                    variant="cleaned",
                    parent_transcript_id=1,
                    text="清洗正文",
                    segments_json='[{"id":"s1","start":0,"end":1,"text":"清洗正文"}]',
                    word_timestamps_json='{"cleaner_model":"qwen3:4b","prompt_version":"v1"}',
                    status="succeeded",
                ),
            ]
        )
        session.commit()
        work_id = work.id

    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/works/{work_id}")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert response.status_code == 200
    transcript = response.json()["transcript"]
    assert transcript["text"] == "清洗正文"
    assert transcript["variant"] == "cleaned"
    assert transcript["source_kind"] == "asr"
    assert transcript["cleaner_model"] == "qwen3:4b"
    assert "raw_payload" not in transcript
    assert "cookie" not in response.text.lower()


@pytest.mark.asyncio
async def test_work_detail_reports_cleaning_fallback_reason(tmp_path: Path, monkeypatch) -> None:
    storage = tmp_path / "library"
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="prov-raw")
        session.add(work)
        session.flush()
        session.add(
            Transcript(
                work_id=work.id,
                engine="subtitle-local",
                model="subtitle",
                source_kind="subtitle",
                variant="raw",
                text="字幕正文",
                segments_json='[{"start":0,"end":1,"text":"字幕正文"}]',
                status="succeeded",
            )
        )
        session.add(
            Job(
                work_id=work.id,
                kind="clean_transcript_work",
                status="failed",
                error_code="handler_failed",
                error_detail="cleaner_down",
            )
        )
        session.commit()
        work_id = work.id

    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/works/{work_id}")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert response.status_code == 200
    transcript = response.json()["transcript"]
    assert transcript["variant"] == "raw"
    assert transcript["source_kind"] == "subtitle"
    assert transcript["cleaning_fallback"] == "cleaner_down"


def test_jobs_api_includes_cleaning_kind_in_summary() -> None:
    from douyin_library.api.routes.jobs import serialize_job

    job = SimpleNamespace(
        id=1,
        kind="clean_transcript_work",
        work_id=2,
        status="failed",
        stage="clean",
        progress=0.5,
        attempts=1,
        heartbeat_at=None,
        worker_id=None,
        error_code="cleaner_down",
    )
    assert serialize_job(job)["kind"] == "clean_transcript_work"
