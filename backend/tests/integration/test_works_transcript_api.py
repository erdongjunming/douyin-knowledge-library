from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.api.routes.works import _work_state
from douyin_library.db import create_schema
from douyin_library.main import app
from douyin_library.models import Analysis, Asset, Job, Transcript, Work


@pytest.mark.asyncio
async def test_work_detail_returns_latest_successful_transcript(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="731", description="知识口述")
        session.add(work)
        session.flush()
        work_id = work.id
        session.add(
            Transcript(
                work_id=work.id,
                engine="qwen3-asr",
                model="Qwen/Qwen3-ASR-0.6B",
                language="Chinese",
                text="这是完整转写正文。",
                segments_json='[{"start":0,"end":2,"text":"这是完整转写正文。"}]',
                status="succeeded",
            )
        )
        session.commit()

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
    assert response.json()["transcript"]["text"] == "这是完整转写正文。"
    assert response.json()["transcript"]["engine"] == "qwen3-asr"


@pytest.mark.asyncio
async def test_work_list_prioritizes_transcribed_items(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        transcribed = Work(aweme_id="732", description="已经转写")
        pending = Work(aweme_id="731", description="最新但尚未转写")
        session.add_all([transcribed, pending])
        session.flush()
        session.add(
            Transcript(
                work_id=transcribed.id,
                engine="qwen3-asr",
                model="Qwen/Qwen3-ASR-0.6B",
                text="优先展示的正文",
                status="succeeded",
            )
        )
        session.commit()

    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/works?limit=1")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert response.status_code == 200
    assert response.json()["items"][0]["aweme_id"] == "732"
    assert response.json()["items"][0]["transcript_preview"] == "优先展示的正文"


@pytest.mark.asyncio
async def test_blank_learning_note_is_not_reported_as_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "library"
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="blank-note", description="空笔记")
        session.add(work)
        session.commit()
        work_id = work.id
    note = storage / "dy-note" / "blank-note" / "learning_note.md"
    note.parent.mkdir(parents=True)
    note.write_text("\ufeff \n", encoding="utf-8")
    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            listing = await client.get("/api/works?limit=10")
            detail = await client.get(f"/api/works/{work_id}")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert listing.status_code == detail.status_code == 200
    assert listing.json()["items"][0]["note_available"] is False
    assert detail.json()["note_available"] is False
    assert detail.json()["note"] is None


@pytest.mark.asyncio
async def test_work_detail_reads_the_complete_note_file_instead_of_analysis_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "library"
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    complete_note = "# 完整学习笔记\n\n## 核心观点\n\n这是应当打开的完整正文。"
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="complete-note", description="完整笔记")
        session.add(work)
        session.flush()
        session.add(
            Analysis(
                work_id=work.id,
                provider="ollama",
                model="qwen3:4b",
                prompt_version="v1",
                summary="只用于列表索引的摘要",
                status="succeeded",
            )
        )
        session.commit()
        work_id = work.id
    note = storage / "dy-note" / "complete-note" / "learning_note.md"
    note.parent.mkdir(parents=True)
    note.write_text(complete_note, encoding="utf-8")

    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            detail = await client.get(f"/api/works/{work_id}")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert detail.status_code == 200
    assert detail.json()["note_available"] is True
    assert detail.json()["note"]["text"] == complete_note
    assert "只用于列表索引的摘要" not in detail.json()["note"]["text"]


@pytest.mark.asyncio
async def test_work_detail_does_not_read_a_note_outside_the_library_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "library"
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    escaped_note = tmp_path / "outside" / "learning_note.md"
    escaped_note.parent.mkdir()
    escaped_note.write_text("# Must not be exposed", encoding="utf-8")
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="../../outside", description="路径边界测试")
        session.add(work)
        session.commit()
        work_id = work.id

    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            listing = await client.get("/api/works?limit=10")
            detail = await client.get(f"/api/works/{work_id}")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert listing.status_code == detail.status_code == 200
    assert listing.json()["items"][0]["note_available"] is False
    assert detail.json()["note_available"] is False
    assert detail.json()["note"] is None


@pytest.mark.asyncio
async def test_work_list_paginates_and_marks_latest_extraction_failure(
    tmp_path: Path, monkeypatch
) -> None:
    storage = tmp_path / "library"
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        works = [Work(aweme_id=f"page-{index}", description=f"资料 {index}") for index in range(3)]
        session.add_all(works)
        session.flush()
        session.add(Asset(work_id=works[0].id, kind="audio", path=str(storage / "audio.m4a")))
        session.add(
            Transcript(
                work_id=works[0].id,
                engine="qwen3-asr",
                model="Qwen/Qwen3-ASR-0.6B",
                text="已完成转写",
                status="succeeded",
            )
        )
        session.add(
            Job(
                work_id=works[1].id,
                kind="download_work",
                status="failed",
                error_code="handler_failed",
                error_detail="media probe failed:",
            )
        )
        session.commit()
        previous_runtime = getattr(app.state, "runtime", None)
        app.state.runtime = SimpleNamespace(engine=engine)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                page = await client.get("/api/works?limit=1&offset=1")
                failures = await client.get("/api/works?limit=10&failed_only=true")
        finally:
            if previous_runtime is None:
                del app.state.runtime
            else:
                app.state.runtime = previous_runtime
            engine.dispose()

    assert page.status_code == 200
    assert page.json()["total"] == 3
    assert page.json()["has_more"] is True
    assert len(page.json()["items"]) == 1
    assert failures.status_code == 200
    assert failures.json()["total"] == 1
    assert failures.json()["failed_total"] == 1
    assert failures.json()["items"][0]["failure"] == {
        "stage": "audio_extraction",
        "reason": "media probe failed（handler_failed）",
    }


def test_note_failure_does_not_override_successful_extraction_state() -> None:
    note_job = SimpleNamespace(
        status="failed",
        error_code="handler_failed",
        error_detail="database or disk is full",
    )

    state = _work_state(
        work_id=1,
        asset_kinds=["audio"],
        transcript_available=True,
        note_available=False,
        latest_jobs={(1, "generate_note"): note_job},
    )

    assert state["extraction_status"] == "succeeded"
    assert state["note_status"] == "failed"
    assert state["failure"] == {
        "stage": "note_generation",
        "reason": "database or disk is full",
    }
