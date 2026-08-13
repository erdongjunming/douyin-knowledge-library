from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.main import app
from douyin_library.models import Transcript, Work


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
