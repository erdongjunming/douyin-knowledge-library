import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.cli import run_transcript_backfill, run_transcript_inventory
from douyin_library.db import create_schema
from douyin_library.models import Asset, Job, Work


def _database_with_audio(root: Path) -> Path:
    database_path = root / "data" / "library.db"
    database_path.parent.mkdir(parents=True)
    engine = create_engine(f"sqlite:///{database_path}")
    create_schema(engine)
    with Session(engine) as session:
        for aweme_id in ("200", "100"):
            work = Work(aweme_id=aweme_id)
            session.add(work)
            session.flush()
            audio_path = root / "audio" / f"{aweme_id}.m4a"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"audio-{aweme_id}".encode("ascii")
            audio_path.write_bytes(payload)
            session.add(
                Asset(
                    work_id=work.id,
                    kind="audio",
                    path=str(audio_path),
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    probe_json=json.dumps({"streams": [{"codec_name": "aac"}]}),
                    verified_at=datetime.now(UTC),
                )
            )
        session.commit()
    engine.dispose()
    return database_path


def test_inventory_is_zero_write_and_repeatable(tmp_path: Path, monkeypatch) -> None:
    database_path = _database_with_audio(tmp_path)
    monkeypatch.setattr(
        "douyin_library.cli.Settings",
        lambda: SimpleNamespace(storage_root=tmp_path),
    )
    before = database_path.read_bytes()

    first = run_transcript_inventory(aweme_ids=None, limit=None)
    second = run_transcript_inventory(aweme_ids=None, limit=None)

    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert database_path.read_bytes() == before
    assert first["selected_ids"] == ["100", "200"]


def test_execute_backfill_only_creates_transcribe_jobs(tmp_path: Path, monkeypatch) -> None:
    database_path = _database_with_audio(tmp_path)
    monkeypatch.setattr(
        "douyin_library.cli.Settings",
        lambda: SimpleNamespace(storage_root=tmp_path),
    )

    result = run_transcript_backfill(aweme_ids=["200"], limit=1, execute=True)

    assert result["queued"] == 1
    engine = create_engine(f"sqlite:///{database_path}")
    with Session(engine) as session:
        jobs = session.query(Job).all()
        assert len(jobs) == 1
        assert jobs[0].kind == "transcribe_work"
        payload = json.loads(jobs[0].payload_json or "{}")
        assert set(payload) == {
            "audio_path",
            "audio_sha256",
            "model",
            "phase",
            "provider",
            "provider_policy",
        }
    engine.dispose()
