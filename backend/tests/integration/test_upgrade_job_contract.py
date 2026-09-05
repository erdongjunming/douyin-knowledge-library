from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.runtime import build_worker_runtime
from douyin_library.models import Work
from douyin_library.notes.service import GenerationResult, NoteGenerationService


@pytest.mark.asyncio
async def test_legacy_upgrade_result_is_returned_to_queue_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="734", media_type="video")
        session.add(work)
        session.commit()
        work_id = work.id

    note_dir = root / "dy-note" / "734"
    note_dir.mkdir(parents=True, exist_ok=True)
    original = "# trusted legacy note\n"
    (note_dir / "learning_note.md").write_text(original, encoding="utf-8")
    policies: list[object] = []

    async def fake_generate(self, aweme_id, **kwargs):
        del self, aweme_id
        policies.append(kwargs.get("provider_policy"))
        return GenerationResult(
            aweme_id="734",
            status="preserved_existing",
            output_dir=str(note_dir),
            provider="ollama",
            quality_gate="passed",
            error="provider_unavailable",
        )

    monkeypatch.setattr(NoteGenerationService, "generate_one", fake_generate)
    runtime = build_worker_runtime(engine, Settings(storage_root=root))
    try:
        job = runtime.queue.create(
            kind="generate_note",
            work_id=work_id,
            payload={"aweme_id": "734", "legacy_only": True, "provider_policy": "auto"},
        )

        assert await runtime.handlers.run_one() is True

        failed = runtime.queue.get(job.id)
        assert failed.status == "failed"
        assert policies == ["local"]
        assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == original
    finally:
        runtime.close()
        engine.dispose()
