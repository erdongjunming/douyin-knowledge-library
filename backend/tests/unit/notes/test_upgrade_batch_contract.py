from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.media.storage import StorageLayout
from douyin_library.notes.service import NoteGenerationService


class _FailingProvider:
    name = "ollama"
    model = "qwen3:8b"
    is_local = True
    context_tokens = 32_768

    async def complete_json(self, **kwargs):
        raise RuntimeError("provider_unavailable")

    async def unload(self):
        return None


class _ExternalSemanticProvider(_FailingProvider):
    name = "external"

    async def complete_json(self, **kwargs):
        try:
            from test_pipeline import ScriptedProvider
        except ModuleNotFoundError:
            from tests.unit.notes.test_pipeline import ScriptedProvider

        return await ScriptedProvider().complete_json(**kwargs)


def _seed_legacy_note(root: Path) -> tuple[object, StorageLayout]:
    root.mkdir(parents=True, exist_ok=True)
    layout = StorageLayout(root)
    layout.initialize()
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    create_schema(engine)
    transcript_dir = layout.transcript_dir("731")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    layout.transcript_text_path("731").write_text(
        "第一步记录问题。第二步整理案例。第三步复述验证。",
        encoding="utf-8",
    )
    layout.transcript_segments_path("731").write_text(
        json.dumps(
            [
                {"start": 0, "end": 7, "text": "第一步记录问题。"},
                {"start": 7, "end": 14, "text": "第二步整理案例。"},
                {"start": 14, "end": 20, "text": "第三步复述验证。"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    layout.metadata_path("731").write_text(
        json.dumps({"aweme_id": "731"}),
        encoding="utf-8",
    )
    layout.audio_path("731").write_bytes(b"audio")
    layout.cover_path("731").write_bytes(b"cover")
    note_dir = layout.note_dir("731")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "learning_note.md").write_text("# trusted legacy\n", encoding="utf-8")
    (note_dir / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    return engine, layout


@pytest.mark.asyncio
async def test_legacy_batch_counts_preserved_result_as_failed(tmp_path: Path) -> None:
    engine, layout = _seed_legacy_note(tmp_path / "library")
    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=layout.root, note_provider_policy="local"),
            provider_factory=lambda name: _FailingProvider(),
        )

        result = await service.run_batch(
            legacy_only=True,
            provider_policy="auto",
        )

    assert result.generated == 0
    assert result.failed == 1
    assert result.skipped == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_legacy_upgrade_rejects_provider_that_is_not_ollama(tmp_path: Path) -> None:
    engine, layout = _seed_legacy_note(tmp_path / "library")
    note_path = layout.note_dir("731") / "learning_note.md"
    original = note_path.read_text(encoding="utf-8")

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=layout.root, note_provider_policy="local"),
            provider_factory=lambda name: _ExternalSemanticProvider(),
        )
        result = await service.generate_one(
            "731",
            legacy_only=True,
            provider_policy="auto",
            cancel_event=asyncio.Event(),
        )

    assert result.status != "generated"
    assert note_path.read_text(encoding="utf-8") == original
    engine.dispose()
