from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.service import NoteGenerationService
from tests.integration.test_teaching_note_regression import (
    JIANG_AWEME_ID,
    JiangTeachingProvider,
    _seed_jiang_history,
    _sha256,
)
from tests.unit.notes.test_pipeline import ScriptedProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["success", "http_error", "quality_failure", "unavailable"])
async def test_upgrade_uses_primary_then_ollama_and_preserves_source_assets(
    tmp_path: Path, primary: str,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    settings = Settings(
        storage_root=layout.root,
        note_provider_policy="auto",
        note_external_base_url="https://synthetic.test/v1",
        note_external_model="synthetic-model",
        note_external_api_key=SecretStr("synthetic-placeholder"),
    )
    scripted = JiangTeachingProvider(malformed_sections=primary == "quality_failure")
    fallback = JiangTeachingProvider()
    requests: list[dict] = []
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if primary == "http_error":
            return httpx.Response(503)
        response = await scripted.complete_json(
            stage=body["response_format"]["json_schema"]["name"]
            .removeprefix("note_").removesuffix("_response"),
            payload=json.loads(body["messages"][1]["content"]),
        )
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(response.parsed_json)}}],
        })

    def factory(name):
        calls.append(name)
        assert name == "ollama"
        return fallback

    try:
        with Session(engine) as session:
            service = NoteGenerationService(session, layout, settings, provider_factory=factory)
            result = await service.generate_one(JIANG_AWEME_ID, legacy_only=True)
        assert result.status == "generated"
        expected = "ollama"
        assert calls == ["ollama"]
        assert result.provider == "ollama"
        note_dir = layout.note_dir(JIANG_AWEME_ID)
        manifest = json.loads((note_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((note_dir / "generation_audit.json").read_text(encoding="utf-8"))
        assert manifest["provider"] == expected
        assert manifest["semantic_complete"] is True
        assert [a["provider"] for a in audit["attempts"]] == calls
        assert {"extract", "outline", "section", "review", "depth_review"} <= set(
            fallback.stages
        )
        assert not requests
        for path, digest in tracked.items():
            if note_dir not in path.parents:
                assert _sha256(path) == digest
        assert any(
            _sha256(path) == tracked[note_dir / "learning_note.md"]
            for path in (note_dir / "versions").rglob("learning_note.md")
        )
    finally:
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["auto", "local", "external", "rule"])
async def test_failed_upgrade_preserves_bundle_without_rule_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    settings = Settings(
        storage_root=layout.root,
        note_external_base_url="https://synthetic.test/v1",
        note_external_model="synthetic-model",
        note_external_api_key=SecretStr("synthetic-placeholder"),
    )
    calls: list[str] = []

    def fail(name):
        calls.append(name)
        raise RuntimeError("synthetic_unavailable")

    def forbidden_rule(*args, **kwargs):
        pytest.fail("a rule note cannot replace an existing note as a semantic upgrade")

    monkeypatch.setattr("douyin_library.notes.service.generate_rule_note", forbidden_rule)
    try:
        with Session(engine) as session:
            result = await NoteGenerationService(
                session, layout, settings, provider_factory=fail,
            ).run_batch(legacy_only=True, provider_policy=policy)
        assert calls == ["ollama"]
        assert result.failed == 1
        assert result.generated == 0
        assert result.items[0]["status"] == "preserved_existing"
        assert result.items[0]["error"] == "legacy_upgrade_no_semantic_provider_succeeded"
        assert {path: _sha256(path) for path in tracked} == tracked
        attempt = next(
            (layout.note_dir(JIANG_AWEME_ID) / "attempts").glob("*/generation_audit.json")
        )
        audit = json.loads(attempt.read_text(encoding="utf-8"))
        assert [item["provider"] for item in audit["attempts"]] == ["ollama"]
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_cancelled_local_upgrade_does_not_fallback(tmp_path: Path) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    settings = Settings(
        storage_root=layout.root,
        note_external_base_url="https://synthetic.test/v1",
        note_external_model="synthetic-model",
        note_external_api_key=SecretStr("synthetic-placeholder"),
    )
    calls: list[str] = []

    class CancelledProvider(JiangTeachingProvider):
        name = "ollama"
        is_local = True

        async def complete_json(self, **kwargs):
            raise JobCancelledError("cancelled_by_user")

    def factory(name):
        calls.append(name)
        return CancelledProvider()

    try:
        with Session(engine) as session:
            with pytest.raises(JobCancelledError):
                await NoteGenerationService(
                    session, layout, settings, provider_factory=factory,
                ).generate_one(JIANG_AWEME_ID, legacy_only=True, cancel_event=asyncio.Event())
        assert calls == ["ollama"]
        assert {path: _sha256(path) for path in tracked} == tracked
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_upgrade_dry_run_makes_no_provider_calls_or_file_changes(tmp_path: Path) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")

    def forbidden(name):
        pytest.fail("dry run cannot invoke a provider")

    try:
        with Session(engine) as session:
            service = NoteGenerationService(
                session, layout, Settings(storage_root=layout.root), provider_factory=forbidden,
            )
            before = {p: p.read_bytes() for p in layout.root.rglob("*") if p.is_file()}
            result = await service.run_batch(legacy_only=True, dry_run=True)
            assert result.would_generate == 1
            assert result.generated == 0
            assert {p: p.read_bytes() for p in layout.root.rglob("*") if p.is_file()} == before
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_incomplete_semantic_attempt_is_audited_as_failed_before_fallback(
    tmp_path: Path,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    settings = Settings(
        storage_root=layout.root,
        note_external_base_url="https://synthetic.test/v1",
        note_external_model="synthetic-model",
        note_external_api_key=SecretStr("synthetic-placeholder"),
    )

    class LegacySemanticProvider(ScriptedProvider):
        name = "ollama"
        is_local = True

    def factory(name):
        return LegacySemanticProvider()

    try:
        with Session(engine) as session:
            result = await NoteGenerationService(
                session, layout, settings, provider_factory=factory,
            ).generate_one(JIANG_AWEME_ID, legacy_only=True)
        assert result.status == "preserved_existing"
        assert result.provider is None
        note_path = layout.note_dir(JIANG_AWEME_ID) / "learning_note.md"
        assert _sha256(note_path) == tracked[note_path]
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_legacy_upgrade_reaudits_short_semantic_bundle_without_replacement(
    tmp_path: Path,
) -> None:
    """A stale passed manifest cannot bypass the current local input gate."""

    engine, layout, _ = _seed_jiang_history(tmp_path / "library")
    try:
        with Session(engine) as session:
            published = await NoteGenerationService(
                session,
                layout,
                Settings(storage_root=layout.root, note_provider_policy="local"),
                provider_factory=lambda name: JiangTeachingProvider(),
            ).generate_one(JIANG_AWEME_ID, legacy_only=True)
        assert published.status == "generated"

        sparse_segments = [
            {
                "id": f"sparse-{index}",
                "start": float(index),
                "end": float(index + 1),
                "text": "嗯。",
            }
            for index in range(4)
        ]
        layout.transcript_text_path(JIANG_AWEME_ID).write_text(
            "\n".join(item["text"] for item in sparse_segments), encoding="utf-8"
        )
        layout.transcript_segments_path(JIANG_AWEME_ID).write_text(
            json.dumps(sparse_segments, ensure_ascii=False), encoding="utf-8"
        )
        note_path = layout.note_dir(JIANG_AWEME_ID) / "learning_note.md"
        note_hash = _sha256(note_path)
        provider_calls: list[str] = []

        def forbidden_provider(name: str):
            provider_calls.append(name)
            pytest.fail("an input-insufficient legacy upgrade must not call a provider")

        with Session(engine) as session:
            result = await NoteGenerationService(
                session,
                layout,
                Settings(storage_root=layout.root, note_provider_policy="local"),
                provider_factory=forbidden_provider,
            ).run_batch(legacy_only=True)

        assert result.selected == 1
        assert result.generated == 0
        assert result.failed == 1
        assert result.items[0]["status"] == "preserved_existing"
        assert result.items[0]["error"] == "legacy_upgrade_input_insufficient"
        assert provider_calls == []
        assert _sha256(note_path) == note_hash
    finally:
        engine.dispose()
