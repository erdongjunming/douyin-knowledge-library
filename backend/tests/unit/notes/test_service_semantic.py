from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Analysis, Creator, ProcessingAudit, Work
from douyin_library.notes.contracts import ProviderUsage
from douyin_library.notes.providers.base import ProviderResponse
from douyin_library.notes.service import (
    NoteCandidate,
    NoteGenerationService,
    _degraded_generation_is_publishable,
)


def _seed(root: Path, aweme_id: str = "731") -> tuple[object, StorageLayout]:
    root.mkdir(parents=True, exist_ok=True)
    layout = StorageLayout(root)
    layout.initialize()
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        creator = Creator(sec_uid=f"sec-{aweme_id}", nickname="知识作者", unique_id="author")
        session.add(creator)
        session.flush()
        work = Work(
            aweme_id=aweme_id,
            creator_id=creator.id,
            description="建立表达系统",
            duration_ms=20_000,
        )
        session.add(work)
        session.commit()
        work_id = work.id
    transcript_dir = layout.transcript_dir(aweme_id)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    text = "第一步记录问题。第二步整理案例。第三步复述验证。"
    layout.transcript_text_path(aweme_id).write_text(text, encoding="utf-8")
    layout.transcript_segments_path(aweme_id).write_text(
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
    layout.metadata_path(aweme_id).write_text(
        json.dumps({"aweme_id": aweme_id, "description": "建立表达系统"}),
        encoding="utf-8",
    )
    layout.audio_path(aweme_id).write_bytes(b"audio")
    layout.cover_path(aweme_id).write_bytes(b"cover")
    return (engine, work_id), layout


class _FailingProvider:
    name = "ollama"
    model = "qwen3:8b"
    is_local = True
    context_tokens = 32_768

    async def complete_json(self, **kwargs):
        raise RuntimeError("provider_unavailable")

    async def unload(self):
        return None


class _DegradedProvider(_FailingProvider):
    supports_teaching_pipeline = True

    async def complete_json(self, **kwargs):
        from test_pipeline import ScriptedProvider

        return await ScriptedProvider().complete_json(**kwargs)


class _TeachingProvider(_FailingProvider):
    supports_teaching_pipeline = True

    def __init__(self) -> None:
        from test_pipeline import ScriptedProvider

        self._legacy = ScriptedProvider()

    @staticmethod
    def _response(payload: dict) -> ProviderResponse:
        return ProviderResponse(
            raw_output="teaching",
            parsed_json=payload,
            usage=ProviderUsage(
                input_tokens=10,
                output_tokens=5,
                api_cost=0.0,
                cost_status="zero_local",
            ),
            diagnostics={"scripted": True},
        )

    async def complete_json(self, **kwargs):
        stage = kwargs["stage"]
        payload = kwargs["payload"]
        if stage == "outline":
            atom_ids = [atom["id"] for atom in payload["atom_catalog"]]
            section_ids = [
                "overview",
                "core_points",
                "methods_steps",
                "examples_results",
                "boundaries_risks",
                "actions",
                "timeline",
                "evidence_limits",
            ]
            return self._response(
                {
                    "sections": [
                        {
                            "section_id": section_id,
                            "title": section_id,
                            "order": order,
                            "atom_ids": atom_ids if order == 1 else [],
                            "required_evidence_types": [],
                            "missing_evidence_policy": "视频未提供这一类信息。",
                            "timestamps": [],
                        }
                        for order, section_id in enumerate(section_ids, 1)
                    ],
                    "included_atom_ids": atom_ids,
                    "unassigned_atom_ids": [],
                    "input_sufficient": True,
                }
            )
        if stage == "section":
            atoms = payload["atoms"]
            claims = [
                {
                    "text": atom["statement"],
                    "atom_ids": [atom["id"]],
                    "timestamps": [
                        value
                        for value in (atom.get("start"), atom.get("end"))
                        if value is not None
                    ]
                    or [0.0],
                }
                for atom in atoms
            ]
            return self._response(
                {
                    "section_id": payload["section"]["section_id"],
                    "title": payload["section"]["title"],
                    "content": "；".join(claim["text"] for claim in claims)
                    or "视频未提供这一类信息。",
                    "claims": claims,
                    "steps": [],
                    "examples": [],
                    "results": [],
                    "conditions": [],
                    "risks": [],
                    "actions": [],
                    "included_atom_ids": [atom["id"] for atom in atoms],
                    "unprovided_items": [],
                    "generated_interpretations": [],
                }
            )
        if stage == "depth_review":
            return self._response(
                {
                    "abstracted_atom_ids": [],
                    "example_without_result_atom_ids": [],
                    "condition_without_scope_atom_ids": [],
                    "reordered_step_atom_ids": [],
                    "lost_literal_atom_ids": [],
                    "lost_warning_or_promotion_atom_ids": [],
                    "lost_uncertainty_markers": [],
                    "unsupported_claims": [],
                    "repair_section_ids": [],
                    "input_sufficient": True,
                }
            )
        return await self._legacy.complete_json(**kwargs)


@pytest.mark.asyncio
async def test_semantic_service_publishes_and_persists_actual_provider(tmp_path: Path) -> None:
    (engine, work_id), layout = _seed(tmp_path / "library")
    settings = Settings(storage_root=tmp_path / "library", note_provider_policy="local")

    class SemanticProvider(_FailingProvider):
        async def complete_json(self, **kwargs):
            from test_pipeline import ScriptedProvider

            return await ScriptedProvider().complete_json(**kwargs)

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            settings,
            provider_factory=lambda name: SemanticProvider(),
        )
        result = await service.generate_one(
            "731", provider_policy="local", cancel_event=asyncio.Event()
        )
        assert result.status == "generated"
        note_dir = layout.note_dir("731")
        assert (note_dir / "generation_manifest.json").is_file()
        manifest = json.loads((note_dir / "generation_manifest.json").read_text())
        assert manifest["provider"] == "ollama"
        analysis = session.scalar(select(Analysis).where(Analysis.work_id == work_id))
        assert analysis is not None
        assert analysis.provider == "ollama"

    engine.dispose()


@pytest.mark.asyncio
async def test_new_input_insufficiency_publishes_readable_failed_gate(tmp_path: Path) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    layout.metadata_path("731").write_text(
        json.dumps(
            {
                "aweme_id": "731",
                "description": "证据不足视频",
                "visual_dependency": {"risk": "high"},
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(storage_root=tmp_path / "library", note_provider_policy="local")

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            settings,
            provider_factory=lambda name: _DegradedProvider(),
        )
        result = await service.generate_one(
            "731", provider_policy="local", cancel_event=asyncio.Event()
        )

    assert result.status == "generated"
    assert result.quality_gate == "passed"
    manifest = json.loads(
        (layout.note_dir("731") / "generation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["semantic_complete"] is False
    assert json.loads(
        (layout.note_dir("731") / "coverage_report.json").read_text(encoding="utf-8")
    )["input_sufficient"] is True
    engine.dispose()


@pytest.mark.asyncio
async def test_failed_semantic_attempt_preserves_existing_bundle(tmp_path: Path) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    note_dir = layout.note_dir("731")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "learning_note.md").write_text("# trusted", encoding="utf-8")
    settings = Settings(storage_root=tmp_path / "library", note_provider_policy="local")

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            settings,
            provider_factory=lambda name: _FailingProvider(),
        )
        result = await service.generate_one(
            "731", force=True, provider_policy="local", cancel_event=asyncio.Event()
        )
        assert result.status in {"preserved_existing", "preserved_stronger_existing"}
        assert (note_dir / "learning_note.md").read_text() == "# trusted"

    engine.dispose()


@pytest.mark.asyncio
async def test_legacy_only_replaces_legacy_bundle_without_force(tmp_path: Path) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    note_dir = layout.note_dir("731")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "learning_note.md").write_text("# legacy", encoding="utf-8")
    (note_dir / "generation_audit.json").write_text(
        json.dumps(
            {
                "provider": "legacy",
                "model": "local-rule-v1",
            }
        ),
        encoding="utf-8",
    )

    settings = Settings(storage_root=tmp_path / "library", note_provider_policy="local")
    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            settings,
            provider_factory=lambda name: _TeachingProvider(),
        )
        result = await service.generate_one(
            "731",
            legacy_only=True,
            provider_policy="local",
            cancel_event=asyncio.Event(),
        )

    assert result.status == "generated"
    manifest = json.loads(
        (note_dir / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["provider"] == "ollama"
    engine.dispose()


@pytest.mark.asyncio
async def test_protected_publish_is_not_marked_as_succeeded(tmp_path: Path) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    note_dir = layout.note_dir("731")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "learning_note.md").write_text("# legacy", encoding="utf-8")
    (note_dir / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )

    settings = Settings(storage_root=tmp_path / "library", note_provider_policy="rule")
    calls: list[str] = []

    def provider_factory(name: str):
        calls.append(name)
        return _FailingProvider()

    with Session(engine) as session:
        service = NoteGenerationService(session, layout, settings)
        service.provider_factory = provider_factory
        result = await service.generate_one(
            "731",
            legacy_only=True,
            provider_policy="rule",
            cancel_event=asyncio.Event(),
        )
        assert result.status == "preserved_existing"
        assert calls == ["ollama"]
        audit = session.scalar(
            select(ProcessingAudit).where(ProcessingAudit.operation == "generate_note")
        )
        assert audit is not None
        assert audit.status == "failed"

    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "quality_gate"),
    [("ollama", "qwen3:4b", "passed"), ("rule", "local-rule-v1", "fallback")],
)
async def test_legacy_only_skips_protected_bundle_without_provider_call(
    tmp_path: Path,
    provider: str,
    model: str,
    quality_gate: str,
) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    note_dir = layout.note_dir("731")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "learning_note.md").write_text("# protected", encoding="utf-8")
    (note_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                    "provider": provider,
                    "model": model,
                    "quality_gate": quality_gate,
                    "semantic_complete": quality_gate == "passed",
                    "generator_version": "semantic-v2",
            }
        ),
        encoding="utf-8",
    )
    if quality_gate == "passed":
        (note_dir / "coverage_report.json").write_text(
            json.dumps(
                {
                    "quality_gate": "passed",
                    "depth_passed": True,
                    "input_sufficient": True,
                    "semantic_complete": True,
                }
            ),
            encoding="utf-8",
        )
    calls = 0

    def provider_factory(name):
        nonlocal calls
        calls += 1
        if quality_gate == "fallback":
            return _FailingProvider()
        raise AssertionError("semantic-complete bundle must not invoke a provider")

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=tmp_path / "library", note_provider_policy="local"),
            provider_factory=provider_factory,
        )
        result = await service.generate_one(
            "731", legacy_only=True, cancel_event=asyncio.Event()
        )

    if quality_gate == "fallback":
        assert result.status == "preserved_existing"
        assert calls == 1
    else:
        assert result.status == "skipped"
        assert calls == 0
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# protected"
    engine.dispose()


@pytest.mark.asyncio
async def test_legacy_only_uses_ollama_when_external_is_not_configured(
    tmp_path: Path,
) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    note_dir = layout.note_dir("731")
    note_dir.mkdir(parents=True, exist_ok=True)
    original = "# legacy\n"
    (note_dir / "learning_note.md").write_text(original, encoding="utf-8")
    (note_dir / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def provider_factory(name):
        calls.append(name)
        return _FailingProvider()

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=tmp_path / "library", note_provider_policy="auto"),
            provider_factory=provider_factory,
        )
        result = await service.generate_one(
            "731",
            legacy_only=True,
            provider_policy="auto",
            cancel_event=asyncio.Event(),
        )

    assert result.status in {"preserved_existing", "preserved_stronger_existing"}
    assert calls == ["ollama"]
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == original
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["auto", "external", "rule"])
async def test_legacy_only_forces_local_policy_even_when_requested(
    tmp_path: Path,
    policy: str,
) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    note_dir = layout.note_dir("731")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "learning_note.md").write_text("# legacy\n", encoding="utf-8")
    (note_dir / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def provider_factory(name):
        calls.append(name)
        return _FailingProvider()

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=tmp_path / "library", note_provider_policy="auto"),
            provider_factory=provider_factory,
        )
        result = await service.generate_one(
            "731",
            legacy_only=True,
            provider_policy=policy,
            cancel_event=asyncio.Event(),
        )
        audit = session.scalar(
            select(ProcessingAudit).where(ProcessingAudit.operation == "generate_note")
        )

    assert result.status in {"preserved_existing", "preserved_stronger_existing"}
    assert calls == ["ollama"]
    assert audit is not None
    request = json.loads(audit.request_json)
    assert request["provider_policy"] == "local"
    assert request["provider_chain"] == ["ollama"]
    engine.dispose()


def test_legacy_batch_defaults_to_three_candidates(tmp_path: Path, monkeypatch) -> None:
    (engine, _), layout = _seed(tmp_path / "library")
    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=tmp_path / "library", note_provider_policy="local"),
        )
        candidates = [
            NoteCandidate(
                aweme_id=str(index),
                transcript_path=layout.transcript_text_path("731"),
                note_dir=layout.note_dir("731"),
                work_id=None,
                completed=False,
            )
            for index in range(5)
        ]
        monkeypatch.setattr(service, "inventory", lambda **kwargs: (5, 0))
        monkeypatch.setattr(service, "discover", lambda **kwargs: candidates)

        result = service.run_batch(legacy_only=True, dry_run=True)

    assert result.selected == 3
    assert result.would_generate == 3
    engine.dispose()


def test_degraded_publish_rejects_duplicate_timeline_indexes() -> None:
    generation = SimpleNamespace(
        coverage=SimpleNamespace(quality_gate="failed", segment_disposition_count=3),
        context_status="within_budget",
        context_failures=(),
        extraction=SimpleNamespace(dispositions=[object()]),
        synthesis=SimpleNamespace(
            timeline=[
                SimpleNamespace(segment_index=0),
                SimpleNamespace(segment_index=1),
                SimpleNamespace(segment_index=1),
                SimpleNamespace(segment_index=2),
            ]
        ),
    )
    note_input = SimpleNamespace(
        segments=[
            SimpleNamespace(index=0),
            SimpleNamespace(index=1),
            SimpleNamespace(index=2),
        ]
    )

    assert _degraded_generation_is_publishable(generation, note_input) is False
