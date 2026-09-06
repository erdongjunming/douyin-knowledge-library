from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Transcript, Work
from douyin_library.notes.contracts import ProviderUsage
from douyin_library.notes.input import InputSegment, NoteInput, _raw_fingerprint_payload
from douyin_library.notes.pipeline import SemanticNotePipeline
from douyin_library.notes.providers.base import ProviderResponse
from douyin_library.notes.render import render_semantic_note
from douyin_library.notes.service import NoteGenerationService

try:
    from test_pipeline import ScriptedProvider
except ModuleNotFoundError:  # pragma: no cover - depends on pytest import mode
    from tests.unit.notes.test_pipeline import ScriptedProvider


JIANG_AWEME_ID = "7636753241384695225"
_SECTION_IDS = (
    "overview",
    "core_points",
    "methods_steps",
    "examples_results",
    "boundaries_risks",
    "actions",
    "timeline",
    "evidence_limits",
)


def _jiang_segments() -> list[dict[str, Any]]:
    return [
        {
            "index": 0,
            "start": 0.0,
            "end": 8.0,
            "text": "姜学长分享一套小预算验证方法。",
        },
        {
            "index": 1,
            "start": 8.0,
            "end": 16.0,
            "text": "第一步，先定义真实问题和成功指标。",
        },
        {
            "index": 2,
            "start": 16.0,
            "end": 24.0,
            "text": "第二步，用不超过100元做一个最小实验。",
        },
        {
            "index": 3,
            "start": 24.0,
            "end": 32.0,
            "text": "第三步，每周复盘实验结果，再决定是否继续。",
        },
        {
            "index": 4,
            "start": 32.0,
            "end": 40.0,
            "text": "案例：姜学长用100元测试一个推广页面，结果首周获得10个有效反馈。",
        },
        {
            "index": 5,
            "start": 40.0,
            "end": 48.0,
            "text": "限制：只有每周复盘才适用；ASR可能把姜学长听成姜校长，姓名需人工核对。",
        },
        {
            "index": 6,
            "start": 48.0,
            "end": 56.0,
            "text": "推广：关注我并领取实验模板，下期继续讲。",
        },
    ]


def _jiang_note_input() -> NoteInput:
    segments = _jiang_segments()
    return NoteInput(
        aweme_id=JIANG_AWEME_ID,
        transcript="".join(item["text"] for item in segments),
        segments=[
            # Keep the fixture in the same shape as the persisted ASR loader.
            InputSegment(
                index=item["index"],
                start=item["start"],
                end=item["end"],
                text=item["text"],
                segment_id=f"jiang-segment-{item['index']}",
            )
            for item in segments
        ],
        metadata={"fixture": "jiang-xuezhang"},
        description="姜学长的小预算验证方法",
        author="姜学长",
        author_unique_id="jiang-xuezhang",
        published_at="2026-08-28T08:00:00+08:00",
        duration_ms=56_000,
        source_url=f"https://www.douyin.com/video/{JIANG_AWEME_ID}",
        source_files={},
        asset_records={},
        warnings=[],
    )


def _response(parsed_json: Mapping[str, Any]) -> ProviderResponse:
    return ProviderResponse(
        raw_output="jiang synthetic provider",
        parsed_json=dict(parsed_json),
        usage=ProviderUsage(
            input_tokens=10,
            output_tokens=5,
            api_cost=0.0,
            cost_status="zero_local",
        ),
        diagnostics={"synthetic": True},
    )


def _jiang_atoms(segment: Mapping[str, Any]) -> list[dict[str, Any]]:
    index = int(segment["index"])
    common = {
        "topic": "小预算验证",
        "source_segment": index,
        "start": segment["start"],
        "end": segment["end"],
        "source_excerpt": segment["text"],
        "relations": [],
        "sequence": None,
        "is_promotional": False,
        "asr_uncertainty": [],
    }
    if index == 0:
        return [
            {
                **common,
                "id": "jiang-overview",
                "kind": "claim",
                "statement": segment["text"],
            }
        ]
    if index in {1, 2, 3}:
        return [
            {
                **common,
                "id": f"jiang-step-{index}",
                "kind": "step",
                "statement": segment["text"],
                "sequence": index,
            }
        ]
    if index == 4:
        return [
            {
                **common,
                "start": 32.0,
                "end": 33.0,
                "id": "jiang-case",
                "kind": "example",
                "statement": "案例：姜学长用100元测试一个推广页面",
                "relations": ["jiang-result"],
            },
            {
                **common,
                "start": 34.0,
                "end": 35.0,
                "id": "jiang-result",
                "kind": "claim",
                "statement": "结果首周获得10个有效反馈",
            },
            {
                **common,
                "start": 36.0,
                "end": 37.0,
                "id": "jiang-budget",
                "kind": "number",
                "statement": "100元",
                "source_excerpt": "100元",
            },
        ]
    if index == 5:
        return [
            {
                **common,
                "start": 40.0,
                "end": 41.0,
                "id": "jiang-condition",
                "kind": "condition",
                "statement": "只有每周复盘才适用",
                "relations": ["jiang-overview"],
            },
            {
                **common,
                "start": 42.0,
                "end": 43.0,
                "id": "jiang-asr-risk",
                "kind": "warning",
                "statement": "ASR可能把姜学长听成姜校长，姓名需人工核对",
                "relations": ["jiang-overview"],
                "asr_uncertainty": ["姓名需人工核对"],
            },
        ]
    return [
        {
            **common,
            # A promotion is a point-in-time call to action in this fixture;
            # the enclosing segment still retains its full 48-56s range.
            "end": None,
            "id": "jiang-promotion",
            "kind": "promotion",
            "statement": segment["text"],
            "is_promotional": True,
        }
    ]


class JiangTeachingProvider(ScriptedProvider):
    """Deterministic local provider for the Jiang teaching-note contract."""

    name = "ollama"
    model = "qwen3:8b"
    supports_teaching_pipeline = True

    def __init__(self, *, malformed_sections: bool = False) -> None:
        super().__init__()
        self.malformed_sections = malformed_sections
        self.section_calls = 0

    async def complete_json(self, **kwargs: Any) -> ProviderResponse:
        stage = kwargs["stage"]
        payload = kwargs["payload"]
        self.stages.append(stage)
        self.payloads.append((stage, payload))
        if stage == "extract":
            atoms = [
                atom
                for segment in payload["segments"]
                for atom in _jiang_atoms(segment)
            ]
            dispositions = [
                {
                    "segment_index": segment["index"],
                    "disposition": "promotion" if segment["index"] == 6 else "included",
                    "atom_ids": [
                        atom["id"] for atom in _jiang_atoms(segment)
                    ],
                    "reason": None,
                }
                for segment in payload["segments"]
            ]
            return _response({"atoms": atoms, "dispositions": dispositions})
        if stage == "outline":
            catalog = payload["atom_catalog"]
            atom_ids = [atom["id"] for atom in catalog]

            def atom_id(*, kind: str | None = None, text: str | None = None) -> str:
                for atom in catalog:
                    if kind is not None and atom["kind"] != kind:
                        continue
                    if text is not None and text not in atom["statement"]:
                        continue
                    return atom["id"]
                raise AssertionError(f"fixture atom not found: {kind=} {text=}")

            by_id = {value: value for value in atom_ids}
            assignments = {
                "overview": [atom_id(text="分享一套小预算")],
                "core_points": [
                    atom_id(text="分享一套小预算"),
                    atom_id(kind="promotion"),
                ],
                "methods_steps": [
                    atom["id"] for atom in catalog if atom["kind"] == "step"
                ],
                "examples_results": [
                    atom_id(kind="example"),
                    atom_id(text="结果首周"),
                    atom_id(kind="number"),
                ],
                "boundaries_risks": [
                    atom_id(text="分享一套小预算"),
                    atom_id(kind="condition"),
                    atom_id(kind="warning"),
                ],
                "actions": [],
                "timeline": [],
                "evidence_limits": [],
            }
            sections = []
            for order, section_id in enumerate(_SECTION_IDS, 1):
                section_atom_ids = [
                    value for value in assignments[section_id] if value in by_id
                ]
                sections.append(
                    {
                        "section_id": section_id,
                        "title": section_id,
                        "order": order,
                        "atom_ids": section_atom_ids,
                        "required_evidence_types": [],
                        "missing_evidence_policy": "视频未提供这一类信息。",
                        "timestamps": [
                            next(
                                atom["timestamps"][0]
                                for atom in payload["atom_catalog"]
                                if atom["id"] == atom_id and atom["timestamps"]
                            )
                            for atom_id in section_atom_ids
                        ],
                    }
                )
            return _response(
                {
                    "sections": sections,
                    "included_atom_ids": atom_ids,
                    "unassigned_atom_ids": [],
                    "input_sufficient": True,
                }
            )
        if stage == "section":
            self.section_calls += 1
            section = payload["section"]
            atoms = payload["atoms"]
            if self.malformed_sections:
                return _response(
                    {
                        "section_id": section["section_id"],
                        "title": section["title"],
                        "content": "无效草稿",
                        "claims": [
                            {"text": "无效草稿", "atom_ids": ["unknown-atom"], "timestamps": [0.0]}
                        ],
                        "steps": [],
                        "examples": [],
                        "results": [],
                        "conditions": [],
                        "risks": [],
                        "actions": [],
                        "included_atom_ids": ["unknown-atom"],
                        "unprovided_items": [],
                        "generated_interpretations": [],
                    }
                )
            claims: list[dict[str, Any]] = []
            steps: list[dict[str, Any]] = []
            examples: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []
            conditions: list[dict[str, Any]] = []
            risks: list[dict[str, Any]] = []
            for atom in atoms:
                timestamp = atom["timestamps"][0] if atom["timestamps"] else 0.0
                claim = {
                    "text": atom["statement"],
                    "atom_ids": [atom["id"]],
                    # Theme validation expects the atom start, while the
                    # rendered claim still prints the complete atom range.
                    "timestamps": [timestamp],
                }
                claims.append(claim)
                if atom["kind"] == "step":
                    steps.append(claim)
                if atom["kind"] == "example":
                    examples.append(claim)
                if atom["kind"] == "claim" and atom["statement"].startswith("结果首周"):
                    results.append(claim)
                if atom["kind"] == "condition":
                    conditions.append(claim)
                if atom["kind"] == "warning":
                    risks.append(claim)
            return _response(
                {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "content": "\n".join(atom["statement"] for atom in atoms)
                    or "视频未提供这一类信息。",
                    "claims": claims,
                    "steps": steps,
                    "examples": examples,
                    "results": results,
                    "conditions": conditions,
                    "risks": risks,
                    "actions": [],
                    "included_atom_ids": [atom["id"] for atom in atoms],
                    "unprovided_items": [],
                    "generated_interpretations": [],
                }
            )
        if stage == "depth_review":
            return _response(
                {
                    "abstracted_atom_ids": [],
                    "example_without_result_atom_ids": [],
                    "condition_without_scope_atom_ids": [],
                    "reordered_step_atom_ids": [],
                    "lost_literal_atom_ids": [],
                    "lost_warning_or_promotion_atom_ids": [],
                    "lost_uncertainty_markers": [],
                    "unsupported_claims": (
                        ["synthetic historical upgrade failure"]
                        if self.malformed_sections
                        else []
                    ),
                    "repair_section_ids": [],
                    "input_sufficient": True,
                }
            )
        if stage == "repair":
            return _response(
                {
                    "extraction_patch": {"atoms": [], "dispositions": []},
                    "synthesis_patch": payload["draft"],
                    "section_drafts": payload.get("section_drafts", []),
                }
            )
        if stage == "review":
            if payload.get("review_mode") == "draft_claims":
                claim_ids = [claim["id"] for claim in payload["draft_claims"]]
                return _response(
                    {
                        "reviewed_claim_ids": claim_ids,
                        "supported_claim_ids": claim_ids,
                        "contradicted_claim_ids": [],
                    }
                )
            return _response(
                {
                    "reviewed_segment_indexes": [
                        segment["index"] for segment in payload["source_segments"]
                    ],
                    "source_omission_candidates": [],
                    "unsupported_draft_claims": [],
                    "disposition_corrections": [],
                }
            )
        raise AssertionError(f"unexpected stage: {stage}")


def _section_body(markdown: str, heading: str, next_heading: str) -> str:
    start = markdown.index(heading) + len(heading)
    end = markdown.index(next_heading, start)
    return markdown[start:end]


@pytest.mark.asyncio
async def test_jiang_teaching_note_keeps_facts_in_topic_sections() -> None:
    note_input = _jiang_note_input()
    provider = JiangTeachingProvider()
    generation = await SemanticNotePipeline(provider).generate(
        note_input,
        asyncio.Event(),
    )
    bundle = render_semantic_note(
        note_input,
        generation,
        generated_at="2026-09-06T10:00:00+08:00",
    )

    assert generation.coverage.quality_gate == "passed"
    assert generation.coverage.semantic_complete is True
    assert generation.section_attempts == {section_id: 1 for section_id in _SECTION_IDS}

    methods = _section_body(bundle.learning_note, "## 方法与步骤", "## 案例与反例")
    examples = _section_body(bundle.learning_note, "## 案例与反例", "## 前提、边界与风险")
    boundaries = _section_body(
        bundle.learning_note,
        "## 前提、边界与风险",
        "## 关键概念",
    )
    promotions = _section_body(bundle.learning_note, "## 推广与行动号召", "## 归档标签")

    assert all(value in methods for value in ("第一步", "第二步", "第三步"))
    assert "案例：姜学长" in examples
    assert "结果首周获得10个有效反馈" in examples
    assert "100元" in examples
    assert "只有每周复盘才适用" in boundaries
    assert "ASR" in boundaries
    assert "姓名需人工核对" in boundaries
    assert "关注我并领取实验模板" in promotions

    for field_name in (
        "themes",
        "methods_and_steps",
        "examples_and_counterexamples",
        "boundaries_and_risks",
        "promotions",
    ):
        for item in bundle.parsed[field_name]:
            assert item["atom_ids"], field_name
            assert item["timestamps"], field_name
    assert all(
        item["id"] and item["timestamps"]
        for item in bundle.parsed["evidence_map"]["items"]
    )
    assert all(entry["atom_ids"] for entry in bundle.parsed["timeline"])
    assert all(
        segment["atom_ids"]
        for segment in bundle.parsed["evidence_map"]["dispositions"]
        if segment["disposition"] in {"included", "promotion"}
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_jiang_history(root: Path) -> tuple[Any, StorageLayout, dict[Path, str]]:
    root.mkdir(parents=True, exist_ok=True)
    layout = StorageLayout(root)
    layout.initialize()
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    create_schema(engine)
    segments = _jiang_segments()
    layout.transcript_dir(JIANG_AWEME_ID).mkdir(parents=True, exist_ok=True)
    transcript = "\n".join(item["text"] for item in segments)
    segment_payload = [
        {
            "id": f"jiang-segment-{item['index']}",
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
        }
        for item in segments
    ]
    fingerprint = _raw_fingerprint_payload(transcript, segment_payload, "asr")
    with Session(engine) as session:
        work = Work(
            aweme_id=JIANG_AWEME_ID,
            description="姜学长的小预算验证方法",
            duration_ms=56_000,
        )
        session.add(work)
        session.flush()
        raw = Transcript(
            work_id=work.id,
            engine="qwen3-asr",
            model="0.6b",
            source_kind="asr",
            variant="raw",
            text=transcript,
            segments_json=json.dumps(segment_payload, ensure_ascii=False),
            status="succeeded",
        )
        session.add(raw)
        session.flush()
        session.add(
            Transcript(
                work_id=work.id,
                engine="qwen3-asr",
                model="0.6b",
                source_kind="asr",
                variant="cleaned",
                parent_transcript_id=raw.id,
                text=transcript,
                segments_json=json.dumps(segment_payload, ensure_ascii=False),
                word_timestamps_json=json.dumps({"raw_fingerprint": fingerprint}),
                status="succeeded",
            )
        )
        session.commit()

    layout.transcript_text_path(JIANG_AWEME_ID).write_text(transcript, encoding="utf-8")
    layout.transcript_segments_path(JIANG_AWEME_ID).write_text(
        json.dumps(segment_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    layout.transcript_cleaned_text_path(JIANG_AWEME_ID).write_text(
        transcript,
        encoding="utf-8",
    )
    layout.transcript_cleaned_json_path(JIANG_AWEME_ID).write_text(
        json.dumps(
            {"raw_fingerprint": fingerprint, "segments": segment_payload},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    layout.metadata_path(JIANG_AWEME_ID).write_text(
        json.dumps(
            {
                "aweme_id": JIANG_AWEME_ID,
                "description": "姜学长的小预算验证方法",
                "author": {"nickname": "姜学长", "unique_id": "jiang-xuezhang"},
                "duration_ms": 56_000,
                "source_url": f"https://www.douyin.com/video/{JIANG_AWEME_ID}",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    layout.audio_path(JIANG_AWEME_ID).write_bytes(b"jiang-audio")
    layout.cover_path(JIANG_AWEME_ID).write_bytes(b"jiang-cover")
    note_dir = layout.note_dir(JIANG_AWEME_ID)
    note_dir.mkdir(parents=True, exist_ok=True)
    old_note = note_dir / "learning_note.md"
    old_note.write_text("# 姜学长旧笔记\n\n仅有旧版摘要。\n", encoding="utf-8")
    (note_dir / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    asset_manifest = note_dir / "assets" / "asset_manifest.json"
    asset_manifest.parent.mkdir(parents=True, exist_ok=True)
    asset_manifest.write_text(
        json.dumps(
            {
                "copies_original_assets": False,
                "assets": {
                    "audio": {"sha256": _sha256(layout.audio_path(JIANG_AWEME_ID))},
                    "cover": {"sha256": _sha256(layout.cover_path(JIANG_AWEME_ID))},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tracked = {
        path: _sha256(path)
        for path in (
            old_note,
            layout.audio_path(JIANG_AWEME_ID),
            layout.cover_path(JIANG_AWEME_ID),
            layout.metadata_path(JIANG_AWEME_ID),
            layout.transcript_text_path(JIANG_AWEME_ID),
            layout.transcript_segments_path(JIANG_AWEME_ID),
            layout.transcript_cleaned_text_path(JIANG_AWEME_ID),
            layout.transcript_cleaned_json_path(JIANG_AWEME_ID),
            asset_manifest,
        )
    }
    return engine, layout, tracked


@pytest.mark.asyncio
async def test_failed_jiang_upgrade_reuses_history_and_preserves_all_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, layout, tracked = _seed_jiang_history(tmp_path / "library")
    provider = JiangTeachingProvider(malformed_sections=True)
    forbidden_calls: list[str] = []

    async def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        forbidden_calls.append("unexpected_transcription_or_media_call")
        raise AssertionError("historical note upgrade must not rerun media or ASR")

    monkeypatch.setattr(
        "douyin_library.transcription.service.QwenAsrSubprocess.run",
        forbidden,
    )
    monkeypatch.setattr(
        "douyin_library.transcription.providers.ollama_cleaner.OllamaTranscriptCleaner.clean",
        forbidden,
    )
    monkeypatch.setattr(
        "douyin_library.transcription.service.TranscriptionService.process",
        forbidden,
    )
    monkeypatch.setattr(
        "douyin_library.media.service.WorkMediaService.process",
        forbidden,
    )

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            Settings(storage_root=layout.root, note_provider_policy="local"),
            provider_factory=lambda name: provider,
        )
        result = await service.generate_one(
            JIANG_AWEME_ID,
            legacy_only=True,
            provider_policy="auto",
            cancel_event=asyncio.Event(),
        )

    assert result.status == "preserved_existing"
    assert provider.name == "ollama"
    assert provider.section_calls == len(_SECTION_IDS) * 2
    assert forbidden_calls == []
    assert {path: _sha256(path) for path in tracked} == tracked
    assert (
        layout.note_dir(JIANG_AWEME_ID) / "learning_note.md"
    ).read_text(encoding="utf-8") == "# 姜学长旧笔记\n\n仅有旧版摘要。\n"
    engine.dispose()
