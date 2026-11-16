from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_pipeline import ScriptedProvider, _note_input

from douyin_library.media.storage import StorageLayout
from douyin_library.notes.contracts import (
    EvidenceAtom,
    ExtractionResult,
    SectionDraft,
    TeachingOutline,
    TeachingOutlineSection,
)
from douyin_library.notes.input import InputSegment, NoteInputLoader
from douyin_library.notes.pipeline import ProviderContextBudgetError, SemanticNotePipeline
from douyin_library.notes.providers.base import ProviderResponse
from douyin_library.notes.render import render_semantic_note


@pytest.mark.asyncio
async def test_valid_cleaned_text_cannot_replace_raw_evidence(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / "library")
    layout.initialize()
    aweme_id = "raw-authority"
    layout.transcript_dir(aweme_id).mkdir(parents=True)
    source = _note_input()
    raw = [
        {"id": f"s{s.index}", "start": s.start, "end": s.end, "text": s.text}
        for s in source.segments
    ]
    cleaned = [dict(s) for s in raw]
    cleaned[5]["text"] = "方法适用；注意保留记录。"
    cleaned_text = "".join(s["text"] for s in cleaned)
    fingerprint = hashlib.sha256(
        json.dumps(
            {"text": source.transcript, "segments": raw, "source_kind": "asr"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    layout.transcript_text_path(aweme_id).write_text(source.transcript, encoding="utf-8")
    layout.transcript_segments_path(aweme_id).write_text(
        json.dumps(raw, ensure_ascii=False), encoding="utf-8"
    )
    layout.transcript_cleaned_text_path(aweme_id).write_text(cleaned_text, encoding="utf-8")
    layout.transcript_cleaned_json_path(aweme_id).write_text(
        json.dumps({"raw_fingerprint": fingerprint, "segments": cleaned}, ensure_ascii=False),
        encoding="utf-8",
    )
    layout.metadata_path(aweme_id).write_text("{}", encoding="utf-8")
    layout.audio_path(aweme_id).write_bytes(b"synthetic-audio")
    layout.transcript_qwen_path(aweme_id).write_text("{}", encoding="utf-8")
    before = {p: p.read_bytes() for p in layout.root.rglob("*") if p.is_file()}

    loaded = NoteInputLoader(layout).load(aweme_id)
    provider = ScriptedProvider()
    await SemanticNotePipeline(provider).generate(loaded, asyncio.Event())

    assert loaded.transcript == source.transcript
    assert loaded.transcript_variant == "raw"
    assert loaded.cleaning_fallback is None
    assert loaded.cleaned_transcript == cleaned_text
    assert loaded.cleaned_segments[5].text == cleaned[5]["text"]
    for stage, payload in provider.payloads:
        if stage == "extract":
            assert payload["segments"][5]["text"] == raw[5]["text"]
        if stage == "review" and "source_segments" in payload:
            assert payload["source_segments"][5]["text"] == raw[5]["text"]
    assert all(p.read_bytes() == content for p, content in before.items())


def _long_evidence():
    texts = ["原始细节" * 1100 + "仅低于100元时适用；不要删除记录。"]
    texts.extend(f"案例{i}：" + "具体结果" * 40 for i in range(1, 8))
    segments = [InputSegment(i, None, None, text, f"s{i}") for i, text in enumerate(texts)]
    source = replace(_note_input(), transcript="".join(texts), segments=segments)
    atoms = [
        EvidenceAtom(
            id=f"A{i}", kind="claim", topic="细节", statement=s.text,
            source_segment=i, source_excerpt=s.text,
        )
        for i, s in enumerate(segments)
    ]
    return source, ExtractionResult(atoms=atoms, dispositions=[])


class TeachingRecorder(ScriptedProvider):
    async def complete_json(self, **kwargs: Any) -> ProviderResponse:
        stage, payload = kwargs["stage"], kwargs["payload"]
        self.payloads.append((stage, payload))
        if stage == "outline":
            ids = [a["id"] for a in payload["atom_catalog"]]
            result = TeachingOutline(
                sections=[
                    TeachingOutlineSection(
                        section_id=sid, title=sid, order=i, atom_ids=ids if i == 2 else [],
                        required_evidence_types=[],
                        missing_evidence_policy="视频未提供这一类信息。",
                    )
                    for i, sid in enumerate((
                        "overview", "core_points", "methods_steps", "examples_results",
                        "boundaries_risks", "actions", "timeline", "evidence_limits",
                    ), 1)
                ],
                included_atom_ids=ids,
            )
        elif stage == "section":
            result = SectionDraft(
                section_id=payload["section"]["section_id"], title="细节",
                content="\n".join(a["statement"] for a in payload["atoms"]),
                included_atom_ids=[a["id"] for a in payload["atoms"]],
            )
        else:
            raise AssertionError(stage)
        return ProviderResponse(
            raw_output=result.model_dump_json(), parsed_json=result.model_dump(mode="json"),
            usage=None, diagnostics={},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("stage", "context_tokens"), [("outline", 26_000), ("section", 32_000)])
async def test_teaching_batches_preserve_every_statement_and_source_tail(
    stage: str, context_tokens: int,
) -> None:
    source, extraction = _long_evidence()
    provider = TeachingRecorder()
    provider.context_tokens = context_tokens
    pipeline = SemanticNotePipeline(provider)
    if stage == "outline":
        await pipeline._build_outline(source, extraction, asyncio.Event(), None, [])
    else:
        outline = TeachingOutline(
            sections=[TeachingOutlineSection(
                section_id="core_points", title="细节", order=1,
                atom_ids=[a.id for a in extraction.atoms], required_evidence_types=[],
                missing_evidence_policy="视频未提供这一类信息。",
            )], included_atom_ids=[a.id for a in extraction.atoms],
        )
        drafts, _ = await pipeline._synthesize_sections(
            source, extraction, outline, asyncio.Event(), None, [],
        )
        assert drafts[0].unprovided_items == []
    payloads = [payload for name, payload in provider.payloads if name == stage]
    assert len(payloads) > 1
    key = "atom_catalog" if stage == "outline" else "atoms"
    delivered = {a["id"]: a["statement"] for p in payloads for a in p[key]}
    assert delivered == {a.id: a.statement for a in extraction.atoms}
    sources = {s["index"]: s["text"] for p in payloads for s in p["source_segments"]}
    assert sources == {s.index: s.text for s in source.segments}
    assert all(p.get("input_truncated", False) is False for p in payloads)
    assert all(p["context_budget"]["within_budget"] for p in payloads)
    assert all(s["start"] is None and s["end"] is None for p in payloads
               for s in p["source_segments"])


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["outline", "section"])
async def test_irreducible_teaching_input_reports_context_failure(stage: str) -> None:
    source, extraction = _long_evidence()
    extraction = extraction.model_copy(update={"atoms": [extraction.atoms[0]]})
    provider = TeachingRecorder()
    provider.context_tokens = 12_000
    pipeline = SemanticNotePipeline(provider)
    if stage == "outline":
        with pytest.raises(ProviderContextBudgetError, match="outline:"):
            await pipeline._build_outline(source, extraction, asyncio.Event(), None, [])
    else:
        outline = TeachingOutline(
            sections=[TeachingOutlineSection(
                section_id="core_points", title="细节", order=1, atom_ids=["A0"],
                required_evidence_types=[], missing_evidence_policy="视频未提供这一类信息。",
            )], included_atom_ids=["A0"],
        )
        drafts, _ = await pipeline._synthesize_sections(
            source, extraction, outline, asyncio.Event(), None, []
        )
        assert drafts[0].unprovided_items == ["core_points"]
    assert provider.payloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("risk", [None, "medium", "high"])
async def test_text_can_pass_with_visual_coverage_explicitly_unreviewed(risk: str | None) -> None:
    source = replace(_note_input(), note_budget={"visual_dependency": {"risk": risk}})
    result = await SemanticNotePipeline(ScriptedProvider()).generate(source, asyncio.Event())
    bundle = render_semantic_note(source, result, generated_at="2026-09-10T00:00:00Z")
    assert result.coverage.quality_gate == "passed"
    for output in (bundle.parsed, bundle.coverage_report, bundle.page_metadata,
                   bundle.brief_json, bundle.note_budget, bundle.asset_manifest):
        assert output["coverage_scope"] == "transcript"
        assert output["text_coverage"]["quality_gate"] == "passed"
        assert output["visual_coverage"]["status"] == "not_reviewed"
        assert output["visual_coverage"]["dependency_risk"] == (risk or "unknown")
        assert output["visual_coverage"]["limitations"]
    assert "画面覆盖：未核验" in bundle.learning_note
    assert "现有材料未提供" in bundle.learning_note


@pytest.mark.asyncio
@pytest.mark.parametrize("warnings", [["synthetic segment"], ["转写稀疏"]])
async def test_text_insufficiency_still_fails_with_visual_risk(warnings: list[str]) -> None:
    source = replace(_note_input(), warnings=warnings,
                     metadata={"visual_dependency": {"risk": "high"}})
    result = await SemanticNotePipeline(ScriptedProvider()).generate(source, asyncio.Event())
    assert result.coverage.quality_gate == "failed"
    assert result.input_sufficient is False
