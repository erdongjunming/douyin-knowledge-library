from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _runner():
    path = Path(__file__).parents[3] / "scripts" / "upgrade_legacy_notes.py"
    spec = importlib.util.spec_from_file_location("upgrade_legacy_notes", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_keeps_three_item_batches() -> None:
    runner = _runner()
    assert runner.chunked([str(index) for index in range(7)]) == [
        ("0", "1", "2"),
        ("3", "4", "5"),
        ("6",),
    ]


def test_runner_retries_recorded_failures() -> None:
    runner = _runner()
    rows = [
        {
            "batch_index": 0,
            "status": "succeeded",
            "aweme_ids": ["a", "b", "c"],
            "source_unchanged": True,
        },
        {"batch_index": 1, "status": "completed_with_failures"},
        {"batch_index": 2, "status": "failed"},
        {
            "batch_index": 3,
            "status": "succeeded",
            "aweme_ids": ["old-inventory"],
            "source_unchanged": True,
        },
    ]
    batches = [("a", "b", "c"), ("d",), ("e",), ("new-inventory",)]
    assert runner.completed_batches(rows, batches) == {0}


def test_runner_skips_terminal_low_information_batch_only_when_all_items_are_preserved() -> None:
    runner = _runner()
    rows = [
        {
            "aweme_ids": ["low", "ok"],
            "status": "completed_with_failures",
            "source_unchanged": True,
            "verification": {"semantic_complete": False, "old_notes_preserved": True},
            "result": {
                "items": [
                    {
                        "aweme_id": "low",
                        "status": "preserved_existing",
                        "error": "legacy_upgrade_input_insufficient",
                    },
                    {"aweme_id": "ok", "status": "generated", "error": None},
                ]
            },
        },
    ]
    assert runner.completed_batches(rows, [("low", "ok")]) == {0}


@pytest.mark.parametrize(
    "row_update",
    [
        {"source_unchanged": False},
        {"verification": {"semantic_complete": False, "old_notes_preserved": False}},
        {
            "result": {
                "items": [
                    {
                        "aweme_id": "low",
                        "status": "preserved_existing",
                        "error": "legacy_upgrade_semantic_incomplete",
                    }
                ]
            }
        },
        {
            "result": {
                "items": [
                    {"aweme_id": "low", "status": "failed", "error": "ollama_unavailable"}
                ]
            }
        },
    ],
)
def test_runner_retries_non_terminal_or_unverified_low_information_batch(row_update) -> None:
    runner = _runner()
    row = {
        "aweme_ids": ["low"],
        "status": "completed_with_failures",
        "source_unchanged": True,
        "verification": {"semantic_complete": False, "old_notes_preserved": True},
        "result": {
            "items": [
                {
                    "aweme_id": "low",
                    "status": "preserved_existing",
                    "error": "legacy_upgrade_input_insufficient",
                }
            ]
        },
    }
    row.update(row_update)
    assert runner.completed_batches([row], [("low",)]) == set()


def test_runner_refuses_remote_ollama_even_with_local_policy() -> None:
    from douyin_library.config import Settings

    runner = _runner()
    with pytest.raises(ValueError, match="loopback"):
        runner.validate_local_settings(Settings(note_ollama_url="https://example.com"))
    runner.validate_local_settings(Settings())


def test_zero_failed_does_not_mark_missing_bundle_complete(tmp_path: Path) -> None:
    runner = _runner()
    result = runner.verify_batch(tmp_path, ("missing",), {})
    assert result["semantic_complete"] is False
    assert result["items"][0]["quality_status"] == "missing"


def test_failed_candidate_checks_old_note_hash(tmp_path: Path) -> None:
    runner = _runner()
    note = tmp_path / "dy-note" / "old" / "learning_note.md"
    note.parent.mkdir(parents=True)
    note.write_text("original", encoding="utf-8")
    before = runner.note_hashes(tmp_path, ("old",))
    assert runner.verify_batch(tmp_path, ("old",), before)["old_notes_preserved"]
    note.write_text("unexpected replacement", encoding="utf-8")
    assert not runner.verify_batch(tmp_path, ("old",), before)["old_notes_preserved"]


def test_resume_uses_first_checkpoint_note_hash_as_preservation_baseline() -> None:
    runner = _runner()
    original = {"exists": True, "bytes": 8, "sha256": "original"}
    overwritten = {"exists": True, "bytes": 12, "sha256": "replacement"}
    checkpoints = [
        {
            "event": "batch_started",
            "batch_index": 4,
            "aweme_ids": ["work"],
            "old_notes_before": {"work": original},
        },
        {
            "event": "batch_started",
            "batch_index": 4,
            "aweme_ids": ["work"],
            "old_notes_before": {"work": overwritten},
        },
    ]

    assert runner.checkpoint_note_baseline(checkpoints, 4, ("work",)) == {"work": original}


def test_only_one_runner_can_hold_storage_lock(tmp_path: Path) -> None:
    runner = _runner()
    lock = tmp_path / "runner.lock"
    with runner.runner_lock(lock):
        with pytest.raises(RuntimeError, match="already_running"):
            with runner.runner_lock(lock):
                pytest.fail("second runner acquired lock")
    with runner.runner_lock(lock):
        pass


def test_incomplete_semantic_bundle_is_not_a_completed_upgrade(tmp_path: Path) -> None:
    runner = _runner()
    note_dir = tmp_path / "dy-note" / "test"
    note_dir.mkdir(parents=True)
    (note_dir / "learning_note.md").write_text("# Ready", encoding="utf-8")
    metadata = {"provider": "ollama", "model": "qwen3:4b", "quality_gate": "passed",
                "semantic_complete": True, "depth_passed": True, "input_sufficient": True}
    for name in ("generation_manifest.json", "coverage_report.json"):
        (note_dir / name).write_text(json.dumps(metadata), encoding="utf-8")
    result = runner.verify_batch(tmp_path, ("test",), {})
    assert result["items"][0]["quality_status"] == "unknown"
    assert not result["semantic_complete"]


def _complete_bundle(root: Path, *, include_manifest: bool = False) -> Path:
    from douyin_library.notes.publisher import GENERATED_FILES

    note_dir = root / "dy-note" / "test"
    transcript_dir = root / "transcript" / "test"
    transcript_dir.mkdir(parents=True)
    transcript = "这是完整转写文本，用于验证历史笔记升级完成状态，并覆盖当前输入证据。"
    (transcript_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    (transcript_dir / "segments.json").write_text(
        json.dumps([
            {"id": "source-0", "start": 0.0, "end": 1.0, "text": transcript},
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    identity = {
        "provider": "ollama", "model": "qwen3:4b", "quality_gate": "passed",
        "generator_version": "semantic-v2", "semantic_complete": True,
        "depth_passed": True, "input_sufficient": True,
    }
    outputs = {
        "learning_note.md": "# Ready\n\nA complete note.",
        "douyin_ai_brief.md": "# Brief",
        "douyin_ai_brief.json": {"title": "Brief", "quality_gate": "passed"},
        "note_budget.json": {"schema": "douyin-note-budget-v2"},
        "note_score.json": {"quality_gate": "passed"},
        "page_metadata.json": identity,
        "coverage_report.json": identity,
        "generation_audit.json": {"status": "succeeded"},
        "evidence_map.json": {"items": []},
        "assets/asset_manifest.json": {"copies_original_assets": False},
        "generation_manifest.json": {
            **identity,
            "generated_files": [*GENERATED_FILES] + (
                ["generation_manifest.json"] if include_manifest else []
            ),
        },
    }
    for name, value in outputs.items():
        path = note_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return note_dir


@pytest.mark.parametrize("include_manifest", [False, True])
def test_complete_bundle_accepts_both_manifest_list_conventions(
    tmp_path: Path, include_manifest: bool,
) -> None:
    _complete_bundle(tmp_path, include_manifest=include_manifest)
    assert _runner().verify_batch(tmp_path, ("test",), {})["semantic_complete"]


def test_complete_manifest_with_current_sparse_input_is_not_complete(tmp_path: Path) -> None:
    _complete_bundle(tmp_path)
    transcript_dir = tmp_path / "transcript" / "test"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    segments = [
        {
            "id": f"sparse-{index}",
            "start": float(index),
            "end": float(index + 1),
            "text": "嗯。",
        }
        for index in range(4)
    ]
    (transcript_dir / "transcript.txt").write_text(
        "\n".join(segment["text"] for segment in segments), encoding="utf-8"
    )
    (transcript_dir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False), encoding="utf-8"
    )

    result = _runner().verify_batch(tmp_path, ("test",), {})

    assert result["items"][0]["quality_status"] == "unknown"
    assert result["semantic_complete"] is False


@pytest.mark.parametrize("value", [None, [], 42, "manifest"])
def test_invalid_manifest_root_fails_closed_without_aborting(tmp_path: Path, value) -> None:
    note_dir = _complete_bundle(tmp_path)
    (note_dir / "generation_manifest.json").write_text(json.dumps(value), encoding="utf-8")
    assert not _runner().verify_batch(tmp_path, ("test",), {})["semantic_complete"]


@pytest.mark.parametrize("name,content", [
    (name, content)
    for name in (
        "generation_audit.json", "evidence_map.json", "note_budget.json", "note_score.json",
        "page_metadata.json", "douyin_ai_brief.json", "assets/asset_manifest.json",
    )
    for content in ("{", "[]", "{}")
    if not (name == "evidence_map.json" and content == "{}")
])
def test_invalid_companion_json_is_not_complete(tmp_path: Path, name: str, content: str) -> None:
    note_dir = _complete_bundle(tmp_path)
    (note_dir / name).write_text(content, encoding="utf-8")
    assert not _runner().verify_batch(tmp_path, ("test",), {})["semantic_complete"]


@pytest.mark.parametrize("listed", [None, [], ["learning_note.md"], {"learning_note.md": True}])
def test_incomplete_manifest_file_list_is_not_complete(tmp_path: Path, listed) -> None:
    note_dir = _complete_bundle(tmp_path)
    path = note_dir / "generation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["generated_files"] = listed
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _runner().verify_batch(tmp_path, ("test",), {})["semantic_complete"]


@pytest.mark.parametrize("extra", ["../transcript.txt", "unknown.json", None, {}])
def test_invalid_manifest_file_entry_is_not_complete(tmp_path: Path, extra) -> None:
    note_dir = _complete_bundle(tmp_path)
    path = note_dir / "generation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["generated_files"].append(extra)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _runner().verify_batch(tmp_path, ("test",), {})["semantic_complete"]


@pytest.mark.parametrize("name,key,value", [
    ("generation_audit.json", "status", "failed"),
    ("generation_audit.json", "status", "running"),
    ("note_score.json", "quality_gate", "failed"),
    ("douyin_ai_brief.json", "quality_gate", "fallback"),
    ("page_metadata.json", "quality_gate", "failed"),
    ("page_metadata.json", "provider", "external"),
    ("page_metadata.json", "model", "other-model"),
])
def test_inconsistent_bundle_cannot_pass_completion(tmp_path: Path, name, key, value) -> None:
    note_dir = _complete_bundle(tmp_path)
    path = note_dir / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not _runner().verify_batch(tmp_path, ("test",), {})["semantic_complete"]


def test_resume_recovers_truncated_final_record_and_preserves_original(tmp_path: Path) -> None:
    runner = _runner()
    path = tmp_path / "audit.jsonl"
    original = b'{"status":"succeeded"}\n{"status":"succ'
    path.write_bytes(original)
    assert runner._read_jsonl(path) == [{"status": "succeeded"}]
    assert [p.read_bytes() for p in tmp_path.glob("audit.jsonl.interrupted-*")] == [original]
    runner._append_jsonl(path, {"status": "failed"})
    assert runner._read_jsonl(path) == [{"status": "succeeded"}, {"status": "failed"}]


def test_middle_record_corruption_is_reported_without_rewriting(tmp_path: Path) -> None:
    runner = _runner()
    path = tmp_path / "audit.jsonl"
    original = '{"status":"succeeded"}\n{\n{"status":"failed"}\n'
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        runner._read_jsonl(path)
    assert path.read_text(encoding="utf-8") == original


def test_append_preserves_valid_record_without_trailing_newline(tmp_path: Path) -> None:
    runner = _runner()
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b'{"status":"succeeded"}')
    runner._append_jsonl(path, {"status": "failed"})
    assert runner._read_jsonl(path) == [{"status": "succeeded"}, {"status": "failed"}]


def test_checkpoint_append_recovers_interrupted_utf8_write(tmp_path: Path) -> None:
    runner = _runner()
    path = tmp_path / "checkpoints.jsonl"
    original = b'{"status":"started"}\n{"value":"\xe4\xb8'
    path.write_bytes(original)
    runner._append_jsonl(path, {"status": "finished"})
    assert runner._read_jsonl(path) == [{"status": "started"}, {"status": "finished"}]
    assert [p.read_bytes() for p in tmp_path.glob("checkpoints.jsonl.interrupted-*")] == [original]


@pytest.mark.parametrize("damage", ["source", "archive", "bundle"])
@pytest.mark.parametrize(
    "prior_status", ["succeeded", "integrity_failed", "completed_with_failures"],
)
def test_resume_rechecks_sources_archives_and_bundle_before_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str, prior_status: str,
) -> None:
    from argparse import Namespace
    from types import SimpleNamespace

    from sqlalchemy import create_engine

    from douyin_library.config import Settings
    from douyin_library.notes import ollama_runtime
    from douyin_library.notes.service import NoteGenerationService

    runner = _runner()
    note_dir = _complete_bundle(tmp_path)
    archive = note_dir / "versions" / "original" / "learning_note.md"
    archive.parent.mkdir(parents=True)
    archive.write_text("Original historical note", encoding="utf-8")
    baseline = {"test": runner._sha256(archive)}
    sources = runner.source_hashes(tmp_path, ("test",))
    audit = tmp_path / "audit" / "batches.jsonl"
    runner._append_jsonl(audit.with_suffix(".checkpoints.jsonl"), {
        "event": "batch_started", "batch_index": 0, "aweme_ids": ["test"],
        "source_hashes_before": sources, "old_notes_before": baseline,
    })
    runner._append_jsonl(audit, {
        "batch_index": 0, "aweme_ids": ["test"], "status": prior_status,
        "source_unchanged": True, "source_hashes_after": sources,
    })
    if damage == "source":
        (tmp_path / "transcript" / "test" / "transcript.txt").write_text(
            "Changed source", encoding="utf-8",
        )
    elif damage == "archive":
        archive.unlink()
    else:
        (note_dir / "evidence_map.json").write_text("{", encoding="utf-8")

    async def unexpected_generation(*args, **kwargs):
        pytest.fail("resume must stop on damaged completed output or lost preservation evidence")

    async def ready(settings):
        return SimpleNamespace(ready=True)

    monkeypatch.setattr(ollama_runtime, "probe_ollama", ready)
    monkeypatch.setattr(NoteGenerationService, "run_batch", unexpected_generation)
    engine = create_engine("sqlite://")
    try:
        code = runner._run_batches(
            Namespace(only_batch=None), Settings(storage_root=tmp_path),
            SimpleNamespace(engine=engine), [("test",)],
            {0} if prior_status == "succeeded" else set(), audit,
        )
    finally:
        engine.dispose()
    assert code == 3
    rows = runner._read_jsonl(audit)
    assert rows[-1]["status"] == "integrity_failed"


def test_archived_old_note_is_preserved_even_if_current_bundle_is_damaged(tmp_path: Path) -> None:
    runner = _runner()
    note_dir = _complete_bundle(tmp_path)
    archive = note_dir / "versions" / "original" / "learning_note.md"
    archive.parent.mkdir(parents=True)
    archive.write_text("original", encoding="utf-8")
    before = {"test": runner._sha256(archive)}
    (note_dir / "coverage_report.json").write_text("{", encoding="utf-8")
    verified = runner.verify_batch(tmp_path, ("test",), before)
    assert verified["old_notes_preserved"] is True
    assert verified["semantic_complete"] is False


@pytest.mark.parametrize("service_status,model_available", [("stopped", False), ("ready", False)])
def test_unavailable_local_model_stops_before_creating_batch_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service_status: str, model_available: bool,
) -> None:
    from argparse import Namespace
    from types import SimpleNamespace

    from sqlalchemy import create_engine

    from douyin_library.config import Settings
    from douyin_library.notes import ollama_runtime
    from douyin_library.notes.service import NoteGenerationService

    runner = _runner()
    audit = tmp_path / "audit" / "batches.jsonl"
    probe_calls = []

    async def probe(settings):
        probe_calls.append(settings.note_ollama_url)
        return ollama_runtime.OllamaStatus(
            executable_found=True, executable_path="ollama.exe", service_status=service_status,
            model_available=model_available, ready=False, model="qwen3:4b", action_required=None,
        )

    async def unexpected_generation(*args, **kwargs):
        pytest.fail("unavailable model must not generate per-video failures")

    monkeypatch.setattr(ollama_runtime, "probe_ollama", probe)
    monkeypatch.setattr(NoteGenerationService, "run_batch", unexpected_generation)
    engine = create_engine("sqlite://")
    try:
        code = runner._run_batches(
            Namespace(only_batch=None), Settings(storage_root=tmp_path),
            SimpleNamespace(engine=engine), [("a",), ("b",)], set(), audit,
        )
    finally:
        engine.dispose()
    assert code == 4
    assert probe_calls == ["http://127.0.0.1:11434"]
    assert not audit.exists()
    assert not audit.with_suffix(".checkpoints.jsonl").exists()
