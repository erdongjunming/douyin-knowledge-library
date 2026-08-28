from __future__ import annotations

import json
from pathlib import Path

import pytest

from douyin_library.notes.state import classify_note_bundle


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _note_dir(tmp_path: Path) -> Path:
    (tmp_path / "learning_note.md").write_text("# note", encoding="utf-8")
    return tmp_path


def test_missing_learning_note_is_missing_even_with_metadata(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "generation_manifest.json",
        {"provider": "ollama", "model": "qwen3:8b", "quality_gate": "passed"},
    )

    state = classify_note_bundle(tmp_path)

    assert state.quality_status == "missing"
    assert state.legacy_eligible is False


def test_unknown_bundle_is_not_eligible_for_legacy_upgrade(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "unknown"
    assert state.legacy_eligible is False


def test_passed_ollama_manifest_is_semantic_complete(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    _write_json(
        note_dir / "generation_manifest.json",
        {
            "generation": {
                "provider": "ollama",
                "model": "qwen3:8b",
                "quality_gate": "passed",
                "generator_version": "semantic-v2",
            }
        },
    )
    coverage = {"schema": "douyin-note-coverage-v1", "quality_gate": "passed"}
    _write_json(note_dir / "coverage_report.json", coverage)

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "semantic_complete"
    assert state.provider == "ollama"
    assert state.model == "qwen3:8b"
    assert state.quality_gate == "passed"
    assert state.generator_version == "semantic-v2"
    assert state.legacy_eligible is False
    assert state.coverage == coverage


@pytest.mark.parametrize(
    "metadata",
    [
        {"provider": "external", "model": "semantic-model", "quality_gate": "fallback"},
        {"provider": "rule", "model": "complete-timeline-v1", "quality_gate": "failed"},
    ],
)
def test_fallback_bundle_is_classified(tmp_path: Path, metadata: dict[str, str]) -> None:
    note_dir = _note_dir(tmp_path)
    _write_json(note_dir / "generation_manifest.json", metadata)

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "fallback"
    assert state.legacy_eligible is False


def test_explicit_fallback_gate_overrides_legacy_generator_identity(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    _write_json(
        note_dir / "generation_manifest.json",
        {
            "provider": "rule",
            "model": "local-rule-v1",
            "quality_gate": "fallback",
        },
    )

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "fallback"
    assert state.legacy_eligible is False


@pytest.mark.parametrize(
    "metadata",
    [
        {"model": "local-rule-v1"},
        {"model_name": "local-rule-v1"},
        {"generator": "local-rule-v1"},
        {"processor": "local-rule-v1"},
        {"generation": {"model": "local-rule-v1"}},
    ],
)
def test_only_explicit_local_rule_v1_is_legacy(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    note_dir = _note_dir(tmp_path)
    _write_json(note_dir / "generation_audit.json", metadata)

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "legacy"
    assert state.legacy_eligible is True


def test_manifest_has_priority_over_audit_and_budget(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    _write_json(
        note_dir / "generation_manifest.json",
        {"provider": "external", "model": "semantic-v1", "quality_gate": "passed"},
    )
    _write_json(note_dir / "generation_audit.json", {"model": "local-rule-v1"})
    _write_json(note_dir / "note_budget.json", {"model": "local-rule-v1"})

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "semantic_complete"
    assert state.provider == "external"


def test_malformed_higher_priority_json_falls_through(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    (note_dir / "generation_manifest.json").write_text("{broken", encoding="utf-8")
    (note_dir / "generation_audit.json").write_text("[not-a-dict]", encoding="utf-8")
    _write_json(note_dir / "note_budget.json", {"model": "local-rule-v1"})

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "legacy"
    assert state.legacy_eligible is True


def test_valid_unknown_manifest_does_not_borrow_lower_priority_identity(
    tmp_path: Path,
) -> None:
    note_dir = _note_dir(tmp_path)
    _write_json(note_dir / "generation_manifest.json", {"status": "complete"})
    _write_json(note_dir / "generation_audit.json", {"model": "local-rule-v1"})

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "unknown"
    assert state.legacy_eligible is False


@pytest.mark.parametrize("bad_coverage", ["{broken", "[]"])
def test_malformed_or_non_object_coverage_is_ignored(
    tmp_path: Path,
    bad_coverage: str,
) -> None:
    note_dir = _note_dir(tmp_path)
    _write_json(note_dir / "generation_audit.json", {"model": "local-rule-v1"})
    (note_dir / "coverage_report.json").write_text(bad_coverage, encoding="utf-8")

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "legacy"
    assert state.coverage is None


def test_all_malformed_metadata_results_in_unknown(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    for name in ("generation_manifest.json", "generation_audit.json", "note_budget.json"):
        (note_dir / name).write_text("{broken", encoding="utf-8")

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "unknown"
    assert state.legacy_eligible is False


def test_deeply_nested_metadata_is_ignored_without_raising(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    nested = '{"generation":' * 100_000 + "{}" + "}" * 100_000
    (note_dir / "generation_manifest.json").write_text(nested, encoding="utf-8")

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "unknown"
    assert state.legacy_eligible is False


def test_oversized_metadata_is_ignored(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    oversized = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "quality_gate": "passed",
        "padding": "x" * (4 * 1024 * 1024),
    }
    _write_json(note_dir / "generation_manifest.json", oversized)

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "unknown"
    assert state.legacy_eligible is False


def test_metadata_with_overlong_integer_is_ignored_without_raising(tmp_path: Path) -> None:
    note_dir = _note_dir(tmp_path)
    manifest = (
        '{"provider":"ollama","model":"qwen3:8b","quality_gate":"passed","value":'
        + "9" * 5_000
        + "}"
    )
    (note_dir / "generation_manifest.json").write_text(manifest, encoding="utf-8")

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "unknown"
    assert state.legacy_eligible is False
