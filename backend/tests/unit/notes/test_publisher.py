from __future__ import annotations

import json
from pathlib import Path

import pytest

from douyin_library.notes.publisher import NoteBundlePublisher, PublishError

GENERATED = (
    "learning_note.md",
    "douyin_ai_brief.md",
    "douyin_ai_brief.json",
    "note_budget.json",
    "note_score.json",
    "page_metadata.json",
    "coverage_report.json",
    "generation_audit.json",
    "assets/asset_manifest.json",
)


def semantic_candidate() -> dict[str, object]:
    outputs: dict[str, object] = {
        "learning_note.md": "# New note\n\n## 一句话总览\n\ncontent\n",
        "douyin_ai_brief.md": "# Brief\n",
        "douyin_ai_brief.json": {"title": "Brief", "quality_gate": "passed"},
        "note_budget.json": {"schema": "douyin-note-budget-v2"},
        "note_score.json": {"quality_gate": "passed", "score": 100},
        "page_metadata.json": {
            "provider": "ollama",
            "model": "qwen3:8b",
            "quality_gate": "passed",
            "generator_version": "semantic-v2",
        },
        "coverage_report.json": {"schema": "douyin-note-coverage-v1", "quality_gate": "passed"},
        "generation_audit.json": {"status": "succeeded"},
        "assets/asset_manifest.json": {"copies_original_assets": False},
        "generation_manifest.json": {
            "provider": "ollama",
            "model": "qwen3:8b",
            "quality_gate": "passed",
            "generator_version": "semantic-v2",
        },
    }
    return outputs


def fallback_candidate() -> dict[str, object]:
    candidate = semantic_candidate()
    for name in ("page_metadata.json", "generation_manifest.json", "note_score.json"):
        value = candidate[name]
        assert isinstance(value, dict)
        value["provider"] = "rule"
        value["model"] = "local-rule-v1"
        value["quality_gate"] = "fallback"
    return candidate


def seed_current_bundle(tmp_path: Path, learning_note: str = "# previous") -> Path:
    note_dir = tmp_path / "dy-note" / "731"
    note_dir.mkdir(parents=True)
    (note_dir / "learning_note.md").write_text(learning_note, encoding="utf-8")
    (note_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                "provider": "ollama",
                "model": "qwen3:8b",
                "quality_gate": "passed",
                "generator_version": "semantic-v1",
            }
        ),
        encoding="utf-8",
    )
    return note_dir


def test_first_publish_writes_validated_bundle(tmp_path: Path) -> None:
    result = NoteBundlePublisher(tmp_path).publish("731", semantic_candidate())

    assert result.status == "published"
    note_dir = tmp_path / "dy-note" / "731"
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8").startswith("# New")
    assert (note_dir / "generation_manifest.json").is_file()
    assert not (note_dir / "versions").exists()


def test_replace_archives_previous_generated_files_without_source_assets(tmp_path: Path) -> None:
    note_dir = seed_current_bundle(tmp_path)
    (note_dir / "source.mp4").write_bytes(b"source")

    result = NoteBundlePublisher(tmp_path).publish("731", semantic_candidate(), force=True)

    assert result.status == "published"
    archives = list((note_dir / "versions").iterdir())
    assert len(archives) == 1
    archive = archives[0]
    assert (archive / "learning_note.md").read_text(encoding="utf-8") == "# previous"
    assert not (archive / "source.mp4").exists()


def test_invalid_candidate_is_rejected_before_touching_current_bundle(tmp_path: Path) -> None:
    note_dir = seed_current_bundle(tmp_path)
    candidate = semantic_candidate()
    del candidate["coverage_report.json"]

    with pytest.raises(PublishError, match="required"):
        NoteBundlePublisher(tmp_path).publish("731", candidate, force=True)

    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# previous"
    assert not (note_dir / "versions").exists()
    assert list((note_dir / "attempts").glob("*/generation_failure.json"))


def test_candidate_rejects_paths_outside_generated_allowlist(tmp_path: Path) -> None:
    candidate = semantic_candidate()
    candidate["../../escape.txt"] = "must not be written"

    with pytest.raises(PublishError, match="unexpected output path"):
        NoteBundlePublisher(tmp_path).publish("731", candidate)

    assert not (tmp_path / "escape.txt").exists()


def test_manifest_is_replaced_last_and_ownership_is_checked_before_it(tmp_path: Path) -> None:
    order: list[str] = []

    def replace(source: Path, target: Path) -> None:
        order.append(target.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)

    def owns_generation() -> bool:
        order.append("ownership")
        return True

    NoteBundlePublisher(tmp_path, replace=replace, ownership_check=owns_generation).publish(
        "731", semantic_candidate()
    )

    assert order[-2:] == ["ownership", "generation_manifest.json"]


def test_failed_replace_restores_previous_bundle(tmp_path: Path) -> None:
    note_dir = seed_current_bundle(tmp_path)
    calls = 0

    def fail_on_third_replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated replace failure")
        source.replace(target)

    publisher = NoteBundlePublisher(tmp_path, replace=fail_on_third_replace)
    with pytest.raises(PublishError):
        publisher.publish("731", semantic_candidate(), force=True)

    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# previous"
    assert not list((note_dir / ".staging").glob("*"))
    assert list((note_dir / "attempts").glob("*/generation_failure.json"))


def test_semantic_publish_requires_current_ownership(tmp_path: Path) -> None:
    note_dir = tmp_path / "dy-note" / "731"
    note_dir.mkdir(parents=True)
    (note_dir / "generation_manifest.json").write_text('{"previous": true}', encoding="utf-8")
    (note_dir / "source.mp4").write_bytes(b"source")

    with pytest.raises(PublishError, match="ownership"):
        NoteBundlePublisher(tmp_path, ownership_check=lambda: False).publish(
            "731", semantic_candidate()
        )

    assert (note_dir / "generation_manifest.json").read_text(
        encoding="utf-8"
    ) == '{"previous": true}'
    assert (note_dir / "source.mp4").read_bytes() == b"source"
    assert not list((note_dir / ".staging").glob("*"))


def test_first_publish_ownership_failure_leaves_no_partial_bundle(tmp_path: Path) -> None:
    with pytest.raises(PublishError, match="ownership"):
        NoteBundlePublisher(tmp_path, ownership_check=lambda: False).publish(
            "731", semantic_candidate()
        )

    note_dir = tmp_path / "dy-note" / "731"
    assert all(not (note_dir / relative).exists() for relative in GENERATED)
    assert not list((note_dir / ".staging").glob("*"))
    assert list((note_dir / "attempts").glob("*/generation_failure.json"))


@pytest.mark.parametrize(
    "filename, value",
    [
        ("douyin_ai_brief.json", {}),
        ("note_budget.json", {}),
        ("note_score.json", {}),
        ("generation_audit.json", {}),
        ("assets/asset_manifest.json", {}),
    ],
)
def test_candidate_requires_sections_for_every_json_artifact(
    tmp_path: Path,
    filename: str,
    value: object,
) -> None:
    candidate = semantic_candidate()
    candidate[filename] = value

    with pytest.raises(PublishError, match="required output section"):
        NoteBundlePublisher(tmp_path).publish("731", candidate)


def test_fallback_never_replaces_existing_note(tmp_path: Path) -> None:
    note_dir = seed_current_bundle(tmp_path)

    result = NoteBundlePublisher(tmp_path).publish("731", fallback_candidate(), force=True)

    assert result.status == "preserved_stronger_existing"
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# previous"


def test_publish_guard_rechecks_protected_state_before_replacement(tmp_path: Path) -> None:
    note_dir = seed_current_bundle(tmp_path)
    guard_calls = 0

    def guard() -> bool:
        nonlocal guard_calls
        guard_calls += 1
        return False

    with pytest.raises(PublishError, match="publish guard"):
        NoteBundlePublisher(tmp_path).publish(
            "731", semantic_candidate(), force=True, publish_guard=guard
        )

    assert guard_calls == 1
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# previous"
    assert not (note_dir / "versions").exists()


class SimulatedProcessLoss(BaseException):
    pass


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("interrupt_at", ["learning_note.md", "generation_manifest.json"])
def test_interrupted_publish_recovers_one_consistent_bundle_on_read(
    tmp_path: Path, existing: bool, interrupt_at: str,
) -> None:
    from douyin_library.notes.state import classify_note_bundle

    candidate = semantic_candidate()
    candidate["generation_manifest.json"]["semantic_complete"] = True
    candidate["coverage_report.json"].update({
        "semantic_complete": True, "depth_passed": True, "input_sufficient": True,
    })
    publisher = NoteBundlePublisher(tmp_path)
    if existing:
        publisher.publish("731", candidate)
    note_dir = tmp_path / "dy-note" / "731"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "source.mp4").write_bytes(b"immutable-source")
    candidate["learning_note.md"] = "# replacement"

    def interrupt(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        if target.name == interrupt_at:
            raise SimulatedProcessLoss()

    with pytest.raises(SimulatedProcessLoss):
        NoteBundlePublisher(tmp_path, replace=interrupt).publish("731", candidate, force=True)

    state = classify_note_bundle(note_dir)
    if interrupt_at == "generation_manifest.json":
        assert state.quality_status == "semantic_complete"
        assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# replacement"
    elif existing:
        assert state.quality_status == "semantic_complete"
        assert (note_dir / "learning_note.md").read_text(encoding="utf-8").startswith("# New")
    else:
        assert state.quality_status == "missing"
        assert not (note_dir / "learning_note.md").exists()
    assert (note_dir / "source.mp4").read_bytes() == b"immutable-source"
    assert not (note_dir / ".publication.json").exists()
    assert not list((note_dir / ".staging").glob("*"))


def test_hash_mismatch_cannot_report_semantic_completeness(tmp_path: Path) -> None:
    from douyin_library.notes.state import classify_note_bundle

    candidate = semantic_candidate()
    candidate["generation_manifest.json"]["semantic_complete"] = True
    candidate["coverage_report.json"].update({
        "semantic_complete": True, "depth_passed": True, "input_sufficient": True,
    })
    NoteBundlePublisher(tmp_path).publish("731", candidate)
    note_dir = tmp_path / "dy-note" / "731"
    assert classify_note_bundle(note_dir).quality_status == "semantic_complete"
    (note_dir / "learning_note.md").write_text("# tampered", encoding="utf-8")
    assert classify_note_bundle(note_dir).quality_status == "unknown"


def test_recovery_rejects_journal_missing_a_snapshot_hash(tmp_path: Path) -> None:
    from douyin_library.notes.state import classify_note_bundle

    candidate = semantic_candidate()
    candidate["generation_manifest.json"]["semantic_complete"] = True
    candidate["coverage_report.json"].update({
        "semantic_complete": True, "depth_passed": True, "input_sufficient": True,
    })
    publisher = NoteBundlePublisher(tmp_path)
    publisher.publish("731", candidate)
    note_dir = tmp_path / "dy-note" / "731"

    def interrupt_after_first_output(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        if target.name == "learning_note.md":
            raise SimulatedProcessLoss()

    candidate["learning_note.md"] = "# replacement"
    with pytest.raises(SimulatedProcessLoss):
        NoteBundlePublisher(
            tmp_path, replace=interrupt_after_first_output,
        ).publish("731", candidate, force=True)

    journal_path = note_dir / ".publication.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    snapshot = note_dir / ".staging" / journal["staging"] / ".previous"
    del journal["previous_hashes"]["learning_note.md"]
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    (snapshot / "learning_note.md").write_text("# corrupted backup", encoding="utf-8")

    state = classify_note_bundle(note_dir)

    assert state.quality_status == "unknown"
    assert journal_path.is_file()
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# replacement"


@pytest.mark.parametrize("existing", [False, True])
def test_cleanup_fsync_failure_preserves_a_committed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    import douyin_library.notes.publisher as publication
    from douyin_library.notes.state import classify_note_bundle

    candidate = semantic_candidate()
    candidate["generation_manifest.json"]["semantic_complete"] = True
    candidate["coverage_report.json"].update({
        "semantic_complete": True, "depth_passed": True, "input_sufficient": True,
    })
    publisher = NoteBundlePublisher(tmp_path)
    if existing:
        publisher.publish("731", candidate)
    note_dir = tmp_path / "dy-note" / "731"
    candidate["learning_note.md"] = "# committed replacement"
    original_sync = publication._fsync_directory

    def fail_after_journal_unlink(path: Path) -> None:
        if path == note_dir and not (note_dir / ".publication.json").exists():
            raise OSError("cleanup directory sync failed")
        original_sync(path)

    monkeypatch.setattr(publication, "_fsync_directory", fail_after_journal_unlink)

    result = publisher.publish("731", candidate, force=True)

    assert result.status == "published"
    assert classify_note_bundle(note_dir).quality_status == "semantic_complete"
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == (
        "# committed replacement"
    )
    assert not (note_dir / ".publication.json").exists()
    assert list((note_dir / ".staging").glob("*/.previous"))


def test_committed_recovery_cleanup_failure_keeps_bundle_readable_and_retriable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from douyin_library.notes.state import classify_note_bundle

    candidate = semantic_candidate()
    candidate["generation_manifest.json"]["semantic_complete"] = True
    candidate["coverage_report.json"].update({
        "semantic_complete": True, "depth_passed": True, "input_sufficient": True,
    })

    def interrupt_after_commit(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        if target.name == "generation_manifest.json":
            raise SimulatedProcessLoss()

    with pytest.raises(SimulatedProcessLoss):
        NoteBundlePublisher(tmp_path, replace=interrupt_after_commit).publish("731", candidate)
    note_dir = tmp_path / "dy-note" / "731"
    original_unlink = Path.unlink

    def fail_journal_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == note_dir / ".publication.json":
            raise PermissionError("journal is temporarily locked")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as cleanup_failure:
        cleanup_failure.setattr(Path, "unlink", fail_journal_unlink)
        assert classify_note_bundle(note_dir).quality_status == "semantic_complete"
        assert (note_dir / ".publication.json").is_file()
        assert list((note_dir / ".staging").glob("*/.previous"))

    assert classify_note_bundle(note_dir).quality_status == "semantic_complete"
    assert not (note_dir / ".publication.json").exists()
    assert not list((note_dir / ".staging").glob("*"))


def test_rollback_cleanup_failure_preserves_original_error_and_restored_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from douyin_library.notes.state import classify_note_bundle

    candidate = semantic_candidate()
    candidate["generation_manifest.json"]["semantic_complete"] = True
    candidate["coverage_report.json"].update({
        "semantic_complete": True, "depth_passed": True, "input_sufficient": True,
    })
    publisher = NoteBundlePublisher(tmp_path)
    publisher.publish("731", candidate)
    note_dir = tmp_path / "dy-note" / "731"
    original_unlink = Path.unlink

    def fail_journal_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == note_dir / ".publication.json":
            raise PermissionError("journal is temporarily locked")
        original_unlink(path, *args, **kwargs)

    candidate["learning_note.md"] = "# rejected replacement"
    monkeypatch.setattr(Path, "unlink", fail_journal_unlink)
    with pytest.raises(PublishError, match="ownership fencing failed"):
        publisher.publish("731", candidate, force=True, ownership_check=lambda: False)

    assert classify_note_bundle(note_dir).quality_status == "semantic_complete"
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8").startswith("# New")
    assert (note_dir / ".publication.json").is_file()


def test_commit_hash_failure_rolls_back_before_cleanup(tmp_path: Path) -> None:
    note_dir = seed_current_bundle(tmp_path)

    def corrupt_note_on_replace(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        if target.name == "learning_note.md":
            target.write_text("# incomplete replacement", encoding="utf-8")

    with pytest.raises(PublishError, match="published bundle hash mismatch"):
        NoteBundlePublisher(tmp_path, replace=corrupt_note_on_replace).publish(
            "731", semantic_candidate(), force=True,
        )

    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# previous"
    assert not (note_dir / ".publication.json").exists()
