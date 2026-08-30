from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path

import pytest

from douyin_library.notes import publisher as publisher_module
from douyin_library.notes.publisher import NoteBundlePublisher, PublishError


def _hold_publication_lock(
    root: str,
    aweme_id: str,
    entered: object,
    release: object,
) -> None:
    with publisher_module.note_publication_lock(Path(root), aweme_id):
        entered.set()
        release.wait(5)


def _semantic_candidate(title: str) -> dict[str, object]:
    return {
        "learning_note.md": f"# {title}\n",
        "douyin_ai_brief.md": "# Brief\n",
        "douyin_ai_brief.json": {"title": title, "quality_gate": "passed"},
        "note_budget.json": {"schema": "douyin-note-budget-v2"},
        "note_score.json": {"quality_gate": "passed", "score": 100},
        "page_metadata.json": {
            "provider": "ollama",
            "model": "qwen3:4b",
            "quality_gate": "passed",
        },
        "coverage_report.json": {"quality_gate": "passed"},
        "generation_audit.json": {"status": "succeeded"},
        "assets/asset_manifest.json": {"copies_original_assets": False},
        "generation_manifest.json": {
            "provider": "ollama",
            "model": "qwen3:4b",
            "quality_gate": "passed",
            "generator_version": "semantic-v2",
        },
    }


def test_publish_lock_closes_guard_to_replacement_race(tmp_path: Path) -> None:
    note_dir = tmp_path / "dy-note" / "734"
    note_dir.mkdir(parents=True)
    (note_dir / "learning_note.md").write_text("# original\n", encoding="utf-8")
    (note_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                "provider": "legacy",
                "model": "legacy",
                "quality_gate": "legacy",
                "generator_version": "legacy",
            }
        ),
        encoding="utf-8",
    )
    first_replacing = threading.Event()
    release_first = threading.Event()
    second_replacing = threading.Event()
    errors: list[BaseException] = []

    def delayed_replace(source: Path, target: Path) -> None:
        if not first_replacing.is_set():
            first_replacing.set()
            assert release_first.wait(5)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)

    def observed_replace(source: Path, target: Path) -> None:
        second_replacing.set()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)

    def run_first() -> None:
        try:
            NoteBundlePublisher(tmp_path, replace=delayed_replace).publish(
                "734", _semantic_candidate("first"), force=True, publish_guard=lambda: True
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def run_second() -> None:
        try:
            NoteBundlePublisher(tmp_path, replace=observed_replace).publish(
                "734", _semantic_candidate("second"), force=True
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=run_first)
    first.start()
    assert first_replacing.wait(5)
    second = threading.Thread(target=run_second)
    second.start()
    second_entered_before_release = second_replacing.wait(0.25)
    release_first.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert second_entered_before_release is False
    assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == "# second\n"


def test_custom_publisher_without_guard_keyword_is_guarded_by_adapter(tmp_path: Path) -> None:
    calls: list[str] = []

    class CustomPublisher:
        def publish(
            self,
            aweme_id: str,
            candidate: object,
            *,
            force: bool = False,
            ownership_check: object = None,
        ) -> object:
            calls.append(aweme_id)
            return type("Result", (), {"status": "published"})()

    with pytest.raises(PublishError, match="publish guard"):
        publisher_module.publish_with_guard(
            CustomPublisher(),
            tmp_path,
            "735",
            _semantic_candidate("guarded"),
            force=True,
            ownership_check=lambda: True,
            publish_guard=lambda: False,
        )
    assert calls == []

    result = publisher_module.publish_with_guard(
        CustomPublisher(),
        tmp_path,
        "735",
        _semantic_candidate("guarded"),
        force=True,
        ownership_check=lambda: True,
        publish_guard=lambda: True,
    )
    assert result.status == "published"
    assert calls == ["735"]


def test_publish_lock_serializes_independent_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    first_entered = context.Event()
    second_entered = context.Event()
    release_first = context.Event()
    release_second = context.Event()
    first = context.Process(
        target=_hold_publication_lock,
        args=(str(tmp_path), "736", first_entered, release_first),
    )
    second = context.Process(
        target=_hold_publication_lock,
        args=(str(tmp_path), "736", second_entered, release_second),
    )
    first.start()
    try:
        assert first_entered.wait(5)
        second.start()
        assert second_entered.wait(0.25) is False
        release_first.set()
        assert second_entered.wait(5)
        release_second.set()
    finally:
        release_first.set()
        release_second.set()
        first.join(5)
        second.join(5)
        if first.is_alive():
            first.terminate()
        if second.is_alive():
            second.terminate()
    assert first.exitcode == 0
    assert second.exitcode == 0
