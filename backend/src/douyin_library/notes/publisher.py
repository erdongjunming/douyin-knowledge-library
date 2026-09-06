from __future__ import annotations

import inspect
import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from douyin_library.notes.generator import GENERATOR_VERSION, NoteBundle

GENERATED_FILES = (
    "learning_note.md",
    "douyin_ai_brief.md",
    "douyin_ai_brief.json",
    "note_budget.json",
    "note_score.json",
    "page_metadata.json",
    "coverage_report.json",
    "generation_audit.json",
    "evidence_map.json",
    "assets/asset_manifest.json",
)
_MANIFEST = "generation_manifest.json"
_ALL_GENERATED_FILES = (*GENERATED_FILES, _MANIFEST)
_PUBLISH_LOCKS: dict[str, threading.RLock] = {}
_PUBLISH_LOCKS_GUARD = threading.Lock()
_PUBLISH_LOCK_STATE = threading.local()


class PublishError(RuntimeError):
    """Raised when a note bundle cannot be validated or published."""


@dataclass(frozen=True, slots=True)
class PublishResult:
    aweme_id: str
    status: str
    output_dir: str
    archive_dir: str | None = None
    error: str | None = None


class NoteBundlePublisher:
    def __init__(
        self,
        root: Path,
        *,
        replace: Callable[[Path, Path], Any] | None = None,
        ownership_check: Callable[..., bool] | None = None,
        generator_version: str = GENERATOR_VERSION,
    ) -> None:
        self.root = Path(root)
        self.replace = replace or _replace
        self.ownership_check = ownership_check or (lambda: True)
        self.generator_version = generator_version

    def publish(
        self,
        aweme_id: str,
        candidate: NoteBundle | Mapping[str, Any],
        *,
        force: bool = False,
        ownership_check: Callable[[], bool] | None = None,
        publish_guard: Callable[[], bool] | None = None,
    ) -> PublishResult:
        note_dir = self.root / "dy-note" / str(aweme_id)
        with note_publication_lock(self.root, aweme_id):
            note_dir.mkdir(parents=True, exist_ok=True)
            staging = note_dir / ".staging" / str(uuid.uuid4())
            rollback = staging / ".previous"
            archive_dir: Path | None = None
            archive_complete = False
            rollback_complete = False
            try:
                outputs = _candidate_outputs(candidate, self.generator_version)
                if "evidence_map.json" not in outputs:
                    outputs["evidence_map.json"] = json.dumps(
                        {"schema": "douyin-note-evidence-map-v1", "items": []},
                        ensure_ascii=False,
                    )
                self._validate_outputs(outputs)
                current_exists = (note_dir / "learning_note.md").is_file()
                if (
                    current_exists
                    and _is_fallback(outputs)
                    and not _current_is_fallback(note_dir)
                ):
                    return PublishResult(
                        str(aweme_id),
                        "preserved_stronger_existing",
                        str(note_dir),
                    )
                if current_exists and not force:
                    return PublishResult(
                        str(aweme_id), "preserved_existing", str(note_dir)
                    )
                self._check_publish_guard(publish_guard)
                # The lock covers the guard through rollback/commit, so no
                # publisher for this work can replace the guarded snapshot.
                _snapshot_generated(note_dir, rollback)
                rollback_complete = True
                _stage_outputs(staging, outputs)
                if current_exists:
                    archive_dir = self._next_archive_path(note_dir)
                    self._archive_current(note_dir, archive_dir)
                    archive_complete = True
                self._replace_outputs(staging, note_dir, outputs)
                self._check_ownership(aweme_id, ownership_check)
                self.replace(staging / _MANIFEST, note_dir / _MANIFEST)
                _fsync_directory(note_dir)
                shutil.rmtree(staging, ignore_errors=True)
                return PublishResult(
                    str(aweme_id),
                    "published",
                    str(note_dir),
                    str(archive_dir) if archive_dir else None,
                )
            except Exception as exc:
                if archive_complete and archive_dir is not None:
                    try:
                        self._restore_archive(note_dir, archive_dir)
                    except Exception as restore_exc:
                        if rollback_complete:
                            try:
                                _restore_snapshot(note_dir, rollback)
                            except Exception as snapshot_exc:
                                restore_exc = (
                                    f"{restore_exc}; snapshot restore failed: "
                                    f"{snapshot_exc}"
                                )
                        exc = PublishError(f"{exc}; restore failed: {restore_exc}")
                elif rollback_complete:
                    try:
                        _restore_snapshot(note_dir, rollback)
                    except Exception as restore_exc:
                        exc = PublishError(f"{exc}; restore failed: {restore_exc}")
                shutil.rmtree(staging, ignore_errors=True)
                self._record_failure(note_dir, aweme_id, exc)
                if isinstance(exc, PublishError):
                    raise
                raise PublishError(str(exc)) from exc


    def _validate_outputs(self, outputs: Mapping[str, str]) -> None:
        extras = sorted(set(outputs).difference(_ALL_GENERATED_FILES))
        if extras:
            raise PublishError(f"unexpected output path: {', '.join(extras)}")
        missing = [name for name in _ALL_GENERATED_FILES if name not in outputs]
        if missing:
            raise PublishError(f"required output missing: {', '.join(missing)}")
        for name in _ALL_GENERATED_FILES:
            content = outputs[name]
            if name.endswith(".json"):
                try:
                    value = json.loads(content)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise PublishError(f"invalid JSON output: {name}") from exc
                if not isinstance(value, dict):
                    raise PublishError(f"JSON output must be an object: {name}")
            elif not content.strip():
                raise PublishError(f"required output is empty: {name}")
        manifest = _json_object(outputs[_MANIFEST], _MANIFEST)
        for key in ("provider", "model", "quality_gate", "generator_version"):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                raise PublishError(f"required manifest section missing: {key}")
        for filename, keys in {
            "douyin_ai_brief.json": (("title", str), ("quality_gate", str)),
            "note_budget.json": (("schema", str),),
            "note_score.json": (("quality_gate", str),),
            "page_metadata.json": (("provider", str), ("model", str), ("quality_gate", str)),
            "coverage_report.json": (("quality_gate", str),),
            "generation_audit.json": (("status", str),),
            "assets/asset_manifest.json": (("copies_original_assets", bool),),
        }.items():
            value = _json_object(outputs[filename], filename)
            if any(not isinstance(value.get(key), expected) for key, expected in keys):
                raise PublishError(f"required output section missing: {filename}")

    def _next_archive_path(self, note_dir: Path) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive = note_dir / "versions" / f"{timestamp}-{self.generator_version}"
        suffix = 1
        while archive.exists():
            archive = note_dir / "versions" / f"{timestamp}-{self.generator_version}-{suffix}"
            suffix += 1
        return archive

    def _archive_current(self, note_dir: Path, archive: Path) -> None:
        archive.mkdir(parents=True, exist_ok=False)
        for relative in _ALL_GENERATED_FILES:
            source = note_dir / relative
            if source.is_file():
                target = archive / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                _fsync_file(target)
        _fsync_directory(archive)

    def _replace_outputs(
        self,
        staging: Path,
        note_dir: Path,
        outputs: Mapping[str, str],
    ) -> None:
        for relative in GENERATED_FILES:
            self.replace(staging / relative, note_dir / relative)

    def _restore_archive(self, note_dir: Path, archive: Path) -> None:
        for relative in _ALL_GENERATED_FILES:
            source = archive / relative
            target = note_dir / relative
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif target.is_file():
                target.unlink()
        _fsync_directory(note_dir)

    def _check_ownership(
        self,
        aweme_id: str,
        ownership_check: Callable[[], bool] | None = None,
    ) -> None:
        callback = ownership_check or self.ownership_check
        try:
            owns = callback()
        except TypeError:
            owns = callback(aweme_id)
        if not owns:
            raise PublishError("ownership fencing failed before manifest commit")

    def _check_publish_guard(self, publish_guard: Callable[[], bool] | None) -> None:
        if publish_guard is None:
            return
        if not publish_guard():
            raise PublishError("publish guard failed before replacement")

    def _record_failure(self, note_dir: Path, aweme_id: str, error: Exception) -> None:
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            path = note_dir / "attempts" / timestamp / "generation_failure.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema": "douyin-note-publish-failure-v1",
                        "aweme_id": str(aweme_id),
                        "error": str(error),
                        "recorded_at": datetime.now(UTC).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _fsync_file(path)
        except OSError:
            pass


def publish_with_guard(
    publisher: Any,
    root: Path,
    aweme_id: str,
    candidate: NoteBundle | Mapping[str, Any],
    *,
    force: bool = False,
    ownership_check: Callable[[], bool] | None = None,
    publish_guard: Callable[[], bool] | None = None,
) -> PublishResult:
    """Publish through legacy custom publishers without weakening the guard."""

    kwargs: dict[str, Any] = {
        "force": force,
        "ownership_check": ownership_check,
    }
    if publish_guard is None or _accepts_keyword(publisher.publish, "publish_guard"):
        if publish_guard is not None:
            kwargs["publish_guard"] = publish_guard
        return publisher.publish(aweme_id, candidate, **kwargs)

    with note_publication_lock(root, aweme_id):
        if not publish_guard():
            raise PublishError("publish guard failed before replacement")
        return publisher.publish(aweme_id, candidate, **kwargs)


def _accepts_keyword(callback: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


@contextmanager
def note_publication_lock(root: Path, aweme_id: str):
    """Serialize one work's publication across threads and worker processes."""

    lock_path = Path(root) / "dy-note" / str(aweme_id) / ".publish.lock"
    key = str(lock_path.resolve())
    with _PUBLISH_LOCKS_GUARD:
        thread_lock = _PUBLISH_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        depths = getattr(_PUBLISH_LOCK_STATE, "depths", None)
        if depths is None:
            depths = {}
            _PUBLISH_LOCK_STATE.depths = depths
        depth = depths.get(key, 0)
        depths[key] = depth + 1
        if depth:
            try:
                yield
            finally:
                depths[key] -= 1
            return

        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as handle:
                _ensure_lock_byte(handle)
                _lock_file(handle)
                try:
                    yield
                finally:
                    _unlock_file(handle)
        finally:
            depths.pop(key, None)


def _ensure_lock_byte(handle: Any) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)


def _lock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _candidate_outputs(
    candidate: NoteBundle | Mapping[str, Any],
    generator_version: str,
) -> dict[str, str]:
    if isinstance(candidate, NoteBundle):
        values: dict[str, Any] = {
            "learning_note.md": candidate.learning_note,
            "douyin_ai_brief.md": candidate.brief_markdown,
            "douyin_ai_brief.json": candidate.brief_json,
            "note_budget.json": candidate.note_budget,
            "note_score.json": candidate.note_score,
            "page_metadata.json": candidate.page_metadata,
            "coverage_report.json": candidate.coverage_report,
            "evidence_map.json": candidate.parsed.get(
                "evidence_map",
                {"schema": "douyin-note-evidence-map-v1", "items": []},
            ),
            "generation_audit.json": {"status": "succeeded"},
            "assets/asset_manifest.json": candidate.asset_manifest,
        }
        metadata = candidate.page_metadata
        values[_MANIFEST] = {
            "schema": "douyin-note-generation-manifest-v1",
            "provider": metadata.get("provider", "unknown"),
            "model": metadata.get("model", "unknown"),
            "quality_gate": metadata.get("quality_gate", "failed"),
            "semantic_complete": bool(metadata.get("semantic_complete", False)),
            "input_sufficient": bool(metadata.get("input_sufficient", False)),
            "generator_version": generator_version,
            "generated_files": list(_ALL_GENERATED_FILES),
        }
    else:
        values = dict(candidate)
    result: dict[str, str] = {}
    for name, value in values.items():
        if isinstance(value, bytes):
            result[name] = value.decode("utf-8")
        elif isinstance(value, str):
            result[name] = value
        else:
            result[name] = json.dumps(value, ensure_ascii=False, indent=2)
    return result


def _is_fallback(outputs: Mapping[str, str]) -> bool:
    for name in (_MANIFEST, "page_metadata.json", "note_score.json"):
        value = json.loads(outputs[name])
        if value.get("quality_gate") == "fallback":
            return True
    return False


def _current_is_fallback(note_dir: Path) -> bool:
    """Return whether the current bundle is already a deterministic fallback.

    A forced retry may refresh an existing fallback, but a fallback must never
    downgrade a semantic bundle.  Malformed or missing metadata is treated as
    stronger/unknown and therefore remains protected.
    """
    for name in (_MANIFEST, "page_metadata.json", "note_score.json"):
        path = note_dir / name
        if not path.is_file():
            return False
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(value, dict) or value.get("quality_gate") != "fallback":
            return False
    return True


def _stage_outputs(staging: Path, outputs: Mapping[str, str]) -> None:
    for relative, content in outputs.items():
        target = staging / relative
        try:
            target.resolve().relative_to(staging.resolve())
        except ValueError as exc:
            raise PublishError(f"invalid staged output path: {relative}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _fsync_file(target)
    _fsync_directory(staging)


def _snapshot_generated(note_dir: Path, snapshot: Path) -> None:
    """Copy the current generated bundle into a private rollback directory."""

    snapshot.mkdir(parents=True, exist_ok=True)
    for relative in _ALL_GENERATED_FILES:
        source = note_dir / relative
        if not source.is_file():
            continue
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        _fsync_file(target)
    _fsync_directory(snapshot)


def _restore_snapshot(note_dir: Path, snapshot: Path) -> None:
    """Restore a pre-publish snapshot and remove files absent from it."""

    for relative in _ALL_GENERATED_FILES:
        source = snapshot / relative
        target = note_dir / relative
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif target.is_file():
            target.unlink()
    _fsync_directory(note_dir)


def _json_object(content: str, name: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise PublishError(f"JSON output must be an object: {name}")
    return value


def _replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "NoteBundlePublisher",
    "PublishError",
    "PublishResult",
    "note_publication_lock",
    "publish_with_guard",
]
