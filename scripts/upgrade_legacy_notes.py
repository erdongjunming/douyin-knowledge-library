"""Run the guarded local semantic upgrade over a fixed historical inventory.

The normal CLI intentionally caps legacy upgrades at three works per request.
This runner keeps that safety boundary while making the complete historical
operation resumable and auditable.  It never enables ``force`` and never uses
the rule or external providers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPO_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


@dataclass(frozen=True, slots=True)
class BatchRecord:
    batch_index: int
    aweme_ids: tuple[str, ...]
    status: str
    result: dict[str, Any]
    source_hashes_before: dict[str, dict[str, Any]]
    source_hashes_after: dict[str, dict[str, Any]]
    recorded_at: str


def chunked(values: list[str], size: int = 3) -> list[tuple[str, ...]]:
    if size != 3:
        raise ValueError("legacy_upgrade_batch_size_must_be_three")
    return [tuple(values[index : index + size]) for index in range(0, len(values), size)]


def discover_legacy_ids(storage_root: Path) -> list[str]:
    """Return a deterministic snapshot of upgrade-eligible note directories."""

    from douyin_library.notes.state import classify_note_bundle

    transcript_root = Path(storage_root) / "transcript"
    note_root = Path(storage_root) / "dy-note"
    result: list[str] = []
    for transcript in sorted(transcript_root.glob("*/transcript.txt")):
        aweme_id = transcript.parent.name
        state = classify_note_bundle(note_root / aweme_id)
        if state.upgrade_eligible:
            result.append(aweme_id)
    return result


def _sha256(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "bytes": None, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"exists": True, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_hashes(storage_root: Path, aweme_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    root = Path(storage_root)
    names = (
        root / "audio" / "{id}.m4a",
        root / "cover" / "{id}.jpg",
        root / "metadata" / "{id}.json",
        root / "transcript" / "{id}" / "transcript.txt",
        root / "transcript" / "{id}" / "segments.json",
        root / "transcript" / "{id}" / "qwen3_asr.json",
        root / "transcript" / "{id}" / "transcript.cleaned.md",
        root / "transcript" / "{id}" / "transcript.cleaned.json",
    )
    return {
        str(path): _sha256(path)
        for aweme_id in aweme_ids
        for template in names
        for path in (Path(str(template).format(id=aweme_id)),)
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip():
            try:
                value = json.loads(line)
            except (ValueError, UnicodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise
                # Only an unterminated final write is recoverable. Preserve
                # the complete original journal before replacing its tail.
                suffix = uuid.uuid4().hex
                backup = path.with_name(f"{path.name}.interrupted-{suffix}")
                temporary = path.with_name(f"{path.name}.recovery-{suffix}")
                with backup.open("xb") as stream:
                    stream.write(original)
                    stream.flush()
                    os.fsync(stream.fileno())
                with temporary.open("xb") as stream:
                    stream.write(b"".join(lines[:index]))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                print(json.dumps({"event": "journal_tail_recovered", "backup": str(backup)}),
                      file=sys.stderr, flush=True)
                break
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _read_jsonl(path)
    with path.open("a+b") as stream:
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def completed_batches(rows: list[dict[str, Any]], batches: list[tuple[str, ...]]) -> set[int]:
    latest = {tuple(row.get("aweme_ids", [])): row for row in rows}
    return {
        index
        for index, batch in enumerate(batches)
        if (row := latest.get(batch))
        and is_terminal_batch(row)
    }


def is_terminal_batch(row: dict[str, Any]) -> bool:
    """Return whether a batch is safe to omit on a resumed run.

    A normal success requires every item to be semantically complete.  The
    only non-success terminal state is a local-input refusal: the service must
    have preserved the existing note and all source assets, and every item in
    the result must either be a successful generation or the exact
    ``legacy_upgrade_input_insufficient`` refusal.  Provider, publication,
    and other semantic failures remain retryable.
    """

    if row.get("source_unchanged") is not True:
        return False
    if row.get("status") == "succeeded":
        return True
    if row.get("status") != "completed_with_failures":
        return False
    verification = row.get("verification")
    if not isinstance(verification, dict) or verification.get("old_notes_preserved") is not True:
        return False
    result = row.get("result")
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list) or not items:
        return False
    return all(
        isinstance(item, dict)
        and (
            item.get("status") == "generated"
            or (
                item.get("status") == "preserved_existing"
                and item.get("error") == "legacy_upgrade_input_insufficient"
            )
        )
        for item in items
    ) and any(
        isinstance(item, dict)
        and item.get("status") == "preserved_existing"
        and item.get("error") == "legacy_upgrade_input_insufficient"
        for item in items
    )


def validate_local_settings(settings: Any) -> None:
    from douyin_library.notes.providers.base import parse_base_url

    parsed = parse_base_url(settings.note_ollama_url)
    if (
        parsed is None
        or not parsed.is_loopback
        or parsed.scheme not in {"http", "https"}
        or parsed.has_credentials
        or parsed.has_query
        or parsed.has_fragment
    ):
        raise ValueError("legacy_upgrade_requires_loopback_ollama")
    if settings.note_ollama_model != "qwen3:4b":
        raise ValueError("legacy_upgrade_requires_qwen3_4b")


def note_hashes(root: Path, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    return {aweme_id: _sha256(root / "dy-note" / aweme_id / "learning_note.md") for aweme_id in ids}


def checkpoint_note_baseline(
    rows: Iterable[dict[str, Any]], batch_index: int, batch: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Return the first complete old-note snapshot recorded for a batch.

    An interrupted run can have already published one or more new root notes
    before it reaches the batch audit append.  On resume, comparing against a
    fresh root snapshot would lose the original preservation baseline.  The
    first durable ``batch_started`` checkpoint is the authoritative baseline.
    """

    for row in rows:
        if (
            row.get("event") != "batch_started"
            or row.get("batch_index") != batch_index
            or tuple(row.get("aweme_ids", [])) != batch
        ):
            continue
        old_notes = row.get("old_notes_before")
        if (
            isinstance(old_notes, dict)
            and all(isinstance(old_notes.get(aweme_id), dict) for aweme_id in batch)
        ):
            return {aweme_id: old_notes[aweme_id] for aweme_id in batch}
    return {}


def verify_batch(root: Path, ids: Iterable[str], before: dict[str, Any]) -> dict[str, Any]:
    from douyin_library.media.storage import StorageLayout
    from douyin_library.notes.input import NoteInputLoader
    from douyin_library.notes.pipeline import _input_is_sufficient
    from douyin_library.notes.publisher import (
        GENERATED_FILES,
        NoteBundlePublisher,
        PublishError,
    )
    from douyin_library.notes.state import classify_note_bundle

    layout = StorageLayout(root)
    required = {*GENERATED_FILES, "generation_manifest.json"}
    validator = NoteBundlePublisher(root)
    items = []
    for aweme_id in ids:
        note_dir = root / "dy-note" / aweme_id
        state = classify_note_bundle(note_dir)
        old = before.get(aweme_id, {})
        current = _sha256(note_dir / "learning_note.md")
        archive = None
        preserved = not old.get("exists") or current == old
        if not preserved:
            archive = next(
                (
                    str(p)
                    for p in (note_dir / "versions").glob("*/learning_note.md")
                    if _sha256(p) == old
                ),
                None,
            )
            preserved = archive is not None
        manifest = {}
        validation_error = None
        try:
            outputs = {
                name: (note_dir / name).read_text(encoding="utf-8-sig")
                for name in sorted(required)
            }
            validator._validate_outputs(outputs)
            manifest = json.loads(outputs["generation_manifest.json"])
            listed = manifest.get("generated_files")
            if (
                not isinstance(listed, list)
                or not all(isinstance(name, str) for name in listed)
                or len(listed) != len(set(listed))
                or set(listed) not in (set(GENERATED_FILES), required)
            ):
                raise PublishError("invalid manifest generated_files")
            parsed = {
                name: json.loads(content)
                for name, content in outputs.items() if name.endswith(".json")
            }
            if parsed["generation_audit.json"].get("status") != "succeeded":
                raise PublishError("generation audit did not succeed")
            for name in ("page_metadata.json", "note_score.json", "douyin_ai_brief.json"):
                if parsed[name].get("quality_gate") != "passed":
                    raise PublishError(f"quality gate did not pass: {name}")
            for key in ("provider", "model"):
                if parsed["page_metadata.json"].get(key) != manifest.get(key):
                    raise PublishError(f"inconsistent page metadata: {key}")
        except (OSError, UnicodeError, ValueError, RecursionError, PublishError) as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
        complete_files = validation_error is None
        identity_ok = manifest.get("provider") == "ollama" and manifest.get("model") == "qwen3:4b"
        try:
            input_sufficient = _input_is_sufficient(
                NoteInputLoader(layout).load(aweme_id)
            )
        except (OSError, UnicodeError, TypeError, ValueError):
            input_sufficient = False
        quality_ok = (
            manifest.get("quality_gate") == "passed"
            and manifest.get("semantic_complete") is True
            and input_sufficient
        )
        semantic_status = (
            "semantic_complete"
            if state.quality_status == "semantic_complete"
            and complete_files
            and identity_ok
            and quality_ok
            else "unknown"
            if state.quality_status == "semantic_complete"
            else state.quality_status
        )
        try:
            readable = bool((note_dir / "learning_note.md").read_text(encoding="utf-8-sig").strip())
        except (OSError, UnicodeError):
            readable = False
        items.append(
            {
                "aweme_id": aweme_id,
                "quality_status": semantic_status,
                "provider": state.provider,
                "model": state.model,
                "input_sufficient": input_sufficient,
                "readable": readable,
                "old_note_preserved": preserved,
                "archive": archive,
                "note_hash": current,
                "validation_error": validation_error,
            }
        )
    return {
        "semantic_complete": all(
            item["quality_status"] == "semantic_complete" and item["readable"] for item in items
        ),
        "old_notes_preserved": all(item["old_note_preserved"] for item in items),
        "items": items,
    }


@contextmanager
def runner_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("legacy_upgrade_already_running") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


async def monitored_batch(service: Any, batch: tuple[str, ...], index: int) -> Any:
    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(30)
            try:
                owned = service.lease_store.for_owner(service.worker_id, None)
                phase = "generating" if owned else "waiting_gpu"
            except Exception:  # noqa: BLE001
                phase = "status_unavailable"
            print(
                json.dumps(
                    {
                        "event": "heartbeat",
                        "batch_index": index,
                        "pid": os.getpid(),
                        "worker_id": service.worker_id,
                        "phase": phase,
                        "at": datetime.now(UTC).isoformat(),
                    }
                ),
                flush=True,
            )

    task = asyncio.create_task(heartbeat())
    try:
        return await service.run_batch(
            aweme_ids=set(batch), provider_policy="local", legacy_only=True, limit=3
        )
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def run(args: argparse.Namespace) -> int:
    from douyin_library.config import Settings

    settings = Settings()
    validate_local_settings(settings)
    with runner_lock(Path(settings.storage_root) / "audit" / "legacy-upgrade.lock"):
        return _run_locked(args, settings)


def _run_locked(args: argparse.Namespace, settings: Any) -> int:
    from douyin_library.notes.ollama_runtime import (
        OllamaProcessManager,
        ensure_ollama_ready,
    )
    from douyin_library.runtime import initialize_runtime

    storage_root = Path(settings.storage_root)
    audit_path = Path(args.audit or (storage_root / "audit" / "legacy-upgrade-batches.jsonl"))
    inventory_path = Path(
        args.inventory or (storage_root / "audit" / "legacy-upgrade-inventory.json")
    )

    if args.refresh_inventory or not inventory_path.is_file():
        ids = discover_legacy_ids(storage_root)
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(
            json.dumps(
                {
                    "schema": "douyin-legacy-upgrade-inventory-v1",
                    "created_at": datetime.now(UTC).isoformat(),
                    "aweme_ids": ids,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        ids = [str(value) for value in payload.get("aweme_ids", [])]

    batches = chunked(ids)
    # A recorded failure is evidence for retry, not a completion marker.  Only
    # an all-item successful batch may be skipped on a resumed run.
    audit_rows = _read_jsonl(audit_path)
    completed = completed_batches(audit_rows, batches)
    print(
        json.dumps(
            {
                "inventory": str(inventory_path),
                "audit": str(audit_path),
                "candidates": len(ids),
                "batches": len(batches),
                "already_recorded": len(completed),
            },
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return 0

    if getattr(args, "enqueue_only", False):
        # Historical invocations share the durable queue with the API and
        # worker pools.  This command only enables scheduling and enqueues one
        # bounded batch; generation remains owned by the fenced worker path.
        from douyin_library.jobs.legacy_upgrade import refresh_legacy_upgrade
        from douyin_library.jobs.queue import JobQueue
        from douyin_library.runtime import initialize_runtime
        from sqlalchemy.orm import Session

        runtime = initialize_runtime(settings)
        try:
            with Session(runtime.engine) as session:
                queue = JobQueue(session)
                result = refresh_legacy_upgrade(
                    queue,
                    settings,
                    limit=3,
                    enable=True,
                )
            print(json.dumps({"event": "legacy_jobs_enqueued", **result}, ensure_ascii=False))
            return 0
        finally:
            runtime.engine.dispose()

    ollama_manager = OllamaProcessManager(settings)
    try:
        readiness = asyncio.run(
            ensure_ollama_ready(settings, manager=ollama_manager),
        )
        if not readiness.ready:
            print(
                json.dumps(
                    {
                        "event": "provider_unavailable",
                        "provider": "ollama",
                        "model": readiness.model,
                        "service_status": readiness.service_status,
                        "model_available": readiness.model_available,
                        "action_required": readiness.action_required,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 4
        runtime = initialize_runtime(settings)
        try:
            return _run_batches(
                args,
                settings,
                runtime,
                batches,
                completed,
                audit_path,
                ollama_manager=ollama_manager,
            )
        finally:
            runtime.engine.dispose()
    finally:
        ollama_manager.stop_if_started()


def _run_batches(
    args,
    settings,
    runtime,
    batches,
    completed,
    audit_path,
    *,
    ollama_manager=None,
) -> int:
    from douyin_library.media.storage import StorageLayout
    from douyin_library.notes.ollama_runtime import ensure_ollama_ready, probe_ollama
    from douyin_library.notes.service import NoteGenerationService
    from sqlalchemy.orm import Session

    storage_root = Path(settings.storage_root)
    stop_file = storage_root / "audit" / "legacy-upgrade.stop"
    audit_rows = _read_jsonl(audit_path)
    checkpoint_path = audit_path.with_suffix(".checkpoints.jsonl")
    checkpoint_rows = _read_jsonl(checkpoint_path)
    failed = False
    try:
        layout = StorageLayout(storage_root)
        with Session(runtime.engine) as session:
            for batch_index, batch in enumerate(batches):
                if args.only_batch is not None and batch_index != args.only_batch:
                    continue
                if stop_file.exists():
                    print(
                        json.dumps({"event": "stopped", "next_batch": batch_index}),
                        flush=True,
                    )
                    return 2
                latest = {
                    tuple(row.get("aweme_ids", [])): row for row in audit_rows
                }.get(batch)
                baseline = checkpoint_note_baseline(checkpoint_rows, batch_index, batch)
                if latest is not None or baseline:
                    if not baseline and latest is not None:
                        baseline = latest.get("old_notes_before", {})
                    verification = verify_batch(storage_root, batch, baseline)
                    current_sources = source_hashes(storage_root, batch)
                    sources_match = (
                        latest is not None
                        and current_sources == latest.get("source_hashes_after")
                    )
                    original_sources = next(
                        (row.get("source_hashes_before") for row in checkpoint_rows
                         if row.get("event") == "batch_started"
                         and row.get("batch_index") == batch_index
                         and tuple(row.get("aweme_ids", [])) == batch),
                        None,
                    )
                    if original_sources is not None:
                        sources_match = current_sources == original_sources and (
                            latest is None or sources_match
                        )
                    require_complete = batch_index in completed or any(
                        tuple(row.get("aweme_ids", [])) == batch and is_terminal_batch(row)
                        for row in audit_rows
                    )
                    # A mixed terminal batch can contain both complete notes
                    # and input refusals. Recheck each current output; a prior
                    # refusal must not conceal damage to another generated note.
                    refused_ids = {
                        item.get("aweme_id")
                        for item in (latest or {}).get("result", {}).get("items", [])
                        if item.get("status") == "preserved_existing"
                        and item.get("error") == "legacy_upgrade_input_insufficient"
                    }
                    outputs_valid = all(
                        item["readable"] and (
                            item["quality_status"] == "semantic_complete"
                            or (
                                not require_complete
                                and item["note_hash"] == baseline.get(item["aweme_id"])
                            )
                            or (
                                item["aweme_id"] in refused_ids
                                and item["input_sufficient"] is False
                                and item["note_hash"] == baseline.get(item["aweme_id"])
                            )
                        ) for item in verification["items"]
                    )
                    if (sources_match and all(aweme_id in baseline for aweme_id in batch)
                            and verification["old_notes_preserved"] and outputs_valid):
                        if require_complete:
                            continue
                    else:
                        failed = True
                        _append_jsonl(audit_path, {
                            "batch_index": batch_index, "aweme_ids": batch,
                            "status": "integrity_failed", "source_unchanged": sources_match,
                            "source_hashes_before": (
                                original_sources or (latest or {}).get("source_hashes_after")
                            ),
                            "source_hashes_after": current_sources, "old_notes_before": baseline,
                            "verification": verification,
                            "result": {"error": "legacy_upgrade_resume_verification_failed"},
                            "recorded_at": datetime.now(UTC).isoformat(),
                        })
                        print(json.dumps({
                            "event": "resume_verification_failed", "batch_index": batch_index,
                        }), flush=True)
                        return 3
                readiness = asyncio.run(
                    ensure_ollama_ready(settings, manager=ollama_manager)
                    if ollama_manager is not None
                    else probe_ollama(settings)
                )
                if not readiness.ready:
                    failed = True
                    print(json.dumps({
                        "event": "provider_unavailable", "next_batch": batch_index,
                        "provider": "ollama", "model": readiness.model,
                        "service_status": readiness.service_status,
                        "model_available": readiness.model_available,
                    }), flush=True)
                    return 4
                before = source_hashes(storage_root, batch)
                old_notes = checkpoint_note_baseline(checkpoint_rows, batch_index, batch)
                if not old_notes:
                    old_notes = note_hashes(storage_root, batch)
                checkpoint = {
                    "event": "batch_started",
                    "batch_index": batch_index,
                    "aweme_ids": batch,
                    "source_hashes_before": before,
                    "old_notes_before": old_notes,
                    "pid": os.getpid(),
                    "at": datetime.now(UTC).isoformat(),
                }
                _append_jsonl(
                    checkpoint_path,
                    checkpoint,
                )
                checkpoint_rows.append(checkpoint)
                print(
                    json.dumps(
                        {
                            "event": "batch_started",
                            "batch_index": batch_index,
                            "aweme_ids": batch,
                        }
                    ),
                    flush=True,
                )
                try:
                    service = NoteGenerationService(
                        session,
                        layout,
                        settings,
                        worker_id=f"legacy-upgrade-{os.getpid()}",
                    )
                    result = asyncio.run(monitored_batch(service, batch, batch_index)).as_dict()
                    status = (
                        "succeeded" if result.get("failed", 0) == 0 else "completed_with_failures"
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {"error": f"{type(exc).__name__}:{exc}"}
                    status = "failed"
                after = source_hashes(storage_root, batch)
                source_unchanged = before == after
                verification = verify_batch(storage_root, batch, old_notes)
                if not source_unchanged or not verification["old_notes_preserved"]:
                    status = "integrity_failed"
                elif not verification["semantic_complete"] and status == "succeeded":
                    status = "completed_with_failures"
                terminal_preserved_low_information = (
                    status == "completed_with_failures" and is_terminal_batch({
                        "status": status, "result": result,
                        "source_unchanged": source_unchanged, "verification": verification,
                    })
                )
                failed = failed or (
                    status != "succeeded" and not terminal_preserved_low_information
                )
                record = BatchRecord(
                    batch_index,
                    batch,
                    status,
                    result,
                    before,
                    after,
                    datetime.now(UTC).isoformat(),
                )
                _append_jsonl(
                    audit_path,
                    {
                        **asdict(record),
                        "source_unchanged": source_unchanged,
                        "old_notes_before": old_notes,
                        "verification": verification,
                        "terminal_preserved_low_information": terminal_preserved_low_information,
                    },
                )
                print(
                    json.dumps(
                        {
                            "batch_index": batch_index,
                            "aweme_ids": list(batch),
                            "status": status,
                            "source_unchanged": source_unchanged,
                            "terminal_preserved_low_information": (
                                terminal_preserved_low_information
                            ),
                            "result": result,
                        },
                        ensure_ascii=False,
                    )
                )
                if status == "integrity_failed":
                    return 3
                if args.only_batch is not None:
                    break
    finally:
        print(json.dumps({"event": "run_finished", "has_failures": failed}), flush=True)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--refresh-inventory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--enqueue-only",
        action="store_true",
        help="enable the durable legacy lane and enqueue one bounded batch",
    )
    parser.add_argument(
        "--direct",
        dest="enqueue_only",
        action="store_false",
        help="run the legacy batch directly (maintenance-only escape hatch)",
    )
    parser.set_defaults(enqueue_only=True)
    parser.add_argument("--only-batch", type=int)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
