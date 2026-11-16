from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from douyin_library.audit import AuditPayloadStore, is_external_payload

_AUDIT_PAYLOAD_COLUMNS = (
    "request_json",
    "response_json",
    "stdout",
    "stderr",
)


def create_database_backup(
    database_path: Path,
    *,
    backup_root: Path,
    min_interval: timedelta,
    retention: int,
    audit_root: Path | None = None,
) -> dict[str, Any] | None:
    database_path = Path(database_path).resolve()
    backup_root = Path(backup_root).resolve()
    resolved_audit_root = (
        Path(audit_root).resolve() if audit_root is not None else None
    )
    if not database_path.is_file() or database_path.stat().st_size == 0:
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    if min_interval < timedelta(0):
        raise ValueError("min_interval_must_be_nonnegative")
    if retention < 1:
        raise ValueError("retention_must_be_at_least_one")

    # The lock is deliberately acquired before checking the interval.  A
    # second process therefore rechecks the newest backup after waiting for a
    # concurrent creator and does not produce duplicate snapshots.
    with _backup_lock(backup_root):
        existing = _list_verified_backups(backup_root)
        now = datetime.now(UTC)
        if existing and min_interval > timedelta(0):
            latest = datetime.fromtimestamp(existing[0].stat().st_mtime, tz=UTC)
            if now - latest < min_interval:
                return None

        stamp = now.strftime("%Y%m%d-%H%M%S")
        target = backup_root / f"library.{stamp}.db"
        audit_target = (
            target.with_suffix(".audit") if resolved_audit_root is not None else None
        )
        if (
            target.exists()
            or target.with_suffix(".manifest.json").exists()
            or (audit_target is not None and audit_target.exists())
        ):
            unique_stamp = f"{stamp}.{uuid.uuid4().hex[:8]}"
            target = backup_root / f"library.{unique_stamp}.db"
            audit_target = (
                target.with_suffix(".audit")
                if resolved_audit_root is not None
                else None
            )
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        audit_temporary = (
            backup_root / f".{audit_target.name}.{uuid.uuid4().hex}.tmp"
            if audit_target is not None
            else None
        )
        manifest_path = target.with_suffix(".manifest.json")
        try:
            quick_check = _create_sqlite_snapshot(database_path, temporary)
            audit_metadata: dict[str, Any] = {}
            if resolved_audit_root is not None and audit_temporary is not None:
                audit_metadata = _create_audit_snapshot(
                    temporary,
                    source_root=resolved_audit_root,
                    destination_root=audit_temporary,
                )
                # The snapshot is staged under a private temporary name.  The
                # manifest must point at the final published tree so it can be
                # used directly for recovery after this function returns.
                audit_metadata["audit_backup"] = str(audit_target)
                for payload in audit_metadata["audit_payloads"]:
                    payload["backup_path"] = str(
                        audit_target / Path(payload["backup_relative_path"])
                    )

            # Publish the database and its external payload tree only after
            # every source and copied payload has passed integrity checks.
            os.replace(temporary, target)
            if audit_target is not None and audit_temporary is not None:
                os.replace(audit_temporary, audit_target)

            manifest = {
                "schema": (
                    "douyin-library-database-backup-v2"
                    if resolved_audit_root is not None
                    else "douyin-library-database-backup-v1"
                ),
                "source": str(database_path),
                "backup": str(target),
                "manifest": str(manifest_path),
                "bytes": target.stat().st_size,
                "sha256": _file_sha256(target),
                "created_at": now.isoformat(),
                "sqlite_quick_check": quick_check,
            }
            manifest.update(audit_metadata)
            _write_json_atomic(manifest_path, manifest)
        except Exception:
            # A snapshot without a manifest is not considered verified.  The
            # target names are unique, so cleanup cannot remove an older
            # backup.  Existing backups remain untouched on every failure.
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            if audit_temporary is not None:
                _remove_tree(audit_temporary)
            if audit_target is not None:
                _remove_tree(audit_target)
            raise
        finally:
            temporary.unlink(missing_ok=True)
            if audit_temporary is not None:
                _remove_tree(audit_temporary)
        _prune_backups(backup_root, retention=retention)
        return manifest


def _create_sqlite_snapshot(source_path: Path, destination_path: Path) -> str:
    """Create and validate an online SQLite snapshot at a temporary path."""

    source = sqlite3.connect(str(source_path), timeout=60)
    destination: sqlite3.Connection | None = None
    try:
        destination = sqlite3.connect(str(destination_path), timeout=60)
        source.backup(destination, pages=4096, sleep=0.01)
        check = destination.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError(f"database_backup_quick_check_failed:{check}")
        return str(check[0])
    finally:
        if destination is not None:
            destination.close()
        source.close()


def _create_audit_snapshot(
    database_snapshot: Path,
    *,
    source_root: Path,
    destination_root: Path,
) -> dict[str, Any]:
    """Copy every external audit payload referenced by a DB snapshot.

    The database snapshot is the source of truth for the reference set.  This
    avoids publishing a payload that belongs to a row which was committed only
    after the SQLite snapshot was taken, and makes the DB/audit pair recoverable
    as one atomic backup unit.
    """

    records = _collect_audit_payloads(database_snapshot, source_root)
    destination_root.mkdir(parents=True, exist_ok=False)
    payloads: list[dict[str, Any]] = []
    total_bytes = 0
    for record in records:
        relative = Path(record["source_relative_path"])
        destination = destination_root.joinpath(*relative.parts)
        _copy_verified_payload(
            Path(record["source_path"]),
            destination,
            expected_bytes=record["bytes"],
            expected_sha256=record["sha256"],
        )
        item = {
            "source_path": record["source_path"],
            "source_relative_path": record["source_relative_path"],
            "backup_relative_path": record["source_relative_path"],
            "backup_path": str(destination),
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "references": record["references"],
        }
        # Keep the singular fields convenient for callers that have one
        # reference, while retaining all references for deduplicated files.
        if len(record["references"]) == 1:
            item["audit_id"] = record["references"][0]["audit_id"]
            item["field"] = record["references"][0]["field"]
        payloads.append(item)
        total_bytes += int(record["bytes"])

    return {
        "audit_root": str(source_root),
        "audit_restore_root": str(source_root),
        "audit_backup": str(destination_root.with_name(destination_root.name)),
        "audit_payload_count": len(payloads),
        "audit_reference_count": sum(len(item["references"]) for item in payloads),
        "audit_payload_bytes": total_bytes,
        "audit_payloads_complete": True,
        "audit_payloads": payloads,
        "audit_errors": [],
        "audit_sha256": _audit_manifest_digest(payloads),
    }


def _collect_audit_payloads(
    database_snapshot: Path,
    source_root: Path,
) -> list[dict[str, Any]]:
    """Read and validate external payload envelopes from a DB snapshot."""

    connection = sqlite3.connect(str(database_snapshot), timeout=60)
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("processing_audit",),
        ).fetchone()
        if table is None:
            return []
        available = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(processing_audit)")
        }
        columns = tuple(column for column in _AUDIT_PAYLOAD_COLUMNS if column in available)
        if not columns:
            return []
        selected = ", ".join(("id", *columns))
        rows = connection.execute(
            f"SELECT {selected} FROM processing_audit ORDER BY id"
        ).fetchall()
    finally:
        connection.close()

    store = AuditPayloadStore(source_root, 0)
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        audit_id = int(row[0])
        for index, column in enumerate(columns, start=1):
            value = row[index]
            if not isinstance(value, str) or not is_external_payload(value):
                continue
            try:
                envelope = json.loads(value)
                # AuditPayloadStore.load performs path, byte-count, digest, and
                # UTF-8 validation.  Read the content here before copying so a
                # malformed source can never produce a successful manifest.
                store.load(value)
                source_path = Path(str(envelope["path"])).expanduser().resolve(
                    strict=True
                )
                relative = source_path.relative_to(source_root).as_posix()
                expected_bytes = int(envelope["bytes"])
                expected_sha256 = str(envelope["sha256"]).lower()
            except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
                raise ValueError(
                    f"audit_payload_validation_failed:{audit_id}:{column}:{exc}"
                ) from exc
            key = (relative, expected_bytes, expected_sha256)
            record = records.get(key)
            if record is None:
                record = {
                    "source_path": str(source_path),
                    "source_relative_path": relative,
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                    "references": [],
                }
                records[key] = record
            record["references"].append(
                {
                    "audit_id": audit_id,
                    "field": column,
                }
            )
    return list(records.values())


def _copy_verified_payload(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    """Copy one payload through a temporary file and verify its digest."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            try:
                os.fsync(target_handle.fileno())
            except OSError:
                # Some mounted/network filesystems do not expose fsync; the
                # digest check still protects the published copy.
                pass
        actual_bytes = temporary.stat().st_size
        actual_sha256 = _file_sha256(temporary)
        if actual_bytes != expected_bytes:
            raise ValueError("audit_backup_payload_bytes_mismatch")
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError("audit_backup_payload_sha256_mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_manifest_digest(payloads: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for payload in sorted(payloads, key=lambda item: item["backup_relative_path"]):
        digest.update(str(payload["backup_relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(payload["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(payload["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@contextmanager
def _backup_lock(backup_root: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """Serialize backup creation using a tiny SQLite lock database.

    A dedicated database keeps the lock portable across Windows and POSIX and
    avoids relying on advisory filesystem-lock semantics.  The transaction is
    held only for the backup operation; callers never sleep while holding a
    separate application/database transaction.
    """

    lock_path = backup_root / ".maintenance-backup.lock.sqlite"
    connection = sqlite3.connect(str(lock_path), timeout=timeout)
    try:
        connection.execute(f"PRAGMA busy_timeout={max(1, int(timeout * 1000))}")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS maintenance_lock "
            "(name TEXT PRIMARY KEY NOT NULL)"
        )
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT OR IGNORE INTO maintenance_lock(name) VALUES (?)",
            ("maintenance:backup",),
        )
        yield
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def vacuum_database(database_path: Path) -> None:
    connection = sqlite3.connect(str(database_path.resolve()), timeout=60)
    try:
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("VACUUM")
    finally:
        connection.close()


def _prune_backups(backup_root: Path, *, retention: int) -> None:
    backups = _list_verified_backups(backup_root)
    for path in backups[retention:]:
        path.unlink(missing_ok=True)
        path.with_suffix(".manifest.json").unlink(missing_ok=True)
        _remove_tree(path.with_suffix(".audit"))


def _list_verified_backups(backup_root: Path) -> list[Path]:
    """Return published database snapshots, excluding interrupted artifacts."""

    return sorted(
        (
            path
            for path in backup_root.glob("library.*.db")
            if path.is_file() and path.with_suffix(".manifest.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _remove_tree(path: Path) -> None:
    """Remove a generated file/directory without following symlinks."""

    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
