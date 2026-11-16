import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.audit import AuditPayloadStore
from douyin_library.db import create_schema
from douyin_library.maintenance import create_database_backup
from douyin_library.models import ProcessingAudit


def test_database_backup_is_consistent_and_rate_limited(tmp_path: Path) -> None:
    database = tmp_path / "library.db"
    connection = sqlite3.connect(database)
    connection.execute("create table sample (value text not null)")
    connection.execute("insert into sample values ('knowledge')")
    connection.commit()
    connection.close()

    first = create_database_backup(
        database,
        backup_root=tmp_path / "backups",
        min_interval=timedelta(hours=24),
        retention=2,
    )
    second = create_database_backup(
        database,
        backup_root=tmp_path / "backups",
        min_interval=timedelta(hours=24),
        retention=2,
    )

    assert first is not None
    assert second is None
    backup = Path(first["backup"])
    assert backup.is_file()
    assert len(first["sha256"]) == 64
    manifest = json.loads(backup.with_suffix(".manifest.json").read_text())
    assert manifest["sqlite_quick_check"] == "ok"
    restored = sqlite3.connect(backup)
    try:
        assert restored.execute("select value from sample").fetchone()[0] == "knowledge"
    finally:
        restored.close()


def test_database_backup_includes_verified_external_audit_payloads(
    tmp_path: Path,
) -> None:
    database = tmp_path / "library.db"
    audit_root = tmp_path / "audit"
    engine = create_engine(f"sqlite:///{database}")
    create_schema(engine)
    try:
        with Session(engine) as session:
            audit = ProcessingAudit(
                operation="note-generation",
                processor="ollama",
                request_json="{}",
                status="succeeded",
            )
            session.add(audit)
            session.flush()
            reference = AuditPayloadStore(audit_root, 1).store(
                audit,
                field="response",
                value=json.dumps({"raw": "full response"}, ensure_ascii=False),
                suffix="json",
            )
            audit.response_json = reference
            session.commit()

        result = create_database_backup(
            database,
            backup_root=tmp_path / "backups",
            audit_root=audit_root,
            min_interval=timedelta(0),
            retention=2,
        )
    finally:
        engine.dispose()

    assert result is not None
    assert result["audit_payloads_complete"] is True
    assert result["audit_payload_count"] == 1
    assert result["audit_payload_bytes"] > 0
    assert result["audit_backup"]
    manifest_path = Path(result["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["audit_root"] == str(audit_root.resolve())
    assert manifest["audit_payloads_complete"] is True
    payload = manifest["audit_payloads"][0]
    expected_relative = Path(json.loads(reference)["path"]).resolve().relative_to(
        audit_root.resolve()
    ).as_posix()
    assert payload["source_relative_path"] == expected_relative
    copied = Path(result["audit_backup"]) / payload["backup_relative_path"]
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == json.dumps(
        {"raw": "full response"}, ensure_ascii=False
    )
    assert copied.stat().st_size == payload["bytes"]
    assert len(payload["sha256"]) == 64


def test_database_backup_rejects_corrupt_payload_and_preserves_previous_backup(
    tmp_path: Path,
) -> None:
    database = tmp_path / "library.db"
    audit_root = tmp_path / "audit"
    engine = create_engine(f"sqlite:///{database}")
    create_schema(engine)
    try:
        with Session(engine) as session:
            audit = ProcessingAudit(
                operation="note-generation",
                processor="ollama",
                request_json="{}",
                status="succeeded",
            )
            session.add(audit)
            session.flush()
            reference = AuditPayloadStore(audit_root, 1).store(
                audit,
                field="response",
                value="trusted response",
                suffix="txt",
            )
            audit.response_json = reference
            session.commit()

        backup_root = tmp_path / "backups"
        first = create_database_backup(
            database,
            backup_root=backup_root,
            audit_root=audit_root,
            min_interval=timedelta(0),
            retention=2,
        )
        assert first is not None
        first_db = Path(first["backup"])
        first_manifest = Path(first["manifest"])
        source_payload = Path(json.loads(reference)["path"])
        source_payload.write_text("tampered response", encoding="utf-8")

        with pytest.raises(ValueError, match="audit_payload_(bytes|sha256)_mismatch"):
            create_database_backup(
                database,
                backup_root=backup_root,
                audit_root=audit_root,
                min_interval=timedelta(0),
                retention=2,
            )
    finally:
        engine.dispose()

    assert first_db.is_file()
    assert first_manifest.is_file()
    assert len(list(backup_root.glob("library.*.db"))) == 1
    assert len(list(backup_root.glob("library.*.manifest.json"))) == 1
    assert len(list(backup_root.glob("library.*.audit"))) == 1
    assert not list(backup_root.glob(".*.tmp"))


def test_database_backup_retention_removes_audit_snapshot_with_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "library.db"
    audit_root = tmp_path / "audit"
    engine = create_engine(f"sqlite:///{database}")
    create_schema(engine)
    try:
        with Session(engine) as session:
            audit = ProcessingAudit(
                operation="note-generation",
                processor="rule",
                request_json="{}",
                status="succeeded",
            )
            session.add(audit)
            session.flush()
            reference = AuditPayloadStore(audit_root, 1).store(
                audit,
                field="response",
                value="payload",
                suffix="txt",
            )
            audit.response_json = reference
            session.commit()

        backup_root = tmp_path / "backups"
        first = create_database_backup(
            database,
            backup_root=backup_root,
            audit_root=audit_root,
            min_interval=timedelta(0),
            retention=1,
        )
        second = create_database_backup(
            database,
            backup_root=backup_root,
            audit_root=audit_root,
            min_interval=timedelta(0),
            retention=1,
        )
    finally:
        engine.dispose()

    assert first is not None and second is not None
    assert not Path(first["backup"]).exists()
    assert not Path(first["manifest"]).exists()
    assert not Path(first["audit_backup"]).exists()
    assert Path(second["backup"]).exists()
    assert Path(second["manifest"]).exists()
    assert Path(second["audit_backup"]).exists()


def test_interrupted_database_artifact_does_not_rate_limit_next_backup(
    tmp_path: Path,
) -> None:
    database = tmp_path / "library.db"
    connection = sqlite3.connect(database)
    connection.execute("create table sample (value text not null)")
    connection.commit()
    connection.close()

    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    orphan = backup_root / "library.20260101-000000.db"
    orphan.write_bytes(database.read_bytes())

    result = create_database_backup(
        database,
        backup_root=backup_root,
        min_interval=timedelta(hours=24),
        retention=2,
    )

    assert result is not None
