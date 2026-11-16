import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.audit import (
    AuditPayloadStore,
    ProcessingAuditRecorder,
    compact_large_audit_payloads,
    compact_large_audit_responses,
)
from douyin_library.db import create_schema
from douyin_library.models import Job, ProcessingAudit, Work


def test_processing_audit_keeps_full_io_but_redacts_credentials(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    input_file = tmp_path / "audio.m4a"
    input_file.write_bytes(b"audio-for-hash")

    with Session(engine) as session:
        work = Work(aweme_id="731")
        session.add(work)
        session.flush()
        job = Job(kind="transcribe_work", work_id=work.id)
        session.add(job)
        session.commit()

        recorder = ProcessingAuditRecorder(session)
        audit = recorder.start(
            operation="transcribe_work",
            processor="qwen3-asr",
            model="Qwen/Qwen3-ASR-0.6B",
            version="1",
            job=job,
            request={
                "audio": str(input_file),
                "cookie": "sessionid=must-not-survive",
                "source_url": "https://example.test/video?msToken=secret&signature=signed",
            },
            input_files=[input_file],
        )
        recorder.succeed(
            audit,
            response={"text": "完整转写", "access_token": "also-secret"},
            stdout="all stdout",
            stderr="all stderr",
        )

        stored = session.scalar(select(ProcessingAudit))
        assert stored is not None
        request = json.loads(stored.request_json)
        response = json.loads(stored.response_json)
        files = json.loads(stored.input_files_json)
        assert request["cookie"] == "[REDACTED]"
        assert request["source_url"] == "https://example.test/video"
        assert response["access_token"] == "[REDACTED]"
        assert stored.stdout == "all stdout"
        assert stored.stderr == "all stderr"
        assert files[0]["sha256"]
        assert files[0]["bytes"] == len(b"audio-for-hash")
        serialized = " ".join(
            value or ""
            for value in (
                stored.request_json,
                stored.response_json,
                stored.stdout,
                stored.stderr,
            )
        )
        assert "must-not-survive" not in serialized
        assert "also-secret" not in serialized
        assert "signed" not in serialized


def test_audit_removes_query_parameters_from_urls_embedded_in_logs() -> None:
    from douyin_library.audit import sanitize_text

    value = (
        "download failed at https://cdn.example.test/media.mp4?signature=secret&expires=1 "
        "before retry"
    )

    sanitized = sanitize_text(value)

    assert sanitized == "download failed at https://cdn.example.test/media.mp4 before retry"
    assert "secret" not in sanitized


def test_large_audit_response_is_externalized_with_hash_and_summary(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    audit_root = tmp_path / "audit"

    with Session(engine) as session:
        recorder = ProcessingAuditRecorder(
            session,
            audit_root=audit_root,
            inline_max_bytes=1024,
        )
        audit = recorder.start(
            operation="sync_favorites",
            processor="test",
            request={},
        )
        recorder.succeed(
            audit,
            response={"status": "succeeded", "pages": ["x" * 4096]},
        )

        reference = json.loads(audit.response_json or "{}")
        payload_path = Path(reference["path"])
        assert reference["kind"] == "external-audit-payload-v1"
        assert reference["bytes"] > 4096
        assert len(reference["sha256"]) == 64
        assert reference["summary"]["fields"]["status"] == "succeeded"
        assert payload_path.is_file()
        assert "x" * 100 in payload_path.read_text(encoding="utf-8")


def test_existing_large_audit_response_can_be_compacted(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        audit = ProcessingAudit(
            operation="sync_favorites",
            processor="legacy",
            request_json="{}",
            response_json=json.dumps({"pages": ["x" * 4096]}),
            status="succeeded",
        )
        session.add(audit)
        session.commit()

        result = compact_large_audit_responses(
            session,
            audit_root=tmp_path / "audit",
            inline_max_bytes=1024,
        )
        session.refresh(audit)

        assert result["migrated"] == 1
        assert result["bytes_externalized"] > 4096
        assert json.loads(audit.response_json or "{}")["kind"] == (
            "external-audit-payload-v1"
        )


def test_compaction_externalizes_all_large_audit_fields_and_is_idempotent(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    large_request = json.dumps({"prompt": "p" * 5000}, ensure_ascii=False)
    large_response = json.dumps({"raw": "r" * 5000}, ensure_ascii=False)
    large_stdout = "stdout-" + "o" * 5000
    large_stderr = "stderr-" + "e" * 5000

    with Session(engine) as session:
        audit = ProcessingAudit(
            operation="legacy-operation",
            processor="legacy",
            request_json=large_request,
            response_json=large_response,
            stdout=large_stdout,
            stderr=large_stderr,
            status="succeeded",
        )
        session.add(audit)
        session.commit()

        result = compact_large_audit_payloads(
            session,
            audit_root=tmp_path / "audit",
            inline_max_bytes=1024,
        )
        session.refresh(audit)
        assert result["migrated_fields"] == 4
        assert result["migrated"] == 1
        for field in ("request_json", "response_json", "stdout", "stderr"):
            value = getattr(audit, field)
            envelope = json.loads(value)
            assert envelope["kind"] == "external-audit-payload-v1"
            suffix = "json" if field.endswith("_json") else "txt"
            assert Path(envelope["path"]).suffix == f".{suffix}"
            assert AuditPayloadStore(tmp_path / "audit", 1024).load(value) is not None

        second = compact_large_audit_payloads(
            session,
            audit_root=tmp_path / "audit",
            inline_max_bytes=1024,
        )
        assert second["migrated_fields"] == 0


def test_audit_payload_load_rejects_escape_and_corruption(tmp_path: Path) -> None:
    store = AuditPayloadStore(tmp_path / "audit", 1)
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        audit = ProcessingAudit(
            operation="load-test",
            processor="test",
            request_json="{}",
            status="running",
        )
        session.add(audit)
        session.flush()
        reference = store.store(
            audit,
            field="response",
            value="payload",
            suffix="txt",
        )
        assert store.load(reference) == "payload"

        envelope = json.loads(reference)
        envelope["path"] = str(tmp_path / "outside.txt")
        (tmp_path / "outside.txt").write_text("payload", encoding="utf-8")
        with pytest.raises(ValueError, match="outside_root"):
            store.load(json.dumps(envelope))

        envelope = json.loads(reference)
        Path(envelope["path"]).write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="audit_payload_(bytes|sha256)"):
            store.load(json.dumps(envelope))
