import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.audit import ProcessingAuditRecorder
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
