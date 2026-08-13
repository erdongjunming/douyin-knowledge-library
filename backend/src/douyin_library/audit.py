from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy.orm import Session

from douyin_library.models import Job, ProcessingAudit

REDACTED = "[REDACTED]"
SENSITIVE_KEY_PARTS = (
    "cookie",
    "token",
    "authorization",
    "password",
    "secret",
    "signature",
    "a_bogus",
)
SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)(cookie|authorization)\s*[:=]\s*[^\r\n]+"),
    re.compile(r"(?i)(sessionid|msToken|a_bogus|x-secsdk-web-signature)=[^;&\s]+"),
)
URL_PATTERN = re.compile(r"https?://[^\s<>'\"]+")


def sanitize_for_audit(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if _sensitive_key(str(key))
                else sanitize_for_audit(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_audit(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def sanitize_text(value: str) -> str:
    sanitized = URL_PATTERN.sub(lambda match: _strip_url_query(match.group(0)), value)
    for pattern in SENSITIVE_TEXT_PATTERNS:
        sanitized = pattern.sub(lambda match: f"{match.group(1)}={REDACTED}", sanitized)
    return sanitized


def _sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _strip_url_query(value: str) -> str:
    if not value.lower().startswith(("http://", "https://")):
        return value
    parsed = urlsplit(value)
    if not parsed.query and not parsed.fragment:
        return value
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def file_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for candidate in paths:
        path = Path(candidate)
        if not path.is_file():
            manifest.append({"path": str(path), "exists": False})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append(
            {
                "path": str(path),
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return manifest


class ProcessingAuditRecorder:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(
        self,
        *,
        operation: str,
        processor: str,
        request: Mapping[str, Any],
        job: Job | None = None,
        work_id: int | None = None,
        sync_run_id: int | None = None,
        model: str | None = None,
        version: str | None = None,
        input_files: Iterable[Path] = (),
    ) -> ProcessingAudit:
        audit = ProcessingAudit(
            job_id=job.id if job else None,
            work_id=work_id if work_id is not None else (job.work_id if job else None),
            sync_run_id=(
                sync_run_id if sync_run_id is not None else (job.sync_run_id if job else None)
            ),
            operation=operation,
            processor=processor,
            model=model,
            version=version,
            request_json=_json(request),
            input_files_json=_json(file_manifest(input_files)),
            status="running",
        )
        self.session.add(audit)
        self.session.commit()
        self.session.refresh(audit)
        return audit

    def succeed(
        self,
        audit: ProcessingAudit,
        *,
        response: Mapping[str, Any],
        stdout: str = "",
        stderr: str = "",
        output_files: Iterable[Path] = (),
        sync_run_id: int | None = None,
    ) -> ProcessingAudit:
        return self._finish(
            audit,
            status="succeeded",
            response=response,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            sync_run_id=sync_run_id,
        )

    def fail(
        self,
        audit: ProcessingAudit,
        *,
        error: str,
        response: Mapping[str, Any] | None = None,
        stdout: str = "",
        stderr: str = "",
        output_files: Iterable[Path] = (),
        sync_run_id: int | None = None,
    ) -> ProcessingAudit:
        return self._finish(
            audit,
            status="failed",
            response=response,
            stdout=stdout,
            stderr=stderr,
            error=error,
            output_files=output_files,
            sync_run_id=sync_run_id,
        )

    def _finish(
        self,
        audit: ProcessingAudit,
        *,
        status: str,
        response: Mapping[str, Any] | None,
        stdout: str,
        stderr: str,
        output_files: Iterable[Path],
        error: str | None = None,
        sync_run_id: int | None = None,
    ) -> ProcessingAudit:
        audit.status = status
        audit.response_json = _json(response) if response is not None else None
        audit.stdout = sanitize_text(stdout)
        audit.stderr = sanitize_text(stderr)
        audit.output_files_json = _json(file_manifest(output_files))
        audit.error = sanitize_text(error) if error else None
        audit.finished_at = datetime.now(UTC)
        if sync_run_id is not None:
            audit.sync_run_id = sync_run_id
        self.session.commit()
        self.session.refresh(audit)
        return audit


def _json(value: Any) -> str:
    return json.dumps(sanitize_for_audit(value), ensure_ascii=False, sort_keys=True)
