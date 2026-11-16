from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
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
EXTERNAL_PAYLOAD_KIND = "external-audit-payload-v1"
_PAYLOAD_FIELD_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


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
    def __init__(
        self,
        session: Session,
        *,
        audit_root: Path | None = None,
        inline_max_bytes: int = 64 * 1024,
    ) -> None:
        self.session = session
        self.payload_store = AuditPayloadStore(audit_root, inline_max_bytes)

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
            request_json="{}",
            input_files_json=_json(file_manifest(input_files)),
            status="running",
        )
        self.session.add(audit)
        self.session.flush()
        audit.request_json = self.payload_store.store(
            audit,
            field="request",
            value=_json(request),
            suffix="json",
        )
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

    def cancel(
        self,
        audit: ProcessingAudit,
        *,
        error: str = "cancelled_by_user",
        stdout: str = "",
        stderr: str = "",
        output_files: Iterable[Path] = (),
    ) -> ProcessingAudit:
        return self._finish(
            audit,
            status="cancelled",
            response=None,
            stdout=stdout,
            stderr=stderr,
            error=error,
            output_files=output_files,
        )

    def skip(
        self,
        audit: ProcessingAudit,
        *,
        reason: str,
        response: Mapping[str, Any] | None = None,
        output_files: Iterable[Path] = (),
    ) -> ProcessingAudit:
        return self._finish(
            audit,
            status="skipped",
            response=response,
            stdout="",
            stderr="",
            error=reason,
            output_files=output_files,
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
        audit.response_json = (
            self.payload_store.store(
                audit,
                field="response",
                value=_json(response),
                suffix="json",
            )
            if response is not None
            else None
        )
        sanitized_stdout = sanitize_text(stdout)
        sanitized_stderr = sanitize_text(stderr)
        audit.stdout = self.payload_store.store(
            audit,
            field="stdout",
            value=sanitized_stdout,
            suffix="txt",
        )
        audit.stderr = self.payload_store.store(
            audit,
            field="stderr",
            value=sanitized_stderr,
            suffix="txt",
        )
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


class AuditPayloadStore:
    def __init__(self, root: Path | None, inline_max_bytes: int) -> None:
        self.root = Path(root).resolve() if root is not None else None
        if inline_max_bytes < 0:
            raise ValueError("inline_max_bytes_must_be_nonnegative")
        self.inline_max_bytes = inline_max_bytes

    def store(
        self,
        audit: ProcessingAudit,
        *,
        field: str,
        value: str,
        suffix: str,
    ) -> str:
        if not _PAYLOAD_FIELD_RE.fullmatch(field):
            raise ValueError(f"invalid_audit_payload_field:{field}")
        if not _PAYLOAD_FIELD_RE.fullmatch(suffix):
            raise ValueError(f"invalid_audit_payload_suffix:{suffix}")
        encoded = value.encode("utf-8")
        if self.root is None or len(encoded) <= self.inline_max_bytes:
            return value
        digest = hashlib.sha256(encoded).hexdigest()
        started_at = audit.started_at or datetime.now(UTC)
        operation = re.sub(r"[^A-Za-z0-9_.-]+", "-", audit.operation).strip("-")
        directory = self.root / "processing" / f"{started_at:%Y}" / f"{started_at:%m}"
        path = directory / f"{audit.id}-{operation}-{field}.{suffix}"
        directory.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            if _file_sha256(path) != digest:
                raise RuntimeError(f"audit_payload_conflict:{path}")
        else:
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(encoded)
                if _file_sha256(temporary) != digest:
                    raise RuntimeError("audit_payload_verification_failed")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        envelope = {
            "kind": EXTERNAL_PAYLOAD_KIND,
            "field": field,
            "path": str(path),
            "bytes": len(encoded),
            "sha256": digest,
            "summary": _payload_summary(value, suffix=suffix),
        }
        return json.dumps(envelope, ensure_ascii=False, sort_keys=True)

    def load(self, reference: str) -> str:
        """Load an external payload after validating its envelope and file digest.

        References are persisted in SQLite and therefore must be treated as
        untrusted input.  In particular, never follow a path outside the
        configured audit root, and never return bytes whose size or digest does
        not match the declaration stored in the envelope.
        """

        try:
            envelope = json.loads(reference)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_audit_payload_reference") from exc
        if not isinstance(envelope, Mapping):
            raise ValueError("invalid_audit_payload_reference")
        if envelope.get("kind") != EXTERNAL_PAYLOAD_KIND:
            raise ValueError("not_an_external_audit_payload")
        if self.root is None:
            raise ValueError("audit_payload_root_not_configured")

        raw_path = envelope.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("invalid_audit_payload_path")
        try:
            path = Path(raw_path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("audit_payload_path_outside_root") from exc
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("audit_payload_path_outside_root") from exc
        if not path.exists():
            raise ValueError("audit_payload_file_missing")
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("audit_payload_file_missing") from exc
        if not path.is_file():
            raise ValueError("audit_payload_file_missing")

        declared_bytes = envelope.get("bytes")
        declared_sha256 = envelope.get("sha256")
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 0
        ):
            raise ValueError("invalid_audit_payload_bytes")
        if (
            not isinstance(declared_sha256, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", declared_sha256)
        ):
            raise ValueError("invalid_audit_payload_sha256")
        actual_bytes = path.stat().st_size
        if actual_bytes != declared_bytes:
            raise ValueError("audit_payload_bytes_mismatch")
        actual_sha256 = _file_sha256(path)
        if actual_sha256.lower() != declared_sha256.lower():
            raise ValueError("audit_payload_sha256_mismatch")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("audit_payload_not_utf8") from exc


def is_external_payload(value: str | None) -> bool:
    if not value or EXTERNAL_PAYLOAD_KIND not in value[:200]:
        return False
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("kind") == EXTERNAL_PAYLOAD_KIND


def compact_large_audit_payloads(
    session: Session,
    *,
    audit_root: Path,
    inline_max_bytes: int,
    fields: Iterable[str] = (
        "request_json",
        "response_json",
        "stdout",
        "stderr",
    ),
) -> dict[str, Any]:
    """Externalize oversized legacy audit columns.

    Each column is committed separately.  If one field fails, already migrated
    fields remain durable and a later invocation can safely resume from the
    remaining inline values.
    """

    from sqlalchemy import select

    requested_fields = tuple(dict.fromkeys(fields))
    allowed_fields = {"request_json", "response_json", "stdout", "stderr"}
    invalid_fields = set(requested_fields) - allowed_fields
    if invalid_fields:
        raise ValueError(f"invalid_audit_fields:{','.join(sorted(invalid_fields))}")
    if inline_max_bytes < 0:
        raise ValueError("inline_max_bytes_must_be_nonnegative")
    store = AuditPayloadStore(audit_root, inline_max_bytes)
    ids = session.scalars(select(ProcessingAudit.id).order_by(ProcessingAudit.id)).all()
    migrated = 0
    migrated_fields = 0
    skipped_fields = 0
    failed_fields = 0
    bytes_externalized = 0
    errors: list[dict[str, Any]] = []
    for audit_id in ids:
        audit = session.get(ProcessingAudit, audit_id)
        if audit is None:
            continue
        audit_migrated = False
        for column in requested_fields:
            value = getattr(audit, column)
            if value is None:
                continue
            if is_external_payload(value):
                # Only skip references that still pass integrity validation;
                # malformed envelopes are reported and left untouched.
                try:
                    store.load(value)
                except ValueError as exc:
                    failed_fields += 1
                    errors.append(
                        {
                            "audit_id": audit_id,
                            "field": column,
                            "error": sanitize_text(str(exc)),
                        }
                    )
                else:
                    skipped_fields += 1
                continue
            encoded_bytes = len(value.encode("utf-8"))
            if encoded_bytes <= inline_max_bytes:
                continue
            suffix = "json" if column.endswith("_json") else "txt"
            logical_field = column.removesuffix("_json")
            sanitized = _sanitize_legacy_payload(value, suffix=suffix)
            try:
                reference = store.store(
                    audit,
                    field=logical_field,
                    value=sanitized,
                    suffix=suffix,
                )
                setattr(audit, column, reference)
                session.commit()
            except Exception as exc:
                session.rollback()
                failed_fields += 1
                errors.append(
                    {
                        "audit_id": audit_id,
                        "field": column,
                        "error": sanitize_text(f"{type(exc).__name__}: {exc}"),
                    }
                )
                continue
            migrated_fields += 1
            bytes_externalized += encoded_bytes
            audit_migrated = True
        if audit_migrated:
            migrated += 1
    return {
        "migrated": migrated,
        "migrated_fields": migrated_fields,
        "skipped_fields": skipped_fields,
        "failed_fields": failed_fields,
        "skipped": skipped_fields,
        "failed": failed_fields,
        "bytes_externalized": bytes_externalized,
        "errors": errors,
    }


def compact_large_audit_responses(
    session: Session,
    *,
    audit_root: Path,
    inline_max_bytes: int,
) -> dict[str, Any]:
    """Backward-compatible response-only compaction wrapper."""

    result = compact_large_audit_payloads(
        session,
        audit_root=audit_root,
        inline_max_bytes=inline_max_bytes,
        fields=("response_json",),
    )
    return {
        "migrated": result["migrated"],
        "bytes_externalized": result["bytes_externalized"],
    }


def _sanitize_legacy_payload(value: str, *, suffix: str) -> str:
    if suffix != "json":
        return sanitize_text(value)
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return sanitize_text(value)
    return _json(parsed)


def _payload_summary(value: str, *, suffix: str) -> dict[str, Any]:
    if suffix != "json":
        return {"type": "text", "characters": len(value), "preview": value[:240]}
    if len(value) > 4 * 1024 * 1024:
        summary: dict[str, Any] = {"type": "json", "characters": len(value)}
        for key in (
            "status",
            "sync_run_id",
            "default_count",
            "folder_count",
            "unique_work_count",
            "discovered_count",
            "new_work_count",
        ):
            scalar_pattern = rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+|true|false|null)'
            match = re.search(scalar_pattern, value[:20000])
            if match:
                try:
                    summary[key] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        return summary
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {"type": "invalid-json", "characters": len(value)}
    if isinstance(payload, Mapping):
        return {
            "type": "object",
            "fields": {
                str(key): _summary_value(item)
                for key, item in list(payload.items())[:40]
            },
        }
    return _summary_value(payload)


def _summary_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {"type": "object", "field_count": len(value)}
    if isinstance(value, list):
        return {"type": "list", "item_count": len(value)}
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
