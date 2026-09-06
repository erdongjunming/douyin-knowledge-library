from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.contracts import ProviderUsage

NoteStage = Literal[
    "extract",
    "outline",
    "synthesize",
    "section",
    "review",
    "depth_review",
    "repair",
]
_SAFE_ERROR_PART = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    raw_output: str
    parsed_json: dict[str, Any]
    usage: ProviderUsage
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParsedBaseUrl:
    scheme: str
    hostname: str
    port: int | None
    path: str
    has_query: bool
    has_fragment: bool
    has_credentials: bool
    is_loopback: bool
    netloc: str
    normalized_url: str


class NoteProviderError(RuntimeError):
    def __init__(self, provider: str, stage: str, code: str) -> None:
        self.provider = provider
        self.stage = stage
        self.code = code
        safe_parts = (_safe_error_part(provider), _safe_error_part(stage), _safe_error_part(code))
        super().__init__(":".join(safe_parts))


@runtime_checkable
class NoteProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def is_local(self) -> bool: ...

    @property
    def context_tokens(self) -> int: ...

    async def complete_json(
        self,
        *,
        stage: NoteStage,
        system_prompt: str,
        payload: Mapping[str, Any],
        json_schema: Mapping[str, Any],
        cancel_event: asyncio.Event,
    ) -> ProviderResponse: ...

    async def unload(self) -> None: ...


async def request_or_cancel[T](
    request: Awaitable[T],
    cancel_event: asyncio.Event,
) -> T:
    if cancel_event.is_set():
        if inspect.iscoroutine(request):
            request.close()
        raise JobCancelledError("cancelled_by_user")
    request_task = asyncio.ensure_future(request)
    cancellation_task = asyncio.create_task(cancel_event.wait())
    try:
        await asyncio.wait(
            {request_task, cancellation_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_event.is_set():
            request_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await request_task
            raise JobCancelledError("cancelled_by_user")
        return await request_task
    except asyncio.CancelledError:
        request_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await request_task
        raise
    finally:
        cancellation_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancellation_task


def load_json_object(
    raw_output: str,
    provider: str,
    stage: NoteStage,
) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_output, parse_constant=_reject_json_constant)
    except (TypeError, ValueError):
        raise NoteProviderError(provider, stage, "invalid_json") from None
    if not isinstance(parsed, dict):
        raise NoteProviderError(provider, stage, "invalid_output")
    return parsed


def parse_base_url(value: str) -> ParsedBaseUrl | None:
    try:
        parsed = urlsplit(value.strip())
        scheme = parsed.scheme.casefold()
        hostname = parsed.hostname
        port = parsed.port
        username = parsed.username
        password = parsed.password
        path = parsed.path.rstrip("/")
        has_query = bool(parsed.query)
        has_fragment = bool(parsed.fragment)
    except (AttributeError, TypeError, ValueError):
        return None
    if not scheme or hostname is None:
        return None
    hostname = hostname.casefold()
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return ParsedBaseUrl(
        scheme=scheme,
        hostname=hostname,
        port=port,
        path=path,
        has_query=has_query,
        has_fragment=has_fragment,
        has_credentials=username is not None or password is not None,
        is_loopback=hostname in {"localhost", "127.0.0.1", "::1"},
        netloc=netloc,
        normalized_url=urlunsplit((scheme, netloc, path, "", "")),
    )


def sanitized_base_url(value: str) -> str:
    parsed = parse_base_url(value)
    return parsed.normalized_url if parsed is not None else ""


def validated_external_base_url(value: str) -> ParsedBaseUrl | None:
    parsed = parse_base_url(value)
    if (
        parsed is None
        or parsed.has_credentials
        or parsed.has_query
        or parsed.has_fragment
    ):
        return None
    if parsed.scheme == "https" or (parsed.scheme == "http" and parsed.is_loopback):
        return parsed
    return None


def _safe_error_part(value: str) -> str:
    return _SAFE_ERROR_PART.sub("_", str(value))[:40] or "unknown"


def _reject_json_constant(value: str) -> None:
    raise ValueError("nonstandard_json_constant")


__all__ = [
    "NoteProvider",
    "NoteProviderError",
    "NoteStage",
    "ParsedBaseUrl",
    "ProviderResponse",
    "load_json_object",
    "parse_base_url",
    "request_or_cancel",
    "sanitized_base_url",
    "validated_external_base_url",
]
