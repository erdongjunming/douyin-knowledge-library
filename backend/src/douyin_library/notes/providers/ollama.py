from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.contracts import ProviderUsage
from douyin_library.notes.providers.base import (
    NoteProviderError,
    NoteStage,
    ProviderResponse,
    load_json_object,
    request_or_cancel,
    sanitized_base_url,
)

AsyncClientFactory = Callable[[], httpx.AsyncClient]


class OllamaNoteProvider:
    name = "ollama"
    is_local = True
    # The Ollama JSON endpoint accepts every typed note stage.  Keeping this
    # capability explicit lets the pipeline retain a legacy fallback for
    # older test/external providers without disabling teaching notes locally.
    supports_teaching_pipeline = True

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        client_factory: AsyncClientFactory | None = None,
    ) -> None:
        base_url = sanitized_base_url(settings.note_ollama_url)
        model = settings.note_ollama_model.strip()
        if not base_url or not model:
            raise ValueError("ollama_provider_configuration_incomplete")
        self.base_url = base_url
        self.model = model
        self.context_tokens = settings.note_context_tokens
        self.keep_alive = settings.note_ollama_keep_alive
        self._timeout = settings.note_request_timeout_seconds
        self._unload_timeout = settings.note_ollama_unload_timeout_seconds
        self._transport = transport
        self._client_factory = client_factory

    async def complete_json(
        self,
        *,
        stage: NoteStage,
        system_prompt: str,
        payload: Mapping[str, Any],
        json_schema: Mapping[str, Any],
        cancel_event: asyncio.Event,
    ) -> ProviderResponse:
        try:
            user_payload = json.dumps(dict(payload), ensure_ascii=False)
        except (TypeError, ValueError):
            raise NoteProviderError(self.name, stage, "invalid_request") from None
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "stream": False,
            "format": dict(json_schema),
            "options": {"temperature": 0, "num_ctx": self.context_tokens},
            "think": False,
            "keep_alive": self.keep_alive,
        }
        try:
            response = await request_or_cancel(
                self._post("/api/chat", request_body),
                cancel_event,
            )
        except JobCancelledError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            raise NoteProviderError(self.name, stage, "transport_error") from None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise NoteProviderError(self.name, stage, "http_error") from None

        response_payload = _response_object(response, self.name, stage)
        message = response_payload.get("message")
        if not isinstance(message, dict):
            raise NoteProviderError(self.name, stage, "invalid_response")
        raw_output = message.get("content")
        if not isinstance(raw_output, str):
            raise NoteProviderError(self.name, stage, "invalid_response")
        parsed_json = load_json_object(raw_output, self.name, stage)
        usage = ProviderUsage(
            input_tokens=_nonnegative_int(response_payload.get("prompt_eval_count")),
            output_tokens=_nonnegative_int(response_payload.get("eval_count")),
            api_cost=0.0,
            cost_status="zero_local",
        )
        return ProviderResponse(
            raw_output=raw_output,
            parsed_json=parsed_json,
            usage=usage,
            diagnostics={
                "provider": self.name,
                "model": self.model,
                "stage": stage,
                "status": "ok",
                "base_url": self.base_url,
            },
        )

    async def unload(self) -> None:
        try:
            response = await asyncio.wait_for(
                self._post(
                    "/api/generate",
                    {"model": self.model, "keep_alive": 0},
                    timeout=self._unload_timeout,
                ),
                timeout=self._unload_timeout,
            )
            response.raise_for_status()
        except TimeoutError:
            raise NoteProviderError(self.name, "unload", "unload_timeout") from None
        except httpx.TimeoutException:
            raise NoteProviderError(self.name, "unload", "unload_timeout") from None
        except asyncio.CancelledError:
            raise
        except Exception:
            raise NoteProviderError(self.name, "unload", "unload_failed") from None

    async def _post(
        self,
        path: str,
        body: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> httpx.Response:
        async with self._make_client(timeout=timeout) as client:
            return await client.post(f"{self.base_url}{path}", json=dict(body))

    def _make_client(self, *, timeout: float | None = None) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=self._timeout if timeout is None else timeout,
        )


def _response_object(
    response: httpx.Response,
    provider: str,
    stage: NoteStage,
) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise NoteProviderError(provider, stage, "invalid_response") from None
    if not isinstance(payload, dict):
        raise NoteProviderError(provider, stage, "invalid_response")
    return payload


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


OllamaProvider = OllamaNoteProvider

__all__ = ["OllamaNoteProvider", "OllamaProvider"]
