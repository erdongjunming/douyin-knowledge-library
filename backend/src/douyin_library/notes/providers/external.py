from __future__ import annotations

import asyncio
import inspect
import json
import math
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
    validated_external_base_url,
)

AsyncClientFactory = Callable[..., httpx.AsyncClient]


class ExternalNoteProvider:
    name = "external"
    is_local = False

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        client_factory: AsyncClientFactory | None = None,
    ) -> None:
        parsed_url = validated_external_base_url(settings.note_external_base_url or "")
        model = (settings.note_external_model or "").strip()
        api_key = settings.note_external_api_key
        key_is_nonempty = bool(api_key and api_key.get_secret_value().strip())
        if parsed_url is None or not model or not key_is_nonempty:
            raise ValueError("external_provider_configuration_incomplete")
        self.base_url = parsed_url.normalized_url
        self.model = model
        self.context_tokens = settings.note_context_tokens
        self._api_key = api_key
        self._input_rate = settings.note_external_input_cost_per_million
        self._output_rate = settings.note_external_output_cost_per_million
        self._timeout = settings.note_request_timeout_seconds
        self._transport = transport
        self._client_factory = client_factory
        self._trust_env = not (parsed_url.scheme == "http" and parsed_url.is_loopback)

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
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"note_{stage}_response",
                    "strict": True,
                    "schema": dict(json_schema),
                },
            },
        }
        try:
            response = await request_or_cancel(
                self._post(request_body),
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
        raw_output = _choice_content(response_payload, self.name, stage)
        parsed_json = load_json_object(raw_output, self.name, stage)
        usage = self._usage(response_payload.get("usage"))
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
        return None

    async def _post(self, body: Mapping[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value().strip()}"}
        async with self._make_client() as client:
            return await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=dict(body),
            )

    def _make_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return _call_client_factory(
                self._client_factory,
                timeout=self._timeout,
                trust_env=self._trust_env,
            )
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=self._timeout,
            trust_env=self._trust_env,
        )

    def _usage(self, raw_usage: Any) -> ProviderUsage:
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        input_tokens = _nonnegative_int(usage.get("prompt_tokens"))
        output_tokens = _nonnegative_int(usage.get("completion_tokens"))
        reported_cost = usage.get("cost")
        if _numeric_cost(reported_cost):
            safe_cost = _safe_cost(reported_cost)
            if safe_cost is not None:
                return ProviderUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    api_cost=safe_cost,
                    cost_status="reported",
                )
            return ProviderUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                api_cost=None,
                cost_status="unknown",
            )
        if (
            input_tokens is not None
            and output_tokens is not None
            and self._input_rate is not None
            and self._output_rate is not None
        ):
            try:
                estimated = (
                    input_tokens * self._input_rate + output_tokens * self._output_rate
                ) / 1_000_000
            except OverflowError:
                estimated = math.inf
            if math.isfinite(estimated) and estimated >= 0:
                return ProviderUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    api_cost=estimated,
                    cost_status="estimated_from_config",
                )
        return ProviderUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_cost=None,
            cost_status="unknown",
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


def _choice_content(payload: dict[str, Any], provider: str, stage: NoteStage) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise NoteProviderError(provider, stage, "invalid_response")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise NoteProviderError(provider, stage, "invalid_response")
    return message["content"]


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_cost(value: Any) -> float | None:
    if not _numeric_cost(value) or value < 0:
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    if isinstance(value, int) and int(converted) != value:
        return None
    return converted


def _numeric_cost(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _call_client_factory(
    factory: AsyncClientFactory,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory()
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return factory(**client_kwargs)
    supported_kwargs = {
        name: value
        for name, value in client_kwargs.items()
        if name in parameters
        and parameters[name].kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return factory(**supported_kwargs)


ExternalProvider = ExternalNoteProvider

__all__ = ["ExternalNoteProvider", "ExternalProvider"]
