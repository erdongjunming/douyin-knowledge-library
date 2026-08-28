import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.providers.base import NoteProviderError
from douyin_library.notes.providers.external import ExternalNoteProvider

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
SECRET = "external-key-must-not-leak"


def _settings(**overrides) -> Settings:
    values = {
        "note_external_base_url": "https://external.test/v1",
        "note_external_model": "external-model",
        "note_external_api_key": SecretStr(SECRET),
        "note_request_timeout_seconds": 60,
        "note_context_tokens": 65536,
    }
    values.update(overrides)
    return Settings(**values)


async def _complete(provider: ExternalNoteProvider, cancel_event=None):
    return await provider.complete_json(
        stage="review",
        system_prompt="system prompt",
        payload={"input": "完整输入"},
        json_schema=SCHEMA,
        cancel_event=cancel_event or asyncio.Event(),
    )


@pytest.mark.asyncio
async def test_external_request_uses_auth_header_and_strict_json_schema_only() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 101, "completion_tokens": 23, "cost": 0.42},
            },
        )

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    result = await _complete(provider)

    request = captured["request"]
    assert isinstance(request, httpx.Request)
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "external-model"
    assert body["temperature"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": '{"input": "完整输入"}'},
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "note_review_response",
            "strict": True,
            "schema": SCHEMA,
        },
    }
    assert result.parsed_json == {"answer": "ok"}
    assert result.usage.input_tokens == 101
    assert result.usage.output_tokens == 23
    assert result.usage.api_cost == 0.42
    assert result.usage.cost_status == "reported"
    assert SECRET not in request.content.decode("utf-8")
    assert SECRET not in str(request.url)
    assert SECRET not in repr(result)
    assert "authorization" not in repr(result.diagnostics).lower()


@pytest.mark.asyncio
async def test_external_cost_is_estimated_only_when_both_rates_and_tokens_exist() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 2_000_000, "completion_tokens": 500_000},
            },
        )

    provider = ExternalNoteProvider(
        _settings(
            note_external_input_cost_per_million=0.25,
            note_external_output_cost_per_million=1.5,
        ),
        transport=httpx.MockTransport(handler),
    )

    result = await _complete(provider)

    assert result.usage.api_cost == pytest.approx(1.25)
    assert result.usage.cost_status == "estimated_from_config"


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_cost", [None, "not-numeric"])
async def test_external_nonnumeric_cost_allows_configured_estimate(reported_cost) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 2_000_000,
                    "completion_tokens": 500_000,
                    "cost": reported_cost,
                },
            },
        )

    provider = ExternalNoteProvider(
        _settings(
            note_external_input_cost_per_million=0.25,
            note_external_output_cost_per_million=1.5,
        ),
        transport=httpx.MockTransport(handler),
    )

    result = await _complete(provider)

    assert result.usage.api_cost == pytest.approx(1.25)
    assert result.usage.cost_status == "estimated_from_config"


@pytest.mark.asyncio
async def test_external_estimated_cost_overflow_is_unknown() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 10**400,
                    "completion_tokens": 1,
                },
            },
        )

    provider = ExternalNoteProvider(
        _settings(
            note_external_input_cost_per_million=1e308,
            note_external_output_cost_per_million=0,
        ),
        transport=httpx.MockTransport(handler),
    )

    result = await _complete(provider)

    assert result.usage.api_cost is None
    assert result.usage.cost_status == "unknown"


@pytest.mark.asyncio
async def test_external_reported_huge_integer_cost_is_unknown() -> None:
    response_body = json.dumps(
        {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost": 10**400,
            },
        }
    ).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=response_body,
            headers={"content-type": "application/json"},
        )

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    result = await _complete(provider)

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.api_cost is None
    assert result.usage.cost_status == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 10, "completion_tokens": 5},
        {"prompt_tokens": 10},
        {"prompt_tokens": 10, "completion_tokens": 5, "cost": -1},
        {"prompt_tokens": 10, "completion_tokens": 5, "cost": "not-numeric"},
    ],
)
async def test_external_unknown_or_invalid_cost_is_never_fabricated(usage) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}], "usage": usage},
        )

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    result = await _complete(provider)

    assert result.usage.api_cost is None
    assert result.usage.cost_status == "unknown"


@pytest.mark.parametrize(
    "overrides",
    [
        {"note_external_base_url": None},
        {"note_external_base_url": "  "},
        {"note_external_model": None},
        {"note_external_model": "\t"},
        {"note_external_api_key": None},
        {"note_external_api_key": SecretStr(" \t")},
    ],
)
def test_external_constructor_rejects_incomplete_configuration_without_leaking_key(
    overrides,
) -> None:
    with pytest.raises(ValueError) as captured:
        ExternalNoteProvider(_settings(**overrides))

    assert "configuration" in str(captured.value)
    assert SECRET not in str(captured.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.test/v1",
        "ftp://api.example.test/v1",
        "https://user:password@api.example.test/v1",
        "https://api.example.test/v1?token=url-secret",
        "https://api.example.test/v1#url-secret",
        "http://[::1/url-secret",
        "http://127.0.0.1:url-secret",
        "https://api.example.test:url-secret/v1",
        "api.example.test/url-secret",
        "https:///url-secret",
    ],
)
def test_external_rejects_unsafe_base_urls_without_leaking_secrets(base_url: str) -> None:
    with pytest.raises(ValueError) as captured:
        ExternalNoteProvider(_settings(note_external_base_url=base_url))

    message = str(captured.value)
    assert "configuration" in message
    assert SECRET not in message
    assert "password" not in message
    assert "url-secret" not in message


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
        ("https://api.example.test/v1", "https://api.example.test/v1"),
    ],
)
def test_external_accepts_https_and_loopback_http_base_urls(
    base_url: str,
    expected: str,
) -> None:
    provider = ExternalNoteProvider(_settings(note_external_base_url=base_url))

    assert provider.base_url == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "expected_trust_env"),
    [
        ("http://127.0.0.1:8000/v1", False),
        ("https://api.example.test/v1", True),
    ],
)
async def test_external_client_controls_environment_proxy_use(
    base_url: str,
    expected_trust_env: bool,
) -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}]},
        )

    def client_factory(**kwargs) -> httpx.AsyncClient:
        captured.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return httpx.AsyncClient(**kwargs)

    provider = ExternalNoteProvider(
        _settings(note_external_base_url=base_url),
        client_factory=client_factory,
    )

    await _complete(provider)

    assert captured["trust_env"] is expected_trust_env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(401, text=f"denied {SECRET} raw-response"), "http_error"),
        (httpx.Response(200, json=[]), "invalid_response"),
        (httpx.Response(200, json={}), "invalid_response"),
        (httpx.Response(200, json={"choices": []}), "invalid_response"),
        (
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "raw-response"}}]},
            ),
            "invalid_json",
        ),
        (
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "[]"}}]},
            ),
            "invalid_output",
        ),
    ],
)
async def test_external_response_failures_are_sanitized(
    response: httpx.Response,
    code: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return response

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(NoteProviderError) as captured:
        await _complete(provider)

    assert captured.value.provider == "external"
    assert captured.value.stage == "review"
    assert captured.value.code == code
    assert SECRET not in str(captured.value)
    assert "raw-response" not in str(captured.value)
    assert len(str(captured.value)) < 100


@pytest.mark.asyncio
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
async def test_external_rejects_nonstandard_json_constants(constant: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": f'{{"answer": {constant}}}'}}
                ]
            },
        )

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(NoteProviderError) as captured:
        await _complete(provider)

    assert captured.value.code == "invalid_json"


@pytest.mark.asyncio
async def test_external_cancellation_cancels_the_in_flight_request_task() -> None:
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise
        raise AssertionError("unreachable")

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))
    cancel_event = asyncio.Event()
    task = asyncio.create_task(_complete(provider, cancel_event))
    await asyncio.wait_for(request_started.wait(), timeout=1)

    cancel_event.set()

    with pytest.raises(JobCancelledError, match="cancelled_by_user"):
        await asyncio.wait_for(task, timeout=1)
    await asyncio.wait_for(request_cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_external_pre_cancel_does_not_start_transport_request() -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))
    cancel_event = asyncio.Event()
    cancel_event.set()

    with pytest.raises(JobCancelledError, match="cancelled_by_user"):
        await _complete(provider, cancel_event)

    assert requests == 0


@pytest.mark.asyncio
async def test_external_unload_is_a_noop() -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500)

    provider = ExternalNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    await provider.unload()

    assert requests == 0
