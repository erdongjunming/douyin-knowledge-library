import asyncio
import json

import httpx
import pytest

from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.providers.base import NoteProviderError
from douyin_library.notes.providers.ollama import OllamaNoteProvider

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _settings(**overrides) -> Settings:
    values = {
        "note_ollama_url": "http://ollama.test",
        "note_ollama_model": "qwen3:8b",
        "note_context_tokens": 32768,
        "note_ollama_keep_alive": "10m",
        "note_request_timeout_seconds": 60,
    }
    values.update(overrides)
    return Settings(**values)


async def _complete(provider: OllamaNoteProvider, cancel_event=None):
    payload = {"transcript": "完整内容" * 4000, "segments": [0, 1, 2]}
    response = await provider.complete_json(
        stage="extract",
        system_prompt="只输出 JSON",
        payload=payload,
        json_schema=SCHEMA,
        cancel_event=cancel_event or asyncio.Event(),
    )
    return payload, response


@pytest.mark.asyncio
async def test_ollama_sends_complete_payload_schema_and_local_options() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"answer":"完成"}'},
                "prompt_eval_count": 123,
                "eval_count": 45,
            },
        )

    provider = OllamaNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    payload, result = await _complete(provider)

    assert captured["path"] == "/api/chat"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is False
    assert body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0, "num_ctx": 32768}
    assert body["think"] is False
    assert body["keep_alive"] == "10m"
    assert body["messages"][0] == {"role": "system", "content": "只输出 JSON"}
    assert json.loads(body["messages"][1]["content"]) == payload
    assert result.raw_output == '{"answer":"完成"}'
    assert result.parsed_json == {"answer": "完成"}
    assert result.usage.input_tokens == 123
    assert result.usage.output_tokens == 45
    assert result.usage.api_cost == 0
    assert result.usage.cost_status == "zero_local"
    assert result.diagnostics == {
        "provider": "ollama",
        "model": "qwen3:8b",
        "stage": "extract",
        "status": "ok",
        "base_url": "http://ollama.test",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(500, text="raw-body-must-not-leak"), "http_error"),
        (httpx.Response(200, json=[]), "invalid_response"),
        (httpx.Response(200, json={}), "invalid_response"),
        (httpx.Response(200, json={"message": {}}), "invalid_response"),
        (
            httpx.Response(
                200,
                json={"message": {"content": "raw-body-must-not-leak"}},
            ),
            "invalid_json",
        ),
        (
            httpx.Response(200, json={"message": {"content": "[]"}}),
            "invalid_output",
        ),
    ],
)
async def test_ollama_response_failures_are_sanitized(
    response: httpx.Response,
    code: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return response

    provider = OllamaNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(NoteProviderError) as captured:
        await _complete(provider)

    assert captured.value.provider == "ollama"
    assert captured.value.stage == "extract"
    assert captured.value.code == code
    assert "raw-body-must-not-leak" not in str(captured.value)
    assert len(str(captured.value)) < 100


@pytest.mark.asyncio
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
async def test_ollama_rejects_nonstandard_json_constants(constant: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": f'{{"answer": {constant}}}'}},
        )

    provider = OllamaNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(NoteProviderError) as captured:
        await _complete(provider)

    assert captured.value.code == "invalid_json"


@pytest.mark.asyncio
async def test_ollama_transport_error_is_sanitized() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private transport diagnostics", request=request)

    provider = OllamaNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(NoteProviderError, match="transport_error") as captured:
        await _complete(provider)

    assert "private transport diagnostics" not in str(captured.value)


@pytest.mark.asyncio
async def test_ollama_cancellation_cancels_the_in_flight_request_task() -> None:
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    requested_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise
        raise AssertionError("unreachable")

    provider = OllamaNoteProvider(_settings(), transport=httpx.MockTransport(handler))
    cancel_event = asyncio.Event()
    task = asyncio.create_task(_complete(provider, cancel_event))
    await asyncio.wait_for(request_started.wait(), timeout=1)

    cancel_event.set()

    with pytest.raises(JobCancelledError, match="cancelled_by_user"):
        await asyncio.wait_for(task, timeout=1)
    await asyncio.wait_for(request_cancelled.wait(), timeout=1)
    assert requested_paths == ["/api/chat"]
    assert "/api/pull" not in requested_paths


@pytest.mark.asyncio
async def test_ollama_pre_cancel_does_not_start_transport_request() -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"message": {"content": "{}"}})

    provider = OllamaNoteProvider(_settings(), transport=httpx.MockTransport(handler))
    cancel_event = asyncio.Event()
    cancel_event.set()

    with pytest.raises(JobCancelledError, match="cancelled_by_user"):
        await _complete(provider, cancel_event)

    assert requests == 0


@pytest.mark.asyncio
async def test_ollama_unload_sets_keep_alive_zero_and_never_pulls() -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"done": True})

    provider = OllamaNoteProvider(_settings(), transport=httpx.MockTransport(handler))

    await provider.unload()

    assert requests == [
        ("/api/generate", {"model": "qwen3:8b", "keep_alive": 0})
    ]
    assert all(path != "/api/pull" for path, _ in requests)


@pytest.mark.asyncio
async def test_ollama_unload_times_out_quickly_and_cancels_request() -> None:
    requested_paths: list[str] = []
    request_cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise
        raise AssertionError("unreachable")

    provider = OllamaNoteProvider(
        _settings(note_ollama_unload_timeout_seconds=0.05),
        transport=httpx.MockTransport(handler),
    )
    started = asyncio.get_running_loop().time()

    with pytest.raises(NoteProviderError) as captured:
        await asyncio.wait_for(provider.unload(), timeout=0.5)

    elapsed = asyncio.get_running_loop().time() - started
    assert captured.value.code == "unload_timeout"
    assert elapsed < 0.5
    await asyncio.wait_for(request_cancelled.wait(), timeout=0.1)
    assert requested_paths == ["/api/generate"]
    assert "/api/pull" not in requested_paths


@pytest.mark.asyncio
async def test_ollama_diagnostics_strip_query_parameters() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.query == b""
        return httpx.Response(200, json={"message": {"content": "{}"}})

    provider = OllamaNoteProvider(
        _settings(note_ollama_url="http://ollama.test?signature=query-secret"),
        transport=httpx.MockTransport(handler),
    )

    _, result = await _complete(provider)

    assert result.diagnostics["base_url"] == "http://ollama.test"
    assert "query-secret" not in repr(result.diagnostics)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://[::1/url-secret",
        "http://127.0.0.1:url-secret",
        "https://api.example.test:url-secret/v1",
        "ollama.test/url-secret",
        "http:///url-secret",
    ],
)
def test_ollama_constructor_rejects_malformed_url_without_leaking_it(
    base_url: str,
) -> None:
    with pytest.raises(ValueError) as captured:
        OllamaNoteProvider(_settings(note_ollama_url=base_url))

    message = str(captured.value)
    assert "configuration" in message
    assert "url-secret" not in message
