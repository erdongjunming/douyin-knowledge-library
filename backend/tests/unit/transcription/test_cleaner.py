from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from douyin_library.config import Settings
from douyin_library.transcription.contracts import TranscriptDraft, TranscriptSegment
from douyin_library.transcription.providers.ollama_cleaner import (
    CleanerFailureReason,
    OllamaTranscriptCleaner,
    TranscriptCleaningError,
)


def _draft(*texts: str) -> TranscriptDraft:
    segments = tuple(
        TranscriptSegment(f"s{i}", float(i - 1), float(i), text)
        for i, text in enumerate(texts, 1)
    )
    return TranscriptDraft(segments=segments, text="\n".join(texts), language="zh")


def _transport(responses: list[dict], requests: list[httpx.Request]):
    iterator = iter(responses)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=next(iterator), request=request)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_cleaner_sends_deterministic_json_chat_request_and_preserves_timestamps() -> None:
    requests: list[httpx.Request] = []
    response = {
        "message": {
            "content": json.dumps(
                {"segments": [{"id": "s1", "start": 0.0, "end": 1.0, "text": "new"}]},
                ensure_ascii=False,
            )
        }
    }
    settings = Settings(
        transcription_cleaner_model="qwen3:4b",
        transcription_cleaner_keep_alive="5m",
    )
    cleaner = OllamaTranscriptCleaner(settings, transport=_transport([response], requests))

    result = await cleaner.clean(_draft("raw"))

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/api/chat"
    assert body["model"] == "qwen3:4b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["keep_alive"] == "5m"
    assert body["options"]["temperature"] == 0
    assert body["format"] == "json"
    assert result.transcript.segments[0].text == "new"
    assert result.transcript.segments[0].start == 0.0
    assert result.transcript.segments[0].end == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,reason",
    [
        ("not-json", CleanerFailureReason.SCHEMA),
        ("```json\n{\"segments\": []}\n```", CleanerFailureReason.SCHEMA),
        (
            json.dumps({"segments": [{"id": "s1", "start": 2, "end": 3, "text": "x"}]}),
            CleanerFailureReason.SCHEMA,
        ),
        (
            json.dumps({"segments": [{"id": "s2", "start": 0, "end": 1, "text": "x"}]}),
            CleanerFailureReason.SCHEMA,
        ),
        (
            json.dumps({"segments": [{"id": "s1", "start": 0, "end": 1, "text": ""}]}),
            CleanerFailureReason.SCHEMA,
        ),
    ],
)
async def test_cleaner_rejects_invalid_or_unsafe_model_output(
    content: str, reason: CleanerFailureReason
) -> None:
    requests: list[httpx.Request] = []
    response = {"message": {"content": content}}
    cleaner = OllamaTranscriptCleaner(Settings(), transport=_transport([response], requests))

    with pytest.raises(TranscriptCleaningError) as exc_info:
        await cleaner.clean(_draft("raw"))

    assert exc_info.value.reason is reason


@pytest.mark.asyncio
async def test_cleaner_rejects_extreme_text_drift() -> None:
    requests: list[httpx.Request] = []
    response = {
        "message": {
            "content": json.dumps(
                {"segments": [{"id": "s1", "start": 0, "end": 1, "text": "x" * 1000}]}
            )
        }
    }
    settings = Settings(transcription_cleaner_drift_ratio=0.5)
    cleaner = OllamaTranscriptCleaner(settings, transport=_transport([response], requests))

    with pytest.raises(TranscriptCleaningError) as exc_info:
        await cleaner.clean(_draft("short"))

    assert exc_info.value.reason is CleanerFailureReason.DRIFT


@pytest.mark.asyncio
async def test_cleaner_batches_with_one_segment_overlap_and_merges_by_id() -> None:
    requests: list[httpx.Request] = []
    texts = ("one" * 140, "two" * 140, "three" * 140)
    responses = [
        {"message": {"content": json.dumps({"segments": [
                {"id": "s1", "start": 0, "end": 1, "text": texts[0]},
                {"id": "s2", "start": 1, "end": 2, "text": texts[1]},
        ]})}},
        {"message": {"content": json.dumps({"segments": [
                {"id": "s2", "start": 1, "end": 2, "text": texts[1]},
                {"id": "s3", "start": 2, "end": 3, "text": texts[2]},
        ]})}},
    ]
    settings = Settings(transcription_cleaner_batch_character_limit=1300)
    cleaner = OllamaTranscriptCleaner(settings, transport=_transport(responses, requests))

    result = await cleaner.clean(_draft(*texts))

    assert len(requests) == 2
    first = json.loads(requests[0].content)["messages"][1]["content"]
    second = json.loads(requests[1].content)["messages"][1]["content"]
    assert [s["id"] for s in json.loads(first)["segments"]] == ["s1", "s2"]
    assert [s["id"] for s in json.loads(second)["segments"]] == ["s2", "s3"]
    assert [s.segment_id for s in result.transcript.segments] == ["s1", "s2", "s3"]


@pytest.mark.asyncio
async def test_cleaner_honors_cancellation() -> None:
    requests: list[httpx.Request] = []
    event = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await event.wait()
        return httpx.Response(200, json={"message": {"content": "{}"}}, request=request)

    cleaner = OllamaTranscriptCleaner(Settings(), transport=httpx.MockTransport(handler))
    cancel = asyncio.Event()
    task = asyncio.create_task(cleaner.clean(_draft("raw"), cancel_event=cancel))
    await asyncio.sleep(0)
    cancel.set()

    with pytest.raises(TranscriptCleaningError) as exc_info:
        await task

    assert exc_info.value.reason is CleanerFailureReason.CANCELLATION


@pytest.mark.asyncio
async def test_cleaner_rejects_result_when_lease_is_lost_during_request() -> None:
    lease_state = {"owned": True}
    response = {
        "message": {
            "content": json.dumps(
                {"segments": [{"id": "s1", "start": 0.0, "end": 1.0, "text": "new"}]}
            )
        }
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        lease_state["owned"] = False
        return httpx.Response(200, json=response, request=request)

    cleaner = OllamaTranscriptCleaner(
        Settings(),
        transport=httpx.MockTransport(handler),
        lease_checker=lambda: lease_state["owned"],
    )

    with pytest.raises(TranscriptCleaningError) as exc_info:
        await cleaner.clean(_draft("raw"))

    assert exc_info.value.reason is CleanerFailureReason.LEASE_LOSS
