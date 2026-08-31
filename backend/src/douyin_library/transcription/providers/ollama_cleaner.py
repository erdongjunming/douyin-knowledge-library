"""Deterministic local Ollama transcript cleaning."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import httpx

from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.notes.providers.base import request_or_cancel, sanitized_base_url
from douyin_library.transcription.contracts import (
    CleanedTranscript,
    CleanerResult,
    TranscriptDraft,
    TranscriptSegment,
)


class CleanerFailureReason(StrEnum):
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    SCHEMA = "schema"
    DRIFT = "drift"
    CANCELLATION = "cancellation"
    LEASE_LOSS = "lease_loss"


class TranscriptCleaningError(RuntimeError):
    def __init__(self, reason: CleanerFailureReason, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"transcript_cleaning_{reason.value}" + (f":{detail}" if detail else ""))


ClientFactory = Callable[..., httpx.AsyncClient]
LeaseChecker = Callable[[], bool]


class OllamaTranscriptCleaner:
    """Clean transcript text while preserving immutable segment identity."""

    name = "ollama_cleaner"
    is_local = True

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        client_factory: ClientFactory | None = None,
        lease_checker: LeaseChecker | None = None,
    ) -> None:
        base_url = sanitized_base_url(settings.note_ollama_url)
        model = settings.transcription_cleaner_model.strip()
        if not base_url or not model:
            raise ValueError("transcription_cleaner_configuration_incomplete")
        self.base_url = base_url
        self.model = model
        self.keep_alive = settings.transcription_cleaner_keep_alive
        self.thinking = settings.transcription_cleaner_thinking
        self.timeout = settings.transcription_cleaner_timeout_seconds
        self.batch_limit = settings.transcription_cleaner_batch_character_limit
        self.drift_ratio = settings.transcription_cleaner_drift_ratio
        self.prompt_version = settings.transcription_cleaner_prompt_version
        self._transport = transport
        self._client_factory = client_factory
        self._lease_checker = lease_checker

    async def clean(
        self,
        transcript: TranscriptDraft,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CleanerResult:
        cancel_event = cancel_event or asyncio.Event()
        if cancel_event.is_set():
            raise TranscriptCleaningError(CleanerFailureReason.CANCELLATION)
        if self._lease_checker is not None and not self._lease_checker():
            raise TranscriptCleaningError(CleanerFailureReason.LEASE_LOSS)
        if not transcript.segments:
            raise TranscriptCleaningError(CleanerFailureReason.SCHEMA, "empty_transcript")

        originals = {segment.segment_id: segment for segment in transcript.segments}
        if len(originals) != len(transcript.segments) or any(
            not isinstance(segment.segment_id, str) or not segment.segment_id
            for segment in transcript.segments
        ):
            raise TranscriptCleaningError(CleanerFailureReason.SCHEMA, "duplicate_or_invalid_id")
        chunks = _batch_segments(transcript.segments, self.batch_limit)
        cleaned_by_id: dict[str, TranscriptSegment] = {}
        for chunk in chunks:
            if self._lease_checker is not None and not self._lease_checker():
                raise TranscriptCleaningError(CleanerFailureReason.LEASE_LOSS)
            output = await self._clean_chunk(chunk, cancel_event)
            if cancel_event.is_set():
                raise TranscriptCleaningError(CleanerFailureReason.CANCELLATION)
            if self._lease_checker is not None and not self._lease_checker():
                raise TranscriptCleaningError(CleanerFailureReason.LEASE_LOSS)
            for segment in output:
                if segment.segment_id in cleaned_by_id:
                    previous = cleaned_by_id[segment.segment_id]
                    if previous.text != segment.text:
                        raise TranscriptCleaningError(CleanerFailureReason.SCHEMA, "duplicate_id")
                    continue
                cleaned_by_id[segment.segment_id] = segment

        if cancel_event.is_set():
            raise TranscriptCleaningError(CleanerFailureReason.CANCELLATION)
        if self._lease_checker is not None and not self._lease_checker():
            raise TranscriptCleaningError(CleanerFailureReason.LEASE_LOSS)
        if tuple(cleaned_by_id) != tuple(originals):
            raise TranscriptCleaningError(CleanerFailureReason.SCHEMA, "missing_or_extra_id")
        segments = tuple(cleaned_by_id[segment.segment_id] for segment in transcript.segments)
        cleaned = CleanedTranscript(
            segments=segments,
            text="\n".join(segment.text for segment in segments),
            language=transcript.language,
        )
        return CleanerResult(
            transcript=cleaned,
            usable=True,
            text_chars=len(cleaned.text),
            segment_count=len(segments),
            timestamp_valid=True,
            placeholder_ratio=0.0,
        )

    async def _clean_chunk(
        self,
        segments: tuple[TranscriptSegment, ...],
        cancel_event: asyncio.Event,
    ) -> tuple[TranscriptSegment, ...]:
        payload = {
            "segments": [
                {"id": s.segment_id, "start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ]
        }
        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Edit transcript text only. Return JSON object with segments. "
                        f"Prompt version: {self.prompt_version}."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "stream": False,
            "format": "json",
            # Cleaning is intentionally deterministic; the config field is
            # retained for compatibility but this provider always disables
            # Ollama reasoning for transcript edits.
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0},
        }
        try:
            request = self._post(request_body)
            response = await asyncio.wait_for(
                request_or_cancel(request, cancel_event), timeout=self.timeout
            )
        except JobCancelledError as exc:
            raise TranscriptCleaningError(CleanerFailureReason.CANCELLATION) from exc
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise TranscriptCleaningError(CleanerFailureReason.TIMEOUT) from exc
        except TranscriptCleaningError:
            raise
        except Exception as exc:
            raise TranscriptCleaningError(CleanerFailureReason.TRANSPORT) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TranscriptCleaningError(CleanerFailureReason.TRANSPORT) from exc
        try:
            body = response.json()
            raw_content = body["message"]["content"]
            parsed = json.loads(raw_content, parse_constant=_reject_constant)
        except Exception as exc:
            raise TranscriptCleaningError(CleanerFailureReason.SCHEMA) from exc
        return _validate_chunk(parsed, segments, self.drift_ratio)

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        async with self._make_client() as client:
            return await client.post(f"{self.base_url}/api/chat", json=body)

    def _make_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(transport=self._transport, timeout=self.timeout, trust_env=False)


def _batch_segments(
    segments: tuple[TranscriptSegment, ...], limit: int
) -> tuple[tuple[TranscriptSegment, ...], ...]:
    batches: list[tuple[TranscriptSegment, ...]] = []
    index = 0
    while index < len(segments):
        total = 0
        end = index
        while end < len(segments):
            cost = len(segments[end].text) + (1 if end > index else 0)
            if end > index and total + cost > limit:
                break
            total += cost
            end += 1
        batches.append(segments[index:end])
        if end >= len(segments):
            break
        # Repeat the final segment as overlap when the batch had at least two
        # segments; always advance for an oversized single segment.
        index = end - 1 if end - index > 1 else end
    return tuple(batches)


def _validate_chunk(
    parsed: Any,
    expected: tuple[TranscriptSegment, ...],
    drift_ratio: float,
) -> tuple[TranscriptSegment, ...]:
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"segments"}
        or not isinstance(parsed.get("segments"), list)
    ):
        raise TranscriptCleaningError(CleanerFailureReason.SCHEMA)
    values = parsed["segments"]
    if len(values) != len(expected):
        raise TranscriptCleaningError(CleanerFailureReason.SCHEMA, "segment_count")
    result: list[TranscriptSegment] = []
    for item, original in zip(values, expected, strict=True):
        if not isinstance(item, dict) or set(item) != {"id", "start", "end", "text"}:
            raise TranscriptCleaningError(CleanerFailureReason.SCHEMA)
        if (
            item.get("id") != original.segment_id
            or item.get("start") != original.start
            or item.get("end") != original.end
        ):
            raise TranscriptCleaningError(CleanerFailureReason.SCHEMA, "segment_identity")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptCleaningError(CleanerFailureReason.SCHEMA, "empty_text")
        baseline = max(len(original.text), 1)
        if abs(len(text) - len(original.text)) / baseline > drift_ratio:
            raise TranscriptCleaningError(CleanerFailureReason.DRIFT)
        result.append(TranscriptSegment(original.segment_id, original.start, original.end, text))
    return tuple(result)


def _reject_constant(value: str) -> None:
    raise ValueError(value)


__all__ = [
    "CleanerFailureReason",
    "OllamaTranscriptCleaner",
    "TranscriptCleaningError",
]
