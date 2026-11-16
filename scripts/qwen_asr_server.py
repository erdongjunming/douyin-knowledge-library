"""Local JSON-lines server that keeps one Qwen3-ASR model loaded."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[requested]


def _device_map(torch: Any, requested: str) -> str:
    return "cuda:0" if requested == "auto" and torch.cuda.is_available() else (
        "cpu" if requested == "auto" else requested
    )


def _transcribe_one(model: Any, audio: Path, language: str) -> dict[str, Any]:
    result = model.transcribe(audio=str(audio), language=language)
    first = result[0] if isinstance(result, list) and result else result
    return {
        "language": _value(first, "language", language),
        "text": str(_value(first, "text", "") or "").strip(),
    }


def _transcribe_chunks(
    model: Any,
    audio: Path,
    language: str,
    chunk_seconds: float,
    soundfile: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    data, sample_rate = soundfile.read(str(audio), always_2d=False)
    total_samples = len(data)
    duration = total_samples / float(sample_rate or 1)
    if chunk_seconds <= 0 or duration <= chunk_seconds * 1.2:
        item = _transcribe_one(model, audio, language)
        item.update({"start": 0.0, "end": round(duration, 3), "chunk_index": 1})
        return [item]

    chunk_size = max(1, int(sample_rate * chunk_seconds))
    segments: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_server_chunks_") as directory:
        for index, start in enumerate(range(0, total_samples, chunk_size), start=1):
            end = min(start + chunk_size, total_samples)
            # Qwen's short-input branch bypasses its own padding. A tiny
            # final WAV can therefore fail STFT reflection padding. Keep the
            # tail with the preceding chunk, preserving every source sample.
            if 0 < total_samples - end < sample_rate * 0.5:
                end = total_samples
            if end <= start:
                continue
            chunk = Path(directory) / f"chunk_{index:04d}.wav"
            soundfile.write(str(chunk), data[start:end], sample_rate)
            item = _transcribe_one(model, chunk, language)
            if item["text"]:
                item.update(
                    {
                        "start": round(start / sample_rate, 3),
                        "end": round(end / sample_rate, 3),
                        "chunk_index": index,
                    }
                )
                segments.append(item)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if end == total_samples:
                break
    return segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resident local Qwen3-ASR JSON-lines server")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args(argv)

    try:
        import soundfile as sf
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = _dtype(torch, args.dtype)
        device_map = _device_map(torch, args.device_map)
        model = Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=args.max_new_tokens,
            max_inference_batch_size=1,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"event": "error", "error": f"model_load_failed:{exc}"},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 3

    print(
        json.dumps(
            {
                "event": "ready",
                "backend": "qwen3-asr",
                "model": args.model,
                "device_map": device_map,
                "dtype": str(dtype).replace("torch.", ""),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for line in sys.stdin:
        request: Any = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("request_not_object")
            request_id = str(request["request_id"])
            audio = Path(str(request["audio"]))
            output = Path(str(request["out"]))
            language = str(request.get("language") or args.language)
            chunk_seconds = float(request.get("chunk_seconds", args.chunk_seconds))
            segments = _transcribe_chunks(model, audio, language, chunk_seconds, sf, torch)
            text = "\n".join(item["text"] for item in segments if item.get("text")).strip()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "model": args.model,
                        "backend": "qwen3-asr",
                        "language": segments[0].get("language", language) if segments else language,
                        "audio": str(audio),
                        "device_map": device_map,
                        "dtype": str(dtype).replace("torch.", ""),
                        "chunk_seconds": chunk_seconds,
                        "segments": segments,
                        "text": text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "request_id": request_id,
                        "returncode": 0,
                        "stdout": json.dumps({"chars": len(text)}, ensure_ascii=False),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            request_id = str(request.get("request_id", "")) if isinstance(request, dict) else ""
            print(
                json.dumps(
                    {
                        "ok": False,
                        "request_id": request_id,
                        "returncode": 1,
                        "stderr": f"qwen_inference_failed:{exc}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
