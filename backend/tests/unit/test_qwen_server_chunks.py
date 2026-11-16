import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("tail_samples", [0, 64, 160, 8000])
def test_chunking_preserves_all_samples_without_tiny_tail(tail_samples: int) -> None:
    path = Path(__file__).resolve().parents[3] / "scripts" / "qwen_asr_server.py"
    spec = importlib.util.spec_from_file_location("qwen_server_chunks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rate = 16000
    data = range(120 * rate + tail_samples)
    chunks = {}

    def write(path, samples, sample_rate):
        assert sample_rate == rate
        chunks[path] = samples

    def transcribe(*, audio, language):
        # Qwen's short-input path does not pad for STFT. The previous
        # implementation sent the final 64/160 samples as a separate WAV.
        assert len(chunks[audio]) >= rate // 2
        return [{"text": "recognized", "language": language}]

    result = module._transcribe_chunks(
        SimpleNamespace(transcribe=transcribe), Path("audio.wav"), "Chinese", 60,
        SimpleNamespace(read=lambda *a, **kw: (data, rate), write=write),
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    flattened = [sample for chunk in chunks.values() for sample in chunk]
    assert flattened == list(data)
    assert result[0]["start"] == 0
    assert result[-1]["end"] == round(len(data) / rate, 3)
    assert all(a["end"] == b["start"] for a, b in zip(result, result[1:], strict=False))
