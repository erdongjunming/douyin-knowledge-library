from pathlib import Path

import pytest

from douyin_library.media.ffmpeg import MediaExtractionError
from douyin_library.media.storage import MediaPipeline, StorageLayout


class FakeMediaTool:
    def extract_audio(self, source: Path, target: Path) -> dict:
        if source.read_bytes() == b"corrupt":
            raise MediaExtractionError("source has no valid audio stream")
        target.write_bytes(b"verified-audio")
        return {"copy_mode": "stream_copy"}

    def extract_cover(self, source: Path, target: Path) -> None:
        target.write_bytes(b"jpeg-cover")

    def probe(self, path: Path) -> dict:
        if path.suffix == ".mp4":
            return {
                "format": {"duration": "42.0"},
                "streams": [{"codec_type": "audio", "codec_name": "aac"}],
            }
        return {
            "format": {"duration": "41.95"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "channels": 2,
                }
            ],
        }

    def analyze_loudness(self, path: Path) -> dict:
        return {"integrated_loudness_lufs": -16.2, "true_peak_dbfs": -1.1}


def test_video_deleted_only_after_verified_audio(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / "library")
    layout.initialize()
    source = tmp_path / "sample_voice.mp4"
    source.write_bytes(b"valid-video")

    result = MediaPipeline(layout, FakeMediaTool()).process("731", source)

    assert result.audio.verified_at is not None
    assert result.audio.sha256
    assert result.cover.exists()
    assert result.metadata.exists()
    assert not source.exists()


def test_failed_audio_keeps_source_video(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / "library")
    layout.initialize()
    source = tmp_path / "corrupt.mp4"
    source.write_bytes(b"corrupt")

    with pytest.raises(MediaExtractionError):
        MediaPipeline(layout, FakeMediaTool()).process("731", source)

    assert source.exists()
