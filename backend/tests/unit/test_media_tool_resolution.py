from pathlib import Path

from douyin_library.config import Settings
from douyin_library.media.ffmpeg import FfmpegMediaTool


def test_media_tool_uses_bundled_ffmpeg_paths(monkeypatch, tmp_path: Path) -> None:
    bin_dir = tmp_path / "tools" / "ffmpeg" / "ffmpeg-build" / "bin"
    bin_dir.mkdir(parents=True)
    ffmpeg = bin_dir / "ffmpeg.exe"
    ffprobe = bin_dir / "ffprobe.exe"
    ffmpeg.write_bytes(b"exe")
    ffprobe.write_bytes(b"exe")
    monkeypatch.setattr("douyin_library.config.shutil.which", lambda _: None)

    tool = FfmpegMediaTool.from_settings(Settings(storage_root=tmp_path))

    assert tool.ffmpeg == str(ffmpeg)
    assert tool.ffprobe == str(ffprobe)
