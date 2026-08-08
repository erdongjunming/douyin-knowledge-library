from douyin_library.media.metadata import can_delete_source_video


def test_cleanup_requires_verified_nonempty_audio_with_matching_duration() -> None:
    assert can_delete_source_video(
        source_duration=42.0,
        audio_duration=41.95,
        audio_bytes=1024,
        sha256="a" * 64,
        verified=True,
    )
    assert not can_delete_source_video(
        source_duration=42.0,
        audio_duration=30.0,
        audio_bytes=1024,
        sha256="a" * 64,
        verified=True,
    )
    assert not can_delete_source_video(
        source_duration=42.0,
        audio_duration=41.95,
        audio_bytes=0,
        sha256="",
        verified=False,
    )
