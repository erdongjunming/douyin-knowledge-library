from math import inf, nan
from pathlib import Path

from douyin_library.config import Settings
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Work
from douyin_library.transcription.contracts import TranscriptSegment
from douyin_library.transcription.providers.subtitle_local import (
    LocalSubtitleProvider,
    normalize_subtitle_text,
    parse_subtitle,
    score_subtitle,
)


def test_parse_srt_removes_cues_and_deduplicates() -> None:
    raw = (
        "1\n00:00:00,000 --> 00:00:01,000\n你好，世界。\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n你好，世界。"
    )
    segments = parse_subtitle(raw, ".srt")
    assert len(segments) == 1
    assert segments[0].text == "你好，世界。"
    assert normalize_subtitle_text(raw, ".srt") == "你好，世界。"


def test_parse_vtt_ass_and_plain_text() -> None:
    vtt = (
        "WEBVTT\n\n00:00.000 --> 00:01.000\n第一句\n\nNOTE\nignore\n\n"
        "00:01.000 --> 00:02.000\n第二句"
    )
    assert [s.text for s in parse_subtitle(vtt, ".vtt")] == ["第一句", "第二句"]
    ass = "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\an8}字幕\\N第二行"
    assert parse_subtitle(ass, ".ass")[0].text == "字幕\n第二行"
    plain = parse_subtitle("  一句有效字幕  \n\n第二行", ".txt")
    assert [s.text for s in plain] == ["一句有效字幕", "第二行"]
    assert all(s.start is None for s in plain)
    ssa = "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,SSA字幕"
    assert parse_subtitle(ssa, ".ssa")[0].text == "SSA字幕"


def test_bad_empty_and_placeholder_subtitles_fail_quality() -> None:
    assert not score_subtitle(parse_subtitle("", ".srt")).usable
    bad = parse_subtitle("1\nnot-a-time --> 00:01:00,000\nhello", ".srt")
    quality = score_subtitle(bad)
    assert not quality.timestamp_valid
    assert not quality.usable
    placeholder = parse_subtitle("[音乐]\n字幕\n", ".txt")
    assert score_subtitle(placeholder).placeholder_ratio > 0


def test_non_finite_timestamps_are_rejected() -> None:
    parsed = parse_subtitle("nan --> inf\n不可用", ".vtt")
    assert not score_subtitle(parsed).timestamp_valid
    direct = (TranscriptSegment("x", nan, inf, "不可用"),)
    quality = score_subtitle(direct)
    assert not quality.timestamp_valid
    assert not quality.usable


def test_one_real_sentence_short_video_is_usable() -> None:
    quality = score_subtitle(
        parse_subtitle("00:00:00.000 --> 00:00:01.000\n今天学习一个方法", ".vtt"), duration_ms=1000
    )
    assert quality.usable


def test_mixed_timestamp_segments_are_invalid_and_quality_has_stable_reason() -> None:
    segments = (
        parse_subtitle("00:00.000 --> 00:01.000\n有时间", ".vtt")[0],
        parse_subtitle("无时间", ".txt")[0],
    )
    quality = score_subtitle(segments)
    assert not quality.timestamp_valid
    assert not quality.usable
    assert quality.reason == "mixed_timestamp"
    assert quality.segment_count == 2
    assert quality.text_chars == 6


def test_duration_changes_minimum_text_threshold() -> None:
    one_char = parse_subtitle("短", ".txt")
    assert score_subtitle(one_char, duration_ms=1_000).usable
    assert not score_subtitle(one_char, duration_ms=120_000).usable


def test_configured_subtitle_root_is_used_without_duplicate_subtitle_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "captions"
    path = root / "42" / "subtitle.ssa"
    path.parent.mkdir(parents=True)
    path.write_text(
        "[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,配置字幕",
        encoding="utf-8",
    )
    settings = Settings(storage_root=tmp_path, transcription_subtitle_root=root)
    draft = LocalSubtitleProvider(settings).discover(Work(aweme_id="42", duration_ms=1000))
    assert draft is not None
    assert draft.text == "配置字幕"


def test_local_provider_uses_first_usable_candidate(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    subtitle_dir = layout.subtitle_dir("42")
    subtitle_dir.mkdir(parents=True)
    (subtitle_dir / "subtitle.srt").write_text("", encoding="utf-8")
    (subtitle_dir / "subtitle.vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\n真实字幕", encoding="utf-8"
    )
    work = Work(aweme_id="42", duration_ms=1000)
    draft = LocalSubtitleProvider(type("Settings", (), {"storage_root": tmp_path})()).discover(work)
    assert draft is not None
    assert draft.provider == "subtitle_local"
    assert draft.text == "真实字幕"
