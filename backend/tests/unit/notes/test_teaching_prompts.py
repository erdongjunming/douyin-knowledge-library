from __future__ import annotations

from typing import get_args

from douyin_library.notes.prompts import (
    DEPTH_REVIEW_PROMPT_VERSION,
    LEGACY_REPAIR_PROMPT_VERSION,
    OUTLINE_PROMPT_VERSION,
    PROMPT_VERSIONS,
    REPAIR_PROMPT_VERSION,
    SECTION_PROMPT_VERSION,
    SYSTEM_PROMPTS,
)
from douyin_library.notes.providers.base import NoteStage


def test_note_stage_contains_all_teaching_stages() -> None:
    assert get_args(NoteStage) == (
        "extract",
        "outline",
        "synthesize",
        "section",
        "review",
        "depth_review",
        "repair",
    )


def test_prompt_versions_and_system_prompts_cover_all_stages() -> None:
    stages = {
        "extract",
        "outline",
        "synthesize",
        "section",
        "review",
        "depth_review",
        "repair",
    }
    assert set(PROMPT_VERSIONS) >= stages
    assert set(SYSTEM_PROMPTS) >= stages
    assert PROMPT_VERSIONS["outline"] == OUTLINE_PROMPT_VERSION
    assert PROMPT_VERSIONS["section"] == SECTION_PROMPT_VERSION
    assert PROMPT_VERSIONS["depth_review"] == DEPTH_REVIEW_PROMPT_VERSION
    assert PROMPT_VERSIONS["repair"] == REPAIR_PROMPT_VERSION
    assert REPAIR_PROMPT_VERSION == "teaching-bounded-repair-v2"
    assert LEGACY_REPAIR_PROMPT_VERSION == "coverage-repair-v1"
    assert len(set(PROMPT_VERSIONS.values())) == len(PROMPT_VERSIONS)


def test_teaching_prompts_preserve_required_evidence_rules() -> None:
    teaching_prompts = (
        SYSTEM_PROMPTS["outline"],
        SYSTEM_PROMPTS["section"],
        SYSTEM_PROMPTS["depth_review"],
        SYSTEM_PROMPTS["repair"],
    )
    required_phrases = (
        "视频未提供这一类信息。",
        "atom_ids",
        "timestamps",
        "例子",
        "结果",
        "条件",
        "风险",
        "数字",
        "专名",
        "推广",
        "ASR",
        "作者原话",
        "生成理解",
        "不得发明",
        "JSON",
    )
    for prompt in teaching_prompts:
        for phrase in required_phrases:
            assert phrase in prompt

    assert "overview" in SYSTEM_PROMPTS["outline"]
    assert "evidence_limits" in SYSTEM_PROMPTS["outline"]
    assert "不修复" in SYSTEM_PROMPTS["depth_review"]
    assert "只修复" in SYSTEM_PROMPTS["repair"]


def test_legacy_prompt_constants_remain_readable() -> None:
    assert "coverage-repair-v1" == LEGACY_REPAIR_PROMPT_VERSION
    assert "evidence-extract-v1" in PROMPT_VERSIONS.values()
    assert "evidence-synthesize-v1" in PROMPT_VERSIONS.values()
    assert "raw-source-review-v1" in PROMPT_VERSIONS.values()
