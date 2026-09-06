from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

NoteQualityStatus = Literal[
    "semantic_complete",
    "fallback",
    "legacy",
    "unknown",
    "missing",
]

_METADATA_FILES = (
    "generation_manifest.json",
    "generation_audit.json",
    "note_budget.json",
)
_MAX_JSON_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NoteBundleState:
    quality_status: NoteQualityStatus
    provider: str | None = None
    model: str | None = None
    quality_gate: str | None = None
    semantic_complete: bool | None = None
    generator_version: str | None = None
    legacy_eligible: bool = False
    coverage: dict[str, Any] | None = None

    @property
    def upgrade_eligible(self) -> bool:
        return self.quality_status in {"legacy", "fallback", "unknown"}


def classify_note_bundle(note_dir: Path) -> NoteBundleState:
    if not (note_dir / "learning_note.md").is_file():
        return NoteBundleState(quality_status="missing")

    coverage = _read_json_object(note_dir / "coverage_report.json")
    metadata = next(
        (
            parsed
            for name in _METADATA_FILES
            if (parsed := _read_json_object(note_dir / name)) is not None
        ),
        None,
    )
    if metadata is None:
        return NoteBundleState(quality_status="unknown", coverage=coverage)

    provider = _string_value(metadata, "provider", "provider_name")
    model = _string_value(metadata, "model", "model_name")
    quality_gate = _string_value(metadata, "quality_gate")
    semantic_complete = _boolean_value(metadata, "semantic_complete")
    generator_version = _string_value(metadata, "generator_version", "version")
    identities = _string_values(
        metadata,
        "model",
        "model_name",
        "generator",
        "processor",
    )

    common = {
        "provider": provider,
        "model": model,
        "quality_gate": quality_gate,
        "semantic_complete": semantic_complete,
        "generator_version": generator_version,
        "coverage": coverage,
    }
    normalized_gate = _normalized(quality_gate)
    normalized_provider = _normalized(provider)
    if normalized_gate == "fallback" or _is_rule_provider(normalized_provider):
        return NoteBundleState(quality_status="fallback", **common)
    if any(_normalized(value) == "local-rule-v1" for value in identities):
        return NoteBundleState(
            quality_status="legacy",
            legacy_eligible=True,
            **common,
        )

    semantic_identity = any(
        _is_semantic_identity(value) for value in (provider, model, *identities)
    )
    rule_identity = any(_is_rule_identity(value) for value in (provider, model, *identities))
    if (
        normalized_gate == "passed"
        and semantic_identity
        and not rule_identity
        and semantic_complete is True
        and _coverage_is_semantic_complete(coverage)
    ):
        return NoteBundleState(quality_status="semantic_complete", **common)
    return NoteBundleState(quality_status="unknown", **common)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _containers(metadata: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    generation = metadata.get("generation")
    if isinstance(generation, dict):
        return metadata, generation
    return (metadata,)


def _string_value(metadata: dict[str, Any], *keys: str) -> str | None:
    for container in _containers(metadata):
        for key in keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _string_values(metadata: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for container in _containers(metadata):
        for key in keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value)
    return values


def _boolean_value(metadata: dict[str, Any], key: str) -> bool | None:
    for container in _containers(metadata):
        value = container.get(key)
        if isinstance(value, bool):
            return value
    return None


def _coverage_is_semantic_complete(coverage: dict[str, Any] | None) -> bool:
    """Require the additive quality gates before claiming completeness.

    A manifest is user-visible metadata and can outlive a failed or partial
    publication attempt.  The coverage report is the authoritative record of
    the source/depth/input checks, so missing fields intentionally fail closed
    and remain eligible for a future upgrade.
    """

    if not isinstance(coverage, dict):
        return False
    return (
        _normalized(coverage.get("quality_gate")) == "passed"
        and coverage.get("depth_passed") is True
        and coverage.get("input_sufficient") is True
        and coverage.get("semantic_complete") is True
    )


def _normalized(value: str | None) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _is_semantic_identity(value: str | None) -> bool:
    normalized = _normalized(value)
    return normalized in {"ollama", "external"} or normalized.startswith(
        ("ollama/", "external/")
    )


def _is_rule_identity(value: str | None) -> bool:
    normalized = _normalized(value)
    return normalized in {"rule", "local-rule"} or normalized.startswith(
        ("rule/", "local-rule", "rule-")
    )


def _is_rule_provider(normalized_provider: str) -> bool:
    return normalized_provider in {"rule", "local-rule"} or normalized_provider.startswith(
        ("rule/", "local-rule")
    )


__all__ = ["NoteBundleState", "NoteQualityStatus", "classify_note_bundle"]
