import pytest
from pydantic import SecretStr

from douyin_library.config import Settings
from douyin_library.notes.providers.selector import (
    external_is_configured,
    select_provider_names,
)


def _settings(**overrides) -> Settings:
    values = {
        "note_external_base_url": "https://api.example.test/v1",
        "note_external_model": "example-model",
        "note_external_api_key": SecretStr("top-secret-key"),
    }
    values.update(overrides)
    return Settings(**values)


def test_provider_policy_chains_are_explicit_and_keep_rule_fallback() -> None:
    settings = _settings()

    assert select_provider_names(settings, "auto") == ["ollama", "external", "rule"]
    assert select_provider_names(settings, "local") == ["ollama", "rule"]
    assert select_provider_names(settings, "external") == ["external", "rule"]
    assert select_provider_names(settings, "rule") == ["rule"]


def test_settings_policy_is_used_when_override_is_absent() -> None:
    settings = _settings(note_provider_policy="local")

    assert select_provider_names(settings) == ["ollama", "rule"]


@pytest.mark.parametrize("override", ["", "AUTO", "invalid"])
def test_explicit_invalid_policy_override_is_rejected(override: str) -> None:
    settings = _settings(note_provider_policy="local")

    with pytest.raises(ValueError, match="invalid_note_provider_policy"):
        select_provider_names(settings, override)  # type: ignore[arg-type]


def test_auto_adds_external_only_for_complete_explicit_configuration() -> None:
    complete = _settings()
    incomplete = (
        _settings(note_external_base_url=None),
        _settings(note_external_model=None),
        _settings(note_external_api_key=None),
        _settings(note_external_base_url="  "),
        _settings(note_external_model="\t"),
        _settings(note_external_api_key=SecretStr("  \t")),
    )

    assert external_is_configured(complete) is True
    for settings in incomplete:
        assert external_is_configured(settings) is False
        assert select_provider_names(settings, "auto") == ["ollama", "rule"]


@pytest.mark.parametrize(
    ("base_url", "expected_configured"),
    [
        ("https://api.example.test/v1", True),
        ("http://localhost:8000/v1", True),
        ("http://127.0.0.1:8000/v1", True),
        ("http://[::1]:8000/v1", True),
        ("http://api.example.test/v1", False),
        ("http://[::1/url-secret", False),
        ("http://127.0.0.1:url-secret", False),
        ("https://api.example.test:url-secret/v1", False),
        ("api.example.test/url-secret", False),
        ("https:///url-secret", False),
    ],
)
def test_external_configuration_uses_provider_url_safety_rule(
    base_url: str,
    expected_configured: bool,
) -> None:
    settings = _settings(note_external_base_url=base_url)

    assert external_is_configured(settings) is expected_configured
    expected_names = (
        ["ollama", "external", "rule"]
        if expected_configured
        else ["ollama", "rule"]
    )
    assert select_provider_names(settings, "auto") == expected_names


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.test/v1",
        "http://[::1/url-secret",
        "http://127.0.0.1:url-secret",
    ],
)
def test_explicit_external_policy_preserves_unsafe_operator_selection(
    base_url: str,
) -> None:
    settings = _settings(note_external_base_url=base_url)

    assert select_provider_names(settings, "external") == ["external", "rule"]


def test_local_policy_never_uses_external_even_when_configured() -> None:
    assert select_provider_names(_settings(), "local") == ["ollama", "rule"]


def test_external_policy_preserves_operator_intent_when_configuration_is_missing() -> None:
    settings = _settings(
        note_external_base_url=None,
        note_external_model=None,
        note_external_api_key=None,
    )

    assert select_provider_names(settings, "external") == ["external", "rule"]


def test_selection_does_not_expose_external_secret() -> None:
    secret = "selection-secret-must-not-leak"
    settings = _settings(note_external_api_key=SecretStr(secret))

    result = select_provider_names(settings)

    assert secret not in repr(result)
    assert secret not in repr(settings)
