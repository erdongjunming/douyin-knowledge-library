from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from douyin_library.config import Settings


def test_default_storage_root_is_d_drive(monkeypatch) -> None:
    monkeypatch.delenv("DKL_STORAGE_ROOT", raising=False)

    settings = Settings()

    assert settings.storage_root == Path(r"D:\DouyinKnowledgeLibrary")
    assert settings.transcription_subtitle_root == Path(r"D:\DouyinKnowledgeLibrary\subtitle")


def test_note_provider_defaults(monkeypatch) -> None:
    for name in (
        "DKL_NOTE_PROVIDER_POLICY",
        "DKL_NOTE_OLLAMA_URL",
        "DKL_NOTE_OLLAMA_MODEL",
        "DKL_NOTE_OLLAMA_EXECUTABLE",
        "DKL_NOTE_OLLAMA_AUTOSTART",
        "DKL_NOTE_OLLAMA_PROBE_TIMEOUT_SECONDS",
        "DKL_NOTE_OLLAMA_UNLOAD_TIMEOUT_SECONDS",
        "DKL_NOTE_REQUEST_TIMEOUT_SECONDS",
        "DKL_NOTE_CONTEXT_TOKENS",
        "DKL_NOTE_OLLAMA_KEEP_ALIVE",
        "DKL_NOTE_EXTERNAL_BASE_URL",
        "DKL_NOTE_EXTERNAL_MODEL",
        "DKL_NOTE_EXTERNAL_API_KEY",
        "DKL_NOTE_EXTERNAL_INPUT_COST_PER_MILLION",
        "DKL_NOTE_EXTERNAL_OUTPUT_COST_PER_MILLION",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.note_provider_policy == "auto"
    assert settings.note_ollama_url == "http://127.0.0.1:11434"
    assert settings.note_ollama_model == "qwen3:4b"
    assert settings.note_ollama_executable is None
    assert settings.note_ollama_autostart is True
    assert settings.note_ollama_probe_timeout_seconds == 2
    assert settings.note_ollama_unload_timeout_seconds == 10
    assert settings.note_request_timeout_seconds == 600
    assert settings.note_context_tokens == 32768
    assert settings.note_ollama_keep_alive == "10m"
    assert settings.note_external_base_url is None
    assert settings.note_external_model is None
    assert settings.note_external_api_key is None
    assert settings.note_external_input_cost_per_million is None
    assert settings.note_external_output_cost_per_million is None


def test_note_provider_settings_load_from_prefixed_environment(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "ollama.exe"
    monkeypatch.setenv("DKL_NOTE_PROVIDER_POLICY", "external")
    monkeypatch.setenv("DKL_NOTE_OLLAMA_EXECUTABLE", str(executable))
    monkeypatch.setenv("DKL_NOTE_OLLAMA_AUTOSTART", "false")
    monkeypatch.setenv("DKL_NOTE_OLLAMA_PROBE_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("DKL_NOTE_OLLAMA_UNLOAD_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("DKL_NOTE_REQUEST_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("DKL_NOTE_CONTEXT_TOKENS", "65536")
    monkeypatch.setenv("DKL_NOTE_OLLAMA_KEEP_ALIVE", "  5m  ")
    monkeypatch.setenv("DKL_NOTE_EXTERNAL_API_KEY", "environment-secret")
    monkeypatch.setenv("DKL_NOTE_EXTERNAL_INPUT_COST_PER_MILLION", "0.25")
    monkeypatch.setenv("DKL_NOTE_EXTERNAL_OUTPUT_COST_PER_MILLION", "1.5")

    settings = Settings()

    assert settings.note_provider_policy == "external"
    assert settings.note_ollama_executable == executable
    assert settings.note_ollama_autostart is False
    assert settings.note_ollama_probe_timeout_seconds == 3.5
    assert settings.note_ollama_unload_timeout_seconds == 0.25
    assert settings.note_request_timeout_seconds == 120
    assert settings.note_context_tokens == 65536
    assert settings.note_ollama_keep_alive == "5m"
    assert settings.note_external_api_key is not None
    assert settings.note_external_api_key.get_secret_value() == "environment-secret"
    assert settings.note_external_input_cost_per_million == 0.25
    assert settings.note_external_output_cost_per_million == 1.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.01])
@pytest.mark.parametrize(
    "field",
    [
        "note_external_input_cost_per_million",
        "note_external_output_cost_per_million",
    ],
)
def test_external_cost_rates_must_be_finite_and_nonnegative(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("note_ollama_probe_timeout_seconds", 0.19),
        ("note_ollama_probe_timeout_seconds", 30.01),
        ("note_ollama_unload_timeout_seconds", 0.049),
        ("note_ollama_unload_timeout_seconds", 60.01),
        ("note_request_timeout_seconds", 9.9),
        ("note_request_timeout_seconds", 3600.1),
        ("note_context_tokens", 4095),
        ("note_context_tokens", 131073),
    ],
)
def test_note_provider_numeric_settings_enforce_bounds(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_ollama_keep_alive_is_stripped_and_must_not_be_empty() -> None:
    assert Settings(note_ollama_keep_alive="  20m  ").note_ollama_keep_alive == "20m"

    with pytest.raises(ValidationError):
        Settings(note_ollama_keep_alive=" \t ")


def test_external_secret_is_masked_by_repr_and_model_dump() -> None:
    secret = "settings-secret-must-not-leak"
    settings = Settings(note_external_api_key=SecretStr(secret))

    assert secret not in repr(settings)
    assert secret not in str(settings.model_dump())
    assert secret not in settings.model_dump_json()
    assert "**********" in settings.model_dump_json()
