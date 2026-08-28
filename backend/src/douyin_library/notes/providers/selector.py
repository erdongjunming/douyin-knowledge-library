from __future__ import annotations

from typing import Literal

from douyin_library.config import Settings
from douyin_library.notes.providers.base import validated_external_base_url

ProviderName = Literal["ollama", "external", "rule"]
ProviderPolicy = Literal["auto", "local", "external", "rule"]


def external_is_configured(settings: Settings) -> bool:
    base_url = settings.note_external_base_url
    model = settings.note_external_model
    api_key = settings.note_external_api_key
    return bool(
        base_url
        and validated_external_base_url(base_url) is not None
        and model
        and model.strip()
        and api_key is not None
        and api_key.get_secret_value().strip()
    )


def select_provider_names(
    settings: Settings,
    policy: ProviderPolicy | None = None,
) -> list[ProviderName]:
    selected_policy = settings.note_provider_policy if policy is None else policy
    if selected_policy == "auto":
        names: list[ProviderName] = ["ollama"]
        if external_is_configured(settings):
            names.append("external")
        names.append("rule")
        return names
    if selected_policy == "local":
        return ["ollama", "rule"]
    if selected_policy == "external":
        return ["external", "rule"]
    if selected_policy == "rule":
        return ["rule"]
    raise ValueError("invalid_note_provider_policy")


__all__ = ["ProviderName", "external_is_configured", "select_provider_names"]
