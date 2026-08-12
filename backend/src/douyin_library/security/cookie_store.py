from __future__ import annotations

from typing import Protocol

import keyring


class CookieStoreUnavailableError(RuntimeError):
    pass


class KeyringBackend(Protocol):
    def set_password(self, service_name: str, username: str, password: str) -> None: ...

    def get_password(self, service_name: str, username: str) -> str | None: ...

    def delete_password(self, service_name: str, username: str) -> None: ...


class CookieStore:
    service_name = "douyin-knowledge-library"
    username = "douyin-cookie"

    def __init__(self, keyring_backend: KeyringBackend | None = None) -> None:
        self._keyring = keyring_backend or keyring

    def save(self, cookie: str) -> None:
        try:
            self._keyring.set_password(self.service_name, self.username, cookie)
        except Exception as exc:  # noqa: BLE001
            raise CookieStoreUnavailableError(
                "Windows credential store is unavailable"
            ) from exc

    def read(self) -> str | None:
        return self._keyring.get_password(self.service_name, self.username)

    def delete(self) -> None:
        try:
            self._keyring.delete_password(self.service_name, self.username)
        except keyring.errors.PasswordDeleteError:
            pass
