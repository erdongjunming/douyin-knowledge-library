from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Protocol

import keyring


class CookieStoreUnavailableError(RuntimeError):
    pass


class KeyringBackend(Protocol):
    def set_password(self, service_name: str, username: str, password: str) -> None: ...

    def get_password(self, service_name: str, username: str) -> str | None: ...

    def delete_password(self, service_name: str, username: str) -> None: ...


@dataclass(frozen=True)
class CookieManifest:
    version: int
    generation: str
    chunk_count: int
    sha256: str


class CookieStore:
    service_name = "douyin-knowledge-library"
    username = "douyin-cookie"
    manifest_username = "douyin-cookie-manifest"
    chunk_prefix = "douyin-cookie-chunk"
    chunk_size = 900
    manifest_version = 1

    def __init__(self, keyring_backend: KeyringBackend | None = None) -> None:
        self._keyring = keyring_backend or keyring

    def save(self, cookie: str) -> None:
        generation = uuid.uuid4().hex
        chunks = [
            cookie[index : index + self.chunk_size]
            for index in range(0, len(cookie), self.chunk_size)
        ] or [""]
        manifest = CookieManifest(
            version=self.manifest_version,
            generation=generation,
            chunk_count=len(chunks),
            sha256=hashlib.sha256(cookie.encode("utf-8")).hexdigest(),
        )
        try:
            old_manifest = self._read_manifest_optional()
            for index, chunk in enumerate(chunks):
                self._keyring.set_password(
                    self.service_name,
                    self.chunk_username(generation, index),
                    chunk,
                )
            self._verify_generation(manifest)
            self._keyring.set_password(
                self.service_name,
                self.manifest_username,
                self._serialize_manifest(manifest),
            )
        except Exception as exc:  # noqa: BLE001
            self._cleanup_generation(generation, len(chunks))
            if isinstance(exc, CookieStoreUnavailableError):
                raise
            raise CookieStoreUnavailableError(
                "Windows credential store is unavailable"
            ) from exc
        self._delete_best_effort(self.username)
        if old_manifest is not None and old_manifest.generation != generation:
            self._cleanup_generation(old_manifest.generation, old_manifest.chunk_count)

    def read(self) -> str | None:
        try:
            manifest = self._read_manifest_optional()
            if manifest is None:
                return self._keyring.get_password(self.service_name, self.username)
            chunks = [
                self._keyring.get_password(
                    self.service_name,
                    self.chunk_username(manifest.generation, index),
                )
                for index in range(manifest.chunk_count)
            ]
            if any(chunk is None for chunk in chunks):
                raise CookieStoreUnavailableError(
                    "Windows credential store is unavailable"
                )
            cookie = "".join(chunk for chunk in chunks if chunk is not None)
            if hashlib.sha256(cookie.encode("utf-8")).hexdigest() != manifest.sha256:
                raise CookieStoreUnavailableError(
                    "Windows credential store is unavailable"
                )
            return cookie
        except CookieStoreUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CookieStoreUnavailableError(
                "Windows credential store is unavailable"
            ) from exc

    def delete(self) -> None:
        try:
            manifest = self._read_manifest_optional()
        except CookieStoreUnavailableError:
            raise
        if manifest is not None:
            self._cleanup_generation(manifest.generation, manifest.chunk_count)
            self._delete_best_effort(self.manifest_username)
        self._delete_best_effort(self.username)

    @classmethod
    def chunk_username(cls, generation: str, index: int) -> str:
        return f"{cls.chunk_prefix}-{generation}-{index:04d}"

    def _read_manifest_optional(self) -> CookieManifest | None:
        try:
            raw = self._keyring.get_password(self.service_name, self.manifest_username)
        except Exception as exc:  # noqa: BLE001
            raise CookieStoreUnavailableError(
                "Windows credential store is unavailable"
            ) from exc
        if raw is None:
            return None
        return self._parse_manifest(raw)

    def _parse_manifest(self, raw: str) -> CookieManifest:
        try:
            payload = json.loads(raw)
            version = payload["version"]
            generation = payload["generation"]
            chunk_count = payload["chunk_count"]
            digest = payload["sha256"]
            if (
                version != self.manifest_version
                or not isinstance(generation, str)
                or not re.fullmatch(r"[A-Za-z0-9]+", generation)
                or not isinstance(chunk_count, int)
                or not 1 <= chunk_count <= 4096
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("invalid cookie manifest")
            return CookieManifest(version, generation, chunk_count, digest)
        except Exception as exc:  # noqa: BLE001
            raise CookieStoreUnavailableError(
                "Windows credential store is unavailable"
            ) from exc

    def _serialize_manifest(self, manifest: CookieManifest) -> str:
        return json.dumps(
            {
                "version": manifest.version,
                "generation": manifest.generation,
                "chunk_count": manifest.chunk_count,
                "sha256": manifest.sha256,
            },
            separators=(",", ":"),
        )

    def _verify_generation(self, manifest: CookieManifest) -> None:
        chunks = [
            self._keyring.get_password(
                self.service_name,
                self.chunk_username(manifest.generation, index),
            )
            for index in range(manifest.chunk_count)
        ]
        if any(chunk is None for chunk in chunks):
            raise CookieStoreUnavailableError(
                "Windows credential store is unavailable"
            )
        cookie = "".join(chunk for chunk in chunks if chunk is not None)
        if hashlib.sha256(cookie.encode("utf-8")).hexdigest() != manifest.sha256:
            raise CookieStoreUnavailableError(
                "Windows credential store is unavailable"
            )

    def _cleanup_generation(self, generation: str, chunk_count: int) -> None:
        for index in range(chunk_count):
            self._delete_best_effort(self.chunk_username(generation, index))

    def _delete_best_effort(self, username: str) -> None:
        try:
            self._keyring.delete_password(self.service_name, username)
        except keyring.errors.PasswordDeleteError:
            pass
        except Exception:
            pass
