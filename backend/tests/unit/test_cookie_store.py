import json

import pytest

from douyin_library.security.cookie_store import CookieStore, CookieStoreUnavailableError


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.set_count = 0
        self.fail_set_at: int | None = None

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.set_count += 1
        if self.fail_set_at == self.set_count:
            raise OSError(1783, "credential payload rejected")
        self.values[(service_name, username)] = password

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.values.get((service_name, username))

    def delete_password(self, service_name: str, username: str) -> None:
        self.values.pop((service_name, username), None)


class UnavailableKeyring(FakeKeyring):
    def set_password(self, service_name: str, username: str, password: str) -> None:
        raise OSError(1312, "credential store unavailable")


def test_cookie_is_only_available_from_chunked_keyring() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)

    store.save("sessionid=secret")

    assert ("douyin-knowledge-library", "douyin-cookie") not in keyring.values
    assert store.read() == "sessionid=secret"


def test_cookie_can_be_deleted_from_keyring() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    store.save("sessionid=secret")

    store.delete()

    assert store.read() is None


def test_cookie_write_failure_is_wrapped_without_exposing_secret() -> None:
    store = CookieStore(keyring_backend=UnavailableKeyring())

    with pytest.raises(CookieStoreUnavailableError) as error:
        store.save("sessionid=super-secret")

    assert str(error.value) == "Windows credential store is unavailable"
    assert "super-secret" not in str(error.value)


def test_long_cookie_is_chunked_and_round_trips() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    cookie = "sessionid=" + "x" * 3500

    store.save(cookie)

    manifest_raw = keyring.values[(store.service_name, store.manifest_username)]
    manifest = json.loads(manifest_raw)
    chunk_values = [
        keyring.values[
            (
                store.service_name,
                store.chunk_username(manifest["generation"], index),
            )
        ]
        for index in range(manifest["chunk_count"])
    ]
    assert all(len(chunk) <= store.chunk_size for chunk in chunk_values)
    assert cookie not in manifest_raw
    assert store.read() == cookie


def test_legacy_single_entry_cookie_remains_readable() -> None:
    keyring = FakeKeyring()
    keyring.values[(CookieStore.service_name, CookieStore.username)] = "sessionid=legacy"

    assert CookieStore(keyring_backend=keyring).read() == "sessionid=legacy"


@pytest.mark.parametrize(
    "manifest",
    [
        {"version": 2, "generation": "abc", "chunk_count": 1, "sha256": "0" * 64},
        {"version": 1, "generation": "", "chunk_count": 1, "sha256": "0" * 64},
        {"version": 1, "generation": "a-b", "chunk_count": 1, "sha256": "0" * 64},
        {"version": 1, "generation": "abc", "chunk_count": 0, "sha256": "0" * 64},
        {"version": 1, "generation": "abc", "chunk_count": 4097, "sha256": "0" * 64},
        {"version": 1, "generation": "abc", "chunk_count": 1, "sha256": "z" * 64},
    ],
)
def test_invalid_manifest_fails_without_exposing_manifest(
    manifest: dict[str, object],
) -> None:
    keyring = FakeKeyring()
    raw = json.dumps(manifest)
    keyring.values[(CookieStore.service_name, CookieStore.manifest_username)] = raw

    with pytest.raises(CookieStoreUnavailableError) as error:
        CookieStore(keyring_backend=keyring).read()

    assert raw not in str(error.value)


def test_failed_replacement_preserves_previously_committed_cookie() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    store.save("sessionid=old")
    keyring.fail_set_at = keyring.set_count + 2

    with pytest.raises(CookieStoreUnavailableError):
        store.save("sessionid=" + "n" * 3500)

    assert store.read() == "sessionid=old"
    assert all("n" * 100 not in value for value in keyring.values.values())


def test_corrupt_chunk_fails_without_exposing_secret() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    store.save("sessionid=" + "secret" * 500)
    manifest = json.loads(keyring.values[(store.service_name, store.manifest_username)])
    first_chunk_key = (
        store.service_name,
        store.chunk_username(manifest["generation"], 0),
    )
    keyring.values[first_chunk_key] += "corrupt"

    with pytest.raises(CookieStoreUnavailableError) as error:
        store.read()

    assert "secret" not in str(error.value)


def test_delete_removes_manifest_chunks_and_legacy_entry() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    keyring.values[(store.service_name, store.username)] = "sessionid=legacy"
    store.save("sessionid=" + "x" * 3500)

    store.delete()

    assert not [key for key in keyring.values if key[0] == store.service_name]
    assert store.read() is None
