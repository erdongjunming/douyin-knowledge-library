from douyin_library.security.cookie_store import CookieStore


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.values[(service_name, username)] = password

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.values.get((service_name, username))

    def delete_password(self, service_name: str, username: str) -> None:
        self.values.pop((service_name, username), None)


def test_cookie_is_only_available_from_keyring() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)

    store.save("sessionid=secret")

    assert keyring.values[("douyin-knowledge-library", "douyin-cookie")] == "sessionid=secret"
    assert store.read() == "sessionid=secret"


def test_cookie_can_be_deleted_from_keyring() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    store.save("sessionid=secret")

    store.delete()

    assert store.read() is None
