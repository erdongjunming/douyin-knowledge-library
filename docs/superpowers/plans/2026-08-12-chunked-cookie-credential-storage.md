# Chunked Cookie Credential Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store complete Douyin cookies as versioned Windows Credential Manager chunks while preserving legacy reads, atomic replacement, and secret-safe errors.

**Architecture:** Keep the HTTP and frontend contracts unchanged and contain the implementation inside `CookieStore`. Write 900-character chunks under a random generation, verify them, then write a small JSON manifest as the commit point; readers verify the manifest and SHA-256 digest, while cleanup handles legacy and obsolete entries without exposing credential contents.

**Tech Stack:** Python 3.14, keyring/Windows Credential Manager, hashlib, json, uuid, pytest, FastAPI, React/Vitest.

---

## File Structure

- Modify `backend/src/douyin_library/security/cookie_store.py`: own chunk naming, manifest parsing, atomic save, verified read, and deletion.
- Modify `backend/tests/unit/test_cookie_store.py`: exercise storage behavior with a stateful fake keyring and injected failures.
- Verify `backend/tests/integration/test_cookie_validation.py`: keep the existing safe API response contract.
- Verify `frontend/src/App.test.tsx` and `frontend/src/App.tsx`: keep the existing localized storage error contract unchanged.

### Task 1: Preserve The Existing Safe Error Baseline

**Files:**
- Modify: `backend/src/douyin_library/security/cookie_store.py`
- Modify: `backend/src/douyin_library/api/routes/settings.py`
- Modify: `backend/tests/unit/test_cookie_store.py`
- Modify: `backend/tests/integration/test_cookie_validation.py`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: Run the focused baseline tests**

Run:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/tests/unit/test_cookie_store.py backend/tests/integration/test_cookie_validation.py -v
Set-Location frontend
npm test -- --run src/App.test.tsx
```

Expected: backend credential tests and all `App.test.tsx` tests pass, including the safe `503 douyin_cookie_store_unavailable` mapping.

- [ ] **Step 2: Run lint and build for the existing baseline**

Run:

```powershell
backend\.venv\Scripts\ruff.exe check backend/src backend/tests
Set-Location frontend
npm run build
```

Expected: both commands exit 0.

- [ ] **Step 3: Commit only the existing safe error changes**

```powershell
git add backend/src/douyin_library/api/routes/settings.py `
  backend/src/douyin_library/security/cookie_store.py `
  backend/tests/integration/test_cookie_validation.py `
  backend/tests/unit/test_cookie_store.py `
  frontend/src/App.test.tsx frontend/src/App.tsx
git commit -m "fix: report unavailable cookie credential storage safely"
```

Expected: the six pre-existing modified files are committed without including the chunking plan or runtime artifacts.

### Task 2: Add Chunked Round-Trip And Legacy Compatibility

**Files:**
- Modify: `backend/tests/unit/test_cookie_store.py`
- Modify: `backend/src/douyin_library/security/cookie_store.py`

- [ ] **Step 1: Add failing tests for long cookies and legacy reads**

Add these tests:

```python
def test_long_cookie_is_chunked_and_round_trips() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    cookie = "sessionid=" + "x" * 3500

    store.save(cookie)

    manifest_raw = keyring.values[(store.service_name, store.manifest_username)]
    manifest = json.loads(manifest_raw)
    chunk_values = [
        keyring.values[(store.service_name, store.chunk_username(manifest["generation"], index))]
        for index in range(manifest["chunk_count"])
    ]
    assert all(len(chunk) <= store.chunk_size for chunk in chunk_values)
    assert cookie not in manifest_raw
    assert store.read() == cookie


def test_legacy_single_entry_cookie_remains_readable() -> None:
    keyring = FakeKeyring()
    keyring.values[(CookieStore.service_name, CookieStore.username)] = "sessionid=legacy"

    assert CookieStore(keyring_backend=keyring).read() == "sessionid=legacy"
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```powershell
backend\.venv\Scripts\python.exe -m pytest `
  backend/tests/unit/test_cookie_store.py::test_long_cookie_is_chunked_and_round_trips `
  backend/tests/unit/test_cookie_store.py::test_legacy_single_entry_cookie_remains_readable -v
```

Expected: the long-cookie test fails because `manifest_username` and chunk storage do not exist; the legacy test continues to pass.

- [ ] **Step 3: Implement manifest and chunk helpers plus verified reads**

Implement these elements in `CookieStore`:

```python
import hashlib
import json
import uuid
from dataclasses import dataclass


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

    @classmethod
    def chunk_username(cls, generation: str, index: int) -> str:
        return f"{cls.chunk_prefix}-{generation}-{index:04d}"

    def save(self, cookie: str) -> None:
        generation = uuid.uuid4().hex
        chunks = [cookie[index : index + self.chunk_size] for index in range(0, len(cookie), self.chunk_size)]
        digest = hashlib.sha256(cookie.encode("utf-8")).hexdigest()
        for index, chunk in enumerate(chunks):
            self._keyring.set_password(self.service_name, self.chunk_username(generation, index), chunk)
        manifest = {
            "version": self.manifest_version,
            "generation": generation,
            "chunk_count": len(chunks),
            "sha256": digest,
        }
        self._keyring.set_password(self.service_name, self.manifest_username, json.dumps(manifest, separators=(",", ":")))

    def read(self) -> str | None:
        manifest_raw = self._keyring.get_password(self.service_name, self.manifest_username)
        if manifest_raw is None:
            return self._keyring.get_password(self.service_name, self.username)
        manifest = self._parse_manifest(manifest_raw)
        chunks = [
            self._keyring.get_password(self.service_name, self.chunk_username(manifest.generation, index))
            for index in range(manifest.chunk_count)
        ]
        if any(chunk is None for chunk in chunks):
            raise CookieStoreUnavailableError("Windows credential store is unavailable")
        cookie = "".join(chunk for chunk in chunks if chunk is not None)
        if hashlib.sha256(cookie.encode("utf-8")).hexdigest() != manifest.sha256:
            raise CookieStoreUnavailableError("Windows credential store is unavailable")
        return cookie
```

Add a parameterized test that stores manifests with an unsupported version, an empty or punctuation-containing generation, a zero or greater-than-4096 chunk count, and a non-hex digest. Each `read()` call must raise `CookieStoreUnavailableError`, and neither the manifest text nor a marker secret may occur in the exception string. Implement `_parse_manifest` to accept only version 1, a non-empty alphanumeric generation, `1 <= chunk_count <= 4096`, and a 64-character hexadecimal SHA-256 digest. Wrap backend and format errors in `CookieStoreUnavailableError` without including input values.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/tests/unit/test_cookie_store.py -v
```

Expected: all current cookie-store tests pass.

- [ ] **Step 5: Commit the round-trip implementation**

```powershell
git add backend/src/douyin_library/security/cookie_store.py backend/tests/unit/test_cookie_store.py
git commit -m "feat: store Douyin cookies in credential chunks"
```

### Task 3: Make Replacement Atomic And Cleanup Complete

**Files:**
- Modify: `backend/tests/unit/test_cookie_store.py`
- Modify: `backend/src/douyin_library/security/cookie_store.py`

- [ ] **Step 1: Add failing rollback, corruption, and deletion tests**

Extend `FakeKeyring` with explicit operation tracking and write failure injection:

```python
class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.operations: list[tuple[str, str]] = []
        self.fail_set_at: int | None = None
        self.set_count = 0

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.set_count += 1
        self.operations.append(("set", username))
        if self.fail_set_at == self.set_count:
            raise OSError(1783, "credential payload rejected")
        self.values[(service_name, username)] = password

    def get_password(self, service_name: str, username: str) -> str | None:
        self.operations.append(("get", username))
        return self.values.get((service_name, username))

    def delete_password(self, service_name: str, username: str) -> None:
        self.operations.append(("delete", username))
        self.values.pop((service_name, username), None)
```

Add these tests:

```python
def test_failed_replacement_preserves_previously_committed_cookie() -> None:
    keyring = FakeKeyring()
    store = CookieStore(keyring_backend=keyring)
    store.save("sessionid=old")
    old_set_count = keyring.set_count
    keyring.fail_set_at = old_set_count + 2

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
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/tests/unit/test_cookie_store.py -v
```

Expected: rollback and complete deletion tests fail because save has no generation rollback or obsolete-entry cleanup.

- [ ] **Step 3: Implement commit-last replacement and cleanup**

Update `save` to:

```python
old_manifest = self._read_manifest_optional()
written_usernames: list[str] = []
try:
    for index, chunk in enumerate(chunks):
        username = self.chunk_username(generation, index)
        self._keyring.set_password(self.service_name, username, chunk)
        written_usernames.append(username)
    self._verify_generation(manifest)
    self._keyring.set_password(self.service_name, self.manifest_username, self._serialize_manifest(manifest))
except Exception as exc:
    for username in written_usernames:
        self._delete_best_effort(username)
    if isinstance(exc, CookieStoreUnavailableError):
        raise
    raise CookieStoreUnavailableError("Windows credential store is unavailable") from exc

self._delete_best_effort(self.username)
if old_manifest is not None:
    for index in range(old_manifest.chunk_count):
        self._delete_best_effort(self.chunk_username(old_manifest.generation, index))
```

Update `delete` to delete active chunks, manifest, and legacy entry. Missing entries are ignored; other failures are wrapped safely. `_verify_generation` must read all new chunks and verify their digest before the manifest commit.

- [ ] **Step 4: Run unit and API integration tests and verify GREEN**

Run:

```powershell
backend\.venv\Scripts\python.exe -m pytest `
  backend/tests/unit/test_cookie_store.py `
  backend/tests/integration/test_cookie_validation.py -v
```

Expected: all focused backend credential tests pass and no response contains cookie text.

- [ ] **Step 5: Commit atomic replacement**

```powershell
git add backend/src/douyin_library/security/cookie_store.py backend/tests/unit/test_cookie_store.py
git commit -m "fix: replace chunked credentials atomically"
```

### Task 4: Full Verification And Runtime Acceptance

**Files:**
- Verify: all backend and frontend source/tests
- Runtime only: local services on `127.0.0.1:8002` and `127.0.0.1:5175`

- [ ] **Step 1: Run all automated checks**

Run:

```powershell
scripts\verify.ps1
```

Expected: the complete backend suite passes, Ruff reports no findings, all frontend tests pass, and the TypeScript/Vite build exits 0.

- [ ] **Step 2: Run a real Windows Credential Manager size smoke test**

Use a temporary service name and a synthetic 5000-character string with the production `CookieStore`. Save, read, compare, and delete it. Print only boolean status and chunk count, never credential contents.

Expected: round-trip is true and the temporary manifest/chunks are removed.

- [ ] **Step 3: Restart the ordinary-user backend**

Stop the current `8002` foreground service, then run:

```powershell
$env:DKL_STORAGE_ROOT = 'D:\DouyinKnowledgeLibrary'
backend\.venv\Scripts\python.exe -m uvicorn douyin_library.main:app --host 127.0.0.1 --port 8002
```

Keep `5175` proxying `/api` to `8002`.

- [ ] **Step 4: Verify the rendered settings page**

Use Microsoft Edge through Playwright because the Browser plugin is unavailable. Open `http://127.0.0.1:5175/`, verify the page title, nonblank DOM, no Vite overlay, no console errors, the connected status, and the Settings credential form.

Expected: page and API health are 200, with no relevant browser errors.

- [ ] **Step 5: Complete user-local real Cookie acceptance**

The user pastes the complete Cookie into the password input and clicks `验证并保存`. Then run only safe status endpoints:

```powershell
Invoke-RestMethod http://127.0.0.1:5175/api/settings/douyin-cookie
Invoke-RestMethod -Method Post http://127.0.0.1:5175/api/settings/douyin-cookie/validate
```

Expected: status is `configured: true`; validation reports `valid: true` and an account nickname without returning Cookie text.

- [ ] **Step 6: Start one real synchronization and monitor it**

From the local UI, trigger either favorites or a configured creator synchronization. Verify `/api/jobs` progresses and `/api/works` receives discovered items; inspect only metadata and asset paths under `D:\DouyinKnowledgeLibrary`, never stored credential contents.

Expected: the job uses the stored cookie and reaches a terminal success state or reports a distinct Douyin validation/risk-control error rather than a credential-storage error.
