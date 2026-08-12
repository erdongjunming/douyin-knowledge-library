# Chunked Cookie Credential Storage Design

## Context

Douyin request cookies can exceed the maximum payload accepted by a single
Windows Generic Credential. A normal-user-session probe accepted a 512-character
value and rejected values of 2048 characters and above with WinError 1783. The
current `CookieStore` writes the complete cookie into one credential and maps all
write failures to `douyin_cookie_store_unavailable`, so an oversized valid cookie
is reported as a session problem.

## Goals

- Store a complete Douyin cookie without writing plaintext secrets to disk.
- Keep the existing `CookieStore.save/read/delete` and HTTP API contracts.
- Read credentials written by the existing single-entry implementation.
- Make a failed replacement leave the previously committed cookie readable.
- Avoid including cookie values in exceptions, logs, responses, or test output.

## Non-Goals

- Exporting browser cookies or reading browser storage.
- Changing Douyin validation, synchronization, or frontend behavior.
- Enumerating or migrating unrelated Windows credentials.

## Considered Approaches

### 1. Versioned Windows Credential chunks (selected)

Split the cookie into conservative 900-character chunks. Store each chunk as a
separate Generic Credential under a unique generation name, then commit a small
JSON manifest last. Windows continues to protect every secret fragment, while a
generation identifier prevents an interrupted save from corrupting the previous
cookie.

### 2. DPAPI-encrypted file

This removes the credential-size limit, but adds file format, ACL, key lifecycle,
and recovery responsibilities. It is unnecessary while Credential Manager can
hold bounded chunks.

### 3. Retain only selected cookie fields

This reduces size but couples storage to undocumented Douyin requirements. A
field omitted today may be required by another endpoint or after a platform
change, so this is not reliable for account and favorites synchronization.

## Credential Layout

The service name remains `douyin-knowledge-library`.

- Legacy entry: `douyin-cookie`
- Manifest entry: `douyin-cookie-manifest`
- Chunk entry: `douyin-cookie-chunk-<generation>-<zero-padded-index>`

The manifest is JSON with the following fields:

```json
{
  "version": 1,
  "generation": "opaque-random-id",
  "chunk_count": 4,
  "sha256": "hex-digest-of-the-complete-cookie"
}
```

The manifest contains no cookie text. The hash detects missing, reordered, or
corrupted chunks and is not used as an authentication secret.

## Save Flow

1. Split the cookie into 900-character chunks and create a random generation ID.
2. Read and retain the currently committed manifest, if one exists.
3. Write every new-generation chunk.
4. Read the new chunks back, concatenate them, and verify the SHA-256 digest.
5. Write the new manifest last. This is the commit point.
6. Delete the legacy entry and the chunks referenced by the previous manifest.

If a chunk write or verification fails before the commit point, delete all
new-generation chunks that may have been written and leave the old manifest and
legacy entry unchanged. Cleanup after the commit point is best effort: the new
cookie remains readable even if an obsolete entry cannot immediately be removed.

## Read Flow

1. Read the manifest.
2. If no manifest exists, return the legacy `douyin-cookie` value.
3. Validate the manifest schema and supported version.
4. Read every referenced chunk in index order.
5. Concatenate the chunks and verify the stored SHA-256 digest.
6. Return the cookie only after integrity verification succeeds.

Malformed manifests, missing chunks, unsupported versions, and digest mismatches
raise a safe `CookieStoreUnavailableError` without embedding credential content.

## Delete Flow

Read the active manifest, delete its referenced chunks, delete the manifest, and
delete the legacy entry. Missing entries are treated as already deleted. Other
backend failures are wrapped without exposing secret values.

## Error Handling And Audit Safety

- No cookie or chunk content is logged or returned by the API.
- Backend exceptions expose only a stable safe message and exception type/code
  metadata where needed for diagnostics.
- The existing HTTP response remains `503 douyin_cookie_store_unavailable` for
  storage failures; the frontend contract does not change.
- The API continues to validate the complete cookie before storing it.

## Tests

Unit tests will prove:

- Long cookies are split into chunks no larger than 900 characters and round-trip.
- The manifest is written after all chunks and contains no cookie text.
- Legacy single-entry credentials remain readable.
- A failed chunk write removes partial new chunks and preserves the old cookie.
- Missing or corrupted chunks fail safely without exposing their content.
- Delete removes the active manifest, active chunks, and legacy entry.
- Keyring write failures remain wrapped in `CookieStoreUnavailableError`.

Existing integration tests will continue to verify that validation happens before
storage and that HTTP responses never echo the cookie. The complete backend test
suite, Ruff, frontend tests, and frontend build must pass before runtime acceptance.

## Runtime Acceptance

After tests pass, restart the ordinary-user backend and its matching frontend
proxy. The user will paste the Cookie locally. Acceptance requires:

- `PUT /api/settings/douyin-cookie` returns success.
- `GET /api/settings/douyin-cookie` returns `{ "configured": true }`.
- Validation reports the logged-in account without returning credential text.
- A real favorites or creator synchronization can read the stored cookie.
