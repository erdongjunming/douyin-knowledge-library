# Audit repairs implementation plan

> Execute the user-approved audit repairs with regression tests and independent review. Preserve existing worktree changes; do not run generation against the real library.

**Goal:** Resolve the seven confirmed audit findings while preserving new-note priority, fenced job ownership, local legacy generation and old-note/source protection.

**Architecture:** Extend existing workers, publication locks and queue contracts. Keep existing note paths compatible by adding a durable publication journal, recovery and integrity verification. Use one cancellable polling hook and bounded SQL task queries; throttle advisory inventory discovery.

**Validation:** Python pytest against temporary storage, frontend Vitest/type checks/build, then independent requirements and regression reviews. Baseline sources are copied to the external temporary directory `C:\Users\CJM\AppData\Local\Temp\dkl-repair-baseline-wn48uxqh`.

## Task 1: Worker recovery, media heartbeats and health

- [x] Add regression tests showing a running job becomes stale after a fast restart and is recovered by subsequent worker polls, without reclaiming a fresh lease.
- [x] Run those tests before implementation and confirm the stale job remains running.
- [x] Add throttled periodic recovery using the configured stale threshold and existing claim fences.
- [x] Add a heartbeat responsiveness regression for slow media processing; move synchronous media work off the event loop, keep ORM work on the owning thread, and bound/terminate FFmpeg work on timeout or cancellation.
- [x] Add tests that worker failure yields degraded health and independent worker liveness is distinct from pool pause configuration.
- [x] Run worker/recovery/media/health tests and self-review.

## Task 2: Recoverable note publication

- [x] Add tests interrupting publication after an output replacement, on first publication and an upgrade; incomplete bundles must not report semantic completeness.
- [x] Run the new tests and observe failures.
- [x] Persist a commit journal before replacement, retain a rollback snapshot until manifest commit, and recover interrupted commits under the existing publication lock before publishing or reading/classifying.
- [x] Include file hashes in new manifests and reject mismatches; preserve compatibility for legacy manifests, raw assets, archives and fencing guards. Protect readers from inconsistent publication snapshots.
- [x] Run publisher/state/API note-reader regression tests and self-review.

## Task 3: Queue API, UI controls and polling

- [x] Add tests for failed legacy generation reporting a blocked failure rather than active queue duplication, plus explicit fenced retry through the existing resume contract.
- [x] Add tests for viewing/resuming/cancelling paused tasks older than the recent 100 rows.
- [x] Implement bounded pagination and status/lane filters, SQL lane aggregation, visible active/paused task controls and explicit failed retry.
- [x] Add fake-timer tests proving repeated refresh events retain one polling chain, avoid overlap and clean up on unmount.
- [x] Extract a shared polling hook, re-probe provider readiness on refresh/interval, and cache expensive note progress briefly without masking refresh results.
- [x] Run frontend tests/type checks and affected backend API tests; self-review.

## Task 4: Scheduler scan throttling and cleanup

- [x] Add a regression proving empty/failed inventories are not rediscovered on every worker idle tick while explicit refresh can scan immediately.
- [x] Persist a bounded discovery cooldown without changing three-item batches, new-note priority, paused-batch behavior or terminal-failure protection.
- [x] Run scheduler/lane tests and fix the two audited import ordering violations.

## Final validation and review

- [x] Independent requirements review of all seven findings.
- [x] Independent review of fences, cancellation, publication interruption and UI polling races; resolve findings and recheck.
- [x] Run the backend suite and frontend tests/build in isolated test storage.
- [x] Check the delta against the external baseline and report code/test results separately from live model generation, which is outside this repair verification.


## Completion record (2026-10-13)

All seven audit repairs are implemented and reviewed:

1. Worker polls periodically reclaim expired claims; fresh leases remain protected.
2. Blocking media processing runs outside the event loop, and FFmpeg has bounded timeout/cancellation cleanup.
3. Health uses actual worker liveness, independently of configured pool pause state.
4. Publication uses a durable journal, recovery snapshots and output hashes; incomplete or altered bundles cannot report semantic completeness. Dry-run classification does not create locks or recover journals.
5. Task queries support bounded pagination and status/lane filters. Paused tasks remain accessible, and terminal legacy failures require explicit retry with clear UI messages.
6. A shared cancellable polling hook prevents overlapping chains. Progress has a short cache, with explicit refresh bypass and provider readiness rechecks.
7. Legacy inventory discovery uses a persisted cooldown, including discovery exceptions, while explicit refresh bypasses it.

Review follow-ups are resolved. Media asset commit, source cleanup and transcription enqueue use immutable claim identities and a conditional database write lock. Pause or fence replacement cannot occur between the ownership check and protected source deletion/enqueue. Failure or cancellation before a valid asset commit preserves the downloaded source. New-note priority, batches of at most three legacy upgrades, local Ollama requirements and quality/semantic publication gates remain enforced.

Final validation:

- Complete backend suite: **939 passed, 1 warning in 61.03s**. The warning is the existing Python sqlite datetime-adapter deprecation in the migration test.
- Complete frontend suite: **43 passed**; TypeScript and production build passed.
- Ruff, Python compilation and standard `git diff --check` passed.
- Final independent publication/fencing review passed, including an additional isolated SQLite competition check for pause/write-lock ordering, stale fences and committed child preservation.
- Compared against the external pre-repair baseline: 32 captured source/test/script files changed and eight scoped source/test files added; no captured baseline/current file is missing. Existing unrelated dirty changes were preserved.

Validation used temporary storage, synthetic media and mocked providers. No real-library generation, model invocation or real service restart was performed; live runtime acceptance is not claimed.
