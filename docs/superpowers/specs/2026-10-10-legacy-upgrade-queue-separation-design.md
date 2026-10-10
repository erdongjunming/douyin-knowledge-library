# Legacy Note Upgrade Queue Separation Design

**Status:** Proposed for review  
**Date:** 2026-10-10

## Goal

Prioritize implementation of a reliable old-note upgrade path while preserving
the existing new-note ingestion behavior. The runtime policy already chosen is
that new-note jobs have priority; legacy upgrades run in bounded batches only
when no runnable new-note job exists.

This slice changes queue identity, refresh behavior, failure recovery, and
progress reporting. It does not start workers, run an upgrade batch, delete
notes, or rewrite source assets.

## Current problem

Both flows currently create `kind="generate_note"` jobs. The legacy flag lives
only in `payload_json`, `claim_next()` has no lane-aware policy, and the refresh
endpoint only recovers stale jobs. The historical upgrade script can also call
the note service directly, bypassing the durable queue. Active-job de-duplication
is a read-then-insert sequence without a database uniqueness boundary. These
facts allow old upgrades to compete with new ingestion, duplicate under
concurrent requests, and be retried repeatedly after failure.

## Invariants

1. New-note ingestion remains behaviorally compatible: its API payload,
   provider selection, publication rules, and existing job kind remain valid.
2. Every note job has a durable lane: `new_note` or `legacy_upgrade`.
   Non-note jobs use `pipeline`.
3. For one work/aweme, at most one active note job exists across both lanes.
4. `claim_next()` chooses an eligible `new_note` before any
   `legacy_upgrade`. A legacy job is claimable only when no eligible new-note
   job exists.
5. Refresh is bounded and observable. It may recover stale work and enqueue a
   small legacy batch, but it never turns a terminal failure into an unlimited
   retry loop.
6. A stale worker cannot publish or mark success after another worker has
   reclaimed the job.
7. A legacy upgrade publishes only a local-Ollama result that passes both the
   semantic quality gate and the completeness gate. Any other result preserves
   the existing note and remains failed.
8. New-note and legacy-upgrade progress are reported independently from their
   actual current state.

## Queue data model

Add a nullable `queue_lane` column to `Job`, with values `pipeline`,
`new_note`, and `legacy_upgrade`. Add a deterministic `dedupe_key` for note
jobs, formatted as `note:<aweme_id>`. Existing rows are backfilled from
`kind` and `payload_json`: `legacy_only=true` becomes `legacy_upgrade`, other
`generate_note` rows become `new_note`, and all other kinds become `pipeline`.

Add a partial unique index for active note jobs on `dedupe_key`, covering
`pending`, `running`, and `paused`. The queue creation path must translate a
unique-conflict into the existing duplicate-active result. Rows without an
aweme identifier continue to use the existing payload validation and are never
silently treated as the same work.

Add a fencing value to a claimed job. Claiming or stale recovery advances the
value; heartbeat, release, handler completion, and publication checks require
the current worker identity and fence. A late worker therefore cannot complete
or publish after refresh has reclaimed its job.

## Scheduling and refresh

`JobQueue.claim_next()` accepts lane-aware selection while retaining the
existing ordering for pipeline work. Its order is:

1. eligible `new_note` jobs, ordered by retry time, creation time, and id;
2. eligible pipeline jobs using the current stage ordering;
3. eligible `legacy_upgrade` jobs, only if no eligible `new_note` remains.

The implementation must preserve the existing ASR/general pool split. It must
not introduce a second general pool for upgrades. A durable legacy slot or
equivalent conditional claim prevents two general workers from occupying the
legacy capacity concurrently.

`POST /api/jobs/refresh` performs these steps in one reported operation:

1. recover stale jobs with a new fence and a recorded recovery attempt;
2. inspect active, cooled, failed, and unqueued counts by lane;
3. enqueue missing new-note jobs using the existing discovery rules;
4. if no runnable new-note job exists, enqueue at most the configured legacy
   batch size (default three), deterministically selected;
5. return counts for recovered, queued-new, queued-legacy, cooled, skipped, and
   per-item conflicts/errors.

Refreshing does not immediately requeue a terminal legacy failure. A legacy
candidate remains cooled until its retry time and maximum failure policy allow
another attempt. Any explicit future retry action must be separate from the
ordinary refresh action.

The direct historical upgrade runner is changed to create or resume
`legacy_upgrade` jobs through the same queue and handler. It must not call the
generation service in a way that bypasses lane priority, fencing, or audit
records.

## Failure and preservation behavior

New-note failure and retry behavior remains unchanged unless required by the
new fencing check. Legacy failures are classified as provider transport/model
failure, invalid or incomplete semantic output, publication conflict, input
insufficiency, cancellation, or stale recovery. Each failure records a bounded
cooldown and machine-readable reason. Repeated refreshes cannot create a fresh
active job while the candidate is cooling down.

The legacy handler must retain the existing publication guard. It requires
local Ollama, `quality_gate=passed`, `semantic_complete=true`, unchanged source
hashes, and a readable candidate note before publication. On failure it leaves
the old Markdown, raw/cleaned transcripts, audio, metadata, and audits intact
and ends the job as failed or cancelled according to the durable state.

## API and UI

The notes API continues accepting `legacy_only` for compatibility, but maps it
to `legacy_upgrade` at queue creation. Normal requests map to `new_note`.
Responses expose independent counts and the effective batch limit.

The task center displays lane and provider, and keeps multi-select, refresh,
pause, resume, and cancel actions. Bulk actions return `applied`, `skipped`, and
`errors` per id so mixed terminal/paused/running selections are visible. The
refresh button reports stale recovery separately from newly queued legacy work.

The progress endpoint exposes independent new-note and legacy-upgrade cohorts.
Presence of `learning_note.md` alone cannot count as a semantic legacy upgrade;
legacy completion requires current semantic state. The task list must retain
the existing recent-job behavior while adding a lane filter or an explicit
legacy view so a 100-row window cannot hide the upgrade state.

## Implementation order

The first implementation slice is legacy-focused and guarded by compatibility
tests:

1. add the migration, lane backfill, dedupe key, and fencing fields;
2. add queue tests for duplicate races, legacy cooldown, stale fencing, and
   new-before-legacy claiming;
3. route legacy discovery and refresh refill through the durable queue;
4. update the legacy handler and historical runner to use the shared path;
5. update independent progress and task-center lane presentation;
6. run the three adversarial review rounds before any real backlog action.

Normal new-note generation is covered by regression tests before each queue
change. No production upgrade is started as part of implementation.

## Three adversarial review rounds

### Round 1: Scheduling and concurrency

Test concurrent new/legacy enqueue, an aweme without a work id, active-job
uniqueness, new-job priority, continuous new arrivals, duplicate general
processes, and the legacy capacity guard.

### Round 2: Runtime ownership and recovery

Test the stale boundary, refresh racing with a late worker, running pause and
cancel, GPU lease expiry/reacquisition, Ollama transport failure, and repeated
refreshes. Assert that only the fenced owner can publish and that recovery adds
cooldown/attempt evidence.

### Round 3: Data protection and UI truth

Test semantic-gate failure, input insufficiency, preserved old-note/source
hashes, queue/audit/filesystem progress disagreement, mixed multi-select
actions, and jobs beyond the recent list window. Assert that failed upgrades
remain `preserved_existing` and that new/legacy progress never shares a false
completion count.

## Acceptance criteria

- Existing new-note API and focused regression suite remain green.
- Two concurrent requests cannot create two active note jobs for one aweme.
- A runnable new-note job is always claimed before a legacy job.
- Refresh recovers stale work and queues at most the configured legacy batch,
  with no unlimited failed-job requeue.
- A late stale worker cannot publish or mark success.
- Legacy publication requires local Ollama, passed quality, semantic
  completeness, and source-hash protection.
- Failed upgrades preserve all existing note/source assets and are visible as
  failed or cooled, never completed.
- New-note and legacy progress remain separately auditable.
- No real worker, Ollama generation, or production storage mutation occurs
  during implementation tests.
