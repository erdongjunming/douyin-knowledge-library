# Legacy Note Upgrade Repair Design

**Status:** Approved design

**Date:** 2026-09-01

## Context

The current web upgrade action can enqueue every legacy note because it sends no
limit. In the live library that produced 724 upgrade jobs from one click. Four
jobs that executed did not replace their notes: Ollama either failed transport or
returned an incomplete draft-claim review, the service selected rule fallback,
and the publisher correctly preserved the stronger existing bundle. The worker
then ignored the service result and marked each no-op job `succeeded`.

The progress endpoint repeats the same mistake at reporting time. It treats a
generic succeeded job as a completed upgrade even when the current bundle is
still legacy, and it defines eligibility using only the `legacy` classification.
Consequently fallback and unknown notes are unreachable, successful counts can
be false, and the denominator changes as files change.

This repair makes upgrade success a semantic, state-based contract. It does not
change the underlying evidence-preserving note architecture: a replacement must
still pass the semantic coverage gate that protects details, examples,
conditions, ordered steps, and source traceability. In particular, it must not
reintroduce the overly concise behavior previously observed in the Jiang
Xuezhang regression sample `7636753241384695225`.

## Goals

- Limit every Web/API upgrade request to at most three notes.
- Upgrade existing `legacy`, `fallback`, and `unknown` bundles, while protecting
  `semantic_complete` bundles.
- Use the local Ollama semantic provider for upgrades and never call an external
  provider as part of an upgrade.
- Replace the current bundle only with an Ollama result whose semantic coverage
  gate passed.
- Preserve the old bundle and fail the attempt and job when Ollama is unavailable,
  invalid, incomplete, or falls through to rule output.
- Accept an omitted or partial `reviewed_claim_ids` field without crashing while
  treating every unreviewed claim as unsupported.
- Continue rejecting unknown claim IDs and contradictory verdicts.
- Report upgrade completion from current semantic bundle state rather than a
  generic job status.
- Show completed semantic upgrades, active jobs, failed outcomes, and remaining
  candidates in the web interface.
- Leave the 724 existing queued jobs and the real library database untouched.

## Non-Goals

- Running, cancelling, deleting, rewriting, or requeueing existing jobs.
- Starting a server, general worker, ASR worker, or Ollama trial during this code
  repair.
- Modifying `D:\DouyinKnowledgeLibrary` or any source audio, cover, metadata,
  transcript, audit, or note file.
- Performing the three-note Ollama acceptance run or the eventual full upgrade.
- Changing normal missing-note generation, CLI full-upgrade authorization, ASR,
  subtitle discovery, transcript cleaning, GPU scheduling, or database schema.
- Relaxing evidence, coverage, publication, archive, rollback, or ownership
  fencing requirements for semantic notes.

## Confirmed Decisions

1. The Web/API upgrade boundary has both a default and a hard cap of three. An
   omitted limit selects three; a value above three returns HTTP 422 and creates
   no jobs. Normal missing-note requests retain their existing limit behavior.
2. The existing wire field `legacy_only` remains for compatibility. In upgrade
   mode it is a historical name for existing non-semantic bundles, specifically
   `legacy`, `fallback`, and `unknown`.
3. `missing` is handled by normal incremental generation, not upgrade mode.
4. `semantic_complete` is never an upgrade candidate and is never replaced by
   this action.
5. Upgrade policy is local Ollama. The frontend sends `provider_policy: "local"`,
   and backend orchestration enforces local policy even for older queued payloads
   that contain `auto`.
6. Rule generation remains available for normal missing-note generation, but it
   is not an acceptable upgrade result and is not published over an existing
   candidate.
7. A job succeeds only when its own attempt publishes a passed semantic bundle.
   A preserved, skipped, fallback, or other no-op result is a failed upgrade.
8. Progress is derived from bundle classification plus deduplicated upgrade-job
   history. It never trusts `Job.status == "succeeded"` by itself.

## State And Eligibility Contract

`classify_note_bundle()` remains the source of truth for the current bundle. The
state model gains an explicit upgrade predicate:

```text
upgrade_eligible = quality_status in {legacy, fallback, unknown}
```

The predicate is false for `semantic_complete` and `missing`. Service discovery
and the per-item generation guard both use this predicate so a direct service
call cannot bypass API filtering.

The publication guard records the candidate's expected classification before
generation. Immediately before replacement it requires the current bundle still
to be upgradeable and unchanged in the fields that identify the guarded state.
If another process changes the bundle, the current attempt does not publish or
claim success.

## Web And API Safety Boundary

For `POST /api/notes/generate` requests with `legacy_only: true`, the route derives
an effective request:

```json
{
  "legacy_only": true,
  "limit": 3,
  "provider_policy": "local"
}
```

The route uses three when `limit` is omitted. A supplied limit above three returns
HTTP 422 before discovery or queue mutation. Dry-run selection follows the same
cap, while `eligible` still reports the full number of current candidates. An
accepted response reports the effective limit and local policy so callers can
verify what was accepted.

The cap is scoped to Web/API upgrade requests. It does not silently change the
normal missing-note route behavior or authorize a larger CLI rollout. Duplicate
active-job handling remains unchanged.

## Local Semantic Upgrade Flow

1. Discover current `legacy`, `fallback`, and `unknown` candidates.
2. Select at most three in deterministic transcript-path order.
3. Enqueue jobs whose payload records `legacy_only: true`, `limit` selection at
   the request boundary, and `provider_policy: "local"`.
4. At execution time, normalize every upgrade job to local policy. This also
   prevents an old queued `auto` payload from calling an external provider.
5. Attempt Ollama semantic generation using the existing evidence extraction,
   synthesis, independent review, repair, and coverage gates.
6. Accept a candidate only when the selected provider is Ollama, a semantic
   generation object exists, and `quality_gate == "passed"`.
7. Stage, archive, and atomically publish through `NoteBundlePublisher` with the
   existing ownership and compare-before-publish guards.
8. Return `generated` only after publication succeeds. Every other outcome
   preserves the old bundle and carries a machine-readable failure reason.
9. The runtime inspects `GenerationResult`. It lets the generic handler finish a
   job successfully only for a generated, passed semantic upgrade. It raises or
   explicitly fails the queue item for fallback, preserved, skipped, or no-op
   results.

Rule output need not be rendered during an upgrade. If the existing provider
selector exposes rule as the last local fallback, orchestration records it as
ineligible for upgrade and stops before publication. This keeps the audit trail
clear without spending work on a bundle that cannot be accepted.

## Claim Review Contract

The review prompt must explain the ID fields explicitly and request a verdict for
every claim in the batch. Runtime validation remains authoritative because a
small local model can omit fields or return only part of the requested partition.

`reviewed_claim_ids` defaults to an empty list when absent. For each review batch:

1. Build `expected_ids` from the claims actually sent.
2. Reject any ID appearing in `reviewed_claim_ids`, `supported_claim_ids`, or
   `contradicted_claim_ids` when it is not in `expected_ids`.
3. Reject any ID appearing in both supported and contradicted verdict sets.
4. Credit a claim as supported only when it is both explicitly reviewed and
   explicitly supported.
5. Treat every expected claim not credited as supported as unsupported. This
   includes omitted and partially reviewed IDs.

Thus an incomplete review is tolerated structurally but fails closed for quality.
It can cause the semantic coverage gate, and therefore the upgrade, to fail; it
cannot crash solely because the model omitted the bookkeeping field, and it
cannot allow unreviewed content to pass.

## Failure And Preservation Semantics

Upgrade failure is an expected domain result, not a successful no-op. Failures
include Ollama transport/model errors, invalid structured output, exhausted
repair, incomplete coverage, rule fallback, publication rejection, ownership
loss, and any preserved/skipped result returned while the bundle remains
non-semantic.

For every failed upgrade:

- the current generated bundle remains byte-for-byte unchanged;
- no fallback bundle replaces it;
- the processing audit records the Ollama stages and sanitized failure reason;
- the batch result increments `failed`, not `generated` or `skipped`;
- the durable job ends as `failed`, not `succeeded`; and
- the candidate remains visible in `remaining` for a future retry.

Publisher fallback protection remains as defense in depth. The service-level
semantic gate prevents unacceptable output from reaching publication in the
first place.

## Authoritative Upgrade Progress

The progress endpoint builds an ID-based cohort rather than combining unrelated
counts:

```text
cohort = current upgrade-candidate IDs
         union IDs ever targeted by a legacy_only upgrade job
```

Native semantic notes with no upgrade history are not part of this cohort.
Keeping job-targeted IDs in the cohort prevents the denominator from shrinking
after a successful replacement.

For each cohort ID, inspect its current bundle and only its latest upgrade job:

- `total`: number of IDs in the cohort.
- `completed`: IDs whose current bundle is `semantic_complete`.
- `remaining`: `total - completed`.
- `pending`: remaining IDs whose latest upgrade job is pending.
- `running`: remaining IDs whose latest upgrade job is running, including a
  running job whose stage is waiting for the GPU.
- `active`: `pending + running`.
- `failed`: remaining IDs whose latest job failed, plus a terminal succeeded job
  whose current bundle is still non-semantic. The latter is reported as
  `upgrade_completed_without_semantic_note` so the four historical false
  successes remain visible without rewriting the database.
- `not_queued`: remaining IDs not represented by an active or failed latest job.
- `percent`: `completed / total * 100`, or zero when the cohort is empty.
- `failures`: up to five latest failed IDs and readable sanitized reasons.

These categories describe different questions: failed and active are subsets of
remaining, while completed and remaining partition total. The endpoint retains
existing top-level note progress fields for compatibility but does not mix them
with semantic-upgrade completion.

## Frontend Behavior

The upgrade button:

- appears when `upgrade.remaining` (or compatibility `upgrade.eligible`) is
  positive;
- is enabled only when local Ollama is ready, even if an external provider is
  configured;
- sends exactly `legacy_only: true`, `limit: 3`, and
  `provider_policy: "local"`; and
- reports queued and already-active counts returned by the API.

A separate `旧笔记语义升级` progress lane displays completed/total, percent,
active, remaining, failed, and the latest failure reason. The existing knowledge
note lane remains about normal note-file generation. This prevents a complete
rule note file from being presented as a completed semantic upgrade.

## Compatibility And Operational Safety

- No migration or payload rewrite is required.
- The `legacy_only` payload key and current response fields remain readable by
  existing callers.
- Existing 724 queued jobs are observed read-only. This repair does not cancel,
  delete, duplicate, or execute them.
- When those jobs are eventually executed by an operator, runtime normalization
  applies the new local-only and semantic-success rules without changing their
  stored payloads first.
- No test uses the real storage root. Tests create temporary transcript, note,
  database, and audit directories.
- No acceptance command starts the local application or worker as part of this
  code-only change.

## Verification Strategy

Implementation follows red-green-refactor. Each boundary first receives a test
that fails for the observed reason, then the smallest production change.

Backend unit tests cover:

- upgrade eligibility for legacy, fallback, and unknown, and exclusion of
  semantic-complete and missing bundles;
- passed Ollama publication as the only successful replacement;
- rule fallback and provider failure preserving the old bundle and returning a
  failed upgrade;
- batch failure accounting;
- omitted and partial `reviewed_claim_ids` failing closed without a parser or
  equality exception;
- unknown IDs in every review list and supported/contradicted overlap remaining
  hard failures; and
- prompt instructions for complete claim-ID accounting.

Backend integration tests cover:

- omitted upgrade limit selecting exactly three of at least five candidates;
- a limit above three returning HTTP 422 and creating no jobs;
- queued payloads recording local policy;
- legacy, fallback, and unknown candidates being queued while semantic-complete
  notes are excluded;
- a failed Ollama upgrade preserving files and marking its job failed;
- a publication no-op not becoming a succeeded job; and
- progress reporting a real semantic completion, active jobs, a failed job, an
  untouched candidate, and a historical false success without counting the false
  success as complete.

Frontend tests cover:

- the exact `{legacy_only: true, limit: 3, provider_policy: "local"}` request;
- external-only readiness not enabling the local upgrade action;
- the candidate count including fallback and unknown notes; and
- accessible rendering of completed, active, failed, remaining, percent, and
  failure detail in the dedicated upgrade lane.

Focused backend and frontend tests run first. Final code verification includes
the complete backend suite, Ruff, compileall, complete frontend tests, production
frontend build, and `git diff --check`. None of these commands target the real
library.

## Acceptance Criteria

The code repair is complete when all of the following hold in temporary tests:

1. One Web/API upgrade action cannot select more than three notes.
2. Legacy, fallback, and unknown bundles are selectable; semantic-complete and
   missing bundles are not.
3. Upgrade execution attempts local Ollama and never external generation.
4. Only a published, coverage-passed Ollama bundle yields a successful job.
5. Rule fallback or incomplete semantic output leaves the old note unchanged and
   produces a failed audit, batch item, and durable job.
6. Omitted/partial review bookkeeping is fail-closed, while unknown/conflicting
   IDs are rejected.
7. API and UI progress agree with current semantic bundle state and expose the
   four historical no-op successes as unresolved failures/remaining work.
8. Existing queues, database rows, and `D:\DouyinKnowledgeLibrary` files remain
   untouched.

Only after this code repair is reviewed and an operator gives separate runtime
authorization may the project dry-run and execute a new three-note Ollama batch.
