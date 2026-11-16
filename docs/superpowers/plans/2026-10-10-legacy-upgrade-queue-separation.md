# Legacy Note Upgrade Queue Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Move old-note upgrades onto an explicit, bounded queue lane with refresh-driven refill and stale-job fencing while keeping new-note ingestion compatible and higher priority.

**Architecture:** Keep \`kind="generate_note"\` for handler compatibility, but persist \`queue_lane\`, \`dedupe_key\`, and a claim fence on each job. The queue selects runnable new-note jobs before legacy upgrades. Refresh recovers stale work and refills at most three legacy jobs only when no runnable new-note work exists. Legacy execution remains local-Ollama and fail-closed through the existing semantic publisher.

**Tech Stack:** Python 3.12, SQLAlchemy/Alembic, SQLite, FastAPI, pytest/pytest-asyncio, React 19, TypeScript, Vitest.

**Safety boundary:** Use temporary databases and storage roots in tests. Do not start the API, workers, Ollama, historical runner, or write \`D:\DouyinKnowledgeLibrary\`. Preserve unrelated dirty worktree changes.

---

### Task 1: Establish regression and migration fixtures

**Files:**
- Modify: \`backend/tests/unit/test_job_state_machine.py\`
- Modify: \`backend/tests/integration/test_migrations.py\`
- Create: \`backend/tests/unit/test_legacy_queue_lane.py\`

- [ ] **Step 1: Write failing compatibility tests**

Create a temporary SQLite schema and assert that a normal \`generate_note\` payload maps to \`new_note\`, while \`legacy_only=true\` maps to \`legacy_upgrade\`.

~~~python
def test_note_jobs_get_distinct_lanes(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        normal = JobQueue(session).create(
            kind="generate_note",
            work_id=1,
            payload={"aweme_id": "new-1", "force": False},
        )
        legacy = JobQueue(session).create(
            kind="generate_note",
            work_id=2,
            payload={"aweme_id": "old-1", "legacy_only": True},
        )
        assert normal.queue_lane == "new_note"
        assert legacy.queue_lane == "legacy_upgrade"
~~~

Add a migration test that checks the new columns, backfill, and active-note partial unique index.

- [ ] **Step 2: Run the focused tests and verify the intended failure**

Run from \`backend\`:

~~~powershell
uv run pytest tests/unit/test_legacy_queue_lane.py tests/unit/test_job_state_machine.py tests/integration/test_migrations.py -q
~~~

Expected: failures report missing lane/migration behavior, not fixture or import errors.

- [ ] **Step 3: Record the new-note regression baseline**

~~~powershell
uv run pytest tests/integration/test_note_generation.py tests/integration/test_upgrade_job_contract.py tests/integration/test_jobs_api.py tests/unit/test_job_state_machine.py -q
~~~

Record pass/fail counts without changing tests for pre-existing failures.

- [ ] **Step 4: Commit only these test changes**

~~~powershell
git add backend/tests/unit/test_legacy_queue_lane.py backend/tests/unit/test_job_state_machine.py backend/tests/integration/test_migrations.py
git commit -m "test: define separated note queue lanes"
~~~

### Task 2: Add durable lane, dedupe, and fencing fields

**Files:**
- Modify: \`backend/src/douyin_library/models.py:201-225\`
- Create: \`backend/migrations/versions/0007_note_queue_lane.py\`
- Modify: \`backend/tests/integration/test_migrations.py\`

- [ ] **Step 1: Extend \`Job\` with compatibility-safe fields**

Add nullable \`queue_lane\` with values \`pipeline\`, \`new_note\`, \`legacy_upgrade\`; nullable \`dedupe_key\`; and integer \`claim_fence\` defaulting to zero. Defaults keep old rows readable before migration backfill.

- [ ] **Step 2: Add migration backfill tests before migration code**

Insert old-style normal, legacy, and non-note rows, apply the migration, and assert the lane values. Assert that an active canonical \`note:<aweme_id>\` duplicate is rejected while historical rows remain intact.

- [ ] **Step 3: Implement \`0007_note_queue_lane.py\`**

Add columns, backfill from \`kind\` and \`payload_json\`, and derive \`dedupe_key\` only when a stable aweme id exists. Create a partial unique index for non-null keys in \`pending\`, \`running\`, or \`paused\`. If historical duplicate keys exist, preserve them with deterministic \`:historical:<job_id>\` suffixes before creating the index. Do not delete or rewrite note/source/audit files.

- [ ] **Step 4: Run migration tests**

~~~powershell
uv run pytest tests/integration/test_migrations.py tests/unit/test_legacy_queue_lane.py -q
~~~

Expected: fresh-schema and upgrade-from-old-row tests pass.

- [ ] **Step 5: Commit this schema slice**

~~~powershell
git add backend/src/douyin_library/models.py backend/migrations/versions/0007_note_queue_lane.py backend/tests/integration/test_migrations.py backend/tests/unit/test_legacy_queue_lane.py
git commit -m "feat: persist note queue lanes and claim fences"
~~~

### Task 3: Make queue creation and claiming lane-aware

**Files:**
- Modify: \`backend/src/douyin_library/jobs/queue.py:20-360\`
- Modify: \`backend/src/douyin_library/jobs/runtime.py:45-60,465-490\`
- Modify: \`backend/tests/unit/test_job_state_machine.py\`
- Modify: \`backend/tests/unit/test_legacy_queue_lane.py\`

- [ ] **Step 1: Write failing queue tests**

Add tests for new-before-legacy claiming, cross-lane active dedupe for one aweme, and a legacy failure that cannot be immediately recreated by refresh.

~~~python
def test_new_note_is_claimed_before_legacy_upgrade(tmp_path: Path) -> None:
    ...

def test_cross_lane_duplicate_is_rejected(tmp_path: Path) -> None:
    ...

def test_failed_legacy_job_is_cooled(tmp_path: Path) -> None:
    ...
~~~

Use real temporary SQLAlchemy sessions; do not mock the queue state machine.

- [ ] **Step 2: Run the tests to confirm current failures**

~~~powershell
uv run pytest tests/unit/test_job_state_machine.py tests/unit/test_legacy_queue_lane.py -q
~~~

Expected: failures show missing lane mapping, dedupe, and ordering.

- [ ] **Step 3: Implement lane and canonical-key resolution**

Add one queue helper that maps explicit lane or compatibility payload: \`legacy_only=true\` to \`legacy_upgrade\`, other note generation to \`new_note\`, all other jobs to \`pipeline\`. Derive \`note:<aweme_id>\` from payload/work. Translate pre-check and SQLite unique conflicts into \`DuplicateActiveJobError\`.

- [ ] **Step 4: Implement claim ordering and the legacy slot**

Keep existing pipeline ordering unchanged. Order eligible new-note jobs before legacy upgrades, then by retry time, creation time, and id. Add a conditional durable legacy slot so only one general worker claims a legacy upgrade at once, releasing it on success, failure, cancellation, and release.

- [ ] **Step 5: Add legacy-only cooldown**

Use a dedicated legacy retry limit/delay. A terminal legacy failure remains failed with cooldown evidence and is excluded from ordinary rediscovery until its retry policy allows it. Do not change normal new-note retry constants.

- [ ] **Step 6: Run queue and note regressions**

~~~powershell
uv run pytest tests/unit/test_job_state_machine.py tests/unit/test_legacy_queue_lane.py tests/integration/test_note_generation.py tests/integration/test_upgrade_job_contract.py -q
~~~

Expected: all new lane tests and existing new-note contracts pass.

- [ ] **Step 7: Commit queue behavior**

~~~powershell
git add backend/src/douyin_library/jobs/queue.py backend/src/douyin_library/jobs/runtime.py backend/tests/unit/test_job_state_machine.py backend/tests/unit/test_legacy_queue_lane.py
git commit -m "feat: prioritize new notes over cooled legacy upgrades"
~~~

### Task 4: Add fenced stale recovery and bounded refresh refill

**Files:**
- Modify: \`backend/src/douyin_library/jobs/queue.py:190-360\`
- Modify: \`backend/src/douyin_library/api/routes/jobs.py:20-90\`
- Modify: \`backend/src/douyin_library/api/routes/notes.py:35-135\`
- Modify: \`backend/tests/integration/test_job_recovery.py\`
- Modify: \`backend/tests/integration/test_jobs_api.py\`
- Create: \`backend/tests/integration/test_legacy_refresh_contract.py\`

- [ ] **Step 1: Write failing recovery and refresh tests**

Assert stale recovery advances the fence and records \`worker_stale_recovered\`. Assert refresh returns separate \`recovered\`, \`queued_new\`, \`queued_legacy\`, \`cooled\`, \`skipped\`, and \`errors\` counts, queues at most three legacy jobs, and suppresses legacy refill whenever a runnable new job exists.

- [ ] **Step 2: Run focused tests**

~~~powershell
uv run pytest tests/integration/test_job_recovery.py tests/integration/test_jobs_api.py tests/integration/test_legacy_refresh_contract.py -q
~~~

Expected: failures show that refresh currently only recovers stale jobs and claims have no fence.

- [ ] **Step 3: Implement fenced claim lifecycle**

Advance \`claim_fence\` on claim and stale recovery. Require job id, status, worker id, and fence in heartbeat, release, success, failure, and handler completion updates. A late worker update that affects zero rows must not publish or mark success.

- [ ] **Step 4: Implement bounded refresh orchestration**

Add a service/queue method that recovers stale jobs, discovers normal missing notes, and then discovers legacy candidates. Enqueue normal candidates with existing rules; enqueue at most three legacy candidates only when no eligible new-note job exists. Never recreate terminal cooled failures. Return per-lane counts and per-item conflict/error details.

- [ ] **Step 5: Preserve API compatibility**

Keep \`legacy_only\` accepted by \`POST /api/notes/generate\`, mapping it to \`legacy_upgrade\`. Keep normal payloads and response keys unchanged. Keep the existing \`recovered\` and \`items\` keys on \`POST /api/jobs/refresh\` while adding the new counters.

- [ ] **Step 6: Run API and note regressions**

~~~powershell
uv run pytest tests/integration/test_job_recovery.py tests/integration/test_jobs_api.py tests/integration/test_legacy_refresh_contract.py tests/integration/test_note_generation.py tests/integration/test_upgrade_job_contract.py -q
~~~

- [ ] **Step 7: Commit recovery and refresh**

~~~powershell
git add backend/src/douyin_library/jobs/queue.py backend/src/douyin_library/api/routes/jobs.py backend/src/douyin_library/api/routes/notes.py backend/tests/integration/test_job_recovery.py backend/tests/integration/test_jobs_api.py backend/tests/integration/test_legacy_refresh_contract.py
git commit -m "feat: refill legacy upgrades through fenced refresh"
~~~

### Task 5: Route the historical runner through the shared queue

**Files:**
- Modify: \`scripts/upgrade_legacy_notes.py:384-415,517-680\`
- Modify: \`backend/src/douyin_library/jobs/runtime.py:320-370\`
- Modify: \`backend/tests/unit/test_legacy_upgrade_runner.py\`
- Modify: \`backend/tests/integration/test_upgrade_job_contract.py\`

- [ ] **Step 1: Write the failing runner contract**

Assert that enqueue-only mode creates/resumes \`legacy_upgrade\` jobs, forces local provider policy, respects the batch limit, and does not call \`NoteGenerationService.run_batch()\` directly.

- [ ] **Step 2: Run the test to expose the bypass**

~~~powershell
uv run pytest tests/unit/test_legacy_upgrade_runner.py tests/integration/test_upgrade_job_contract.py -q
~~~

Expected: failure because the script currently calls the service batch directly.

- [ ] **Step 3: Replace direct generation with queue submission**

Reuse the refresh/refill method. Keep the script lock as an invocation guard, but let durable queue dedupe, lane priority, cooldown, fencing, and audit records govern execution. Enqueue-only mode prints JSON counts and does not start a worker.

- [ ] **Step 4: Run runner and normal-generation contracts**

~~~powershell
uv run pytest tests/unit/test_legacy_upgrade_runner.py tests/integration/test_upgrade_job_contract.py tests/integration/test_note_generation.py -q
~~~

- [ ] **Step 5: Commit the shared runner path**

~~~powershell
git add scripts/upgrade_legacy_notes.py backend/src/douyin_library/jobs/runtime.py backend/tests/unit/test_legacy_upgrade_runner.py backend/tests/integration/test_upgrade_job_contract.py
git commit -m "feat: route historical upgrades through durable jobs"
~~~

### Task 6: Make progress and task-center controls truthful by lane

**Files:**
- Modify: \`backend/src/douyin_library/api/routes/jobs.py:35-220\`
- Modify: \`backend/src/douyin_library/api/routes/notes.py:170-345\`
- Modify: \`frontend/src/App.tsx:473-552,850-940\`
- Modify: \`frontend/src/App.test.tsx\`
- Create: \`frontend/src/LegacyUpgradeQueue.test.tsx\`

- [ ] **Step 1: Write failing API and frontend tests**

Backend tests must distinguish new and legacy counts and exclude fallback/rule output from semantic completion. Frontend tests must show the lane, report legacy refill separately, and display applied/skipped/error results for mixed selections.

- [ ] **Step 2: Run focused frontend tests**

~~~powershell
Set-Location frontend
npm test -- --run App.test.tsx LegacyUpgradeQueue.test.tsx
~~~

Expected: failures show missing lane labels and refresh/bulk-result behavior.

- [ ] **Step 3: Implement independent progress serialization**

Expose separate normal and legacy cohorts with queued, running, failed, cooled, not_queued, remaining, and percent. Count legacy complete only when the current bundle is semantic-complete; a generic succeeded job is insufficient.

- [ ] **Step 4: Update task-center controls**

Extend the job wire type with lane/provider/cooldown. Render lane labels and refresh results. Make bulk pause/resume return and display per-item applied/skipped/errors. Keep existing normal-note buttons and request bodies unchanged.

- [ ] **Step 5: Run frontend regression and build**

~~~powershell
npm test -- --run
npm run build
~~~

Expected: all existing/new tests pass and TypeScript/Vite build succeeds.

- [ ] **Step 6: Commit the UI/progress slice**

~~~powershell
git add backend/src/douyin_library/api/routes/jobs.py backend/src/douyin_library/api/routes/notes.py frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/LegacyUpgradeQueue.test.tsx
git commit -m "feat: expose independent note queue progress"
~~~

### Task 7: Run three adversarial review rounds and final verification

**Files:**
- Create: \`backend/tests/integration/test_legacy_queue_adversarial.py\`
- Modify: \`backend/tests/integration/test_job_recovery.py\`
- Modify: \`frontend/src/LegacyUpgradeQueue.test.tsx\`

- [ ] **Step 1: Round 1 — scheduling and concurrency**

Test concurrent normal/legacy enqueue for one aweme, workless payload dedupe, new-note priority, continuous new arrivals, two general claimers, and the single legacy slot.

- [ ] **Step 2: Round 2 — runtime ownership and recovery**

Test the exact stale cutoff, refresh racing with late completion, running pause/cancel, GPU lease loss/reacquisition, Ollama transport failure, and repeated refresh. Assert a late fence cannot publish and repeated refresh cannot create a fresh cooled-failure job.

- [ ] **Step 3: Round 3 — data and UI truth**

Test semantic-gate failure, insufficient input, unchanged old-note/source hashes, queue/audit/filesystem disagreement, mixed multi-select, and jobs outside the recent list window. Assert \`preserved_existing\`, independent progress, and per-item bulk results.

- [ ] **Step 4: Run complete verification**

From \`backend\`:

~~~powershell
uv run pytest -q
uv run ruff check src tests
python -m compileall -q src
~~~

From \`frontend\`:

~~~powershell
npm test -- --run
npm run build
~~~

From the repository root:

~~~powershell
git diff --check
~~~

- [ ] **Step 5: Confirm safety boundaries**

Verify test logs contain no worker/Ollama/API startup, no production storage writes, and no unrelated staged files. Report existing dirty changes separately from this implementation.

- [ ] **Step 6: Commit adversarial tests**

~~~powershell
git add backend/tests/integration/test_legacy_queue_adversarial.py backend/tests/integration/test_job_recovery.py frontend/src/LegacyUpgradeQueue.test.tsx
git commit -m "test: adversarially review legacy queue isolation"
~~~

