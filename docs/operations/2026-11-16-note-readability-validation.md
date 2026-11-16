# Note Readability Implementation and Validation

Implementation plan: `docs/superpowers/plans/2026-11-16-note-readability-synthesis.md`.

## Implementation

The six steps were implemented in their specified order:

1. Replaced the synthesis prompt instruction with the exact three-line wording
   from the plan; bumped `SYNTHESIZE_PROMPT_VERSION` to `evidence-synthesize-v2`.
2. Used rendering option (a): retained the eight teaching chapters, combined the
   four synthesis groups into one concise theme review, preserved promotions and
   the raw source timeline, and deduplicated complete quotations. Rendered the
   existing structured steps field, which was previously omitted. Empty nested
   groups no longer emit repeated missing-information placeholders.
3. Added `_methods_section_suspicious` with the specified sequence-signal regex
   and threshold of four matches. An empty methods bucket produces
   `methods_steps_suspicious_missing`; no automatic repair was added.
4. Added the three deterministic readability checks: at least three missing
   placeholders, more than 20% short verbless core points, and more than 900 output
   characters per unique atom. Equal-weight binary components produce a score
   of 0, 33, 67, or 100. Readability warnings downgrade an evidence-passing bundle
   to `passed_with_warnings`, with the score in the Markdown header and
   `note_score.json`, and warning codes in `generation_audit.json`. Evidence
   coverage remains a separate hard gate. Header scoring converges before the
   bundle is returned, including at the character-budget boundary.
5. Added a glossary using synthesis concepts and repeated Chinese phrases with
   stopword filtering, using only the standard library. Added a timestamped ASR
   doubt list for uncertain atoms and unintelligible segments, including segments
   without extracted atoms.
6. Added the exact outline and section instructions from the plan and bumped
   `OUTLINE_PROMPT_VERSION` and `SECTION_PROMPT_VERSION` to v2.

## Files

- `backend/src/douyin_library/notes/prompts.py`
- `backend/src/douyin_library/notes/render.py`
- `backend/src/douyin_library/notes/coverage.py`
- `backend/src/douyin_library/notes/service.py`
- `backend/src/douyin_library/notes/state.py`
- `backend/src/douyin_library/jobs/handlers.py`
- `backend/tests/unit/notes/test_readability.py`
- `backend/tests/unit/notes/test_render.py`
- `backend/tests/unit/notes/test_teaching_prompts.py`
- `backend/tests/integration/test_teaching_note_regression.py`
- This validation record.

## Automated Verification

Executed from `backend`:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -k note -q
```

- Before implementation: 599 passed, 348 deselected.
- After implementation and fixes: 623 passed, 348 deselected, one existing SQLite
  datetime-adapter deprecation warning; 42.50 seconds.
- Focused Ruff checks passed for `coverage.py`, `render.py`, and
  `test_readability.py`.
- The task-only patch passed whitespace validation.
- Independent code review and re-review found no remaining actionable findings;
  the review also checked 21 combinations around readability score boundaries.

## Live Regeneration

Executed the primary sample commands from `backend`, once each:

```powershell
.venv\Scripts\python.exe -m douyin_library.cli generate-notes --dry-run --aweme-id 6978749729874218247
.venv\Scripts\python.exe -m douyin_library.cli generate-notes --force --aweme-id 6978749729874218247
```

The dry run skipped the existing note. The force command exited 0 but published
no new semantic note: `generated=0`, `skipped=1`,
`status=preserved_stronger_existing`. Processing audit 7901 records an Ollama
evidence-gate failure followed by rule fallback; the publication guard retained
the stronger existing semantic bundle. Extraction retry feedback included
`extraction atom mapping does not match its source segment`; later review retry
feedback reported incomplete `reviewed_claim_ids`. No GPU-contention error was
reported, and the command was not retried. All 17 checked original source/bundle
file SHA256 hashes remained unchanged.

Three additional samples were selected reproducibly with seed 20261116 and run
in one force command:

```powershell
.venv\Scripts\python.exe -m douyin_library.cli generate-notes --force --aweme-id 7560638582618508603 --aweme-id 7561631117192023348 --aweme-id 7631186363039187377
```

| Sample | Processing audit | Result |
| --- | --- | --- |
| 7560638582618508603 | 7904 | Evidence gate failed; rule fallback published. |
| 7561631117192023348 | 7906 | Evidence gate failed; rule fallback published. |
| 7631186363039187377 | 7908 | Ollama semantic generation published; `passed_with_warnings`, `semantic_complete=true`. |

The three-sample command exited 0: `generated=3`, `skipped=0`, `failed=0`.
The first two are explicitly rule fallbacks and do not satisfy semantic-note
acceptance. All three publications retained prior versions. No command-level
retry was performed.

The third publication has 272 lines (previously 362), 23 atoms, readability 67,
four missing-information placeholders, and no unknown atom references. Its
methods section is nonempty. However, one full atom excerpt occurs five times,
so the published output does not meet the repetition limit. This command began
at 17:01:52 local time and imported the renderer before its final 17:02:55 fix.
Reconstructing audit 7908 and rendering it with the current code produces 188
lines, preserves every raw segment, and has no atom excerpt occurring more than
twice. The four placeholders and readability warning remain. That final-code
check did not publish or rerun the model; the on-disk published note remains the
272-line result from the earlier renderer.

## Archived Renderer Checks

As supplemental verification, prior audited semantic generations were rendered
with the new renderer without publishing the results:

| Sample | Previous lines | New lines | Missing placeholders | Readability | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 6978749729874218247 | 630 | 323 | 3 | 67 | passed_with_warnings |
| 7631186363039187377 | 362 | 188 | 4 | 67 | passed_with_warnings |

Both preserved every raw segment and had no atom excerpt occurring more than
twice. The other two selected samples had old fallback-schema audits, so they
could not reconstruct a semantic `SynthesisResult` for this supplemental check.

These are renderer-only results. They do not establish fresh-generation
acceptance. The primary sample's four-method classification and at-most-two
missing-placeholder requirements remain unverified for a fresh semantic note;
the archived rendering still has three placeholders and exceeds the expected
300-line target. Source text was not truncated to satisfy the line target.

## Scope and Deviations

- The warning status also required small compatibility changes in `state.py` and
  `jobs/handlers.py`, so a semantically complete warning-status bundle is not
  mistaken for a failed or legacy note. Evidence, depth, and completeness checks
  remain required.
- Readability is computed against the rendered Markdown in `render.py` and
  propagated through `service.py`, ensuring artifacts and audits agree.
- Complete quotations are deduplicated; short literals embedded in raw source
  text are preserved even if the original source repeats them. A universal
  substring-count limit would conflict with source fidelity.
- Existing extraction/review/repair validation, dependencies, and source assets
  were not changed by this task. No media was downloaded and only the four
  explicitly selected notes were targeted for regeneration.
- This checkout already contained substantial uncommitted changes, including
  prerequisite edits in the touched files and an unrelated staged test rename.
  The task-only patch was saved separately. Eight touched files already differed
  from `HEAD` by 1,066 insertions and 94 deletions before this task. These changes
  depend on additional uncommitted publisher, queue, control, and lease code;
  committing only the touched files would not create a standalone validated
  snapshot. No commit was created: approval to include those existing changes
  has not been received. The staged rename and other existing work have been
  preserved. Intended task commit message:
  `fix(notes): improve synthesis readability and warning gates`.
