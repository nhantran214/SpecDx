# SDD ledger — plan: docs/superpowers/plans/2026-11-19-specdx-p3a-s3-offline.md

Spec: `docs/superpowers/specs/2026-11-19-specdx-s2-s3-design.md`. §5.2, §8.4, §13c and §13d are binding, and §13d overrides §5.2.

Worktree: `/home/ethan/AILab/SpecDx/.claude/worktrees/specdx-p3a`, branch `worktree-specdx-p3a`, base d8ad819 (master).

## Pre-flight scan

| Pair or task | What is produced and consumed | Finding |
|---|---|---|
| T1 → T2, T3 | `SOPBENCH_COMMIT`, `SOPBENCH_DIR`, `DOMAINS` | consistent |
| T2 → T3 | `TrajectoryName`; `TrajectoryError` (defined in T2, imported by T3) | consistent after the plan fix |
| T3 → T4 | `S3Task(user_prompt, …)`, `S3Run(run_key, …)` | consistent |
| T4 → T5 | `S3CellData(data, violated, excluded_tool_error)` | consistent |
| T2 → T5 | `OPEN_WEIGHTS`, used by `compose()` | consistent |
| T6 → T7, T8 | `STAGES`, `run_stage`, `already_done`, `update_status` | consistent. T8 adds `status --done` to the T7 CLI, and lists `cli.py` among its files |
| T7 → T8 | `specdx p3 <stage>` | consistent |
| T1 itself | the test imports `REPO_ROOT` | it comes from `specdx.config` |
| T1 environment | the worktree has no `.venv` | F1 |
| T5 itself | the tests assert concrete defect types that depend on V3Params behaviour | F2 |
| T7 itself | `local_stage` runs pytest as a subprocess | F3 |
| T8 itself | the tests run `run_all.sh`, which rebuilds `results/p3/STATUS.md` inside the repository | F4 |
| T9 itself | needs network, and downloads about 1–2 GB into the worktree's `work/` | F5 |

## Rulings

- **F1.** Ruling: Task 1's implementer first creates the venv (`python3.13 -m venv .venv && .venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"`) and records the baseline `-m "not db"` count. On master it is 702 passed, 1 skipped, 96 deselected.
  - Why: the worktree is fresh.
  - If wrong: no cost.
- **F2.** Ruling: Task 5's tests describe what the model really does at V3Params. If a constructed scenario does not yield the planned type, the implementer strengthens the scenario (more raters, families or tasks) and reports it. They never change `specdx/infer/`.
  - Why: the design keeps the method unchanged.
  - If wrong: a weaker test.
- **F3.** Ruling: `local_stage` takes an injectable command runner. Its unit test injects a fake runner and never runs pytest recursively.
  - Why: this avoids running the suite from inside the suite.
  - If wrong: no cost.
- **F4.** Ruling: `run_all.sh` and `specdx p3` honour `P3_RESULTS` (default `results/p3`) and `P3_WORK` (default `work`). Tests point both at `tmp_path`.
  - Why: tests must not write into the repository.
  - If wrong: no cost.
- **F5.** Ruling: Task 9 runs on this machine with network access. Downloads go to the worktree's git-ignored `work/sopbench-output`, and only the manifest is committed.
  - Why: `results/` and `work/` are untracked.
  - If wrong: re-download.

## Tasks
Task 1: dispatched (sonnet), BASE d8ad819
Task 1: DONE_WITH_CONCERNS at eac885d. The suite went from 702 to 705 passed.
  Ruling (concern 1): the worktree venv also installs `.[baselines]`, and the git-ignored `results/` was copied from the main checkout, because existing tests read results/p2-*. This amends F1. — Why: the master baseline needs both. — If wrong: none (results/ is untracked).
  Ruling (concern 2): the "ruff clean" requirement is now `ruff check specdx tests scripts experiments`. The existing E501 in paper/figures/drawio/gen_figures.py:163 (from commit 0d951a6) is out of scope. — Why: it is not in this plan's files. — If wrong: one line left unfixed.
  Ruling (concern 3): the commit trailer stays as instructed. — Why: it follows the session's attribution rule. — If wrong: cosmetic.
Task 1: review dispatched (sonnet). The package is review-task1.diff, which excludes vendor/sopbench (56 files, 297,933 lines, checked by verify_vendor.sh).
Task 1: minor (deferred), four items:
  - VENDORED.md:14 says "and nothing else", which is no longer true;
  - the mktemp dirs in verify_vendor.sh are never removed;
  - `-x __pycache__` could hide a stray directory;
  - the pin test does not check verify_vendor.sh.
Task 1: complete (commits d8ad819..eac885d, review clean)
Task 2: dispatched (sonnet), BASE eac885d
Task 2: DONE at 1d2ef62 (715 passed; the real 455-entry tree raises no unknown model). Review dispatched (sonnet).
Task 2: the review approved the task. task-2-report.md was missing, so the implementer was asked to write it (no code change).
Task 2: minor (deferred):
  - fetched_at changes even when nothing was downloaded;
  - the file write is not atomic, but self-heals through the sha256 check;
  - no test covers a re-download after a sha256 mismatch, or the lexical-path tie-break;
  - an opener exception propagates raw; Task 7's CLI handles it.
Task 2: complete (commits eac885d..1d2ef62, review clean). task-2-report.md has now been written.
Task 3: dispatched (sonnet), BASE 1d2ef62
Task 3: DONE at d9f12ab. Suite: 720 passed. The real library file loads with 66 records and 66 runs. Review dispatched (sonnet).
Task 3: minor (deferred; flagged for the final fix wave):
  - (a) The alignment check compares only user_prompt. hotel has 1 duplicated prompt and university has 5, so a shift inside a run of identical prompts goes undetected. A stronger guard also compares record["task"]["action_should_succeed"], and if needed user_instruction, with the task.
  - (b) Malformed records raise a raw KeyError or IndexError; wrap them in TrajectoryError.
  - (c) There is no test for a missing domain file.
  - (d) A redundant Path().
  - (e) The test's tuple assertions are hard to read.
Task 3: complete (commits 1d2ef62..d9f12ab, review clean)
Task 4: dispatched (sonnet), BASE d9f12ab
Task 4: DONE at faf4cf0. Suite: 725 passed. Review dispatched (sonnet).
Task 4: minor (deferred; flagged for the final fix wave):
  - (a) cells_for does not check that a run's task is in `tasks`, so `by_cell` raises KeyError;
  - (b) the domains of the runs and of `tell_for` are not cross-checked;
  - (c) prompt disagreement is compared on raw strings (disclosed);
  - (d) tool-error runs feed the Tell corpus (disclosed).
Task 4: complete (commits d9f12ab..faf4cf0, review clean)
Task 5: dispatched (opus), BASE faf4cf0. Stopped by a rate limit with nothing written; resumed on 2026-11-20.
Task 5: DONE_WITH_CONCERNS at dc1d01d. Suite: 730 passed.
  - F2 applied: the ambiguity scenario needs at least 12 agreeing tasks; the test uses 20.
  - Concerns:
    - DEFAULT_PARAMS replaces the default argument (ruff B008);
    - a check cell inside a contradiction is also reported as hidden, which triple-counts;
    - for P3b, ambiguity needs about 12 or more tasks per rater.
  - Review dispatched (opus); duplicate reporting is a named risk.
Task 5: review returned Needs fixes, with 2 plan-mandated Important findings.
  - **I1.** The plan has the hitting sets read only runs with a non-empty V(b). `diagnose_v3`'s `hitting_candidates` reads every run of the scenario, so its clean runs dilute the mass. The plan's filter creates false contradictions.
    Ruling: follow the method. Pass every run of the task, clean or not, with `fit.weights`, exactly as `hitting_candidates` does. Add a test in which clean runs plus two runs with disjoint single violations give no contradiction.
    Why: the spec (the design) keeps the method unchanged, and the plan's filter was a slip.
    If wrong: fewer contradiction candidates.
  - **I2.** The plan reports unrefuted multi-clause candidates as contradictions with no note. `diagnose_v3` reports a contradiction only after a refutation attempt fails, and otherwise records it in `spent`.
    Ruling: keep reporting them offline, because no executor exists to refute them. Each one carries `note="refutation not attempted (offline)"`, and its witnesses are the run keys whose V(b) meets H, sorted, as `diagnose_v3` does. Task 7's tables must count them separately as unrefuted candidates.
    Why: without them the offline replication could never show a contradiction, and the note keeps them honest.
    If wrong: a relabel.
  - Task 5: minor (deferred):
    - triple reporting matches the method; the reports explain it;
    - there is no test for a tool-error run inside `diagnose_domain`;
    - the ambiguity test sits near the 12-task transition;
    - `PROVISIONAL` must be attached by Task 7.
Task 5: fix round 1 dispatched (resumed implementer), FIX_BASE dc1d01d.
Task 5: fix round 1/5: 2 addressed, 0 open (I1 hitting sets over every run; I2 offline note and witnesses); commits dc1d01d..ad47e45. The re-reviewer ran with the system python (no psycopg). The controller ran tests/subjects with .venv: 29 passed.
Task 5: complete (commits faf4cf0..ad47e45, review clean after 1 fix round)
Task 6: dispatched (sonnet), BASE ad47e45
Task 6: DONE at a964702. Suite: 737 passed. Deviations: an extra optional `git` callable, and a `work` parameter that defaults to results.parent/"work", so Task 7 must pass `work`. Review dispatched (sonnet).
Task 6: minor (deferred), 6 items:
  - (1) the `work` fallback is results.parent/work. Carried into Task 7/8: always pass `work`.
  - (2) `_pid_alive` accepts pid ≤ 0.
  - (3) stale pid files: run_all.sh must remove them (carried into Task 8).
  - (4) a stale data.json is left when data is None.
  - (5) VERDICTS is unused.
  - (6) test gaps: truncation, failed rows, pid precedence, Subject/Split, escaping.
Task 6: complete (commits ad47e45..a964702, review clean)
Task 7: dispatched (sonnet), BASE a964702. Rulings carried into the dispatch:
  - F3 and F4;
  - always pass `work`;
  - I2: unrefuted contradictions get their own column, plus a "not refuted offline" line;
  - fetch network errors give exit 2.
Task 7: DONE_WITH_CONCERNS at 0da801a. Suite: 745 passed.
  - local's ruff check is scoped to specdx/tests/scripts/experiments, consistent with the Task 1 ruling.
  - A regex in test_docs was widened to allow digits.
  - Table 1 still has a plain contradiction column, which is always 0 offline.
  - Review dispatched (sonnet).
Task 7: minor (deferred; flagged for the final fix wave):
  - (1) A missing manifest file raises an uncaught OSError. It should be excluded, or give exit 2.
  - (2) The replication test does not check the unrefuted-column count numerically, and it ends with a stray print.
  - (3) The excluded-file test does not check that the excluded runs are absent.
  - (4) A missing ruff binary raises; the smoke maps exit 2 to 1; the command lines are not asserted.
  - (5) If every file is excluded, the stage still reports DONE.
  - (6) The broad KeyError/IndexError catch; the private _cell import.
  - (7) The --manifest wiring in the CLI is awkward.
Task 7: complete (commits a964702..0da801a, review clean)
Task 8: dispatched (sonnet), BASE 0da801a. Carried into the dispatch:
  - F4;
  - pid files written and removed by trap (Task 6 minor 3);
  - print each report path after its stage.
Task 8: DONE at 9e90ada. Suite: 745 passed.
  - 6 old test_docs tests were deleted (P2b and old-script content).
  - RUNNING §13 is kept as history.
  - Review dispatched (sonnet).
Task 8: minor (deferred; flagged for the final fix wave):
  - (1) RUNNING §13 still gives old run_all commands as live instructions; it needs a "historical" note.
  - (2) Stale hints in cli.py:85 and the test_preregistration_baselines skip reason.
  - (3) An obsolete README troubleshooting row.
  - (4) `--help` exits 2.
  - (5) A failing rebuild_status loses the stage's exit code.
  - (6) The usage "(pending)" marks can drift from STAGES.
Task 8: complete (commits 0da801a..9e90ada, review clean)
Task 9: dispatched (sonnet), BASE 9e90ada (real fetch and replication on this machine)
Task 9: DONE_WITH_CONCERNS at add6d72 (manifest only).
  - Run figures: fetch took 2m04s and pulled 942 MB in 187 files (28 models, 6 families); replication took 2m09s; a second `all` run skipped the finished stages. No files were excluded.
  - Controller check of concern 1 (prompt disagreement): in bank, the 27 files share 2 distinct task-0 prompts. 9 older files lack the 7-line "constraints are organized hierarchically" preamble (AND/OR/CHAIN explanation); the rest is identical. So it is a prompt *version* (an older SOPBench render), not a task-identity problem, because alignment is checked on user_prompt.
    Ruling: in the final fix wave, record the prompt version as Reading.variant (0 for the majority, 1 for the other), report the split per domain, and keep both in the fit. The variant is a Tell variant, as in v3.
    Why: different raters saw slightly different T, so this must be disclosed.
    If wrong: one more column.
  - Concern 2: Table 2 is saturated at score 1.000, with 39–233 hidden findings per domain. Likely causes:
    - systematic shared agent errors, which SOPBench is designed to provoke and which the model cannot separate from a mis-key when every family errs alike (identifiability, Kim et al.);
    - real SOPBench C bugs (e.g. bank apply_credit_card);
    - uncalibrated V3Params.
    Ruling: scientific, not a code defect. Carry it to P3b's recalibration and to the paper's threats section. Raise it with the author at the end of P3a.
  - Concern 3: no confirmed contradictions (expected offline); 2 unrefuted candidates appear in the open composition.
Task 9: minor (deferred):
  - report headers say "(dirty)" because the manifest was untracked at run time;
  - hotel weak counts differ across compositions (2 vs 22), to watch in P3b;
  - the per-version prompt split belongs to the fix wave.
Task 9: complete (commits 9e90ada..add6d72, review clean; all 187 sha256 verified)
All tasks complete. Final review next.
Final review: dispatched (opus) on review-final.diff (d8ad819..add6d72, excluding vendor and data). Its focus is whether the saturated hidden findings are an implementation artefact.
Final review: "With fixes": 0 Critical, 6 Important, 10 Minor; 7 deferred minors need fixing (final-review.md). Polarity and anchors were verified on all 21,323 runs, so the saturated hidden findings are not an implementation artefact.
Correction to the Task 9 line: 5 files gave zero runs because they have no evaluations (I-2), so "No files were excluded" was wrong.
Rulings (final fix wave):
  - **I-1.** Ruling: S3Run.model is the canonical name, and compose, θ and the raters all key on it. — Why: the fireworks duplicates are the same weights. — If wrong: none.
  - **I-2.** Ruling: surface LoadStats; a file with zero runs is listed as excluded ("no evaluations"), with a missing-evaluation count per domain and a test. — Why: no silent drops. — If wrong: none.
  - **I-3.** Ruling (amends the Task 9 ruling):
    - The prompt version is decided per run: within each task, the most common prompt is variant 0, and every other distinct prompt is variant 1, 2 and so on.
    - This goes into `Reading.variant`.
    - `tell_for` uses the variant-0 render.
    - The report gives the variant split per domain and notes that the variant is confounded with model generation.
    - Why: files mix renders. If wrong: none.
  - **I-4.** Ruling: replication checks each file's sha256 against the manifest before loading it. A mismatch excludes the file and names the reason. — Why: it closes the chain. — If wrong: none.
  - **I-5.** Ruling: alignment also compares action_should_succeed, user_known and constraint_parameters. — Why: university tasks 0–6 share a prompt but differ in their label. — If wrong: none.
  - **I-6.** Ruling (amends design §13d): clean = db_match ∧ constraints ∧ process ∧ (the outcome reading equals its anchor).
    - Why: the method's own rule says a deviant reading at an anchored value cell violates its golden clause and is not clean (v3 typing). §13d's omission was a design slip.
    - If wrong: fewer weak findings.
  - **M-1.** Ruling: document the third offline adaptation (no observability stage, because every non-error run reads all four cells) in the diagnose_domain docstring.
  - **M-2.** Ruling: the report adds a count of findings at score ≥0.999 and distinct tasks per type. Table 2 uses the number of deviating readings as a secondary sort key.
  - **M-3.** Ruling: when the recorded commit differs from HEAD, `run_all.sh` / `p3 status` warns. The fixer deletes the stale replication and local metadata and re-runs both.
  - **M-7.** Ruling: the status row and the plain contradiction column read "n/a offline".
  - **Must-fix deferred minors.** Ruling: fix T1 (VENDORED line), T7(5) (everything excluded, or zero readings, gives exit 2), T7(2) (numeric assertions, no stray print), and T8(1)–(3) (historical banner, stale hints, README row).
  - **M-10.** Ruling: the controller updates global-constraints.md to the narrowed ruff gate.
  - **Saturation (design question).** Ruling: surface it to the author at the finish as a §13d / §13a.2 design question: per-task cells, pooled β and 4 correlated readings per act. A negative control comes before P3b calibration. Not fixed here. — Why: it changes the method's design, which is the author's call. — If wrong: P3b plans around it anyway.
  - Everything else in the triage table stays deferred, as the reviewer recommends.
Final fix wave: dispatched (opus), FIX_BASE add6d72.
Final fix wave: DONE_WITH_CONCERNS. Commits 09d49fc, 60799aa and 8fe2e08; 770 passed.
  - Re-run: hidden 896 → 896; weak 145 → 0; excluded files 0 → 8 (no evaluations); bank open/hosted 9/17 → 11/15; all 187 sha256 match.
  - Concerns:
    - with I-6, S3 base can never yield a weak finding;
    - the T1 VENDORED.md edit was refused by the permission classifier and reverted;
    - §13d still says T is identical across models.
  - Scoped re-review dispatched (opus).
Final re-review: every ruled item is addressed except T1; there is no new Critical or Important breakage. The re-reviewer confirmed on real data that the new `clean` equals SOPBench's own `success` (20,567 of 20,567 runs).
Residual adjudication:
  - T1: parked. Ruling: vendor/VENDORED.md:11 stays unchanged. — Why: the vendor rule and the permission classifier forbid the edit; the author can exempt our provenance note or move it. — If wrong: one stale doc line.
  - N-1: parked. Ruling: the stale-commit warning stays, although it fires permanently for fetch-sopbench. — Why: it is minor noise, and P3b can exempt data-only stages. — If wrong: users learn to ignore the warning.
  - N-2: parked. Ruling: the §13d text gaps go into P3b's plan. The gaps are: citing the clean = SOPBench `success` equivalence; noting that weak findings vanish on base cells; and line 379 ("T identical") contradicting I-3. — Why: these are documentation fixes in a design that P3b amends anyway. — If wrong: a misleading design line until P3b.
  - Out of scope, deferred to P3b:
    - baseline/README.md:141-144 still gives the old run_all steps;
    - the 33 should-fail "forbidden action done" cases should be kept as an untyped count;
    - the stale-commit check ignores a dirty tree;
    - nothing guards against duplicate canonical entries in the manifest;
    - forward note: C-mutants that drop a check must unanchor the cell and drop its conjunct from clean, and the field-level cells are needed for db/constraints/process weak mutants.
Final review: complete. Next: finishing-a-development-branch.
