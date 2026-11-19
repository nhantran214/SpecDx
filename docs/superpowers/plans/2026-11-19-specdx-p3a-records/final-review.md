# P3a final review (d8ad819..add6d72)

Reviewer: final review (opus), 2026-11-20. Read-only, apart from this file. Suite run once: 745 passed, 1 skipped, 96 deselected; `ruff check specdx tests scripts experiments` is clean, and the working tree was clean before and after.

Evidence scripts (not committed) are in the session scratchpad: `probe.py`, `align.py`, `spot.py`, `weak.py`, `prompts.py`, `strong.py`. All of them read only `experiments/p3-sopbench-trajectories.json`, `work/sopbench-output/` (the worktree's copy) and `results/p3/replication.data.json`.

## Strengths

- **Faithful to the method.** The offline path calls `fit_anchored` and `type_cells` exactly as `diagnose_v3` does. `soft_hitting_sets` gets the same `tau=0.05`, `max_size=3` and default `min_mass`, over every run of the scenario, weighted by `fit.weights[run_key]`, which is what `scores.hitting_candidates` does (ruling I1). The rest also matches v3:
  - Contradiction witnesses are the runs whose V(b) meets H, sorted (ruling I2).
  - Singletons are left to the cell path.
  - Scores equal v3's support.
  - Merging is unnecessary, because every S3 location is task-specific, so `merge_defects(identity)` would be a no-op.
- **Polarity is right, checked against the oracle source.** In `vendor/sopbench/env/evaluator.py:268-271`, `action_successfully_called` is true only when the target action was called and the agent-side system returned truthy. It is not the "matches expectation" flag, which is `action_called_correctly`. So reading `succeeds` from `action_successfully_called` and anchoring on `action_should_succeed` is correct (`cells.py:50,66`).
- **Alignment holds end to end on the real data.** For all 21,323 evaluated runs in 7 domains, the vendored task's `should_succeed` equals the evaluator's own stored `evaluations[0].action_should_succeed`, and `user_goal` equals both `evaluations[0].user_goal` and `record.task.user_goal`. There are 0 mismatches (`align.py`). The vendored `action_should_succeed` values are all real bools (830), so `bool()` is safe.
- **The spot checks match the raw trajectories.**
  - `bank:12:outcome` (cancel_credit_card, should succeed): all 26 raters refuse. gpt-4.1 checks the user, checks the card, sees balance 0, then logs out without cancelling.
  - `library:13` (borrow_book, should fail): every rater borrows.
  - `hotel:102` (modify_reservation, should fail): 16 of 25 modify.

  These are real, population-wide behaviours, not join or polarity errors.
- **Stage reports follow §13c.** They have the header rows, metadata JSON and `data.json`, and the dashboard is regenerated after each stage. The pid file is removed by an EXIT trap. `P3_RESULTS`/`P3_WORK` reach both bash and the CLI, and tests never write into the repository (ruling F4). `local_stage` takes an injected runner, so there is no recursive pytest (ruling F3).
- **Network access is confined to two injected callables** (`http_get`, `github_tree`), and a truncated tree listing is refused.

## Issues

### Critical

None. No run is joined to the wrong task, no anchor or reading is inverted, and the core diagnosis matches `diagnose_v3` apart from the ruled adaptations.

### Important

**I-1. `compose()` assigns the open and hosted compositions from the raw model name, so two open-weight models count as hosted in 5 of 7 domains.**
- Where: `specdx/subjects/sopbench/diagnose.py:78,80`, with `load.py:94` (`model=name.model`).
- What is wrong: `OPEN_WEIGHTS` is keyed by the canonical name, but `S3Run.model` is the raw file name. So `llama3.1-70b-it-fireworks` and `qwen2.5-72b-it-fireworks`, which are in the manifest for bank, dmv, healthcare, library and online_market, fail `r.model in OPEN_WEIGHTS`.
- Evidence: Table 1 gives bank `open` 9 raters and `hosted` 17. The correct split is 11 and 15. hotel and university are right only because they have no fireworks files.
- Why it matters: the `open`/`hosted` rows of Table 1 are wrong. §8.4's "detection versus population composition" and H5 consume them. θ in `data.json` is also keyed by the raw name, so the same model has different ids across domains, which breaks any cross-domain aggregation in P3b.
- Fix:
  - Add `canonical` to `S3Run`, filled from `TrajectoryName.canonical`, and use it as the rater id (`run_key`, `Reading.model`, the `models` map). Alternatively, set `model=name.canonical` directly.
  - Have `compose` test `canonical`.
  - Add a fireworks-named run to `test_compose_splits_open_and_hosted_raters`.

**I-2. Files and records without `evaluations` are dropped silently. The report says "Excluded files: None" while 5 whole model–domain files give zero runs.**
- Where: `specdx/report/p3.py:139` (`loaded, _ = load_runs(...)` discards `LoadStats`).
- Evidence (`probe`/`align`):
  - `gpt-5-mini` has 0 evaluated records in bank, dmv and university. Each file has 3 interactions per record and no evaluations.
  - `qwen2.5-72b-instruct` (react) has 0 in hotel and university.
  - Stray records: hotel gemini-1.5-pro has 2, online_market deepseek-r1 has 3.
  - So fetch reports "187 files, 28 models", while replication works on 24–27 raters per domain. The only trace is the `raters` column. The Task 9 ledger line "No files were excluded" is therefore inaccurate.
- Why it matters: the population is misstated. A reader cannot tell that gpt-5-mini is missing from 3 domains.
- Fix:
  - Aggregate `LoadStats` per domain.
  - List every zero-run file under "Excluded files" with the reason "no evaluations (unevaluated upstream)".
  - Add a `missing evaluations` count to Table 1 or to a per-domain line.
  - Add a test with a fake file that has no evaluations.

**I-3. The prompt-version ruling (Task 9) is not implemented, and as worded it needs refining.**
- Where: `specdx/subjects/sopbench/cells.py:33-34,71` (variant always 0; T is the first run's prompt).
- Evidence (`prompts.py`):
  - In bank, dmv, healthcare, library and online_market, the same 8–10 earliest-generation models saw the old render: claude-3-5-sonnet, gemini-1.5-pro, gemini-2.0-flash(-thinking), gpt-4o(-mini), llama3.1-70b, qwen2.5-72b, deepseek-r1, and o1 in healthcare. Every other model saw the new render. hotel and university are uniform.
  - The version is not constant within a file. deepseek-r1 has 111 old and 23 new prompts in bank, and 122 old and 2 new in healthcare.
  - `tell_for` takes the first file in path order, which is claude-3-5-sonnet, so **T is the minority (old) render** in 5 domains.
- Why it matters:
  - The variant must be per run, not per file.
  - It is confounded with model generation, so offline a variant effect cannot be separated from α. The paper must say so rather than claim the variant is controlled.
  - T is the text that ddmin localises against in P3b, so it should be the majority render.
- Fix:
  - Compute the variant per run, as 0 for the majority render of that task and 1 otherwise.
  - Build T from the majority render.
  - Report the split per domain with the model list.

**I-4. Replication never checks the downloaded files against the manifest's sha256.**
- Where: `specdx/report/p3.py:131-143`.
- What is wrong: the manifest records a sha256 per file, but the stage loads whatever sits at `work/sopbench-output/<domain>/<name>`. Combined with Task 2's non-atomic write, the one unverified link in manifest → selection → load is the bytes actually read. The Task 9 "all 187 sha256 verified" was a manual controller check.
- Fix: hash each file before `load_runs`, and exclude it with the reason `sha256 mismatch` (and fail the stage, exit 2, if any mismatch). This costs a few seconds for 942 MB.

**I-5. The duplicate-prompt alignment guard (Task 3 minor (a)) is load-bearing, not cosmetic.**
- Where: `specdx/subjects/sopbench/load.py:84`.
- What is wrong: university tasks 0–6 share one `user_prompt` but differ in `action_should_succeed` (1 True, 6 False) and have 7 distinct initial databases. A shift inside that block passes the guard and **inverts the outcome anchor**.
- Current exposure: the real data are clean (see Strengths), so this is not a live defect.
- Fix: also compare `action_should_succeed`, `user_known` and `constraint_parameters` (all three round-trip exactly, 42/42 in university), and `user_goal` against the group key.

**I-6 (spec §13d; raise with the author, not an implementation defect). Most `weak` findings are structural.**
- Where: `cells.py:65` implements §13d's `clean = db_match ∧ constraints ∧ process`.
- Why the findings are structural: the evaluator replays the agent's *own* calls on the strict system. A refusal on a should-succeed task therefore has all three checks satisfied, so it is clean, and `weak_finding` fires on the outcome cell whenever any rater refuses. But SOPBench's real C does catch this case, through `action_called_correctly` (`evaluator.py:292-300`).
- Numbers: 112 of the 145 weak findings (all composition) are on should-succeed outcome cells. The other 33, on should-fail cells, are the interesting ones. There the agent performed the forbidden action and the strict replay agreed, which means the strict system permitted it. That is a genuine C-internal inconsistency candidate and a good lead for P3b.
- Fix options:
  - Have the outcome reading's `clean` also require `succeeded == should_succeed`.
  - Or drop weak typing on outcome cells for S3.
  - Or report the two kinds separately.

  Decide before P3b's C-mutant scoring of weak checks.

### Minor

- **M-1. An un-ruled third offline adaptation.** `diagnose.py` skips v3's observability stage 1 and `_note`. S3 cells have `steps=()`, so `diagnose_v3` would type every finding UNOBSERVABLE, and `_note` would crash on `steps[0]`. The adaptation is correct, since every non-error run reads all four cells. Document it in the `diagnose_domain` docstring and add it to the ledger beside I1 and I2.
- **M-2. Saturated scores make Table 2 and the key number misleading.**
  - 719 of 896 hidden findings have score 1.000, so the "10 highest" are just the first ties in location-string order (`p3.py:173`).
  - "1573 findings" counts each co-occurring ambiguity twice and up to 4 cells per task. The 896 hidden findings fall on 513 distinct tasks: hotel 186/195, library 59/66, online_market 99/172, bank 57/134, dmv 45/97, healthcare 43/124, university 24/42.
  - Fix: add "findings at ≥0.999", distinct tasks per type, and a secondary sort key (for example, the number of deviating readings).
- **M-3. The skip-when-done rule ignores the commit** (`stages.py:193`). After the fix wave, `./run_all.sh all` will skip `replication` and keep the pre-fix report. Delete `results/p3/replication.json` and `local.json` before re-running, or add `--force`. A cheap guard is to warn when the recorded commit ≠ HEAD.
- **M-4. Any exception becomes exit 2** (`stages.py:175-177`), which conflates a crash with §13c's "usage or environment error". Acceptable, but say so in RUNNING.
- **M-5. Records with `evaluations` but no `interactions`** raise `IndexError` at `load.py:102`, and the whole file is excluded instead of the record being skipped. This was not observed: such records also lack evaluations in the real data.
- **M-6. `select` cannot know that a preferred-mode file is unevaluated.** For example, gpt-5-mini fc is kept even if a lower-ranked file were evaluated. This cannot be fixed from names alone. Disclose it via I-2.
- **M-7. The `status` row is always "not run"**, because `p3 status` writes no metadata. Table 1's plain `contradiction` column is always 0 offline. Mark both "n/a offline".
- **M-8. `--manifest` and `--results` defaults are cwd-relative** (`cli.py` p3 parser). `specdx p3 replication` run outside the repo root fails with exit 2. `run_all.sh` does `cd "$ROOT"`, so the script itself is fine.
- **M-9. Test gaps.** None of these hides a live bug except the first.
  - `compose` is tested only on canonical names, which hid I-1.
  - No shell-level test covers `all`: skip on done, stop at the first failure, exit-code propagation.
  - The replication tests check headings, not numbers (Task 7 deferred (2)).
  - No test covers a missing-evaluation file (I-2).
- **M-10. Global constraints still say `ruff check .`,** while the Task 1 ruling narrowed it to `specdx tests scripts experiments`. Update `global-constraints.md` so later plans do not re-litigate it.

## Triage of the deferred minors

| Task | Item | Verdict |
|---|---|---|
| T1 | VENDORED.md:11 "and nothing else" is false now | **Fix before merge** (one line) |
| T1 | mktemp dirs not removed; `-x __pycache__`; the pin test ignores verify_vendor.sh | Can stay deferred |
| T2 | `fetched_at` churn; opener exception raw; no re-download/tie-break tests | Can stay deferred |
| T2 | non-atomic write | Deferred **only if I-4 lands** (the sha check at load makes it self-detecting); otherwise fix |
| T3 (a) | duplicate `user_prompt` guard | **Fix before merge** (I-5): university 0–6 differ in `action_should_succeed` |
| T3 (b)–(e) | raw KeyError (already caught per file in replication); domain-file test; cosmetics | Can stay deferred |
| T4 (a), (b) | `cells_for` KeyError; domain cross-check | Can stay deferred (replication passes consistent inputs) |
| T4 (c), (d) | raw-string prompt compare; tool-error runs in Tell | Subsumed by I-3 (fix with it) |
| T5 | triple reporting | Stay, but report distinct tasks (M-2) |
| T5 | tool-error test in `diagnose_domain`; ambiguity test near the transition | Can stay deferred |
| T5 | `PROVISIONAL` attached by Task 7 | Done (verified in `replication.md`) |
| T6 (1), (2), (4), (5), (6) | work fallback; pid ≤ 0; stale data.json; VERDICTS unused; test gaps | Can stay deferred |
| T6 (3) | stale pid files | Done (EXIT trap) |
| T7 (5) | all files excluded → still DONE | **Fix before merge**: with an empty `work/sopbench-output/<domain>`, every file becomes "excluded", `compose` yields [] and the stage reports `DONE, 0 findings`. It should be exit 2. Same for a domain with zero readings. |
| T7 (1) | missing trajectory file | Fine once (5) is fixed (it surfaces as excluded) |
| T7 (2) | numeric assertion; stray `print` | **Fix with I-2's test** (cheap) |
| T7 (3), (4), (6), (7) | | Can stay deferred |
| T8 (1) | RUNNING §13 still gives live `run_all.sh g3/gate/study/all --yes` instructions (RUNNING.md:227-244) | **Fix before merge**: add a "historical, these commands no longer exist" banner |
| T8 (2) | `cli.py:85` hint "run ./run_all.sh freeze-baselines" names a removed stage; the skip reason in test_preregistration_baselines.py:31 | **Fix before merge** (cheap, user-facing) |
| T8 (3) | obsolete README troubleshooting row | **Fix before merge** (cheap) |
| T8 (4)–(6) | `--help` exit 2; rebuild_status failure loses the exit code; "(pending)" drift | Can stay deferred |
| T9 | "(dirty)" headers | Can stay deferred (re-run after the fix wave anyway, M-3) |
| T9 | hotel weak 2 vs 22 across compositions | Defer to P3b, and read it with I-1 and I-6 in mind |
| T9 | per-version prompt split | **Fix before merge** (I-3) |

## Disagreements with rulings

1. **Task 9, concern 2 ("scientific, not a code defect; carry to P3b's recalibration").** I agree it is not a code defect in the core path. I disagree that recalibrating V3Params is the remedy, and with the three-cause reading as stated. The evidence points at the §13d/§13a.2 design applied to this data:
   - **Base rates.** The mean per-cell deviation from the anchor is 25–35% (bank: outcome 0.25, constraints 0.31, process 0.34). Hidden findings cover 186/195 hotel tasks and 59/66 library tasks. A detector that flags 95% of tasks has no discriminative power at any τ_H.
   - **The anchor is overwhelmed in library.** θ ranks qwen2.5-14b (0.95) and qwen3.5-4b (0.91) highest, and **gpt-5 lowest (0.52)**: the fit has adopted the majority reading as the truth. §13b's "no θ collapse" gate should count a θ-ordering inversion of this kind as a warning sign.
   - **Hidden findings even when the strongest raters agree.** In hotel, 95 of 233 hidden findings have all top-5-θ raters (gpt-5, gpt-5-mini, o4-mini-high, …) agreeing with the key. They arise through the ambiguous-pick path (532 of 896 hidden findings are ambiguous-pick), because a pair explains both camps.
   - **Correlated readings.** One agent act gives up to 4 readings by construction (a forbidden action → outcome, db_match, constraints and process all deviate), which breaks conditional independence and inflates confidence.
   - **Pooled β.** β is shared across all tasks of a check or goal, so per-task difficulty heterogeneity, which SOPBench is built to have, is absorbed as defects.

   Recommendation for P3b, before any calibration:
   - a negative control, for example anchors permuted within `user_goal`, or the frontier-only population, checking whether the hidden rate drops;
   - a check of whether the hidden rate simply tracks per-task SOPBench success;
   - reporting distinct tasks.

   Raise this with the author as a design question about §13d, not only as calibration.
2. **Task 9 prompt-version ruling.** I agree in substance, but "0 for the majority, 1 for the other" must be per run (files are mixed). The ledger should also record that the variant is confounded with model generation, and that T was the minority render (I-3).
3. **Task 9 "No files were excluded".** This is accurate for TrajectoryMismatch and TrajectoryError, but 5 files gave zero runs (I-2). Correct the ledger line.
4. **Ruling I2.** I agree. Minor: the always-zero plain `contradiction` column should say "n/a offline" (M-7).

## Declined to judge

- **Byte identity of `vendor/sopbench`.** It is excluded from the package; I had no clone and no network.
- **Whether the manifest's sha256 values match upstream.** That needs network access.
- **Whether the top findings are true SOPBench defects** (for example, bank:12, where every model refuses a zero-balance card cancellation, or library:13). That needs reading the rendered SOP and human judgment. I only checked that the readings faithfully reflect the trajectories.
- **The statistical behaviour of `fit_anchored` and `type_cells` themselves.** They are unchanged code outside this plan, so I judged only how P3a feeds them.
- **RUNNING and README prose** beyond the deferred items, and the paper framing.

## Assessment

**Ready to merge: With fixes.**

The S3 chain is correct where it matters most. There are no mis-joins, polarity and anchors are right, and alignment is verified against the evaluator's own fields for every run. The offline diagnosis is `diagnose_v3` minus refutation, plus the two ruled adaptations and one undocumented but sound adaptation (M-1). The runner is sound.

Before merge, fix:
- I-1 (open/hosted composition);
- I-2 (silent model drops);
- I-3 (the prompt variant, already ruled for this wave);
- I-4 (sha check at load);
- I-5 (the alignment guard);
- T7 (5) (all excluded → DONE);
- the cheap documentation and hint items (T1, T8 (1)–(3)).

Then delete `results/p3/replication.json` and re-run `./run_all.sh replication` (M-3).

I-6 and disagreement 1 need an author decision, not code, and should be settled before P3b builds on the replication. The current replication numbers should not be read as findings. They show that, on SOPBench's published population, the unchanged method with per-task cells flags nearly every task.
