# P3a final fix wave: re-review (add6d72..8fe2e08)

Reviewer: re-review of the single fix wave, 2026-11-20. This review was read-only apart from this file.

What I did:
- Read `review-add6d72..8fe2e08.diff` once.
- Ran the touched test files only: `tests/subjects/test_sopbench_{cells,load,diagnose}.py`, `tests/report/test_p3.py` and `tests/report/test_stages.py` gave `50 passed`. I did not re-run the whole suite.
- Ran one focused data check, an I-6 equivalence check against the real trajectories (named point 1).
- Read the real re-run outputs in `results/p3/`.

## Finding Verdicts

| Item | Verdict | Where |
|---|---|---|
| I-1 canonical rater | ADDRESSED | `specdx/subjects/sopbench/load.py:114` (`model=name.canonical`); `compose` at `diagnose.py:81-88` now tests canonical names against `OPEN_WEIGHTS`. Tests: `tests/subjects/test_sopbench_load.py` (`test_a_run_is_keyed_by_the_canonical_model_name`), `tests/subjects/test_sopbench_diagnose.py` (`test_a_fireworks_file_loads_as_its_open_weight_model`). Real run: bank is open 11 / hosted 15, the split the review gave as correct. `select` keys on (domain, canonical) (`trajectories.py:119-123`), and the manifest has no duplicate (domain, canonical) pair, so no two runs share a `run_key`. |
| I-2 no silent drops | ADDRESSED | `specdx/report/p3.py:134-180` (`_load_domain` keeps `LoadStats`; a file with zero runs is excluded with `MISSING_EVALUATIONS`, `p3.py:48,162`). The Loading table is at `p3.py:293-296`. Test: `test_a_file_without_evaluations_is_excluded_and_counted`. 8 files are now listed, against the review's 5; they reconcile (named point 4). |
| I-3 prompt variants per run | ADDRESSED | `cells.py:28-36` (`_renders`: order by descending count, ties broken by the string); `cells.py:39-44` (`prompt_variants`); `cells.py:55` (`tell_for` builds T from `renders[task][0]`, which is variant 0); `cells.py:96` (`Reading.variant`). The variants are computed once per domain over all loaded runs (`p3.py:228`) and passed to every composition. The confound is disclosed at `p3.py:299-304`, and in the real report at `results/p3/replication.md:177`. Nothing in `specdx/infer` reads `Reading.variant`, so the fit is unchanged, which matches the identical `all` hidden and ambiguity counts. |
| I-4 sha256 at load | ADDRESSED | `p3.py:107-113` streams the file in 1 MiB blocks. `p3.py:152-156`: a mismatch excludes the file with `sha256 mismatch (manifest …, file …)`, and it is listed under Excluded files. `p3.py:163-164`: a missing file is excluded with its reason. The stage does not fail on a single mismatch: the ruling says "exclude and name the reason", which differs from the review's suggested exit 2. It does fail when every file is excluded. Test: `test_a_file_whose_sha256_differs_from_the_manifest_is_excluded`. |
| I-5 alignment guard | ADDRESSED | `load.py:80-88` (`_aligned` checks `user_prompt`, `user_goal`, `action_should_succeed`, `user_known` and `constraint_parameters`); `load.py:104`. Tests: the university 0↔1 swap and the parametrised per-field test. |
| I-6 clean needs the outcome anchor | ADDRESSED | `cells.py:90`. The design amendment is at `docs/superpowers/specs/2026-11-19-specdx-s2-s3-design.md:375-376`, and `global-constraints.md:36` matches it. Test: `test_clean_also_needs_the_outcome_reading_to_equal_its_anchor`. The rule is correct and faithful to v3 (named point 1). |
| M-1 third adaptation documented | ADDRESSED | `specdx/subjects/sopbench/diagnose.py:49-55`. |
| M-2 saturation, distinct tasks, secondary sort | ADDRESSED | `p3.py:122-130` (`_deviating`, `rank_findings` sorting by (-score, -deviating, key)); Table 2 uses the ranked order with a "deviating readings" column; Table 3 at `p3.py:285-291` (findings ≥0.999, distinct tasks per kind); `data.json` gets `saturation` and per-defect `deviating`. |
| M-3 stale-result warning | ADDRESSED | `specdx/report/stages.py:198-211`; `specdx/cli.py:221-224`; `docs/RUNNING.md:173`. `run_all.sh:79,138` redirects only stdout, so the warning reaches the terminal. The stale metadata was deleted, and `results/p3/replication.json` and `local.json` were re-recorded at `8fe2e08 (clean)`. |
| M-7 n/a offline | ADDRESSED | Table 1's contradiction column: `p3.py:47,249`, with the preamble at `p3.py:273-275`. The status row: `stages.py:249`. Real output: `results/p3/STATUS.md` row `status`. |
| T1 VENDORED.md:11 | **NOT ADDRESSED** (blocked) | `vendor/VENDORED.md:11` still reads "prints `vendor/tonomyx matches 34c1d2c…` and nothing else", which is false now that `verify_vendor.sh` also prints a sopbench line. The edit was refused by the permission classifier and conflicts with the "`vendor/` is never edited" rule. The implementer reverted it, and the working tree is clean. This is a residual item for the author (named point 5). |
| T7(5) all excluded or zero readings → exit 2 | ADDRESSED | `p3.py:225-227` (no runs); `p3.py:232-237` (the `all` composition has no readings, so the domain is skipped); `p3.py:326-328` returns `StageOutcome(2, "FAIL", …)`. Tests: `test_every_file_excluded_fails_with_exit_2`, `test_a_domain_with_zero_readings_fails_with_exit_2`. |
| T7(2) numeric assertions, no print | ADDRESSED | In `tests/report/test_p3.py`, the first replication test now checks (raters, runs, excluded) per composition, the hidden and unrefuted counts against `defects`, a literal Table 1 row and the Loading dict. The `print` is gone. |
| T8(1) RUNNING §13 historical banner | ADDRESSED | `docs/RUNNING.md:227`. It points to §11, which is the right section (§11 is "P3: SOPBench and τ²-bench"). |
| T8(2) stale hints | ADDRESSED | `specdx/cli.py:85-86` points to `specdx freeze-baselines`, which exists (`cli.py:157`). `tests/test_preregistration_baselines.py:31` is updated. |
| T8(3) README troubleshooting row | ADDRESSED | `README.md:243`. |

### Named points

**1. I-6: is `clean = db_match ∧ constraints ∧ process ∧ (outcome == anchor)` faithful, and is weak unreachable?**

*The rule is correct and faithful.*
- In v3, a value cell's anchor is "what C declares for the cell" (v3 design §2.3). §5.4 asks that a weak deviant "satisfies **every clause of C** relevant at the item". The outcome anchor is C's declared clause, so a reading that deviates from it violates C.
- The grounding is stronger than the amendment states. SOPBench's own verdict (`vendor/sopbench/env/evaluator.py:294-300`) is `success = no_tool_call_error ∧ constraint_not_violated ∧ database_match ∧ action_called_correctly ∧ dirgraph_satisfied`, with `action_called_correctly = (action_should_succeed == action_successfully_called)`. Tool-error runs are excluded before the readings are built, so the amended `clean` is exactly C's own `success` verdict.
- I checked this on the real data. On all 20,567 evaluated runs without a tool error (21,323 − 756 excluded), the amended `clean` equals the stored `evaluations[0].success` in 20,567 of 20,567 cases, with 0 disagreements.

*"S3 can never produce a weak finding on the base" is correct, and at the default V3Params it holds strictly, not only empirically.*
- `weak_finding` (`specdx/infer/typing.py:82-97`) returns `None` on constraint cells. So the outcome cell is the only S3 cell where a weak finding can arise.
- It needs a MAP set {x} with posterior ≥ τ_W = 0.9, and a clean reading with a value ≠ x.
  - If x = anchor, every reading with a value ≠ x is unclean by the new conjunct.
  - If x ≠ anchor, then `p_defect` ≥ 0.9 ≥ τ_H = 0.5. A hidden finding fires, and `resolve_calls` (`typing.py:120-131`) discards the weak finding.
- So under τ_W ≥ τ_H, no weak finding can occur on S3's base cells. The implementer's "only where the MAP set differs from the anchor" is the right corner, and that corner is suppressed too.
- This is the v3-consistent outcome. v3 §2.3 says that unanchored cells are "the place where weak checks show", §5.4 says that "mutants W1 and W2 delete the anchor", and resolution 14 in `typing.py` says "weak truths are unanchored cells only". Every S3 base cell is anchored.
- The ruling's "If wrong: fewer weak findings" understated this. The count is zero. That includes the 33 should-fail "doer passes the strict replay" cases, which the review had called the interesting ones (see Out-of-Scope O-3).

*Forward compatibility (a note, not a blocker).*
- For P3b, `clean` must come from the clauses the (mutant) C still has, and the anchors must follow the same set. Concretely:
  - A C-mutant that drops `action_called_correctly` must both remove the outcome conjunct from `clean` and unanchor the outcome cell (anchor `None`, the W1/W2 analogue). Only then can a weak finding appear.
  - A mutant that drops `db_match`, `constraints` or `process` can never yield a weak finding on the current four cells. Those cells are constraint cells, and outcome readings at the anchor are not deviant.
  - Weak findings from dropped database checks need the field-level value cells that §13d defers to P3b.
- The current code hardcodes `CHECKS` and the `should` anchor (`cells.py:66-90`), but it does not make any of this impossible. Adding a "clauses C keeps" parameter to `cells_for` is a local change.

**2. I-3 variants.**
- Per run and per task, by frequency, with ties broken by the string (`cells.py:35`).
- `tell_for` uses variant 0 (`cells.py:55`), and the test puts the minority render first and expects T from the majority.
- The confound is disclosed in the report (`p3.py:299-304`; `results/p3/replication.md:177`).
- The disclosure has one nuance, under O-4.

**3. I-4.**
- Streamed in 1 MiB blocks (`p3.py:107-113`).
- A mismatch excludes the file with a reason that names both hash prefixes (`p3.py:152-156`), and the file is listed under Excluded files.

**4. I-2 count: 8 against the review's 5.** The two counts reconcile, and the review undercounted.
- The review's own pre-fix rater counts already imply 8 excluded files: bank 26 of 27 files, dmv 25/26, healthcare 27/28, hotel 25/26, library 26/27, online_market 26/27, university 24/26. That is 1+1+1+1+1+1+2 = 8.
- The review's explicit list missed three files: gpt-5-mini in healthcare and online_market, and gpt-4.1-mini in library.
- The rater counts are unchanged before and after the fix, so the same 8 files were dropped silently before.
- The Loading table (`results/p3/replication.md:154-162`) reconciles exactly:
  - files − loaded = excluded in every domain;
  - records − runs = missing evaluations: 134, 97, 124, 197 (= 195 + 2 stray), 66, 175 (= 172 + 3 stray), 84 (= 2 × 42);
  - Σ runs = 21,323, which is the review's count of evaluated runs.

**5. T1.** NOT ADDRESSED. This is a residual item. `vendor/VENDORED.md:11` is still false. The author has two options:
- edit it personally, or explicitly exempt `vendor/VENDORED.md`, our own provenance note, from the vendor rule;
- or record the correction outside `vendor/`, for example in `docs/RUNNING.md` or next to `scripts/verify_vendor.sh`.

**6. The design's §13d amendment text** (`docs/superpowers/specs/2026-11-19-specdx-s2-s3-design.md:375-376`). It is accurate as far as it goes. The v3 §5.4 quote is verbatim, the "golden clause" reading agrees with v3 §2.3, and `Reading.clean` is the right code reference. Three precision gaps remain, all Minor:
- **(a)** It does not say that the amended `clean` is SOPBench's own `success` verdict (`evaluator.py:294-300`) on runs without a tool error. That is the strongest justification, and it was verified on 20,567 of 20,567 runs.
- **(b)** Its last sentence ("every such outcome cell became a weak check") only explains the should-succeed refusals. It does not record the consequence: weak findings disappear on should-fail cells too, and weak findings are structurally unreachable on S3's all-anchored base cells. P3b should not plan weak-check scoring on base cells without unanchoring mutants or field-level cells.
- **(c)** Two lines below, line 379 still says that T "is identical across models for the same task". I-3 has shown this is false on the published runs (see O-5).

## New Breakage in the Fix Diff

No Critical or Important breakage.

- **N-1 (Minor): the stale-commit warning is permanent noise for `fetch-sopbench`.**
  - Where: `specdx/report/stages.py:198-207` and `specdx/cli.py:221-224`.
  - Every `rebuild_status`, which runs after every stage, warns that `fetch-sopbench was recorded at commit 9e90ada`.
  - The fetched bytes do not depend on the code commit, and they are now sha-verified at load, so this warning never signals a real problem.
  - Possible fixes: exempt stages whose output is commit-independent, or suppress the warning on `status` rebuilds and keep it on `--done`. If it stays, users will learn to ignore the warning that matters for `replication`.
- **N-2 (Minor, documentation): the §13d amendment's precision gaps (a) and (b) above** (`design.md:376`).

## Out-of-Scope Observations

- **O-1.** `baseline/README.md:141-144` still gives live instructions for `./run_all.sh baselines-dev / freeze-baselines / study / report`. This is the same defect as T8(1), in a file the ruling did not name. It is a project-authored guide (commit 9d1ad8b), not a vendored clone.
- **O-2.** `stale_commit` ignores a dirty tree. A stage recorded at HEAD with a dirty tree is not flagged. This is acceptable for M-3's purpose.
- **O-3.** After I-6, the review's "interesting 33" are no longer surfaced by any typed finding: should-fail tasks where the agent did the forbidden action and all three replayed checks passed. Under v3's typing that is correct, because C's `action_called_correctly` catches them. As a lead for P3b, they could be kept as an untyped diagnostic count in the report (outcome-deviant ∧ all checks satisfied, per domain). This is the author's call.
- **O-4.** The confound sentence says that "the earlier models saw an older render". That holds for variant 1. But variant 2 is a third render seen by o4-mini, a later model: 75 runs in healthcare and 4 in library. The per-domain model list shows this, but the sentence over-generalises slightly.
- **O-5.** Design §13d line 379 ("identical across models for the same task") contradicts I-3 and the report. It was not ruled, so it was left alone. It should be amended together with (a) and (b).
- **O-6.** Replication has no guard against two manifest entries with the same canonical model in one domain. `select` prevents this, but a hand-edited manifest would make `run_key`s collide silently in `prompt_variants` and in the readings. This is negligible now.

## Verdict

**Fix round: Findings remain open.**
- **T1** (`vendor/VENDORED.md:11` "and nothing else") is NOT ADDRESSED. It is blocked by the vendor rule and the permission classifier, and needs an author decision (named point 5).

Every other ruled item is ADDRESSED: I-1 to I-6, M-1, M-2, M-3, M-7, T7(5), T7(2) and T8(1)–(3). The fix diff introduces no Critical or Important breakage. There are two Minor items: N-1 (a permanent stale warning for fetch-sopbench) and N-2 (the precision of the §13d amendment).

I-6 is faithful to v3: the amended `clean` equals SOPBench's own `success` on 20,567 of 20,567 runs. Under τ_W ≥ τ_H, weak findings are strictly unreachable on S3's anchored base cells. P3b's C-mutants will need to unanchor the outcome cell and drop its conjunct, or add field-level value cells, before a weak finding can appear. This is a forward note, not a blocker.
