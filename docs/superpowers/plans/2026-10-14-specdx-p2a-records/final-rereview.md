# Scoped re-review of the final fix wave: SpecDx P2a, 98edf32..6ac5228

Reviewer: scoped re-review, 2026-10-19. Read-only. I made no changes to the worktree, the index, HEAD or any branch, and I did not touch `/home/ethan/AILab/TonomyX-paper/work`. I made no Ollama or LLM calls, did not run calibrate, g0, simulate or p2_replay (only their `--help`), and dispatched no subagents. My only scratch outputs are under the session scratchpad.

**Independent checks I ran:**

| Check | Result |
|---|---|
| Truth mix: `generate()` on the calibrated config, base arm, seeds 0–9 (pure Python, no LLM) | hidden 32.5 constraint / 2.4 value (share 0.9312); ambiguity 58.3 unanchored value, 7.6 anchored constraint + 0.5 anchored value = 8.1; weak 152.5. All match §10 and the fix report. |
| Calibration figures against `results/p2-calibration.json` | 140 columns; constraint_share 0.342857 (34.3%); pooled_share 0.364286 (36.4%); anchored_share 0.0326. `calibrate.py:346-347` computes anchored_share over value columns only, so "of the value columns" is correct. The config holds forms (0.4, 0.3, 0.3), unanchored (0.5, 0.2, 0.3), kappa_true 30, defect_rate 0.15 and seed 0. |
| Generator constants against `specdx/sim/generator.py` | `forms`, `unanchored` and `kappa_true` at lines 50-52 and 57; Dirichlet(30·tendency) at line 151; A3 `dirichlet(np.full(len(vocab), 0.3))` at line 152-153. The form order (mis, amb, empty) and the unanchored order (none, amb, empty) match the document's glosses. |
| M1 gap from `results/p2-g0.json` | 0.42656 − 0.33838 = 0.08818, so 0.088 is correct. |
| SHAs (`git log`, read-only) | `a97a67c` = "G0 passed — calibration, G0 record, V3Params frozen, and the study"; `e9aac9b` = "G1 passed" and adds only `results/p2-g1.md`. `46beddf..a97a67c` under `specdx/` touches only the frozen block (`PARAMS_FROZEN` and the V3Params values). `a97a67c..98edf32` touches nothing under `specdx/` or `experiments/p2_replay.py`. Task 13's base was 46beddf (progress.md:243). The md5 `db1e1823…` matches progress.md:4 and the plan. |
| CLI `--help` (calibrate, `g0 run`, `specdx simulate`, `p2_replay.py`) | Every flag in the Reproduce block exists: `--out`; `--calibration/--out/--params-out`; `--sweep/--config/--params/--out`; `--out`. |
| G0 jq comparison run read-only on the committed file | `jq -S 'del(.seconds, .calibration)' results/p2-g0.json` exits 0. The diff of the file against itself is empty. The only string or volatile top-level keys are `verdict`, `calibration` and `seconds`, so deleting those two is the right set. |
| G3 jq against the Report shape | `Report.to_json` (`loop.py:145-156`) emits `cells` = `cell_table`, a dict keyed by cell id with `readings: {model: count}` (`loop.py:171-181`). `Reading.model` is the interpreter's model name (`cells.py:395`), the same names that `models_by_family` holds (`pilot.py:104`, a family-to-list mapping). So `$g[0].models_by_family[][]` and `.cells[].readings[$m] // 0` are correct, and an empty `cells` gives 0. |
| Protected paths (`git diff --stat 98edf32 6ac5228`) | Nothing changed under `results/`, `experiments/p2-g0-params.json`, `experiments/p2-gate.json` or `vendor/`. In `anchored.py` only the docstrings at 282-286 and 375-380 changed; the frozen block (41-60) is untouched. |
| Focused tests: `tests/test_docs.py`, `tests/test_preregistration_v3.py`, the two new `test_g0.py` tests | All passed, none skipped (jq is installed, so the jq test really ran). `git status` was clean afterwards. |

The fix report names its covering tests and gives the suite result (630 passed, ruff clean). Its claims match the diff, with the exceptions recorded under M9.

---

### Finding Verdicts

**I1. G3's vacuous θ pass, and the dead check (4).** ADDRESSED.
- **The G3 row.** `experiments/p2-preregistration.md:97` now reads "θ_m ≥ 0.5 and at least one reading for each population model", plus "(check (4) is replaced, see the amendment below)".
- **The dated amendment.** `p2-preregistration.md:102` dates it 2026-10-19, "after this file was frozen and before any G2 or G3 run", mirroring ruling 21. It gives the σ(1.0) = 0.731 and Beta(7, 3) mode 0.75 reasons, both arithmetically correct, and says the change only makes G3 stricter. Line 5 records both dated additions.
- **The RUNNING check.** `docs/RUNNING.md:188` and `:196-198` replace check (4) with the v3 reading check. The check matches the real Report shape (see the table above). `:202` makes a `NO READINGS` line a stop.
- **Tests.** Needles are in `tests/test_preregistration_v3.py`. `tests/test_docs.py` runs the documented jq on a synthetic report of the `cell_table` shape and asserts `ok, ok, NO READINGS`.

**I2. Disclose what the calibrated simulation exercises.** ADDRESSED.
- **The new section.** §10 (`p2-preregistration.md:162-180`) states `forms`, `unanchored`, `kappa_true` 30, the A3 0.3 concentration and the defect rate 0.15. It gives the calibrated mix (140 columns, 34.3%, 36.4%, 0.0326 of value columns) and the per-specification truth mix for seeds 0–9. All of these figures match my own generator run and the calibration JSON.
- **The reading consequences.** §10 says that G0's hidden evidence is mainly constraint cells, that the ambiguity and weak counts come from the uncalibrated `unanchored`, and that the chosen shares equal the planted `forms`.
- **Tests.** `test_section_10_generator_constants_and_truth_mix_are_the_codes` recomputes all of this from the code. A small wording inaccuracy remains in the "37 of 53" sentence; it is recorded as new Minor breakage N2 below and does not reopen I2.

**M1. 0.089 should be 0.088.** ADDRESSED.
- `p2-preregistration.md:112` reads "0.088 (G0)", which is correct (0.08818).
- The needle was updated, and `test_section_8_g0_figures_…` now derives the gap from `results/p2-g0.json`.

**M2. Pin "kept within 0.25–3".** ADDRESSED. The needle is added to `test_the_pre_registration_states_each_value`. The text is at `p2-preregistration.md:62`.

**M3. RUNNING's resume wording.** ADDRESSED. `docs/RUNNING.md:186` reads "re-run the same command once the tunnel is back; finished runs come from the cache (§7)". It is pinned by `test_the_resume_wording_says_to_rerun_the_same_command`.

**M4. The ε prior's rationale and name (wording only).** ADDRESSED.
- **Where.** `specdx/infer/anchored.py:375-380` and `p2-preregistration.md:53`.
- **What it now says.** ε is the posterior mean under Dirichlet(κu), equivalently the mode under Dirichlet(1 + κu), with κ pseudo-counts towards u. The literal MAP, (n + κu − 1)/(N + κ − K), would apply κ − K and is not used. All three statements are mathematically correct.
- **What was removed.** The unreachable "literal mode is 0" reason is gone.
- **What did not change.** The formula and the code are unchanged; `_error_update`'s body is identical.

**M5. What θ means.** ADDRESSED. `anchored.py:282-286` now says "θ: the accuracy at β = 0", and notes that β's location is fixed only by its N(0, τ²) prior. It is pinned in `test_the_anchored_wording_matches_the_corrected_rationale`.

**M6. The out-of-sample seeds had been seen.** ADDRESSED. `p2-preregistration.md:114` (§6) says "out of sample for the selection, but not unseen", and `:142` (§8) says "the same "next ten" seeds the amended preview had shown, §6". Both are pinned by needles.

**M7. `g0 run` overwrites the frozen parameters file.** ADDRESSED.
- **The guard.** At `specdx/sim/g0.py:424-427`, while `PARAMS_FROZEN` is set and no `--params-out` is given, `run` returns 2 before doing any work. An explicit path is passed through at `:429-430`. `run()` accepts `params_out` (`g0.py:337`) and creates its parent directory (`:364`).
- **The test.** `test_run_refuses_to_overwrite_the_frozen_parameters_file` covers the refusal, the pass-through and the behaviour when parameters are not frozen.

**M8. A crash exits 1, the same status as FAIL.** ADDRESSED.
- At `g0.py:434-436`, a catch-all `except Exception` now returns 2, placed after the specific handlers. KeyboardInterrupt and SystemExit are not swallowed. PASS stays 0 and FAIL stays 1.
- It is covered by `test_an_unexpected_exception_exits_2_…`.

**M9. A reproduction recipe where readers look.** NOT ADDRESSED. The block exists at `docs/RUNNING.md:213-233` and its SHAs, flags and G0 jq are all correct, but it does not run as written from the repository root.
- **Where readers run it.** RUNNING's own setup is `git clone … specdx && cd specdx` (`RUNNING.md:22`), and its §11 commands assume the main checkout. In the main checkout, `work/` is the run cache: `SPECDX_WORK` = `/home/ethan/AILab/TonomyX-paper/work`, and `WORK_DIR` defaults to `REPO_ROOT/work` (`specdx/config.py:14`).
- **The G1 step fails.** It writes `--out work/repro/p2-g1.md` (`RUNNING.md:229-230`). That path resolves under WORK_DIR, so `experiments/p2_replay.py:236` refuses it with "lies under the read-only run cache" and exits 2. I checked the path resolution in Python without running the script: `Path('work/repro/p2-g1.md').resolve().is_relative_to(WORK_DIR)` is True from `/home/ethan/AILab/TonomyX-paper` and False only from a separate worktree.
- **The other steps write into the cache directory.** `mkdir -p work/repro`, calibrate, g0 and simulate (`RUNNING.md:222-228`) all write there, which is the directory the same paragraph (`:219`) calls read-only.
- **The time estimates are wrong.** "~25 min" for G0 and "~10 min" for the study (`:226`, `:228`) contradict the committed run times: G0 took 3354 s (`results/p2-g0.json` `.seconds`, about 56 min) and the study took 1195 s (task-13-report.md:30, about 20 min).
- **Fix (one line each).**
  - Use a scratch directory outside `SPECDX_WORK`, for example `R=$(mktemp -d)` or a git-ignored `repro/` at the repository root, in all five commands and in the `cmp`/`diff` line.
  - Change the estimates to about 1 h and about 20 min.

**M10. The gate Markdown does not show non-convergence.** ADDRESSED.
- **The field.** `GateResult.converged` (`specdx/eval/pilot.py:169`) is filled for the base (`:306`) and for each mutant that ran (`:344`). A mutant that failed is absent, which is correct. The value is passed positionally in the right field order (`:361-362`) and emitted in `to_json` (`:198`).
- **The Markdown.** `gate_markdown` adds a "Convergence … never a criterion" section (`:432-441`).
- **No decision changes.** `gate_verdict` is called with `primary`, `primary_untyped` and the scorable counts only (`:363-364`), and `passed` is still `passed and not failed`. So no gate decision depends on `converged`.
- **The test.** It asserts that passed, wins, headline, primary, scores, literal_wins and scorable are identical between the converged and non-converged runs. `GateResult` is constructed nowhere else (I grepped for it).

### New Breakage in the Fix Diff

- **N1. Important: the Reproduce block's G1 step exits 2 from the repository root, and the block writes into the run-cache directory.** `docs/RUNNING.md:222-230` against `experiments/p2_replay.py:236` and `specdx/config.py:14`. The details and the fix are under M9; this is the same item, listed here because the fix diff introduced it. It is loud, it corrupts no data and it affects no gate, but a reader following M9's recipe from the documented working directory cannot complete it. The fix report says the commands were not executed (final-fix-report.md:62).
- **N2. Minor: the "37 of G1's 53" sentence miscounts the event cells.** At `experiments/p2-preregistration.md:178`, the sentence says G0 "barely exercises value cells with multi-valued vocabularies, where 37 of G1's 53 real hidden findings are". Of those 37, 8 are absent golden events. Event cells are kind `event` with the binary vocabulary {present, absent} (`specdx/infer/cells.py:358-365`), not value cells. The correct split is 29 value cells and 8 binary event cells (§9: 15 + 4 + 10 value, 8 event, 16 constraint). The conclusion stands either way. A precise wording would be "37 … are on value or event cells (29 value, 8 event)".
- **N3. Minor: the run-time estimates in the Reproduce block.** `RUNNING.md:226` and `:228` say ~25 min and ~10 min, against the recorded 56 and 20 min (see M9).

### Out-of-Scope Observations

- `RUNNING.md:233` says "The G1 replay uses the frozen V3Params in the code, so it is the same at any later commit". That holds at 6ac5228, where only docstrings changed under `specdx/infer`. "Any later commit" is forward-looking, because P2b may change the typing or loop code that `p2_replay` imports. "At any later commit that does not change specdx/" would be exact. Non-blocking.
- `p2-preregistration.md:168` says "a flat Dirichlet with concentration 0.3". The code draws a symmetric Dirichlet(0.3, …, 0.3) (`generator.py:152`), which is a flat base measure, not a flat density. This is defensible under the spec's own "u flat" usage. Non-blocking.

### Verdict

**Fix round:** Findings remain open.
- **M9, NOT ADDRESSED.** The Reproduce block fails at its G1 step from the repository root, because p2_replay refuses `--out` under the cache. Steps 1–3 also write into the cache directory, and the G0 and study time estimates contradict the records. This is the same item as the new Important breakage N1.
- **Addressed, with no new breakage beyond N1–N3:** I1, I2, M1–M8 and M10. N2 and N3 are Minor.

The fix is a one-line path change across the block, the two time estimates and, optionally, N2's wording. It needs no code or result change.
