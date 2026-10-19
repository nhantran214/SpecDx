# Final whole-branch review: SpecDx P2a (4ec6037..98edf32)

Reviewer: final-review pass, 2026-10-19. Read-only. The worktree, index and HEAD are unchanged, and `/home/ethan/AILab/TonomyX-paper/work` was not touched. I made no LLM, Ollama or docker calls and did not export SPECDX_WORK. I wrote only scratch scripts in the session scratchpad, plus this file.

## How the review was done (passes)

The diff covers 58 files and +52.9k lines, most of it result data. I reviewed it in five passes:
1. **Context.** The ledger, the global constraints and the spec (all of v3.1 plus its diff).
2. **Core maths.** `anchored.py`, `typing.py`, the `scores.py` refactor and `cells.py`.
3. **Simulation.** `sim/generator.py`, `metrics.py`, `baselines.py`, `study.py`, `g0.py` and `calibrate.py`.
4. **Integration.** `loop.py`, `workbench.py`, `cli.py`, `screen.py`, `pilot.py`, `b2_single.py`, `p2_replay.py` and `p1b_replay.py`.
5. **Documents and records.** The pre-registration, `RUNNING.md`, the G0/G1 records and the pre-registration tests.

**Independent checks I ran:**

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest` | **621 passed** (36.5 s) |
| `.venv/bin/ruff check .` | **All checks passed** |
| `git status` | Clean |
| **G0 criteria reproduced.** `g0.check(load_config(results/p2-calibration.json), V3Params(), 10)` at HEAD, 138 s. | Bit-identical to `results/p2-g0.json["criteria"]`: θ error 0.011192233136510975, all 48 F1 values, and non-convergence {calibrated 0, a3 2, a5 1, own 1}. The calibration's `config` equals `p2-g0.json["base"]`. |
| **G0 selection reproduced** for the chosen prior setting: `g0.search` restricted to π 0.2, shares (0.4, 0.3, 0.3), τ 1, (μα, σα) = (1, 1), with all 27 threshold settings. | 27 of 27 candidates identical to the recorded ones (objective, per-arm F1, non-convergence). Chosen = recorded; objective 0.633040682748974. |
| **EM monotonicity.** My own loop over `_e_step`, `_newton` and `_error_update` on calibrated-scale synthetic specifications: 3 arms × 2 seeds × 5 starts × 60 iterations, with `_log_posterior` after every iteration. | Worst change between iterations: **0.0**. It never decreased. |
| **Cross-check.** `results/p2-sim.json` arms sweep against G0's criteria. | The base arm equals G0's calibrated point exactly (v3 0.858 / 0.807 / 0.338), from an independent code path (`run_sweep`). |
| **Frozen parameters.** `V3Params()` against `experiments/p2-g0-params.json` against `p2-g0.json["chosen"]["params"]`. | Equal; `PARAMS_FROZEN = True`. The code has not changed since G0 (`a97a67c..98edf32` touches no `specdx/` file), and `46beddf..a97a67c` touches only the frozen block. |
| **Truth mix of the calibrated simulation** (`generate`, seeds 0–9). | Per specification: 32.5 hidden truths on constraint cells against 2.4 on value cells; 58.3 ambiguity truths on unanchored value cells against 8.1 on anchored cells; 152.5 weak truths. See I2. |

---

## Strengths

- **The EM matches spec §3.4 and pre-registration item 6.** The E-step posterior over the 22 sets and r = Σ post(A)·P(faithful | y, A) are both correct. The α/β M-step is the expected complete-data objective r·η − q·log(1 + e^η), with q = P(A ≠ ∅); a Newton step with a backtracking line search makes it a GEM step. The ε M-step is the exact data-augmentation EM for the truncated error model: geometric rejected draws ε(k)/(1 − ε(A)) on the slots in A, weighted by post(A) and by the readings outside A. `_log_posterior` uses the same prior on ε as the M-step, so EM is monotone in the very quantity the multi-start compares, and my check confirms it empirically.
- **The multi-start is right.**
  - The starts are T3's median plus a common α ∈ {0, 1, 2, 3}, each with β = 0 and ε = u.
  - The comparison is on the marginal log posterior with A_c summed out, and ties go to the earlier start.
  - Each start's `StartFit` is reported, and the winner's convergence flag and iteration count reach the Report JSON and the gate lines.
- **One typing core, shared by everything that types.**
  - `typing.resolve_calls` serves `type_cells`, G0's `counts_at` and all three baselines.
  - `counts_at` mirrors `co_occur` in both directions plus both weak suppressions.
  - So v3 and the baselines differ only in their evidence model, which is what makes G0 (iii) a fair comparison.
- **The G0 machinery is sound and reproducible.**
  - The grid and the selection rule are fixed in code. `conservative()` implements "higher P(|A| = 1), then smaller π".
  - The out-of-sample seeds (base + 10…19) are disjoint from the search seeds.
  - The committed G0 record reproduces bit-for-bit from the committed code and calibration.
- **Read-only discipline is enforced in code.**
  - `p2_replay` and `calibrate` open no Workbench, database or Ollama, and extract the PDFs into a `TemporaryDirectory`.
  - `p2_replay` refuses an `--out` under WORK_DIR and refuses provisional parameters.
  - Every crash in `p2_replay` exits 2.
- **Integration is clean.**
  - The `Localizer` signature change (`persists`) is carried through the loop, the workbench and the tests.
  - `ambiguity_cells` is keyed exactly as `merge_defects(identity)` merges, so the oracle re-reads the cell behind the surviving note.
  - The localiser is gated on p_pair ≥ τ_A, so it never runs an oracle that cannot pass.
  - `GateConfig.params` is never read from JSON.
  - The G2 stop rule is counted over the admitted population. `screen --from` makes no run and exits 3 on STOP.
- **The pre-registration is unusually candid.** It records the option-B relaxation, the fact that G0 would have failed under the bar registered before 2026-10-15, the post-preview G0-altering changes, the selection margin of 8e-5, and the nominal R-SO pass.

---

## Issues

### Critical

None.

### Important

**I1. G3's no-collapse check passes vacuously for a population model that never acts, and the documented smoke check that should catch this still tests v2's constant.**
- **Where:**
  - `docs/RUNNING.md:188-199` (the G3 block);
  - `experiments/p2-preregistration.md:97` (the G3 row);
  - `specdx/infer/anchored.py:447`, where a model without readings keeps α = μ_α (the Newton gradient on its α is the prior's only);
  - plan `2026-10-11-specdx-p1b.md:7461-7465` (check 4).
- **What:**
  - Under the frozen parameters, a population model with zero readings gets θ = σ(μ_α) = σ(1.0) = 0.731. That passes G3's "θ_m ≥ 0.5 for each population model", and RUNNING's jq prints `ok` for it.
  - G3 also requires "every P1b smoke check passes". Check (4) flags a non-acting model by testing whether |θ − 0.75| < 1e-9, which is v2's Beta(7, 3) mode. Under v3 θ is 0.7311, so check (4) can never fire.
- **Why it matters:**
  - This is exactly the vacuous pass that ruling 21 closed for G1 (i), but G3 was not given the same guard.
  - It is a realistic case: mistral-small3.2:24b has act rate 0.346, and G3 runs at budget 60.
  - Spec §5.3 and G2 need four acting models. A model that never acts at G3 means the gate would run with three, which is the regime where spec §4.4 and spike 2 say v3.1 loses its ambiguity advantage.
- **Fix:** an author ruling before G2. The G3 text is still before G2, so it can change.
  - Mirror ruling 21 in the G3 row: "θ_m ≥ 0.5 and at least one reading for each population model".
  - Replace check (4) in RUNNING §11 with a v3 check. The Report already carries per-cell reading counts by model, for example `jq '[.cells[].readings | keys[]] | unique' results/p2-smoke-base.json`, which must list every population model.
  - Add a needle for the new wording to `test_preregistration_v3.py`.

**I2. The pre-registration does not say what the calibrated simulation actually exercises, nor the generator constants behind G0's truth mix.**
- **Where:**
  - `results/p2-calibration.json:30` (`anchored_share` 0.0326);
  - `specdx/sim/generator.py:50-51` (`forms` = (0.4, 0.3, 0.3), `unanchored` = (0.5, 0.2, 0.3)) and line 57 (`kappa_true` 30), plus the A3 concentration of 0.3 at line 154;
  - `experiments/p2-preregistration.md:61` (item 14) and §8.
- **What:**
  - Under ruling F12(c), only about 3% of value columns are anchored at the calibrated point. Seeds 0–9 of the calibrated configuration give, per specification:
    - **Hidden truths:** 32.5 on constraint cells (V = {satisfied, violated}) against 2.4 on value cells, so about 93% are binary constraint cells.
    - **Ambiguity truths:** 58.3 on unanchored value cells against 8.1 on anchored cells.
  - The ambiguity and weak truth counts are set mostly by the uncalibrated constant `unanchored` = (0.5, 0.2, 0.3).
  - The planted anchored form shares (0.4, 0.3, 0.3) are exactly the (s_mis, s_amb, s_∅) that G0 chose.
  - The pre-registration states only "the defect rate is planted at 0.15".
- **Why it matters:**
  - G0 (ii) "hidden ≥ 0.6" will be read as evidence for detecting hidden requirements. Yet 37 of G1's 53 real hidden findings are on value or event cells: mis-keys over multi-valued vocabularies, which G0 barely exercises.
  - A reader should also know that the selected form prior coincides with the generator's planted shares. That is not a leak, but it is a property of selecting by simulation.
  - The document's own title claims complete disclosure, and it is about to be the frozen artefact.
- **Fix (documents only, no code or result changes):**
  - Add to item 14, or to §8, the generator constants (`forms`, `unanchored`, `kappa_true` 30, the A3 concentration 0.3) and the calibrated truth mix above. State that G0's hidden-requirement evidence is mainly on constraint cells, and that the chosen shares equal the planted ones.
  - Add needles to the pre-registration test.

### Minor

**Already ruled into this wave (N1–N3): must fix before merge.**
- **M1 (N1).** `experiments/p2-preregistration.md:110` says "0.089 (G0)", but the exact difference is 0.4266 − 0.3384 = **0.0882**, so it should read 0.088. The needle at `tests/test_preregistration_v3.py:213` must change with it, or the suite fails.
- **M2 (N2).** `experiments/p2-preregistration.md:62` states "kept within 0.25–3", but no needle pins that text; `test_preregistration_v3.py:282` pins only the code. Add `"kept within 0.25–3"` to `test_the_pre_registration_states_each_value`.
- **M3 (N3).** `docs/RUNNING.md:186` says "so the gate resumes (§7) once the tunnel is back", which reads as automatic. §7 says to run the same command again, with finished runs coming from the cache. Reword to "re-run the same command once the tunnel is back; finished runs come from the cache (§7)".

**New. M4 and M5 are wording in the frozen document or code, cheap enough to fold into the same wave; M6 onwards are optional.**
- **M4. The rationale for the ε prior is wrong, and its name is loose.**
  - **Where:** `specdx/infer/anchored.py:374-376` and `experiments/p2-preregistration.md:53`.
  - **What:**
    - The formula (counts + κu)/(N + κ) is the posterior **mean** under Dirichlet(κu), equivalently the mode under Dirichlet(1 + κu). It is not "the mode of the Dirichlet κ·u_c posterior".
    - The stated reason for leaving the spec's literal MAP ("the literal mode is 0 wherever κ·u + count < 1") cannot occur: |V_c ∪ OTHER| ≤ 7 gives κu ≥ 10/7 > 1.
    - So the code applies κ = 10 pseudo-counts towards flat, where the spec's literal Dirichlet(κu) MAP would apply κ − K.
  - **Fix:** correct the wording and add one sentence noting the effective prior. The formula itself is disclosed and pinned, and G0 selected under it, so it stays.
- **M5. "θ is accuracy at a cell of median difficulty" is not what the code does.**
  - **Where:** `specdx/infer/anchored.py:282-283`, which repeats spec §3.5.
  - **What:** the location of β is fixed only by its N(0, τ²) prior, not by T3's median-β = 0 constraint, so θ_m = σ(α_m) is the accuracy at β = 0.
  - **Fix:** reword the docstring. G0 (i), with error 0.011, shows the two are close in practice.
- **M6. The out-of-sample seeds had already been seen.**
  - **What:** G0's out-of-sample seeds base + 10…19 (`g0.py` `check(offset=replicates)`) are the "next ten seeds" whose amended-preview result (weak 0.427, ledger line 79) the author saw before choosing option B.
  - **Why it matters:** they are out of sample with respect to the selection, but not unseen. The pre-registration (§6, §8) cites the out-of-sample 0.427 as support.
  - **Fix:** add one clause saying so. Line 110 implies it but does not say it.
- **M7. Re-running G0 overwrites the frozen parameters file.**
  - **Where:** `specdx/sim/g0.py:363`, `run(..., params_out=PARAMS_FILE)`.
  - **What:** a reader who reproduces G0 with `python -m specdx.sim.g0 run` overwrites `experiments/p2-g0-params.json`.
  - **Fix:** refuse while `PARAMS_FROZEN` is set unless `--params-out` is given, or write next to `--out`.
- **M8. A crashed G0 exits 1, the same status as FAIL.** This is Task 11's deferred minor 1.
  - **Where:** `specdx/sim/g0.py:422` catches only OSError, ValueError and KeyError.
  - **Why it matters:** the ruling handled it operationally for the one G0 run, but anyone reproducing G0 will meet it.
  - **Fix:** a cheap catch-all mapped to exit 2, as in `p2_replay`. I mildly disagree with leaving it as an operational ruling.
- **M9. No reproduction recipe sits where readers will look.**
  - **What:** the commands for calibrate → `g0 run` → `simulate --config results/p2-calibration.json` → `p2_replay` exist only in plan Task 13/14. The G0 and G1 records carry no code SHA or date (Task 14's deferred minor).
  - **Fix:** a short "Reproduce" block in the pre-registration or RUNNING, naming the commits (G0 ran at 46beddf, whose simulation code is identical to HEAD).
- **M10. The gate's Markdown does not show non-convergence** (Task 10's deferred minor). The JSON carries it per mutant. Fix before gate v3 launches in P2b, not necessarily in this merge.

---

## Triage of the deferred minors (`minor (deferred)` lines in progress.md)

**Must fix before merge:** N1, N2 and N3 (M1–M3 above). M4 and M5 are wording in the frozen document or code and are cheap enough for the same wave. I1 and I2 are new.

**Can stay deferred, with reasons:**

| Task | Deferred minor | Reason it can stay |
|---|---|---|
| 1 | Unverified range comment; screen and smoke runs not distinguished | The context digest test is sufficient |
| 2 | 28+ digit numerals; report wording; identity()/tell_texts normalise as text; en dash counts as a boundary | Ruled intended; no such data in the pack |
| 3 | Event location from raw spellings | Plan-mandated; the gate risk is mutant-specific |
| 3 | A reading is clean if SOME item is | Disclosed as pre-registration item 22 |
| 3 | Test gaps, duplicated constants, fixture rebuild, run_key-only sort | run_key is unique |
| 4 | All-NaN posterior with ε = 0 on OTHER | Unreachable: the fitted ε is always positive on OTHER |
| 4 | values[0] / duplicates unchecked; sigmoid overflow below −709 | The overflow gives p = 0 with a warning, which is harmless |
| 4 | Shares unvalidated; argmax-only T2′ test; `names.index` is O(C·G) | Negligible at C ≈ 10³ |
| 4 | map_set tie-break | Deterministic |
| 5 | The line-search floor applies the step at t ≤ 1e-8 | My monotonicity run saw no drop |
| 5 | Test tightening | Optional |
| 6 | A3 not paired with the base; arm and length validation; magic 0.3; A5 guess per cell; A = V faithful with probability 1 | Changing any would alter G0 after the fact. The 0.3 belongs in I2's disclosure. |
| 7 | Redundant `max`; missing direction-A test; `assert` used for narrowing | The asserts guard invariants that always hold |
| 7 | No type_cells-level unanchored weak test | Optional |
| 8 | Float comparison at the 0.05 boundary; γ_min at k = 2; single 0.6 target; test gaps | Informational outputs only |
| 9 | Test gaps; ambiguity_cells tie-break | The tie-break matches merge_defects' first-wins |
| 9 | Design note | Disclosed as item 24 |
| 10 | `p1_dev_diagnostics` AttributeError | Its docstring already says it is pinned to 547abbd and does not reproduce later |
| 10 | `k_max` dead but pinned; determinism test does not permute; `_persists` built for skipped ambiguities | Cosmetic |
| 10 | gate_markdown silent on convergence | M10: before the gate, in P2b |
| 11 | `--replicates` unvalidated in g0; lambda style | Cosmetic |
| 11 | Crash exits 1 | See M8: recommended, not blocking |
| 12 | `unmatched_rows` rebuild (performance); grn12-only edge (documented); no end-to-end exit-code test | Optional |
| 13 | Non-convergence, close margins, docstring citation | All disclosed in pre-registration §8; the file now exists |
| 14 | "1 cells"; record lacks SHA and date | Cosmetic; folded into M9 |

## Where I disagree with a ruling

- **Resolution 6 (as amended by the Task 5 ruling).** I disagree only with its stated reason, "the literal mode is 0 …", which is unreachable at κ = 10 and K ≤ 7. I accept the formula (M4).
- **Task 11's ruling to handle crash → exit 1 operationally.** It was fine for the single G0 run, but reproducibility argues for the code fix (M8).
- **Pre-registration G3 row.** Ruling 21 should have been carried to G3 (I1).

---

## Recommendations

1. Take one author ruling on I1, the G3 reading guard, and fold it in with N1–N3, I2, M4 and M5 as one final wave. Then re-run the pre-registration tests and the full suite. The expected count is 621 plus any new needles.
2. Before P2b's G3, replace the P1b check (4) in RUNNING §11 with the v3 reading-count check.
3. Add the reproduction block (M9) and guard `g0 run`'s writing of the parameters file (M7). Both are cheap and protect the frozen artefacts once readers try to reproduce them.

---

## Declined to judge

Each item below was considered and set aside as outside the plan or spec, or as an author decision.

| Considered | Why set aside |
|---|---|
| Option B, the weak-check bar made information-only | An author decision amended into spec §8.2 and §13 item 15, and disclosed |
| F12(c)'s "at least half its cells" rule for anchored columns | An author/plan ruling, disclosed with its sensitivities. I flag only what it implies for disclosure (I2). |
| Calibrating accuracy at plurality-equals-anchor cells, which may bias it upward | Resolution 15, planned and disclosed |
| F1 co-occurrence adding hidden false positives on ambiguity mutants | Required by spec v1 §2.4; noted in the ruling's "if wrong" |
| Hidden suppressing weak | The Task 7 ruling, disclosed as item 26 |
| G1 (ii)'s nominal pass and the vocabulary question | Author decision (2), disclosed in pre-registration §9 |
| F17: B2 and refuter order by θ_m against spec §3.5's w_b wording | The spec is internally inconsistent (§8.3 says θ_m) and the ruling resolves it |
| The pilot's preference for an unrun model at σ(μ_α) = 0.731 | Unchanged in kind from v2's Beta mode; P2b behaviour |
| RUNNING §11's G2 `jq` temp file under `work/` | Same pattern as P1b's §5; `work/` is writable in P2b |
| Performance (`names.index`, `unmatched_rows`, the size of per-cell JSON in the gate lines) | Not a correctness concern at this scale |
| The content of `results/p2-g0.json`, `p2-sim.json`, `p2-calibration.json` and `p2-spikes/**` | Skip-as-data. I spot-checked only where a claim depends on them, and reproduced G0's criteria and the chosen candidate. |
| The plan's own amendment diff | Per the brief |
| The vendor tree | Not in the changed-file list |

---

## Assessment

**Ready to merge: With fixes.**

The code is correct and coherent across tasks.
- **Maths and tests.** The EM is exact and monotone, the multi-start is right, typing is shared by the real pipeline, G0 and the baselines, and the suite and lint are clean.
- **Records.** The committed G0 record reproduces bit-for-bit from the committed code and calibration. The frozen parameters agree across the code, the parameters file and the record.

The issues that remain are in the frozen documents, not the algorithms.
- G3 as written can pass vacuously for a model that never acts (I1).
- The pre-registration does not state that G0's hidden-requirement evidence is mostly binary constraint cells under uncalibrated generator constants (I2).
- Three or more figure, needle and wording corrections are already ruled (N1–N3), with M4 and M5 to join them.

All of these are cheap. They should land before the pre-registration is treated as frozen for G2.

**Counts: Critical 0, Important 2, Minor 10.**
