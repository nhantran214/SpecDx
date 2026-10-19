# SDD ledger — plan: docs/superpowers/plans/2026-10-14-specdx-p2a.md
Spec: docs/superpowers/specs/2026-10-14-specdx-v3-design.md (v3.1 at 5b406a2; binding; delta on v2 and v1).
Worktree: /home/ethan/AILab/TonomyX-paper/.claude/worktrees/specdx-p2a, branch worktree-specdx-p2a, base 4ec6037 (master).
Execution: subagent-driven (user's choice, 2026-10-15). The P1 run cache at /home/ethan/AILab/TonomyX-paper/work is read-only in P2a (md5 db1e1823c66df102648c69eb6a8789b3, 6505 lines).
Plan pre-check (by the plan author, scratch): suite counts T1 448 … T15 595. A full G0 preview on the pre-check code PASSED (results/p2-spikes/g0-preview-full.md; weak 0.403, a margin of 0.003).
Global constraints extracted to global-constraints.md (plan lines 74–226).

## Pre-flight scan
The full table (68 rows: 53 task pairs, 15 single tasks; 8 Important and 18 Minor findings) is in preflight.md. Every step applies cleanly. The findings concern behaviour and tests.
Rulings (applied to the plan text by an amendment pass, then re-pre-checked; spec v3.1 is binding):
- Ruling F1: on an anchored cell, ambiguity and hidden requirement co-occur, as spec §3.1, §5.1 and §5.3 (v1 §2.4) require.
  - When P(|A|=2) ≥ τ_A, the cell reports the ambiguity and also its hidden candidate (subtype ambiguous-pick, score max(P(A≠{d}), P(|A|=2))).
  - When P(A≠{d}) ≥ τ_H and the most probable defect form is a pair, both are reported too.
  - Unanchored cells report ambiguity only.
  - Mirrored in g0.counts_at, with a Task 7 test.
  — Why: the spec is binding, and the plan omitted it. — If wrong: extra hidden findings on ambiguity mutants (v1 semantics); G0 changes after the preview was seen, which is disclosed (F23).
- Ruling F2: G0 counts non-converged fits per arm and setting (search) and per point (check), and writes them to p2-g0.json and .md, with a test. Reporting only. — If wrong: none.
- Ruling F3: fix Task 3's soline assertion as proposed (so, ma_sp, ma_soline). — If wrong: none.
- Ruling F4: fix Task 7's ambiguity-suppresses-weak test so that removing the `continue` fails it. — If wrong: none.
- Ruling F5: add `_log_joint(batch, lik)` in Task 4, used by Task 5's `_log_posterior`. The arithmetic is identical. — If wrong: none.
- Ruling F6: diagnose_v3 reuses v2's multi-clause candidate path through a shared helper in scores.py that both `diagnose` and `diagnose_v3` call; v2 tests (test_scores, test_replay) must pass unchanged. — Why: spec §11 says hitting sets are "reused unchanged". — If wrong: a subtle change to v2's path, which those tests guard.
- Ruling F7: deduplicate within the new code.
  - One `calibrate.smoke_runs` serves both calibrate and p2_replay.
  - P1_SETTINGS, P1_SCHEMA and NoRefutation live in calibrate; p2_replay imports them.
  - `by_cell` is defined once, in baselines.
  - experiments/p1b_replay.py stays self-contained (an archived script whose results are pinned), and the tasks say so.
  — If wrong: minor.
- Ruling F8: replace the trivially true assertions.
  - 6079: assert the ordering only if the implementer confirms it is monotone; otherwise drop the line.
  - 4538/7718: test a gate JSON that carries a "params" key, and assert the loaded params come from code (V3Params()), never from the JSON.
  — If wrong: none.
- Ruling F9: correct the docstrings (Report.spent, CellDiagnosis.spent, the weights of non-readers). — If wrong: none.
- Ruling F10: pre-registration §3 records all planning resolutions 1–18 plus these rulings, with needles. — If wrong: none.
- Ruling F11: pin hitting-set max_size 3 and the dev split (5, 20261006) in the pre-registration test. — If wrong: none.
- Ruling F12 (G0-altering, disclosed):
  - (a) the calibration's plurality and dispersion use population members only, per resolution 15;
  - (c) the cell-mix shares are computed per column, matching the generator;
  - (b) readings per cell follow from the planned budget (resolution 15), and the calibrated `reads` is reported beside the simulated reads.
  — Why: the code departed from its own resolution and from spec §8.1. These are correctness fixes, not tuning. — If wrong: G0's outcome may differ from the preview, possibly a FAIL that stops P2 honestly.
- Ruling F13: the calibration raises CalibrationError instead of its four silent fallbacks. — If wrong: none (they never fire on the real cache).
- Ruling F14: Task 15 updates RUNNING.md §5 to the v3.1 screen rule (0.3, at least 2 families, at least 4 acting models, exit 3). — If wrong: none.
- Ruling F15: RUNNING §11 reports convergence but does not stop on it. The G3 criteria are exactly spec §8.2's. — If wrong: a missed stop the spec never asked for.
- Ruling F16: cell_table carries "sets" (the posterior over A_c, at most 22 pairs), per spec §3.5/§5.5, with a test. — If wrong: a larger report.
- Ruling F17 (planning resolution 17): B2 and the refuter's interpreter order use θ_m (spec §8.3, §11; per-model quantities), and w_b weights the hitting sets. Recorded in the pre-registration. — If wrong: a different refuter order.
- Ruling F18 (planning resolution 18): §5.2's V(b) is v2 §3.4's, over every clause of C. — Why: the literal "constraint cells" reading would drop golden+invariant conflicts, which are exactly the C2 mutants' truths. — If wrong: none against the spec's intent.
- Ruling F19: diagnose_v3 sorts witnesses and passes the group to the refuter sorted by run_key; a reversed multi-clause case is added to the determinism test. — If wrong: none.
- Ruling F20:
  - `specdx simulate` defaults to §8.1's minimums (10 replicates, 5 for regime), and labels lower counts "exploratory";
  - `regime` is added to the help;
  - the report adds T2′'s chance line and T4(c)'s bound;
  - Task 13 gains a step that runs the full study at the frozen parameters and commits it.
  — If wrong: CPU time.
- Ruling F21: Task 15's test pins the committed results/p2-g1.md (PASS line and θ). — If wrong: none.
- Ruling F22: keep G0's seeds (the preview is known). The pre-registration states that the criteria are judged on the selection replicates, and G0 adds an information-only out-of-sample report (seeds base+10…19). — If wrong: none.
- Ruling F23: update the plan header and the controller note to record the full G0 preview (PASS, its chosen parameters). Pre-registration §6 discloses the preview and lists every post-preview change (F1, F12). — If wrong: none.
- Ruling F24: the generator's single-form groups are recorded as a limitation of resolution 14 in the pre-registration (no generator change after the preview). — If wrong: none.
- Ruling F25: Task 15 keeps only its new check; the repeat of Task 13's frozen-parameter test is removed. — If wrong: none.
- Ruling F26: Task 13 runs G0 as a background job with bounded polling, reading the final log line and the exit status; no `tail -f`. — If wrong: none.

## Progress
Task 1: dispatched (haiku), BASE 4ec6037. The plan amendment pass (rulings F1–F26) follows Task 1, to avoid two pytest processes on one DB.
Task 1: implemented f13fe11 (448 passed; cache md5 verified). Review dispatched (sonnet).
Plan amendment pass: dispatched (resume of the plan author, opus). It applies F1–F26 to the plan text, re-runs the pre-check from f13fe11 and re-runs the full G0 preview on the amended code. Task 2 waits for it.
Task 1: review (sonnet) Approved, 0 Critical/Important. The reviewer independently recomputed the last cached P1b run's key from the worktree code: it reproduces run_key cab22d69c445888e977345ee under context 9c796b06…; the cache md5 is unchanged.
Task 1: ⚠️ resolved: the counts, ruff, venv and imports are in the implementer's report (448 passed, imports inside the worktree).
Task 1: complete (commits 4ec6037..f13fe11, review clean)
Task 1: minor (deferred): the test comment's "lines 6319-6505" range is unverified beyond line 6505; the test does not distinguish screen runs from smoke runs (they share one context)
Plan amendment pass: done, committed f191a25. All 26 rulings are applied, with resolutions 17 and 18 added.
  - F8: `_matching_concentration` is non-increasing over shares 0.30–1.00, so the ordering is asserted.
  - Gap resolved by the amender: under F12(c), a partly anchored column counts as anchored if at least half its cells are.
  - New counts: T1 448, T2 469, T3 479, T4 488, T5 495, T6 504, T7 510, T8 519, T9 526, T10 530, T11 544, T12 548, T13 549, T15 602.
  - Pre-check passed from f13fe11. Cache md5 unchanged.
AMENDED FULL G0 PREVIEW: FAIL on (ii) weak, 0.343 (calibrated). Hidden .861, ambiguity .837; (i), (iii) and the ambiguity win hold.
  Sensitivity (full grid, information only):
  - F1 alone: PASS (weak .403);
  - F12(a) alone: PASS (.401);
  - F12(c) under three rules for partly anchored columns: weak .343 (≥ half), .306 (mean fraction), .236 (any). All FAIL.
  - Next ten seeds: weak .427 (same criteria pass).
  - Seed-set swing ≈ 0.08.
  Evidence: results/p2-spikes/amended-preview/.
  STOP POINT for the author before Task 11: my own ruling F12(c) flips G0. Every path from here is outcome-informed. Tasks 2–10 do not depend on it and continue meanwhile.
Task 2: dispatched (haiku), BASE f191a25
Task 2: implemented 2789780 (469 passed). Review dispatched (sonnet).
Task 2: review (sonnet) Needs fixes, 1 Important (plan-mandated): the numeral rule runs for text columns, so 0015→15 and 0912…→912…, and identity() on FK columns can collide.
  Ruling: numeral canonicalisation applies only to numeric column types (int/numeric/decimal and the like, per the pack's column types); text columns get NFC, trim, whitespace collapse and casefold only. Add tests:
  - a zero-padded id in a text column stays as it is;
  - `35,0` in a numeric column → 35;
  - `1,000` in a numeric column follows the decimal-comma convention the spec's `35,0` example implies (→ 1), stated in a comment.
  — Why: §2.5's examples are numbers, and ids must keep their identity. — If wrong: a numeral written into a text column (e.g. "35.0") stays distinct from "35" (rare).
  Ruling: also add one-line comments: the hyphen-only boundary class is intentional (en dash not excluded), and the composite-key join in executor_generated is latent (only evt has a composite key, and it is excluded). — If wrong: none.
Task 2: minor (deferred): no test for 28+ digit numerals; the report's test count wording (16 parametrized cases, not 19)
Task 2: fix round 1 dispatched (resume of the implementer, haiku), FIX_BASE 2789780
Task 2: fix round 1 implemented cd4f578 (472 passed; TEST-COUNT OFFSET +3 against the plan). Re-review dispatched (sonnet).
Task 2: fix round 1/5 (comments addressed; the numeral finding is partly open: `bigint`, the pack's most frequent numeric type, is missing from the numeric set; commits 2789780..cd4f578)
  Ruling (re-review item 2): identity() keeps normalising key and FK columns as text, without numeral canonicalisation. — Why: keys are ids, and the ruling's aim is id safety. — If wrong: a numeric key written "35.0" would not match "35" (no such key in the pack).
Task 2: fix round 2 dispatched (resume, haiku): add bigint to the numeric set, with a test.
Task 2: fix round 2 implemented a6df776 (473 passed; offset +4). Re-review dispatched.
Task 2: fix round 2/5 (1 addressed, 0 open; commits cd4f578..a6df776)
Task 2: complete (commits f191a25..a6df776, review clean after 2 fix rounds). TEST-COUNT OFFSET now +4 (473 actual against the plan's 469).
Task 2: minor (deferred): identity() and tell_texts/occurs normalise as text (numerals untouched on those paths; ruled intended); an en dash counts as a token boundary
Task 3: dispatched (sonnet), BASE a6df776; expected count = plan + 4 (479 → 483)
Task 3: implemented d2e4443 (483 passed; test_cells 35). Review dispatched (opus: the foundation of every later task).
Task 3: review (opus; resumed after a rate limit) Approved, 0 Critical/Important. All four named risks hold: lifted anchors, acted and scope filtering, constraint evaluation with applies, and a clean flag (§5.4 wording) that is conservative relative to resolution 14.
Task 3: ⚠️ resolved: acted_steps and violations are unchanged v2 code, already reviewed in P1b; draft_tables coverage is data, and its so/po tests pass; clean against the simulation is checked at Task 6.
Task 3: complete (commits a6df776..d2e4443, review clean)
Task 3: minor (deferred):
  - event-cell location/id is built from raw ref/actor spellings, so a non-golden variant can split a location (plan-mandated);
  - a reading is clean if SOME in-scope acted item is clean (record in the pre-registration);
  - no tests for anchored-scope constraints unread elsewhere, for a lifted clause becoming per-scenario, or for multi-step clean;
  - duplicated _PLACEHOLDER and _TRUE/_FALSE;
  - the _run fixture is rebuilt per call;
  - sorting by run_key alone.
Task 4: dispatched (sonnet), BASE d2e4443; expected count = plan + 4 (488 → 492)
Task 4: implemented 06ecfc0 (492 passed). Review dispatched (opus: core model math).
Task 4: review (opus) Approved, 0 Critical/Important. The code matches the brief verbatim. Five mutation tests are killed, the hand-computed likelihood was re-derived, the prior sums to 1 for n = 0..6, and the T2′ test has a negative control at p = |A|/K.
Task 4: ⚠️ carried into Task 5: ε must stay positive on OTHER (κ·u) and zero on unused slots; _encode raises a KeyError for a cell missing from by_cell.
Task 4: complete (commits d2e4443..06ecfc0, review clean)
Task 4: minor (deferred):
  - cell_posterior can go all-NaN with a user ε that is 0 on OTHER (unreachable from the fit); no guard;
  - cell_posterior does not check values[0] == anchor or duplicate values;
  - the sigmoid overflows below −709;
  - V3Params shares are not validated;
  - the n = 1 anchored prior renormalisation should be recorded in the pre-registration;
  - the T2′ test checks the argmax only;
  - names.index is O(C·G);
  - map_set ties break to the last set.
Task 5: dispatched (sonnet), BASE 06ecfc0; expected count = plan + 4 (495 → 499)
Task 5: implemented 37b27d5 (499 passed; ε invariant confirmed). Review dispatched (opus).
Task 5: review (opus) Needs fixes, 1 Important (plan-mandated, resolution 6). The ε update (counts+κu)/(N+κ) drops the −log(1−ε(A)) normaliser of §3.2's truncated error model, so EM is not monotone in _log_posterior: up to −7.4 nats per iteration, and on the test world fixed points end up 118 nats apart. "MAP by EM" (§3.4) does not hold.
  Ruling (amends resolution 6): use the exact EM for the truncated error model by data augmentation.
  - For each set A ≠ ∅, add post(A)·N_err,A·ε(k)/(1−ε(A)) expected rejected draws to each slot k ∈ A, where N_err,A is the cell's expected readings outside A.
  - Then ε = (counts + rejected + κu)/(total + κ), kept zero on unused slots and positive on OTHER.
  - Add a test that _log_posterior never decreases across EM iterations from a fixed start (tolerance 1e-9).
  — Why: spec §3.4 binding; the multi-start comparison needs a monotone EM. — If wrong: none against the spec. It changes G0 again (disclosed in the pre-registration with F1/F12).
  Ruling: also fix Minors 1, 2 and 4:
  - the family map is built from sorted readings, and raises on a conflict;
  - each start records (score, converged, iterations) in `starts`;
  - remove the unused `params` argument of _error_update.
  — If wrong: none.
Task 5: minor (deferred): the line-search floor applies the step at t ≤ 1e-8; the no-collapse test could assert θ < 0.5 at the T3 fixed point; Review Focus 1 "acted" is decided upstream (Tasks 3/10)
Task 5: fix round 1 dispatched (resume of the implementer, sonnet), FIX_BASE 37b27d5
Task 5: fix round 1 implemented 8d926af (502 passed; offset +7). The starts now end within 3.4 nats and all converge. Re-review dispatched (opus: math).
Task 5: fix round 1/5 (4 addressed, 0 open; commits 37b27d5..8d926af). The re-reviewer ran 240 EM runs: the worst drop was 3.6e-12, and the fixed point is stationary in ε.
Task 5: complete (commits 06ecfc0..8d926af, review clean after 1 fix round). TEST-COUNT OFFSET now +7 (502). Interface change: AnchoredFit.starts is dict[str, StartFit(score, converged, iterations)].
Carry into the owning task of plan line 7723 (r.fit.starts formatted with {v:.1f}): use v.score, since starts now map to StartFit. Task 6: dispatched (sonnet), BASE 8d926af; expected count = plan + 7 (504 → 511)
Task 6: implemented 780e001 (511 passed). Review dispatched (sonnet).
Task 6: review (sonnet) Approved, 0 Critical/Important. The base-arm sampler was verified empirically against §3.2, over 200k draws, including the V∖A renormalisation and ∅. Truths and clean follow resolution 14. Seeding is local.
Task 6: complete (commits 8d926af..780e001, review clean)
Task 6: minor (deferred):
  - A3 is not paired with the base at the same seed, because the extra Dirichlet draws shift the stream. Not changed: it would alter G0 after the preview.
  - No validation of the arm name or the act/alphas lengths.
  - Two tests restate the implementation, and there is no direct error-side sampling test.
  - The A3 concentration 0.3 is a magic number.
  - The A5 guess is per cell.
  - The generator reads faithfully with probability 1 when A = V (the fitter keeps OTHER).
Task 7: dispatched (sonnet), BASE 780e001; expected count = plan + 7 (510 → 517)
Task 7: implemented 31c67b3 (517 passed). Review dispatched (opus).
Task 7: review (opus) Approved, with 1 Important (plan-mandated): an anchored mis-key cell can also emit a weak check (a clean reading of d is a "deviant"), double-typing one defect, and a weak false positive against resolution 14's truth labels.
  Ruling: option (a): a cell that reports a hidden finding reports no weak check, mirroring the ambiguity suppression. Mirror it in Task 11's counts_at. — Why: the deviant there IS C's own value; the defect is the hidden requirement; resolution 14 makes weak truths unanchored cells. — If wrong: a real weak check co-located with a hidden requirement goes unreported. Changes G0 (disclosed).
  Ruling (carry into Task 10): direction-B ambiguities (p_pair < τ_A) cannot pass the ddmin oracle (P(|A|=2) ≥ τ_A), so the localiser runs only for ambiguities with p_pair ≥ τ_A; the others are reported unlocalised. — If wrong: fewer localised ambiguities.
  ⚠️ carry into Task 11: counts_at must mirror co_occur (both directions), the ambiguity→weak suppression and the new hidden→weak suppression, with its equivalence test.
Task 7: minor (deferred): max(p_defect, p_pair) is redundant (always p_defect); no test that direction A overrides an existing subtype; assert used for narrowing
Task 7: fix round 1 dispatched (resume, sonnet), FIX_BASE 31c67b3
Task 7: fix round 1 implemented 251d3e4 (518 passed; offset +8). Re-review dispatched.
Task 7: fix round 1/5 (1 addressed, 0 open; commits 31c67b3..251d3e4)
Task 7: complete (commits 780e001..251d3e4, review clean after 1 fix round). TEST-COUNT OFFSET now +8 (518).
Task 7: minor (deferred): no type_cells-level test of an unanchored weak finding. An anchored cell with MAP {d} cannot yield weak: a deviant at an anchored value cell violates its golden clause and so is not clean; no ruling needed.
Task 8: dispatched (sonnet), BASE 251d3e4; expected count = plan + 8 (519 → 527)
Task 8: implemented 36ee8af (527 passed; study calls type_cells). Review dispatched (sonnet).
Task 8: review (sonnet) Needs fixes, 2 Important.
  #1 (plan-mandated): the baselines lack v3's cell-level post-processing.
  - They emit weak on cells they also call hidden or ambiguous (anchored weak false positives: majority 3/20, stage2 8/39 over three replicates).
  - Majority reports an anchored ambiguity without its hidden requirement (it missed 8 of 25).
  - This biases the W/T/L map and G0 (iii) towards v3.
  Ruling: apply exactly v3's post-processing (the same one used in type_cells) to every baseline:
  - an anchored ambiguity also reports hidden;
  - a cell reporting hidden or ambiguity reports no weak.
  So methods differ only in their evidence model. Add a test that no baseline reports weak on a cell it reports hidden or ambiguous, and that an anchored baseline ambiguity co-reports hidden. — Why: G0 (iii) is a fairness comparison. — If wrong: none (the baselines get stronger, so G0 is harder for v3). Changes G0 (disclosed).
  #2: --replicates is not validated (0 falls back silently; negative values run zero replicates). Ruling: reject < 1 with exit 2, with a test. — If wrong: none.
Task 8: minor (deferred):
  - versus compares floats at the 0.05 boundary;
  - gamma_min at k = 2 is degenerate;
  - "runs needed" uses one 0.6 target for all types;
  - test gaps (default replicates, the margin boundary, by_cell keys, F1 on an anchored pair).
Task 8: fix round 1 dispatched (resume, sonnet), FIX_BASE 36ee8af
Task 8: fix round 1 implemented 88ba872 (529 passed; offset +10). Re-review dispatched.
Task 8: fix round 1/5 (2 addressed, 0 open; commits 36ee8af..88ba872). type_cells output was identical before and after over 48 fitted specs.
Task 8: complete (commits 251d3e4..88ba872, review clean after 1 fix round). TEST-COUNT OFFSET now +10 (529). New shared helper: typing.resolve_calls(anchored, calls).
Task 9: dispatched (sonnet), BASE 88ba872; expected count = plan + 10 (526 → 536)
Task 9: implemented 7ec7258 (536 passed; F6 helpers shared with the v2 diagnose). Review dispatched (opus).
Task 9: review (opus) Approved, 0 Critical/Important. v2's diagnose is unchanged after the F6 refactor; the unattempted and spent paths match resolution 8 and I2; it is deterministic; locations match matching.py.
Task 9: ⚠️ carried into Task 10: build `violated` as v2 §3.4's V(b) over every clause (resolution 18); the differential against the base lives in the loop.
Task 9: complete (commits 88ba872..7ec7258, review clean)
Task 9: minor (deferred):
  - no tests for an unattempted multi-clause candidate going to spent, for an unobservable multi-clause candidate, for spent after a probe-refuted cell, or for a conflict-truth match;
  - the ambiguity_cells tie-break keeps the first;
  - Design note: dropping singleton hitting sets after they prune supersets means a clause X with high hitting-set support but a cell posterior < τ_H is never refuted (v2 would have refuted {X}). Follows resolution 8. For the author.
Task 10: dispatched (sonnet), BASE 7ec7258; expected count = plan + 10 (530 → 540)
Task 10: implemented a00f1c7 (541 passed; offset +11; the localiser is gated on p_pair ≥ τ_A, with a test). Review dispatched (opus).
Task 10: review (opus) Approved, 0 Critical/Important. The brief was re-applied mechanically and matches HEAD, apart from the τ_A guard. All 5 named risks hold: run keys untouched; the oracle uses fitted parameters on acted re-runs; the differential is unchanged; report and gate fields are present; G2 counts over the admitted population, with a run-free rescreen (which reproduces spec §7's population).
Task 10: ⚠️ resolved: the implementer's full suite (541) includes the DB tests; the upstream modules were reviewed in Tasks 3–9.
Task 10: complete (commits 7ec7258..a00f1c7, review clean). TEST-COUNT OFFSET now +11 (541).
Task 10: minor (deferred):
  - experiments/p1_dev_diagnostics.py now raises an AttributeError on cfg.p_single (it is pinned to 547abbd; add a docstring or a local constant);
  - SpecDxConfig.k_max is dead, yet still pinned by test_preregistration;
  - the loop-level determinism test does not permute input;
  - gate_markdown does not mention non-convergence;
  - _persists is built even for skipped ambiguities.
STOP before Task 11: the author's decision on the G0 weak criterion (A/B/C/D) is pending. Asked 2026-10-15; no answer yet. Since the amended preview, the G0-altering changes are F1, F12(a,c), Task 5 exact EM, Task 7 hidden→weak suppression and Task 8 baseline post-processing.
AUTHOR DECISION (2026-10-15): option B. G0's weak-check bar is information only; criterion (iii), including weak, is unchanged; the gate's weak rule is unchanged. Spec amended in 7b5b645 (§8.2, §13 item 15).
Task 11: dispatched (sonnet), BASE 7b5b645; expected count = plan + 11 (544 → 555).
  Carries:
  - option B in the G0 verdict;
  - counts_at mirrors type_cells through typing.resolve_calls (F1, both suppressions), with an equivalence test;
  - StartFit;
  - the baselines already share the post-processing.
Task 11: implemented 420636c (555 passed). Option B is in the verdict; counts_at uses resolve_calls. Review dispatched (opus). Carry into Tasks 13 and 14: weak lives under ii_weak_information.
Task 11: review (opus) Approved, 0 Critical/Important. Read-only cache, the selection rule (weak stays in the objective per §9), counts_at ≡ type_cells, option B in the verdict, and freeze all hold.
Task 11: ⚠️ resolved: calibrate's shared helpers are consumed in Task 12; resolution 16's point is checked in Task 13.
Task 11: complete (commits 7b5b645..420636c, review clean)
Task 11: Ruling (minors 1, 2, 3 and 5: crash exit 1 reads as FAIL; a missing cache exits 1; dropped screen runs are not count-checked; freeze does not refuse FAIL): handled operationally in Task 13. The controller checks that p2-g0.md and .json exist before reading the exit status, that the calibration maps 105 runs (60 smoke + 63 screen − 18 shared), and freezes only on PASS. — If wrong: none (no code change, so G0 is unaltered).
Task 11: Ruling (minor 4, calibration scope): _top_error, |V| and reads use every screened model's readings, including non-population models. Kept, consistent with resolution 15; disclosed in the pre-registration (Task 15). — If wrong: a slightly mis-calibrated error concentration in G0.
Task 11: minor (deferred): `--replicates` is unvalidated in g0 main; the lambda style nit. Housekeeping: global-constraints.md G0 (ii) is updated to option B.
Task 12: dispatched (sonnet), BASE 420636c; expected count = plan + 11 (548 → 559). Carries: r.fit.starts values are StartFit (use .score); option B.
Task 12: implemented 08379b9 (559 passed; render uses v.score). Review dispatched (sonnet).
Task 12: review (sonnet) Needs fixes, 1 Important: an uncaught exception exits 1, the same status as G1 FAIL.
  Ruling: any exception other than the handled ones exits 2 (an error), with a test (load_cache raising OSError gives 2). — If wrong: none.
  Ruling: also fix cheap Minors that protect G1's meaning:
  - (i) requires at least 1 reading for each NO_COLLAPSE model; a named model with no reading fails (i), stated in the report — otherwise a prior-only θ ≥ 0.5 passes vacuously;
  - add a render test with StartFit;
  - render lists each start's converged and iterations;
  - the title is "PROVISIONAL (not G1)" when the parameters are not frozen;
  - refuse --out under WORK_DIR.
  — If wrong: none.
Task 12: minor (deferred): unmatched_rows rebuilds the golden set per cell; the grn12-only "not observable" edge is documented, not changed.
Task 12: fix round 1 dispatched (resume, sonnet), FIX_BASE 08379b9
Task 12: fix round 1 implemented 6ebd199 (563 passed; offset +15). Re-review dispatched.
Task 12: fix round 1/5 (5 addressed, 1 open: the crash→2 guard does not cover mkdir/render/write_text/verdict after the try; commits 08379b9..6ebd199). Round 2 dispatched (resume).
Task 12: fix round 2 implemented 46beddf (564 passed; offset +16). Re-review dispatched.
Task 12: fix round 2/5 (1 addressed, 0 open; commits 6ebd199..46beddf)
Task 12: complete (commits 420636c..46beddf, review clean after 2 fix rounds). TEST-COUNT OFFSET now +16 (564).
Task 12: minor (deferred): no end-to-end test pins exit 0 for PASS or exit 1 for FAIL
Task 13: dispatched (sonnet), BASE 46beddf; expected count = plan + 16 (549 → 565).
  Operational checks (ruling, Task 11):
  - the outputs exist before the exit status is read;
  - calibration runs = 105;
  - freeze only on PASS;
  - weak is read from ii_weak_information;
  - cache md5 and listing are checked before and after every command.
Task 13: G0 PASS (exit 0), commit a97a67c.
  - (i) θ error .0112; (ii) hidden .858, ambiguity .807; (iii) all hold, with the ambiguity win (.807 vs majority .714).
  - Weak .338: information only, under option B. Out of sample: weak .427, and the out-of-sample check would also PASS.
  - Chosen: π .2, shares (.4,.3,.3), τ 1.0, (μα,σα)=(1,1), τ_H .5, τ_A .3, τ_W .9.
  - Non-converged: 60/1080 search fits.
  - Runtimes: G0 3354 s, study 1195 s.
  - Cache md5 and listing unchanged; 565 passed.
  Review dispatched (sonnet).
Task 13: review (sonnet) Approved, 0 Critical/Important. The verdict was recomputed from p2-g0.json: PASS (ambiguity win .092; worst (iii) gap: A3 weak −.029). The chosen setting is the unique maximum, with a margin of ~8e-5 over tau_h .7. Freeze touched only the marked block, equal to the params file and test-pinned. Calibration: 105 runs, the §7 population.
Task 13: ⚠️ resolved: the controller verified the cache md5 (db1e1823…) and the listing hash (5f3760e7…), unchanged after G0 and the study.
Task 13: complete (commits 46beddf..a97a67c, review clean). Suite 565.
Task 13: minor (deferred, for the paper):
  - non-convergence at the chosen setting: A3 2/10, A5 1/10;
  - some regime-map cells converge only 0.6–0.8;
  - the selection margin is ~8e-5 among near-identical settings;
  - the anchored.py docstring cites experiments/p2-preregistration.md, created in Task 15.
Task 14: dispatched (sonnet), BASE a97a67c
Task 14: G1 PASS (exit 0, 17 s), commit e9aac9b.
  - (i) θ: gpt-oss .989 (2628 readings), qwen3:30b .948 (1944).
  - (ii) R-SO golden:so:SO-0031:trang_thai is reported, with P(A≠{d}) = 1.0 (mis .952).
  - Cache unchanged.
  VALIDITY NOTE for the author/paper: both models read 'draft' (English) at R-SO's cell, not 'Đã duyệt'. G1 (ii) passes, but the evidence may reflect English status vocabulary rather than the R-SO reading. If the models write English statuses generally, base status cells become hidden mis-keys, and the differential could mask status mutants (C2) at the gate. Check this in P2b's G3 smoke run and disclose it in the pre-registration.
  Unmatched-row counts are listed in p2-g1.md.
Task 14: review dispatched (sonnet)
Task 14: review (sonnet) Approved. The record supports PASS under the frozen parameters, consistent with the brief.
  Important validity caveat, for the author:
  - (ii) passes nominally: the models wrote 'draft', not 'Đã duyệt'.
  - The earlier diagnosis's row counts suggest English 'draft' dominates status readings (so ≈94%, inv ≈89%, po ≈57%).
  - So many anchored Vietnamese status cells likely type as hidden mis-keys (53/296 smoke cells are hidden).
  - At the gate, the differential would likely mask status-column mutants (e.g. the H1 status sites).
Task 14: complete (commits a97a67c..e9aac9b, review clean)
Task 14: minor (deferred): "1 cells" pluralisation; the record lacks date, SHA and exit code
STOP before Task 15 (it freezes the pre-registration): the author must decide on the English status-vocabulary validity issue.
AUTHOR DECISION (2026-10-19): option (1). Quantify the vocabulary issue first, read-only, base smoke only, no mutants. Diagnostic dispatched (opus); output vocab-diagnosis.md.
Vocabulary diagnosis (opus, read-only, committed as results/p2-vocab-diagnosis.md). The hypothesis mostly fails:
  - 53 hidden findings; 43 are not vocabulary;
  - 4 have an out-of-vocabulary MAP value, and only 1 is a true translation (inv draft/Nháp);
  - T itself writes "SO-0031 draft" etc., so the status findings are real T↔C conflicts;
  - C declares no domain, and a domain rule removes 0 findings.
  The cache is unchanged. Recommendation to the author: option (2), freeze and disclose.
AUTHOR DECISION (2026-10-19): option (2). Freeze the pre-registration as is and disclose the vocabulary diagnosis, the R-SO caveat and the status-mutant risk. Task 15: dispatched (sonnet), BASE 3b7d428; expected count about plan + 16 (602 → ~618).
Task 15: implemented 15db088 (618 passed). Review dispatched (opus: pre-registration integrity).
Task 15: review (opus) Needs fixes, 6 Important (accuracy and completeness of the frozen document):
  1. the wrong best hidden baseline (anchor-only .807, not .766);
  2. the superseded ε M-step in §3 item 6;
  3. the §9 arithmetic (43 vs 46);
  4. the diagnosis summary is incomplete or inaccurate;
  5. the R-SO caveat understates (APR-0102 is never written);
  6. the option-B timeline is incomplete.
  Ruling: fix all 6 Important plus Minors 1–7. — Why: the document is about to be frozen. — If wrong: none.
Task 15: fix round 1 dispatched (resume, sonnet), FIX_BASE 15db088 (review saved as task-15-review.md)
Task 15: fix round 1 done at 98edf32 (621 passed); scoped re-review dispatched (opus) on review-15db088..98edf32.diff
Task 15: re-review clean (all of I1–I6 and M1–M7 ADDRESSED). New Minors: N1 prereg:110 0.089→0.088 against the JSON; N2 needle for "0.25–3"; N3 RUNNING.md:186 resume wording.
  Ruling: carry N1–N3 into the final fix wave. — Why: none is load-bearing, and they need no separate round. — If wrong: a frozen document carries a 0.001 error until that wave.
Task 15: complete (98edf32)
Final review: dispatched (opus), MERGE_BASE 4ec6037, HEAD 98edf32
Final review: With fixes. 0 Critical, 2 Important, 10 Minor (final-review.md). Tests: 621 passed. G0 reproduces bit for bit; EM monotone.
  Ruling (I1): amend the G3 row so θ_m ≥ 0.5 AND ≥1 reading per population model; replace P1b check (4) in RUNNING §11 with a v3 reading-count check; disclose as an amendment dated 2026-10-19, made before any G2/G3 run. — Why: the same vacuous pass that ruling 21 closed for G1; no G2/G3 data exists, so this is not outcome-informed. — If wrong: the author reverts one row; G3 is stricter than registered.
  Ruling (I2): add a disclosure of the generator constants and the calibrated truth mix (≈93% of hidden truths are constraint cells; chosen shares = planted shares) plus needles. — Why: the document claims complete disclosure. — If wrong: none (documents only).
  Ruling (M1–M9): fix in this wave. M4 is wording only; the ε formula stays. M7: refuse to overwrite the frozen params file. M8: catch-all → exit 2. M9: Reproduce block with SHAs. — Why: all cheap, and they protect the frozen artefacts. — If wrong: none.
  Ruling (M10): fix in this wave: gate_markdown shows non-convergence. — Why: small, and it removes a P2b prerequisite. — If wrong: one cosmetic line.
  Ruling: every other deferred minor stays deferred, per the reviewer's triage table. — Why: the reviewer agrees each is not load-bearing. — If wrong: listed in final-review.md.
Final fix wave: dispatched (fresh, sonnet), FIX_BASE 98edf32
Final fix wave: done, 16a3957 6ac5228 (630 passed, ruff clean); scoped re-review dispatched (opus)
Final re-review: 11 of 12 ADDRESSED; M9 open plus new N1 (Important: the Reproduce recipe writes into work/, the read-only cache, and its p2_replay step is refused there), N2 (Minor: 37 = 29 value + 8 event cells), N3 (Minor: run times 56 and 20 min). See final-rereview.md.
  Ruling (residual adjudication): fix N1–N3 now, in a docs-only follow-up from the same implementer, then have the controller check that the text matches. — Why: a recipe that writes into the read-only cache contradicts a hard constraint, and readers will copy it. The edit is docs only, with no code or results. — If wrong: one extra docs commit.
Final follow-up: e402b8b (N1–N3; 630 passed); controller checked RUNNING §12 (mktemp -d, nothing under work/) and prereg:178 (29 value + 8 event). Final review: complete.
