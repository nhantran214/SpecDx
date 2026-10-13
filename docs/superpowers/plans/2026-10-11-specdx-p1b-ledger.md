# SDD ledger — plan: docs/superpowers/plans/2026-10-11-specdx-p1b.md
Spec: docs/superpowers/specs/2026-10-11-specdx-v2-design.md (binding; delta on 2026-10-06-specdx-design.md and its §11).
Worktree: /home/ethan/AILab/TonomyX-paper/.claude/worktrees/specdx-p1b, branch worktree-specdx-p1b, base e66bfa5.
Execution: subagent-driven (user's choice, 2026-10-12). The P1 run cache is at /home/ethan/AILab/TonomyX-paper/work (never modify).
Pre-existing ruling (author, 2026-10-12): Task 7 GO is a mechanism check (κ of the non-writer lowest; no θ collapse below 0.5; ≥ 50% of P1 base findings Unobservable). The θ ordering moves to Task 18, after the smoke run.

## Pre-flight scan
The full table (172 rows: 102 task pairs, 18 per-task, 13 findings, 22 edit checks, 5 reviewer-defect, 12 constraint/spec) is in .superpowers/sdd/2026-10-11-specdx-p1b/preflight.md.
- No signature, replace-target, import or test-count conflicts.
- The scanner ran Tasks 2–7's code in a scratch copy, and the three hand edits work: Task 7 gives 3 tests, Task 17 has 15 needles, and the Task 18 jq check works.
- Findings were applied to the plan in commit 5dcf5fb (spec F10 too). Rulings:
  - Ruling F1: the replay docstring now states the mechanism check. — Why: it contradicted go(). — If wrong: none.
  - Ruling F2: remove the unused QWEN3 constant. — If wrong: none.
  - Ruling F3: accept that statuses and the verdict are recomputed in render(). — Why: trivial cost. — If wrong: none.
  - Ruling F4: pin THETA_FLOOR and UNOBSERVABLE_SHARE inside the existing go test, so test counts stay unchanged. verdict() stays covered by composition (go plus unobservable_share). — If wrong: a verdict() wiring bug is caught only when the replay runs.
  - Ruling F5: in the pre-registration, the θ ordering moves to §4 ("before the gate"), and the closing GO line names all three conditions. — Why: the frozen document must be unambiguous. — If wrong: none.
  - Ruling F6: RUNNING.md gets the θ-ordering jq check between the smoke run and the gate. — Why: spec §6 makes it a stop condition. — If wrong: none.
  - Ruling F7: Task 18 reports κ and θ with the act rates from results/p1b-screen.json, to the controller, who relays to the author. — If wrong: none.
  - Ruling F8: a tie in κ at float precision is accepted as practically impossible; jq max_by picks deterministically. — If wrong: an arbitrary but reproducible top model on an exact tie.
  - Ruling F9: Step 6 reads "Verdict line after the title"; the Step 8 wording is corrected. — If wrong: none.
  - Ruling F10: accept "at least one model acted" in criterion (b), and spec §6 is updated to match. — Why: without it (b) is vacuously true. — If wrong: stricter GO.
  - Ruling F11: the disclosure states both first-draft conditions. — Why: honesty of the pre-registration. — If wrong: none.
  - Ruling F12: ollama_digests raises ValueError when a model is missing on only some hosts (KeyError only when it is missing everywhere), and the test is updated. — Why: hosts that disagree are an inconsistency like differing digests, not "not installed" (Review Focus 3). — If wrong: a screen with mixed hosts stops instead of skipping the model.
  - Ruling F13: accepted. `doctor` may create the worktree's own git-ignored work/. — Why: harmless. — If wrong: an empty directory.

## Progress
Task 1: dispatched (haiku), BASE 5dcf5fb
Task 1: Ruling: no subagent review for an empty environment commit. The controller verified it instead: .venv imports specdx and tonomyx from the worktree, and 262 passed. — If wrong: none (no code).
Task 1: complete (commits 5dcf5fb..7d4b207, controller-verified)
Task 1: minor (deferred): the commit message's Co-Authored-By trailer is on the subject line (no blank line)
Task 2: dispatched (haiku), BASE 7d4b207
Task 2: implemented 8443c0d (268 passed); review dispatched (sonnet)
Task 2: complete (commits 7d4b207..8443c0d, review clean)
Task 2: ⚠️ resolved. Ruling: the pre-registration (Task 17) must state the Rasch penalty exactly as (λ/2)·Σ(κ² + δ²), the N(0, 1/λ) ridge that act.py implements. — Why: a λ‖x‖² reading would change κ and δ. — If wrong: none. Carry into the Task 17 dispatch.
Task 2: minor (deferred): de_scope uses .get with [] (a silent fallback); fit_rasch(lam=0) raises a bare LinAlgError, and non-convergence returns silently; the penalty, the omega/m_min overrides and needs_gate are not pinned by tests; slice_rows is a frozenset (duplicates collapse) and duplicate Acts are double-counted
Task 3: dispatched (haiku), BASE 8443c0d
Task 3: implemented 3262b7d (272 passed); review dispatched (sonnet)
Task 3: complete (commits 8443c0d..3262b7d, review clean)
Task 3: ⚠️ resolved. Loop wiring (observations() / acted_violations()) is in Task 6, plan lines ~1972-1999.
Task 3: ⚠️ ruling. The class key no longer implies the violated set (it is projection only, so cross-slice clauses can differ within a class). Task 6's `_weak` used `{v.klass: v.violated}`, which is last-view-wins and order-dependent. Task 6 dispatch carries this correction:
  - `failing = {v.klass for v in views if v.violated}`;
  - return None if `failing & map_set`;
  - eligible = classes with no failing view.
  `_hidden_score` already has any-view semantics and stays as is.
  — If wrong: weak becomes stricter on mixed classes.
Task 3: minor (deferred): the SpecLens defaults drafting=frozenset() and initial=None fail open; acts() and views() both re-project; no test of acts() vs views().acted on a healthy run, of needs_gate, or of acted_violations through the real lens
Task 4: dispatched (haiku), BASE 3262b7d
Task 4: implemented 23782f3 (275 passed); review dispatched (sonnet)
Task 4: complete (commits 3262b7d..23782f3, review clean). The inversion-regression test is present, and the reviewer confirmed that it fails under P1's plug-in ε.
Task 4: ⚠️ resolved. The production caller is switched in Task 6 (plan ~line 2000: fit(observations(views), models=population, ...)). The reviewer measured that the unfiltered input still inverts (qwen 0.009, llama 0.997), so the acted filter is load-bearing.
Task 4: minor (deferred): the models= family is overridden by an observation's family; alpha+beta=2 with a zero-observation model divides by zero; the family effect is now unshrunk and a silent model contributes logit(0.75); non-convergence at iters is silent; no test of ItemPosterior.error or of the θ clamp
Task 5: dispatched (haiku), BASE 23782f3
Task 5: implemented 726e637 (280 passed); review dispatched (sonnet)
Task 5: review Approved with 1 Important (plan-mandated test gap). The _TIE tolerance has no regression test; setting _TIE = 0 passes all 5 tests. Rulings:
  Ruling: add a deterministic tie regression test, as an assertion inside an existing test function, so test counts are unchanged. — Why: it is the one named risk the plan leaves unguarded. — If wrong: none.
  Ruling: also fix Minor 3. permutation_test sorts readings by (model, variant, klass) first, so the p-value is independent of reading order. — Why: byte-identical reports (Task 14) and resume must not rest on upstream order. — If wrong: none.
  Ruling: also fix Minor 2. The BH comparison becomes `p * m <= rank * q + 1e-12`, so exact-boundary p-values like (1 + k) / 2001 are not lost to float error. — If wrong: none (the old error was conservative).
Task 5: minor (deferred): no test discriminates stratified from global permutation; eligibility is pooled across models, so p = 1 items count in BH's m; the report's line counts are wrong
Task 5: fix round 1/5 (2 addressed, 1 open: the tie test does not discriminate _TIE=0, per the implementer's own report; commits 726e637..2e76e3c). Re-review skipped because the open item was self-reported. Round 2 sent with a search recipe.
Task 5: fix round 2/5 (3 addressed, 0 open; commits 726e637..8e673a8). The re-reviewer confirmed the tie assertion fails under _TIE=0 (p 0.588 vs 0.509), and that the old BH comparison rejects 0 of 43.
Task 5: complete (commits 23782f3..8e673a8, review clean after 2 fix rounds)
Task 5: minor (deferred): the reading-order assertion does not discriminate (its data gives p = 1); the pinned Monte Carlo p depends on the NumPy Generator.permuted stream across versions; there are in-body and duplicate imports in the test
Task 6: dispatched (sonnet, multi-file integration), BASE 8e673a8. It carries the Task 3 _weak ruling, and the Task 5 ⚠️ that S2 must get observable items only and that BH must be applied by the caller.
Task 6: implemented d97df4f (283 passed), DONE_WITH_CONCERNS (_weak ruling applied; p1_dev_diagnostics.py imports _hidden_score/_weak/run_weights whose semantics changed); review dispatched (opus)
Task 6: review (opus) Needs fixes. 2 Important, both plan-mandated. Rulings:
  Ruling I1 + Minor 1: an item that reports an ambiguity does not also report weak. Its ambiguity members are excluded from _weak eligibility.
    - For an S2-driven ambiguity (S1 < 0.5), the members are the two most frequent classes among the competent acted readings at the item (ties broken by class key), not ranked[:2] of near-zero p_member.
    - Why: every S2 ambiguity also became a weak FP (measured: weak 1.0 outranks ambiguity 0.962). An unstable member choice also changes the (step, fields) identity, which breaks the differential.
    - If wrong: a genuine weak check co-located with an ambiguity is not reported (rare).
  Ruling I2: Diagnosis exposes `spent`, the locations of singletons dropped below τ_H whose refutation was attempted by probe (not symbolic repair) or never attempted. Report carries `spent`, and run_specdx's known = the v1 §11 item 13 set ∪ base.spent.
    - Why: τ_H dropped those singletons from base.defects, so the known-last ordering lost them and every mutant re-spent probes on base noise.
    - If wrong: a mutant's genuinely changed candidate at such a location is probed later.
    - Carry into Task 10 (which extends Report): preserve `spent` and the known union.
  Ruling Minor 2: Hidden(c) is maximised over the scenario's **observable** items only, consistent with "only observable items carry evidence". — If wrong: lower hidden scores where evidence exists only at unobservable items.
  Ruling: every new check in this fix goes in as assertions inside existing test functions, keeping the plan's test-count chain at 283. — If wrong: none.
Task 6: minor (deferred): no test pins the _weak ruling for both view orders (fold in if cheap); the gate is tested only with observable=frozenset(); pilot escalation counts non-acted views; the Unobservable note wording
Task 6: fix round 1 implemented af28129 (283 passed). Note for Task 10: add Report.spent to to_json.
Task 6: fix round 1/5 (4 addressed, 0 open; commits d97df4f..af28129). Mutation checks confirmed each fix is guarded.
Task 6: complete (commits 8e673a8..af28129, review clean after 1 fix round)
Task 6: minor (deferred):
  - ruling (b), _competent_pair, is not exercised by any test (the old rule passes all tests);
  - scores.py imports specdx.active.refute (infer→active dependency; move SYMBOLIC_WITNESS to types.py);
  - the comment above known omits base.spent;
  - the S1 ranked tie-break uses dict order;
  - Report.spent is not in to_json (Task 10).
Task 7: dispatched (sonnet; runs read-only on the P1 cache), BASE af28129
Task 7: implemented 0d30803 (286 passed). Replay verdict GO: (a) llama lowest κ on 21/21; (b) no acting model θ<0.5; (c) 61/91 = 67.0% Unobservable. Base act/κ/θ: llama 0.158/-2.319/0.961, qwen2.5 0.281/-1.244/0.610, qwen3 0.340/-0.824/0.743. Work unchanged (md5).
Task 7: review Approved with 1 Important (plan-mandated): the replay's read-only claim is not enforced. _never's RuntimeError is caught by Runner and cached as an error run; this was reproduced on a scratch cache. Rulings:
  Ruling: enforce read-only.
    - After Workbench.open, the script swaps bench.runner.cache for a ReadOnlyRunCache whose put raises ReplayWriteError(BaseException).
    - _never raises the same error, so `except Exception` cannot swallow it.
    - The docstring claims are corrected: tell/*.json and the mkdir happen on a corpus-cache miss.
    — Why: the P1 cache is irreplaceable. — If wrong: none.
  Ruling: also fix these cheap Minors.
    - Pass cfg.ambig_threshold and cfg.weak_threshold to diagnose (parity with the loop).
    - A mapping stop exits with status 2, distinct from NO-GO's 1.
    - main asserts 21 replays with the base first.
    - go() requires every other population model to be present in kappa.
    — If wrong: none.
  Ruling: the new checks are assertions inside the existing tests (count stays 286). Re-run the replay and confirm the report is byte-identical and the work md5 is unchanged.
Task 7: minor (deferred): the tests do not pin the literal P1 digest payload, the exact-200 guard, or p1_status; load_cache re-parses the cache and reads the private _steps_digest; the docstring overclaims the pinning; 15 mutants share κ (the Task 17 write-up should say so)
Task 7: fix round 1/5 (4 addressed, 0 open; commits 0d30803..50036a4). The GO was re-confirmed: report byte-identical, work md5 unchanged.
Task 7: complete (commits af28129..50036a4, review clean after 1 fix round). GO → continue to Task 8.
Task 7: minor (deferred): an uncaught ReplayWriteError or IndexError exits 1, the same as NO-GO; main's wiring is untested; db.use runs before the factory on a cache miss (unreachable)
Task 8: dispatched (sonnet, multi-file harness), BASE 50036a4
Task 8: implemented 2b81bde (315 passed); review dispatched (opus: interpreter-visible change)
Task 8: review (opus) Needs fixes, 1 Important (plan-mandated): READ_ONLY_TOOLS omits match_three_way, a vendor read-only tool granted to DE-FN-02. So DE-FN-02:reconcile is never nudged after a read-only reconciliation. Rulings:
  Ruling (write detection):
    - Invert to an explicit WRITE_TOOLS allow-list. wrote() counts only an ok outcome of a tool in WRITE_TOOLS.
    - Keep READ_ONLY_TOOLS (now including match_three_way) for documentation.
    - Add a test that every tool in build_registry().names() | {SEARCH_SPEC} is in exactly one of the two sets, so a new vendor tool fails the test until it is classified.
    — Why: a deny-list silently treats unknown tools as writes, and that skips nudges. — If wrong: an unknown write tool causes an extra nudge, which the model can answer with "xong".
  Ruling: also fix these Minors.
    - No nudge on the last round (`step < max_steps`).
    - Parsed text calls are recorded as native calls in the history (set reply["tool_calls"] before executing), as §2.1 requires ("executed like a native call").
    - Hash NUDGE into interpreter_settings so that the context digest covers its text.
    - Add executor-level tests: a drafting step with only read-only ok calls is nudged; an outage or ReadTimeout during a nudge round keeps infra_failure and the counts.
    — If wrong: none.
Task 8: minor (deferred): after a nudge, claims[step] holds the reply to the nudge; echoed native calls in a final text reply re-execute (spec-mandated; watch in the Task 18 smoke run); the valid-call rate is 0/0 when only invalid text calls exist (Task 13 must define it)
Task 8: fix round 1 implemented e2e8402 (320 passed). TEST-COUNT OFFSET: +5 from Task 8 on (plan says 315; actual 320). Later dispatches: expected = plan count + 5.
Task 8: fix round 1/5 (5 addressed, 0 open; commits 2b81bde..e2e8402). The reviewer confirmed the WRITE_TOOLS classification against the vendor implementations.
Task 8: complete (commits 50036a4..e2e8402, review clean after 1 fix round)
Task 8: minor (deferred): READ_ONLY_TOOLS is documentation only (the completeness test cannot catch a misplaced tool); a read-only drafting step reaches DONE status through the vendor's finish, so later code must key on slice gain, not on status
Task 9: dispatched (haiku), BASE e2e8402; test counts are plan + 5
Task 9: implemented 226cf2e (341 = 336+5 passed); review dispatched (sonnet)
Task 9: review Needs fixes, 1 Important: dd/mm/yyyy dates, times and fractions split into integers (resolution 9 says dates are not numbers). The real base corpus does not trigger it. Rulings:
  Ruling: add / and : to the _NUMBER guards, and add dd/mm/yyyy, hh:mm and a fraction to the "dropped" parametrisation. — If wrong: none.
  Ruling: also fix two Minors: multi-word keywords allow \s+ between words (NBSP, double space); add exceeds/exceeded cases (resolution 9 names them). Extra parametrised cases are allowed; record the new suite count. — If wrong: none.
Task 9: minor (deferred): the English inflection suffix applies to every keyword (capes); _near returns the leftmost marker, not the nearest; offset is relative to NFC text; ranges and thousand separators are inconsistent; no test for the _physical skip or for the canon-Tell PDF path; the physical condition is duplicated with space.py
Task 9: ⚠️ resolved: queueing the T-boundary scenarios and reporting the literals belong to Task 10 (check it there).
Task 9: fix round 1 implemented c358217 (347 passed). TEST-COUNT OFFSET now +11 (plan 336 at Task 9; actual 347).
Task 9: fix round 1/5 (3 addressed, 0 open; commits 226cf2e..c358217)
Task 9: complete (commits e2e8402..c358217, review clean after 1 fix round)
Task 9: minor (deferred): a bare number directly before ':' or '/' is dropped ("tối đa 5:"); the "phần trăm" unit is not NBSP-tolerant; the NBSP in the test is invisible
Task 10: dispatched (sonnet), BASE c358217; expected count = plan + 11. Carries: preserve Report.spent and the known union (Task 6 ruling) and add spent to to_json; queue T-boundary after canon; report literals via TLiteral.as_dict().
Task 10: implemented 0b183e8 (354 passed; offset now +12: one extra visited-skip test). Concerns: no Report.from_json (base report is in-process, ok); no competent model → ambiguity unlocalised (by design). Review dispatched (opus).
Task 10: review (opus) Needs fixes, 1 Important (test-only, plan-mandated): no test ever produces a non-empty Unobservable list (IdleLens cannot), so the field GO criterion (c) relies on is unguarded. Rulings:
  Ruling: inject a non-empty diagnosis.unobservable via the monkeypatch pattern and assert it reaches report.unobservable and to_json; rename the misnamed test. — If wrong: none.
  Ruling (⚠️ from review): oracle_interpreters requires θ ≥ 0.5 AND at least one acted behaviour in the specification. Under resolution 5, a model with no acted behaviour sits at the prior mode (0.75) with no evidence, and would waste ddmin runs that can never yield an acted view. — Spec §3.6 says "competent (θ ≥ 0.5)"; a prior-only θ is not evidence of competence. — If wrong: an oracle with fewer interpreters, possibly none (then unlocalised).
  Ruling: also add cheap tests: a unit test for split_persists' acted filter and & fields; an empty oracle gives unlocalised and ddmin_runs 0; plus the skip budget assertion and `assert report.items`. Fix the stale comments. New test functions are allowed; record the new count.
Task 10: minor (deferred): to_json omits ambiguity_keys (matters only if a base report is reloaded; check in Task 12)
Task 10: fix round 1 dispatch lost to the session rate limit (no commits; worktree clean at 0b183e8). Review saved to task-10-review.md, findings to task-10-fix1-findings.md. Fix round 1 re-dispatched to a fresh implementer (sonnet), FIX_BASE 0b183e8 (2026-10-12, new session).
Task 10: fix round 1 implemented dc0d42b (358 passed; offset now +16). Concern: the empty-oracle test calls split_persists directly. Re-review dispatched (sonnet).
Task 10: fix round 1/5 (F1, F2 and all six F3 items addressed, 0 open; commits 0b183e8..dc0d42b). No new breakage.
Task 10: complete (commits c358217..dc0d42b, review clean after 1 fix round). TEST-COUNT OFFSET now +16 (358 actual).
Task 10: minor (deferred): the empty-oracle test calls split_persists directly with localize_ambiguity patched. An ambiguity with an empty oracle IS reachable in the loop (every acting model θ < 0.5), and a loop-level test would monkeypatch oracle_interpreters → []. The report.runs and `assert report.items` checks are redundant with existing assertions.
Task 11: dispatched, BASE dc0d42b; expected count = plan + 16
Task 11: implementer dispatched (haiku). P1 mutant-pack fingerprint before Task 11 (md5 of sorted per-file md5s under work/mutants): b5478d9e37f21973db35a1006dd9ea0b
Task 11: implemented cd4fb44 (364 passed; Step 6: 36 mutants, differing []). P1 pack fingerprint after: b5478d9e… unchanged. Review dispatched (opus: P1-pack safety).
Task 11: review (opus) Approved, 0 Critical/Important. The reviewer checked every rewrite/delete path, and ran scratch crash and concurrency experiments: P1 packs are safe (copytree with dirs_exist_ok=False; rmtree only on staging).
Task 11: ⚠️ resolved: (b) no file under the P1 work/ has an mtime on or after 2026-10-12 00:00, so Step 6 wrote nothing. (c) The implementer ran the suite: 364. (a) carry into the Task 12/17 reviews: scorability is computed from base runs only, and observed_at is never an input to any method.
Task 11: complete (commits dc0d42b..cd4fb44, review clean)
Task 11: minor (deferred):
  - the settle copytree is not atomic: a crash mid-copy leaves a torn new pack, which later builds refuse (fail-safe, and it cannot affect the 36 existing P1 packs). Fix: copy to a sibling tmp, then os.rename;
  - target validity has no guard beyond the 20 dev mutants (all 36 are correct today): run the golden-run test over all mutants and assert step targets are in STEPS;
  - the refusal test tampers only with the first-built pack; there is no test for an event-kind mismatch or the unknown-kind ValueError;
  - the RED evidence for the two operator tests was not observed (the collection error masked them), though the reviewer confirmed they fail on the old code.
Task 12: dispatched (sonnet), BASE cd4fb44; expected count = plan + 16
Task 12: implemented 70a4184 (368 passed). No separate RED run captured (tests and code applied together). Review dispatched (opus: gate scoring).
Task 12: review (opus) Needs fixes, 1 Important (plan-mandated test gap): the verdict wiring in run_gate is untested. test_pilot.py:215's hidden loss holds whatever the scorable count, because B2 ties at 1.0. Mutations that drop scorable= or decide on the secondary tables pass all 16 tests. Rulings (task-12-fix1-findings.md):
  Ruling F1: in the scorability test, a recorder around gate_verdict asserts the primary tables, scorable counts and min_scorable; fix the comment. — If wrong: none.
  Ruling F2: also fix these cheap Minors.
    - Alias the observability.scorable import (the property reads as self-recursion).
    - The per-mutant "scorable" is `m.id in keep`, not a repeated threshold.
    - The cli gate calls config.check_population() right after GateConfig.load, before Workbench.open or all_mutants.
    - Add a delta_shift unit test (sign; a one-sided item is dropped) and assert the base-line and to_json fields.
    — If wrong: none.
  Ruling: counts(True) keeps mutants that failed to run. — Why: scorability is a base-run property, and any failure already forces INCOMPLETE. — If wrong: "N of M scorable" overstates by the failed ones on an INCOMPLETE gate.
  Ruling: check_population does not raise without a "screen" key. — Why: P1 configs must still load. — If wrong: an unscreened P1b gate. Carry into Task 17: experiments/p1b-gate.json must name "screen", and Task 18 runs from the worktree root (the screen path resolves against CWD).
Task 12: minor (deferred): no RED run was captured (brief Step 2 skipped)
Task 12: fix round 1 dispatched (resume of the implementer, sonnet), FIX_BASE 70a4184
Task 12: fix round 1 implemented 4deb67f (370 passed; offset now +18). Re-review dispatched (sonnet).
Carry into Task 14: the Task 10 fix added split_persists unit tests in tests/test_workbench.py with a fake runner. Task 14 batches split_persists (brief ~line 503), so that fake needs run_batch too; the brief's fake list omits it. Also, loop.py's oracle call now passes acted= (Task 10 fix).
Task 12: fix round 1/5 (F1 and all five F2 items addressed, 0 open; commits 70a4184..4deb67f). The recorder delegates to the real gate_verdict, and M1–M5 fail.
Task 12: complete (commits cd4fb44..4deb67f, review clean after 1 fix round). TEST-COUNT OFFSET now +18 (370 actual).
Task 12: minor (deferred): a whitespace-only line was left in run_gate where `keep = scorable(touched)` used to be
Task 13: dispatched (haiku), BASE 4deb67f; expected count = plan + 18 (358 → 376). The Task 8 deferred minor (a valid-call rate of 0/0) is resolved by the brief: zero calls give 0.0, so the model is not admitted.
Task 13: implemented 56bce90 (376 passed). Review dispatched (sonnet).
Task 13: review (sonnet) Approved, 0 Critical/Important. Screen JSON round-trips through load_population; infrastructure errors propagate uncached; outcome-free.
Task 13: ⚠️ resolved: the experiments/profiles/ files and load_profile tests ship in Task 15 (per its brief).
Task 13: complete (commits 4deb67f..56bce90, review clean)
Task 13: minor (deferred):
  - the screen CLI uses args.out.with_suffix(".json"), which replaces a dotted prefix (latent; Task 18's prefix has no dot);
  - errored screen runs are indistinguishable from idle ones (no error count in note);
  - there is no test of a model under two families;
  - .json is written before .md.
Task 14: dispatched (sonnet), BASE 56bce90; expected count = plan + 18 (363 → 381). Carries: the test_workbench.py fake needs run_batch; keep oracle acted=, the early check_population and the screen subcommand.
Task 14: implemented 41ecf5b (381 passed), DONE_WITH_CONCERNS: on probe-budget exhaustion, DirectedProbe dispatches the already-charged attempts (brief code). Review dispatched (opus: concurrency/cache).
Task 14: review (opus) Approved, 0 Critical/Important. Scratch comparisons of the sequential Runner and ParallelRunner (1 and 3 workers) gave identical results, byte-identical runs.jsonl and executed=2. The probe-budget concern was verdicted as no deviation (every charged attempt is executed, and the witness is the first in submission order, per planning resolution 11).
Task 14: ⚠️ resolved: Task 15's brief assigns hosts round robin per worker (test_workers_take_the_ollama_hosts_round_robin).
Task 14: ⚠️ Ruling: "byte-identical to N = 1" holds for deterministic executors and for any re-run over the run cache. Fresh real-model runs with OLLAMA_NUM_PARALLEL > 1 are not guaranteed bit-identical (batched inference). Carry into the Task 16 (RUNNING.md) and Task 17 (pre-registration) dispatches: state it that way. — Why: it is honest about what the test proves. — If wrong: none.
Task 14: complete (commits 56bce90..41ecf5b, review clean)
Task 14: minor (deferred):
  - after a failure, in-flight worker runs keep going and nothing calls close(), so exit waits (fix: try/finally close with cancel_futures in the CLI);
  - no test of a duplicate miss within one batch or of hits mixed with misses;
  - the byte-identical test does not compare the two runs.jsonl files;
  - --workers accepts 0 or negative values (silently sequential).
Task 15: dispatched (sonnet), BASE 41ecf5b; expected count = plan + 18 (375 → 393)
Task 15: implemented ed255a2 (393 passed). Step 8 was not run by the implementer, because the controller's dispatch forbade real-Ollama calls.
  Controller ran Step 8 (read-only):
  - doctor exited 0; every package, postgresql, vendor and ollama check is ok; vram ok at 16 GiB (the NVML mismatch of 2026-10-11 is gone);
  - missing: qwen3:30b-a3b, gpt-oss:20b, mistral-small3.2:24b;
  - cache-info: 6300 runs (llama3.1:8b 2058, qwen2.5:7b-instruct-q4_K_M 2037, qwen3:14b 2205), all "(not recorded: a P1 run)";
  - P1 work/ files modified today: 0 before, 0 after.
  Review dispatched (sonnet).
Task 15: review (sonnet) Approved, 0 Critical/Important. Digests resolve before any put (sequential, and parallel through serial key computation). Hosts are bound per worker through a per-call closure. doctor and cache-info write nothing beyond mkdir of work/.
Task 15: ⚠️ resolved: the controller's Step 8 run covers the real profile and doctor behaviour; the implementer ran the suite (393).
Task 15: complete (commits 41ecf5b..ed255a2, review clean)
Task 15: minor (deferred):
  - scripts/verify_vendor.sh's mktemp -d has no cleanup trap, so every doctor run leaks an extract into /tmp;
  - a partial host fetch leaves `listed` half-filled and gives a misleading KeyError later;
  - doctor's /api/show post is outside the try, so a timeout crashes doctor;
  - doctor warns where ollama_digests raises (a model missing on some hosts);
  - Focus 3 is pinned only at the ollama_digests level, not at the Runner/cache level;
  - SPECDX_OLLAMA_URL="" causes an IndexError at import.
Task 16: dispatched (haiku), BASE ed255a2; expected count = plan + 18 (389 → 407). Carries the Task 14 byte-identical ruling.
Task 16: implemented 4c35528 (407 passed). Review dispatched (sonnet).
Task 16: review (sonnet) Approved, 0 Critical/Important. Every command, flag, exit code and path in the guide matches cli.py; F6's θ-ordering check is placed between the smoke run and the gate. (The first handback lacked the verdicts; the controller asked for them, and they were supplied.)
Task 16: ⚠️ resolved:
  (1) Task 18's brief prefixes every interpreter command with SPECDX_WORK;
  (2) Task 17 creates experiments/p1b-gate.json and p1b-preregistration.md;
  (3) Task 17's brief config carries "screen": "results/p1b-screen.json";
  (4) smoke-run appends to the P1 cache are intended (Task 18 preamble).
Task 16: complete (commits ed255a2..4c35528, review clean)
Task 16: minor (deferred): test_docs checks only subcommand names, not flags; the §2 `docker compose up -d` line is for a fresh machine (Task 18 skips it here)
Task 17: dispatched (sonnet), BASE 4c35528; expected count = plan + 18 (412 → 430).
  Carries:
  - the Task 2 Rasch penalty wording;
  - the Task 14 byte-identical ruling;
  - the Task 7 shared-κ disclosure (replay: 14 of 20 dev mutants share one κ triple at 3 decimals, 2 share another; 7 distinct triples across the 21 specifications);
  - the Task 11 ⚠️(a) targets-are-metadata statement;
  - "screen" kept in p1b-gate.json.
Task 17: implemented 788a30a (430 passed; 23 in test_preregistration). All four carried rulings applied; the shared-κ counts were confirmed (14 + 2 + 5 singletons = 21; 7 triples). Review dispatched (opus: frozen pre-registration).
Task 17: review (opus) Needs fixes. Every stated number matches the code, and the 4 carried rulings are accurate. 4 Important, all plan-mandated:
  I1: the oracle row omits the Task 10 acted requirement;
  I2: τ = 0.05 and parts of resolutions 5 and 6 are missing;
  I3: the "test pins every value" claim is false;
  I4: the disclosure omits the first-draft NO-GO on 21/21.
  Rulings (task-17-fix1-findings.md):
  Ruling: fix all four. — Why: the plan says resolutions 1–8 and the pre-check numbers are recorded there, the frozen document must state what the code does, and the Task 10 ruling postdates the brief. — If wrong: none.
  Ruling I3: extend the test to pin every importable stated value (fit_rasch tol/iters, SpecDxConfig k_max/p_single/tau/ambig_threshold, TEMPERATURE, the gate's 3 wins, screen scenario/variants/seeds), and narrow the sentence for literals. — If wrong: none.
  Ruling: also fix Minors 1–6 (criterion labels, the shared-κ bullet, provenance, "200 per specification", an "Other rules the code applies" section, needles). — Why: cheap, and the document is about to be frozen. — If wrong: a longer document.
Task 17: fix round 1 dispatched (resume of the implementer, sonnet), FIX_BASE 788a30a
Task 17: fix round 1 implemented 882100b (441 passed; test_preregistration 34; offset now +29). Re-review dispatched (opus: frozen document, statements must match the code).
Task 17: fix round 1/5 (I1–I4 and Minors 1–5 addressed, 0 open; commits 788a30a..882100b). Every added sentence was verified against the code at its cited file:line.
Task 17: complete (commits 4c35528..882100b, review clean after 1 fix round). TEST-COUNT OFFSET now +29 (441 actual; Task 18's precondition "412" → 441).
Task 17: Ruling (out-of-scope observation, confirmed by the controller): the pre-registration and resolution 5 say the family effect is "estimated and reported", but Posterior.family_effect (latent.py:204) never reaches Report.to_json, the gate's base line or any output. The final fix wave adds family_effect to Report/to_json and to the gate base line, so the frozen document's statement holds. — Why: resolution 5 and spec §3.3 keep the family effect; weakening the document would drop a promised output. — If wrong: one extra field in the report JSON.
Task 17: note for the author: the document records the replay verdict (GO) while its rule says nothing changes after a P1b result is seen. This is disclosed, but the author should confirm before freezing.
Final review: dispatched (opus), range e66bfa5..882100b (27 commits, 64 files). It triages the deferred minors and carries the family_effect ruling.
Final review (opus): With fixes. 1 Critical (C1, the family effect is not reported; the existing ruling), 2 Important, 7 Minor; 0 must-fix deferred minors. 441 passed, ruff clean.
  Disclosure: the reviewer ran `ollama --version` (a version query, no model call, cache untouched), a slip against the no-real-Ollama rule.
  Rulings (final-fix-findings.md):
  Ruling C1: add family_effect to Report/to_json, the gate base line and mutant lines, with a docstring note on logit(0.75). — If wrong: none.
  Ruling I1: gate mutant lines carry theta, kappa, family_effect, items and literals; the base line carries literals and family_effect (spec §3.7/§3.8; a plan gap). — If wrong: larger gate JSONL.
  Ruling I2: operational plus documents, no executor change.
    - Task 18 uses --workers 1 unless the human sets OLLAMA_NUM_PARALLEL ≥ 2 and confirms 100% GPU under two-worker load.
    - Correct RUNNING.md, the pre-registration's execution paragraph and the plan's Task 18 text.
    - Add smoke-run checks (ReadTimeout claims = 0, error runs per model, parsed_from_text and duplicate evt rows, every population model acted on the base).
    — Why: a queued timeout would be cached as behaviour in the irreplaceable cache. — If wrong: a slower Task 18.
  Ruling M1:
    - `specdx run` calls check_population();
    - `--variant base` runs with localize=False, matching the gate's base run.
    — Why: ddmin on the base is spent for nothing. — If wrong: a user of `specdx run` cannot get localised base ambiguities.
  Ruling M3: Workbench.open takes the main-database lock for every worker count. — If wrong: a test-harness adjustment.
  Ruling: also fix M5 (whitespace), M6 (resume wording) and M7 (pin p1_dev_diagnostics to 547abbd); update the Task 18 counts. — If wrong: none.
  Ruling M2: keep the prior-only θ in the pilot, the refuter order and B2's best interpreter. — Why: the screen's act rate ≥ 0.6 makes every population model act on happy, and the new smoke check (every model acted on the base) stops Task 18 otherwise. — If wrong: B2 could pick a non-acting model on some mutant, which is now visible in the per-mutant θ/κ (I1).
  Ruling: the Task 6 escalation item (the pilot escalation counts non-acted views) goes to the author before Task 18. — Why: research-design choice, irreversible once the gate runs, and spec §3.8 says escalation is unchanged. — If wrong: n/a (asking).
  Before P2/P3 (not P1b): pin numpy (the S2 Monte Carlo stream) and guard target validity over all 36 mutants.
Final fix wave: dispatched (sonnet), FIX_BASE 882100b
Final fix wave: implemented 5262909 (code) and 0227dcb (docs). 445 passed; test_preregistration 34. P1 work/ unchanged (0 files modified today; pack fingerprint b5478d9e… unchanged). Scoped re-review dispatched (sonnet).
Final fix wave: re-review (sonnet): all findings addressed (C1, I1, M1, M3, M5, I2 in all its parts, M6, M7), no new breakage. The θ = 0.75 smoke proxy cannot false-negative (n_m = 0 gives exactly 6/8). The smoke-check field names match Behaviour.to_json, and the checks ran on 300 copied P1 lines. The main lock at open is re-entrant per URL.
Final review: residual parked. docs/RUNNING.md §5, §6 and quick-start items 5–6 still show `--workers 2`.
  Ruling: parked and surfaced to the author. — Why: the guide is generic; its §4 now states the workers ≤ OLLAMA_NUM_PARALLEL rule, and the Task 18 plan uses --workers 1. — If wrong: an operator copying §5 on a host without OLLAMA_NUM_PARALLEL gets queue timeouts cached as behaviour.
STATUS 2026-10-12: Tasks 1–17 complete and reviewed, and the final whole-branch review fix wave is clean. HEAD 0227dcb, 445 passed, ruff clean.
  Task 18 (real models, model pulls, appends to the P1 cache) is held for the author's go-ahead. Open author decisions:
  (1) Task 18 go-ahead and workers (1 unless OLLAMA_NUM_PARALLEL ≥ 2);
  (2) whether the pilot escalation counts only acted views;
  (3) confirm the pre-registration records the replay GO;
  (4) the RUNNING.md --workers 2 examples.
  Workspace kept (Task 18 pending).
AUTHOR DECISIONS (2026-10-12, user replied "Đồng ý" to the controller's five-point summary, accepting each recommendation):
  (1) Task 18 is approved: pull the 3 missing gpu16 models, then the screen, the smoke run and gate v2, appending to the P1 cache;
  (2) --workers 1 (OLLAMA_NUM_PARALLEL unset);
  (3) pilot escalation's class trigger counts only acted views; the spec §3.8 line and the pre-registration are updated;
  (4) the pre-registration may record the replay GO (no change);
  (5) the RUNNING.md examples become --workers 1, with a note on when 2 is allowed.
Author-decisions change: dispatched (sonnet), BASE 0227dcb. Brief: author-decisions-brief.md.
Task 18: started 2026-10-12 (author-approved).
  Step 1 facts:
  - GPU: NVIDIA GeForce RTX 5070 Ti, 16303 MiB.
  - The Ollama service Environment has no OLLAMA_NUM_PARALLEL, so --workers 1 throughout.
  - Disk: 126 GB free.
  - Installed before pulls: granite3.3:8b fd429f23b909, llama3.1:8b 46e0c10c039e, qwen2.5:14b 7cdf5a0187d5, qwen3:14b bdbd181c33f2.
  Ruling: pull only the 3 missing candidates (qwen3:30b-a3b, gpt-oss:20b, mistral-small3.2:24b), not the brief's loop over all 7. — Why: re-pulling an installed tag can silently update it to a new build (a new digest), changing models against P1 for no gain. — If wrong: none (the installed tags already pass doctor).
  Pulls running in the background, in parallel with the author-decisions change (pulls touch no repo or DB).
Author-decisions change: implemented 1cdc98c (447 passed). Review dispatched (sonnet).
Author-decisions change: review (sonnet) Approved, 0 Critical/Important. The controller confirmed 447 passed, ruff clean, tree clean at 1cdc98c.
Author-decisions change: complete (commits 0227dcb..1cdc98c). The Task 18 precondition count is now 447 (the plan text says 445).
Author-decisions: minor (deferred): `_unacted` test helper is unannotated; there is no loop-level escalation test
Task 18 Step 1 done:
  - pulled mistral-small3.2:24b 5a408ab55df5, gpt-oss:20b f38aa0c53da5 and qwen3:30b-a3b ad815644918f;
  - qwen3:14b is served at 100% GPU (RTX 5070 Ti, 16303 MiB).
Task 18 Step 2 done:
  - doctor exited 0; all 7 gpu16 models have native tools and ok digests; vram ok; 81 GB free.
Task 18 Step 3:
  - the screen was launched detached (setsid nohup, --workers 1); the exit code goes to work/p1b-screen.exit;
  - the P1 runs.jsonl had 6300 lines before the screen (work/p1b-screen.from).
Task 18 Step 3 progress (17:03 KST): 23/63 runs. qwen3:14b runs take about 45 s. qwen3:30b-a3b is offloaded 31%/69% CPU/GPU at num_ctx 16384, about 15 min per run. One ReadTimeout claim so far: qwen2.5:14b v0 s1, step DE-FN-03:draft_payment, with --workers 1, so it is not queueing (a slow call, behaviour under the P1 rule). No error runs.
  Concern for the gate: if a CPU-offloaded model (qwen3:30b-a3b, possibly mistral-small3.2:24b) is admitted, the gate's up to ~8,000 runs at about 15 min each would take far beyond the plan's 1–3 days. Estimate the gate runtime from the frozen population before launching, and report to the author if it is beyond the plan's estimate.
Task 18 Step 3 done: the screen exited 0, 63 runs from 15:01 to 19:07 KST (4 h 7 min). Population: openai gpt-oss:20b (act 0.914, valid 0.988) and qwen qwen3:30b-a3b (act 0.802, valid 0.952).
  Not admitted:
  - qwen3:14b: act 0.407;
  - qwen2.5:14b: act 0.580;
  - mistral-small3.2:24b: act 0.346;
  - granite3.3:8b: act 0.025;
  - llama3.1:8b: act 0.148.
  Mean seconds per run: qwen3:30b-a3b 943.7 (CPU-offloaded), gpt-oss:20b 140.0, qwen2.5:14b 157.4, mistral 212.7, granite 96.9, qwen3:14b 51.8, llama 46.3. Errors: 0. Ollama call failures: 1 (qwen2.5:14b, a ReadTimeout).
Task 18 Step 4 done: the population is frozen into experiments/p1b-gate.json; check_population ok; test_preregistration 34 passed; committed c22db31. P1 runs.jsonl is now 6363 lines (6300 + 63).
STOP before Step 5 (the author decides).
  Gate runtime estimate with this population at --workers 1: at most 21 × 380 = 7,980 runs; P1 executed about 300 per specification, so about 6,300. Half are qwen3:30b-a3b, so the mean is about 540 s per run, about 40–50 days. The plan said 1–3 days. The smoke run alone is about 60 × 540 s, about 9 h.
Task 18 Step 5: the smoke run was launched detached at 19:09 KST (budget 60, --workers 1). SMOKE_FROM=6363; exit code to work/p1b-smoke.exit. The gate is held for the author's runtime decision.
Task 18 Step 5 done (the smoke run exited 0 at 2026-10-13 18:33 KST): 142 runs (60 population + 100 probe), 0 errors.
  Smoke checks:
  (1) ReadTimeout runs: 0;
  (2) gpt-oss:20b 74 runs and 0 errors, qwen3:30b-a3b 68 runs and 0 errors;
  (3) parsed_from_text 11, duplicate-evt runs 0;
  (4) both models acted (449 acted behaviours across 36 items).
  θ ordering: OK (highest κ gpt-oss:20b, κ 2.146 and θ 0.0967 > qwen3:30b-a3b θ 0.0781).
  Report: 36 items, all observable; 0 Unobservable; defects 9 contradiction, 1 hidden, 4 weak; family_effect ±0.117.
  Mean seconds per run: gpt-oss population 154 / probe 192; qwen3:30b-a3b population 937 / probe 1083.
CONCERN: θ is far below 0.5 for both models (0.097 and 0.078), so neither is "competent". S2 was tested at 0 of 36 items, and the ddmin oracle will be empty. The P1 replay gave θ 0.61–0.96. A read-only θ diagnosis was dispatched (opus) before any gate launch: genuine, class-key artefact or bug?
Author decision 2026-10-13: option B (a larger GPU via SPECDX_OLLAMA_URL), and "find ways to make each run less long".
θ DIAGNOSIS (opus, theta-diagnosis.md): verdict (d), mostly (a) genuine; no code bug.
  - Agreement is low: 26.8% of same-item pairs between models, 29.6% within a model, and same-paraphrase pairs agree barely more (seed noise).
  - The class key adds some noise (invented soline ids, case and language variants of status, placeholders, inherited rows; columns.py:53–76, items.py:139–140). Removing all of it still gives θ only 0.381 / 0.100.
  - Spec-level mechanism: with ε item-specific, shared by models and allowed on admissible classes, the likelihood does not identify the θ level (a one-parameter family). θ ≈ the share of a model's runs in the plurality joint class. Re-estimated ε ≈ P1's plug-in (TV 0.009), so resolution 4 changes nothing here.
  - P1's high θ measured agreement on empty projections (DE-OP-01, P2, P6), with priors on a flat likelihood.
  - The θ-ordering pre-check passes on paper (0.097 > 0.078), but a bootstrap gives P(θ_gpt > θ_qwen) = 0.48, so it carries no information.
  - Gate prognosis:
    - Ambiguity cannot be found: S1 ≤ 0.119, and S2 and the ddmin oracle are inert.
    - Weak is plurality noise: all 4 base weak findings are noise.
    - Hidden is depressed.
    - Contradiction is unaffected.
    - So NOT PASSED is very likely (it needs 3 of 4 wins).
Gate held. Decision to the author: run the pre-registered gate anyway on a big GPU (days, a likely NOT PASSED), or stop P1b before the gate with a disclosed pre-gate diagnosis and redesign stage 2 / ≡_S (v3).
AUTHOR DECISION (2026-10-14): option B. Stop P1b before gate v2; no mutant run, no gate result, test split untouched. The pre-registration and the frozen experiments/p1b-gate.json are unchanged.
  Artifacts committed:
  - results/p1b-outcome.md;
  - results/p1b-theta-diagnosis.md;
  - results/p1b-smoke-base.json;
  - experiments/p1b_theta_diagnosis/ (archived scripts, `# ruff: noqa`; the controller re-ran load.py and summ.py, which reproduce θ, and the P1 runs.jsonl md5 is unchanged);
  - this ledger, as docs/superpowers/plans/2026-10-11-specdx-p1b-ledger.md.
  The P1 cache holds 6505 runs (6300 + 63 screen + 142 smoke). The mutant-pack fingerprint b5478d9e… is unchanged.
  Next: finishing-a-development-branch; then v3 design (brainstorming) on θ identifiability and comparison granularity.
