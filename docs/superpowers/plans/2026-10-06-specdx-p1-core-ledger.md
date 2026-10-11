# SDD ledger — plan: docs/superpowers/plans/2026-10-06-specdx-p1-core.md

Spec: docs/superpowers/specs/2026-10-06-specdx-design.md (binding authority).
Worktree: .claude/worktrees/specdx-p1, branch worktree-specdx-p1, started from fbfc839 (master).
Models: implementers haiku (plan carries complete code), Task 20 sonnet; task reviewers sonnet; final review opus.

## Pre-flight scan

Evidence base: before execution, every code block of Tasks 1–19 was materialised in a scratch directory and run together. 130 tests were collected and passed; the default-port test was deselected because the scratch run used port 55499. That shared run is evidence that every producer→consumer interface below matches; each row names the interface checked.

### Task pairs sharing a file or interface

| Producer → consumer | Interface | Finding |
|---|---|---|
| T1 → all | specdx/config.py (BASE_PACK, PDF_DIR, WORK_DIR, DATABASE_URL, OLLAMA_URL, settings_for) | consistent; settings_for(pack, url) used by T3 |
| T2 → T3,T5,T6,T9–T19 | types.py (Scenario.params/signature/deliveries, State, InterpreterId.key, Behaviour json, Defect.key, RefuteResult, Refuter) | consistent |
| T3 → T4–T19 | tests/conftest.py fixtures pack/db/golden | consistent; db re-selects base pack on each use |
| T3 → T6,T7,T10,T13,T17,T19 | db.golden_tables, pack_digest, PackDatabase.use/isolated/materialize/golden_state | consistent |
| T4 → T6,T10,T19 | ColumnModel.of/.classes/.keys/.project/.diff | consistent |
| T5 → T6,T13 | lift_value, lift_invariant, instantiate, lift_expectations, column_type | consistent |
| T6 → T8,T10,T13,T17,T19 | WorkStep(.tables), load_work_orders, render_inputs; ClauseSet.from_pack/applies/violations/[id]; PAID constant | consistent |
| T7 → T9,T10,T15,T17,T19 | TellCorpus API, Chunk/Sentence, pdf_chunks; SpecVariant.digest | consistent |
| T8 → T9 tests,T10,T19 tests | run_case, CaseRun, reference_executor | consistent |
| T9 → T16,T19 | committee(models_by_family,…), pilot(models_by_family, theta); PLACEHOLDERS; SpecInterpreterExecutor | fixed during planning (committee previously derived families from names) |
| T10 → T14,T15,T16,T19 | RunCache, Runner(.executed), ItemView, SpecLens | consistent |
| T11 → T14,T16 | Observation, Posterior, ItemPosterior.map_set, fit | consistent |
| T12 → T14 | soft_hitting_sets | consistent |
| T13 → T19 | CombinedRefuter, SymbolicRepair, DirectedProbe | consistent |
| T14 → T16 | diagnose(…, diff, tau, ambig_threshold, weak_threshold) | consistent |
| T15 → T16,T19 | CellBandit, information_gain, should_escalate; localize_ambiguity | consistent |
| T16 → T19 | SpecDxConfig(localize_tests), LoopContext, Report(.behaviours), run_specdx | consistent |
| T17 → T18,T19 | GroundTruth, Mutant, all_mutants, dev_split | consistent |
| T18 → T19 | b2_report, b3_report, Prf, evaluate_reports(typed=), gate_verdict | consistent |
| T9 → T20 | variants.json rewritten by make-variants | consistent (save_variants keeps _note) |
| T19 → T20 | experiments/p1-gate.json may be edited after capability pilot | consistent |

### Each task's own consistency (its tests against its code, the files it creates against the files it later touches)

| Task | Result |
|---|---|
| T1 | ok — 3 tests; .gitignore now lists .claude/ (Ruling 1) |
| T2 | ok — 9 |
| T3 | ok — 7 |
| T4 | ok — 6 |
| T5 | ok — 5 |
| T6 | ok — 11 |
| T7 | plan said 11 tests, actual 10 — corrected (Ruling 2) |
| T8 | ok — 5 |
| T9 | ok — 8 |
| T10 | ok — 8 |
| T11 | ok — 7 |
| T12 | ok — 6 |
| T13 | ok — 5 |
| T14 | ok — 5 |
| T15 | ok — 9 |
| T16 | ok — 4 |
| T17 | ok — 2 (step 4) + 6 = 8 |
| T18 | ok — 6 |
| T19 | plan said 7 tests, actual 8 — corrected (Ruling 2) |
| T20 | operational (real models), no code |

Rubric-defect scan: no test that asserts nothing, no verbatim duplicated logic block mandated. Note: the plan imports private names from frozen vendored TonomyX (_Record, _arguments, _invented_numbers, _plain, _record_inspection, _record_shipment) — deliberate reuse of frozen code, recorded as Ruling 3.

## Rulings

- Ruling 1: add `.claude/` to the plan's .gitignore (Task 1) — the worktree lives under the main checkout's .claude/worktrees, and the main checkout must not pick it up as untracked content — if wrong, one harmless ignore line.
- Ruling 2: correct expected test counts in the plan for Task 7 (10) and Task 19 (8); commit 00b0186 — measured by collecting the materialised tests — if wrong, an implementer sees a count mismatch and reports it.
- Ruling 3: importing private helpers from vendor/tonomyx is accepted — vendor is frozen at a pinned commit, so the names cannot drift; the alternative is duplicating frozen logic — if wrong, a future vendor bump breaks imports loudly at test time.

## Progress
Task 1: complete (commits 00b0186..fa9835c, review clean)
Task 1: minor (deferred): scripts/verify_vendor.sh never removes its mktemp dir (add trap rm -rf)
Task 1: minor (deferred): .gitignore `work/` is unanchored (prefer `/work/`)
Task 1: minor (deferred): tests/test_vendor.py default-port check fails if SPECDX_DATABASE_URL overrides the port
Task 2: ⚠️ resolved — Behaviour/State are never hashed downstream (all lookups key by run_key; plan code checked)
Task 2: complete (commits fa9835c..e10cbc0, review clean)
Task 2: minor (deferred): Scenario.deliveries() yields a zero-unit receipt when received < grn_count (only enumerate_scenarios guards it)
Task 2: minor (deferred): cell_of tests miss the discount == cap boundary
Task 3: review — 3 Important (all plan-mandated), Ruling: fix all three —
  Ruling: PackDatabase.use() must forget the loaded pack (clear _digest/_settings/_pack) before drop_everything, plus a test that a failing pack load leaves use(BASE_PACK) able to reload — mutants that fail to load are expected input (Review Focus 2) — if wrong: one extra reload per failure.
  Ruling: rollback test must check from a separate connection after isolated() exits — the plan's version cannot fail — if wrong: none.
  Ruling: materialize()/reconcile_value() let psycopg.OperationalError and psycopg.InterfaceError propagate and treat only other psycopg.DatabaseError as refusal/ERROR (RaiseException is InternalError, CardinalityViolation is ProgrammingError, both DatabaseError) — infrastructure faults must not be scored as spec refusals — if wrong: a genuine refusal class raised as OperationalError would crash a run instead of scoring None (visible, not silent).
Task 3: fix round 1/5 (3 addressed, 0 open; commits b2434ed..3713311)
Task 3: complete (commits e10cbc0..3713311, review clean after 1 fix round)
Task 3: minor (deferred): split_background annotated tuple[dict, ...] instead of tuple[dict, dict]
Task 3: minor (deferred): write_rows silently drops keys that are not schema columns
Task 3: minor (deferred): PackDatabase accepts any URL and drops its public schema — no guard against 55432/55433
Task 4: ⚠️ resolved — diff naming all columns of a one-sided row is fine (field matching tests truth ∈ reported); callers pass state.tables (Task 10 SpecLens)
Task 4: complete (commits 3713311..207df92, review clean)
Task 4: minor (deferred): ColumnModel frozen dataclass with dict fields is not hashable; evt projection hardcodes (actor, case_id)
Task 4: minor (deferred): column tests miss one-sided non-evt rows, apr.noi_dung free text, evt=REF exclusion
Task 5: ⚠️ resolved — structural applicability is enforced by ClauseSet.applies in Task 6 (test_lifted_clauses_do_not_apply_to_two_receipts)
Task 5: complete (commits 207df92..a282860, review clean)
Task 5: minor (deferred): lift_expectations lifts a numeric key present in only some canon scenarios; a literal 0 in an int column would lift to {discount_pct} (latent: no such cell today)
Task 5: minor (deferred): instantiate uses ^…$ (matches trailing newline) and raises KeyError on unknown placeholders
Task 6: review — 1 Important (plan-mandated): Vietnamese clause descriptions. Ruling: keep — Clause.description is shown to interpreters in DirectedProbe's Vietnamese goal suffix (Task 13), so it is prompt text, which the constraint allows — if wrong: six strings to translate later.
Task 6: ⚠️ resolved — de_scope and guardrail triggers act through execution (refusals, trigger errors) rather than as state clauses, consistent with spec §11 item 7's clause families; Ruling: no extra clause family in P1 — if wrong: scope/trigger defects surface only indirectly (step status, golden mismatch).
Task 6: ⚠️ resolved — invariants without scenario literals stay anchored to grn12 per spec §11 item 1 (intended)
Task 6: complete (commits a282860..ae67095, review clean after ruling)
Task 6: minor (deferred): no behavioural tests for reconcile (error = violation), guard_pay, vocabulary, blank, golden_event families; lifted id set not pinned
Task 6: minor (deferred): add a comment that Clause.description is prompt text (Vietnamese by design) and why master-table blank columns are skipped
Task 6: minor (deferred): Clause/WorkStep frozen with dict fields (unhashable); unknown expectation key raises KeyError in evaluate
Task 7: review — 1 Important (plan-mandated): PDF chunk cache keyed only by PDF bytes, so chunking changes are served stale. Ruling: fix — key on sha256(pdf bytes + f"{doc}|{words}|{CHUNKING_VERSION}") with CHUNKING_VERSION = 1 — reproducibility of the Tell corpus is a research requirement — if wrong: one re-extraction.
Task 7: fix round 1/5 (1 addressed, 0 open; commits 49a1302..69fd265)
Task 7: complete (commits ae67095..69fd265, review clean after 1 fix round)
Task 7: minor (deferred): pdftotext output decoded with locale encoding (pass encoding="utf-8"); cache write not atomic; stray PDF name → bare KeyError
Task 7: minor (deferred): chunks exceed 300 words where paragraphs are long (BA:004 = 694 words, TOC); 35 "sentences" over 100 words from flattened tables
Task 7: minor (deferred): with_chunk accepts duplicate chunk ids; SpecVariant.digest recomputes pack hash on every call
Task 8: ⚠️ resolved — Ruling: spec §6.3 reproduction is satisfied by checking every clause (incl. the scenarios.json expectation clauses) on happy/discount8/grn12 plus the golden projection and event chain on grn12 — the vendored flow's outcome is defined by those same expectations and TonomyX's own golden test covers grn12 only — if wrong: a divergence in fields outside the expectations on happy/discount8 goes unnoticed.
Task 8: complete (commits 69fd265..a121298, review clean)
Task 8: minor (deferred): CaseRun.refused/claims and the _persona error branch are untested; render_inputs/human_context/TaskEnvelope sit outside per-step error handling
Task 8: minor (deferred): reference decide_po advances the PO regardless of quyet_dinh (reference-only, approve fixed)
Task 9: review — 2 Important (both plan-mandated).
  Ruling: search_spec replies are built from whole passages under SEARCH_REPLY_LIMIT = 12000 chars (top passage always kept, its text trimmed if it alone exceeds), with SEARCH_K = 5; other tool replies keep the 4000 limit; and every chat request sets options num_ctx = 16384 (NUM_CTX constant) — the 4000-char slice left 2–3 of 5 passages and could break JSON, so retrieval loss would masquerade as spec defects; larger replies need a context that Ollama's default would silently truncate — if wrong: longer prompts, more VRAM per run (16k ctx fits a 14B Q4 model on 16 GB).
  Ruling: gated (non-search_spec) tool calls are delegated to a vendored OllamaExecutor(model, tools)._execute(call, ctx, record) instead of a copied method body — guarantees enforcement identical to TonomyX (spec §6.3) — if wrong: coupling to a private method of the frozen vendor (covered by Ruling 3).
Task 9: fix round 1/5 (2 addressed, 1 new Important open — malformed-argument and search failures no longer recorded in record.calls; plus 1 new Minor off-by-one in the top-passage trim; commits f297c97..8260f4d)
Task 9: fix round 2/5 (2 addressed, 0 open; commits 8260f4d..a6c9307)
Task 9: ⚠️ confirmed real gap — the standing prompt lists only table names for read/draft scopes, while spec §2.1 puts the schema in T and the vendored prompt listed columns; interpreters would fail on column names and that would read as spec defects. Enters fix round 3.
Task 9: ⚠️ resolved — Ruling: keep search_kb in interpreters' tool set (it is the canon's de_scope, i.e. T); mutants edit the pack KB files as well as the corpus, so they are consistent; only ddmin's corpus-only sentence removal can leak KB text via search_kb for DE-SM-01/02/CO-01 — accepted as localisation noise — if wrong: ambiguity localisation on KB sentences is weaker for draft_quote.
Task 9: fix round 3/5 (1 addressed, 0 open; commits a6c9307..f24f439)
Task 9: complete (commits a121298..f24f439, review clean after 3 fix rounds)
Task 9: minor (deferred): an Ollama HTTP error returns a bare ERROR envelope, dropping tool calls already recorded in that step (use record.envelope)
Task 9: minor (deferred): think=False hard-coded for qwen3*; pilot() raises IndexError on an empty family list
Task 9: minor (deferred): executor tests miss max_steps exhaustion, unknown tool, HTTP 5xx, variant/suffix, save/load_variants round trip
Task 10: review — 4 Important (3 plan-mandated, reproduced) + 2 ⚠️. Rulings:
  Ruling: RunCache loads bytes, splits on b"\n", decodes+parses each line inside try (catch ValueError, KeyError, TypeError) — the plan's read_text/splitlines refused the whole file on a multibyte cut and split records on U+2028/U+0085 — if wrong: none.
  Ruling: RunCache.put writes a leading "\n" when the file (as loaded) ends without one — otherwise the first record after a crash is glued to the partial line and lost — if wrong: one blank line in the JSONL (skipped on load).
  Ruling: `blank` clauses are relevant only to steps whose slice owns the whole table (no keyed-only attribution) — a blank clause checks every row, so attributing it to keyed steps flips classes of steps that did not write the row; run-level V(b) still carries it for MHS — if wrong: blank violations on apr/cand rows are not attributed to any step's class.
  Ruling: Runner re-raises psycopg.OperationalError and psycopg.InterfaceError instead of caching them as error behaviours — infrastructure faults are not interpreter behaviour (same rule as db.materialize) — if wrong: a DB outage aborts the loop loudly instead of producing cached error rows.
  Ruling: the run key also includes a digest of the work-order steps — editing work_orders.json must not serve stale runs (same reasoning as the Task 7 cache key) — if wrong: one cache miss per work-order change.
  ⚠️ resolved: the spec's "step" key component maps to the probe suffix (runs are whole cases, so a per-step key does not exist).
Task 10: fix round 1/5 (5 addressed in code, 3 open — the multibyte-cut, U+2028 and blank-relevance regression tests are vacuous; commits 42aec23..3e6a508)
Task 10: fix round 2/5 (2 addressed, 1 open — the U+2028 half of the line-separator test holds a plain space; commits 3e6a508..e3dc918)
Task 10: fix round 3/5 (1 addressed, 0 open; commits e3dc918..85a6280)
Task 10: complete (commits f24f439..85a6280, review clean after 3 fix rounds)
Task 10: minor (deferred): invariants keyed on non-key columns (gates-count, pending-2, th01-*) are relevant to no step (run-level V only)
Task 10: minor (deferred): cache returns shared objects (dataclasses.replace is shallow); key-sensitivity to variant/model digest/suffix untested
Note for dispatches: never send raw special Unicode in SendMessage text (U+2028 was normalised to a space); use chr() forms.
Task 11: ⚠️ resolved — Ruling: keep ε as the smoothed empirical distribution over all outputs (plan-mandated) — if wrong: dominant-class responsibilities sit slightly below θ-driven values. run_key is unique within an item (one view per behaviour per step; committee skips pilot ids; probe behaviours never enter the posterior). Reviewer measured the family-pull risk: good-model θ 0.05/0.25/0.5/0.7 at 2/3/5/11 items — real runs have 11 items per behaviour.
Task 11: complete (commits 85a6280..c210a34, review clean)
Task 11: minor (deferred): E-step recomputes θ-independent structure each iteration (3.7 s for 60 items × 150 obs); map_set tie-break prefers the lexicographically larger set
Task 11: minor (deferred): fit lacks input checks (duplicate run_key, model in two families, theta0 ∉ (0,1)) and a convergence flag; pooling test cannot fail
Task 12: ⚠️ resolved — Ruling: no PySAT fallback in P1; the hitting-set universe holds only clauses violated with ≥1% weighted mass, far below 100 in practice — if wrong: diagnose slows to seconds per scenario when an interpreter population violates ~100 clauses at once.
Task 12: complete (commits c210a34..2f7f5e5, review clean)
Task 12: minor (deferred): min_mass pruning unsound when tau < min_mass (benign at defaults); no docstring/input validation; duplicate violation sets not merged
Task 13: review — 2 Important (#2 plan-mandated; #1 reproduced on the real DB). Rulings:
  Ruling: SymbolicRepair rejects any candidate (and witness) whose protected tables differ from the start state — cases, insp, ncr, ship always; grn unless its (ma_grn, po, sl_nhan) rows equal the scenario's delivery truth (repair may move an agent's wrong receipt towards the truth, never elsewhere) — spec §3.6/§11: repair keeps scenario facts untouched; editing a receipt turned a Contradiction into a Hidden requirement — if wrong: a few repairs that touch environment rows are no longer found.
  Ruling: DirectedProbe builds its goal from a per-scenario description — new `describe(clause, scenario)` in spec/clauses.py renders lifted golden/invariant clauses and expectations from the instantiated payload, other clauses keep their description — lifted clause descriptions carried canon literals (15) on non-anchor scenarios — if wrong: goal wording only.
  ⚠️ resolved: the loop passes ranked_interpreters (θ-descending) to the refuter factory and builds a fresh CombinedRefuter every iteration (Task 16), so the memo spans one fixed behaviour set; in-memory intermediate evaluation is plan design (soundness rests on the DB read-back).
Task 13: fix round 1/5 (2 addressed, 1 new Important open — _facts_kept rejects a GRN unchanged from a deviant start; plus Minor sorted(dict) crash and canon literal kept in lifted-invariant descriptions; commits efff7cb..d514886)
Task 13: fix round 2/5 (3 addressed, 0 open; commits d514886..8411af5)
Task 13: complete (commits 2f7f5e5..8411af5, review clean after 2 fix rounds)
Task 13: minor (deferred): the _facts_kept multi-row regression test uses one ncr row (cannot catch a return to sorted(dict))
Task 13: minor (deferred): repair budget ignores baseline evaluations; probe `allowed` is the union over starts (repair uses per-start); `attempts` overstated; probe seed base 100 is a magic number; no repair for `vocabulary`
Task 14: review — 2 Important (plan-mandated) + 2 further spec-formula divergences. Rulings (spec §3.4 is binding):
  Ruling: Hidden score = max over the scenario's items of Σ_A post(A)·1[A contains a class whose step-relevant V equals {c}], over items where some view violates c; fall back to the hitting-set support only when no item's views mention c — the plan's support is confined to [0.95, 1] and cannot rank — if wrong: hidden scores less comparable to contradiction scores.
  Ruling: Weak score = Σ_A post(A)·1[some eligible clean class (≥ weak_min_runs runs, not in the MAP set) lies outside A] (1.0 if an eligible class is pooled as rare); the reported class = highest P(outside), then most runs, then smallest name (deterministic) — spec defines a union probability; the plan's max and set-order tie-break were PYTHONHASHSEED-dependent — if wrong: weak scores slightly higher with several deviant classes.
  Ruling: Ambiguity location = union of pairwise field diffs among the MAP-set classes when the MAP set has ≥ 2 classes, else the top two by p_member — spec: fields that differ between the classes of A_s — if wrong: wider location sets (matching uses intersection, so recall cannot drop).
Task 14: fix round 1/5 (3 addressed + defects[0] filtering, 0 open; commits a7a3b52..61aee6c)
Task 14: complete (commits 8411af5..61aee6c, review clean after 1 fix round)
Task 14: minor (deferred): no test distinguishes the MAP-set union ambiguity location from the top-two location; untested: dropped refuted sets of size ≥ 2, threshold gating, weak_min_runs, merge across scenarios
Task 15: review — 2 Important (plan-mandated). Rulings:
  Ruling: ddmin memoises the oracle on tuple(subset) (cache hits do not count against max_tests) and skips the complement pass when n == 2 — measured 18/36 and 58/133 duplicate probes, each a pilot of LLM runs — if wrong: none (same results, fewer probes).
  Ruling: the budget test counts predicate calls (≤ 3) and asserts an early, non-minimal result — the plan's assertion also held with no budget — if wrong: none.
  ⚠️ resolved: cells key on recv/disc relations and grn_count; grn_count is the only parameter that changes which lifted clauses apply, so the cell captures clause activation; ε is wired from SpecDxConfig.eps in the loop (Task 16); the ddmin pilot is seeded (seed 1, fixed variant), so the oracle is as deterministic as Ollama allows.
Task 15: fix round 1/5 (2 addressed, 0 open; commits 1d3ba15..0dbe29d)
Task 15: complete (commits 61aee6c..0dbe29d, review clean after 1 fix round)
Task 15: minor (deferred): information_gain treats a first-seen hypothesis as prior 1e-6 (large reward); one zero-gain probe can close a cell; ddmin complements use equality (fine for unique ids)
Task 16: ⚠️ resolved — localize_tests is wired into the ddmin budget by the Task 19 workbench localiser (max_tests=config.localize_tests). Ruling: runs-to-detection reconstruction (RQ4) is deferred to P3; the Report keeps the ordered behaviour list, which is enough to replay prefixes — if wrong: P3 adds per-iteration snapshots.
Task 16: complete (commits 0dbe29d..01af770, review clean)
Task 16: minor (deferred): lens.violations/views computed twice per pilot behaviour; fresh refuter per iteration repeats repair work (probe runs are served from the Runner cache); localisation does not re-merge/re-rank
Task 16: minor (deferred): pilot/ranked interpreters assume variant 0 and seed 1 are in the configuration; synthesize branch and to_json untested; budget test asserts <= rather than ==
Task 17: review — 1 Important (plan-mandated): A2/A3 tests tautological, 4 Tell mutants untested, and a drifted site string would silently yield an equivalent mutant. Ruling: all_mutants raises ValueError when a Tell edit matches nothing (A1: base corpus has no such sentence; A2: no KB file, principle or corpus sentence holds the old text) or when a Tell mutant's truth is empty; plus a test over all A mutants (truth non-empty, corpus digest differs from base) and literal-location checks — the design promises no equivalent mutants — if wrong: none.
Task 17: ⚠️ resolved — witness_state(C2) is None by design; run_gate (Task 19) never calls witness_state; C2 is proven by the DB-refusal test.
Task 17: fix round 1/5 (1 addressed, 0 open; commits df86172..ef1551d)
Task 17: complete (commits 01af770..ef1551d, review clean after 1 fix round)
Task 17: minor (deferred): A2 principle-match flag is vacuous (meta file never round-trips byte-identically) — guard rests on non-empty truth; check `old in meta["nguyen_tac"]` instead
Task 17: minor (deferred): all_mutants is one long function; A1 truth from base.corpus vs mutant corpora from pdf can disagree if callers pass different inputs; regenerate_sql hides generator stderr; copy_pack has no guard against dst inside vendor/
Task 18: review — 1 Important (plan-mandated): untyped false positives were charged to the reported type, inflating baseline untyped columns. Ruling: when typed=False, charge every unmatched finding to the mutant's true type — the untyped columns exist to read the gate fairly across types — if wrong: none (typed verdict unchanged).
Task 18: ⚠️ resolved — Ruling: B3 rows are runs (documented in the module); B2 uses canon scenarios only while B3 uses every run — B2 is naive differential testing on the canon scenarios, B3 is a population statistic — if wrong: B3 sees more data than B2 (only helps the baseline).
Task 18: fix round 1/5 (1 addressed, 0 open; commits 34d0085..251d987)
Task 18: complete (commits ef1551d..251d987, review clean after 1 fix round)
Task 18: minor (deferred): B2 returns [] if the best-θ model has no ok canon run (no fallback); evaluate_reports iterates reports not truths; matching tests miss k truncation and the sentence kind
Task 19: review — 2 Important (#2 plan-mandated). Rulings:
  Ruling: fail fast when a configured Tell variant does not exist (Workbench.loop_context raises ValueError if max(config.variants) >= variants.count); `make-variants` refuses to overwrite existing paraphrases unless --force; Runner takes `context_digest: str = ""` folded into the run key, and the Workbench sets it to a digest of the loaded TellVariants — otherwise v1/v2 runs are cached as errors and survive a later make-variants — if wrong: a cache miss after any variants edit (intended).
  Ruling: ollama_digests resolves lazily on first call, normalises an untagged name to ":latest", and raises (no silent fallback) for a model Ollama does not list or when Ollama is unreachable — the model digest must be in every run key — if wrong: runs fail loudly when Ollama is down instead of caching by bare name; `mutants` never needs Ollama because the lookup is lazy.
Note: Task 19 commit carries a "Claude Sonnet 5.5" trailer (the subagent's own model) — left as is.
Task 20: Ruling: split Task 20. Steps 1–5 (pulls, capability pilot, variants, mutants, smoke run) run first. The final whole-branch review and its single fix dispatch come next. Step 6 (the multi-day gate) launches only after that, and Step 7 follows when the gate ends. — Why: the gate must run on reviewed code, and the fix implementer's test suite rebuilds the database, which the brief forbids while a gate is running. — If wrong: the gate starts about an hour later.
Task 19: fix round 1/5 dispatched (commit dcc3ec4)
Task 19: fix round 1/5 (2 addressed, 0 open; commits 7b240d0..dcc3ec4)
Task 19: complete (commits 251d987..dcc3ec4, review clean after 1 fix round)
Task 19: minor (deferred): empty/negative variants tuple not guarded; "host:port/model" treated as tagged; digest KeyError for an unseen model lands in "could not run" (visible)
Task 20: Step 1 done (4 models pulled by controller); Steps 2–5 dispatched to sonnet implementer; final whole-branch review (opus) running on fbfc839..dcc3ec4 in parallel (read-only)
Final review (opus, fbfc839..dcc3ec4): With fixes. Findings: 1 Critical, 8 Important, 14 Minor. Rulings:
  Ruling C1 (Ollama outage cached as behaviour): fix as proposed. Retry httpx.TransportError (except ReadTimeout) and HTTP 5xx with logged backoff. When retries are exhausted, set an executor infra flag; Runner raises an uncached InfrastructureError. — An outage must never become data. — If wrong: none.
  Ruling I1: run_gate re-raises psycopg Operational/InterfaceError, httpx.TransportError and InfrastructureError. Any failed mutant makes the headline INCOMPLETE (passed=False). — If wrong: none.
  Ruling I2 (typed gate trivially passed for C/A/W because baselines only emit hidden): the gate verdict compares SpecDx typed F1 per type against each baseline's UNTYPED F1 for that type. The baselines get the type for free (Task 18 charges untyped FPs to the truth type), so a win needs better detection, not just labelling. The table keeps the literal typed-vs-typed comparison too; recorded as spec §11 clarification. — Spec §4.8 meant a test SpecDx could fail; baselines cannot type by construction. — If wrong: the gate is stricter than the literal text; the literal verdict is still printed and recomputable from per-mutant output (I8).
  Ruling I3: ddmin tests fails([]) first and returns unlocalised when the split persists without the pool. A localisation is unlocalised when the ddmin budget runs out before 1-minimality. Unlocalised ambiguity defects keep type ambiguity with an empty location (they cannot match a truth). The ambiguity differential against the base report uses the pre-localisation key (step + field set), and localisation applies to the mutant report only. — If wrong: lower ambiguity recall for SpecDx (conservative).
  Ruling I4: one stderr log line per executed (non-cached) run. Report counts population, probe and ddmin runs. New config field probe_budget (default 100, also in p1-gate.json) caps DirectedProbe and ddmin runs per specification. Candidates are refuted in descending score order. A singleton whose refutation was never attempted (budget exhausted) is typed Hidden, because Contradiction needs a failed refutation attempt. — If wrong: some C mutants are typed H when the budget is tight (visible in per-mutant output).
  Ruling I5: RUN_SCHEMA_VERSION constant plus executor settings (temperature, max_steps, timeout, NUM_CTX, SEARCH_K, SEARCH_REPLY_LIMIT) fold into the Workbench context_digest. Bump the version on any interpreter-visible change. — If wrong: an extra cache miss.
  Ruling I6: PackDatabase takes a process-lifetime pg advisory lock (re-entrant within one process) and fails fast if another process holds it. Refuse DATABASE_URL ports 55432/55433. — If wrong: pytest cannot run while a gate runs (already forbidden).
  Ruling I7: cross-cutting clauses are relevant only to the steps that own the rows they read. expect:apr_0102 → steps whose slice keys evt=SO-0031; guard:pay → steps owning pay; reconcile:consistency → steps owning vinv. No table-overlap fallback for these kinds. — Extends the Task 10 blank ruling. — If wrong: fewer items per clause.
  Ruling I8: write per-mutant results (truth, SpecDx/B2/B3 defects, matches, run counts, failure) to results/p1-gate-mutants.jsonl. — If wrong: none.
  Ruling minors fixed now: cache hit replaces interpreter too; Behaviour records model_digest (§4.2); evt count-invariant repair proposal (event append/remove) for H2 typing; run weights include θ (spec §3.3 line 163 is explicit; it beats the plan); check_model raise_for_status (an unpulled model is an error); docker-compose restart: unless-stopped; one end-to-end DB test with a scripted deviant executor through loop_context → run_specdx → run_gate on one C2 and one W1 mutant.
  Ruling minors kept as deviations (spec §11 entries): localiser k=5, not 20 (bounds ddmin cost in P1; ambiguities sourced beyond rank 5 go unlocalised, which is conservative). Step-level errors enter the class key, while run-level errors never get a class (C1 keeps infrastructure out). B2 uses all variants × seeds of the best model (more data, so it only helps the baseline).
  Ruling minors deferred: A1–A3 kill witnesses, prompt_eval_count truncation logging (check in the smoke log instead), column pin test breadth.
  Ruling Task 20 smoke: after the fix, rerun the smoke with budget 60 (covers grn12) and read the run log for failure/timeout rates before the gate. The budget-24 smoke runs are superseded by the RUN_SCHEMA_VERSION key change.
Task 20: Steps 2+4 done (commit a84ed85): usable = qwen3:14b, qwen2.5:14b, qwen2.5:7b, llama3.1:8b, granite3.3:8b; not usable = mistral-nemo:12b, phi4-mini, command-r7b; no family swap. build-tell BA 137 / DEMO 50 / SAD 19; 36 mutants. Step 3 BLOCKED (paraphrase prompt makes models invent placeholders). Step 5 diagnostic (variants [0], budget 24): 413 runs in 4h07m, all ok, unfinished; refuter runs dominate.
  Ruling: fold the paraphrase fix (item P) into the final fix wave — it is a code defect, and the wave is the only code-changing dispatch left — if wrong: none.
Final fix wave dispatched (opus) from a84ed85 with final-fix-brief.md (C1, I1–I8, P, minors, §11 docs)
Final fix wave: commits a84ed85..5500785 (9 commits), 227 passed, ruff clean. DONE_WITH_CONCERNS.
  Ruling: accept Deviation 5. Ollama's HTTP 500 "error parsing tool call" stays interpreter behaviour, not a retried infrastructure fault. — §6.5 counts a malformed tool call as behaviour, and fixed seeds make retries repeat it. — If wrong: a real 500 carrying that exact text goes uncaught (unlikely).
  Ruling: accept Deviation 1. An unattempted multi-clause candidate is dropped. — A contradiction needs a failed refutation attempt, and Hidden is defined for singletons only. — If wrong: lower contradiction recall when the budget is tight (conservative).
  Ruling (concern 1): refutation and localisation get separate per-specification budgets. probe_budget = 100 covers DirectedProbe only. A new localize_budget = 80 covers ddmin, so ambiguity is not starved by refutation (about 37 hitting sets × 9 runs measured). — Otherwise the ambiguity F1 is 0 by budget, and the gate needs C, H and W all to win. — If wrong: about +80 runs per specification (about 17 h over the gate).
  Ruling (concern 2): the paraphrase guard also requires equal sets of snake_case identifiers (for example search_spec, discount_pct). — A paraphrased identifier changes what an interpreter calls. — If wrong: more rejected paraphrases.
  Concern 3: the controller ran docker update --restart unless-stopped specdx-postgres (policy confirmed).
  Concern 4 accepted: open_units stays at 3 steps (P6 owns the whole po table).
Final fix wave: addendum commit 77647de (separate budgets, identifier guard); 231 passed
Task 20: Step 3 rerun BLOCKED, nothing committed. qwen3:14b passes the guard, but the paraphrases still fail semantically:
  - draft_purchase_order keeps inventing "trước" (a previous order);
  - record_goods_receipt drops `deliveries` (no underscore, so the identifier regex does not protect it);
  - one paraphrase adds a condition ("khi bán hàng");
  - "khớp lệnh" appears as jargon;
  - two variants of draft_payment are identical.
  An 87-seed sweep found no acceptable draft_purchase_order or record_goods_receipt.
  Ruling: fix it in one small dispatch, bundled with any load-bearing residual from the fix-wave re-review:
    (a) protected identifiers = snake_case tokens ∪ the work-order input keys that appear in the text (`deliveries`); prompt_for names them and rejection compares them;
    (b) the generic prompt adds "Không thêm, bớt hay thay đổi chi tiết, điều kiện hay đối tượng nào; không dùng thuật ngữ của lĩnh vực khác." (no added, removed or changed detail, condition or referent; no other-domain jargon);
    (c) reject a paraphrase identical to variant 0 or to an earlier variant of the same text, and retry with the next seed;
    (d) the CLI gets `make-variants --only KEY[,KEY] --variant N` to regenerate single statements without --force.
  Semantic acceptance stays a recorded read by the Task 20 agent, because the Tell variants are frozen once.
  — Why: variants are one axis of the interpreter population (§3.1), so they cannot be dropped; they must keep meaning.
  — If wrong: a few extra paraphrase calls.
  — Fallback if qwen3:14b still cannot produce draft_purchase_order or record_goods_receipt after this: allow --model per statement, using qwen2.5:14b (another usable model). Record which model produced each text in variants.json `_note`.
Final fix wave: scoped re-review (opus). Every finding and every post-brief ruling is addressed except I2 for A/W. One new Important: unlocalised ambiguities merge into one defect (loop.py:227-230). Plus 4 Minors.
  Ruling (I2 residual): the baselines are structurally blind to A and W. Their clause ids match 0/7 A truths and 0/13 W truths, but 9/9 C and 7/7 H, as measured by the re-reviewer. So:
    - C and H keep the comparison "SpecDx typed F1 > max baseline untyped F1".
    - A and W need SpecDx typed F1 ≥ 0.4. That is above the 0.333 that one TP with zero FP gives on 5 truths, so at least 2 TPs are required.
    - The gate still needs 3 of 4 types. gate_markdown names the rule used for each type, and spec §11 item 12 is corrected.
    — Why: a test SpecDx can fail on every type.
    — If wrong: recompute from the JSONL; the threshold is visible in the table.
  Ruling: fix the unlocalised-ambiguity merge before launch, using the reviewer's one-line key. Its effect cannot be recovered after the run.
  Ruling (minors fixed now):
    - the generic except in Runner checks executor.infra_failure;
    - _mutant_line records the pre-differential ambiguity keys and the keys the differential dropped, so the cross-scenario erasure risk can be measured;
    - the lock tests use try-lock;
    - an unknown expectation key raises.
  Ruling (wedged Ollama): a run in which every LLM call ended in ReadTimeout raises InfrastructureError and is not cached.
    — Why: a wedged server is infrastructure.
    — If wrong: a model that times out on every step stops the gate loudly.
  Ruling (standing-text drift): label lines of the standing instruction ("Label: {p}." segments) must appear verbatim in a paraphrase. prompt_for lists them, and rejection checks them. Only the prose is paraphrased.
  Ruling (process): because I ran the final review before Task 20 finished, these residuals are Task 20 blockers. They go as ONE bundled dispatch to the fix-wave agent, together with the paraphrase ruling above (a)–(d), followed by one scoped re-review. This deviates from "no second fix wave".
    — Why: the multi-day gate must not run on known-wrong code.
    — If wrong: about one extra hour.
Bundle 2: commits ce0af78..355dacd (4), 256 passed, ruff clean. Ruling: accept "--variant N" restricted to existing indices 1..count-1 — a partial new index would break the prompt for that variant — if wrong: none.
Bundle 2 re-review (sonnet): every finding addressed, no new Critical or Important breakage. Controller fixed Minor 1 (the exact-arithmetic bar) in commit 'fix: the absolute gate bar is checked in exact arithmetic'. Minors 2–5 deferred (TEMPLATE_NAMES lacks `lines`, which no text uses; no digest-change test; theoretical C1/D overlap; --variant without --only is ignored).
Final fix wave: complete (a84ed85..HEAD; one fix wave plus Bundle 2, both re-reviewed).
Task 20: Step 3, rerun 2, BLOCKED. 14/20 paraphrases are acceptable. 6 have no faithful text after 3 remakes each with qwen3:14b and qwen2.5:14b: decide_quote [1,2], decide_po [1,2], draft_sales_invoice [2], check_availability [1].
  Ruling: the controller added `make-variants --only … --seed S` (remakes no longer cycle on a fixed seed) and `--original` (the entry keeps the variant-0 wording, provenance "original: no faithful paraphrase"). 258 passed. For each failing entry, try seeds 3–8 with qwen3:14b; if none is faithful, use --original. Never hand-write. — Why: the user requires a fully automatic pipeline, and keeping the original wording is the honest automatic fallback. — If wrong: less wording diversity on the decide_* steps; provenance shows exactly where.
  Ruling: accept the agent's doubtful accepts. The trailing spaces in standing [1] are whitespace only. "đối chiếu" / "ba bên" are standard synonyms for three-way matching. "đề nghị thanh toán" ≈ "đề nghị chi". "lập" ≈ "soạn" for an invoice draft, and the tools only allow drafting. — If wrong: a little wording drift on 4 entries, recorded in the report.
  Ruling: start Step 3 from work/variants.candidate-rr2.json. It is the CLI's own output from rerun 2 at HEAD 355dacd, which has the same paraphrase code, so this is not a hand-edit. — If wrong: none.
Task 20: Step 3 complete (commit 1326093). 18 of 20 entries are paraphrased by qwen3:14b, some with seed 5 or 7. decide_quote [2] and decide_po [2] keep the original wording.
Task 20: Step 5 smoke (budget 60, base). 160 executed runs in 98 min (60 population + 100 probe, 0 ddmin), all ok. No retry, outage or timeout. Median 33 s per run, max 160 s. The base report holds 91 findings: 78 hidden (mostly unattempted), 12 contradiction, 1 weak.
  Ruling (controller commit be4b95b): on a mutant, refute the base report's locations last. An unattempted Hidden finding at a base location (any type) is base noise in the differential. Recorded in spec §11 item 13.
    — Why: base noise had spent the probe budget, so the mutant's own contradiction candidate would be typed Hidden by default, and unattempted type flips would become false positives.
    — If wrong: rescore from the cache (scoring code is not in run keys).
Task 20: Step 6 gate launched 2026-10-07 17:53 KST (PID 3544504, log work/gate.log, out results/p1-gate.json and results/p1-gate-mutants.jsonl). Estimate: about 380 runs per specification × 21 at about 100 runs/hour, roughly 3–3.5 days. Resume: rerun the same command, and cached runs are reused.
Controller commits review (opus, 355dacd..be4b95b): Needs fixes. Two Important findings: (a) known-last puts C2 truths last, because the base symbolically repaired them; (b) float ties counted as wins. The gate was stopped in its base phase (17 new runs cached) to land the fix.
  Ruling (commit cda573c): known = base contradictions plus base hidden findings not refuted by symbolic repair. The comparisons use exact F1. A merged hidden finding keeps an attempted note. The (b) noise rule requires type hidden. The variants _note is corrected (outside the digest). Spec §11 item 13 is updated, including the one-sided rule. 262 passed.
    — Why: symbolic repair costs no probe on the base, so a mutant that defeats it must be probed in score order.
    — If wrong: rescore; ordering only matters when the budget binds.
  Ledger correction: the smoke base hidden findings are 41 refuted by symbolic repair and 37 unattempted (not "mostly unattempted").
  Deferred minors: CLI flags silently ignored without --only (--seed with --original, --variant).
  ⚠️ For the verdict read: check whether the gate base holds the dev H1 truth clauses (H1-02, H1-03, H1-04 were base hidden findings in the smoke). If so, the key differential masks them for every method (spec §4.3), and H recall is capped near 2/5 for all methods.
Task 20: gate restarted 2026-10-07 18:08 KST (PID 3575998) on cda573c. The attempt-1 log is work/gate-attempt1.log.
Task 20: gate at 20:09 KST. The base is done: 200 population, 100 probe and 0 ddmin runs, 123 executed. SpecDx base findings: 12 C, 78 H, 1 W. B2 has 92 and B3 has 0. Mutant A1-01 is in progress, with no errors.
  ⚠️ confirmed: the dev H1 truth clauses (INV-0031:trang_thai, PAY-0031:trang_thai, APR-0210:muc_tu_hanh) are Hidden in the base report for both SpecDx and B2. The spec §4.3 key differential masks them for every method, which caps H recall near 2/5. Report this alongside the verdict.
  Watcher stopped at its 2 h limit and was not re-armed.
Task 20: gate PAUSED by user request 2026-10-08 ~13:39 KST (SIGSTOP, PID 3575998). Done: base + 5 mutants (A1-01, A1-xx, A2-xx, A3-00 per jsonl), C1-00 in progress; 1897 executed runs. Resume: kill -CONT 3575998 (or, if the process is gone, rerun the gate command — cache).
Task 20: complete (commit 3e83e9b) — gate NOT PASSED; diagnosis in results/p1-gate-diagnosis.md
