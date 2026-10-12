# P1b pre-registration

- **Spec:** `docs/superpowers/specs/2026-10-11-specdx-v2-design.md` (v2), a delta on `2026-10-06-specdx-design.md` (v1).
- **Plan:** `docs/superpowers/plans/2026-10-11-specdx-p1b.md`.
- **Rule:** this file is committed before the capability screen runs. Nothing in it changes after any P1b result is seen. `tests/test_preregistration.py` pins every named constant below to the code (for example `MAX_NUDGES`, `RASCH_L2`, `TAU_H`, the screen constants, `SpecDxConfig` defaults); the oracle's seed 1 is a literal in `specdx/active/loop.py` and is not pinned by a constant.

## 1. Offline replay

The mechanism-check criterion below was committed in the plan (commits e66bfa5 and 5dcf5fb) before the replay ran. The replay then ran, read-only, and its verdict was **GO** (`results/p1b-replay.md`, commits 0d30803 and 50036a4). The v2 inference ran on the P1 run cache: the base specification and the 20 development mutants, with the old population (qwen3:14b, qwen2.5:7b-instruct-q4_K_M, llama3.1:8b) and their 200 population runs per specification, 4,200 in all. No run was executed and no refutation was attempted.

- **GO criterion: a mechanism check** (author's ruling, 2026-10-12). It holds when all three are true:
  - (a) On the base and on each of the 20 development mutants, stage 1 gives llama3.1:8b, which never writes, a κ strictly below every other model's.
  - (b) On the same 21 specifications, every model with at least one acted behaviour has θ ≥ 0.5, and at least one model acted.
  - (c) At least 50% of the P1 base findings become Unobservable.
- **Caveat on (a).** The 21 specifications of criterion (a) are not 21 independent stage-1 fits: in the replay (`results/p1b-replay.md`), 14 of the 20 development mutants share one κ triple to three decimals and 2 share another, so there are 7 distinct κ triples across the 21 specifications.
- **Disclosure: the criterion was revised after a planning pre-check.**
  - The first draft required both of these on every specification: the model with the highest κ does not have the lowest θ, and θ(qwen3:14b) ≥ θ(llama3.1:8b).
  - Under that criterion the replay verdict was NO-GO on all 21 specifications: θ(llama3.1:8b) = 0.961 against θ(qwen3:14b) = 0.743. κ ranks qwen3 > qwen2.5 > llama, and 61 of 91 P1 base findings become Unobservable.
  - Only the 24 items of the three non-drafting steps (check_availability, P2, P6) are observable on the P1 population, and llama "acts" there by completing the step without writing.
  - The verdict does not depend on resolutions 2 and 4: with P1's plug-in ε, θ is 0.918 against 0.515; if every interpreter step must write to act, llama never acts and keeps 0.75 against 0.507.
  - The author's reason: on the P1 population only one family (qwen) writes, and Proposition 2′ needs at least two acting families, so a θ ordering there carries no information about v2.
  - The author therefore moved the θ ordering to the screened population (§4) and kept the mechanism check for the P1 cache.
- **Reported in addition:** the number of P1 base findings that become Unobservable, Oₛ per item, and the S2 p-values.
- **NO-GO** would have stopped P1b before any infrastructure work, with the result going to the author.

## 2. Interpreter population

| Rule | Value |
|---|---|
| Harness nudges per step | 2 (only for a step whose token drafts tables and that made no successful write call) |
| Screen runs per candidate | 9: base specification, scenario `happy`, Tell variants 0–2 × seeds 1–3, through the harness |
| Admission | act rate ≥ 0.6 and valid-call rate ≥ 0.8 |
| Ranking | act rate, then valid-call rate, then model name |
| Population | ≤ 2 per family, ≤ 4 in all, ≥ 2 families |
| Stop rule | fewer than 2 families admitted: P1b stops; thresholds are never lowered |

- **Act rate:** the fraction of (run, interpreter step) pairs with aᵦₛ = 1.
- **Valid-call rate:** the fraction of tool calls (native, or parsed from text) whose name is one of the step's tools and whose arguments validate against its schema.
- **Text calls:** a JSON object anywhere in the reply, fenced or in `<tool_call>` tags, naming the tool with `name` or `function` and giving the arguments with `arguments` or `parameters`.

## 3. Inference

| Parameter | Value |
|---|---|
| Act indicator aᵦₛ | the step completed (`done` or `needs_gate`) and, when its actor drafts tables (`de_scope.draft_tables`), its slice ends with a row the starting state did not hold |
| Stage 1 (Rasch) | P(a = 1) = σ(κₘ − δₛ), penalised by the ridge term (λ/2)·Σ(κ² + δ²) with λ = 1.0 (a N(0, 1/λ) prior on every κ and δ), mean δ fixed at 0, Newton to a change below 1e-6, at most 100 iterations |
| Observable item | Oₛ ≥ ω = 0.3 and at least m_min = 3 acted behaviours |
| Classes | the ≡_S projection of the step's slice (locked values and blank presence), without the step status |
| Stage 2 data | acted behaviours only |
| θ prior | Beta(7, 3); θ is the closed-form MAP, M-step θₘ = (Sₘ + 6) / (nₘ + 8); a listed model with no acted behaviour gets 0.75. The family effect (a family's mean logit θ minus the population's) is estimated and reported, but does not feed θ |
| Run weight | the mean, over the items where the run acted, of P([b] ∈ Aₛ) × θₘ; 0 for a run that acted nowhere |
| Error distribution ε | re-estimated in each EM iteration from the outputs not explained as faithful (add-one smoothing) |
| K_max, P(\|A\| = 1), τ | K_max = 6, P(\|A\| = 1) = 0.9, τ = 0.05 (the hitting sets cover at least 1 − τ of the weighted mass) |
| V(b) | violated clauses relevant to an item where b acted |
| Typing | only candidates whose every clause is read by an observable item; others are Unobservable, never scored |
| Hidden | refuted or unattempted singleton with Hidden(c) ≥ τ_H = 0.5; Hidden(c) is 0 when no acted view supports c, and is maximised over the scenario's observable items only |
| Weak | v1 rule on an observable item; a weak deviant is compared with the most probable class of the (clean) MAP set |
| S1 | P(\|Aₛ\| ≥ 2), reported at ≥ 0.5 |
| S2 | competent models θₘ ≥ 0.5; plug-in I(K; V \| M); 2000 permutations of V within each model, seeded; p = (1 + #{I_perm ≥ I_obs}) / (R + 1); tested with ≥ 6 behaviours, ≥ 2 classes, ≥ 2 variants; Benjamini–Hochberg at q = 0.1 per specification |
| Ambiguity score | max(S1, 1 − pₛ), with 1 − pₛ = 0 when S2 did not test the item |
| ddmin oracle | one re-run per (model with θₘ ≥ 0.5 and at least one acted behaviour in the specification, Tell variant) at seed 1, most competent first; a prior-only θ (0.75, no acted behaviour) is not evidence of competence, and with no such model the ambiguity stays unlocalised; passes while two acted re-runs differ on the ambiguity's fields |
| T-boundary scenarios | number within 40 characters of a comparison keyword in the same sentence; % or phần trăm → discount_pct; near số lượng, đơn vị, units or quantity → received; v − 1, v, v + 1 on the anchor; at most 12 per specification; queued after the canon scenarios |
| Budgets | population 200, probe 100, localise 80 per specification |
| Sampling | temperature 0.7, Tell variants 0–2, seeds 1–3 |

**Other rules the code applies.**

- Weak: the members of an ambiguity reported at the item are excluded from its weak check.
- An S2-only ambiguity takes the two most frequent classes among the acted readings of competent models, ties by class key, and is reported only when its members differ on some field.
- T-boundary selection: only physical scenarios (grn_count ≤ received ≤ ordered + 1, 0 ≤ discount ≤ 100); canon points and repeats are skipped; the literal order (source, chunk, sentence, offset) decides which 12 are kept; dates, times and fractions are not literals.
- Harness: text calls are parsed only when the reply has no native call, and several run in order; a model with zero calls has valid-call rate 0; there is no nudge on the last round.
- All attempts of one DirectedProbe candidate are charged, then dispatched as one batch; the witness is the first in submission order (planning resolution 11).
- The Rasch Newton iteration uses step halving.

**Execution and reproduction.** A report made with N workers is byte-identical to the one-worker report over the same cached runs; fresh runs with `OLLAMA_NUM_PARALLEL` above 1 are not guaranteed to be bit-identical, because the model server batches inference.

## 4. Evaluation

- **Mutants:** the same 20 development mutants as P1 (`dev_per_type` 5, `rng_seed` 20261006). The test split is not run.
- **Truth targets (`observed_at`):** they are benchmark metadata. They are used only to decide scorability from the base runs, and are never an input to any diagnosis method.

  | Operator | Targets |
  |---|---|
  | C1 | the row the conflicting invariants select |
  | C2 | the golden clause's row |
  | H1 | the golden row |
  | H2 | any event with the demanded event's reference (no admissible run writes the demanded event itself) |
  | A1, A2, A3 | the steps the edited rule governs: the discount rules → draft_quote and P2; the GRN-exception rule → record_goods_receipt, reconcile and draft_payment; the gate count → draft_quote, P2, draft_purchase_order, P6 and draft_payment; the volume discount → draft_quote |
  | W1 | the golden case rows of the blanked field's table |
  | W2 | the event kind and reference |

- **Scorable mutant:** at least k = 3 base behaviours touch a target. A behaviour touches a row or event when it acted at an item whose slice covers it and its end state holds it, and a step when it acted at that step. Computed once from the base runs, before any mutant runs, and shared by SpecDx, B2 and B3.
- **Tables:** the primary table scores the scorable mutants and decides the verdict. The secondary table scores all 20 and reports the number excluded per type. The literal typed-against-typed table is kept.
- **Rule:** contradiction and hidden win when SpecDx's typed F1 > max(B2, B3) untyped F1; ambiguity and weak win when SpecDx's typed F1 ≥ 0.4; F1 in exact arithmetic. A type with fewer than 3 scorable mutants is a loss. The gate passes on at least 3 wins of 4.
- **θ ordering before the gate (Task 18).** After the smoke run on the base specification (budget 60), the population model with the highest κ must have a θ strictly above the lowest θ of the other models. If it does not, the gate is not launched, and the result goes to the author.
- **Outcome:** PASSED, NOT PASSED or INCOMPLETE. After a second NOT PASSED, P1b stops; a third iteration needs the author's decision.
- **GO** for the gate is given only by three things together: the replay of §1, the screen's population of §2, and the θ ordering above.
