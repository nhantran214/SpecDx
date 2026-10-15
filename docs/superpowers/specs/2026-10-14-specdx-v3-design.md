# SpecDx v3.1: anchored cell-level diagnosis (design delta)

- **Status:** approved in brainstorming on 2026-10-14, section by section.
  - This is a **delta** on `2026-10-11-specdx-v2-design.md` (v2), which is itself a delta on `2026-10-06-specdx-design.md` (v1).
  - Everything in v1 (with its §11 clarifications) and in v2 stays binding unless this document replaces it.
  - "v1 §x" and "v2 §x" refer to those documents.
- **Revision v3.1 (2026-10-14).** Amended after two simulation spikes, before any implementation (§13 items 8–14; evidence in `results/p2-spikes.md`):
  - flat-pooled errors with κ = 10;
  - difficulty grouped by value column and by clause;
  - multi-start EM;
  - no Spearman rule;
  - Theorem T2′;
  - revised G0 criteria;
  - a four-model population requirement.
- **Phase:** P2, after P1b stopped before gate v2 (`results/p1b-outcome.md`).
- **Evidence this design answers:**
  - `results/p1b-theta-diagnosis.md`, with its scripts in `experiments/p1b_theta_diagnosis/`;
  - `results/p1b-smoke-base.json`;
  - `results/p1b-screen.md`.

---

## 0. Summary

P1b's smoke run showed two failures, and v3 answers each.

**Failure 1: the stage-2 model of v2 does not identify the level of θ once interpreters write.**
- The error distribution ε was per item, shared by every model, and allowed to put mass on the admissible classes. Along a one-parameter family of (θ, ε), the data fit equally well.
- θ ended up near the share of a model's runs in the plurality class: 0.097 and 0.078 for the screened population.
- Every component gated on θ ≥ 0.5 was therefore inert: S2, the ddmin oracle and the competent-pair ambiguity.

**Failure 2: readings were compared as one joint class key over a whole multi-row slice.**
- Independent variation in several fields multiplied into fragmentation. Only 26.8% of same-item run pairs agreed between models, and 29.6% within one model.
- Surrogate ids that the models invent, and rows inherited from other steps, added noise.

**What v3 does instead.** v3 replaces stage 2 and the class key with an **anchored cell model**.
- **Unit of evidence (§2).** One cell: a (table, row, column) value, or one cross-cutting clause, read at an item. Each cell carries the specification's own declared value d_c as its **anchor**.
- **Latent truth (§3).** Each cell has a latent entailment set A_c, the readings that T admits. A_c is one of:
  - {d_c}: no defect;
  - a set without d_c: a hidden requirement;
  - a two-value set: an ambiguity (reported with a hidden requirement when C picks one of the readings, as v1 §2.4 requires);
  - ∅: T is silent.
- **Readings (§3).** A reading falls in A_c with probability equal to the model's **accuracy** p_{m,c} = σ(α_m − β_{g(c)}), where the difficulty β is shared by a group g(c): the column of a value cell, or the clause of a constraint cell. Otherwise the reading is drawn from an error distribution restricted to values **outside** A_c, and that distribution is pulled towards a flat pool.
  - Because θ is defined as accuracy, error mass can never sit on the truth.
  - Because errors are pulled towards a flat pool, silence (∅) predicts flat readings. A sharp split then identifies a pair, and a sharp mode identifies a value, from a single model (T2′).
  - Under the stated assumptions, the model is identifiable (§4).
- **Typing signatures (§4).** Each defect type has a distinct **accuracy-gradient signature**: how the reading distribution at a cell changes with model accuracy.
  - The mass on a value in A_c rises with accuracy.
  - The mass on an error value falls.
  - When A_c = ∅, no value depends on accuracy at all.
- **Typing (§5).** Typing follows from the posterior over A_c, and the v1 definitions are kept:
  - refutation still separates a hidden requirement from a contradiction of size 1;
  - multi-clause contradictions still go through the hitting sets and refutation of v1 and v2.
- **Evidence (§8).** The evidence has four pillars:
  1. theorems with proofs (§4, Appendix A);
  2. a pre-registered simulation study that needs no GPU;
  3. an offline replay on cached base runs;
  4. gate v3 on the same 20 development mutants.

  A run of GO steps (G0–G3) stands before any new GPU use.

Interpreter-visible settings do not change, so run keys are unchanged (`RUN_SCHEMA_VERSION` stays 3).
- **Schema-3 runs** (the P1b screen and smoke runs) remain valid cache hits for new runs.
- **P1's schema-2 runs** remain usable for offline replay only.

Development is offline.

---

## 1. Intent and decisions (2026-10-14)

| Decision | Choice | Source |
|---|---|---|
| Goal of v3 | Turn the P1b finding into the main contribution: a new evidence model that is identifiable when readings are noisy and fragmented; re-pre-register the gate for it | User (option 2 of 3) |
| Hardware | Design and develop on the local RTX 5070 Ti (16 GB); the official runs use a 48 GB GPU; small and medium models are preferred, so runs stay short | User (option b) |
| Success criteria | A theorem on identifiability, a simulation study, and the benchmark as empirical evidence | User (option c) |
| Evidence model | The specification as a fallible rater at cell level (anchored cell model) | User (approach A of A/B/C) |
| Source of accuracy contrast | Relax the capability screen to admit more small and medium models (the v3 weakened-reader fallback was dropped in v3.1) | User (option a of a/b/c) |

The constraints of v1 §1.2 and v2 §1 stand, with v2 §2.3's larger-GPU profiles used for the official runs. The test split is not touched in P2.

---

## 2. Evidence unit: anchored cells (replaces the class key of v2 §3.3 for stage 2)

### 2.1 Items

Items are unchanged from v2 §3.1: (scenario, interpreter step). Stage 1 (v2 §3.2: the act indicator, the Rasch fit and observability) is unchanged. Only behaviours that **acted** at an item contribute readings at that item's cells.

### 2.2 Cells

A cell is the smallest unit of reading. There are two kinds.

- **Value cell.**
  - A (table, row, column) with the column LOCKED under v1 §2.2's ColumnModel (the blank-column clauses of v1 §2.2 are constraint cells).
  - The reading of behaviour b is the normalised value in b's end state (§2.5), or `absent` when the row is not in b's state.
  - Value cells also cover golden events: one cell per (kind, ref, actor), with readings `present` and `absent`.
- **Constraint cell.**
  - One cross-cutting clause of C evaluated at the item: invariant, expectation, blank column, reconcile, guard, or step completion.
  - The reading is `satisfied` or `violated` on b's end state, as v1's `evaluate` computes it.

### 2.3 Anchors

The anchor d_c is what C declares for the cell.

- **Value cell.** The golden value, lifted to the scenario where v1's lifting applies (`golden:<table>:<key>:<column>`), or `present` for a golden event.
- **Constraint cell.** `satisfied`.
- **Unanchored cells.** A value cell for which C declares nothing is **unanchored**. It is still read. It contributes no evidence to accuracy, and it is the place where weak checks show (§5.4).

### 2.4 Cell scope

A cell is read only at the items whose step may write it. This removes rows that other steps write into a shared slice (P1b diagnosis §2).

- **Value cells.** The item's step must draft the cell's table (`draft_tables` in `de_scope.csv`, planning resolution 2 of P1b).
- **Constraint cells.** The step's tables must intersect the clause's tables (v2's relevance rule).

### 2.5 Normalisation (syntactic only)

Each value is normalised in this order:
1. Unicode NFC.
2. Trimmed.
3. Inner whitespace collapsed to one space.
4. Casefolded.
5. Numerals canonicalised: `35`, `35.0` and `35,0` all become `35`, and dates in v1's canonical form are kept as they are.

No semantic mapping is applied. `draft` and `Nháp` stay distinct readings. Whether one of them is an interpreter error is decided by the model, not by normalisation.

### 2.6 Row matching

Rows are matched by natural key. A key column of the canon is classified as **executor-generated** when none of its golden values occurs anywhere in the Tell corpus (T) of the specification. This test is automatic and uses T only.

Rows of a table with an executor-generated key are matched on the remaining LOCKED foreign-key columns. For example, `soline` rows match on (`so`, `ma_sp`) rather than on `ma_soline`. A row that matches no golden row yields readings only at unanchored cells.

### 2.7 Pooling across scenarios

A lifted cell (a golden clause lifted by v1's lifting, or a cross-cutting clause) is **one cell across every scenario where it applies**. Its readings are pooled over those scenarios, and it has one latent A_c. Cells that do not lift stay per scenario. Pooling raises the number of readings per model and cell, which the finite-sample bound of §4 depends on.

A pooled value cell compares readings in the **lifted vocabulary** (clarified at planning, 2026-10-14):
- the cell's anchor is the clause's placeholder, for example `{qty}`;
- in scenario s, a reading is replaced by the placeholder that `lift_value(reading, column type, s)` returns, when it returns one; otherwise it stays as normalised text.

So "the reading equals this scenario's value of `qty`" is the same reading in every scenario.

---

## 3. The anchored cell model (replaces v2 §3.3)

### 3.1 Latent entailment set

For each cell c, V_c is the set of candidate values, built as follows:
1. d_c, when the cell is anchored;
2. then the most frequent observed values, ties broken by the normalised string, up to K_max = 6 values in all;
3. plus one `OTHER` bucket for every remaining value. `OTHER` is never part of A_c.

The latent A_c is one of:
- a single value {x} ⊆ V_c;
- a two-value set {x, y} ⊆ V_c (|A_c| ≤ 2);
- ∅.

| A_c | Meaning |
|---|---|
| {d_c} | No defect at c |
| {a}, a ≠ d_c | T entails a value C does not declare (hidden requirement, *mis-key* subtype) |
| {d_c, a} | T admits two readings, and C picks one: an ambiguity **and** a hidden requirement at c (v1 §2.4: "when T is ambiguous and C picks one reading, both are reported"; *ambiguous-pick* subtype) |
| {a, a'} without d_c | An ambiguity and a hidden requirement (ambiguous-pick subtype) |
| ∅ | T does not determine c, but C fixes it (hidden requirement, *unstated* subtype). An unanchored cell with A_c = ∅ is simply undetermined and is not a defect |

The prior over A_c is defined as follows. The defect forms are:
- *mis*: a single value other than d_c;
- *amb*: any two-value set;
- *∅*: the empty set.

The prior then assigns:
- **Anchored cells.** P(A_c = {d_c}) = 1 − π. Each form f receives π · s_f, with s_mis + s_amb + s_∅ = 1, uniform over the candidate sets within the form.
- **Unanchored cells.** There is no d_c. A single value has total prior 1 − π, uniform over values. Two-value sets have π · s_amb / (s_amb + s_∅), and ∅ has π · s_∅ / (s_amb + s_∅).
- **Selection.** π and the shares s_f are chosen at G0 (§9).

### 3.2 Reading model

Behaviour b of model m reads cell c as follows.

- **When A_c ≠ ∅:**
  - with probability p_{m,c} = σ(α_m − β_{g(c)}), the reading is uniform on A_c (faithful);
  - otherwise it is drawn from ε_c(· | V_c ∖ A_c), the error distribution conditioned on values outside A_c.
- **When A_c = ∅:** the reading is drawn from ε_c, and no reading is faithful.

The parameters are:
- α_m: model ability;
- β_g: the difficulty of the cell's group g(c). For a value cell the group is its column, pooled over rows and scenarios; for a constraint cell it is its clause. At the planning-time check, the spread of difficulty within a value column was at sampling-noise level, while within a clause kind it was about 1.6, so difficulty is never shared by a whole clause kind;
- ε_c: a full distribution over V_c, used only through its restriction to V_c ∖ A_c.

The resulting marginal for model m at c is:
- P_m(y) = p_{m,c} · 1{y ∈ A_c}/|A_c| + (1 − p_{m,c}) · ε_c(y | V_c ∖ A_c), when A_c ≠ ∅;
- P_m(y) = ε_c(y), when A_c = ∅.

Readings are conditionally independent given A_c and the parameters. A behaviour contributes at most one reading per cell. Readings of different cells within one behaviour are treated as independent given their own A_c; this is the cell-level analogue of v1's independence across items.

### 3.3 Priors

- α_m ~ N(μ_α, σ_α²).
- β_g ~ N(0, τ²). A group may be hard, but an extreme difficulty is penalised.
- ε_c ~ Dirichlet(κ · u_c), where u_c is the **flat** distribution over V_c, and κ = 10 (fixed).
  - Value labels mean nothing across columns, so a flat profile is the only pool beyond the column.
  - The strong pull stops ∅ from fitting an arbitrary sharp split or mode. Spike 1 showed that a free per-cell ε_c makes silence absorb almost every pair (§13 item 8).
  - The flat pool is a regularisation, not a claim that errors are flat: real errors concentrate, and the simulation's A3 arm measures the effect.
- μ_α, σ_α and τ are chosen at G0 (§9). τ's grid includes 1 and 2.

### 3.4 Fitting

MAP by EM, deterministic.

- **Initialisation (multi-start).**
  - EM runs from T3's median start and from a common α ∈ {0, 1, 2, 3}, with every β_g = 0 and ε_c = u_c. It keeps the fit with the highest log posterior.
  - Why: on the base data, more than half of the anchored cells are read off their anchor by every model (case `apr` rows that nobody writes, golden event actors that nobody writes). That is past T3's breakdown point. From the median start alone, EM reached a θ-collapsed fixed point about 200 nats below the best one.
- **E-step.**
  - For each cell, compute the posterior over all candidate sets: 1 + 6 + 15 = 22 when |V_c ∖ OTHER| = 6, including ∅.
  - The posterior is proportional to prior(A) · Π_b P_{m(b)}(y_{b,c} | A).
  - Then compute each reading's faithful probability r_{b,c} = Σ_A post(A) · P(faithful | y, A).
- **M-step.**
  - α and β are updated by Newton steps on the expected complete-data log posterior. It is a penalised logistic regression with r_{b,c} as fractional successes.
  - ε_c is the Dirichlet MAP of the expected error counts.
- **Convergence.** The fit stops when the largest parameter change is below 1e-6, or after 200 iterations (pre-registered). Non-convergence is reported, never silent.

### 3.5 Outputs

- **Per cell:** the posterior over A_c, the MAP set, and the posterior mass of each form (none, mis-key, unstated, ambiguity).
- **Per model:** α_m and θ_m = σ(α_m), the accuracy at a cell of median difficulty. The family effect is reported as in v2 §3.3.
- **Per behaviour:** the weight w_b, the mean of r_{b,c} over the cells it read. It replaces v2's run weight in the hitting sets (§5.2), the refuter order, and B2's choice of best interpreter.
- **No hard competence threshold.** No component is switched on or off by a threshold on θ; all posterior quantities are used as weights.

---

## 4. Theory (replaces the stage-2 claims of v2 §3.9)

### 4.1 Assumptions (per cell; c is suppressed)

- **(A1) Independence.** Given A and the parameters, the readings of a model's behaviours are independent. Across Tell variants they are not identically distributed: P_m is then the average distribution over variants.
- **(A2) Accuracy parametrisation.** When A ≠ ∅, P_m(A) = p_m, and faithful readings are uniform on A, with |A| ≤ 2.
- **(A3) Shared, pooled errors.** The error distribution ε (restricted to V ∖ A) is the same for every model at the cell, and it is drawn around a known pool u, with ε ~ Dirichlet(κ u) and u flat in v3.1. T2′ uses the limit ε = u.
- **(A4) Monotone accuracy.** p_{m,c} = σ(α_m − β_{g(c)}). The order of models by accuracy is therefore the same at every cell.
- **(A5) Silence.** When A = ∅, every model reads from ε: competence does not help where T says nothing.

### 4.2 Accuracy-gradient signatures

Take two models with p_1 > p_2, write Δ = p_1 − p_2, and define D(y) = P_1(y) − P_2(y). Then:

| Form of A | D(y) for y ∈ A | D(y) for y ∉ A |
|---|---|---|
| {x} | Δ > 0 | −Δ·ε(y) ≤ 0 |
| {x, a} | Δ/2 > 0 | −Δ·ε(y) ≤ 0 |
| ∅ | — | 0 for every y |

An error value concentrated on a (that is, ε(a) large, a ∉ A) has D(a) < 0. An ambiguous value a ∈ A has D(a) > 0. This sign difference is what separates ambiguity from a concentrated error. A two-model contrast reaches only the population-level signatures; the finite-sample power depends on Δ and n (Theorem T4).

### 4.3 Theorems (proofs in Appendix A)

- **T1 (single value, one model).** Let A = {x}. Suppose p_m > (1 − p_m) · max_y ε(y), the plurality condition (PC_m). Then x is the unique mode of P_m, and p_m = P_m(x).
- **T2 (all four forms, two models).** Under (A1)–(A5), with p_1 > p_2, A = {y : D(y) > 0}. In particular A = ∅ exactly when D ≡ 0. The forms no defect, mis-key, ambiguity and silence are therefore identified from two models with distinct accuracy. Under (A4), Δ_c > 0 at every cell whenever α_1 ≠ α_2.
- **T3 (model contrasts from anchors, breakdown ½).**
  - At an anchored cell with A_c = {d_c}, ℓ_c := logit P_{1,c}(d_c) − logit P_{2,c}(d_c) = α_1 − α_2.
  - If more than half of the anchored cells are non-defective, then median_c ℓ_c = α_1 − α_2.
  - So the model order and the contrasts are identified, even where PC fails at individual cells.
  - The levels α_m are identified given a location constraint on β: the median of β_{g(c)} over the non-defective anchored cells is 0.
- **T4 (finite samples).** Let n be the smaller number of readings per model at the cell, and |V| the number of candidate values.
  - **(a) The two-model contrast rule.** Â = {y : D̂(y) > Δ/4} errs with probability at most 4|V| exp(−n Δ²/32).
  - **(b) The plug-in mode under PC.** With margin γ = p − (1 − p) · max ε, the mode errs with probability at most 2|V| exp(−n γ²/2).
  - **(c) The minimum-distance rule under T2′.** Let γ_min be the smallest sup-norm distance between the predicted distributions of two candidate sets. The rule that picks the candidate whose predicted distribution is closest to P̂ errs with probability at most 2|V| exp(−n γ_min²/2).
- **T2′ (all four forms, one model, known error pool).** Let ε = u be known and flat over V, with K = |V|, and let p be the model's accuracy at the cell.
  - If A ≠ ∅, then P_m equals p/|A| on A and the floor (1 − p)/(K − |A|) off A.
  - If A = ∅, then P_m is flat at 1/K.
  - Hence A is identified from P_m alone whenever p ≠ |A|/K, that is, whenever the faithful share differs from what chance gives. This needs no second model.
  - With p known (from α and β_g), the forms are separated by γ_min > 0 under the same condition.

### 4.4 Scope of the theory

- **The bounds are conservative.** They come from Hoeffding's inequality. The MAP estimator with priors is what v3 runs, and the simulation (§8.1) measures its actual power.
- **Single-model typing.** Mis-key hidden requirements and weak checks rest on T1. They need no accuracy contrast, only a plurality margin.
- **Ambiguity and the unstated subtype.** In v3.1 these rest on T2′, through the error pool, and on T2, through accuracy contrasts, jointly.
- **The advantage is regime-dependent** (spike 2, §13 item 12). v3.1 beats the simple heuristics on ambiguity only at high cell accuracy (≥ 0.9) with four models. At accuracy 0.6–0.8, or with two models, majority vote does as well or better. Hidden requirements mostly tie with anchor-only. The calibrated population sits in the winning regime, and the paper reports the whole regime map.
- **(A3) may fail.** Models can make different errors, and real errors concentrate: in the calibration, the most frequent wrong value holds 0.84 of the error mass. The flat pool is a regularisation, and the simulation's A3 arm is part of G0 (§8.2).
- **(A5) may fail.** A strong model may guess an unstated value from domain knowledge. The effect is only to move a finding from the *unstated* to the *mis-key* subtype. Both are hidden requirements.

---

## 5. Typing (replaces v2 §3.5–§3.6)

The v1 §2.4 definitions are kept, and so are the matching rules of v1 §4.6.

### 5.1 Hidden requirement and contradiction of size 1

- **Candidate.** An anchored cell c with posterior P(A_c ≠ {d_c}) ≥ τ_H, the total posterior mass of every defect form.
- **Refutation.** The candidate goes to refutation (v1 §3.6, v2 refutation: symbolic repair, then directed probe).
  - If a state satisfies c together with the other clauses, the finding is a **hidden requirement** at clause c. Its subtype follows the MAP form:
    - mis-key: {a};
    - ambiguous-pick: a two-value set, reported together with the ambiguity of §5.3;
    - unstated: ∅.
  - If no state does, the finding is a **contradiction** of size 1 (v1 §3.3).

### 5.2 Contradictions of two or more clauses

These follow v1 §3.3 and v2 §3.4: hitting sets over the acted violation sets of constraint cells, then refutation (Proposition 3, unchanged). The behaviour weight is w_b (§3.5) instead of v2's θ-based weight. The P1b smoke run showed this component working independently of θ.

### 5.3 Ambiguity

- **Report.** A cell with P(|A_c| = 2) ≥ τ_A is reported as an **ambiguity**.
- **ddmin oracle.** The ambiguity is localised in T by ddmin over the sentences of T (v1 §3.7, v2 §3.6). Its oracle is redefined as follows.
  - A ddmin test re-runs the population models that have at least one acted behaviour in the specification, for each Tell variant, at seed 1. This is v2 §3.6's oracle without the θ threshold.
  - The test passes while P(|A_c| = 2), computed on the re-run readings with α, β and ε held at their fitted values, stays ≥ τ_A.
- **Budget.** The localisation budget is unchanged (80).
- **Co-occurrence.** Ambiguity and hidden requirement co-occur as in v1 §2.4.
- **No rank-correlation rule.** v3's Spearman robustness rule is removed: with about 6 readings per model and cell it passed at about 0% of true pairs (spike 1). Robustness to (A3) is instead required of the whole method, through G0's A3 arm.
- **Population.** Ambiguity requires at least four acting models (§7, G2).

### 5.4 Weak check

A cell c is a candidate when two conditions hold:
- its MAP set is {x} with posterior ≥ τ_W;
- some acted behaviour read y ≠ x at c and still **satisfies every clause of C** relevant at the item. This is checked symbolically on that behaviour's state.

Each candidate is reported as a weak check at the cell's column (for an event cell, its (kind, ref)), per v1 §4.6. Unanchored cells are the main source of weak checks: mutants W1 and W2 delete the anchor.

### 5.5 Differential and reporting

The differential against the base (v1 §11 item 13) and the ordering of base locations are unchanged. Every report carries:
- the per-cell posteriors;
- θ_m and the family effect;
- the T literals (v2 §3.7);
- κ, δ and Oₛ from stage 1.

---

## 6. What stays and what goes

| Component | v3 |
|---|---|
| Neutral harness, nudges, `RUN_SCHEMA_VERSION` 3 | **Kept unchanged**. Cached schema-3 runs stay valid cache hits |
| Stage 1 (Rasch act model), observability ω = 0.3, m_min = 3 | Kept |
| T-guided scenarios, bandit loop, budgets 200/100/80, escalation counting acted views only | Kept |
| Refutation (symbolic repair, directed probe), Proposition 3 | Kept |
| ddmin over T | Kept; new oracle (§5.3) |
| Frozen mutant packs, scorability (k = 3), primary and secondary tables | Kept |
| Parallel workers, `doctor`, `cache-info`, profiles | Kept; the official runs use `gpu48` |
| Joint class key over a slice (v2 §3.3) | **Replaced** by anchored cells (§2) |
| Stage 2: Beta(7, 3) θ with item-shared ε (v2 §3.3) | **Replaced** by the anchored cell model (§3). The v2 model is kept in code as a simulation baseline and an ablation |
| S1/S2 ambiguity signals, permutation test, BH (v2 §3.6) | **Removed**. Ambiguity comes from P(|A_c| = 2). The P1b diagnosis showed that paraphrase effects are swamped by seed noise |
| Every "θ ≥ 0.5 is competent" gate | **Removed**. Weights come from the posterior |

---

## 7. Population (amends v2 §2.2)

- **Admission.** The act-rate threshold drops from 0.6 to **0.3**. The other rules are unchanged:
  - valid-call rate ≥ 0.8;
  - at most 2 models per family and 4 in all;
  - at least 2 families;
  - same scenario, variants and seeds.
- **Why.** More raters raise the readings per cell, and spike 2 shows that ambiguity needs four acting models: with two, it loses to majority vote at 13 of 15 regime points. T2 also uses distinct accuracies.
- **Four-model requirement (v3.1).** The admitted population must hold at least 4 acting models across at least 2 families; otherwise STOP (G2).
- **Applying it.** The threshold is applied to the cached P1b screen (`results/p1b-screen.json`); no new screen runs are needed. The population is chosen by the unchanged ranking:

| Family | Model | Act rate | Valid-call rate |
|---|---|---:|---:|
| qwen | qwen3:30b-a3b | 0.802 | 0.952 |
| qwen | qwen2.5:14b | 0.580 | 0.943 |
| openai | gpt-oss:20b | 0.914 | 0.988 |
| mistral | mistral-small3.2:24b | 0.346 | 0.914 |

  Within the qwen family, qwen3:14b (0.407) is ranked out by the per-family cap.
- **Disclosure.** The new threshold was chosen during design, knowing which cached screen results it admits. The screen is outcome-free: it reads no mutant, truth or diagnosis. The change is recorded in the pre-registration before any v3 result.
- **No weakened reader.** v3's weakened-reader fallback is dropped in v3.1. T2′ identifies ambiguity without an accuracy contrast, and spike 2 shows the remaining limit is the number of models, not their contrast.

---

## 8. Evaluation (amends v2 §4)

### 8.1 Simulation study (pre-registered; no GPU)

- **Generator.** Synthetic specifications. Each has:
  - C cells of both kinds (value and constraint), anchored and unanchored, with |V| drawn from the observed distribution;
  - M ∈ {2, 3, 4} models with abilities α_m;
  - group difficulties β_g ~ N(0, τ_true²), with a within-group arm in which each cell's difficulty spreads around its group's by τ_within = 0.5;
  - error distributions drawn from a Dirichlet, including concentrated errors;
  - planted defects of every form at controlled rates;
  - per model, readings from 3 variants × 3 seeds, pooled over S scenarios, with act rates per model.
  - **Misspecification arms:** model-specific error distributions (A3 violated) and model-dependent silence (A5 violated).
- **Calibration.** Generator parameters are calibrated to the P1b base smoke data and the cached screen runs. The calibrated quantities are:
  - per-cell accuracy;
  - dispersion;
  - error concentration;
  - act rates;
  - readings per cell.

  Calibration uses summary statistics of base runs only, never a mutant.
- **Metrics:**
  - accuracy recovery, |θ̂_m − θ_m|;
  - typed F1 for hidden requirement, ambiguity and weak check (contradiction is symbolic and is not simulated);
  - calibration of the posterior defect probabilities;
  - runs needed to reach a given F1.
- **Baselines:**
  - v2's stage 2 with joint classes;
  - majority vote over models;
  - "deviates from the anchor ⇒ defect" (anchor-only).
- **Sweeps:**
  - Δ (the accuracy gap);
  - M;
  - n (readings per model per cell);
  - τ_true;
  - defect rate;
  - error concentration;
  - the misspecification arms;
  - **the regime map:** mean cell accuracy ∈ {0.6, 0.7, 0.8, 0.9, 0.95} × accuracy spread Δ ∈ {0, 0.1, 0.2} × M ∈ {2, 4}, giving win, tie or loss against the best baseline per type, with a margin of 0.05.

  The phase diagrams are compared with the boundaries that T2, T2′ and T4 predict.
- **Replicates.** At least 10 replicates per arm at the calibrated point, and 5 per point of the regime map. Spike 1 showed that 3 replicates are too noisy for G0's thresholds.

### 8.2 GO steps

| Step | What | GPU | Pass criterion (pre-registered here) |
|---|---|---|---|
| **G0** simulation | Fit at the calibrated parameters and the planned budget, with at least 10 replicates; choose the free parameters of §9 by the selection rule | No | (i) mean \|θ̂ − θ\| ≤ 0.05. (ii) Simulated typed F1 ≥ 0.6 for hidden requirement and ≥ 0.4 for ambiguity. Simulated weak-check F1 is reported against the gate's 0.4 bar **for information only**; it does not gate (§13 item 15). (iii) Regime-based: at the calibrated point, under the A3 and A5 arms, and at the population's own M and Δ, v3.1 is never below the best baseline by more than 0.05 on any type; and at the calibrated point it beats the best baseline on ambiguity by more than 0.05. If G0 fails, stop and report |
| **G1** base replay | Fit v3.1 with the parameters frozen at G0 on the cached base runs: the P1b smoke run and the P1 base runs | No | (i) θ_m ≥ 0.5 for gpt-oss:20b and qwen3:30b-a3b (no collapse). (ii) The real defect R-SO (v1 §4.4: `SO-0031` stays `Chờ duyệt` after `APR-0102` is approved) is reported as a hidden requirement, judged **on the P1b smoke run only** (author ruling, 2026-10-14: P1's two acted readings at that cell equal the anchor). If no acted behaviour reached the cell, G1 records "not observable" for (ii) and does not fail on it. On failure, stop and report; parameters are not re-tuned |
| Freeze | The v3 pre-registration (§9) is committed with a test that pins every value | No | — |
| **G2** population | Apply §7 to the cached screen and freeze the population into the gate configuration | No | At least 2 families and at least 4 acting models admitted, otherwise STOP |
| **G3** smoke | The base specification on the 48 GB host, population budget 60, with the checks of P1b Task 18 Step 5 | Hours | Every P1b smoke check passes, and θ_m ≥ 0.5 for each population model |
| **Gate v3** | The 20 development mutants | Days | §8.3 |

### 8.3 Gate v3

- **Rule.** The v2 §4.2 rule is unchanged, so iterations stay comparable:
  - contradiction and hidden requirement win when SpecDx's typed F1 exceeds max(B2, B3) untyped F1;
  - ambiguity and weak check win when SpecDx's typed F1 is ≥ 0.4;
  - a type with fewer than 3 scorable mutants is a loss;
  - the gate passes on 3 or more wins of 4.
- **Exact arithmetic.** Scoring uses v2's exact-arithmetic F1.
- **Baselines.** B2 uses the interpreter with the highest θ_m (§3.5). B3 is unchanged.
- **Second iteration on the development split.** This is the second gate on the same 20 development mutants. P1's gate ran on them, and P1b stopped before its gate. v2 §4.2 makes another iteration the author's decision, and the author took it on 2026-10-14. Integrity rules:
  - every threshold and prior is chosen by simulation and base data only (G0, G1);
  - the development mutants are used once, at the gate;
  - the test split stays untouched.

### 8.4 Deferred to P3

These are deferred:
- the hypotheses of v1 §1.4 on the test split;
- ablations: anchor removed (back to latent truth), the joint class key instead of cells, a free per-cell ε instead of the flat pool, per-cell difficulty instead of grouped difficulty, the v2 stage 2;
- the remaining real defects R-12 and R-DE;
- S2 (τ²-bench).

---

## 9. Pre-registration v3

The v3 pre-registration is a document (`experiments/p2-preregistration.md`) plus a test that pins every value. It is committed after G1 and before G2.

| Group | Values |
|---|---|
| Unchanged from v1 and v2 | temperature 0.7; Tell variants 0–2; seeds 1–3; budgets 200/100/80; scorability k = 3; K_max = 6; the gate rule; dev split (`dev_per_type` 5, `rng_seed` 20261006); harness nudges 2; ω = 0.3, m_min = 3; Rasch λ = 1.0; T-boundary scenarios ≤ 12; workers ≤ `OLLAMA_NUM_PARALLEL` |
| New, fixed by this spec | \|A_c\| ≤ 2; the candidate rule of §3.1; the normalisation of §2.5; the row-matching rule of §2.6; the cell scope of §2.4; pooling of lifted cells (§2.7); EM tolerance 1e-6 and at most 200 iterations; the screen's act-rate threshold 0.3; κ = 10 with a flat pool u_c; difficulty grouped by value column and by clause; multi-start EM (T3's start plus a common α ∈ {0, 1, 2, 3}); G0 with at least 10 replicates, and τ ∈ {1, 2} in the grid; at least 4 acting models; the G0–G3 criteria of §8.2 |
| Chosen at G0 by the selection rule, then frozen | π and the form shares (s_mis, s_amb, s_∅); τ; μ_α, σ_α; τ_H, τ_A, τ_W |
| Selection rule (fixed now) | A grid over the free parameters, written in the simulation module before G0 runs. The objective is the mean simulated typed F1 over hidden requirement, ambiguity and weak check, at the calibrated parameters and under the misspecification arms equally weighted. Ties go to the more conservative setting (higher P(\|A\| = 1), smaller π) |

---

## 10. Hardware and runtime

- **Development, simulation, G0 and G1** run on the local machine, with no GPU needed.
- **G3 and the gate** run on a 48 GB GPU through `SPECDX_OLLAMA_URL` over an SSH tunnel.
  - The Ollama version stays 0.30.4.
  - The model digests must equal the cached runs' digests, as `doctor` checks.
  - Server settings: `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_KEEP_ALIVE=-1`, with `--workers 2`.
- **Memory.** The four population models need about 60 GB to stay resident together. On 48 GB they swap; v2 §5.1 groups dispatch by model, which limits the swaps. An 80 GB GPU removes the swaps.
- **Runtime estimate:** 3–8 days for the gate. It is measured at G3 and reported to the author before launching. The gate is resumable.

---

## 11. Implementation notes (amends v2 §6)

### Modules

| Module | Responsibility |
|---|---|
| `specdx/infer/cells.py` | Cell extraction (value and constraint), anchors, cell scope, normalisation, row matching, pooling |
| `specdx/infer/anchored.py` | The anchored cell model: priors, EM, outputs |
| `specdx/infer/typing.py` | v3 typing (§5). Hitting sets and refutation are reused unchanged; it replaces the typing parts of `scores.py` |
| `specdx/sim/` | Generator, fit, metrics, sweeps, and the `specdx simulate` command |
| `experiments/p2_replay.py` | G1, read-only over the run cache, following `experiments/p1b_replay.py` |
| `specdx/eval/screen.py` | Act-rate threshold 0.3 |

`specdx/active/loop.py` uses the v3 diagnosis, the v3 ddmin oracle and w_b. B2 chooses the interpreter by θ_m. `sensitivity.py` and v2's stage 2 stay in the tree, only as simulation baselines.

### Tests

- Anchored-model recovery on synthetic data at known parameters, including all four forms.
- A regression on the P1b smoke data: θ does not collapse.
- Normalisation and row matching on the real base pack, including the `soline` surrogate rule.
- Cell scope excludes inherited rows.
- Typing: refutation separates hidden requirement from contradiction of size 1.
- T2′: a single model separates the forms when the error pool is known.
- Weak checks on unanchored cells.
- Determinism.
- The pre-registration test.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| (A3) fails: errors concentrate, or are model-specific, and mimic ambiguity | High P(\|A\| = 1) prior; the A3 arm in G0's selection objective and in criterion (iii) |
| The advantage is regime-dependent (high accuracy, four models) | G2 requires 4 acting models; G3 checks θ; G0 (iii) is judged at the population's own M and Δ; the paper reports the full regime map, including where heuristics suffice |
| Hard cells mistaken for hidden requirements | The β prior (§3.3); refutation (§5.1) |
| The row-matching rule misclassifies a key | Unit tests on the base pack; G1 inspection of unmatched rows |
| Defects are not sparse on some specification | T3's breakdown point is ½. The base has few findings (14 in the P1b smoke run, most of them noise under v2) |
| Inherited rows still leak | Cell scope (§2.4); a test |
| Swaps on 48 GB lengthen the gate | Measured at G3; an 80 GB GPU removes them |
| The second iteration on the dev split invites tuning | §8.3 integrity rules; G0 and G1 parameters frozen before the population is applied; the development mutants used once |

---

## 13. Decisions log (2026-10-14, brainstorming)

1. **Goal.** The new evidence model is the main contribution (option 2 of: minimal repair / new model / broader reframe).
2. **Hardware.** Develop locally; official runs on 48 GB; small and medium models preferred.
3. **Success criteria.** Theorem, simulation and benchmark (option c).
4. **Evidence model.** Anchored cell model, the specification as a fallible rater (approach A; B and C were rejected).
   - B: three views from seeds cannot separate two models without an anchor.
   - C: interventions multiply runs.
5. **Typing names.** Aligned with v1 §2.4. Mis-key and unstated are both hidden requirements; a contradiction is internal to C and stays symbolic.
6. **Accuracy contrast.** Relaxed screen (act ≥ 0.3). The weakened-reader fallback was dropped in v3.1 (item 11).
7. **Gate.** The v2 rule is kept. This is the second iteration on the dev split, by the author's decision.

**v3.1 amendments.** All of these were decided on 2026-10-14, after the P2a plan's pre-check and two simulation spikes. No mutant was looked at, and nothing was pre-registered yet. Evidence: `results/p2-spikes.md`.

8. **Spike 1** (per-column difficulty) failed. Ambiguity typed F1 stayed near 0 under every difficulty model. The real competitor of a pair is ∅ with a free per-cell ε_c, which fits any split; the Spearman rule passed at about 0% of true pairs.
9. **Errors.** ε_c ~ Dirichlet(κ u_c), with a flat pool and κ = 10, the best of κ ∈ {10, 50, fixed} in spike 2.
10. **Difficulty** is grouped by value column and by clause (within-column spread ≈ 0; within clause kind ≈ 1.6). EM is multi-start (the plan's pre-check finding).
11. **The Spearman rule and the weakened reader are dropped.** T2′ is added.
12. **Spike 2 at the calibrated point** (10 replicates; hidden / ambiguity / weak F1):
    - v3.1: 0.83 / 0.81 / 0.45;
    - majority vote: 0.73 / 0.69 / 0.40;
    - anchor-only: 0.78 / 0 / 0;
    - v2's stage 2: 0.45 / 0 / 0.11.
    The mean θ error is 0.016. The regime map shows the advantage only at high accuracy with four models.
13. **G0 criteria revised** (author approval, 2026-10-14):
    - weak check ≥ 0.4, the gate's bar, instead of 0.6, which no method reached;
    - criterion (iii) is regime-based, and includes a required win on ambiguity;
    - at least 10 replicates.
14. **Population and G1.** G2 requires at least 4 acting models. G1 (ii) is judged on the P1b smoke run only (author ruling).
15. **Weak check in G0 is information only** (author ruling, 2026-10-15; option B of A/B/C/D). After the plan's pre-flight rulings, a full G0 preview failed on weak alone. The sensitivity previews (`results/p2-spikes/amended-preview/`) showed that simulated weak-check F1 depends on two things outside the method:
    - **A calibration choice the base data do not determine.** The generator cannot represent value columns that are anchored in some scenarios only. Weak F1 was 0.40 per cell, and 0.24–0.34 under three per-column rules.
    - **Seed noise.** Weak F1 moves by about ±0.08 between seed sets.

    G0's weak-check bar therefore measures generator assumptions rather than the method, and it is reported but does not gate. Two things still hold:
    - The gate's own weak-check rule (typed F1 ≥ 0.4 on the development mutants, §8.3) is unchanged.
    - Criterion (iii) still includes weak checks: v3.1 may not fall more than 0.05 below the best baseline on any type.

    Every G0-altering change made after the first preview is disclosed in the pre-registration:
    - F1 (co-occurrence);
    - F12(a,c) (calibration);
    - the exact EM for the truncated error model;
    - suppression of weak checks on cells that report a hidden requirement;
    - the baselines sharing v3.1's cell-level post-processing.

---

## 14. Delivery (amends v2 §9)

- **P2a, no GPU:**
  - cells, the anchored model, typing and the simulation;
  - G0 and G1;
  - the frozen pre-registration.
- **P2b, 48 GB GPU:**
  - G2, G3, then gate v3.
  - P2 is done when gate v3 is PASSED, or when a NOT PASSED is reported, or when a GO step stops it with a report.

---

## Appendix A. Proofs

Notation per cell: P_m(y) is model m's reading distribution, and D(y) = P_1(y) − P_2(y), with p_1 > p_2 and Δ = p_1 − p_2.

### Lemma (signatures)

Under (A2), (A3) and (A5):
- if A ≠ ∅, then D(y) = Δ · (1{y ∈ A}/|A| − ε̃(y)), where ε̃ = ε(· | V ∖ A) vanishes on A;
- if A = ∅, then D ≡ 0.

*Proof.* For A ≠ ∅, P_m(y) = p_m 1{y ∈ A}/|A| + (1 − p_m) ε̃(y). Subtracting the two models gives (p_1 − p_2) 1{y ∈ A}/|A| − (p_1 − p_2) ε̃(y). For A = ∅, (A5) gives P_1 = P_2 = ε. ∎

### T1

Let A = {x}. Then P_m(x) = p_m, because ε̃(x) = 0, and P_m(y) = (1 − p_m) ε̃(y) for y ≠ x. PC_m states that p_m > (1 − p_m) max_y ε̃(y), so x is the unique maximiser of P_m, and p_m = max_y P_m(y). ∎

### T2

By the lemma:
- if A ≠ ∅, then D(y) = Δ/|A| > 0 for y ∈ A, and D(y) = −Δ ε̃(y) ≤ 0 for y ∉ A, so {y : D(y) > 0} = A;
- if A = ∅, then D ≡ 0, so {y : D(y) > 0} = ∅ = A.

Under (A4), p_{1,c} − p_{2,c} = σ(α_1 − β_{g(c)}) − σ(α_2 − β_{g(c)}), which has the sign of α_1 − α_2 at every finite β. ∎

### T2′

Let ε = u be flat over V, with K = |V|.
- **A ≠ ∅.** Then ε̃ = u(· | V ∖ A) equals 1/(K − |A|) off A. So P_m(y) = p/|A| for y ∈ A, and P_m(y) = (1 − p)/(K − |A|) for y ∉ A.
- **A = ∅.** Then P_m(y) = 1/K for every y.

Suppose two candidate sets A ≠ A′ gave the same P_m. If both are non-empty, P_m takes the value p/|A| exactly on A and a single other value off A. The two values differ when p/|A| ≠ (1 − p)/(K − |A|), which is equivalent to p ≠ |A|/K. Then A is recoverable as the level set of P_m that does not equal the floor, so A = A′. If one of them is ∅, P_m is flat, and a non-empty A yields a flat P_m only when p = |A|/K. Hence A is identified whenever p ≠ |A|/K. ∎

### T3

At an anchored cell with A_c = {d_c}, P_{m,c}(d_c) = p_{m,c} = σ(α_m − β_{g(c)}) by (A2). Hence logit P_{1,c}(d_c) − logit P_{2,c}(d_c) = α_1 − α_2: the same value v at every non-defective anchored cell.

If more than half of the anchored cells are non-defective, more than half of the values ℓ_c equal v, so the median equals v. The order of the models follows from the sign of v. Given the contrasts, the median of β_{g(c)} over the non-defective anchored cells being 0 fixes the levels:
- for any model m, the median over those cells of logit P_{m,c}(d_c) is α_m − median β_{g(c)} = α_m. ∎

### T4

**(a) The two-model contrast rule.** Let P̂_m be the empirical distribution of n independent readings (A1). By Hoeffding's inequality, for each y and m:

P(|P̂_m(y) − P_m(y)| ≥ Δ/8) ≤ 2 exp(−2n(Δ/8)²) = 2 exp(−nΔ²/32).

By a union bound over y ∈ V and m ∈ {1, 2}, with probability at least 1 − 4|V| exp(−nΔ²/32), every |D̂(y) − D(y)| < Δ/4. On that event:
- for y ∈ A, D(y) ≥ Δ/2 (since |A| ≤ 2), so D̂(y) > Δ/4;
- for y ∉ A, D(y) ≤ 0, so D̂(y) < Δ/4.

Hence Â = A, and this includes A = ∅, where D ≡ 0.

**(c) The minimum-distance rule.** If every |P̂(y) − P(y)| < γ_min/2, then P̂ is closer in sup norm to the true candidate's predicted distribution than to any other's. That holds because any two candidates' predictions differ by at least γ_min at some y. Hoeffding's inequality and a union bound over V give the failure probability 2|V| exp(−n γ_min²/2).

**(b) The plug-in mode.** If every |P̂(y) − P(y)| < γ/2, the mode of P̂ equals the mode x of P, because P(x) − P(y) ≥ γ for every y ≠ x. Hoeffding's inequality and a union bound over V give the failure probability 2|V| exp(−2n(γ/2)²) = 2|V| exp(−nγ²/2). ∎

*Remark.* Readings across Tell variants are independent but not identically distributed. Hoeffding's inequality applies to their average with P_m taken as the variant-averaged distribution, as (A1) states. Pooling lifted cells across scenarios (§2.7) adds readings to n, but only if (A2)–(A4) hold for the pooled cell. A cell's difficulty may then differ across scenarios; the simulation measures the effect of pooling such cells.
