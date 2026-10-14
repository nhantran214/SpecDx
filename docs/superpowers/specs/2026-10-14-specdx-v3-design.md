# SpecDx v3: anchored cell-level diagnosis (design delta)

- **Status:** approved in brainstorming on 2026-10-14, section by section.
  - This is a **delta** on `2026-10-11-specdx-v2-design.md` (v2), which is itself a delta on `2026-10-06-specdx-design.md` (v1).
  - Everything in v1 (with its §11 clarifications) and in v2 stays binding unless this document replaces it.
  - "v1 §x" and "v2 §x" refer to those documents.
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
- **Readings (§3).** A reading falls in A_c with probability equal to the model's **accuracy** p_{m,c} = σ(α_m − β_c). Otherwise it is drawn from an error distribution restricted to values **outside** A_c.
  - Because θ is defined as accuracy, error mass can never sit on the truth.
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
| Source of accuracy contrast | Relax the capability screen to admit more small and medium models; a controlled "weakened reader" is the written fallback | User (option a of a/b/c) |

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
  - with probability p_{m,c} = σ(α_m − β_c), the reading is uniform on A_c (faithful);
  - otherwise it is drawn from ε_c(· | V_c ∖ A_c), the error distribution conditioned on values outside A_c.
- **When A_c = ∅:** the reading is drawn from ε_c, and no reading is faithful.

The parameters are:
- α_m: model ability;
- β_c: cell difficulty;
- ε_c: a full distribution over V_c, used only through its restriction to V_c ∖ A_c.

The resulting marginal for model m at c is:
- P_m(y) = p_{m,c} · 1{y ∈ A_c}/|A_c| + (1 − p_{m,c}) · ε_c(y | V_c ∖ A_c), when A_c ≠ ∅;
- P_m(y) = ε_c(y), when A_c = ∅.

Readings are conditionally independent given A_c and the parameters. A behaviour contributes at most one reading per cell. Readings of different cells within one behaviour are treated as independent given their own A_c; this is the cell-level analogue of v1's independence across items.

### 3.3 Priors

- α_m ~ N(μ_α, σ_α²).
- β_c ~ N(0, τ²). A cell may be hard, but an extreme difficulty is penalised. This prior separates "a hard cell" from "a hidden requirement whose anchor is chosen only at noise level".
- ε_c ~ Dirichlet(κ · ε̄_col(c)), where ε̄_col(c) is the pooled empirical distribution of readings in cell c's column (or clause kind) over all cells. Each cell has only a few readings, and this pooling keeps ε_c from overfitting them.
- μ_α, σ_α, τ and κ are chosen at G0 (§9).

### 3.4 Fitting

MAP by EM, deterministic.

- **Initialisation.**
  - α_m is initialised by the median estimator of Theorem T3 over the anchored cells, with β_c = 0.
  - ε_c starts at ε̄_col(c).
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
- **(A3) Shared errors.** The error distribution ε (restricted to V ∖ A) is the same for every model at the cell.
- **(A4) Monotone accuracy.** p_{m,c} = σ(α_m − β_c). The order of models by accuracy is therefore the same at every cell.
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
  - The levels α_m are identified given a location constraint on β: the median of β_c over the non-defective anchored cells is 0.
- **T4 (finite samples).** Let n be the smaller number of readings per model at the cell, and |V| the number of candidate values.
  - **(a) The two-model contrast rule.** Â = {y : D̂(y) > Δ/4} errs with probability at most 4|V| exp(−n Δ²/32).
  - **(b) The plug-in mode under PC.** With margin γ = p − (1 − p) · max ε, the mode errs with probability at most 2|V| exp(−n γ²/2).

### 4.4 Scope of the theory

- **The bounds are conservative.** They come from Hoeffding's inequality. The MAP estimator with priors is what v3 runs, and the simulation (§8.1) measures its actual power.
- **Single-model typing.** Mis-key hidden requirements and weak checks rest on T1. They need no accuracy contrast, only a plurality margin.
- **Contrast-dependent typing.** Ambiguity and the *unstated* subtype rest on T2. They need Δ > 0, and their power scales with Δ²·n. With Δ ≈ 0, the priors decide, and they decide conservatively: P(|A| = 1) is high, and β is shrunk.
- **(A3) may fail.** Models can make different errors. A model-specific error concentrated on a value can mimic ambiguity. The mitigations are in §12, and the simulation sweeps this violation.
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
- **Robustness rule.** With three or more models of distinct α, an ambiguity must also show the accuracy-gradient sign at both values. The mass on each value must rise with α across the models, with Spearman correlation ≥ 0.5. This guards against (A3) violations (§12).

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
- **Why.** Theorem T2 needs models of distinct accuracy, and more raters raise the readings per cell. The P1b population (2 models, nearly equal θ) leaves ambiguity and the unstated subtype to the priors.
- **Applying it.** The threshold is applied to the cached P1b screen (`results/p1b-screen.json`); no new screen runs are needed. The population is chosen by the unchanged ranking:

| Family | Model | Act rate | Valid-call rate |
|---|---|---:|---:|
| qwen | qwen3:30b-a3b | 0.802 | 0.952 |
| qwen | qwen2.5:14b | 0.580 | 0.943 |
| openai | gpt-oss:20b | 0.914 | 0.988 |
| mistral | mistral-small3.2:24b | 0.346 | 0.914 |

  Within the qwen family, qwen3:14b (0.407) is ranked out by the per-family cap.
- **Disclosure.** The new threshold was chosen during design, knowing which cached screen results it admits. The screen is outcome-free: it reads no mutant, truth or diagnosis. The change is recorded in the pre-registration before any v3 result.
- **Fallback: a controlled weakened reader.** It is used only if G0 shows that the contrast of this population is too small for ambiguity. The weakened reader is the population's highest-act model with `search_k` = 1 instead of 5, registered as a separate interpreter. It is interpreter-visible, so it gets new run keys. It costs about one extra population member's share of the budget.

---

## 8. Evaluation (amends v2 §4)

### 8.1 Simulation study (pre-registered; no GPU)

- **Generator.** Synthetic specifications. Each has:
  - C cells of both kinds (value and constraint), anchored and unanchored, with |V| drawn from the observed distribution;
  - M ∈ {2, 3, 4} models with abilities α_m;
  - cell difficulties β_c ~ N(0, τ_true²);
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
  - the misspecification arms.

  The phase diagrams are compared with the boundaries that T2–T4 predict.

### 8.2 GO steps

| Step | What | GPU | Pass criterion (pre-registered here) |
|---|---|---|---|
| **G0** simulation | Fit at the calibrated parameters and the planned budget; choose the free parameters of §9 by the selection rule | No | (i) mean \|θ̂ − θ\| ≤ 0.05; (ii) simulated typed F1 ≥ 0.6 for hidden requirement and for weak check, and ≥ 0.4 for ambiguity; (iii) v3 beats all three baselines on hidden requirement and ambiguity. If only (ii)-ambiguity fails, switch to the weakened reader (§7) and re-run G0 once. If G0 still fails, stop and report |
| **G1** base replay | Fit v3 with the parameters frozen at G0 on the cached base runs: the P1b smoke run and the P1 base runs | No | (i) θ_m ≥ 0.5 for gpt-oss:20b and qwen3:30b-a3b (no collapse); (ii) the real defect R-SO (v1 §4.4: `SO-0031` stays `Chờ duyệt` after `APR-0102` is approved) is reported as a hidden requirement, if any acted behaviour reached its cell. If none did, G1 records "not observable" for (ii) and does not fail on it. On failure, stop and report; parameters are not re-tuned |
| Freeze | The v3 pre-registration (§9) is committed with a test that pins every value | No | — |
| **G2** population | Apply §7 to the cached screen and freeze the population into the gate configuration | No | At least 2 families admitted, otherwise STOP (v2 rule) |
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
- ablations: anchor removed (back to latent truth), the joint class key instead of cells, the accuracy-gradient robustness rule removed, the v2 stage 2;
- the remaining real defects R-12 and R-DE;
- S2 (τ²-bench).

---

## 9. Pre-registration v3

The v3 pre-registration is a document (`experiments/p2-preregistration.md`) plus a test that pins every value. It is committed after G1 and before G2.

| Group | Values |
|---|---|
| Unchanged from v1 and v2 | temperature 0.7; Tell variants 0–2; seeds 1–3; budgets 200/100/80; scorability k = 3; K_max = 6; the gate rule; dev split (`dev_per_type` 5, `rng_seed` 20261006); harness nudges 2; ω = 0.3, m_min = 3; Rasch λ = 1.0; T-boundary scenarios ≤ 12; workers ≤ `OLLAMA_NUM_PARALLEL` |
| New, fixed by this spec | \|A_c\| ≤ 2; the candidate rule of §3.1; the normalisation of §2.5; the row-matching rule of §2.6; the cell scope of §2.4; pooling of lifted cells (§2.7); EM tolerance 1e-6 and at most 200 iterations; the screen's act-rate threshold 0.3; the ambiguity robustness rule (Spearman ≥ 0.5 with three or more models); the G0–G3 criteria of §8.2 |
| Chosen at G0 by the selection rule, then frozen | π and the form shares (s_mis, s_amb, s_∅); τ; κ; μ_α, σ_α; τ_H, τ_A, τ_W |
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
- The ambiguity robustness rule.
- Weak checks on unanchored cells.
- Determinism.
- The pre-registration test.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| (A3) fails: model-specific errors mimic ambiguity | High P(\|A\| = 1) prior; the Spearman robustness rule with three or more models (§5.3); the simulation's misspecification arm in G0's selection objective |
| Δ too small for ambiguity and the unstated subtype | The relaxed screen (§7); the weakened-reader fallback; the G0 criterion stops the work before any GPU is used |
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
6. **Accuracy contrast.** Relaxed screen (act ≥ 0.3), with a weakened-reader fallback.
7. **Gate.** The v2 rule is kept. This is the second iteration on the dev split, by the author's decision.

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

Under (A4), p_{1,c} − p_{2,c} = σ(α_1 − β_c) − σ(α_2 − β_c), which has the sign of α_1 − α_2 at every finite β_c. ∎

### T3

At an anchored cell with A_c = {d_c}, P_{m,c}(d_c) = p_{m,c} = σ(α_m − β_c) by (A2). Hence logit P_{1,c}(d_c) − logit P_{2,c}(d_c) = α_1 − α_2: the same value v at every non-defective anchored cell.

If more than half of the anchored cells are non-defective, more than half of the values ℓ_c equal v, so the median equals v. The order of the models follows from the sign of v. Given the contrasts, the median of β_c over the non-defective anchored cells being 0 fixes the levels:
- for any model m, the median over those cells of logit P_{m,c}(d_c) is α_m − median β_c = α_m. ∎

### T4

**(a) The two-model contrast rule.** Let P̂_m be the empirical distribution of n independent readings (A1). By Hoeffding's inequality, for each y and m:

P(|P̂_m(y) − P_m(y)| ≥ Δ/8) ≤ 2 exp(−2n(Δ/8)²) = 2 exp(−nΔ²/32).

By a union bound over y ∈ V and m ∈ {1, 2}, with probability at least 1 − 4|V| exp(−nΔ²/32), every |D̂(y) − D(y)| < Δ/4. On that event:
- for y ∈ A, D(y) ≥ Δ/2 (since |A| ≤ 2), so D̂(y) > Δ/4;
- for y ∉ A, D(y) ≤ 0, so D̂(y) < Δ/4.

Hence Â = A, and this includes A = ∅, where D ≡ 0.

**(b) The plug-in mode.** If every |P̂(y) − P(y)| < γ/2, the mode of P̂ equals the mode x of P, because P(x) − P(y) ≥ γ for every y ≠ x. Hoeffding's inequality and a union bound over V give the failure probability 2|V| exp(−2n(γ/2)²) = 2|V| exp(−nγ²/2). ∎

*Remark.* Readings across Tell variants are independent but not identically distributed. Hoeffding's inequality applies to their average with P_m taken as the variant-averaged distribution, as (A1) states. Pooling lifted cells across scenarios (§2.7) adds readings to n, but only if (A2)–(A4) hold for the pooled cell. A cell's difficulty may then differ across scenarios; the simulation measures the effect of pooling such cells.
