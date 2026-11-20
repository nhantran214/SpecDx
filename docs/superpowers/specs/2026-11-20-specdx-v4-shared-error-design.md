# SpecDx v4 design: shared agent errors, the statistics–logic division, and the frontier arm

**Date:** 2026-11-20
**Status:** draft for the author's approval (revision 2, after the builder check)

**Amends:**
- the v3 design: §3 (the anchored cell model), §4 (theory) and §5 (typing);
- the P3 design `2026-11-19-specdx-s2-s3-design.md`: §3 (constraints), §9 (comparison), §10 (novelty), §11 and §13b–§13d.

**Keeps:**
- cells, anchors and the set-valued latent A_c;
- hitting sets, refutation and ddmin;
- subjects S2 (τ²-bench with Verified) and S3 (SOPBench);
- the reduced run plan.

---

## 1. Intent and why

### What P3a found
- **Scale of the problem.** P3a replicated the method offline on 21,323 published SOPBench runs from 28 models. It flagged a hidden requirement on 513 of about 830 tasks; in the hotel domain, 186 of 195.
- **Not an implementation artefact.** The final review verified anchors and polarity on every run.
- **The causes are in the model:**
  - Difficulty β is pooled by group, so a hard task looks as easy as its group, and strong models' deviations count as evidence against C.
  - The pair path takes over: 532 of the 896 hidden findings are of the ambiguous-pick kind.
  - One act gives up to four correlated readings.
  - On multi-valued cells, the flat error pool also stops a shared mistake from being explained as an error.
- **The visible symptom.** The fit lets the majority win. In the library domain, gpt-5 gets the lowest θ of any model.
- **Why it matters.** Kim et al. (ICML'25) show that LLM errors are correlated. A method that separates specification defects from agent errors must survive shared agent errors, so this is the paper's weakest point.

### What the author decided (2026-11-20)
1. **Factor 1, approach B.** Make the shared agent error a first-class explanation in the model.
2. **Question 1, option (a): divide the labour between statistics and logic.**
   - Where the population shows an ability gradient at a cell, the statistics decide.
   - Where every model deviates alike, readings alone cannot decide, and the theory says so (§4).
   - Such cells go to execution witnesses (§5.2), or are reported as undetermined.
3. **Question 2, option (i).** S3's cells stay binary. The attractor's evidence comes from simulation and from S2's multi-valued fields. On S3, SpecDx's difference from IRT audits rests on:
   - set-valued truth;
   - the anchor;
   - typing;
   - execution witnesses;
   - the lower bound.
4. **Factor 3.** The primary comparison stays fully local, on one open-model population shared by every method. A frontier evaluation is added with Gemini. All of Gemini 2.5 Pro, 3.1 Pro and 3.0 Flash are available. There is no spending cap, but **every experiment that calls Gemini reports its tokens and its cost in USD** (§7.4).

### The honest theoretical position
v3's T2 already shows that the ability gradient separates a defect from a concentrated error: D(y) > 0 on A, and D(y) ≤ 0 off A, whatever ε is. v4 adds nothing to identifiability in the limit.

v4's contributions are threefold:
- **(i) An estimator that can express shared errors.** It has per-cell difficulty and an attractor, so a fitted model can actually use T2's signature.
- **(ii) A finite-sample lower bound.** It says when *no* method that reads only behaviour can decide. This is new, and it justifies the undetermined outcome.
- **(iii) A division of labour.** Execution witnesses resolve what statistics provably cannot.

## 2. Goals, non-goals, anti-goals

### Goals
- **G1.** The v4 reading model (§3).
- **G2.** Theory (§4): T5, a lower bound of Le Cam's kind; T6, a matching upper bound; and the undetermined rule derived from them.
- **G3.** Execution witnesses for cells that statistics cannot decide (§5.2).
- **G4.** Evidence:
  - simulation (G0v4);
  - a placebo control on real runs;
  - seeded mutants;
  - S2's real defects.
- **G5.** Ablations.
- **G6.** The frontier arm, with per-experiment cost reports.

### Non-goals
- External ability information inside the fit (approach C).
- Hosted models in the primary comparison.
- Changing cells, anchors, hitting sets or ddmin.
- Making S3's cells multi-valued.

### Anti-goals
- Tuning v4 until SOPBench's base looks plausible. Every choice is fixed on the simulation and the development split before any test split runs.
- Typing a cell the data cannot decide. Such a cell is undetermined, or goes to a witness, and it is always counted.
- Claiming any novelty for T5 identifiability over v3's T2.
- Any Gemini experiment that does not report its tokens and its cost.

## 3. The v4 reading model (replaces v3 §3.2–§3.3)

### 3.1 Per-cell difficulty
- p_{m,c} = σ(α_m − β_c), with β_c ~ N(β_{g(c)}, τ_c²) and β_g ~ N(0, τ²).
- Each cell has its own difficulty, shrunk toward its group.

### 3.2 Flat-plus-attractor errors, for non-silent sets only
- **When A_c ≠ ∅,** the error distribution on V_c ∖ A_c is (1 − λ_c)·u + λ_c·δ_{e_c}, with e_c ∈ V_c ∖ A_c ∖ {OTHER} and λ_c ~ Beta(a_λ, b_λ).
- **When A_c = ∅,** the readings keep v3's flat-pool ε_c ~ Dirichlet(κ·u_c) with κ = 10, and there is no attractor.
  - This keeps T2′: silence stays identifiable against a flat pool by a single model.
  - It also stops ∅ with an attractor from mimicking {x}.
- **On binary cells** (all of S3) the attractor is vacuous, and v4 reduces to per-cell difficulty plus the anchored set-valued model. This is disclosed as such.

### 3.3 Composite weights apply to shared parameters only
- An **act group** is the set of cells that one interpreter action determines together:
  - in S3, the four cells of a task;
  - in S2, the database fields one tool call writes;
  - otherwise, a single cell.
- In the M-step updates of α and β_g, each reading of a group of size k carries weight 1/k.
- Each cell's own posterior over A_c, β_c and λ_c uses its readings unweighted, because each act informs a cell only once.

### 3.4 Abilities
These are unchanged from v3: α_m ~ N(μ_α, σ_α²), additive for S2 pairs.

### 3.5 Fitting
- **Outer loop.** α and β_g are fitted by v3's EM, with multi-start, run to convergence.
- **Per cell.** The posterior over A_c uses a **profile likelihood**: for each candidate A, maximise over (β_c, λ_c, e_c) under their priors, then combine with the prior over A.
  - This replaces v3's single plug-in β per cell.
  - With a plug-in, EM locks onto one explanation and hides non-identifiability. With profiles, every candidate gets its best case.

## 4. Theory (amends v3 §4)

### Assumptions
- **(A3′)** replaces (A3). For A ≠ ∅, errors lie in the flat-plus-attractor family, shared by all models at the cell, with λ unrestricted. For A = ∅, the flat pool applies.
- **(A4)** and **(A5)** are kept.
- (A1) and (A2) are as in v3.

### Lemma (T2 per cell)
T2 holds per cell under (A3′): A_c = {y : D_c(y) > 0}. This is v3's T2, unchanged in substance. It is stated per cell because v4 makes difficulty per cell.

### T5: lower bound
- **Setting.** Consider the two hypotheses at a cell:
  - **H** (mis-key): A = {a}, at per-cell accuracy p_H;
  - **S** (shared error): A = {d}, λ = 1, e = a, at accuracy p_S.
- **Statement.** For every test based on M models' readings, with n readings each and with α known, the error is at least ½·(1 − TV). TV is bounded by a function of n·Σ_m (p_{m,H} − (1 − p_{m,S}))².
- **When the bound bites.** Choose p_S per cell to minimise the bound. The minimum is 0 exactly when the population's accuracies at the cell can be matched by a common shift. That happens when the ability spread at the cell is small relative to 1/√n.
- **In words.** Unanimous or saturated deviation, or a population with small spread, makes H and S statistically indistinguishable *for any method*.
- **Proof route.** Le Cam's two-point method with a Pinsker or Hellinger bound on products of categorical distributions.

### T6: upper bound
- The profile-likelihood rule (§3.5), or v3's two-model contrast rule, decides with error ≤ C·exp(−c·n·Δ_c²).
- Here Δ_c is the largest accuracy contrast at the cell, from v3's T4(a), stated per cell.
- T5 and T6 together give the rate. This gives the undetermined rule (§5.1) its principled threshold.

### What the theory says to the paper
- Readings decide only where the population's ability gradient at the cell is large enough. The number of readings needed scales as 1/Δ_c².
- Elsewhere, behaviour-only methods (all of the paper's baselines) cannot decide either. SpecDx then uses execution, or says "undetermined".
- Population diversity (H5) becomes a quantitative condition.

## 5. Typing (amends v3 §5)

### 5.1 Undetermined
- **Definition.** A cell is statistically undetermined when the profile-likelihood ratio between the best defect form and the best no-defect explanation is below τ_U.
  - The no-defect explanation includes the shared-error explanation.
  - τ_U is set so that T6's error at the cell's estimated Δ_c is at most 0.05.
- **Routing.** Undetermined cells are passed to §5.2, and are never typed by statistics.

### 5.2 Execution witnesses: the logic half
For a statistically undetermined anchored cell, SpecDx asks the subject's own executable checker whether the anchor's behaviour is consistent.

**S3 (SOPBench):**
- **The check.** Replay the task's target action with the task's `user_known` parameters on the strict system. The strict system (`env/` `<Domain>_Strict`) is the code form of the constraints. The task's `directed_action_graph` gives the prerequisites, satisfied in graph order.
- **If the strict system's outcome differs from `action_should_succeed`:** C contradicts its own executable constraints. This is a C defect (hidden requirement, subtype *self-inconsistent*), and the witness is the replay.
- **If the outcome agrees:** the anchor is executable and consistent. The cell is a **shared agent error**, with the replay as witness.
- **If the replay is impossible:** an exception or a missing parameter leaves the cell undetermined, and it is counted.

**S2 (τ²-bench):**
- **The check.** Replay the task's reference actions in τ²-bench's environment.
- **If the replay fails**, because a tool refuses or an assertion fails: C is inconsistent with the environment. This is a C defect, witnessed.
- **Otherwise** the cell stays undetermined. τ²-bench has no executable form of its policy text.

**Elsewhere,** undetermined cells stay undetermined. Every report counts undetermined cells by the type each would otherwise have had, and counts witnessed shared errors and witnessed defects separately.

### 5.3 Hidden, ambiguity and weak
- These are v3's definitions on cells that statistics decided. The thresholds are re-chosen at G0v4.
- Shared-error cells decided statistically (A = {d}, λ_c ≥ τ_λ) and those witnessed in §5.2 go to an informational table, "where agents commonly fail". They are not defects.

## 6. Validation

### 6.1 G0v4: simulation, per subject (CPU)

**Calibration.** The simulator is calibrated per subject on that subject's development split. The calibration covers:
- the number of raters and the readings per rater;
- the spread of θ;
- vocabulary sizes;
- the deviation and error concentration rates;
- act-group sizes, including correlated readings within a group.

For S3, the development split is the published runs of a seeded draw of 3 of the 7 domains. The other 4 domains are held out for the placebo and the replication. S2 is calibrated in P4c, on its development tasks.

Each subject's parameters are frozen separately.

**Arms.** The simulation has four arms:
- **Ordinary:** v3's arm.
- **Shared-error:** hard cells with A = {d}, λ ~ U(0.5, 1).
- **Homogeneous ability:** equal or near-equal α.
- **Saturated:** a defect or shared error that every model reads alike.

**Criteria:**
1. On the shared-error arm, the hidden false-positive rate among determined cells is at most 0.1. v3's rate is reported.
2. On the ordinary arm, hidden and ambiguity F1 on determined cells is at least v3's minus 0.05, and the undetermined rate is reported.
3. On the homogeneous and saturated arms, at least 80% of planted defects and shared errors are undetermined. This is T5's prediction.

### 6.2 Placebo control on real runs (CPU, the 4 held-out S3 domains)

**Procedure.**
- Within each task, permute the raters' whole act vectors (all four cells together) across raters, with 20 seeds per domain.
- Refit everything, including α.

**Measure.** v4's typed findings on cells whose readings are not unanimous, so that permutation can change them, under placebo as a fraction of those on real readings. v3's ratio is reported beside it.

**Criterion.** v4's ratio is at most 0.2.

Unanimous cells are unaffected by permutation. They are reported separately, with their undetermined and witness outcomes.

### 6.3 Seeded and real defects
- Mutants and S2's real defects are evaluated as designed (P3 §6–§7), with v4 as SpecDx and v3 as an ablation.
- Witnessed findings are scored like any finding.

### 6.4 Ablations (H3)

| Ablation | Where it is evaluated |
|---|---|
| Without per-cell β | S2, S3 |
| Without attractor | S2 and simulation only; vacuous on S3, as disclosed |
| Without composite weights | S2, S3 |
| Without witnesses | S2, S3 |
| Plug-in instead of profile | S2, S3 |
| Full v3 | S2, S3 |

## 7. The frontier arm

### 7.1 Principle
- **Primary comparison:** unchanged, local open models, as in P3 §9.1.
- **Secondary frontier evaluation:** every LLM-reading baseline runs with its own paper's backend wherever possible, which is the setting most favourable to it.

### 7.2 Models

| Arm | Frontier setting | Fidelity |
|---|---|---|
| AgentSuite | Gemini-2.5-Pro (thinking), its paper's default judge | exact |
| BenchGuard | Gemini 3.1 Pro and Gemini 3.0 Flash, two of its paper's five backends | exact (subset) |
| Who&When | Gemini 2.5 Pro instead of GPT-4o | substitution, disclosed |
| auto-bench-audit | not re-run: it needs the Claude Code, Codex or Cursor CLIs | type-B table only |
| Run-reading baselines | read the frontier population's runs | exact |

### 7.3 SpecDx v4's frontier population
- **S3:** SOPBench's published trajectories, 6 families.
- **S2 base:** τ²-bench's published runs (OpenAI and Anthropic agents, 4 trials each), plus new Gemini agent runs with a Gemini user simulator.
- **S2 Verified and T-mutants:** Gemini runs mixed with the local open population, so that at least 2 families are present. This needs the 48 GB GPU as well as the API.

### 7.4 Constraints and cost reporting (amends P3 §3)
- **Scope.** Only the frontier arm may call a hosted API, and only Gemini.
- **Key handling.** The key comes from `GEMINI_API_KEY`. It is never written to logs, caches or reports.
- **Calls.** Every call goes through the cached gateway, appending to its own file `work/gemini-cache.jsonl`. Temperature is 0 where allowed. Model ids are pinned in the pre-registration.
- **Inputs.** Public benchmark texts only.
- **No spending cap**, by the author's decision.
- **Cost report per experiment.** Every stage that calls Gemini writes a "Cost" section in its Markdown report:
  - per model: calls, input tokens, output tokens and thinking tokens, with cache hits counted separately;
  - the price per million tokens used, with the date of the price table;
  - the cost in USD per model and in total.
- **Price table.** It lives in a committed file, with its source and date. A price missing from it fails the stage, rather than guessing.
- **Dashboard.** `results/p3/STATUS.md` gains a cumulative Gemini cost column and total.

## 8. Approaches considered

| Approach | What it is | Outcome |
|---|---|---|
| A. Per-cell 2PL with discrimination | IRT with item difficulty and discrimination | Not chosen: close to Land & Bikel's IRT audit |
| **B. Shared-error component** | §3 | **Chosen**, with option (a)'s statistics–logic division and option (i)'s binary S3 |
| C. External ability anchoring | Fix α from known-clean tasks | Not chosen: needs clean tasks, and the methodological contribution is thin |

## 9. Delivery

These phases replace P3b–P3d.

| Phase | Content | Hardware |
|---|---|---|
| P4a | v4 model and typing; T5/T6 proofs; simulator arms; G0v4 for S3; S3 witnesses (strict replay); placebo and ablations on the held-out S3 domains | CPU |
| P4b | S3 on the GPU with v4 (mutants, baselines, pre-registration, test) | 48 GB GPU |
| P4c | S2 with v4: G0v4 for S2, witnesses (reference replay), real defects | 48 GB GPU |
| P4d | Frontier arm with Gemini; new baselines; type-B table; cost reports | API and 48 GB GPU |
| P4e | Paper rewrite | none |

## 10. Left to the builder
- The numerical method for the per-A profile maximisation.
- Deriving act groups for S2's tool calls.
- The prerequisite order for the strict replay, within §5.2, from the action graph.
- Module layout. v3 stays importable as an ablation.

## 11. Decisions log (2026-11-20)

| # | Decision | By | Why |
|---|---|---|---|
| 1 | Approach B | author | Novelty; answers the correlated-error critique |
| 2 | Option (a): statistics decide where the gradient exists, witnesses or "undetermined" elsewhere | author | Honest and theorem-backed; uses SpecDx's logic half |
| 3 | Option (i): S3 cells stay binary | author | Limits the work; the attractor's evidence comes from simulation and S2 |
| 4 | Gemini: no cap, a cost report per experiment; all three models available | author | — |
| 5 | Primary comparison local; Gemini frontier arm | author | Fairness at both cost levels |
| 6 | No novelty claimed for T5-style identifiability; the new theory is the lower bound and the undetermined rule | Claude | The builder check showed T2 already covers it |
| 7 | ∅ keeps the flat pool; attractor only for A ≠ ∅ | Claude | Keeps T2′; stops ∅ from mimicking {x} |
| 8 | Composite weights only in the α and β_g updates | Claude | Over-counting is in the shared parameters; per-cell evidence stays intact |
| 9 | Profile likelihood per candidate A | Claude | A plug-in β hides non-identifiability |
| 10 | G0v4 calibrated per subject on its development split; S3 split by domain, 3 development and 4 held out | Claude | The old calibration is on the dropped subject; avoids circularity with the placebo |
| 11 | Placebo permutes whole act vectors, refits α, and counts only cells permutation can change | Claude | Targets the gradient mechanism without the unanimous-cell artefact |
| 12 | The S2 Verified frontier population mixes Gemini with local models | Claude | The ≥2-family rule |
