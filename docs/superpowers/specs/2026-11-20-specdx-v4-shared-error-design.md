# SpecDx v4 design: shared agent errors, the statistics–logic division, and the frontier arm

**Date:** 2026-11-20
**Status:** draft for the author's approval (revision 2, after the builder check); amended 2026-11-25 under ruling TH-1 to match the proved statements of `docs/theory/specdx-v4-bounds.md` (§4, §3.2, §5.1, §5.3, §6.4; see §12, with the disclosures of §12.2)

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
- **On binary cells** (all of S3; amended 2026-11-25, ruling F-I1 under TH-1) the attractor is forced, not free, and v4 reduces to per-cell difficulty plus the anchored set-valued model.
  - The error side of {d} (or {a}) has two slots, the other value and OTHER, so the attractor e is the other value. With no OTHER reading, λ's likelihood profile sits at 1 (theory note Lemma 1.1).
  - The attractor is not inert: removing it shifts log_lr by (n_a − n_d)·log 2. That shift is a count of readings, not a model comparison.
  - λ's posterior on such a cell is a count rule. Under Beta(1, 3) it is λ_{d} = (n_dev − 2)/(n_dev + 2), which reaches 0.5 exactly when n_dev ≥ 6. It therefore carries no shared-error evidence.
  - **So statistical shared error is never emitted on a cell with at most two candidate values.** On S3, shared error is claimed only through execution witnesses (§5.2), consistent with the statistics–logic division (decision 2). λ stays in the likelihood.

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

*Amended 2026-11-25 (ruling TH-1).* This section now states what the theory note `docs/theory/specdx-v4-bounds.md` proves. Where the earlier draft differed, the departure is named D1–D5 as in the note's §7. The paper's wording follows the note.

### Assumptions
- **(A3′)** replaces (A3).
  - For A ≠ ∅, errors lie in the flat-plus-attractor family, the same for every model at the cell, with λ ∈ [0, 1] unrestricted.
  - For A = ∅, every model reads from the same flat pool ε.
  - *Wording change (Task 9).* The nuisances range over their prior supports: β_c over the reals, λ over [0, 1], e over V∖A∖{OTHER}, and ε over the **open** simplex. The code profiles β_c over the population-relative box of ruling T1-L′.
- **(A4)** is kept, made per cell: p_m = σ(α_m − β_c). **(A5)** is kept.
- (A1) and (A2) are as in v3.
- **Knowledge assumptions.** T5, T5-S and T6 take α as known.
  - The lower bounds stay valid when α is unknown.
  - T6 covers the α the code is given; the error of the fitted α is not covered (open point O1).

### Lemma (T2 per cell)
T2 holds per cell under (A3′): A_c = {y : D_c(y) > 0}. This is v3's T2, unchanged in substance, and no novelty is claimed for it. It is stated per cell because v4 makes difficulty per cell.

### T5: lower bound (the mis-key side)
- **Setting.** Consider two hypotheses at a cell:
  - **H** (mis-key): A = {a}, at per-cell accuracy p_H;
  - **S** (shared error): A = {d}, λ = 1, e = a, at accuracy p_S.

  They differ only in the sign of the ability slope of the reading a.
- **Statement (D1).** For every test based on the readings, with α known, the error is at least ½·(1 − TV).
  - The earlier sentence "TV is bounded by a function of n·Σ_m (p_{m,H} − (1 − p_{m,S}))²" is false as a uniform statement. The note gives a counterexample near saturation (Remark T5.6).
  - What is proved instead is the weighted form TV ≤ √(½·Σ_m n_m Δ_m²/(y_m(1 − y_m))) and the exact Hellinger exponent h (Proposition T5.1).
  - A sandwich (Proposition T5.4) makes the dependence on n·Δ_c² exact up to constants when the accuracies are bounded away from 0 and 1.
- **When the bound bites (D2).**
  - The minimum over p_S is 0 **exactly** when the abilities at the cell have no spread (Theorem T5.2(i)).
  - The infimum over both nuisances is always 0, through saturation. So the global risk is ½, and the bounds are stated for **local** risks R_{β_H} (T5.2(ii), (iv)).
  - The local exponent's scale is N·sd(α)², proved two-sided: ½·η_H²(1 − η_H)²·N·sd_n(α)² ≤ h* ≤ ⅛·N·sd_n(α)² (Proposition T5.4(a′), Corollary T5.3).
  - So the near-chance regime is sd(α)·√(n·M) small, up to factors set by the accuracy level. It is not "spread small relative to 1/√n".
- **In words (D5).** Unanimous or saturated deviation, unanimous or saturated **agreement**, and a population with small spread all make H and S statistically indistinguishable *for any method*.
  - The code types an all-agree cell "no defect" once N is large enough (N ≥ 2, 5 or 12 at τ_u = 1, 2 or 3; Proposition T5.5(c)). That is the candidate-value convention: an unobserved value is not a candidate. It is not evidence.
  - Below that N, an all-agree cell is undetermined.
- **Proof route.** Le Cam's two-point method, with the Hellinger affinity and Pinsker's inequality, on products of Bernoulli laws (binary cells).

### T5-S: lower bound (the shared-error side)
- Every rule's local risk at a shared-error cell is bounded below against each defect form the code's statistic tests: {a}, {d, a} and ∅ (Theorem T5-S).
- The shared-error side is harder than the mis-key side, so more true shared errors than true mis-keys are expected to be undetermined.

### T6: upper bound (D3)
- **What is proved.** For **binary** cells, which covers all of S3, the code's profile-likelihood rule decides with error at most C·exp(−n_min·Δ_c²/4) (Theorem T6(c), Corollary T6.1).
  - c = ¼.
  - C = (2G_B + G_∅)·e^{(τ+½)/2} is polynomial in N, not an absolute constant.
  - An extra term covers the all-d corner.
- Δ_c is the largest accuracy contrast at the cell, from v3's T4(a), stated per cell.
- **Cells with |V| ≥ 3** (S2): only the contrast rules are proved (v3's T4(a) per cell, T6(b)). The profile rule there is open point O3.
- **Matching.** T5, T5-S and T6(c) match term by term on both sides: the same exponents, up to a factor 2, with polynomial prefactors (theory note §4.4).

### τ_U (D4)
- τ_U is **not** derived from T6. The earlier draft said "τ_U is set so that T6's error at the cell's estimated Δ_c is at most 0.05". That is neither what the code does nor something T6's constants can deliver: certifying 0.05 that way would need τ ≈ 10 nats.
- τ_U is one constant, selected at G0v4 by simulation from {1, 2, 3} nats and judged by criterion 1 with a Wilson upper bound (§6.1).
- **Its error control rests on the simulation, not on the theory**, and the paper says so.

### What the theory says to the paper
- Readings decide only where the population's ability gradient at the cell is large enough. The number of readings needed scales as 1/Δ_c².
- Elsewhere, behaviour-only methods (all of the paper's baselines) cannot decide either. SpecDx then uses execution, or says "undetermined".
- Population diversity (H5) becomes a quantitative, per-cell condition. For example, no rule reaches local risk 0.05 at a mis-key cell unless sd(α) ≥ 2.58/√(n·M) over the models that read it.
- **Do not claim:**
  - novelty for identifiability (v3's T2);
  - that τ_U is derived from T6 (D4);
  - any non-trivial bound on the global risk, which is ½.

## 5. Typing (amends v3 §5)

### 5.1 Undetermined
- **Definition.** A cell is statistically undetermined when the profile-likelihood ratio between the best defect form and the best no-defect explanation is below τ_U in absolute value.
  - The no-defect explanation includes the shared-error explanation.
  - The ratio is prior-free (ruling T1-L), and β_c is profiled over the population-relative box of ruling T1-L′.
  - *Amended 2026-11-25 (TH-1, D4).* τ_U is one constant, selected at G0v4 by simulation (§6.1) from {1, 2, 3} nats. It is not certified by T6 (§4).
- **Routing.** Undetermined cells are passed to §5.2, and are never typed by statistics.

### 5.2 Execution witnesses: the logic half
*Amended 2026-11-25:* only an undetermined outcome cell whose plurality deviates from the anchor goes to the witness (W-1); see §12.1.

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
  - *Amended 2026-11-25 (F-I1).* Statistical shared error is emitted only on cells with three or more candidate values (§3.2). On S3's binary cells, shared errors are the witnessed ones only.

## 6. Validation

### 6.1 G0v4: simulation, per subject (CPU)

**Calibration.** The simulator is calibrated per subject on that subject's development split. The calibration covers:
- the number of raters and the readings per rater;
- the spread of θ;
- vocabulary sizes;
- the deviation and error concentration rates;
- act-group sizes, including correlated readings within a group.

For S3, the development split is the published runs of a seeded draw of 3 of the 7 domains. The other 4 domains are held out for the placebo and the replication. S2 is calibrated in P4c, on its development tasks.

*Amended 2026-11-25:* the S3 simulator copies each development domain's group layout and fits each simulated domain on its own (F-C1), and the selection counts an undetermined planted cell as a miss (F-I2); see §12.1.

**Difficulty spreads (ruling R-5).** A cell's logit accuracy x_c = logit((hits + 0.5)/(n + 1)) spreads by binomial sampling as well as by difficulty, with sampling variance about v_c = 1/(n·p·(1 − p)) (p clipped to [0.01, 0.99]). By the method of moments:
- the within-group variance is the pooled variance of the cells about their group means, minus the mean v_c;
- the between-group variance is the variance of the group means, minus the mean over groups of (σ_w² + v̄_g)/n_g.

Both sds are floored at 0.25 (SPREAD_FLOOR), so pure sampling noise calibrates to the floor, and capped at 3. Otherwise the simulated within-group spread would be inflated relative to the real structure that cell_sd is matched against. Only cells with readings count, and a user goal whose tasks have no reading (every run a tool error) is not in the layout (ruling R-1).

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
*Amended 2026-11-25:* the placebo counts distinct cells and permutes only among a task's runs without a tool error (P-C); see §12.1.

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
| Without attractor | S2 and simulation only. Not run on S3: on binary cells the attractor is forced, and removing it only shifts log_lr by (n_a − n_d)·log 2, a count (§3.2; amended 2026-11-25, F-I1) |
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

## 12. Amendment of 2026-11-25 (ruling TH-1) and disclosures

### 12.1 What the amendment changes
- **§4** follows the theory note: D1 (T5's form), D2 (when T5 bites, and its scale), D3 (T6 on binary cells, polynomial constant), D4 (τ_U from the simulation, not from T6), D5 (unanimous agreement), and the (A3′) wording.
- **§3.2 and §5.3** (ruling F-I1): on binary cells the attractor is forced and λ is a count rule, so no statistical shared error is emitted there; on S3 shared error comes from witnesses only.
- **§5.1** (D4): τ_U's definition and where it comes from.
- **§6.4** (F-I1): the attractor ablation's S3 entry.
- **Pre-freeze corrections, not deviations.** Before any official run and before the G0v4 parameters are frozen, the final fix wave applied these rulings:
  - F-C1: the G0v4 simulator copies each development domain's S3 group layout and fits each simulated domain on its own;
  - F-I2a and F-I2b: the selection objective counts an undetermined planted cell as a miss, and criterion 2.ambiguity is reported as non-binding when v3's ambiguity F1 is below 0.05;
  - W-1: only an undetermined outcome cell whose plurality deviates from the anchor goes to the witness;
  - P-C: the placebo counts distinct cells and permutes among the runs without a tool error.
  - R-1 and R-5: the S3 calibration leaves out goals with no read task and subtracts the binomial sampling variance from both difficulty spreads (§6.1);
  - R-3 and R-4: only the s3-g0v4 stage's own search, on the development split, can be frozen, and the frozen file records the commit the search ran at.

### 12.2 Disclosures
- **Held-out data seen before the official run.** Two looks at held-out S3 domains happened during development:
  - the smoke run on bank: its v4 counts and its placebo ratio (0.192);
  - the Task 7 witness spot check on the first 30 tasks of bank and library.
- **Which rulings came before those looks.** W-1 (Task 9) and P-C (Task 8) were ruled before the smoke run. F-C1, F-I1 and F-I2 come from code and simulator review, not from bank numbers.
- **Smoke runs no longer touch held-out data.** From the final fix wave on, smoke s3-v4 and s3-placebo run on the first development domain only (ruling F-I3).
