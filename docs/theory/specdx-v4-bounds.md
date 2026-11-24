# SpecDx v4 theory: when readings cannot tell a mis-key from a shared error

**Date:** 2026-11-25
**Status:** theory note for P4a Task 9. It implements design v4 §4 (`docs/superpowers/specs/2026-11-20-specdx-v4-shared-error-design.md`).

**Matches the code at:** `specdx/infer/anchored_v4.py` and `specdx/infer/typing_v4.py` at commit 2bd4fe9. These rulings are in force: T1-L′, the ∅ addendum, T2-U′, T3-N and G0-V.

**Numbers:** every number in §6 comes from `scripts/v4_bounds_table.py`, and `tests/test_v4_bounds_table.py` tests it.

---

## 0. Summary, and what is new

At one anchored cell, v4 must tell apart two explanations of the readings that disagree with the anchor d:
- **H, a mis-key.** The specification entails a value a ≠ d (A_c = {a}).
- **S, a shared agent error.** The specification is right (A_c = {d}), but every agent that errs makes the same error a (λ = 1, e = a).

Under v4's accuracy model the two differ only in the **sign of the ability slope** of the reading a. Write P(a) for the probability that a model reads a:
- under H, logit P(a) = α_m − β_H;
- under S, logit P(a) = β_S − α_m.

Everything below follows from this mirror image (Proposition T5.1).

**What is old.** None of these is claimed as new:
- Identifiability in the limit is v3's T2 (Lemma T2 below). It is restated per cell, with **no novelty claimed**: v4 adds nothing to identifiability.
- The technique is Le Cam's two-point method, with the Hellinger affinity and Pinsker's inequality. It is textbook material (Le Cam 1973; Tsybakov 2009, ch. 2).
- The contrast rule of T6(b) is v3's T4(a), applied per cell. In psychometric terms it is the classical item-analysis signal: an item whose "correct" answer is anti-correlated with ability is suspected of being mis-keyed.

**What this note adds.** It is a direct application of the old tools to the mis-key/shared-error pair:
1. the mirror-image construction;
2. the exact exponent of the bound, as a Jensen gap of the Bernoulli log-partition function;
3. a characterisation of when the bound reaches chance (no spread, or saturation), and why the β_c box must be relative to the population (ruling T1-L′);
4. upper bounds for three rules, the last being the statistic the code actually computes, with the same exponent up to a factor 2;
5. what follows for the undetermined outcome, the execution witnesses and population diversity (H5).

We are not aware of this two-point construction in the literature on crowd labelling or on auditing specifications. It is still an application, and should be presented as one.

**Where this note departs from design §4.** The departures are listed in §7. In short:
- two sentences of the design's T5 are false as written (D1, D2);
- T6 is proven for the code's statistic only on binary cells, with a constant that is polynomial in N rather than absolute (D3);
- the τ_U calibration the design describes is not what the code does (D4).

---

## 1. Setting

### 1.1 Notation

The cell index c is suppressed where it is clear.

**The cell.**
- d is the anchor.
- V is the tuple of candidate values. It is built by `candidate_values`: d first, then the cell's observed values by frequency, at most K_max = 6 in all. One OTHER slot holds every remaining value. k = |V| + 1 is the number of slots.
- A ⊆ V∖{OTHER} is the latent entailment set: ∅, a single value, or a pair.

**The models.**
- Models m = 1, …, M have readings at the cell. They are the "raters present".
- Model m has n_m ≥ 1 readings. N = Σ_m n_m, n_min = min_m n_m, and n_m = n when all are equal.
- α_m is model m's ability, α_max and α_min the largest and smallest, and ᾱ_n = Σ_m n_m α_m / N the reading-weighted mean.
- σ(t) = 1/(1 + e^{−t}). Its derivative is σ′(t) = σ(t)(1 − σ(t)), with σ′ ≤ ¼ and σ′(t) ≤ e^{−|t|}.
- A(θ) = log(1 + e^θ) is the Bernoulli log-partition function, so A′ = σ and A″ = σ′.

**The reading law (design §3).** The accuracy is p_m = σ(α_m − β), with β = β_c the cell's own difficulty.
- **A ≠ ∅.** A reading falls in A with probability p_m, uniformly over A. Otherwise it is drawn from the error law ε̃ on the slots outside A: ε̃ = (1 − λ)·u_{∖A} + λ·δ_e. Here u_{∖A} is flat over the slots outside A (OTHER included), λ ∈ [0, 1], and e ∈ V∖A∖{OTHER}.
- **A = ∅.** Every reading comes from the flat-pool ε.

### 1.2 Assumptions

(A1) and (A2) are v3's. (A3′) replaces (A3). (A4) is v3's, made per cell. (A5) is v3's.

- **(A1) Independence.** Given A and the parameters, all readings at the cell are independent: across models, and across one model's runs.
- **(A2) Accuracy.** When A ≠ ∅, P_m(A) = p_m, faithful readings are uniform on A, and |A| ≤ 2.
- **(A3′) Shared errors.**
  - For A ≠ ∅, the error law is flat-plus-attractor, the same for every model at the cell, with λ ∈ [0, 1] unrestricted.
  - For A = ∅, it is the flat pool.
- **(A4) Monotone accuracy, per cell.** p_{m,c} = σ(α_m − β_c).
- **(A5) Silence.** When A = ∅, every model reads from ε, so no model is faithful.

**Knowledge assumptions.** T5 and T6 take α as known, and take the truth to lie in the model. Not knowing α can only make the decision harder, so T5 stays valid when α is unknown: the minimax risk over a larger class is at least as large. T6 is stated for the α the code is given. In the code that α is fitted by `fit_v4`, with composite weights and every cell included (ruling T2-U′). Error in α is not covered by T6 (§7, open point O1).

### 1.3 Binary cells

Every S3 cell is binary. An outcome cell reads `succeeds` or `refused`; a check cell reads `satisfied` or `violated`.

**Lemma 1.1 (the binary reduction).** Take an anchored cell whose readings all lie in {d, a}, with a ≠ d. Let N_m(d) and N_m(a) be model m's counts, and N_d and N_a the totals.

If both values are observed, then V = (d, a), the slots are d, a and OTHER (k = 3), and the candidate sets are {d}, {a}, {d, a} and ∅. Each set's likelihood, maximised over λ and e as the code does, is as follows.

1. **A = {d}.** The attractor is forced to e = a. The law is P(d) = p, P(a) = (1 − p)(1 + λ)/2 and P(OTHER) = (1 − p)(1 − λ)/2. On data without OTHER, the λ-dependent part of the log-likelihood is N_a·log((1 + λ)/2). This is non-decreasing in λ and equals 0 at λ = 1, so the profile sits at λ = 1, where
   log L = Σ_m [N_m(d)·log σ(α_m − β) + N_m(a)·log σ(β − α_m)].
   If N_a = 0, λ does not enter.
2. **A = {a}.** This is case 1 with d and a swapped. At λ = 1, e = d:
   log L = Σ_m [N_m(a)·log σ(α_m − β) + N_m(d)·log σ(β − α_m)].
3. **A = {d, a}.** No value is left for an attractor, so errors go to OTHER. The law is P(d) = P(a) = p/2, and the code subtracts log 2 per reading inside A. So
   log L = Σ_m N_m·log(σ(α_m − β)/2).
4. **A = ∅.** The code uses the pool ε̂ = (N_slot + κ/3)/(N + κ), with κ = 10 (ledger addendum to T1-L). So
   log L = N_d·log ε̂_d + N_a·log ε̂_a.

If a is never observed, then V = (d,), and the candidate sets are only {d} and ∅.

*Proof.*
- **Case 1.** With A = {d} the slots outside A are a and OTHER, two of them, and the only value among them is a. So e = a, and ε̃(a) = (1 − λ)/2 + λ = (1 + λ)/2 and ε̃(OTHER) = (1 − λ)/2. A reading of a contributes log(1 − p) + log((1 + λ)/2). There are no OTHER readings. Summing over the readings gives the λ-part N_a·log((1 + λ)/2), which is maximal at λ = 1 with value 0. What remains is Σ N_m(d)·log p_m + N_m(a)·log(1 − p_m), and 1 − σ(α − β) = σ(β − α).
- **Case 2.** The same, with d and a swapped.
- **Case 3.** With A = {d, a} the only slot outside A is OTHER. So `choices` is empty in `_profile`, there is no attractor, and every reading in A has probability p_m/2.
- **Case 4.** This is `_empty_fit` with per_slot = (N_d, N_a, 0) and k = 3.
- **The last sentence.** `candidate_values(d, observed)` lists d and then only observed values. ∎

So on S3 the attractor is vacuous, as design §3.2 discloses. H and S are the two logistic families:
- P(a) = σ(α_m − β) for {a};
- P(a) = σ(β − α_m) for {d}.

### 1.4 The statistic the code computes

**Definition 1.2 (log_lr, ruling T1-L′ with the ∅ addendum).**
- **The box.** B = [α_min − b, α_max + b], with b = `BOX_MARGIN` = 10 and the abilities taken over the raters present at the cell. Its width is W = α_max − α_min + 2b.
- **A ≠ ∅.** ℓ̂(A) = sup over β ∈ B, λ ∈ [0, 1] and e of the data log-likelihood of A. Priors do not enter.
- **A = ∅.** ℓ̂(∅) is the data log-likelihood at the pool ε̂ of Lemma 1.1(4).
- **On an anchored cell:**
  log_lr = max over the defect sets A ≠ {d} of ℓ̂(A), minus ℓ̂({d}).
- **Typing (ruling T3-N):**
  - |log_lr| < τ_u: undetermined, and nothing else;
  - log_lr ≤ −τ_u: no defect type, though possibly a shared error, read from the {d} set's λ;
  - log_lr ≥ τ_u: v3's finders.

  τ_u is selected at G0v4 from the grid {1, 2, 3} nats.

The posterior over sets, which carries the priors, only types. It never decides between undetermined and determined.

---

## 2. Lemma T2 per cell (v3's T2; no novelty claimed)

**Lemma T2.** Take two models that read the cell, with α_1 > α_2, and a finite β_c. Let Δ_c := p_{1,c} − p_{2,c} = σ(α_1 − β_c) − σ(α_2 − β_c) and D(y) := P_1(y) − P_2(y). Under (A2), (A3′) and (A5):
- if A ≠ ∅, then D(y) = Δ_c·(1{y ∈ A}/|A| − ε̃(y));
- if A = ∅, then D ≡ 0.

Since Δ_c > 0, it follows that A = {y : D(y) > 0}.

*Proof.* This is v3's Appendix A, word for word, with β_{g(c)} replaced by β_c.
- **A ≠ ∅.** P_m(y) = p_m·1{y ∈ A}/|A| + (1 − p_m)·ε̃(y). The error law ε̃ is the same for both models by (A3′), so subtracting gives D(y) = (p_1 − p_2)·1{y ∈ A}/|A| − (p_1 − p_2)·ε̃(y). Since ε̃ vanishes on A:
  - D(y) = Δ_c/|A| > 0 for y ∈ A;
  - D(y) = −Δ_c·ε̃(y) ≤ 0 for y ∉ A.
- **A = ∅.** By (A5), P_1 = P_2 = ε.
- **Δ_c > 0.** σ is strictly increasing, and α_1 − β_c > α_2 − β_c. ∎

The only use of (A3′) is that ε̃ is a law on the slots outside A shared by the two models. The attractor, including λ = 1, changes nothing.

In the H/S example:
- under H, D(a) = Δ_c > 0;
- under S with λ = 1, D(a) = −Δ_c < 0.

**In the limit, then, H and S are always separated when Δ_c > 0.** T5 and T6 concern the rate at which they separate.

---

## 3. T5: the lower bound

### 3.1 Tools

All laws below live on finite sets. Write BC(P, Q) := Σ_y √(P(y)·Q(y)) (the Bhattacharyya coefficient) and TV(P, Q) := ½·Σ_y |P(y) − Q(y)|.

**Lemma 3.1 (Le Cam's two-point lemma).** For any randomised test φ, where φ(y) ∈ [0, 1] is the probability of answering "Q":

P(φ answers Q) + Q(φ answers P) ≥ Σ_y min(P(y), Q(y)) = 1 − TV(P, Q).

Hence max{P(φ answers Q), Q(φ answers P)} ≥ ½·(1 − TV(P, Q)). The likelihood-ratio test φ* = 1{Q(y) > P(y)} attains the first bound with equality.

*Proof.*
1. The sum of the two errors is Σ_y [P(y)·φ(y) + Q(y)·(1 − φ(y))]. Each summand is a convex combination of P(y) and Q(y), so it is at least min(P(y), Q(y)). φ* puts all its weight on the smaller one.
2. min(p, q) = (p + q − |p − q|)/2. Summing gives Σ min = 1 − ½·Σ|P − Q| = 1 − TV.
3. A maximum is at least the average. ∎

**Abstaining rules.** Let a rule R output H, S or U (undetermined). Suppose R decides correctly with probability at least 1 − δ under both P_H and P_S. Then the test φ = 1{R = H} has
- P_H(φ ≠ H) = P_H(R ≠ H) ≤ δ, and
- P_S(φ = H) ≤ P_S(R ≠ S) ≤ δ.

So every lower bound below also bounds "the probability of not deciding correctly", for rules that may answer "undetermined".

**Lemma 3.2 (Hellinger affinity).**
- (a) TV ≤ √(1 − BC²).
- (b) 1 − TV ≥ ½·BC².
- (c) BC(P_1 ⊗ P_2, Q_1 ⊗ Q_2) = BC(P_1, Q_1)·BC(P_2, Q_2).
- (d) Σ_y min(P, Q) ≤ BC.
- (e) For u ∈ [0, 1], 1 − √(1 − u) ≥ u/2.

*Proof.*
- **(a)** Σ|p − q| = Σ|√p − √q|·(√p + √q). By Cauchy–Schwarz this is at most √(Σ(√p − √q)²)·√(Σ(√p + √q)²) = √(2 − 2BC)·√(2 + 2BC) = 2·√(1 − BC²). Divide by 2.
- **(b)** BC = Σ √(min·max) ≤ √(Σ min)·√(Σ max), by Cauchy–Schwarz. Since Σ max = 2 − Σ min ≤ 2, this gives BC² ≤ 2·Σ min = 2(1 − TV).
- **(c)** Σ_{y_1, y_2} √(P_1(y_1) P_2(y_2) Q_1(y_1) Q_2(y_2)) factorises into the two sums.
- **(d)** min(p, q) ≤ √(pq).
- **(e)** √(1 − u) ≤ 1 − u/2, because (1 − u/2)² = 1 − u + u²/4. ∎

**Lemma 3.3 (Pinsker, and KL ≤ χ²).** This is used only in Remark T5.6; no theorem rests on it.
- **Pinsker's inequality.** TV(P, Q) ≤ √(KL(P‖Q)/2) (Tsybakov 2009, ch. 2). It is cited, not proven here. KL adds over independent coordinates.
- **KL ≤ χ².** By Jensen's inequality for the concave log, KL(P‖Q) = Σ P·log(P/Q) ≤ log Σ P²/Q = log(1 + χ²(P‖Q)) ≤ χ²(P‖Q).
- **Two Bernoulli laws.** χ²(Bern x ‖ Bern y) = (x − y)²/y + (x − y)²/(1 − y) = (x − y)²/(y(1 − y)).

### 3.2 The theorem

**Theorem T5 (lower bound).** Take an anchored cell with candidate values V ∋ d, a, where a ≠ d, and with M models of known abilities α_m and n_m ≥ 1 readings each, under (A1).

**The classes.**
- 𝓗 is the class of mis-key configurations (A = {a}, β_H, λ_H, e_H).
- 𝓢 is the class of shared-error configurations (A = {d}, β_S, λ_S, e_S).

**The worst-case error.** For any test φ of all the readings, possibly randomised:
R(φ) := max{ sup_𝓗 P(φ = S), sup_𝓢 P(φ = H) }.

**The two points.** Take any β_H, β_S ∈ ℝ, and the points
- H° = ({a}, β_H, λ = 1, e = d), and
- S° = ({d}, β_S, λ = 1, e = a).

Write x_m = σ(α_m − β_H) and y_m = σ(β_S − α_m) = 1 − p_{m,S}, and
h(β_H, β_S) := Σ_m n_m·(−log BC(Bern x_m, Bern y_m)).

Then:
inf_φ R(φ) ≥ ½·(1 − TV(P_{H°}, P_{S°})) ≥ ½·(1 − √(1 − e^{−2h(β_H, β_S)})) ≥ ¼·e^{−2h(β_H, β_S)}.

**The Pinsker form.** TV(P_{H°}, P_{S°}) ≤ √(½·Σ_m n_m·KL(Bern x_m ‖ Bern y_m)).

**The local form.** For every fixed β_H, the bound holds with h replaced by h*(β_H) := inf_{β_S ∈ ℝ} h(β_H, β_S).

**Abstaining rules.** All of these bounds also bound the probability of not deciding correctly, for rules that may abstain (§3.1).

*Proof.*
1. **Two points suffice.** H° ∈ 𝓗 and S° ∈ 𝓢. So R(φ) ≥ max{P_{H°}(φ = S), P_{S°}(φ = H)}.
2. **Le Cam.** Apply Lemma 3.1 to the joint laws of all readings, which live on a finite set. This gives inf_φ max{…} ≥ ½·(1 − TV(P_{H°}, P_{S°})).
3. **Every reading is a Bernoulli.**
   - Under H°, a model reads a with probability x_m (faithful). Otherwise it errs, and with λ = 1 and e = d the error is d. So P(a) = x_m, P(d) = 1 − x_m, and every other slot has probability 0.
   - Under S°, P(d) = p_{m,S} and P(a) = 1 − p_{m,S} = y_m.

   Both laws live on {d, a}. So each reading is Bern(x_m) against Bern(y_m) in the event "reading = a".
4. **Tensorise.** By (A1) the joint laws are products. Lemma 3.2(c) gives BC(P_{H°}, P_{S°}) = Π_m BC_m^{n_m} = e^{−h}.
5. **Hellinger.** Lemma 3.2(a) gives TV ≤ √(1 − e^{−2h}), and Lemma 3.2(e) gives ½·(1 − √(1 − e^{−2h})) ≥ ¼·e^{−2h}.
6. **Pinsker.** This form is Lemma 3.3 applied to the product law.
7. **The local form.** Step 1 holds for every β_S. Take the supremum of the bound over β_S, which is the infimum of h. ∎

**Remark (the choice of the H point).** The design leaves H's error law open. T5 fixes λ_H = 1 and e_H = d.
- On a binary cell this choice is forced. Readings lie in {d, a}, so OTHER is never read and the profile sits at λ = 1 (Lemma 1.1).
- On cells with three or more values the pair is **not** claimed to be least favourable. If H and S both err mostly onto a common third value b, both laws put most of their mass on b. For example:
  - H: P(a) = 0.1 and P(b) = 0.9;
  - S: P(d) = 0.1 and P(b) = 0.9.

  This pair has BC = 0.9, above the 0.6 of the λ = 1 pair at the same accuracies. So T5 as stated is a valid lower bound, but on |V| ≥ 3 cells it is not the strongest one available.

### 3.3 The exponent in closed form

**Proposition T5.1 (the mirror image).** For Bernoulli laws with natural parameters θ_1 and θ_2 (so that P(a) = σ(θ)):

−log BC = ½·A(θ_1) + ½·A(θ_2) − A((θ_1 + θ_2)/2).

Now reparametrise the pair (β_H, β_S) by
- c := (β_S − β_H)/2, a common logit, and
- γ := (β_H + β_S)/2, a pivot ability,

and put δ_m := α_m − γ. Then:
- H's logits are θ^H_m = α_m − β_H = c + δ_m;
- S's logits are θ^S_m = β_S − α_m = c − δ_m;
- h(β_H, β_S) = Σ_m n_m·ψ_c(α_m − γ), with ψ_c(δ) := ½·A(c + δ) + ½·A(c − δ) − A(c).

ψ_c has four properties:
- **(P1)** It is even, and ψ_c(0) = 0.
- **(P2)** ψ_c(δ) > 0 for δ ≠ 0, and it increases in |δ|.
- **(P3)** ½·δ²·min_{|t−c|≤|δ|} σ′(t) ≤ ψ_c(δ) ≤ ½·δ²·max_{|t−c|≤|δ|} σ′(t) ≤ δ²/8.
- **(P4)** For fixed δ, ψ_c(δ) → 0 as |c| → ∞.

*Proof.*
- **The formula.** √(σ(θ_1)σ(θ_2)) = e^{(θ_1+θ_2)/2} / √((1 + e^{θ_1})(1 + e^{θ_2})), and √((1 − σ(θ_1))(1 − σ(θ_2))) = 1 / √((1 + e^{θ_1})(1 + e^{θ_2})). Their sum is (1 + e^{(θ_1+θ_2)/2}) / √((1 + e^{θ_1})(1 + e^{θ_2})) = exp(A((θ_1+θ_2)/2) − ½A(θ_1) − ½A(θ_2)). Take −log.
- **The reparametrisation.** β_H = γ − c and β_S = γ + c. So α_m − β_H = (α_m − γ) + c and β_S − α_m = c − (α_m − γ). The midpoint of the two logits is c.
- **(P1)** Evident from the definition.
- **(P2)** dψ_c/dδ = ½·(σ(c + δ) − σ(c − δ)). This is > 0 for δ > 0, since σ is strictly increasing. With (P1), (P2) follows.
- **(P3)** Taylor's theorem with Lagrange remainder gives A(c ± δ) = A(c) ± δ·A′(c) + ½·δ²·A″(ξ_±), with ξ_± between c and c ± δ. Adding the two, ψ_c(δ) = ¼·δ²·(σ′(ξ_+) + σ′(ξ_−)). Bound each σ′ by its minimum or maximum on [c − |δ|, c + |δ|], and use σ′ ≤ ¼.
- **(P4)** Follows from (P3), since max_{|t−c|≤|δ|} σ′(t) ≤ e^{−(|c|−|δ|)} → 0. ∎

**Reading the result.** At pivot γ and common logit c, the S configuration reproduces H's law model by model, except that each model's distance δ_m from the pivot appears with the opposite sign. The only thing that separates H from S is how far the models' abilities spread around the pivot, measured through the curvature σ′ of the log-partition function at the cell's accuracy level.

### 3.4 When the bound reaches chance

**Theorem T5.2 (characterisation).** Assume n_m ≥ 1 for every m.

- **(i) Exact indistinguishability.** h(β_H, β_S) = 0 if and only if α_m = (β_H + β_S)/2 for every m, and h = 0 if and only if TV = 0.
  - So a pair with TV = 0, where the bound equals ½ exactly, exists if and only if all the α_m are equal: **no spread**.
  - In that case every β_H has the exact mimic β_S = 2ᾱ − β_H.
- **(ii) Over unrestricted nuisances, always chance.** For every α, inf over (β_H, β_S) ∈ ℝ² of h = 0. If the spread is positive, the infimum is not attained.
  - Consequently inf_φ R(φ) = ½ exactly: no test beats a fair coin uniformly over 𝓗 ∪ 𝓢.
  - Every useful guarantee must therefore be **local**, depending on the cell's accuracy level.
- **(iii) Local and away from saturation.** Fix a finite β_H and suppose the spread is positive. Then h*(β_H) > 0. When n_m = n for all m, h*(β_H) = n·h*_1(β_H), so the local bound falls exponentially in n.
- **(iv) Saturation.** Fix α with positive spread. Then h*(β_H) → 0 as β_H → ±∞. Quantitatively, let c := ᾱ_n − β_H and D := max_m |α_m − ᾱ_n|. Whenever |c| ≥ D:
  h*(β_H) ≤ ½·σ′(|c| − D)·Σ_m n_m·(α_m − ᾱ_n)²,
  and σ′(|c| − D) ≤ e^{−(|c| − D)}.

*Proof.*
- **(i)**
  - By Proposition T5.1, h = Σ n_m·ψ_c(δ_m), a sum of non-negative terms. By (P2) it is zero exactly when every δ_m = 0, that is, when every α_m equals γ.
  - If h = 0, then BC = 1, so Σ(√p − √q)² = 2 − 2BC = 0 and the laws coincide, giving TV = 0. If TV = 0, the laws coincide and BC = 1.
  - With all α_m = ᾱ, the pivot γ = ᾱ means β_S = 2ᾱ − β_H.
- **(ii)**
  - Take γ = ᾱ_n, so δ_m = α_m − ᾱ_n is fixed, and let c → ±∞. By (P4) every ψ_c(δ_m) → 0, so h → 0.
  - The infimum is not attained when the spread is positive, by (i).
  - By Lemma 3.2(a), TV → 0 along this sequence, so sup over pairs of ½(1 − TV) = ½. Hence inf_φ R(φ) ≥ ½. A fair coin has R = ½.
- **(iii)**
  - β_S ↦ h(β_H, β_S) is continuous, and at every finite β_S it is positive by (i).
  - As β_S → +∞, y_m → 1, so BC_m → √x_m and h → −½·Σ n_m·log x_m. This is > 0 because x_m < 1 for finite β_H.
  - As β_S → −∞, h → −½·Σ n_m·log(1 − x_m) > 0.
  - Choose R so large that h > half of the smaller limit outside [−R, R]. On [−R, R], h attains a positive minimum. So h* > 0.
  - Linearity in n: h is n times its value at n = 1, so the minimiser does not depend on n.
- **(iv)**
  - h*(β_H) ≤ h at the pivot γ = ᾱ_n, where c = ᾱ_n − β_H and δ_m = α_m − ᾱ_n.
  - By (P3), h ≤ ½·Σ n_m·δ_m²·max_{|t−c|≤D} σ′(t).
  - σ′ is even and decreasing on [0, ∞). So when |c| ≥ D its maximum over [c − D, c + D] is at the end point nearest 0, where |t| = |c| − D.
  - Finally, σ′(t) = e^{−|t|}/(1 + e^{−|t|})² ≤ e^{−|t|}.
  - As |c| → ∞ the bound → 0. ∎

**In words.** A unanimous deviation is the H-saturated end: β_H → −∞, so every model reads a. A population with small spread is case (i). By Theorem T5.2 both make H and S statistically indistinguishable **for any method that reads only behaviour**. (iv) gives the rate: what remains is about the expected number of non-a readings, Σ n_m·(1 − x_m), weighted by the spread. The symmetric end, β_H → +∞ with every model reading d, is a **unanimous agreement**. It is just as undecidable; see Proposition T5.5(c).

### 3.5 The bound in terms of the spread and the accuracy contrast

**Corollary T5.3 (spread condition).** Let c := ᾱ_n − β_H, D := max_m |α_m − ᾱ_n| and s := max_{|t−c|≤D} σ′(t) ≤ ¼. For every β_H:

inf_φ R(φ) ≥ ½ − ½·√(s·Σ_m n_m (α_m − ᾱ_n)²) ≥ ½ − ¼·√(Σ_m n_m (α_m − ᾱ_n)²).

With n_m = n, the right-hand side is ½ − ¼·√(n·M)·sd(α), where sd(α) is the population standard deviation.

**A necessary condition.** If some test has R(φ) ≤ δ < ½, then:
- h*(β_H) ≥ h_δ := −½·log(4δ(1 − δ));
- hence Σ_m n_m (α_m − ᾱ_n)² ≥ 2h_δ/s ≥ 8·h_δ;
- with n_m = n, sd(α) ≥ √(8h_δ/(nM)).

At δ = 0.05, h_δ = 0.830 and the condition is sd(α) ≥ 2.577/√(nM).

*Proof.*
1. By T5, inf_φ R ≥ ½·(1 − √(1 − e^{−2h*})).
2. Since 1 − e^{−2h} ≤ 2h, this is at least ½ − √(h*/2).
3. At the pivot γ = ᾱ_n, (P3) gives h* ≤ ½·s·Σ n_m δ_m². So √(h*/2) ≤ ½·√(s·Σ n_m δ_m²).
4. With s ≤ ¼ the second form follows. Σ n_m (α_m − γ)² is minimised over γ at ᾱ_n, and equals n·M·sd(α)² when n_m = n.
5. **The necessary condition.** ½·(1 − √(1 − e^{−2h})) ≤ δ is equivalent to e^{−2h} ≤ 1 − (1 − 2δ)² = 4δ(1 − δ), that is, h ≥ h_δ. Then h_δ ≤ h* ≤ ½·s·Σ n_m δ_m². ∎

**Proposition T5.4 (sandwich in the accuracy contrast).** Let Δ_c := x_max − x_min = σ(α_max − β_H) − σ(α_min − β_H). This is H's accuracy contrast at the cell, the design's Δ_c. Then:

- **(a)** For every β_S, h(β_H, β_S) ≥ n_min·Δ_c²/4. Hence h*(β_H) ≥ n_min·Δ_c²/4.
- **(b)** With c, D as in Corollary T5.3, let η := min over |t − c| ≤ D of min(σ(t), 1 − σ(t)). Then h*(β_H) ≤ N·Δ_c² / (8·η²(1 − η)²).

*Proof.*

**(a)** First, a bound for two Bernoulli laws. Let m̄ = (x + y)/2. Since (√x − √y)² = (x − y)²/(√x + √y)²:

1 − BC = ½·(x − y)²·[1/(√x + √y)² + 1/(√(1−x) + √(1−y))²].

Also (√x + √y)² ≤ 2(x + y) = 4m̄, and (√(1−x) + √(1−y))² ≤ 4(1 − m̄). Hence:

−log BC ≥ 1 − BC ≥ (x − y)²/(8·m̄(1 − m̄)) ≥ (x − y)²/2.

The first step uses −log u ≥ 1 − u; the last uses m̄(1 − m̄) ≤ ¼.

Now let model 1 have the largest ability and model 2 the smallest, and write Δ_m := x_m − y_m.
- y_m = σ(β_S − α_m) decreases in α_m, so y_1 ≤ y_2.
- So Δ_1 − Δ_2 = (x_1 − x_2) + (y_2 − y_1) ≥ Δ_c, hence |Δ_1| + |Δ_2| ≥ Δ_c.
- By (u + v)² ≤ 2(u² + v²), Δ_1² + Δ_2² ≥ Δ_c²/2.
- Therefore h ≥ Σ_m n_m Δ_m²/2 ≥ n_min·(Δ_1² + Δ_2²)/2 ≥ n_min·Δ_c²/4.

**(b)**
- At the pivot γ = ᾱ_n, h* ≤ ½·s·Σ n_m δ_m² ≤ ⅛·N·R², where R := α_max − α_min. This uses |δ_m| ≤ R, which holds because ᾱ_n ∈ [α_min, α_max].
- By the mean value theorem, Δ_c = R·σ′(ξ) for some ξ in [c + δ_min, c + δ_max] ⊆ [c − D, c + D].
- On that interval σ(ξ) ∈ [η, 1 − η], and u(1 − u) is smallest at the ends of [η, 1 − η]. So σ′(ξ) ≥ η(1 − η), and R ≤ Δ_c/(η(1 − η)).
- Substituting, h* ≤ N·Δ_c²/(8·η²(1 − η)²). ∎

**So, for fixed M and accuracies bounded away from 0 and 1** (fixed η), T5's exponent is Θ(n·Δ_c²). This is the design's statement that "the readings needed scale as 1/Δ_c²", made precise.

**Remark T5.6 (what the design's T5 sentence gets wrong; departure D1).** The design says "TV is bounded by a function of n·Σ_m (p_{m,H} − (1 − p_{m,S}))²". In the notation here, the quantity is n·Σ_m (x_m − y_m)². As a uniform statement this is false.

**Counterexample.** Take one model with
- x = 1 − e^{−t} and y = 1 − e^{−2t}, so that Δ = e^{−t} − e^{−2t} ≤ e^{−t};
- n = e^{2t} readings.

Then:
- n·Δ² ≤ 1.
- 1 − BC = ½·(e^{−t/2} − e^{−t})² + ½·(√(1 − e^{−t}) − √(1 − e^{−2t}))² ≥ ½·e^{−t}·(1 − e^{−t/2})². So BC^n ≤ exp(−n(1 − BC)) → 0.
- By Lemma 3.2(d), TV ≥ 1 − BC^n → 1.

No function f with f < 1 on bounded arguments can bound this TV. The pair is realisable in the model: by Proposition T5.1, the logits c + δ and c − δ cover any pair of reals.

**The correct statements are weighted:**
- TV ≤ √(½·Σ_m n_m·(x_m − y_m)²/(y_m(1 − y_m))), by Lemma 3.3;
- equivalently, TV ≤ √(n·Σ_m Δ_m²/(2η(1 − η))) when every y_m ∈ [η, 1 − η].

Near saturation a single "rare" reading carries far more information than Δ² suggests. The Hellinger exponent h and the variance-weighted χ² capture this; n·ΣΔ² does not.

### 3.6 The box, and unanimous cells

**Proposition T5.5 (the box and unanimous cells).** Take a binary anchored cell with the statistic of Definition 1.2, the box B = [α_min − b, α_max + b], and N readings.

- **(a) Unanimous deviation: every reading is a.** Then V = (d, a), and
  - ℓ̂({a}) = Σ_m n_m·log σ(α_m − α_min + b);
  - ℓ̂({d}) = Σ_m n_m·log σ(α_max + b − α_m).

  Both lie in [−N·log(1 + e^{−b}), 0]. Hence
  |log_lr| ≤ N·log(1 + e^{−b}) ≤ N·e^{−b},

  **whatever β_g is**. The cell is undetermined for every τ_u > N·e^{−10}, that is, for N < τ_u·e^{10} ≈ 22,026·τ_u.

- **(b) The superseded box β_g ± 6 (ruling T1-L).** On the same data, when {a} is the best defect set:
  log_lr = Σ_m n_m·[log σ(α_m − β_g + 6) − log σ(β_g + 6 − α_m)].

  This depends on β_g. Take the 30-rater population of the code's test (α ~ N(1, 1), seed 20261120, 20 readings each). The formula gives 51.3, 19.5, 6.78, 1.11 and −3.38 at β_g = −2, −1, 0, 1 and 2. That is a confident defect, then undetermined, then a confident "no defect", from the same unanimous data.

  The ledger's values (+8.6 and −2.3) were measured on an earlier code state. The formula here is for the current code with that box substituted. The current box gives 0.0034, which matches the ledger.

- **(c) Unanimous agreement: every reading is d.** a is not observed, so V = (d,), the candidate sets are {d} and ∅, and the pool has k = 2 slots.
  - ℓ̂(∅) = N·log((N + κ/2)/(N + κ)).
  - ℓ̂({d}) = Σ_m n_m·log σ(α_m − α_min + b), which lies in [−N·log(1 + e^{−b}), 0].
  - Hence log_lr lies between N·log((N + κ/2)/(N + κ)) and that value plus N·log(1 + e^{−b}). It tends to −κ/2 = −5 as N → ∞.
  - Up to the O(N·e^{−10}) term, the cell is typed "no defect" exactly when N·log((N + κ)/(N + κ/2)) ≥ τ_u: from N ≥ 2 at τ_u = 1, N ≥ 5 at τ_u = 2, and N ≥ 12 at τ_u = 3.

*Proof.*
- **(a)**
  - On all-a data the {a} likelihood is Σ n_m·log σ(α_m − β) (Lemma 1.1(2)). It decreases in β, since d/dβ log σ(α − β) = −(1 − σ) < 0. So its supremum over B is at β = α_min − b.
  - The {d} likelihood is Σ n_m·log σ(β − α_m) (Lemma 1.1(1)). It increases in β, so its supremum is at β = α_max + b.
  - In both, every argument of log σ is at least b, so every term lies in [log σ(b), 0] = [−log(1 + e^{−b}), 0].
  - Every log-likelihood of a law on a finite set is ≤ 0. So the best defect set satisfies max_A ℓ̂(A) ≤ 0, giving log_lr ≤ −ℓ̂({d}) ≤ N·log(1 + e^{−b}).
  - Also max_A ℓ̂(A) ≥ ℓ̂({a}), giving log_lr ≥ ℓ̂({a}) − 0 ≥ −N·log(1 + e^{−b}).
  - Finally, log(1 + u) ≤ u.
- **(b)** The same monotonicity applies, with the box [β_g − 6, β_g + 6]. The numbers come from evaluating the formula; the check is recorded in the Task 9 report.
- **(c)**
  - With V = (d,), the slots are d and OTHER, k = 2, and ε̂_d = (N + κ/2)/(N + κ). So ℓ̂(∅) = N·log ε̂_d.
  - ℓ̂({d}) = Σ n_m·log σ(α_m − β). It decreases in β, so its supremum over B is at β = α_min − b, and each term lies in [−log(1 + e^{−b}), 0].
  - Subtract. The thresholds come from evaluating N·log((N + 10)/(N + 5)). ∎

**Why the box must be relative to the population (ruling T1-L′).** By Theorem T5.2(iv), unanimous data carry vanishing information about H against S. The only thing that can move a profile statistic on such data is how far each hypothesis is allowed to saturate.
- With a box centred on β_g (part b), the two hypotheses saturate to different floors whenever β_g sits off-centre relative to the population. The statistic then reports where the box sits, not the data.
- With the population-relative box (part a), each hypothesis can push every model to at least σ(b) of its own value, so both saturate to the same floor. The statistic is then O(N·e^{−b}), as T5 requires.

The margin b = 10 is a convention. Its effect is bounded by part (a).

**What the code does with unanimous agreement (part c).** Statistically, by T5.2(iv) at the end β_H → +∞, a unanimous agreement is exactly as undecidable as a unanimous deviation:
- under (A3′), a mis-key {a} in which every model errs onto d (λ_H = 1) produces all-d data;
- so does {d} at high accuracy.

The code does not compare these two. Because a is not observed, {a} is not a candidate, and what remains is ∅ against {d}. That comparison is T2′'s single-model separation through the flat pool, worth at most κ/2 = 5 nats.

So the code types an all-agree cell "no defect" **by its candidate-value convention**, which amounts to a zero prior on unobserved values, and **not by the data**. Small all-agree cells (N below the thresholds of part (c)) are undetermined instead.

---

## 4. T6: upper bounds

### 4.1 T6(a): the oracle likelihood-ratio test

**Theorem T6(a).** Take the simple pair H°, S° of T5, with known (β_H, β_S). Let φ* answer S when P_{S°}(readings) > P_{H°}(readings). Then:

P_{H°}(φ* = S) + P_{S°}(φ* = H) = 1 − TV ≤ BC = e^{−h(β_H, β_S)}.

Together with T5, the minimax error of the two-point problem therefore lies in [¼·e^{−2h}, e^{−h}]. **The exponent h is the right one, up to a factor 2.**

*Proof.* The equality is Lemma 3.1's equality case. The inequality is Lemma 3.2(d) together with (c). ∎

### 4.2 T6(b): contrast rules that need no nuisance (v3's T4(a), per cell)

**Theorem T6(b).** Let w_m := α_m − ᾱ, where ᾱ is the unweighted mean. Let F_m be the fraction of model m's readings equal to a, and T := Σ_m w_m·F_m. The rule answers H when T > 0 and S otherwise. Write V_w := Σ_m w_m²/n_m. Then:

- **Under H.** For every configuration of 𝓗, whatever λ_H and e_H:
  P(rule = S) ≤ exp(−2·G_H²/V_w), where G_H := Σ_m w_m·x_m > 0 whenever the spread is positive.
- **Under S.** For every configuration of 𝓢:
  P(rule = H) ≤ exp(−2·r²·G_S²/V_w), where G_S := Σ_m w_m·σ(α_m − β_S) > 0 and r := ε̃_S(a) ∈ [0, 1]. At λ_S = 1, e_S = a, r = 1.
- **For M = 2 with n_m = n,** both bounds equal exp(−n·Δ_c²), with Δ_c the true configuration's accuracy contrast.

Separately, v3's four-form rule Â = {y : D̂(y) > Δ_c/4} on two models errs with probability at most 4|V|·exp(−n·Δ_c²/32) under (A3′). Lemma T2 holds verbatim, so v3's proof (Hoeffding's inequality and a union bound) applies unchanged with Δ_c per cell.

*Proof.*
1. **The statistic is a sum of independent bounded terms.** T = Σ_m Σ_{i ≤ n_m} (w_m/n_m)·1{Y_{m,i} = a}. Term (m, i) ranges over an interval of length |w_m|/n_m. By Hoeffding's inequality, P(T − E T ≤ −t) ≤ exp(−2t² / Σ_m n_m (w_m/n_m)²) = exp(−2t²/V_w), and likewise for the upper tail.
2. **Under H**, a is the faithful value and errors never equal a. So E F_m = x_m and E T = G_H. Take t = G_H.
3. **Under S**, E F_m = (1 − p_{m,S})·r. Since Σ_m w_m = 0, E T = r·(Σ_m w_m − Σ_m w_m p_{m,S}) = −r·G_S. Take t = r·G_S on the upper tail.
4. **Positivity.** Σ_m (α_m − ᾱ)·x_m = (1/(2M))·Σ_{m,m′} (α_m − α_{m′})(x_m − x_{m′}). Every summand is ≥ 0, since x increases in α, and some summand is > 0 when the spread is positive. The same argument applies to G_S.
5. **M = 2.** w = ±(α_1 − α_2)/2, so G = ½(α_1 − α_2)·Δ_c and V_w = ½(α_1 − α_2)²/n. Hence 2G²/V_w = n·Δ_c². ∎

**Remarks.**
- The rule uses only α. It needs neither β, λ nor e, so its bound is uniform over the nuisances; only the value of G depends on them.
- It is the per-cell form of the classical negative-discrimination check.
- Hoeffding's inequality ignores the variance, so near saturation the rule is loose:
  - by Cauchy–Schwarz, 2G_H²/V_w ≤ 2n·Σ_m (x_m − x̄)² ≈ 2n·σ′(c)²·Σ_m δ_m²;
  - T5's exponent is ≈ ½·n·σ′(c)·Σ_m δ_m²;
  - the ratio, 4σ′(c), vanishes at saturation.

  T6(c) does not lose this factor.

### 4.3 T6(c): the code's profile-likelihood rule

**Lemma 4.1 (Chernoff bound over a grid).** Let P be a law on a finite set 𝒵. Let {Q_θ : θ ∈ [u, v]} be laws on 𝒵, each with total mass ≤ 1, such that for every z with P(z) > 0, θ ↦ log Q_θ(z) is finite and L-Lipschitz. Let G := ⌈(v − u)·L⌉ + 1, let Z ~ P, and let ρ(Z) := sup_θ log Q_θ(Z) − log P(Z). Then for every t ∈ ℝ:
- **(i)** P(ρ ≥ t) ≤ G·e^{½ − t};
- **(ii)** P(ρ > −t) ≤ G·e^{(t + ½)/2}·sup_θ BC(P, Q_θ).

*Proof.*
1. **The grid.** Take θ_j = u + j/L for j = 0, …, ⌈(v − u)L⌉ − 1, together with θ = v. This is at most G points, with spacing at most 1/L, so every θ ∈ [u, v] lies within 1/(2L) of a grid point. By the Lipschitz property, log Q_θ(z) ≤ max_j log Q_{θ_j}(z) + ½. Hence ρ ≥ t implies that some log(Q_{θ_j}/P)(Z) ≥ t − ½, and ρ > −t implies that some log(Q_{θ_j}/P)(Z) > −t − ½.
2. **(i), by Markov.** P(log(Q_j/P) ≥ s) ≤ e^{−s}·E_P[Q_j/P] = e^{−s}·Σ_{P>0} Q_j ≤ e^{−s}. Put s = t − ½ and apply a union bound over the grid.
3. **(ii), by Chernoff with exponent ½.** P(log(Q_j/P) > −s) ≤ e^{s/2}·E_P[(Q_j/P)^{1/2}] = e^{s/2}·Σ_{P>0} √(P·Q_j) ≤ e^{s/2}·BC(P, Q_j). Put s = t + ½ and apply a union bound. ∎

**Theorem T6(c) (the profile-LR rule on a binary cell).** Take an anchored binary cell whose readings lie in {d, a}, with N readings, known α, (A1), and the statistic of Definition 1.2. The truth is either
- H° = ({a}, β_H, 1, d) with β_H ∈ B, or
- S° = ({d}, β_S, 1, a) with β_S ∈ B.

**Constants.**
- G_B := ⌈N·W⌉ + 1, with W = α_max − α_min + 2b.
- L_N := log(1 + 3N/κ), and G_∅ := ⌈2N·L_N⌉ + 1.
- κ_N := N·log((N + κ)/(N + 2κ/3)), and κ′_N := N·log((N + κ)/(N + κ/2)).

**Exponents.** Each is the infimum over a family's parameter range of −log BC between the truth's law and the family's law:
- h*_B(H→{d}) is the inf over β ∈ B of h(β_H, β). It is T5's quantity with β_S restricted to B.
- h*_B(S→{a}) is the inf over β ∈ B.
- h*_B(S→{d,a}) is the inf over β ∈ B.
- h*(S→∅) is the inf over q ∈ (0, 1), against i.i.d. Bern(q) laws on {d, a}.

Then for every τ > N·log(1 + e^{−b}):

- **(i) Confident error under H:**
  P_H(log_lr ≤ −τ) ≤ G_B·e^{½ − τ} + Π_m (1 − x_m)^{n_m}.
- **(ii) Miss under H, either undetermined or wrong:**
  P_H(log_lr < τ) ≤ G_B·e^{(τ + ½)/2}·e^{−h*_B(H→{d})}.
- **(iii) Confident error under S:**
  P_S(log_lr ≥ τ) ≤ (2G_B + G_∅·e^{−κ_N})·e^{½ − τ}.
- **(iv) Miss under S:**
  P_S(log_lr > −τ) ≤ e^{(τ + ½)/2}·[G_B·(e^{−h*_B(S→{a})} + e^{−h*_B(S→{d,a})}) + G_∅·e^{−κ_N/2}·e^{−h*(S→∅)}] + Π_m p_{m,S}^{n_m}·1[κ′_N < τ + N·log(1 + e^{−b})].

*Proof.* There are three kinds of data: both values observed; all a; all d. In the first two, V = (d, a).

**Case 1: V = (d, a).**

*The families.* By Lemma 1.1:
- ℓ̂({a}), ℓ̂({d}) and ℓ̂({d, a}) are suprema over β ∈ B of the log-likelihoods of three families Q_{a,β}, Q_{d,β} and Q_{da,β}. Each is a product over readings, with probabilities σ(α − β), σ(β − α) and σ(α − β)/2.
- The derivative in β of the log of each reading's probability lies in [−1, 1]. So each family is N-Lipschitz on B (L = N, v − u = W, G = G_B).
- For ∅, write s := ε̂_d + ε̂_a = (N + 2κ/3)/(N + κ) and q̂ := ε̂_d/s. Then ℓ̂(∅) = N·log s + N_d·log q̂ + N_a·log(1 − q̂) = −κ_N + log Bern(q̂)^{⊗N}(Z).
- q̂ = (N_d + κ/3)/(N + 2κ/3), so logit q̂ = log((N_d + κ/3)/(N_a + κ/3)) ∈ [−L_N, L_N].
- In θ = logit q, the Bern(q) log-likelihood has derivative N_d·(1 − σ(θ)) − N_a·σ(θ) ∈ [−N, N]. So ℓ̂(∅) ≤ −κ_N + sup_{|θ|≤L_N} log Q_{∅,θ}(Z), with L = N, v − u = 2L_N and G = G_∅.

*The truth lies in its own family.* Since β_H ∈ B, log P_{H°}(Z) = log Q_{a,β_H}(Z) ≤ ℓ̂({a}). Likewise log P_{S°}(Z) ≤ ℓ̂({d}).

- **(i)** log_lr ≤ −τ gives ℓ̂({d}) ≥ max_A ℓ̂(A) + τ ≥ ℓ̂({a}) + τ ≥ log P_H(Z) + τ. That is ρ_{d} ≥ τ. Apply Lemma 4.1(i).
- **(ii)** log_lr < τ gives ℓ̂({d}) > ℓ̂({a}) − τ ≥ log P_H − τ. That is ρ_{d} > −τ. Apply Lemma 4.1(ii); here sup_β BC = e^{−h*_B(H→{d})}.
- **(iii)** log_lr ≥ τ means some defect set A has ℓ̂(A) ≥ ℓ̂({d}) + τ ≥ log P_S + τ. That is:
  - ρ_{a} ≥ τ, or
  - ρ_{da} ≥ τ, or
  - sup_θ log Q_{∅,θ} − log P_S ≥ τ + κ_N.

  Lemma 4.1(i) and a union bound give 2G_B·e^{½−τ} + G_∅·e^{½−τ−κ_N}.
- **(iv)** log_lr > −τ means some defect set has ℓ̂(A) > log P_S − τ. That is ρ_{a} > −τ, or ρ_{da} > −τ, or sup_θ log Q_{∅,θ} − log P_S > −(τ − κ_N). Lemma 4.1(ii) and a union bound give the bracket.

**Case 2: all readings are d,** so V = (d,). By Proposition T5.5(c), log_lr lies in [−κ′_N, −κ′_N + N·log(1 + e^{−b})].

- **Under H.**
  - For (i), the event {all d} has probability Π_m (1 − x_m)^{n_m}; add it.
  - For (ii), on all-d data, sup_{β∈B} log Q_{d,β}(Z) = ℓ̂({d}) ≥ −N·log(1 + e^{−b}) > −τ ≥ log P_H(Z) − τ. So these data already lie in the event {ρ_{d} > −τ} bounded in Case 1, and nothing is added.
- **Under S.**
  - For (iii), log_lr ≤ N·log(1 + e^{−b}) − κ′_N < τ, so the event is empty.
  - For (iv), log_lr > −τ requires κ′_N < τ + N·log(1 + e^{−b}). The event {all d} has probability Π_m p_{m,S}^{n_m}. ∎

**Corollary T6.1 (the rate in Δ_c).** Each of h*_B(H→{d}), h*_B(S→{a}), h*_B(S→{d,a}) and h*(S→∅) is at least n_min·Δ_c²/4, where Δ_c is the true configuration's accuracy contrast between its most and least able models. Hence:

- P_H(log_lr < τ) ≤ G_B·e^{(τ+½)/2}·e^{−n_min·Δ_c²/4};
- P_S(log_lr > −τ) ≤ e^{(τ+½)/2}·(2G_B + G_∅·e^{−κ_N/2})·e^{−n_min·Δ_c²/4}, plus the all-d term of (iv).

*Proof.*
- **H→{d}.** This is Proposition T5.4(a), which holds for every β_S, so in particular on B.
- **S→{a}.** The same argument with H and S swapped. S's P(a) = 1 − p_{m,S} decreases in α, by Δ_c between the extreme models, while the {a} family's P(a) increases. So the per-model gaps satisfy Δ_1 − Δ_2 ≥ Δ_c.
- **S→∅.** The ∅ law Bern(q) is the same for every model. So the gaps Δ_m = P_S(a) − (1 − q) satisfy Δ_2 − Δ_1 = Δ_c. As in T5.4(a), Σ n_m Δ_m²/2 ≥ n_min·Δ_c²/4.
- **S→{d,a}.**
  - Take f = 1{d} − 1{a}, so |f| ≤ 1. Then TV_m ≥ ½·|E_P f − E_Q f| = ½·|2p_{m,S} − 1|, since E_Q f = p/2 − p/2 = 0.
  - Lemma 3.2(a) gives TV² ≤ (1 − BC)(1 + BC) ≤ 2(1 − BC). So 1 − BC_m ≥ (2p_{m,S} − 1)²/8.
  - Between the extreme models, (2p_1 − 1) − (2p_2 − 1) = 2Δ_c. So the two squares sum to at least 2Δ_c².
  - Hence Σ n_m·(1 − BC_m) ≥ n_min·Δ_c²/4, and −log BC ≥ 1 − BC. ∎

**T5 and T6 together.**
- **Exponent h\*.** T5 gives every rule an error of at least ¼·e^{−2h*_B(β_H)} at some point of the pair (H, best mimic in B). T6(c)(ii) gives the code's rule an error of at most G_B·e^{(τ+½)/2}·e^{−h*_B(β_H)}. **The exponents agree up to the factor 2.** The prefactor is polynomial in N and e^{τ/2}.
- **Rate n·Δ_c².** Proposition T5.4(b) and Corollary T6.1 bound the error between
  ¼·exp(−n·M·Δ_c²/(4η²(1 − η)²)) and C_N(τ)·exp(−n·Δ_c²/4).

  The rate n·Δ_c² is the same; the constants differ by M/(η(1 − η))².

On the S side the code's statistic faces more alternatives than T5's pair: it also faces ∅ and {d, a}. The binding term is usually the ∅ one. Table 5 shows this: more S cells than H cells come out undetermined.

### 4.4 How τ_u relates to the bounds

1. **τ_u is a confidence level in nats, and it does not depend on the data.** T6(c)(i) and (iii) bound confident errors by a constant times e^{−τ_u}, uniformly in the nuisance and without needing Δ_c, apart from the all-d term in (i). T6(c)(ii) and (iv) bound misses by a constant times e^{τ_u/2}·e^{−h*}.
   - Raising τ_u trades confident errors for undetermined cells at the exchange rate e^{−τ} against e^{τ/2}.
   - Cells with h* ≲ log G_B + τ_u/2 are where the code will say "undetermined".
   - T5 says that cells with h* < h_δ are undecidable at level δ by any method.
   - Between the two lies a band of width about log G_B + τ_u/2 − h_δ nats. There the code abstains although some method might not need to.
2. **What the design says and what the code does (departure D4).**
   - The design says: "τ_U is set so that T6's error at the cell's estimated Δ_c is at most 0.05."
   - The code does not do this. τ_u is one constant, selected at G0v4 by simulation from {1, 2, 3}, and judged by criterion 1 with the Wilson bound (ruling G0-V).
   - T6's explicit constants cannot certify such a τ_u. At N = 26 and sd(α) = 1 with equal spacing, G_B = 608. Driving bound (i) below 0.05 needs τ ≥ ½ + log(608/0.05) = 9.9 nats; bound (iii) needs 10.6.
   - **The error control of the selected τ_u therefore rests on G0v4, not on T6.** Table 5 shows confident-error rates at τ_u = 2 of at most 2.5% across the illustrated settings.
3. **A heuristic, not proven here.** In regular cases, Wilks' theorem suggests that maximising over the one-dimensional β_c raises the log-likelihood by about ½χ²₁, which is O(1), not by log G_B ≈ log(N·W). The union bound over the grid is the conservative part of Lemma 4.1.

---

## 5. What follows for the method and the paper

### 5.1 Undetermined cells are a property of the information

- T5 bounds **every** function of the readings. That includes majority vote, Dawid–Skene, IRT-style audits, all of the paper's baselines, and the code's own statistic.
- At cells where h*(β_H) is small, no such method can tell a mis-key from a shared error. Those cells are:
  - unanimous ones;
  - those read by a population with little ability spread;
  - those where every model finds the task easy, or every model finds it impossible.
- Saying "undetermined" there is the correct output. Counting such cells, as the anti-goal requires, is reporting the information limit honestly; it is not an admission of weakness.

Specifically:
- **Unanimous deviation is always undetermined in the code,** for N < 22,026·τ_u (Proposition T5.5(a)). This holds for any β_g (ruling T1-L′).
- **Unanimous agreement is typed "no defect" by the candidate-value convention,** worth at most κ/2 = 5 nats, once N ≥ 2, 5 or 12 at τ_u = 1, 2 or 3 (Proposition T5.5(c)). Below those N it is undetermined. The theory treats it as symmetric with unanimous deviation. The paper should say that "no defect" on an all-agree cell is a convention: an unobserved value is not a candidate.
- **Ruling T2-U′** keeps undetermined cells in the α update. This is consistent with T5. A saturated cell's likelihood is nearly flat in α under every candidate set, so it carries little information about α as well as about A_c. Keeping it nudges α little, and excluding it by its observed |log_lr| would select on the outcome.

### 5.2 Why execution witnesses escape the bound

- T5 is a statement about laws of **readings**. An execution witness (design §5.2) adds an observation from outside the reading model: the strict system's verdict on the task's own constraints. Its information is not limited by h*.
- The division of labour is therefore forced, not merely convenient:
  - statistics decide where h* is large;
  - logic, or "undetermined", applies elsewhere.
- A witness resolves the H/S question only for the cells it is applied to. In the code, every undetermined outcome cell on S3 goes to the witness (`diagnose_domain_v4`). If the replay agrees with the anchor, the cell is counted as a "witnessed shared error".
- **Caution.** By Proposition T5.5(c), a small cell where all models agree with the anchor (N below the thresholds) is also undetermined. When its replay agrees, it would be booked as a shared error although no agent deviated. See §7, open point O2.

### 5.3 Population diversity (H5), quantitatively

- **The necessary condition.** By Corollary T5.3, at any accuracy level no method reaches error 0.05 at a cell unless
  sd(α) ≥ 2.58/√(n·M),
  over the models that read that cell. For S3-like settings with one reading per model:
  - 0.51 logits with 26 models;
  - 1.29 logits with 4 models.
- **Near saturation the requirement grows.** By Table 4 it is about 1.3 times larger at p̄ = 0.8 and 2.4 times at p̄ = 0.95.
- **What counts.**
  - It is the spread in **ability at the cell**, not the number of model families.
  - The relevant M is per cell. Tool-error runs are excluded from the readings, so M shrinks.
  - A cell that every model finds easy (or impossible) loses the spread through σ′, whatever the abilities are.
- So H5 becomes a checkable, per-cell condition. A run plan can be judged in advance by Table 4.

### 5.4 Wording for the paper

**Claim:**
- T5 as an application of Le Cam's two-point method to a mis-key/shared-error pair that is new to this setting;
- the closed-form exponent and its characterisation;
- T6(c) as the matching guarantee for the statistic actually used.

**Do not claim:**
- novelty for identifiability (v3's T2);
- that τ_u is derived from T6 (D4).

**Disclose these limits:**
- α is taken as known;
- (A1) is assumed;
- T6(c) is binary-only;
- the box margin b = 10 is a convention.

---

## 6. Numerical illustration

Produced by `.venv/bin/python scripts/v4_bounds_table.py --mc` on 2026-11-25. It runs on the CPU and needs no network.

**Settings.**
- M models with equally spaced abilities, centred at 0, with population standard deviation sd ∈ {0, 0.25, 0.5, 1, 2, 3}.
- n ∈ {1, 2, 8} readings per model.
- H's accuracy at the mean ability is p̄ = σ(ᾱ − β_H) ∈ {0.5, 0.8, 0.95}.

S3 has one reading per run and cell, with about 26 models per domain and n ≈ 1. The calibrated S3 spread (`calibrate_s3` on the development split) cannot be computed on this machine, because the SOPBench runs are not present. The sd columns bracket it.

**Table 1** gives the T5 bound ½·(1 − √(1 − e^{−2h*})), with h* the infimum over β_S ∈ ℝ.

**Table 2** gives the exact two-point bound ½·(1 − TV) for M = 4, at the best β_S found by a search around the Hellinger minimiser. Any β_S gives a valid bound. Table 2 shows that the Hellinger form is conservative by a factor of about 1.5.

**Table 3** gives the per-reading exponent and the contrast.

**Table 4** gives the smallest sd(α) at which the bound falls to 0.05, from Corollary T5.3's closed form and by bisection.

### Table 1. T5 lower bound on the minimax error (Hellinger form)

| M | n | p̄ | sd 0 | sd 0.25 | sd 0.5 | sd 1 | sd 2 | sd 3 |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.50 | 0.500 | 0.377 | 0.267 | 0.110 | 0.010 | 0.001 |
| 4 | 1 | 0.80 | 0.500 | 0.401 | 0.310 | 0.166 | 0.039 | 0.009 |
| 4 | 1 | 0.95 | 0.500 | 0.446 | 0.394 | 0.299 | 0.151 | 0.056 |
| 4 | 2 | 0.50 | 0.500 | 0.329 | 0.188 | 0.040 | 0.000 | 0.000 |
| 4 | 2 | 0.80 | 0.500 | 0.362 | 0.241 | 0.084 | 0.006 | 0.000 |
| 4 | 2 | 0.95 | 0.500 | 0.424 | 0.351 | 0.228 | 0.071 | 0.011 |
| 4 | 8 | 0.50 | 0.500 | 0.187 | 0.036 | 0.000 | 0.000 | 0.000 |
| 4 | 8 | 0.80 | 0.500 | 0.239 | 0.077 | 0.002 | 0.000 | 0.000 |
| 4 | 8 | 0.95 | 0.500 | 0.350 | 0.222 | 0.065 | 0.001 | 0.000 |
| 26 | 1 | 0.50 | 0.500 | 0.212 | 0.054 | 0.001 | 0.000 | 0.000 |
| 26 | 1 | 0.80 | 0.500 | 0.261 | 0.100 | 0.005 | 0.000 | 0.000 |
| 26 | 1 | 0.95 | 0.500 | 0.364 | 0.245 | 0.086 | 0.003 | 0.000 |
| 26 | 2 | 0.50 | 0.500 | 0.128 | 0.010 | 0.000 | 0.000 | 0.000 |
| 26 | 2 | 0.80 | 0.500 | 0.182 | 0.034 | 0.000 | 0.000 | 0.000 |
| 26 | 2 | 0.95 | 0.500 | 0.311 | 0.163 | 0.026 | 0.000 | 0.000 |
| 26 | 8 | 0.50 | 0.500 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 |
| 26 | 8 | 0.80 | 0.500 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 |
| 26 | 8 | 0.95 | 0.500 | 0.161 | 0.023 | 0.000 | 0.000 | 0.000 |

### Table 2. Exact two-point bound ½(1 − TV), M = 4

| M | n | p̄ | sd 0 | sd 0.25 | sd 0.5 | sd 1 | sd 2 | sd 3 |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.50 | 0.500 | 0.403 | 0.313 | 0.171 | 0.039 | 0.003 |
| 4 | 1 | 0.80 | 0.500 | 0.434 | 0.370 | 0.250 | 0.074 | 0.018 |
| 4 | 1 | 0.95 | 0.500 | 0.479 | 0.457 | 0.405 | 0.252 | 0.105 |
| 4 | 2 | 0.50 | 0.500 | 0.362 | 0.241 | 0.085 | 0.005 | 0.000 |
| 4 | 2 | 0.80 | 0.500 | 0.395 | 0.298 | 0.143 | 0.018 | 0.001 |
| 4 | 2 | 0.95 | 0.500 | 0.461 | 0.421 | 0.331 | 0.129 | 0.022 |
| 4 | 8 | 0.50 | 0.500 | 0.240 | 0.080 | 0.003 | 0.000 | 0.000 |
| 4 | 8 | 0.80 | 0.500 | 0.287 | 0.132 | 0.015 | 0.000 | 0.000 |
| 4 | 8 | 0.95 | 0.500 | 0.389 | 0.285 | 0.120 | 0.005 | 0.000 |

### Table 3. Per-reading exponent h₁* and contrast Δ_c (n = 1)

| M | p̄ | quantity | sd 0 | sd 0.25 | sd 0.5 | sd 1 | sd 2 | sd 3 |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.50 | h₁* | 0.0000 | 0.0311 | 0.1229 | 0.4694 | 1.6229 | 2.7913 |
| 4 | 0.50 | Δ_c | 0.000 | 0.166 | 0.323 | 0.586 | 0.872 | 0.965 |
| 4 | 0.80 | h₁* | 0.0000 | 0.0199 | 0.0783 | 0.2943 | 0.9479 | 1.6895 |
| 4 | 0.80 | Δ_c | 0.000 | 0.107 | 0.215 | 0.427 | 0.769 | 0.929 |
| 4 | 0.95 | h₁* | 0.0000 | 0.0059 | 0.0232 | 0.0880 | 0.3344 | 0.7784 |
| 4 | 0.95 | Δ_c | 0.000 | 0.032 | 0.067 | 0.154 | 0.431 | 0.746 |
| 26 | 0.50 | h₁* | 0.0000 | 0.2022 | 0.7978 | 3.0362 | 10.4372 | 18.0925 |
| 26 | 0.50 | Δ_c | 0.000 | 0.205 | 0.394 | 0.682 | 0.931 | 0.987 |
| 26 | 0.80 | h₁* | 0.0000 | 0.1293 | 0.5092 | 1.9144 | 6.1574 | 10.9169 |
| 26 | 0.80 | Δ_c | 0.000 | 0.133 | 0.267 | 0.525 | 0.866 | 0.972 |
| 26 | 0.95 | h₁* | 0.0000 | 0.0384 | 0.1510 | 0.5761 | 2.2032 | 5.0605 |
| 26 | 0.95 | Δ_c | 0.000 | 0.040 | 0.086 | 0.208 | 0.594 | 0.886 |

### Table 4. Smallest sd(α) at which the bound falls to 0.05

| M | n | closed form (any p̄) | p̄ = 0.50 | p̄ = 0.80 | p̄ = 0.95 |
|---|---|---|---|---|---|
| 4 | 1 | 1.289 | 1.362 | 1.835 | 3.094 |
| 4 | 2 | 0.911 | 0.937 | 1.210 | 2.223 |
| 4 | 8 | 0.456 | 0.459 | 0.578 | 1.091 |
| 26 | 1 | 0.505 | 0.510 | 0.643 | 1.212 |
| 26 | 2 | 0.357 | 0.359 | 0.451 | 0.842 |
| 26 | 8 | 0.179 | 0.179 | 0.224 | 0.413 |

### Table 5. The code's log_lr on simulated binary cells (p̄ = 0.8, 400 cells each, tau_u = 2)

| M | n | sd | T5 bound | H: defect | H: undet. | H: no defect | S: defect | S: undet. | S: no defect |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.5 | 0.310 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | 1 | 1 | 0.166 | 0.393 | 0.608 | 0.000 | 0.022 | 0.978 | 0.000 |
| 4 | 1 | 2 | 0.039 | 0.757 | 0.242 | 0.000 | 0.000 | 0.537 | 0.463 |
| 4 | 8 | 0.5 | 0.077 | 0.603 | 0.383 | 0.015 | 0.020 | 0.390 | 0.590 |
| 4 | 8 | 1 | 0.002 | 0.965 | 0.035 | 0.000 | 0.005 | 0.058 | 0.938 |
| 4 | 8 | 2 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.013 | 0.988 |
| 26 | 1 | 0.5 | 0.100 | 0.510 | 0.468 | 0.022 | 0.025 | 0.475 | 0.500 |
| 26 | 1 | 1 | 0.005 | 0.953 | 0.043 | 0.005 | 0.007 | 0.095 | 0.897 |
| 26 | 1 | 2 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.015 | 0.985 |
| 26 | 8 | 0.5 | 0.000 | 0.998 | 0.003 | 0.000 | 0.000 | 0.022 | 0.978 |
| 26 | 8 | 1 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 26 | 8 | 2 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

**How to read Table 5.**
- Each row simulates H cells and S cells at the same accuracy level p̄ = 0.8, and types them with `profile_cell`'s log_lr at the code's default τ_u = 2.
- The T5 column is the lower bound for the pair (H, best mimic). The mimic is an S configuration at accuracy about 1 − p̄, not the S of this table. So the H and S columns are not individually bounded by it. They show the same ordering: where the bound is far from 0, most cells are undetermined.
- Confident errors (H typed "no defect", or S typed "defect") stay at or below 2.5%.

**Reading the tables together.**
- With 26 models and one reading each, a population spread of 0.25 logits leaves every method at an error of at least 26% at p̄ = 0.8 (Table 1). A spread of 1 logit makes the problem decidable in principle (0.5%), and the code then decides about 95% of H cells and 90% of S cells (Table 5).
- With 4 models and one reading each, even a spread of 1 logit leaves at least 17% (Table 1; 25% by the exact bound in Table 2). The code then answers "undetermined" on 61% of H cells and 98% of S cells.
- Saturation (p̄ = 0.95) roughly doubles the spread needed (Table 4).

---

## 7. Departures from design §4, and open points

**D1 (T5, "TV is bounded by a function of n·Σ(p_H − (1 − p_S))²").**
- **The design's claim:** this is false as a uniform statement (Remark T5.6, with a counterexample near saturation).
- **What is proven instead:** the weighted forms:
  - TV ≤ √(½·Σ n_m Δ_m²/(y_m(1 − y_m)));
  - the exact Hellinger exponent h (Proposition T5.1);
  - the sandwich of Proposition T5.4, which makes the dependence on n·Δ_c² exact up to constants when accuracies are bounded away from 0 and 1.

**D2 (T5, "the minimum is 0 exactly when accuracies can be matched by a common shift; that happens when the spread is small relative to 1/√n").**
- **The design's claim:** this mixes two statements, and the scale is wrong.
- **What is proven instead:**
  - The minimum is 0 **exactly** when there is no spread (Theorem T5.2(i)).
  - The infimum over both nuisances is **always** 0, through saturation, so local statements are needed (T5.2(ii) and (iv)).
  - The near-chance regime is sd(α)·√(n·M·s) small, with s = max σ′ near the cell's accuracy level (Corollary T5.3). It is not sd(α)·√n.

**D3 (T6, "the profile-likelihood rule decides with error ≤ C·exp(−c·n·Δ_c²)").**
- **What is proven:** for **binary** cells, which covers all of S3 (Theorem T6(c), Corollary T6.1), with:
  - c = ¼;
  - C = (2G_B + G_∅)·e^{(τ+½)/2}, which is polynomial in N, not an absolute constant;
  - an extra term for the all-d corner.
- **Cells with |V| ≥ 3:** only the contrast rules (T6(b) and v3's T4(a) per cell) are proven. The profile rule there faces more candidate sets and the attractor; that is open point O3.
- **H's error law in T5:** fixed at λ_H = 1, e_H = d. On |V| ≥ 3 cells this pair is not claimed least favourable (see the remark after T5).

**D4 (§5.1, "τ_U is set so that T6's error at the cell's estimated Δ_c is at most 0.05").**
- **What the code does instead:** this is neither what the code does nor something T6's constants can deliver. The code uses one τ_u from {1, 2, 3}, chosen at G0v4 under ruling G0-V. Certifying 0.05 by T6(c) would need τ ≈ 10 nats.
- **Where the error control comes from:** the simulation.

**D5 (unanimous agreement).**
- **The design's claim:** the design speaks of "unanimous or saturated deviation".
- **What is proven instead:**
  - By T5 the agreement end (β_H → +∞) is equally undecidable.
  - The code types it "no defect" by the candidate-value convention once N is large enough (Proposition T5.5(c)). Below that N it is undetermined.

**Open points (reported; not fixed by this task):**
- **O1.** T6 assumes α is exact, but the code plugs in the EM estimate. No bound covers the error from α.
- **O2.** `diagnose_domain_v4` sends **every** undetermined outcome cell to the witness. That includes small all-agree cells (N < 5 at τ_u = 2, N < 12 at τ_u = 3). If the replay agrees, such a cell is counted as a "witnessed shared error" although no agent deviated. A guard such as "at least one reading off the anchor" before the witness would fix the label. On S3 with about 26 raters per cell this cannot happen at τ_u ≤ 3, but it can in small compositions.
- **O3.** T6(c) for cells with |V| ≥ 3 values (S2).
- **O4.** The union bound in Lemma 4.1 costs log G_B ≈ log(N·W) nats. A chaining or Laplace argument should reduce this to O(½·log N), but that is not proven here.

---

## References

- L. Le Cam (1973). Convergence of estimates under dimensionality restrictions. *Annals of Statistics* 1(1), 38–53.
- A. B. Tsybakov (2009). *Introduction to Nonparametric Estimation*. Springer, ch. 2 (two-point lower bounds; Hellinger, total variation and Kullback–Leibler distances; Pinsker's inequality).
- W. Hoeffding (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58(301), 13–30.
- SpecDx v3 design, `docs/superpowers/specs/2026-10-14-specdx-v3-design.md`: §4 and Appendix A (T2, T2′, T4).
- SpecDx v4 design, `docs/superpowers/specs/2026-11-20-specdx-v4-shared-error-design.md`: §3–§5.
