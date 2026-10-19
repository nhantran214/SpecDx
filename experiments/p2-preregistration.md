# P2 (SpecDx v3.1) pre-registration

- **Spec:** `docs/superpowers/specs/2026-10-14-specdx-v3-design.md` (v3.1), a delta on `2026-10-11-specdx-v2-design.md` (v2), itself a delta on `2026-10-06-specdx-design.md` (v1). The v3.1 amendments and their evidence are in its §13 items 8–15 and `results/p2-spikes.md`.
- **Plan:** `docs/superpowers/plans/2026-10-14-specdx-p2a.md`, as amended by the author's rulings F1–F26 on the pre-flight scan (2026-10-15).
- **Rule:** this file is committed after G1 and before G2 (spec v3.1 §9). Nothing in it changes after any P2b result is seen. Two dated additions were made between the freeze and any G2 or G3 run (2026-10-19): the G3 amendment (§5) and the generator disclosure (§10); corrections of wording and figures are in the git history. `tests/test_preregistration_v3.py` pins every named constant below to the code, the values chosen at G0 to §4's table and §8's figures to the committed G0 and G1 records; `tests/test_g0_frozen.py` pins them to `experiments/p2-g0-params.json`.
- **Integrity (spec §8.3):** every threshold and prior was chosen by simulation and base data only (G0, G1). The development mutants are used once, at gate v3. The test split stays untouched. What was seen before G0 and how it shaped G0 is disclosed in §6, the execution-time rulings in §7, the records in §8, the vocabulary diagnosis in §9 and what the calibrated simulation exercises in §10.

## 1. Values unchanged from v1 and v2

| Value | Registered |
|---|---|
| Temperature | 0.7 |
| Tell variants / seeds | 0–2 / 1–3 |
| Budgets (population / probe / localise) | 200 / 100 / 80 |
| Scorability | k = 3 |
| Candidate values per cell | K_max = 6, plus OTHER |
| Hitting-set tolerance / maximum size | τ = 0.05 / 3 |
| Development split | `dev_per_type` 5, `rng_seed` 20261006 |
| Harness nudges per step | 2 |
| Observability | ω = 0.3, m_min = 3 |
| Rasch L2 penalty | λ = 1.0 |
| T-boundary scenarios per specification | at most 12 |
| Workers | at most `OLLAMA_NUM_PARALLEL` |
| Gate rule | contradiction and hidden: SpecDx typed F1 > max(B2, B3) untyped F1; ambiguity and weak: SpecDx typed F1 ≥ 0.4; a type with fewer than 3 scorable mutants is a loss; at least 3 wins of 4; exact-arithmetic F1 |

## 2. Values fixed by spec v3.1

| Value | Registered |
|---|---|
| Entailment sets | \|A_c\| ≤ 2: one value, a pair, or ∅; 22 candidate sets with six values |
| Candidate values (§3.1) | d_c first, then the most frequent readings, ties by the normalised string, K_max = 6 in all; OTHER is never in A_c |
| Normalisation (§2.5) | NFC, trim, whitespace collapsed, casefold, numerals canonicalised; no semantic mapping |
| Row matching (§2.6) | by natural key; a key none of whose golden values occurs in T as a token is executor-generated, and its rows match on the remaining LOCKED foreign-key columns |
| Cell scope (§2.4) | value cells at the steps that draft their table; constraint cells at the steps v2's relevance rule gives |
| Pooling (§2.7) | a lifted golden clause or a cross-cutting clause is one cell across the scenarios where it applies, read in the lifted vocabulary |
| Errors (§3.3) | ε_c ~ Dirichlet(κ·u_c), u_c flat over V_c, κ = 10, fixed |
| Difficulty (§3.2) | p = σ(α_m − β_{g(c)}); g(c) is a value cell's column (pooled over rows and scenarios), an event cell's kind, a constraint cell's clause; β_g ~ N(0, τ²) |
| EM (§3.4) | multi-start from T3's median start and a common α of 0, 1, 2 and 3, keeping the highest log posterior; tolerance 1e-6 on the largest parameter change, at most 200 iterations, non-convergence reported |
| Ambiguity (§5.3) | P(\|A_c\| = 2) ≥ τ_A; no rank-correlation rule and no accuracy-contrast gate; on an anchored cell an ambiguity and its hidden requirement co-occur (§3.1, §5.1, §5.3) |
| Screen and population (§7) | act rate ≥ 0.3, valid-call rate ≥ 0.8, ≤ 2 per family, ≤ 4 in all; at least 2 families and at least 4 acting models, otherwise STOP (G2) |
| G0 (§8.1–§8.2) | at least 10 replicates per arm; τ ∈ {1, 2} in the grid; the regime map (accuracy 0.6–0.95 × Δ 0–0.2 × M 2 and 4, win, tie or loss with a margin of 0.05), 5 per point of the regime map |
| GO criteria | G0–G3 as in §5 below |

## 3. Planning-time resolutions and rulings (P2a plan)

These were settled while planning P2a, before G0 ran, or by the author's rulings F1–F26 on the plan's pre-flight scan (2026-10-15). Each is in the plan's list with its task.

1. **Lifted vocabulary.** A pooled value cell's anchor is the clause's placeholder (`{ordered}`); a reading becomes the placeholder `lift_value` gives in its scenario. When the reading equals several parameters (happy and discount8 order and receive 15), it becomes the anchor's own placeholder if the anchor's parameter is among them.
2. **Booleans.** A bool column's t/f (golden CSVs) and true/false (end states) are one value, as C's own comparison treats them.
3. **Scope of tables no step drafts.** apr and evt rows are written through tools; their cells are read at the steps whose work-order slice keys the row (`apr=APR-0102`, `evt=SO-0031`). Key and matching columns are row identity, not readings.
4. **Readings.** A behaviour reads a cell once, if it acted at any in-scope item; several matching rows give the smallest normalised value.
5. **Difficulty groups.** A value cell's group is its column, pooled over rows and scenarios; an event cell's is its event kind (`evt:DRAFTED`), the event table's analogue of a column; a constraint cell's is its clause.
6. **The M-step.** α and β: r as fractional successes out of q_c = P(A_c ≠ ∅) trials per reading, summed over each difficulty group. ε_c: the exact EM step of the truncated error model (the Task 5 ruling, which amends this resolution; §3.2 draws an error from ε renormalised over V∖A): the posterior mean under Dirichlet(κ·u_c) given all draws (equivalently the mode under Dirichlet(1 + κ·u_c)), ε_c(k) = (counts_k + rejected_k + κ·u_c(k)) / (total + κ). Its effective prior is κ pseudo-counts towards u_c, flat over V_c; the spec's literal MAP under Dirichlet(κ·u_c) would apply κ − K, with K the number of slots, and is not used. The counts are the expected error counts (a reading outside A, or any reading under ∅); under A ≠ ∅ each error reading also rejects ε(k)/(1 − ε(A)) draws on each slot k in A, weighted by the posterior of A.
7. **Refutation of a pooled candidate.** It runs in the scenario with the most acted readings outside d_c, ties to the smaller scenario id.
8. **Singleton hitting sets defer to the cell path.** It is the only source of hidden requirements and of contradictions of size 1; hitting sets of two or more clauses feed multi-clause contradictions. `spent` holds the multi-clause candidates that cost a probe run, or would have, and found nothing.
9. **Defect locations.** Findings reuse v2's location formats, so the matching code is unchanged. A hidden finding's subtype (mis-key, ambiguous-pick, unstated) is informational and never scored.
10. **Provisional parameters.** V3Params' defaults were labelled provisional until G0 froze its choice into the marked block; κ = 10 is a constant outside that block.
11. **Typing without an accuracy contrast, and co-occurrence.** An ambiguity is P(|A_c| = 2) ≥ τ_A for any number of models; the hidden subtype follows the most probable defect form (§5.1's "MAP form"). On an anchored cell an ambiguity and its hidden requirement co-occur (ruling F1): a reported ambiguity brings its hidden candidate (ambiguous-pick, scored max(P(A_c ≠ {d_c}), P(|A_c| = 2))), and a hidden candidate whose most probable form is a pair brings its ambiguity. An unanchored cell reports the ambiguity alone.
12. **The ddmin oracle.** It re-runs every population model with an acted behaviour, highest θ_m first, under each Tell variant at seed 1, and tests P(|A_c| = 2) ≥ τ_A on the re-runs' readings with α, β_{g(c)}, ε_c and V_c held at their fitted values.
13. **G2 on the cached screen.** `specdx screen --from results/p1b-screen.json` applies the current admission rules to the cached measurements, with no run, database or model.
14. **Simulation design.** A defect form and a difficulty belong to a column; truths are pairs (ambiguity), anchored cells with A_c ≠ {d_c} (hidden) and unanchored determined cells (weak); the defect rate is planted at 0.15; the regime map spreads accuracies by Δ around the mean, clipped to 0.51–0.995. Limitation (ruling F24): a simulated difficulty group is a single-form column whose cells differ only by scenario, while a real value column pools rows of different forms and anchoring under one β; G0 does not exercise that case.
15. **Calibration.** Base runs only. Accuracy, the plurality that selects its cells, and the dispersion of logit accuracy are measured on the population's readings; the dispersion is kept within 0.25–3; the error concentration is the nearest of six (0.1, 0.2, 0.5, 1, 2, 5) that reproduces the mean top-error share; the cell mix and |V| are shares of columns, as the generator draws them. Readings per cell follow from the planned budget (⌊200 / (models × 9)⌋ scenarios); the base runs' readings per cell are reported beside the simulated ones. A statistic without data raises instead of falling back (rulings F12, F13).
16. **G0 (iii)'s "population's own M and Δ"** is the regime-map point nearest the calibrated population: its number of models, the grid accuracy nearest its mean accuracy and the grid spread nearest its accuracy range.
17. **θ_m and w_b.** B2 and the refuter's interpreter order use θ_m (§8.3, §11: per-model quantities); w_b weights the hitting sets (§3.5) (ruling F17).
18. **§5.2's V(b).** It is v2 §3.4's acted violation set, over every clause of C, golden values and events included: the literal "constraint cells" would drop the golden-and-invariant conflicts that are the C2 mutants' truths (ruling F18).

## 4. Values chosen at G0

Selection rule (spec §9, fixed in `specdx/sim/g0.py` before G0 ran): the grid π ∈ {0.05, 0.1, 0.2}; (s_mis, s_amb, s_∅) ∈ {(0.6, 0.2, 0.2), (0.4, 0.3, 0.3), (0.2, 0.4, 0.4)}; τ ∈ {1, 2}; (μ_α, σ_α) ∈ {(0, 1.5), (1, 1)}; τ_H ∈ {0.5, 0.7, 0.9}; τ_A ∈ {0.3, 0.5, 0.7}; τ_W ∈ {0.6, 0.8, 0.9}. Ten replicates per arm. The objective is the mean simulated typed F1 over hidden requirement, ambiguity and weak check, at the calibrated parameters (`results/p2-calibration.json`) and under the A3 and A5 arms, equally weighted. Ties go to higher P(|A| = 1) on an anchored cell, then smaller π, then grid order.

G0's criteria are judged on the selection replicates, the same ten seeds per arm that chose the parameters (ruling F22). `results/p2-g0.md` also reports the criteria on the next ten seeds, for information only, and counts the fits that did not converge (ruling F2).

<!-- G0 table: begin -->
| parameter | field | value |
|---|---|---:|
| π | `pi` | 0.2 |
| s_mis | `s_mis` | 0.4 |
| s_amb | `s_amb` | 0.3 |
| s_∅ | `s_empty` | 0.3 |
| τ (β prior sd) | `beta_sd` | 1.0 |
| μ_α | `mu_alpha` | 1.0 |
| σ_α | `sigma_alpha` | 1.0 |
| τ_H | `tau_h` | 0.5 |
| τ_A | `tau_a` | 0.3 |
| τ_W | `tau_w` | 0.9 |
<!-- G0 table: end -->

G0's report is `results/p2-g0.md`; G1's is `results/p2-g1.md`. Both passed. The simulation study at the frozen parameters (§8.1, every sweep and the regime map) is `results/p2-sim.md`. The commands that reproduce G0 and G1, and the commits that produced them, are in `docs/RUNNING.md` §12.

## 5. GO steps and gate v3

| Step | Pass criterion |
|---|---|
| G0 | (i) mean \|θ̂ − θ\| ≤ 0.05; (ii) typed F1 ≥ 0.6 for hidden requirement and ≥ 0.4 for ambiguity; the weak check's ≥ 0.4 is reported for information only and does not gate the verdict (author ruling, option B, 2026-10-15; spec §8.2 and §13 item 15; the gate's own weak rule in §1 is unchanged); (iii) at the calibrated point, under the A3 and A5 arms, and at the population's own M and Δ, v3.1 is never below the best baseline by more than 0.05 on any type, and at the calibrated point it beats the best baseline on ambiguity by more than 0.05. If G0 fails, stop and report |
| G1 | (i) θ_m ≥ 0.5 for gpt-oss:20b and qwen3:30b-a3b; (ii) R-SO reported as a hidden requirement, judged on the P1b smoke run only (author ruling, 2026-10-14), or "not observable" if no acted behaviour read its cell |
| G2 | `specdx screen --from results/p1b-screen.json` admits at least 2 families and at least 4 acting models; the population is frozen into `experiments/p2-gate.json` |
| G3 | the base specification on the 48 GB host, budget 60: every P1b smoke check passes (check (4) is replaced and check (1) is refined, see the amendments below), and θ_m ≥ 0.5 and at least one reading for each population model; convergence is reported, not a criterion (ruling F15) |
| Gate v3 | the 20 development mutants under §1's gate rule; B2 uses the interpreter with the highest θ_m |

A failing GO step stops P2 with a report; no parameter is re-tuned.

**G3 amendment (2026-10-19).** The words "and at least one reading for each population model" were added to the G3 row after this file was frozen and before any G2 or G3 run, mirroring ruling 21 for G1 (i). Reason: a population model with no reading keeps α = μ_α, so its θ is σ(1.0) = 0.731, which passes θ ≥ 0.5 vacuously. P1b smoke check (4) was meant to catch a model that never acts, but it tests |θ − 0.75| < 1e-9, v2's Beta(7, 3) mode, and under v3 it can never fire. It is replaced by a v3 check, in `docs/RUNNING.md` §11, that every population model has readings in the smoke run's cells. The amendment is not outcome-informed (no G2 or G3 data exists) and can only make G3 stricter. A model that never acts at G3 would also leave the gate with fewer than the four acting models G2 requires.

**G3 amendment 2 (2026-10-19): check (1).** P1b smoke check (1) stops P2 when any smoke run holds a cached "ollama call failed: ReadTimeout", because a call that waited in a queue (more concurrent requests than `OLLAMA_NUM_PARALLEL`, or a wait for a model load) would be cached as model behaviour. The check cannot tell such a wait from a model that exhausts the 180 s timeout on its own. On 2026-10-19, after the first amendment, the local smoke run of `run_all.sh local` (16 GB GPU, one worker, so nothing could queue; a scratch cache, part of no registered result) cached one ReadTimeout: the host's Ollama log shows qwen2.5:14b decoding 12,362 tokens at about 69 tokens/s for the whole 180 s, a runaway generation. The P1 cache holds one such run in 6,505. Under the literal check, a single runaway at G3 would stop P2.

Check (1) now reads: every cached ReadTimeout among the smoke run's new runs must be matched by a runaway in the GPU host's Ollama log over the smoke run's window, where a runaway is a request the client cancelled after it had spent at least 90% of the timeout decoding (n_decoded / tg ≥ 0.9 × 180 s, from llama-server's progress lines). A request that waited for a slot or a model decodes little before the cancellation and still fails the check; so does a timeout with no host log to classify it. The rule is mechanical (`specdx/eval/ollama_log.py`, tested on the real 2026-10-19 excerpt). The interpreter settings and run keys are unchanged; a cap on generated tokens was rejected because it would change every run key and invalidate the cached P1b runs.

Unlike the first amendment, this one relaxes G3, and it was prompted by an observed run, though not by any G2 or G3 data: no G3 run has been made. The G2 population had been previewed in a scratch directory the same day; it is deterministic from the cached screen and equals the one §6 already states.

## 6. Population (spec §7) and disclosure

The 0.3 act rate was chosen during design, knowing which cached screen results it admits. The screen is outcome-free: it reads no mutant, truth or diagnosis. Applied to `results/p1b-screen.json` it admits gpt-oss:20b (openai), qwen3:30b-a3b and qwen2.5:14b (qwen) and mistral-small3.2:24b (mistral), four acting models in three families; qwen3:14b (0.407) is ranked out by the per-family cap. The four-model requirement and the other v3.1 amendments were decided after the P2a plan's pre-check and two simulation spikes, before G0, with no mutant seen (spec §13 items 8–14; item 15, the weak-check ruling, came after the previews and is in §6).

**G0 previews (ruling F23).** Before execution began, a full G0 (36 × 27 settings, 10 replicates) ran on the plan's pre-check code: `results/p2-spikes/g0-preview-full.md`, PASS, choosing π 0.1, shares (0.6, 0.2, 0.2), τ 2, (μ_α, σ_α) = (1, 1), τ_H 0.9, τ_A 0.3 and τ_W 0.8, with weak-check F1 0.403. After it, two rulings changed what G0 runs:
- F1: on an anchored cell an ambiguity and its hidden requirement co-occur (§3 item 11);
- F12 (a), (c): the calibration's plurality and dispersion use the population's readings only, and the cell mix is counted per column, a column being anchored if at least half its cells are (§3 item 15).

A second full preview ran on the amended code during the plan's re-check (`results/p2-spikes/amended-preview/g0-preview-amended.md`): FAIL on (ii) weak alone, 0.343 at the calibrated point, with hidden 0.861 and ambiguity 0.837. Sensitivity previews of the same code, full grid each and for information only (`results/p2-spikes/amended-preview/`): F1 alone passes with weak 0.403; F12 (a) alone passes with weak 0.401 per cell; F12 (c) under three rules for a partly anchored column gives weak 0.343 (at least half its cells), 0.306 (the mean anchored fraction) and 0.236 (any cell), so 0.24–0.34 under the per-column rules, each a FAIL. Weak-check F1 differed by 0.084 (amended preview) and 0.088 (G0) between the selection seeds and the next ten, both upward (0.343 to 0.427, then 0.338 to 0.427; spec §13 item 15 rounds this to ±0.08).

**G0's weak-check bar is information only (author decision, option B, 2026-10-15; spec §8.2 and §13 item 15).** After seeing those previews the author chose option B of A/B/C/D: G0's weak-check F1 ≥ 0.4 is reported but does not gate the verdict. The decision was made on 2026-10-15, before G0 ran. Criterion (iii), weak included, and the gate's weak rule are unchanged. The real G0 gave weak 0.338 on the selection replicates (information only, below 0.4) and 0.427 out of sample (above it). Those out-of-sample seeds are the "next ten" whose amended-preview result (weak 0.427) the author had already seen before choosing option B: out of sample for the selection, but not unseen. The bar was relaxed knowing the previews' outcome. Under G0 (ii) as registered before 2026-10-15, G0 fails (weak 0.338 < 0.4) and P2 would have stopped; a reader should weigh G0's PASS accordingly.

**Every G0-altering change made after the first full preview** (`results/p2-spikes/g0-preview-full.md`), none tuning a grid value. F1 and F12 correct the code to the spec and to the plan's own resolution, and were ruled before the amended preview. Tasks 5, 7 and 8 were ruled after the amended preview's weak failure was known, and the last two change weak typing.
- F1: ambiguity and hidden co-occur on an anchored cell;
- F12 (a, c): calibration over the population's members, and per-column cell-mix shares, a column being anchored if at least half its cells are;
- Task 5: the exact EM for the truncated error model, by data augmentation (§3 item 6's M-step);
- Task 7: a cell that reports a hidden requirement or an ambiguity reports no weak check;
- Task 8: the baselines share v3.1's cell-level post-processing.

From the amended preview's weak failure on, every path is outcome-informed: the second full preview ran on the code before Tasks 5, 7 and 8, so it covers F1 and F12 only. No preview ran on the final code. G0 was the first run on it, and it chose different parameters from both previews (π 0.2 against 0.1).

No grid, threshold or seed was changed after either preview. The one change to a criterion is the weak-check bar above.

## 7. Execution-time rulings

These rulings were taken while the plan executed, before G0 or G1 produced a result. Each is in `.superpowers/sdd/2026-10-14-specdx-p2a/progress.md` with its task.

19. **Numerals** are canonicalised only in int, bigint and numeric columns; ids and other text columns keep their form, with NFC, trim, whitespace collapse and casefold (Task 2).
20. **The ddmin localiser** runs only for ambiguities with P(\|A\| = 2) ≥ τ_A; the others are reported unlocalised, because the oracle tests that very bound (Task 10).
21. **G1 (i)** requires at least one reading for each named model; a named model with no reading fails (i), so a prior-only θ_m cannot pass vacuously (Task 12).
22. **A reading is clean** if some in-scope acted item is clean (Task 3).
23. **The anchored prior is renormalised** when a defect form has no candidate set (Task 4).
24. **Singleton hitting sets defer to the cell path**, so a clause with high hitting-set support but a cell posterior below τ_H is not refuted; v2 would have refuted it (Task 9, design note, resolution 8).
25. **Calibration's top error, |V| and reads** use every screened model's readings, including models outside the population (Task 11, kept).
26. **Hidden suppresses weak.** A cell that reports a hidden requirement or an ambiguity reports no weak check (Task 7, mirrored in G0's counts and in the baselines): the deviant reading there is C's own value, the defect is the hidden requirement, and a weak truth is an unanchored cell (§3 item 14). A real weak check at a cell that also holds a hidden requirement goes unreported. This changes weak typing, and it was ruled after the amended preview's weak failure was known (§6).

## 8. The G0 and G1 records

**G0 (`results/p2-g0.md`): PASS** on the selection replicates. Chosen parameters (§4's table, `experiments/p2-g0-params.json`): π 0.2, shares (0.4, 0.3, 0.3), τ 1.0, (μ_α, σ_α) = (1, 1), τ_H 0.5, τ_A 0.3, τ_W 0.9; objective 0.6330. At the calibrated point v3.1 scores hidden 0.858, ambiguity 0.807, weak 0.338 (information only), against the best baseline's 0.807 (anchor-only), 0.714 (majority) and 0.316 (majority); v3.1's margin over the best baseline is 0.051 on hidden and 0.022 on weak; mean |θ̂ − θ| 0.0112; the ambiguity win over the best baseline is 0.092. The worst (iii) gap is A3's weak check, −0.029 against 0.05 allowed. Out of sample (seeds base + 10…19, information only; the same "next ten" seeds the amended preview had shown, §6) the same criteria would pass, weak 0.427.

*Non-convergence (§3.4, reported, never a criterion).* 60 of 1080 search fits did not converge. At the chosen setting 2 of 10 A3 fits and 1 of 10 A5 fits did not; the base arm had none.

*Selection margin.* The chosen setting is the unique maximum of the 972 in `results/p2-g0.json`, but by 8.06e-5 over the runner-up (τ_H 0.7, the rest equal). The top 8 settings lie within 5.5e-4 of each other and share π 0.2, shares (0.4, 0.3, 0.3), τ 1 and τ_A 0.3; only τ_H, the α prior and τ_W vary among them, so the choice among those is close to arbitrary.

**G1 (`results/p2-g1.md`): PASS** on the P1b smoke run (base, 60 runs, 296 cells, 4572 readings). (i) θ_m is 0.989 for gpt-oss:20b and 0.948 for qwen3:30b-a3b; (ii) R-SO (`golden:so:SO-0031:trang_thai`) is reported as a hidden requirement, with P(A_c ≠ {d_c}) = 1.000. The run is typed without refutation 0/53/19/4 (contradictions, hidden requirements, ambiguities, weak checks).

## 9. The vocabulary diagnosis, the R-SO caveat and the gate risk

After G1 the review noted that both models read `draft` at R-SO's cell, not an approved status. The author ordered a read-only diagnosis on the base smoke run only, with no mutant: `results/p2-vocab-diagnosis.md`. Then the author ruled to freeze this pre-registration as it stands and to disclose what the diagnosis found. Findings:
- Of the 53 hidden base findings, 43 lie outside the status columns (15 `apr` cells of rows never written, 8 absent golden events, 16 violated constraint clauses and 4 other value cells); of the 10 status findings, 3 more are `apr` rows never written. Only 4 findings have an out-of-vocabulary MAP value, and only 1 of those (`inv.trang_thai`, `draft` for `nháp`) is a pure translation.
- T itself writes "draft" for SO-0031, INV-0031 and PAY-0031, so the status findings are conflicts between T and C, not measurement artefacts. C declares no status domain.
- A rule that treats out-of-domain readings as errors removes none of the 53 hidden findings, but it does change what is reported. Under the diagnosis's R1 the reported counts (contradictions, hidden requirements, ambiguities, weak checks) go from 0/53/19/4 to 0/54/16/2: one new hidden finding, and fewer ambiguities and weak checks. No such rule is in v3.1, and none is added.
- Vocabulary matters more for the two types with an absolute bar: 4 of the 19 reported ambiguity locations and 2 of the 4 weak-check locations involve out-of-vocabulary status readings. Both types face the absolute 0.4 bar at the gate.

**The R-SO caveat.** No acted run wrote APR-0102: every reading of its `apr` cells and of `evt:APPROVED:APR-0102` is `absent`, for both models. The approval that R-SO is conditioned on (SO stays "Chờ duyệt" after APR-0102 is approved) never happened in the smoke run. So G1 (ii) passes nominally: the readers follow T's "draft" while C has "Chờ duyệt", and SpecDx flags a T↔C status conflict at that cell. It does not pass because the readers inferred "Đã duyệt" after an approval, and it is not evidence that SpecDx recovers the approval-conditioned defect.

**The gate risk.** Mutants may be masked by the base differential wherever the base specification already reports a hidden finding. That holds for the status columns (10 of the 53) and, more widely, for the structural findings: 18 `apr` cells, 8 absent golden events and 16 violated constraint clauses. This is disclosed before gate v3; the rule and the development split are unchanged, and the risk is reported with the gate's results, never mitigated by re-tuning.

## 10. What the calibrated simulation exercises (disclosure, 2026-10-19)

G0 chose the parameters on synthetic specifications from `specdx/sim/generator.py`, configured by `results/p2-calibration.json`. Calibration measures accuracy, act rates, dispersion, error concentration, the cell mix and |V| (§3 item 15). The following constants are not measured; they are fixed in the generator:
- `forms` = (0.4, 0.3, 0.3): the shares of mis, amb and empty among the defects planted on anchored columns;
- `unanchored` = (0.5, 0.2, 0.3): the shares of a determined value, a pair and silence on unanchored columns;
- `kappa_true` = 30: a cell's ε_c ~ Dirichlet(30 · tendency);
- the A3 arm draws each model's errors from a flat Dirichlet with concentration 0.3;
- the defect rate 0.15 (§3 item 14).

The calibrated cell mix is 140 columns, of which 34.3% constraint columns and 36.4% pooled; the anchored share is 0.0326 of the value columns (ruling F12 (c)), so at the calibrated point about 3% of value columns are anchored.

Per specification, over seeds 0–9 of the calibrated configuration (base arm), the planted truths are:
- **hidden:** 32.5 on constraint cells (V = {satisfied, violated}) and 2.4 on value cells, so about 93% of the hidden truths are binary constraint cells;
- **ambiguity:** 58.3 on unanchored value cells and 8.1 on anchored cells (7.6 constraint, 0.5 value).

What this means for reading G0:
- G0 (ii)'s "hidden ≥ 0.6" is evidence mainly about hidden requirements on constraint cells. It barely exercises value and event cells, where 37 of G1's 53 real hidden findings are (53 findings, 16 of them violated constraint clauses, §9; the 37 are 29 value cells and 8 event cells, present or absent). Only the value cells among them have multi-valued vocabularies.
- The ambiguity and weak truth counts are set mostly by the uncalibrated `unanchored` constant, not by calibration.
- The form shares G0 chose (0.4, 0.3, 0.3; §4) equal the generator's planted `forms`. This is not a leak (the shares were selected from a grid of three by simulated F1), but it is a property of selecting a prior by simulation: the selected prior matches the planted one.
