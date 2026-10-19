## Global Constraints

- `vendor/` is never edited. `vendor/tonomyx/` is byte-identical to TonomyX commit `34c1d2ce70583b2de54573d28ba4f49e0af4cc7f`.
- The plan executes in a git worktree on branch `worktree-specdx-p2a`, with its own `.venv` (Task 1).
- **The P1 run cache.** `/home/ethan/AILab/TonomyX-paper/work` is **read-only in P2a**: never write, move or delete anything there.
  - Only Tasks 13 and 14 read it, with `SPECDX_WORK=/home/ethan/AILab/TonomyX-paper/work` prefixed to one command at a time. Never `export` it, so tests never see it.
  - Read-only access goes through `specdx/sim/calibrate.py` and `experiments/p2_replay.py`. They read `runs.jsonl` and extract the Tell PDFs into a temporary directory. They open no `Workbench`, no database and no Ollama, and write only their `--out` file.
  - Before and after any command that reads the cache, check:
    - `md5sum /home/ethan/AILab/TonomyX-paper/work/runs.jsonl`, expected `db1e1823c66df102648c69eb6a8789b3` (6,505 lines);
    - `ls -la --time-style=full-iso /home/ethan/AILab/TonomyX-paper/work | md5sum`.

    A change stops the plan.
  - No command that builds mutants (`specdx mutants`, `specdx gate`, `specdx run --variant <mutant>`) runs in P2a.
- **Database.**
  - Port 55434 only (`postgresql://tonomyx:tonomyx@localhost:55434/tonomyx`); never touch 55432 or 55433.
  - Every DB test is marked `@pytest.mark.db`.
  - One pytest process at a time: the advisory lock refuses a second.
  - The `specdx-postgres` container already runs; never `docker compose up` from the worktree.
- **No LLM calls in P2a**: no Ollama, not even a version or tags query. No test calls a real Ollama.
- **Run keys.** `RUN_SCHEMA_VERSION` stays 3 and `interpreter_settings()` is unchanged (spec §0). The P1b screen and smoke runs stay cache hits; Task 1 pins this.
- **Code.**
  - Python 3.13, typed, `ruff check .` clean at line length 100 (pyproject.toml).
  - Comments are sparse and explain why.
  - Identifiers, comments, tests and documents are in English. Vietnamese appears only in Tell text, clause values and `docs/RUNNING.md`'s quick-start.
- **Parameters.** `V3Params`' defaults are **provisional** until Task 13 freezes G0's choice. Every test whose outcome depends on a parameter value passes explicit `V3Params(...)`. G1 refuses to run on provisional parameters.
- **Pre-registered values (spec v3.1 §9), unchanged from v1 and v2:**
  - temperature 0.7; Tell variants 0–2; seeds 1–3;
  - budgets 200/100/80; scorability k = 3; K_max = 6;
  - the gate rule; dev split (`dev_per_type` 5, `rng_seed` 20261006);
  - harness nudges 2; ω = 0.3, m_min = 3; Rasch λ = 1.0;
  - T-boundary scenarios ≤ 12; workers ≤ `OLLAMA_NUM_PARALLEL`.
- **New, fixed by spec v3.1 §9:**
  - |A_c| ≤ 2; the candidate rule of §3.1; the normalisation of §2.5;
  - the row-matching rule of §2.6; the cell scope of §2.4; pooling of lifted cells (§2.7);
  - EM tolerance 1e-6 and at most 200 iterations;
  - the screen's act-rate threshold 0.3;
  - κ = 10 with a flat pool u_c;
  - difficulty grouped by value column, event kind and clause (resolution 5);
  - multi-start EM (T3's start plus a common α ∈ {0, 1, 2, 3});
  - G0 with at least 10 replicates, and τ ∈ {1, 2} in the grid;
  - at least 4 acting models;
  - the G0–G3 criteria of §8.2.
- **Chosen at G0 by the selection rule, then frozen:** π and the form shares (s_mis, s_amb, s_∅); τ; μ_α, σ_α; τ_H, τ_A, τ_W. κ is not a G0 parameter.
- **Selection rule (§9).** A grid over the free parameters, written in the simulation module before G0 runs. The objective is the mean simulated typed F1 over hidden requirement, ambiguity and weak check, at the calibrated parameters and under the misspecification arms equally weighted. Ties go to the more conservative setting (higher P(|A| = 1), smaller π).
- **GO criteria (spec v3.1 §8.2), copied exactly:**
  - **G0:**
    - (i) mean |θ̂ − θ| ≤ 0.05.
    - (ii) Simulated typed F1 ≥ 0.6 for hidden requirement and ≥ 0.4 for ambiguity. [AMENDED 7b5b645, author option B: weak-check F1 against 0.4 is reported for information only and does not gate; (iii) still includes weak.]
    - (iii) Regime-based: at the calibrated point, under the A3 and A5 arms, and at the population's own M and Δ, v3.1 is never below the best baseline by more than 0.05 on any type; and at the calibrated point it beats the best baseline on ambiguity by more than 0.05.
    - If G0 fails, stop and report.
  - **G1:**
    - (i) θ_m ≥ 0.5 for gpt-oss:20b and qwen3:30b-a3b (no collapse).
    - (ii) The real defect R-SO (v1 §4.4: `SO-0031` stays `Chờ duyệt` after `APR-0102` is approved) is reported as a hidden requirement, judged **on the P1b smoke run only** (author ruling, 2026-10-14: P1's two acted readings at that cell equal the anchor). If no acted behaviour reached the cell, G1 records "not observable" for (ii) and does not fail on it.
    - On failure, stop and report; parameters are not re-tuned.
  - **G2:** at least 2 families and at least 4 acting models admitted, otherwise STOP. **G3:** every P1b smoke check passes, and θ_m ≥ 0.5 for each population model. (Both in P2b.)
- **Gate v3 (§8.3).** The v2 rule, unchanged:
  - contradiction and hidden win when SpecDx's typed F1 exceeds max(B2, B3) untyped F1;
  - ambiguity and weak win when SpecDx's typed F1 is ≥ 0.4;
  - a type with fewer than 3 scorable mutants is a loss;
  - 3 or more wins of 4 pass;
  - exact-arithmetic F1;
  - B2 uses the interpreter with the highest θ_m.
- **Never tune on the development or test split.** P2a runs no mutant.
- **Commands.** Run tests with `.venv/bin/python -m pytest` (pyproject's `addopts = "-q"` prints `N passed`; do not add `-q`). Lint with `.venv/bin/ruff check .`.

## Review Focus

Five conditions the spec implies but the main tests would not exercise. Each has a test in the task that owns the code:

1. **A model with no acted reading, or a cell read by one model only.**
   - The fit must keep the model with no acted reading at the prior mean.
   - A cell read by one model must still be typed: an ambiguity or an unstated subtype needs no accuracy contrast under the flat pool (T2′).
   - *Tests:* Task 5 `test_weights_family_effect_and_a_model_without_readings`; Task 7 `test_an_ambiguity_needs_tau_a_and_a_pair_but_no_accuracy_contrast` and `test_the_subtype_follows_the_most_probable_defect_form_even_for_one_model`; Task 10 `test_a_population_that_never_acts_types_nothing_and_still_reports`.
2. **A pooled lifted cell across scenarios with different parameter values.** This includes happy and discount8, where ordered = received = 15 and `lift_value` returns no placeholder. "Equals this scenario's value of the anchor's parameter" must be one reading everywhere.
   - *Tests:* Task 3 `test_a_lifted_golden_cell_is_one_cell_across_scenarios_in_the_lifted_vocabulary` and `test_lift_reading_keeps_text_and_prefers_the_anchor_on_a_tie`.
3. **Readings that agree, or that are all distinct.**
   - An anchored cell where every reading agrees on one non-anchor value must be a mis-key, not silence.
   - One where every reading is distinct puts the defect mass on ∅.
   - With the flat pool, one model separates all four forms.
   - *Tests:* Task 4 `test_readings_that_all_agree_on_another_value_are_a_mis_key_not_silence`, `test_readings_that_are_all_distinct_put_the_defect_mass_on_silence` and `test_one_model_separates_all_four_forms_when_the_error_pool_is_flat`.
4. **EM that does not converge** must say so in the Report JSON (`converged`, `iterations`), never silently.
   - *Tests:* Task 5 `test_non_convergence_is_reported`; Task 10 `test_non_convergence_surfaces_in_the_report_json`.
5. **Identical input must give a byte-identical fit and report**, whatever the order of readings, so that resume and parallel workers stay exact.
   - *Tests:* Task 3 `test_extraction_does_not_depend_on_the_order_of_behaviours`; Task 5 `test_the_fit_is_deterministic_and_ignores_the_order_of_readings`; Task 9 `test_the_diagnosis_is_deterministic`; Task 10 `test_identical_input_gives_a_byte_identical_report`.

Two further cases are pinned in their owning tasks' main tests:
- executor-generated key detection on token boundaries (`SOL-1` against `SOL-10`): Task 2's `test_a_key_occurs_only_on_token_boundaries` and `test_the_rule_reads_t_only`;
- G2's four-model rule: Task 10's `test_the_screen_stops_without_two_families_and_four_acting_models`.

## Planning-time resolutions (spec v3.1 left these open)

Each is implemented in the named task and recorded in `experiments/p2-preregistration.md` (Task 15).

1. **Lifted vocabulary (§2.7, clarified 2026-10-14).** *(Task 3)*
   - A pooled value cell's anchor is the clause's placeholder (`{ordered}`).
   - In scenario s, a reading becomes the placeholder `lift_value(reading, column type, s)` returns.
   - `lift_value` returns none when a value equals two parameters at once (happy and discount8 order and receive 15). The reading then becomes the anchor's own placeholder if the anchor's parameter is among them, and otherwise stays as normalised text.
2. **Booleans (§2.5).** The golden CSVs write bool columns as `t`/`f`, while end states, read through the typed database, write `true`/`false`. C's own comparison (`clauses._same`) treats them as equal. A bool column's values are canonicalised to `true`/`false`: a type-level canonicalisation like the numeral rule, not a semantic mapping. *(Task 2)*
3. **Scope of tables no step drafts (§2.4).** *(Task 3)*
   - `draft_tables` covers so, soline, po, grn, inv, vinv and pay.
   - apr rows and events are written through tools (`create_approval`, the persona's decision, `log_event`). Their cells are therefore read at the steps whose work-order slice keys the row (`apr=APR-0102`, `evt=SO-0031`).
   - Under the literal rule, no apr or event cell would ever be read. That contradicts §2.2 (golden events are value cells) and §5.4 (weak checks on event cells).
   - Key and matching columns are row identity, not readings.
4. **One reading per behaviour and cell (§3.2).** A behaviour reads a cell if it acted at any in-scope item of its scenario. Several matching rows give the smallest normalised value; `absent` when no row matches. *(Task 3)*
5. **Difficulty groups (§3.2).** *(Tasks 3, 5)*
   - A value cell's group is its column, pooled over rows and scenarios.
   - An event cell's group is its event kind (`evt:DRAFTED`), the event table's analogue of a column.
   - A constraint cell's group is its clause.
   - `Cell.group` holds the group; the v3 ε̄_col pooling it once served is gone.
6. **The M-step (§3.4).** *(Tasks 4–5)*
   - α and β: Newton steps on Σ [r·η − q·log(1 + e^η)] − priors.
     - r is the faithful probability.
     - q_c = P(A_c ≠ ∅) trials per reading: under ∅ no reading is a trial of accuracy.
     - The sums run over each difficulty group.
   - ε_c: the mode of Dirichlet(1 + κ·u_c + expected error counts), that is (counts + κ·u_c)/(N + κ), with κ = 10 and u_c flat over V_c and OTHER.
     - The literal mode of Dirichlet(κ·u_c) is 0 wherever κ·u + count < 1.
     - A zero error mass makes every reading of that value impossible outside A_c.
7. **Refutation of a pooled hidden-requirement candidate** runs in the scenario with the most acted readings outside d_c, ties broken by the smaller scenario id. *(Task 9)*
8. **Singleton hitting sets defer to the cell path**, the only source of hidden requirements and of contradictions of size 1. Hitting sets of two or more clauses feed multi-clause contradictions. *(Task 9)*
   - `spent` (P1b ruling I2) keeps its purpose: the locations that cost a probe run, or would have, and found nothing.
   - In v3.1 a cell candidate always yields a finding, so these are the multi-clause candidates refuted by a probe run or never attempted.
9. **Defect locations reuse v2's formats**, so `specdx/eval/matching.py` keeps working. *(Task 9)*
   - Hidden and contradiction findings sit at the clause id (`golden:po:PO-0015:trang_thai`, `inv:po-15`, `event:DRAFTED:SO-0031:AI`).
   - Ambiguity and weak findings sit at `table.column` or `evt:KIND:REF`; a constraint cell's ambiguity sits at its clause id.
   - Their note is `scenario|step`: the scenario with the most readings (ties to the smaller id) and the cell's first in-scope step.
   - `Defect` gains an informational `subtype` (mis-key, ambiguous-pick, unstated), which is never scored.
10. **Provisional `V3Params` defaults** are labelled as such in the code, and every parameter-dependent test passes explicit parameters. Task 13 rewrites the marked block with G0's choice (`PARAMS_FROZEN = True`). κ = 10 is a module constant outside that block. *(Tasks 4, 11, 13)*
11. **Ambiguity and the hidden subtype need no accuracy contrast (§5.3, T2′).** *(Task 7)*
    - The ambiguity rule is P(|A_c| = 2) ≥ τ_A, and the subtype follows the most probable defect form, for any number of models.
    - A cell with one candidate value has no pair, so it is never an ambiguity.
12. **The ddmin oracle (§5.3)** re-runs, under each Tell variant at seed 1, every population model with at least one acted behaviour in the specification, highest θ_m first. It then asks the loop's test: P(|A_c| = 2) on the re-runs' readings of the ambiguity's cell, with α, β_{g(c)}, ε_c and V_c held at their fitted values, is at least τ_A. *(Task 10)*
13. **G2 (§7).** `specdx screen --from results/p1b-screen.json --out results/p2-screen` applies the current admission rules to the cached screen's measurements, with no run, database or model. The stop rule is fewer than 2 families or fewer than 4 acting models in the population. *(Task 10)*
14. **Simulation design (§8.1).** *(Tasks 6, 8, 11)*
    - A defect form and a difficulty β_g belong to a column (all its per-scenario cells, or its one pooled cell), as a defect belongs to a rule. The within-group arm spreads cell difficulties around the group's by τ_within = 0.5.
    - Truths:
      - weak-check truths are unanchored cells that T determines (|A_c| = 1);
      - ambiguity truths are pairs;
      - hidden truths are anchored cells with A_c ≠ {d_c}.
    - A behaviour is clean at an item when all its anchored readings there equal their anchors. Faithful readings are uniform over A_c.
    - Arms: the A3 arm gives each model its own concentrated error distribution; the A5 arm lets the strongest model read one guessed value with its accuracy at silent cells.
    - The defect rate is planted at 0.15, not calibrated.
    - Regime map:
      - accuracies are spread evenly by Δ around the mean, with every α shifted, each clipped to 0.51–0.995;
      - act rates follow the calibrated population's order;
      - scenarios stay at the calibrated count.
15. **Calibration (§8.1)** reads base runs only. *(Task 11)*
    - Per-model accuracy: measured at anchored cells whose population plurality is the anchor.
    - Dispersion: the standard deviation of those cells' logit accuracy, kept within 0.25–3.
    - Error concentration: the nearest of six Dirichlet concentrations that reproduces the mean top-error share.
    - Act rates come from the screen; the cell mix and |V| (observed distinct values plus one) come from the base runs.
    - The planned budget 200 escalates ⌊200 / (models × 9)⌋ scenarios.
16. **G0 (iii)'s "population's own M and Δ"** is the regime-map point nearest the calibrated population: its number of models, the grid accuracy nearest its mean accuracy, and the grid spread nearest its accuracy range. For the calibrated population that is M = 4, accuracy 0.95, Δ = 0.1. *(Task 11)*

