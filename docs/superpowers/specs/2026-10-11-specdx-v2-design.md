# SpecDx v2: act-then-read diagnosis with observability (design delta)

- **Status:** approved in brainstorming on 2026-10-11, section by section. This is a **delta** on `2026-10-06-specdx-design.md` (v1). Everything in v1 and in its §11 clarifications stays binding unless this document replaces it. Section numbers written "v1 §x" refer to v1.
- **Phase:** P1b, the revision that v1 §4.8 requires after the P1 gate came back NOT PASSED.
- **Evidence:**
  - `results/p1-gate.md` and `results/p1-gate-diagnosis.md` give the verdict and the per-mutant outcome.
  - `results/p1-dev-diagnostics.md` gives the measurements this design answers. Script: `experiments/p1_dev_diagnostics.py`.

---

## 0. Summary

On the 20 development mutants, P1 won none of the four defect types. The diagnostics trace the failure to two root causes that compound each other:

1. **The inference rewards consistent failure.**
   - Competence θ measured agreement, not correctness. `llama3.1:8b` wrote no row in any of its 64 base runs, yet received θ = 0.55. `qwen3:14b`, the model with the fewest violations, received θ = 0.003.
   - Classes mostly encoded step status (`done` against `error`), not readings of T. In 67 of 87 multi-class items, the two most frequent classes differ only in step status.
2. **The population rarely acts.**
   - 99% of golden-state violations are rows that no run wrote.
   - The truth events of the W2 mutants are absent from all 200 runs, on the base and on the mutant alike.
   - The H1 truths sit on rows that no `grn12` run writes.
   - No inference can see a defect on a row that no interpreter reaches.

v2 answers both causes:

- **Population (§2).**
  - A neutral execution harness lets interpreters act through tools.
  - A capability screen admits only models that act.
  - Stronger tool-calling models are added, with hardware profiles for larger GPUs.
- **Inference (§3).**
  - A **two-stage act→read latent-class model** separates *capability failure* (stage 1, Rasch) from *readings of T* (stage 2, set-valued Dawid–Skene on acted behaviours only). θ is anchored by a Beta prior.
  - Diagnosis becomes **observability-aware**. A candidate on rows that the population does not reach is reported as *unobservable*, never as a defect.
  - Ambiguity gains a second signal, **paraphrase sensitivity**: a permutation test of class against Tell variant among competent models.
  - **T-guided scenario synthesis** places scenarios at the numeric thresholds stated in T.
- **Evaluation (§4).**
  - A pre-registered **observability-aware scoring** applies the same criterion to every method, computed from base runs only.
  - The gate is re-run on the same 20 development mutants under the same rule.
- **Infrastructure (§5).**
  - `--workers N` runs independent interpreter runs in parallel, with one database per worker. The report is identical to a sequential run.
  - Portability: remote and multiple Ollama hosts, hardware profiles, `specdx doctor`, `specdx cache-info`, and `docs/RUNNING.md`.

Contributions to the paper, in addition to v1's:

- **C-v2.1.** An act→read latent-class model with an anchored competence prior, which separates capability failure from specification divergence.
- **C-v2.2.** Observability-aware diagnosis, with a non-identifiability result that justifies abstention (Proposition 4).
- **C-v2.3.** Ambiguity detection by paraphrase sensitivity, and T-guided boundary scenario synthesis.

---

## 1. Intent and constraints (additions to v1 §1)

- **Unchanged:**
  - fully automatic on a local machine, with no human experts and no hand-written data;
  - novelty and contribution for a Q1 venue;
  - no tuning on the test split;
  - S1 first.
- **New: portability.** The study must run unchanged on other machines with larger GPUs, or with a remote Ollama server. A step-by-step guide ships with the code (§5.2).
- **New: throughput.** With parallel workers on a large GPU, a gate should take about one day instead of three to four.
- **Pre-registration.** Every threshold and rule in this document is committed in `experiments/p1b-preregistration.md` before the capability screen runs. Nothing in it changes after any P1b result is seen.

---

## 2. Interpreter population (replaces v1 §4.2's selection; extends v1 §2.3)

### 2.1 Neutral execution harness

The harness sits at the executor's chat boundary (`specdx/interpreters/`), and `vendor/` stays untouched. Its texts are fixed constants, identical for every specification, and contain **no business value or rule**.

- **Tool calls written as text.**
  - Applies when a model reply has no `tool_calls` but its content contains a tool call. Accepted forms:
    - a JSON object with `name` and `arguments` (or `function` and `parameters`);
    - optionally fenced in ```` ``` ```` or wrapped in `<tool_call>…</tool_call>`.
  - The parsed call is used only if its name is one of the step's tools and its arguments validate against that tool's schema.
  - It is then executed like a native call and marked `parsed_from_text`.
- **Act before done.**
  - Applies when a model ends a step with a final answer, the step has made **no successful write call**, and its work order drafts at least one table.
  - The harness then appends this user message, at most 2 times per step:

    > `NUDGE = "Bạn chưa thực hiện hành động nào bằng công cụ. Nếu còn việc phải làm, hãy dùng công cụ; nếu thật sự đã xong, hãy trả lời lại 'xong'."`

  - This only restates rule 1 of the standing instruction. The model remains free to confirm that it is done.
- **Recording.** `Behaviour` records the per-step `nudges` count and the number of `parsed_from_text` calls. They are observations, never part of a class key.

### 2.2 Capability screen (automatic, outcome-free)

- **Purpose.** It decides which installed candidate models enter the population. It measures **capability to act**, not agreement with C. Scoring against C would be circular when the specification under diagnosis is itself defective.
- **Protocol.** Each candidate model runs the base specification, canon scenario `happy`, Tell variants {0, 1, 2} × seeds {1, 2, 3}: 9 runs through the harness.
- **Metrics:**
  - *act rate*: the fraction of interpreter steps (kinds `de` and `persona`) with aᵦₛ = 1, as defined in §3.1;
  - *valid-call rate*: the fraction of tool calls, native or parsed, that validate against the tool schema.
- **Admission:** act rate ≥ **0.6** and valid-call rate ≥ **0.8**.
- **Population.** Rank the admitted models by act rate, breaking ties by valid-call rate and then by name. Keep at most **2 per family** and at most **M_max = 4** in total. The population must contain at least **2 families** (v1 §4.8).
- **Stop rule.** If fewer than 2 families are admitted, P1b stops and reports. The thresholds are never lowered.
- **Persistence.** Screen results are written to `results/p1b-screen.md` and `results/p1b-screen.json`. The chosen population is frozen into the gate config.

### 2.3 Candidate models and hardware profiles

Profiles live in `experiments/profiles/{gpu16,gpu24,gpu48,gpu80}.json`. Each lists candidate models by family, a suggested `workers`, and a suggested `OLLAMA_NUM_PARALLEL`. A profile only widens the candidate list; the screen still decides admission, and no rule or threshold depends on the profile.

| Profile | Candidates (family: models) |
|---|---|
| gpu16 | qwen: qwen3:14b, qwen2.5:14b, qwen3:30b-a3b · openai: gpt-oss:20b · mistral: mistral-small3.2:24b · ibm: granite3.3:8b · meta: llama3.1:8b |
| gpu24 | gpu16, plus qwen3:32b and google: gemma3:27b |
| gpu48 | gpu24, plus meta: llama3.3:70b (q4) and qwen2.5:72b (q4) |
| gpu80 | gpu48, plus gpt-oss:120b |

Model tags are examples current on 2026-10-11. `specdx doctor` (§5.2) checks that each one is present and supports native tool calling (v1 §4.2), and records its digest.

---

## 3. Inference (replaces v1 §3.2; amends v1 §3.3–§3.5, §3.7, §3.9, §3.10)

### 3.1 Items and the act indicator

An item s is a pair (scenario, step), as in v1, restricted to interpreter steps (kinds `de` and `persona`; `env` steps are not interpreter behaviour).

For behaviour b and item s, **aᵦₛ = 1** when both hold:
- the step ended with status `done`;
- if the step's work order drafts tables, the final state holds at least one row or event in the step's slice that the initial state did not hold.

Rows are attributed to steps by the work-order `slice`. A row in a slice shared by several steps counts for each of them. In every other case aᵦₛ = 0 (error, timeout, refusal, or no write).

### 3.2 Stage 1: act (Rasch)

- **Model:** P(aᵦₛ = 1) = σ(κₘ − δₛ), where κₘ is model m's capability to act and δₛ is item difficulty.
- **Fit:** penalised maximum likelihood with an L2 penalty λ = **1.0** on κ and δ, and mean δ fixed at 0. Newton iterations run until the change is below 1e-6, with at most 100 iterations.
- **Observability:** Oₛ = mean over the population of σ(κₘ − δₛ).
- **Observable item:** Oₛ ≥ **ω = 0.3** and at least **m_min = 3** behaviours with aᵦₛ = 1.

### 3.3 Stage 2: read (set-valued Dawid–Skene on acted behaviours)

- **Classes:** the projection of the behaviour onto the **locked fields** of the step's slice (v1 §2.2 column model). Step status is **not** part of the class key.
- **Generation.** The v1 form is kept: P([b] = k | Aₛ, θₘ) = θₘ·w_{s,k}·𝟙[k ∈ Aₛ] + (1 − θₘ)·ε_{s,k}. The family random effect, K_max = 6 and P(|A| = 1) = 0.9 are kept as well.
- **Data:** only behaviours with aᵦₛ = 1 enter stage 2. The others contribute nothing.
- **Anchor:** θₘ has a **Beta(7, 3)** prior. The M-step is MAP: θₘ = (S_m + α − 1) / (n_m + α + β − 2), where S_m is the expected number of admissible behaviours and n_m the number of acted behaviours.
- **Outputs:**
  - θₘ;
  - P(k ∈ Aₛ);
  - P(|Aₛ| ≥ 2);
  - each acted behaviour's admissibility weight, P([b] ∈ Aₛ) × θₘ (v1 §3.3).

  A non-acted behaviour has weight 0.

### 3.4 Violation sets and hitting sets (amends v1 §3.3)

- V(b) holds only the clauses that b violates **and** that are relevant to an item where aᵦₛ = 1 (relevance as in v1 and §11 item 4, plus the Task 10 and final-review rulings).
- A clause relevant only to items where b did not act is *missing* for b, not violated.
- Soft hitting sets are computed exactly as in v1, over these V(b) with the §3.3 weights.

### 3.5 Typing with observability (amends v1 §3.4)

A candidate is typed only if **every** clause in it is relevant to an observable item (§3.2). Any other candidate goes to the **Unobservable** list. That list is reported, but it is not a defect type and it is never scored as a TP or FP.

- **Contradiction:** as in v1 and §11 item 13. Refutation was attempted and failed.
- **Hidden(c):** a singleton that was refuted (or whose refutation was never attempted, §11 item 13), **and** whose Hidden(c) score is at least **τ_H = 0.5**. P1 reported 68 of 78 hidden findings with a score of 0.
- **Weak:** as in v1, and in addition the item must be observable and its MAP set must contain a clean admissible class.

### 3.6 Ambiguity: two signals (amends v1 §3.4 Ambig and §3.7)

Both signals are computed per observable item.

- **S1, set-valued split.** S1 = P(|Aₛ| ≥ 2) from stage 2. It covers all acted behaviours, weighted by θ as stage 2 does.
- **S2, paraphrase sensitivity.** It covers only the acted behaviours of **competent** models (θₘ ≥ **0.5**).
  - **Statistic:** the plug-in conditional mutual information I(K; V | M) between class K and Tell variant V, stratified by model M.
  - **Permutation:** permute V within each model stratum, R = **2000** times with a seeded RNG.
  - **p-value:** pₛ = (1 + #{I_perm ≥ I_obs}) / (R + 1).
  - **Eligibility:** an item is tested only if it has at least 6 such behaviours, at least 2 classes, and at least 2 variants.
  - **Multiple testing:** Benjamini–Hochberg across the tested items of one specification, at **q = 0.1**.
- **Report.** An ambiguity is reported when S1 ≥ 0.5 **or** S2 is significant after the BH correction.
  - Score: max(S1, 1 − pₛ).
  - Location: the fields that differ between the classes (v1 rule).
  - Identity for merging and for the differential: (step, fields).
- **ddmin oracle** (replaces v1 §3.7's oracle).
  - **Re-run:** one behaviour per (competent model, variant) at seed 1.
  - **Pass:** the test passes while that re-run still yields at least 2 distinct classes on the ambiguity's fields.
  - **Budget and outcome:** `localize_budget` and the unlocalised rule (§11 items 9 and 13) are unchanged.

### 3.7 T-guided scenario synthesis (amends v1 §3.5)

- **Extraction** runs on the Tell corpus *of the specification being diagnosed*, so a mutant uses its own T.
  - **Pattern:** a number, with an optional unit (`%`, `phần trăm`, `đơn vị`, `cái`, `chiếc`), within ±40 characters and in the same sentence as a comparison keyword.
  - **Vietnamese keywords:** tối đa, tối thiểu, không vượt quá, vượt quá, vượt, trên, dưới, ít nhất, nhiều nhất, trần, ngưỡng, lớn hơn, nhỏ hơn.
  - **English keywords (S2):** at most, at least, exceed, above, below, maximum, minimum, cap, threshold, more than, less than.
- **Binding to parameters:**
  - a percentage binds to `discount_pct`;
  - a number near a quantity word (số lượng, đơn vị, units, quantity) binds to `received`;
  - anything else is dropped.
- **Domain.** Every bound literal v adds v − 1, v and v + 1 to that parameter's domain in `boundary_domain`, inside the physical limits that `enumerate_scenarios` already enforces.
- **Queue.** The new **T-boundary scenarios** vary one parameter from the anchor, at most **12** per specification. They are ordered by source, chunk id and offset, and enter the queue after the canon scenarios and before bandit synthesis. The bandit's candidate set includes the extended domain.
- **Report:** the extracted literals are written per specification, for audit.

### 3.8 Main loop (amends v1 §3.8)

- **Queue:** canon scenarios, then T-boundary scenarios, then the Thompson-sampling bandit, as in v1.
- **Per iteration:** pilot, then escalation, then stage 1 and stage 2, then diagnosis (§3.4–§3.6).
- **Unchanged:** budgets, escalation and the differential, including §11 item 13's ordering of base locations.
- **Report additions:**
  - κₘ, θₘ and Oₛ per item;
  - the Unobservable list;
  - per-item S1, S2 and pₛ;
  - the extracted T literals.

### 3.9 Theory (adds to v1 §3.9)

- **Proposition 2′ (identifiability under abstention).** Assume:
  - (i) acting is independent of the reading given (κₘ, δₛ), i.e. missing at random given capability and difficulty;
  - (ii) on acted behaviours, mean family competence is at least ½ + γ;
  - (iii) n_f^act(s) families have at least one acted behaviour at s.

  Then the probability of misclassifying Aₛ is at most exp(−2·n_f^act(s)·γ²). Observability therefore enters the rate through n_f^act(s).
- **Proposition 4 (non-identifiability without action).** If no acted behaviour touches the rows that clause c reads, then the distribution of every behaviour-based observation is the same under "c is a hidden requirement" and under "c is correct but unreachable by P". So no behaviour-based method, including B2 and B3, can tell them apart, and abstention (*Unobservable*) is the only calibrated output.
  - *Proof sketch:* the observations do not depend on c's truth value when c's rows never appear.
  - This motivates both §3.5 and the evaluation's scorability criterion (§4.1).
- **Risk to assumption (i).** Ambiguous text may itself cause abstention. The report therefore includes δₛ per item, and the per-mutant output includes the base-to-mutant δₛ shift, so a violation is visible.

### 3.10 Pre-registered hyper-parameters (adds to v1 §3.10)

| Parameter | Value |
|---|---|
| Harness nudges per step | 2 |
| Screen: runs / act rate / valid-call rate | 9 / ≥ 0.6 / ≥ 0.8 |
| Population: per family / total / minimum families | ≤ 2 / ≤ 4 / ≥ 2 |
| Rasch L2 penalty λ | 1.0 |
| Observability ω / m_min | 0.3 / 3 |
| θ prior | Beta(7, 3) |
| Competent model for S2 | θₘ ≥ 0.5 |
| Hidden threshold τ_H | 0.5 |
| S2: permutations / minimum behaviours / BH q | 2000 / 6 / 0.1 |
| T-boundary scenarios per specification | ≤ 12 |
| Scorability k (§4.1) | 3 |
| Budgets (population / probe / localise) | 200 / 100 / 80 (unchanged) |

---

## 4. Evaluation (amends v1 §4)

### 4.1 Observability-aware scoring (pre-registered)

- **Truth targets.** Every mutation operator declares `observed_at`, the rows, fields or steps its truth needs. This is benchmark metadata and is never an input to any method.

  | Operator | `observed_at` |
  |---|---|
  | C1 | the rows the pair of invariants reads |
  | C2 | the golden clause's row |
  | H1 | the golden row |
  | H2 | the event kind and reference |
  | A1, A2, A3 | the step(s) the edited rule governs, declared per site |
  | W1 | the blanked field's row |
  | W2 | the event kind and reference |

- **Scorable mutant.** A mutant is scorable if, in the **base** specification's runs, at least **k = 3** acted behaviours touch `observed_at`. "Touch" means:
  - for a row, field or event: an acted behaviour at an item whose slice covers it, whose final state holds it;
  - for a step: an acted behaviour at that step.
- **Fairness.** The scorable set is computed once from base runs, before any mutant runs. It is identical for SpecDx, B2 and B3, and it is written to the results.
- **Tables:**
  - the **primary** table scores the scorable mutants only;
  - a **secondary** table scores all 20, reports the number excluded per type, and keeps the literal typed-against-typed table (§11 item 12).

### 4.2 Gate v2 (replaces the P1 gate run; v1 §4.8 is otherwise kept)

- **Mutants:** the same 20 development mutants (the same `dev_split` seed); the test split is untouched.
- **Rule** (§11 item 12, unchanged):
  - contradiction and hidden: SpecDx typed F1 > max(B2, B3) untyped F1, in exact arithmetic;
  - ambiguity and weak: SpecDx typed F1 ≥ 0.4;
  - at least 3 wins out of 4.
- **New:** a type with **fewer than 3 scorable mutants counts as a loss**.
- **Baselines:** B2 and B3 keep their v1 §4.5 definitions and run on the **same population and the same runs**. B2's "best interpreter" is the one with the highest stage-2 θ.
- **Outcome:** PASSED, NOT PASSED or INCOMPLETE (as in P1). After a second NOT PASSED, P1b stops and reports. A third iteration on the development split needs the author's decision.

### 4.3 Ablations and hypotheses (additions for P3)

| Id | Ablation |
|---|---|
| A8 | No stage 1: v1 classes, which include step status |
| A9 | No capability screen: every installed tool-capable model |
| A10 | No harness: no text-call parsing and no nudges |
| A11 | Ambiguity by S1 only (no S2) |
| A12 | No T-guided scenarios |
| A13 | No observability gating: Unobservable candidates are typed as in v1 |

- **H6.** Observability gating (A13) lowers base-specification false positives without lowering typed recall on scorable mutants.
- **H7.** The harness and the screen (A9, A10) raise the fraction of scorable mutants.

---

## 5. Execution infrastructure

### 5.1 Parallel workers (`--workers N`, default 1)

- **Databases.** There are N PostgreSQL databases `tonomyx_w0 … tonomyx_w{N−1}` in the same server, created on demand. Each worker holds its own advisory lock and its own `PackDatabase`, so per-run isolation is unchanged.
- **What runs in parallel:** the independent runs of one loop step, that is, the pilot, committee escalation, DirectedProbe attempts for one candidate, and the runs of one ddmin test. Inference stays sequential.
- **Determinism:**
  - budgets are charged before dispatch;
  - results are collected in **submission order**;
  - a single writer appends to the run cache.

  The report is byte-identical to N = 1, and a test enforces this.
- **Model grouping.** Runs are grouped by model within a batch, to limit model swaps.
- **Ollama hosts.** `SPECDX_OLLAMA_URL` accepts a comma-separated list. Workers are assigned to hosts round-robin. Every host must report the same digest for each model tag; `doctor` checks this, and a mismatch is an error.

### 5.2 Portability

- **Endpoints and paths** come from the environment, as now: `SPECDX_OLLAMA_URL`, `SPECDX_DATABASE_URL`, `SPECDX_WORK`. SpecDx and PostgreSQL can therefore run on a CPU machine against a remote GPU host.
- **The run cache is portable.** Keys are content digests plus the Ollama model digest, so copying `work/` to another machine reuses every run whose model digest matches.
- **`specdx cache-info`** prints the cached runs per model and digest, and per specification.
- **`specdx doctor`** checks:
  - the Python version and packages;
  - PostgreSQL reachability, and that the port is not 55432 or 55433;
  - the vendor integrity check (`scripts/verify_vendor.sh`);
  - Ollama reachability for every host, model presence, native tool calling, and digest agreement across hosts;
  - free disk;
  - VRAM, read from `nvidia-smi` when it is local.

  It prints a recommended profile and the commands to pull the missing models.
- **`docs/RUNNING.md`** is in English with a Vietnamese quick-start. It covers:
  - prerequisites (Python 3.13, Docker, Ollama, poppler);
  - setup;
  - choosing a profile;
  - the remote or multi-GPU Ollama setup, including `OLLAMA_NUM_PARALLEL ≥ N`;
  - the screen;
  - the gate;
  - pause and resume (`kill -STOP` / `kill -CONT`; never start a second process on the same database);
  - moving the cache;
  - reading results;
  - troubleshooting.

---

## 6. Implementation notes (amends v1 §6)

- **New modules:**
  - `specdx/interpreters/harness.py`: text-call parsing and the nudge constant;
  - `specdx/infer/act.py`: the act indicator and the Rasch fit;
  - `specdx/infer/sensitivity.py`: S2;
  - `specdx/scenarios/tguided.py`: literal extraction and binding;
  - `specdx/eval/observability.py`: `observed_at` and scorability;
  - `specdx/eval/screen.py`: the capability screen;
  - `specdx/parallel.py`: the worker pool and database pool;
  - `specdx/doctor.py`;
  - `experiments/profiles/*.json`;
  - `experiments/p1b-preregistration.md`;
  - `experiments/p1b-gate.json`;
  - `docs/RUNNING.md`.
- **Changed modules:**
  - `infer/latent.py` (stage 2 data and the Beta prior);
  - `infer/items.py` (classes without status; act-filtered violation sets);
  - `infer/scores.py` (observability typing, τ_H, S1 and S2 combination);
  - `active/loop.py` (the queue, the report fields, parallel dispatch);
  - `active/ddmin.py` (the new oracle);
  - `scenarios/space.py` (the extended domain);
  - `mutate/operators.py` (`observed_at`);
  - `eval/pilot.py` (screen, scorability, the new tables);
  - `cli.py` (`screen`, `doctor`, `cache-info`, `--workers`, `--profile`);
  - `observe/run.py` (Behaviour fields; `RUN_SCHEMA_VERSION = 3`, because the harness changes what interpreters see).
- **Tests (additions to v1 §6.4):**
  - Rasch parameter recovery on synthetic data;
  - the inversion regression: a synthetic population with a consistently non-acting model, shaped after the P1 measurements, cannot pull θ of acting models below the prior mean;
  - observability gating;
  - S2 controls type-I error under a synthetic null (rejection rate ≤ q + tolerance) and has power on a planted variant effect;
  - T-literal extraction on Vietnamese and English fixture sentences;
  - the harness: parsing every accepted text form and rejecting invalid names and arguments; nudge count and message;
  - workers: N = 1 and N = 3 reports are byte-identical with a scripted executor;
  - `doctor` and `cache-info` with mocked endpoints;
  - scorability on the real base pack with scripted behaviours.
- **Offline replay check, before any new run:** run the v2 inference on the P1 cache (old population). It uses no GPU and is a **mechanism check** (author's ruling, 2026-10-12). It holds when:
  - (a) on the base and on every development mutant, stage 1 gives the model that never writes (llama3.1:8b) the lowest κ;
  - (b) every model with acted behaviour keeps θ ≥ 0.5, so there is no collapse like P1's 0.003, and at least one model acted;
  - (c) at least 50% of the P1 base findings become Unobservable.

  A θ ordering between models is not judged on the P1 cache: only one family writes there, and Proposition 2′ needs two. That ordering is checked on the screened population after the smoke run, before gate v2. A failure stops P1b before the gate.

---

## 7. Risks and mitigations (additions to v1 §7)

| Risk | Mitigation |
|---|---|
| Nudges push action where inaction is correct | The nudge allows confirming "xong"; nudges are recorded; ablation A10 |
| No two families pass the screen on 16 GB | Profiles for larger GPUs; the stop rule never lowers thresholds |
| Ambiguity causes abstention (violates MAR) | δₛ is reported, with base-to-mutant shifts in the per-mutant output |
| Regex extraction misses thresholds | Extracted literals are reported per specification; ablation A12 |
| Parallelism changes results | Submission-order collection, a single cache writer, and a byte-identical test |
| Scorability is seen as cherry-picking | Pre-registered, computed from base runs only, shared by all methods; the all-mutant table is always reported |

---

## 8. Decisions log (2026-10-11, brainstorming)

1. Fix **both** root causes, inference and population.
2. Population: neutral harness, outcome-free capability screen, stronger models. No neutral playbook.
3. Inference: PA1, the act→read model with an anchored prior and observability, plus PA3, ambiguity by paraphrase sensitivity.
4. Evaluation: observability-aware scoring and T-guided scenarios. Operators A and W are **not** redesigned.
5. Portability: runnable on larger GPUs and on remote Ollama, with `docs/RUNNING.md`.
6. Parallel workers are included in v2.

## 9. Delivery (amends v1 §9)

| Phase | Content | Done when |
|---|---|---|
| P1b | §2, §3, §4.1–§4.2, §5 and §6; pre-registration committed; offline replay check; screen; gate v2 | Gate v2 is PASSED, or a reported NOT PASSED |
| P2 | v1 P2 (S2: τ²-bench), on v2 | As in v1 |
| P3 | v1 P3, plus ablations A8–A13 and hypotheses H6–H7 | As in v1 |
