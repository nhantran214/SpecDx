# SpecDx P3 design: re-scoping to τ²-bench and SOPBench

**Date:** 2026-11-19 · **Status:** draft for the author's approval · **Supersedes for the paper:** the subject, evaluation and delivery parts of v1 (§1.3, §4, §9), v2, v3 (§8, §9, §14) and the baselines design (subjects and splits). The method (v3 §2–§5: cells, the anchored cell model, theory, typing) and the baseline implementations are kept.

---

## 1. Intent and why

SpecDx diagnoses defects in an executable business specification S = ⟨T, C, Π⟩ by running a population of LLM interpreters and separating specification defects from agent errors. It types each defect (contradiction, hidden requirement, ambiguity, weak check) with a location and witnesses.

On 2026-11-19 the author re-scoped the paper for ACM TOSEM. The author's own words, in order:

1. Stop experimenting on S1, TonomyX CASE-0031: it belongs to the FAA project's scope.
2. Experiment only on S2 = τ²-bench (with τ²-Bench-Verified) and S3 = SOPBench.
3. Results must be comparable and assessable against published methods and studies.
4. The paper must prove SpecDx's novelty and difference from prior papers.

Two further choices were made by the author in the conversation:

- **Comparison style.** Option A is the main result: every baseline is re-run by us on the same data, with the same local models, the same runs and the same scoring. A secondary table of type B compares, per real defect, with what published audits reported, without re-running them.
- **Build approach.** Approach 3 (§4): all primary runs are local, plus a cheap replication of SpecDx's inference on the trajectories the benchmarks themselves published.

**Why this re-scope.** It separates SpecDx from FAA completely, and it puts SpecDx on the benchmarks its closest competitors use, so that reviewers can compare like with like. It also answers the single-subject threat that the current draft carries.

## 2. Goals, non-goals, anti-goals

**Goals**

- G1. SpecDx runs end to end on S2 and S3 through subject adapters, with no CASE-0031 code on the experimental path.
- G2. Real-defect evaluation on S2: SpecDx rediscovers the defects that τ²-Bench-Verified repaired, scored by a pre-registered, mechanical mapping (§6).
- G3. Seeded-defect evaluation on S2 and S3: mutants of every type, a development split for tuning and a test split run once (§7).
- G4. A head-to-head comparison with published methods, all re-run under identical conditions (§9.1), plus the type-B table (§9.2).
- G5. A replication of SpecDx's inference on the published trajectories (§8.4), showing how the result depends on population composition, frontier models included.
- G6. Evidence of novelty: a positioning table, ablations of each new component, and the theorems (§10).

**Non-goals**

- Any new run, result or claim on TonomyX CASE-0031. The existing CASE-0031 code stays in the repository as an internal regression fixture and is not reported.
- The τ²-bench telecom domain, voice and multilingual variants. Verified repaired only airline and retail.
- Hosted model APIs. Reading trajectories that others published with hosted models is allowed; calling a hosted API is not.
- New human labels of any kind.

**Anti-goals** (failures even if the code works)

- Baselines that are weaker than their papers for reasons we control, such as a wrong prompt, a truncated input or a missing component, without saying so. Every adaptation is listed in the paper.
- Any threshold, prior or parameter tuned on a test split, or a test split run more than once.
- A novelty claim that no experiment or theorem in the paper supports.
- Frontier trajectories mixed into the primary comparison. They appear only in the replication (§8.4).

## 3. Constraints

- **Local models.** Ollama 0.30.4 serving open-weight models. Development on the 16 GB GPU; official runs on the 48 GB GPU over the SSH tunnel, as already set up (`docs/RUNNING.md` §11; `run_all.sh` keeps its host checks).
- **Pinned subjects.** Byte-identical, never edited, recorded in `vendor/VENDORED.md`:
  - τ²-bench at the commit τ²-Bench-Verified forked from (the base, with its original defects). The current `main` already contains the repaired task text (for example retail task 0 already says "the same one"), so the base must be that earlier commit, not `main`.
  - τ²-Bench-Verified at a pinned commit, for `FIXES.md` and the repaired tasks.
  - SOPBench at a pinned commit, with its published trajectories in `output/` (about 2 GB, kept outside git).
- **Licences.** τ²-bench and τ²-Bench-Verified are MIT. SOPBench code is MIT and its data CC BY 4.0. The paper cites them.
- **Caches.** Every interpreter run goes through the append-only `RunCache`; baseline LLM calls go through the cached gateway. A new cache file per subject: `work/runs-s2.jsonl` and `work/runs-s3.jsonl`. The P1 cache `work/runs.jsonl` is never touched.
- **Pre-registration.** One pre-registration per subject, frozen before that subject's first test-split run (§11).
- **Automation.** No human judgement enters a metric. Labels come only from mutation operators and from `FIXES.md` through the frozen mapping.
- **Separation from FAA.** Nothing is imported from FAA or `aebench`; no FAA data (including `FAA_data/bench_v1`) is used.

## 4. Approaches considered

| Approach | What it is | Trade-off |
|---|---|---|
| 1. All local | Adapters for both subjects; a local population runs the base and every mutant | Uniform and fair; the most GPU time; no evidence about strong models |
| 2. Hybrid | Published trajectories serve as the population on the base; local runs only for mutants | Cheapest and includes frontier models, but the base and mutant populations differ, which confounds every comparison; τ²-bench publishes only 4 agents from 2 vendors, so S2 needs local runs anyway |
| **3. Local primary plus replication (chosen)** | Approach 1 for every primary result; SpecDx's inference (and the run-reading baselines) re-applied to the published trajectories as a secondary analysis, with no new runs | Keeps the primary comparison clean, answers "what about strong models?", adds evidence for H5 and for the type-B table; small extra cost |

The author chose approach 3 on 2026-11-19.

## 5. The subjects as specifications

Both adapters produce the objects the method already consumes: a Tell corpus of sentences, a clause set, a scenario list, cells with anchors, and interpreter runs. Cell, anchor, typing and refutation semantics are those of v3 §2–§5.

### 5.1 S2: τ²-bench airline and retail

| SpecDx | τ²-bench |
|---|---|
| T, read by the agent | `policy.md` of the domain and the tool descriptions |
| T, read by the user simulator | each task's `user_scenario.instructions` (`reason_for_call`, `task_instructions`, `known_info`, `unknown_info`) and `description.purpose` |
| Π | the tasks of the domain (airline and retail, base commit) |
| C | the target database end state (fresh environment plus replay of `evaluation_criteria.actions`), `communicate_info` by substring matching, and `nl_assertions` where the task's `reward_basis` uses them |
| cell | (table, key, field) of the database, restricted to fields that differ between the initial and the target or any predicted end state of the task; plus one cell per `communicate_info` item |
| anchor | the target end state's value |
| interpreter | a pair (agent model, user-simulator model) with a seed |

**The user simulator is part of the population.** About two thirds of Verified's 53 fixes change the user's instructions (`reason_for_call`, `task_instructions`), not the policy. Such a defect shows up only when different user simulators read the same instruction differently. The user simulator therefore comes from the population, crossed with the agents (§8.2), and the ddmin localiser works over sentences of both the policy and the user instructions.

**Local judging only.** τ²-bench's default communicate judge is a hosted model; SpecDx uses the original substring scorer. `nl_assertions` judged by an LLM use the gateway's local judge, as BenchGuard and Who&When already do.

### 5.2 S3: SOPBench

| SpecDx | SOPBench |
|---|---|
| T | the rendered SOP: the domain system prompt plus the constraint descriptions in the `structured` format (`reference/domains/*.md`) |
| Π | the 830 tasks of the 7 domains (bank, DMV, healthcare, online market, university, library, hotel), each with its `constraint_parameters`, `user_known` and goal |
| C | the oracle: action-level constraint checks, final-database equality with the oracle's end state, and the directed action graph (`dirgraph_satisfied`) |
| cell | (collection, key, field) of the final database; plus one constraint cell per constraint instance of the task |
| anchor | the oracle's end-state value; the numeric `constraint_parameters` (for example `maximum_owed_balance` 500) are anchors of the constraint cells |
| interpreter | (agent model, seed); SOPBench's user is scripted, so no user simulator is varied |

## 6. Real defects on S2

- **Source.** `FIXES.md` of τ²-Bench-Verified: 53 fix entries (25 airline, 28 retail), each with its location, before and after text, and a reason, 24 of them quoting a policy clause.
- **Mapping to SpecDx types.** The mapping is frozen in the S2 pre-registration before any S2 run. It is mechanical, decided by the fix's location and the change's direction:

| Fix | SpecDx type | Location |
|---|---|---|
| `actions` (or `new_item_ids`, passenger data) changed, so the target end state changes | hidden requirement (mis-keyed cell) | the target cells that change |
| Only the user's instructions changed, by adding a constraint or disambiguating wording | ambiguity | the changed sentences of `user_scenario` |
| The task was impossible as written (Verified's "logical consistency") | contradiction | the clauses of the task |
| A required `communicate_info` or `nl_assertion` added | weak check | the added item |
| Both instructions and actions changed | both rows above, scored separately | |

- **Unmappable entries.** Entries the rule cannot place (for example a pure wording change with no behavioural effect) are listed as such and excluded from recall, with the count reported.
- **Matching.** A SpecDx finding matches a real defect when the type agrees and the location overlaps, using the existing `match_findings`.

## 7. Seeded defects (mutants)

Operators are generalised from the CASE-0031 operators to the two subjects:

| Type | T-mutant (needs new runs) | C-mutant (re-scored on existing runs) |
|---|---|---|
| hidden requirement | delete the T sentence that states a rule C still checks | change one target cell, or one constraint parameter in C only |
| ambiguity | replace a precise phrase with a vaguer paraphrase ("the same product" → "a similar product") | none |
| contradiction | add a T rule that conflicts with C | add a C check that conflicts with an existing one |
| weak check | none | drop one C check, or widen a tolerance |

- **Split.** Per subject: a development split of 5 mutants per type, and a test split of the rest, by a seeded draw recorded in the pre-registration.
- **Cost.** C-mutants change only the checks, so every existing run (local and published) is re-scored offline, without GPU. Only T-mutants need new interpreter runs.

## 8. Population and runs

### 8.1 Screen

The v3.1 capability screen is re-run per subject on its development tasks: act rate ≥ 0.3, valid-call rate ≥ 0.8, at least 2 families and 4 acting models, at most 2 per family. A subject that fails the screen stops with a report.

### 8.2 S2 pairing

Agents and user simulators are both drawn from the screened population. Each run pairs an agent with a user simulator of a different family, in a balanced design: every agent meets every other family's user simulator equally often. The pairing and its seeds are pre-registered.

### 8.3 Budget

The budget follows v3 (population runs, probe runs, localisation re-runs, with the active scenario choice). Runs per task and seeds are set in the pre-registration from the G3-style timing of the screen, as P2b already measured.

### 8.4 Replication on published trajectories (secondary)

- **S3.** SOPBench publishes, per domain, runs of about 30 models (GPT, Claude, Gemini, Llama, Qwen, Gemma, DeepSeek), each with the initial and final database, the constraint list, the constraint parameters and every evaluator field. SpecDx's inference and the run-reading baselines are applied to them as a second population.
- **S2.** τ²-bench publishes 4 agents (from 2 vendors) × 4 trials on the base commit. This is reported only as an illustration, since it is below the population rule.
- **Use.** Reported separately: detection versus population composition (open only, frontier only, mixed). It feeds H5 and the type-B table, and never enters the primary comparison.

## 9. Comparison with published methods

### 9.1 Primary (type A): re-run under identical conditions

Every arm reads the same specifications, the same runs and the same models, and is scored by the same matching code. Thresholds are tuned on the development split; the test split is run once.

| Arm | Method | Status |
|---|---|---|
| B1 | BenchGuard (Tu et al., COLM 2026) | implemented; adapter repackaged for S2/S3 tasks |
| B2 | Who&When (Zhang et al., ICML 2025) | implemented |
| B3 | Forced-ceiling IRT audit (Land & Bikel, EMNLP 2026) | implemented |
| B4 | Semantic entropy (Farquhar et al., Nature 2024) | implemented |
| B5 | MUS on C alone (PySAT) | implemented |
| TI | Dawid–Skene, GLAD (crowd-kit) | implemented |
| JEST | N+1-version rule (Park et al., ICSE 2021) | implemented |
| new | AgentSuite (Suh et al. 2026) | to add: code released, evaluated on τ²-style tasks; the closest competitor |
| new | auto-bench-audit (Wang et al. 2026) | to add: code released |

Each baseline that the original paper ran with a hosted model is run with the strongest model of the local population. The paper states this adaptation, and §8.4's replication shows how SpecDx itself moves with stronger models.

### 9.2 Secondary (type B): what published audits reported

A table with one row per `FIXES.md` entry. Its columns record whether the entry was reported by:
- each published audit that names τ²-bench defects: Bhat et al. 2026 (validity audit), Cao et al. 2026 (policy loopholes), Zhu et al. 2025 (ABC), AgentSuite, BenchGuard;
- SpecDx.

Cells are filled only from those papers' text and released artefacts. When an audit gives no per-task list, the cell is "not reported", never inferred.

## 10. Proving novelty

1. **Positioning table** (Related Work and Introduction). Rows: the closest works (AgentSuite, BenchGuard, auto-bench-audit, Land & Bikel, Nahum et al. 2025, Hu et al. 2026 "Verifying the Verifiers", Cao et al. 2026, MANTRA, Who&When, JEST). Columns:
   - evidence source (artefact reading, a single trace, or a population of executions);
   - truth model (single label, or set-valued and anchored);
   - separates specification defects from agent errors statistically;
   - typed into four classes;
   - logical witnesses;
   - localisation to sentences of T;
   - identifiability theory;
   - local open models only.
2. **Head-to-head results** of §9.1 per type and per subject, with paired tests (Holm-corrected) and the equal-cost table.
3. **Ablations.** Each new component is replaced by its nearest standard alternative:
   - the anchored cell model, by majority, Dawid–Skene or GLAD;
   - refutation, by none;
   - ddmin localisation, by none;
   - active scenario choice, by random choice;
   - a multi-family population, by a single family at equal budget.

   Each ablation must lower typed F1 or raise runs-to-detection (H3).
4. **Theory.** T1–T4 of v3 §4, restated for the two subjects; G0-style simulation recalibrated on S2/S3 development data.
5. **Claim register.** An appendix that lists each novelty claim with the table, figure or theorem that supports it. A claim with no entry is removed.

## 11. Pre-registration and calibration

- **Recalibration.** `V3Params` were calibrated on the CASE-0031 cache, so they are recalibrated: a G0-style simulation calibrated on each subject's development tasks and development runs. Its selection rule is v3's; its outcome freezes the S2 and S3 parameters.
- **Two pre-registrations.** `experiments/p3-s2-prereg.md` and `experiments/p3-s3-prereg.md`, each frozen and committed before its subject's test split runs. Each contains:
  - pinned commits;
  - the population and the pairing;
  - seeds and budgets;
  - parameters;
  - the real-defect mapping (S2 only);
  - the mutant split;
  - the baselines' settings and selection rules;
  - the analyses.
- **Code gate.** The comparison command refuses a test split until the subject's pre-registration is committed and the working tree is clean. This is the existing exit code 4 of `run_all.sh`.

## 12. Paper changes

- The motivating example moves to a τ²-bench defect. Candidates are retail task 0 ("a similar one" versus "the same one") or Retail Task 7's wrong ground truth reported by Bhat et al.
- The subject section, the evaluation design, the results and the threats sections are rewritten for S2 and S3.
- CASE-0031 is not mentioned as an evaluation subject.
- The cover letter cites FAA and states that no data, runs or code are shared.

## 13. Delivery phases

| Phase | Content | Hardware |
|---|---|---|
| P3a | Vendoring and the two adapters (§5); the SOPBench replication reader (§8.4) | CPU |
| P3b | Real-defect mapping and mutant operators (§6, §7); the type-B table's sources | CPU |
| P3c | Screens (measuring seconds per run) and the development grid; recalibration (§11) | 48 GB GPU |
| P3d | Baselines on development; selection; the two pre-registrations frozen | 48 GB GPU |
| P3e | Test splits run once; replication; ablations | 48 GB GPU |
| P3f | Paper rewrite | none |

S3 goes first in each phase: it is cheaper (scripted user, published trajectories) and gives an early signal.

## 13a. Settled after the builder check (Claude's calls, 2026-11-19)

1. **The rater in S2.** A rater is the pair (agent model, user-simulator model), parameterised additively: α_pair = α_agent + α_user.
   - T1–T4 hold unchanged, because they assume only a per-rater accuracy, and the additive form only ties those parameters.
   - A pair counts as cross-family when its two members differ in family. The population rules count agent families.
   - The G0-style simulator generates pairs the same way.
   - Why: it keeps the theory, needs far fewer parameters than one α per pair, and still lets user-side misreadings show up as differences in α_user.
2. **Units.**
   - One domain is one specification: 2 in S2 (airline, retail) and 7 in S3. Each has its own base, mutants and budget.
   - Scenarios are the benchmark's existing tasks. The active loop chooses which task to run next and never creates new tasks, because a new task would need a new user script and target, which only a human can write faithfully.
   - Cells are per task. Difficulty β is grouped by (table, field) across tasks, as v3 groups β by column.
   - H4 is tested as active task choice versus random task order at equal budget.
3. **Real-defect scoring and the role of Verified.**
   - **Tasks.** The development tasks of S2 are the tasks that `FIXES.md` does not touch. Every touched task is used only at test, where its labels are read for the first time.
   - **Metrics.**
     - The primary metric is typed recall over the mappable fixes.
     - Precision is reported against the union of `FIXES.md` and the per-task defects published by other audits (§9.2), since `FIXES.md` alone is not exhaustive.
     - SpecDx's other findings are listed and counted as "unconfirmed", not silently dropped.
   - **Negative control.** The Verified version is also run. Findings on the repaired tasks should disappear there.
4. **Where S2 mutants go.** They are planted in the Verified version, so that their background has no known real defect. C-mutants re-score the Verified runs from item 3, so the negative control and the mutant background share one set of runs.
5. **Harness.**
   - Interpreters run inside each benchmark's own harness: τ²-bench's orchestrator and user simulator, and SOPBench's agent loop. The prompts are the benchmarks' own, and there are no Tell paraphrases.
   - Variation comes from seeds and, in S2, from user simulators.
   - The ddmin oracle re-runs with sentences deleted, over seeds.
   - The context is raised to whatever the longest policy plus tool schema needs (at least 32,768 tokens), so nothing is truncated.
   - Why: it keeps the runs comparable with the published trajectories (§8.4) and with the benchmarks' published numbers.

## 13b. Reduced run plan and per-subject gate (author's answers, 2026-11-19)

The author keeps the 48 GB GPU but finds 3–4 weeks of continuous GPU time too long. Less important experiments are cut; for example, no experiment varies the population of LLMs on the GPU. The target is about **1–1.5 weeks of GPU time**. Where this section conflicts with §7, §8, §9 or §10, this section wins.

**Population.** One population per subject: exactly 4 acting models from at least 2 families, which is the method's minimum (v3.1 §7). There is no population-size or model-mix experiment on the GPU. In S2, the 4 models also serve as user simulators, in 4 fixed cross-family pairs.

**Runs: a fixed grid instead of the active loop.**
- Every task of a specification is run by every interpreter, 2 seeds each, which gives 8 runs per task.
- **S2.** All airline and retail tasks, on the original and on Verified: about 2,600 runs.
- **S3.** A seeded sample of 50 tasks per domain across all 7 domains: about 2,800 runs. All 830 tasks remain covered by the published-trajectory replication.
- **Mutants.** Only the tasks a mutant affects are re-run, with the same grid.
- **Caps.** Refutation probes are capped at 40 per specification, and ddmin localisation at 40 re-runs per specification, highest-scoring ambiguities first.

**Mutants: shifted to C.**
- Hidden requirement, contradiction and weak check are planted as C-mutants. They re-score existing runs and need no GPU.
- Only ambiguity, which needs a change to T, is planted as a T-mutant: 10 per subject (5 development, 5 test).
- The design therefore loses T-side hidden-requirement mutants (deleting a T rule). The paper states this, and the real defects of S2 cover that type.

**Ablations: offline only.**
- *Anchored model against majority, Dawid–Skene and GLAD; refutation off; ddmin off.* These are computed on the existing runs.
- *H4 (active task choice against random order).* Replayed on the full grid: both policies draw from the same cached runs, at equal budget.
- *H5 (families against seeds).* Computed offline on SOPBench's published trajectories (about 30 models) and on subsets of the local grid.

**Baselines: capped LLM cost.** Every arm stays in.
- Who&When judges k = 3 failing runs per clause, for the 30 clauses violated most per specification.
- Semantic entropy's localisation uses the same 40-re-run budget as SpecDx.
- BenchGuard, AgentSuite and auto-bench-audit audit each task once per specification.

**Estimated GPU time** (48 GB, 2 parallel streams):

| Part | Time |
|---|---|
| Screens | 0.5 day |
| S2 grid | 2.5 days |
| S3 grid | 0.5 day |
| Mutant re-runs | 0.5 day |
| Probes and localisation | 0.5 day |
| LLM baselines | 2 days |
| **Total** | **about 6–7 days** |

These are extrapolated from the CASE-0031 timings and are uncertain. The screens measure the real seconds per run, and the pre-registration freezes the plan only after that measurement. If the measured total exceeds 10 days, cuts are made in this order:
1. the S3 sample, 50 → 30 tasks per domain;
2. S2 T-mutants, retail only;
3. the AgentSuite and auto-bench-audit audits on development mutants only.

**Gate per subject.** Each subject has its own gate on its development split:
- no θ collapse;
- θ_m ≥ 0.5 with at least one reading for every population model;
- v3 §8.3's gate rule on the development mutants.

A subject that fails stops alone, with an honest report. The paper continues with the other subject. If both fail, the paper reports a negative result.

**Success criteria.** These are the hypotheses of v1 §1.4, pre-registered per subject:
- H1: typed F1 above the best baseline, per type.
- H2: recall on S2's real defects above the best baseline. This is stronger than v1's "above B1", because AgentSuite and BenchGuard are now in the comparison.
- H3: the ablations.
- H4: the active task choice, offline.
- H5: families against seeds, offline.

## 13c. Running everything: `run_all.sh` and one Markdown report per experiment (author's request, 2026-11-19)

`run_all.sh` is rewritten for P3. The CASE-0031 stages (`g2`, `g3`, `gate` on `p2-gate.json`) are removed from it. `./run_all.sh all` runs every stage below in order. Each stage also runs on its own, and a stage whose report already exists with exit 0 is skipped on re-run, so that the whole pipeline resumes.

| Stage | What it does | Report |
|---|---|---|
| `local` | Smoke test on the 16 GB machine: tests, adapters on 2 tasks per subject, nothing written to `results/` | `results/p3/local.md` |
| `s3-screen`, `s2-screen` | Capability screen; measured seconds per run | `results/p3/s3-screen.md`, `results/p3/s2-screen.md` |
| `s3-dev`, `s2-dev` | Development grid (base, Verified for S2), C-mutant re-scoring, development T-mutants | `results/p3/s3-dev.md`, `results/p3/s2-dev.md` |
| `s3-calibrate`, `s2-calibrate` | G0-style recalibration; frozen parameters | `results/p3/s3-calibration.md`, `results/p3/s2-calibration.md` |
| `s3-gate`, `s2-gate` | Per-subject gate (§13b) | `results/p3/s3-gate.md`, `results/p3/s2-gate.md` |
| `s3-baselines-dev`, `s2-baselines-dev` | All baselines on development; selection | `results/p3/s3-baselines-dev.md`, `results/p3/s2-baselines-dev.md` |
| `freeze` | Writes and commits each passing subject's pre-registration | `experiments/p3-s3-prereg.md`, `experiments/p3-s2-prereg.md` |
| `s3-test`, `s2-test` | Test split, once: SpecDx and every baseline | `results/p3/s3-test.md`, `results/p3/s2-test.md` |
| `s2-real` | Real defects: typed recall and precision; the type-B table (§9.2) | `results/p3/s2-real-defects.md` |
| `ablations` | Offline ablations, H3–H5 (§13b) | `results/p3/ablations.md` |
| `replication` | SpecDx on the published trajectories (§8.4), CPU only | `results/p3/replication.md` |
| `status` | Rebuilds the dashboard | `results/p3/STATUS.md` |

**Report format.** Every report starts with:
- the date and time of start and end, and the duration;
- the commit SHA, and whether the working tree was clean;
- the exact command and its exit code;
- the subject, specification and split.

Then it gives the stage's key numbers and its verdict (PASS, FAIL or STOP where a rule applies), and links to the JSON with the full data.

**Dashboard.** After every stage, `results/p3/STATUS.md` is rebuilt. It has one row per stage, with its status (not run, running, done, failed, skipped), end time, duration, the one key number (for example "θ min 0.71" or "typed F1 0.64"), and a link to its report. The GPU stages also show the measured run time against the §13b estimate.

**Exit codes**, as before:
- 0: done;
- 1: a check failed;
- 2: a usage or environment error;
- 3: a screen stop;
- 4: a test split locked (gate not passed or pre-registration not committed).

A subject's failure stops that subject's later stages only; the other subject continues (§13b).

## 14. Left to the builder

- Adapter internals: how databases are flattened to cells, and how the target end state is materialised (replaying `actions` in τ²-bench's own environment code).
- The exact act-rate measurement for each subject's tools.
- Mutant operator details within §7's table, as long as each mutant has a known type and location.
- How AgentSuite and auto-bench-audit are packaged behind the existing baseline interface.
- How the published-trajectory files are streamed (they are large).
- The structure of `run_all.sh` stages for P3.

## 15. Decisions log (2026-11-19)

| # | Decision | By | Why |
|---|---|---|---|
| 1 | Drop CASE-0031 from experiments | author | FAA's scope |
| 2 | Subjects: τ²-bench + Verified, SOPBench | author | published, independent, comparable |
| 3 | Comparison: A primary, B secondary | author | fairness plus contact with published audits |
| 4 | Approach 3: local primary plus replication | author | clean comparison and evidence on strong models |
| 5 | τ²-bench base = the commit Verified forked from | Claude | `main` already contains the repairs |
| 6 | User simulator drawn from the population, cross-family pairs | Claude | two thirds of the real defects sit in user instructions |
| 7 | Telecom, voice and multilingual out of scope | Claude | no real-defect labels; GPU budget |
| 8 | Substring communicate scorer; local LLM for `nl_assertions` | Claude | no hosted APIs |
| 9 | Recalibrate `V3Params` per subject | Claude | the frozen ones came from CASE-0031 |
| 10 | Add AgentSuite and auto-bench-audit as baselines | Claude | closest competitors with released code |
| 11 | CASE-0031 code kept as an internal regression fixture | Claude | cheap regression value; not reported |
| 12 | Rater = (agent, user-simulator) pair, additive α (§13a) | Claude | keeps T1–T4; few parameters |
| 13 | A domain is a specification; tasks are the scenarios; no new tasks (§13a) | Claude | faithful to the benchmarks |
| 14 | FIXES-touched tasks used only at test; Verified run as a negative control and mutant background (§13a) | Claude | no label leakage; a clean background |
| 15 | Benchmarks' own harnesses; no Tell paraphrases (§13a) | Claude | comparability with published runs and numbers |
| 16 | About 1–1.5 weeks of 48 GB GPU; a fixed grid; C-mutants; offline ablations (§13b) | author (budget), Claude (cuts) | 3–4 weeks is too long |
| 17 | A gate per subject; a failing subject stops alone (§13b) | author | the paper survives one failure |
