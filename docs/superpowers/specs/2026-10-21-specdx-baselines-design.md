# SpecDx baselines and the comparison study (design delta)

- **Status:** approved in brainstorming on 2026-10-21, section by section (answers C, C, B, 1; sections 1–5 approved).
- **Delta on:** `2026-10-14-specdx-v3-design.md` (v3), itself a delta on v2 and v1. Everything there stays binding unless this document replaces it.
- **Replaces:**
  - v1 §4.5's baseline list B1–B5;
  - the v3 gate's restriction to B2/B3, for every comparison *other than* the gate verdict. The gate verdict itself is unchanged (§2.4).
- **Evidence:** the verified survey in `related_work/` (612 papers, `related_work/README.md`) and the candidate notes in `baseline/*/NOTES.md`.

---

## 0. Summary

SpecDx is compared with **seven baselines**, each the closest published method for one defect type or one SpecDx component. Every baseline runs on the same specifications, the same population of open models, the same cached runs where it consumes runs, and the same scoring code as SpecDx. Choices that could favour a baseline (model, strategy, threshold) are made on the development split and frozen before the test split is run once.

`run_all.sh` gains six steps: `baselines-dev`, `freeze-baselines`, `study`, `report`, `s2` and an extended `local`. `all` now runs the whole pipeline up to the report, stopping at the first gate it does not pass.

---

## 1. Decisions (2026-10-21)

| # | Decision | Choice | Source |
|---|---|---|---|
| 1 | Number of baselines | Keep only those most related to SpecDx (§2) | User |
| 2 | Methods built on closed models | Allowed when directly related; they run on SpecDx's **open-model population** instead | User |
| 3 | Licences | Drop baselines whose code carries no licence, i.e. whose use would need the authors' permission before publication: ClarifyGPT, SpecFix, AgentSuite/COBA | User |
| 4 | Scope of `run_all.sh` | Staged: the gate and the baselines on the dev split; the S1 test-split study only after the gate PASSED; S2 is a separate later step (option C) | User |
| 5 | Budget for LLM-calling baselines | Native protocol of each method, with every cost recorded, plus an **equal-cost table** read from SpecDx's anytime checkpoints (option C) | User |
| 6 | Models for LLM-calling baselines | Each of the four population models; the best per baseline is chosen on the dev split and frozen (option B) | User |
| 7 | Architecture | In-process adapters in `specdx/baselines/`, one shared LLM gateway, original libraries where they exist (option 1) | User |

---

## 2. The baselines

### 2.1 Set

| Slot | Baseline | Source | Code used | LLM | Types reported |
|---|---|---|---|---|---|
| **B1** | BenchGuard (Tu et al., arXiv 2604.24955; COLM 2026) | `tu2026who` | Original Python package (Apache-2.0), pinned at commit `7d6a52e` | yes | all four (§2.3) |
| **B2** | Who&When failure attribution (Zhang et al., ICML 2025) | `zhang2025which` | Prompts ported from `mingyin1/Agents_Failure_Attribution` (MIT, `b2bae5c`), with attribution | yes | hidden |
| **B3** | IRT audit, forced-ceiling indicator (Land & Bikel, arXiv 2605.30504; EMNLP 2026) | `land2026auditing` | Reimplemented from the paper. The existing 2PL (`b3_irt.py`) stays as the secondary arm B3-2PL | no | hidden |
| **B4** | Semantic entropy, discrete (Farquhar et al., Nature 2024; Kuhn et al., ICLR 2023) | `farquhar2024detecting` | Reimplemented from the paper (reference code BSD-3-Clause-Clear) | no | ambiguity |
| **B5** | MUS enumeration on C alone (PySAT, SAT 2018) | `ignatiev2018pysat` | `python-sat` (MIT) | no | contradiction |
| **TI** | GLAD and Dawid–Skene (NeurIPS 2009; JRSS-C 1979) | `whitehill2009whose`, `dawid1979maximum` | `crowd-kit` (Apache-2.0) | no | hidden |
| **JEST** | N+1-version rule (Park et al., ICSE 2021) | `park2021jest` | Reimplemented from the paper | no | hidden |

**Dropped:**
- ClarifyGPT, SpecFix and AgentSuite/COBA: no licence (decision 3). They are discussed in the related work section, not run.
- BenchJack, IntellAgent, active testing, ProbDD and cleanlab: lower relevance; they are component-level alternatives only.

The old v1 B2 ("best single interpreter, every violation flagged") is renamed **B2-naive**. It stays in the gate verdict (§2.4) and is reported as a secondary arm.

### 2.2 Inputs

Each baseline receives a `BaselineInput` for one specification (the base or a mutant). It holds:
- the SpecDx `Report` for that specification, including:
  - the behaviours (cached runs);
  - the cells and their readings;
  - the violation lens;
- the Tell corpus T, split into sentences;
- the clause set C' (lifted) and the golden state.

| Baseline | What it reads |
|---|---|
| B1 | A BenchGuard task directory per specification (§2.3) |
| B2 | Logs of failing runs (§2.5) |
| B3, TI, JEST | The run × clause and model × cell matrices against the anchors d_c |
| B4 | The readings per cell |
| B5 | C' encoded over each scenario's finite state domain |

### 2.3 B1: BenchGuard packaging and mapping

**Packaging.** Each specification becomes one BenchGuard benchmark under `work/baselines/benchguard/<spec>/`, with one task per scenario:

| BenchGuard file | Content |
|---|---|
| `instruction.md` | The scenario statement plus the T sentences retrieved for it, numbered `[T:<id>]` |
| `tests/` | One file per clause of C' (`<clause_id>.sql` or `.txt`), so that evidence citing a file names a clause |
| `solution/` | The golden end state (lifted to the scenario) |
| run directory | SpecDx's runs for the scenario as agent results (BenchGuard's "Full" tier) |

**Mapping.** Fixed now and pinned by a test:

| BenchGuard subcategory | SpecDx type | Location |
|---|---|---|
| `INST-INCOMPLETE` | Ambiguity | The `[T:<id>]` sentences quoted in the evidence |
| `INST-CONTRADICT` | Contradiction if the evidence cites two or more clause files; otherwise hidden requirement at the cited clause | Clause set, or clause |
| `INST-INFEASIBLE` | Contradiction | The cited clause files |
| `GT-LOGIC`, `GT-DATA`, `GT-FMT` | Hidden requirement | The clause(s) whose golden cells the evidence cites |
| `EVAL-MISMATCH`, `EVAL-COVERAGE`, `EVAL-JUDGE-BIAS`, `EVAL-TOLERANCE` | Hidden requirement at the cited clause; **also** a weak check at that clause's column when the finding's text matches `/\b(accept|accepts|pass(es)?|allow(s)?|lenient|loose|permissive)\b/i` | Clause; column |
| `EVAL-STOCHASTIC`, `ENV-*` | Dropped: outside SpecDx's defect space | — |

**Ranking score.** Each finding is scored as severity rank × confidence rank, with CRITICAL = 4 … LOW = 1 and CONFIRMED = 3 … POSSIBLE = 1. Both BUG and WARNING findings are kept. The minimum score is a threshold chosen at freeze (§3.3).

**LLM calls.** BenchGuard's protocol code is unchanged. Only `LLMClient`'s completion method is replaced, so that every call goes through the gateway (§4.2). If the pinned package fails on the packaged tasks, the fallback is to run the protocol prompts of `benchguard-skill` (Apache-2.0) through the gateway; the paper states which was used.

### 2.4 The gate verdict is unchanged

The gate v3 rule (v3 §8.3) keeps B2-naive and B3-2PL as pre-registered. The new baselines are evaluated alongside it and never change the gate's verdict.

### 2.5 B2: Who&When on SpecDx runs

**Log.** One failing run (status `ok`, at least one violated clause) becomes a multi-agent log:
- an agent is a step's digital employee or gate actor (e.g. `DE-FN-02:reconcile`), in flow order;
- a step's message is its claim, its refused calls and the rows it wrote, from `state.tables` restricted to the step's draft tables;
- the task text is the scenario statement plus the retrieved T sentences;
- the violated clauses are given as the failure.

**Verdict.** The judge names the decisive agent and step, as in Who&When, and in addition chooses between:
- `agent_error`;
- `spec_defect: <clause_id>`, meaning the agent followed T and the clause is wrong.

**Strategies.** All-at-once, step-by-step and binary search; the prompts are ported verbatim, plus the added verdict.

**Sampling (fixed now).** Per (specification, violated clause): at most **k = 5** failing runs, the 5 smallest by `sha256(run_key)`.

**Aggregation.** A clause is reported as a hidden requirement when the share of its sampled runs judged `spec_defect` at that clause is at least τ_B2 (§3.3). Its ranking score is that share.

### 2.6 B3: the forced-ceiling IRT indicator

**Data.** Subjects are models; items are anchored cells and clauses. A response is 1 when a reading equals d_c (cells) or the behaviour satisfies the clause (clauses). Responses are pooled over a model's runs as binomial counts.

**Fit.** As in Land & Bikel:
- a constrained model with guessing c_i fixed at the chance floor (1/|V_c| for cells, 0 for clauses) and ceiling d_i = 1;
- a second fit with a free ceiling d_i;
- the score is the likelihood ratio Δℓ_i between the two.

Items are ranked by Δℓ_i, and those with Δℓ_i ≥ τ_B3 are reported as hidden requirements.

**Secondary arm.** B3-2PL, the existing code, unchanged.

### 2.7 B4: discrete semantic entropy

- **Score.** For each cell, the readings (normalised as in v3 §2.5) are clustered by exact value. The entropy over the clusters is computed both pooled over the population and per family; the score is the larger of the two.
- **Report.** Cells with entropy ≥ τ_B4 are reported as ambiguities.
- **Localisation.** Each flagged cell is localised in T by **the same ddmin oracle and budget (80)** as SpecDx (v3 §5.3), so the two methods differ only in detection.

### 2.8 B5: MUS on C alone

- **Encoding.** Each scenario's lifted clauses C' are encoded over a finite state domain: the scenario's keys; status values from the schema's check constraints; integers in [0, 2·max literal].
- **Enumeration.** MUSes are enumerated with PySAT (MCS enumeration plus `Hitman`), up to size 3, as in v1 §3.3. Each MUS is reported as a contradiction.
- **Unencodable clauses.** Clauses that call PostgreSQL functions that cannot be encoded (e.g. `fn_reconcile`) are excluded and listed in the report. No threshold.

### 2.9 TI and JEST

- **TI.** GLAD and Dawid–Skene from crowd-kit, with task = cell, worker = model and label = normalised reading. A cell is reported as a hidden requirement when the aggregate ≠ d_c. The ranking score is 1 − the posterior of d_c. No threshold beyond the aggregate.
- **JEST.** A cell or clause is reported as a hidden requirement when the share of models deviating from d_c (a majority of the model's runs deviate) is at least τ_JEST. The score is that share.

---

## 3. Fairness protocol

### 3.1 Same input and evidence

- **Specifications.** Every method runs on the same base and mutants:
  - the 20 development mutants (`dev_per_type` 5, `rng_seed` 20261006);
  - for the study, the test split: the complement of `dev_split` in `specdx mutants`. The current operators yield 35 mutants (`results/p1-mutants.tsv`), so the test split has 15. Enlarging the mutant set toward v1 §4.3's 150–250 is outside this design (§8).
- **Differential.** v1 §11 item 13's differential against the base applies to every method.
- **Shared runs.** B2, B3, B4, TI and JEST consume **exactly the cached SpecDx runs** of the specification, including the scenarios SpecDx synthesised. Any bias this introduces favours the baselines.

### 3.2 Same models

B1 and B2:
- run on each of the four population models (v3 §7), under the pinned digests and Ollama 0.30.4;
- use each method's original generation settings (temperature and so on), with seed 1;
- send every call through the gateway's cache.

### 3.3 Selection on the dev split, frozen before the test split

`freeze-baselines` chooses, on the base plus the 20 development mutants:
- for B1: the model, and the minimum finding score;
- for B2: the model, the strategy and τ_B2;
- for B3: τ_B3;
- for B4: τ_B4;
- for JEST: τ_JEST.

**Rule.** Each choice maximises the mean typed F1 over the types the baseline reports, on the primary (scorable) table. Ties go to the smaller model, then to the higher threshold.

**Grids** (fixed now):
- thresholds: deciles of the dev scores;
- models: the four in the population;
- B2 strategies: all three.

**Recording.** The choices are written to `experiments/p3-baselines-prereg.json` and `.md`, pinned by `tests/test_preregistration_baselines.py`, and committed.

**Asymmetry.** SpecDx's thresholds stay frozen from G0 (simulation only). The baselines are tuned on real development data, which is a conservative asymmetry.

### 3.4 Same scoring

For every method:
- `match_findings`, typed and untyped, as in v1 §4.6 and v2;
- the primary table on scorable mutants;
- exact-arithmetic F1;
- the threshold-free metrics of v1 §4.6: average precision per type and Detection@k for k ∈ {1, 3, 5}.

### 3.5 Same cost: two tables

- **Table A, each method at its native protocol.** Columns: F1/AP/Detection@k, and cost (LLM calls, prompt and completion tokens, and GPU-seconds from Ollama's `prompt_eval_count`, `eval_count` and `total_duration`).
  - SpecDx's cost is that of its runs, probes and ddmin calls.
  - The baselines that consume SpecDx's runs are charged the same runs plus their own calls (B4 and SpecDx both pay for ddmin).
- **Table B, equal cost.** For each LLM-calling baseline, SpecDx is scored at the anytime checkpoint (§4.4) whose GPU-seconds are closest to, and not above, the baseline's.

### 3.6 Statistics

As in v1 §4.7:
- exact McNemar per mutant on typed detection, SpecDx against each baseline, with Holm correction across baselines;
- paired risk differences;
- 95% bootstrap CIs over mutants (10,000 resamples, seed 20261021).

### 3.7 Integrity

- The test split is run once, and only when both hold:
  1. the gate is PASSED;
  2. the baseline pre-registration is committed and the working tree is clean.
- `study` refuses any configuration that differs from the frozen one.

---

## 4. Architecture

### 4.1 Modules

| File | Responsibility |
|---|---|
| `specdx/baselines/api.py` | `BaselineInput`, `BaselineOutput(defects, scores, cost, failures)`, the `Baseline` protocol, the registry |
| `specdx/baselines/llm.py` | The gateway (§4.2) |
| `specdx/baselines/benchguard.py` | Packaging, the client override, finding → `Defect` mapping (§2.3) |
| `specdx/baselines/whowhen.py` | Log building, the ported prompts and three strategies, sampling, aggregation (§2.5) |
| `specdx/baselines/irt_audit.py` | Forced-ceiling indicator (§2.6) |
| `specdx/baselines/semantic_entropy.py` | §2.7 |
| `specdx/baselines/mus_c.py` | §2.8 |
| `specdx/baselines/truth_inference.py` | GLAD and DS (§2.9) |
| `specdx/baselines/jest_rule.py` | §2.9 |
| `specdx/eval/study.py` | Runs a split (SpecDx and the baselines), scores it, computes statistics, writes both tables and the Markdown report |

### 4.2 LLM gateway

- **Endpoint.** An OpenAI-compatible client against `SPECDX_OLLAMA_URL`.
- **Cache.** `work/baseline-llm.jsonl`, append-only and separate from `work/runs.jsonl`. The key is sha256 of (model digest, messages, generation parameters, seed).
- **Recorded per call:** tokens, duration and status.
- **Failures.** A timeout or error is cached as `status: error`. It is counted and never treated as a finding.
- **Digests.** A model digest that differs from the pinned one stops the run.

### 4.3 CLI

| Command | Effect |
|---|---|
| `specdx baselines --split dev --config experiments/p2-gate.json --out results/p3-baselines-dev.json` | Runs every baseline × model on base plus dev |
| `specdx freeze-baselines --from results/p3-baselines-dev.json --out experiments/p3-baselines-prereg` | Writes the frozen choices |
| `specdx study --split test --config … --baselines experiments/p3-baselines-prereg.json --out results/p3-study.json` | Runs SpecDx and the frozen baselines on the test split |
| `specdx report --dev results/p3-baselines-dev.json [--test results/p3-study.json]` | Writes tables A and B, the statistics, and `results/p3-report.md` |

Each specification's SpecDx `Report` is obtained by `run_specdx` against the run cache. For the dev split this means no new interpreter runs beyond the gate's.

### 4.4 SpecDx anytime checkpoints (output only)

- **What is recorded.** The loop records the defect list and the cumulative cost at population-run counts {25, 50, 100, 150, 200} (and at the end), in the report's `checkpoints` field.
- **Effect on behaviour.** Recording a checkpoint fits the same model on the runs so far. It never feeds back into scenario choice, escalation or stopping.
- **Test.** The final report must be byte-identical with and without checkpoints.

### 4.5 Dependencies

- A new optional extra `[baselines]`: `crowd-kit`, `python-sat`, and `benchguard @ git+https://github.com/XinmingTu/BenchGuard@7d6a52e` (brings `litellm`).
- The clones in `baseline/` are reference copies only; nothing imports from them.

---

## 5. `run_all.sh`

### 5.1 Steps

The existing steps (`local`, `g2`, `preflight`, `g3`, `gate`, `status`) keep their behaviour. `local` and `all` are extended.

| Step | What | Precondition |
|---|---|---|
| `local` | Existing smoke tests, plus: baseline unit tests; each baseline once on the cached base with one model; a gateway round-trip; validation of the BenchGuard package. Writes only to a scratch directory | — |
| `baselines-dev` | `specdx baselines --split dev`, detached when run alone | `results/p2-gate.json` exists (PASSED or NOT PASSED) |
| `freeze-baselines` | `specdx freeze-baselines`; runs the pin test; commits the pre-registration | `baselines-dev` finished with no failed specification |
| `study` | `specdx study --split test`, detached when run alone | Gate PASSED; pre-registration committed; working tree clean. Otherwise exit 4 |
| `report` | `specdx report` for dev, and for test when present | The inputs exist |
| `s2` | Prints that S2 needs the τ² adapter (P3 follow-up) | Always exits 2 |
| `status` | Gate, baselines-dev and study: pid, log tail, cache line counts | — |
| `all [--yes]` | `g2` → `preflight` → `g3` → `gate` → `baselines-dev` → `freeze-baselines` → `study` → `report`, in the foreground (run inside tmux). It stops at the first unmet precondition. A NOT PASSED gate still runs `baselines-dev` and the dev `report`, then exits 4 | — |

### 5.2 Exit status

| Code | Meaning |
|---|---|
| 0 | Done |
| 1 | A check failed |
| 2 | Usage or environment error |
| 3 | G2 stop rule |
| **4** | Gate NOT PASSED; the test study is locked |

### 5.3 Error handling

- **One failed specification.** A baseline that fails on one specification marks that specification `failed` for that baseline only. That baseline's verdict is INCOMPLETE; SpecDx and the other baselines are unaffected.
- **Infrastructure errors** (PostgreSQL, Ollama, the tunnel) stop the step. Re-running resumes from both caches.
- **Cache integrity.** After `freeze-baselines`, `run_all.sh` checks the md5 of the baseline cache's committed prefix, as it already does for `runs.jsonl`.

---

## 6. Testing

### 6.1 Unit tests (no LLM)

| Component | What is tested |
|---|---|
| B3 | A planted miskeyed clause ranks first by Δℓ |
| B4 | A 50/50 cell scores above a consensus cell; flagged cells go through the shared ddmin |
| TI | Recovery on planted strong and weak models |
| JEST | Behaviour at the threshold boundary |
| B5 | A planted size-2 MUS is returned; unencodable clauses are excluded and listed |
| B2 | Log order; deterministic k = 5 sampling; aggregation of planted verdicts |
| B1 | The packaged base validates against `STANDARD_FORMAT`; every mapping row maps; evidence quotes resolve to the right T sentences |

### 6.2 Gateway

Uses `httpx.MockTransport`:
- a cache hit sends no request;
- the key changes with the digest;
- tokens and durations are recorded;
- a timeout is cached as `error`;
- a digest mismatch stops the run.

### 6.3 Evaluation

- Checkpoints leave the final report byte-identical.
- `study.py` statistics match hand-computed values (McNemar, Holm, seeded bootstrap, AP, Detection@k) on fixtures.
- Table B picks the right checkpoint.
- `study` refuses to run:
  - without a PASSED gate;
  - without a committed pre-registration;
  - on a dirty working tree;
  - with a configuration that differs from the frozen one.
- `tests/test_preregistration_baselines.py` pins every frozen value, together with k = 5 and the grids.

### 6.4 Integration

`run_all.sh local` runs every baseline once on the base with one model, including BenchGuard end to end, and writes nothing to `work/`, `results/` or `experiments/`.

---

## 7. Values fixed now

| Value | Setting |
|---|---|
| Baselines | The seven of §2.1, plus B2-naive and B3-2PL as secondary arms |
| B1 | The mapping table and the weak-check regex of §2.3; the score formula |
| B2 | k = 5 with sha256 sampling |
| Selection | The rule and grids of §3.3 |
| SpecDx checkpoints | {25, 50, 100, 150, 200} |
| Bootstrap | 10,000 resamples, seed 20261021 |
| Exit codes | §5.2 |

Chosen at `freeze-baselines`, then frozen: the models, the B2 strategy, and τ_B1min, τ_B2, τ_B3, τ_B4, τ_JEST.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| BenchGuard's Python package is out of date (its README says so) | Pinned commit; fallback to the skill's protocol prompts through the same gateway (§2.3), disclosed in the paper |
| Local models make BenchGuard and Who&When much weaker than in their papers | Each baseline's best model is chosen on dev (decision 6), and the threats to validity say so. Decision 2 allows a frontier-model arm later if the venue requires it, but it is not part of this design |
| Who&When cost | k = 5 runs per (specification, clause); the cost is reported |
| Unencodable clauses limit B5 | Listed per run; R-12 (`fn_reconcile`) is the known case |
| The gate is NOT PASSED | The dev comparison is still produced and reported; the test study stays locked (exit 4) |
| The S1 test split has only 15 mutants (35 in all), far below v1 §4.3's 150–250, so per-type McNemar tests have little power | Report effect sizes with bootstrap CIs alongside the p-values. Enlarging the operator sites is a separate decision for the author before `study` runs; `study` uses whatever `specdx mutants` yields, minus the dev split |

---

## 9. Amendments at planning (2026-10-21)

These were found while the plan was being written against the code. Where a rule below conflicts with an earlier section, it replaces that section.

1. **The cost unit of Table B is model requests, not GPU-seconds (replaces §3.5).**
   - **The problem.** Cached SpecDx runs record neither tokens nor durations. Per step, they keep only `tool_calls`, `valid_calls`, `nudges` and `parsed_from_text`.
   - **The common unit** is one chat request to Ollama.
   - **New runs.** The executor gains four harness counters for runs made from now on: `requests`, `prompt_tokens`, `eval_tokens`, `duration_ms`. They are observations only, so run keys do not change.
   - **Runs cached before the counter.** A step costs at most `tool_calls + nudges + 1` requests. That is an upper bound: it overstates SpecDx's cost, which is the conservative direction.
   - **Baselines.** Their requests are counted exactly by the gateway.
   - **Reporting.** Tokens and GPU-seconds are reported wherever they were recorded.
2. **Checkpoints are taken before localisation (refines §4.4).**
   - **What a checkpoint holds.** The defect list after the diagnosis and after the base-ambiguity differential, but before ddmin localisation.
   - **Consequence.** Ambiguity findings at an intermediate checkpoint carry no sentence location and match no ambiguity truth. Table B marks ambiguity as "unlocalised" at those checkpoints. The final checkpoint is the final report.
3. **B5 explores the golden state's neighbourhood instead of encoding SQL into SAT (replaces §2.8).**
   - **Exploration.** For each canon scenario, B5 starts from the golden state, materialised by PostgreSQL so that derived values (`fn_reconcile`) are real. It then explores up to two rounds of single-clause repairs with the existing `propose`, within at most 200 clause evaluations per scenario, the bound of v1 §3.6.
   - **MUS candidates.** Each explored state's violation set is a correction set. The minimal hitting sets of size ≤ 3 over these sets (PySAT `Hitman`) are the MUS candidates, reported as contradictions, including size 1.
   - **Why.** No SQL is re-encoded, so no clause has to be excluded. This is v1's B5: "symbolic repair … with no interpreters".
4. **B1 runs in BenchGuard's definition-audit mode only (refines §2.3).**
   - **Inputs.** Instruction, tests and solution: `AuditMode.DEFINITION`, `agent_evidence=off`.
   - **Why no traces.** The results-verification mode needs agent trials in Harbor's trial format. SpecDx runs cannot fill it without inventing fields, so the "with traces" arm is dropped.
   - **How the gateway is wired.** It is injected by replacing the pipeline's `ProtocolRunner` with one built on a `LLMClient` subclass whose `complete` and `acomplete` call the gateway. BenchGuard's code is not modified.
   - **Generation settings.** Temperature follows BenchGuard's default (0.0); `max_tokens` is 4096.
