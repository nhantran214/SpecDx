# Baselines: what is compared with SpecDx and how to run it

SpecDx is compared with seven published methods, plus two secondary arms taken from SpecDx's own gate.
- Every baseline is implemented inside SpecDx, in `specdx/baselines/`, behind one interface.
- Every baseline runs on the same specifications as SpecDx, and every baseline that reads interpreter runs reads the same cached SpecDx runs.
- The two LLM-based baselines call the same open-weight population models, with the same digests, through one cached gateway.
- All baselines are scored by the same matching code as SpecDx.

This file explains which baselines are used, how each one is run, and how to run the experiments.

The numbered folders hold, for each baseline:
- `NOTES.md`: paper, original code and commit, licence, adaptation notes;
- a local PDF of the paper;
- a shallow clone of the original code (`code*/`) for reference.

Clones and PDFs are git-ignored. **Nothing is imported from these clones**: the code that runs is in `specdx/baselines/`, and its third-party dependencies are installed from PyPI or a pinned git commit.

---

## 1. The baselines in use

| Slot | Registry name | Method (paper) | Implementation | Folder | Uses an LLM | Reads SpecDx runs | Threshold tuned on dev | Reports |
|---|---|---|---|---|---|---|---|---|
| **B1** | `benchguard` | BenchGuard: LLM cross-verification of a benchmark's instruction, tests and solution (Tu et al., COLM 2026) | `benchguard.py`, using the original package at commit `7d6a52e` | [04_benchguard](04_benchguard/NOTES.md) | yes | no | yes (+ model) | all four types |
| **B2** | `whowhen` | Who&When automated failure attribution (Zhang et al., ICML 2025) | `whowhen.py`, with prompts ported verbatim (MIT) | [05_who_and_when](05_who_and_when/NOTES.md) | yes | yes | yes (+ model, strategy) | hidden requirement |
| **B3** | `irt_audit` | Forced-ceiling IRT mislabel indicator (Land & Bikel, EMNLP 2026) | `irt_audit.py`, reimplemented from the paper | [06_irt_audit](06_irt_audit/NOTES.md) | no | yes | yes | hidden requirement |
| **B4** | `semantic_entropy` | Discrete semantic entropy (Farquhar et al., Nature 2024; Kuhn et al., ICLR 2023) | `semantic_entropy.py`, reimplemented from the papers | [10_semantic_entropy](10_semantic_entropy/NOTES.md) | no (its localisation re-runs interpreters) | yes | yes | ambiguity |
| **B5** | `b5_mus` | Contradictions in C alone: MUS by hitting sets (PySAT, SAT 2018) | `mus_c.py`, using `python-sat` | [11_pysat_marco](11_pysat_marco/NOTES.md) (PySAT part) | no | no | no (τ = 0) | contradiction |
| TI | `glad` | GLAD (Whitehill et al., NeurIPS 2009) | `truth_inference.py`, using `crowd-kit` | [08_crowdkit](08_crowdkit/NOTES.md) | no | yes | no (τ = 0) | hidden requirement |
| TI | `dawid_skene` | Dawid–Skene (JRSS-C 1979) | `truth_inference.py`, using `crowd-kit` | [08_crowdkit](08_crowdkit/NOTES.md) | no | yes | no (τ = 0) | hidden requirement |
| JEST | `jest` | N+1-version rule (Park et al., ICSE 2021) | `jest_rule.py`, reimplemented from the paper | [15_jest_spec_vote](15_jest_spec_vote/NOTES.md) | no | yes | yes | hidden requirement |
| secondary | `b2_naive` | Best single interpreter, every violated clause flagged (SpecDx's gate baseline) | `secondary.py`, wrapping `b2_single.py` | — | no | yes | no (τ = 0) | hidden requirement |
| secondary | `b3_2pl` | Plain 2PL IRT, negative discrimination (SpecDx's gate baseline) | `secondary.py`, wrapping `b3_irt.py` | — | no | yes | no (τ = 0) | hidden requirement |

### How each baseline is run on SpecDx

#### B1 BenchGuard
- **Packaging.** The anchor canon scenario of each specification becomes one BenchGuard task:
  - `instruction.md` holds the scenario statement and the relevant sentences of T, numbered `[T:<id>]`, capped at 8000 characters;
  - `tests/` holds one file per clause of C that applies;
  - `solution/` holds the golden end state. If the database refuses that state, which C2 mutants do on purpose, the unmaterialised golden tables are used.
- **Audit.** BenchGuard runs in definition-audit mode, with its own protocols unchanged. Every LLM call goes through SpecDx's gateway, at temperature 0 with `max_tokens` 4096.
- **Mapping findings to types.** The mapping is frozen:

  | BenchGuard finding | SpecDx type and location |
  |---|---|
  | `INST-INCOMPLETE` | ambiguity, at the quoted `[T:id]` sentences |
  | `INST-CONTRADICT` | contradiction when it cites two or more clauses; otherwise hidden requirement |
  | `INST-INFEASIBLE` | contradiction |
  | `GT-*` | hidden requirement at the cited clause |
  | `EVAL-*` | hidden requirement at the cited clause; also a weak check on the clause's column when the finding says the check *accepts* or *passes* too much |
  | `ENV-*`, `EVAL-STOCHASTIC` | dropped |

- **Score.** Severity rank × confidence rank.

#### B2 Who&When
- **Logs.** Each failing SpecDx run becomes a multi-agent log with one entry per work step: the step's claim, its refused tool calls and its rows.
- **Sampling.** For each violated clause, at most 5 failing runs are judged. Runs already chosen for other clauses are reused first; all choices are ordered by sha256 of the run key.
- **Localisation.** The judge finds the decisive step with one of the three original strategies (`all_at_once`, `step_by_step`, `binary_search`).
- **Verdict.** One added call decides `agent_error` or `spec_defect <clause>`. A clause's score is the share of its sampled runs judged a specification defect at that clause.
- **Settings.** Generation follows the original's open-model path: temperature 0.6, top_p 0.95, seed 1, `max_tokens` 1024.
- **Unparseable replies.** They are counted, never turned into findings.

#### B3 forced-ceiling IRT
- **Data.** Subjects are interpreters (model, Tell variant, seed); items are anchored cells.
- **Fit.** Item parameters are refitted with the ceiling forced to 0 and to 1, θ held fixed and c free.
- **Score.** Δℓ = ℓ(d=0) − ℓ(d=1), per response. Only Δℓ > 0 is reported.

#### B4 semantic entropy
- **Score.** The entropy of each cell's readings, pooled and per model family; the larger of the two counts.
- **Localisation.** Flagged cells are localised in T by SpecDx's own ddmin localiser, with the same budget of 80 re-runs, highest score first. The re-runs are interpreter runs, so they go to Ollama.

#### B5 MUS on C
- **Exploration.** Starting from the anchor's golden state, explore up to two rounds of single-clause repairs, at most 200 evaluations.
- **MUS.** Each explored state's violated set is a correction set. Minimal hitting sets of size ≤ 3 (PySAT `Hitman`) are reported as contradictions. No interpreter is involved.

#### GLAD and Dawid–Skene
- **Input.** Task = cell, worker = interpreter, label = its modal reading; `n_iter` = 100.
- **Report.** A cell is reported only when the aggregate label differs from the value the specification declares. The score is 1 − P(declared value).

#### JEST
- **Score.** The share of models whose modal reading of a cell deviates from the declared value.

#### Secondary arms
- `b2_naive` and `b3_2pl` are the gate's own B2 and B3, run unchanged.

### Thresholds, models and strategies (chosen on the development set only)

- **Grid.**
  - Tuned baselines (`benchguard`, `whowhen`, `irt_audit`, `semantic_entropy`, `jest`) are scored on the development set at every threshold in {0, deciles of their scores}.
  - The LLM baselines are also scored for each population model, and `whowhen` for each strategy.
- **Rule.** The choice maximises the mean typed F1 over the types the baseline reports, on scorable mutants. Ties go to the smaller model, then the higher threshold.
- **Untuned arms.** `glad`, `dawid_skene`, `b5_mus`, `b2_naive` and `b3_2pl` keep τ = 0.
- **Freezing.** The choices are written to `experiments/p3-baselines-prereg.{json,md}` and committed before the test set runs. They cannot be replaced without `--force`.

### Cost accounting

- **Unit.** Costs are counted in model requests, plus prompt and output tokens and GPU time wherever Ollama recorded them.
- **What each baseline pays.**
  - A baseline that reads SpecDx's runs is charged those population runs plus its own calls.
  - BenchGuard and B5 pay only their own calls.
- **The two tables.**
  - Table A reports each method at its native protocol.
  - Table B reports SpecDx at each LLM baseline's cost, read from SpecDx's anytime checkpoints at 25, 50, 100, 150 and 200 population runs, and at the end. Ambiguity at an intermediate checkpoint is not yet localised.

---

## 2. Before running

1. SpecDx is installed and its tests pass. PostgreSQL (`specdx-postgres`, port 55434) is up.
2. Install the baselines' dependencies once:
   ```bash
   .venv/bin/python -m pip install -e ".[baselines]"     # crowd-kit, python-sat, BenchGuard @ 7d6a52e
   .venv/bin/python -c "import crowdkit, pysat, benchguard; print('ok')"
   ```
3. The population is frozen in `experiments/p2-gate.json`, and its models are pulled on the Ollama host with the same digests as the cached SpecDx runs. Nothing else needs pulling: BenchGuard and Who&When reuse those models. A missing model or a different digest stops the run before any call.
4. The gate has finished: `results/p2-gate.json` exists. The baselines run after the gate, on the same cached runs.
5. `SPECDX_OLLAMA_URL` points at the Ollama host, for example `http://localhost:11435` through the SSH tunnel.

## 3. Smoke test (no GPU time beyond a few calls)

```bash
./run_all.sh local
```

Besides SpecDx's own smoke run, this:
- checks the baseline packages;
- runs every baseline arm once on the smoke base with one model;
- lists arms that failed;
- fails if any LLM call's prompt plus its output budget reached the 16384-token context window (`near_ctx`).

Fix any failure here before running the GPU steps.

## 4. Running the comparison

The usual way is `./run_all.sh all [--yes]`, which runs the gate and then these steps. Each step can also be run on its own:

```bash
export SPECDX_OLLAMA_URL=http://localhost:11435

./run_all.sh baselines-dev       # every arm, every population model, base + 20 development mutants
./run_all.sh freeze-baselines    # choose and commit each baseline's model, strategy and threshold
./run_all.sh study               # SpecDx + frozen baselines on the test mutants (gate PASSED only)
./run_all.sh report              # results/p3-report.md
./run_all.sh status              # pid, log tail and cache sizes of the running steps
```

| Step | Reads | Writes | Notes |
|---|---|---|---|
| `baselines-dev` | `experiments/p2-gate.json`, `work/runs.jsonl` | `results/p3-baselines-dev.json`, `work/baseline-llm.jsonl` | Detached, with a pid and log in `work/p3-baselines-dev.*`. Prints an estimate of arms, specifications and judged logs once the base report is in. Checks the run cache and the Ollama digests before launching |
| `freeze-baselines` | `results/p3-baselines-dev.json` | `experiments/p3-baselines-prereg.{json,md}` (committed) | Refuses if any development specification failed. Records the sha256 of the gate configuration and gate result, and the line count and md5 of the baseline cache |
| `study` | the pre-registration, `experiments/p2-gate.json` | `results/p3-study.json` | Exit 4 unless: the gate PASSED; the pre-registration is committed; the tree is clean; the fingerprints match. Embeds the thresholds it used. Detached (`work/p3-study.*`) |
| `report` | the dev result, and the study result if present | `results/p3-report.md` | The dev section shows Table A twice, at the in-sample thresholds and at τ = 0. The test section uses the thresholds embedded in the study |

If the gate is NOT PASSED, `all` runs `baselines-dev` and the development report, then exits 4. The test set stays locked.

### The underlying commands

These are the commands `run_all.sh` runs. `study` accepts only these two files:

```bash
.venv/bin/specdx baselines --split dev --config experiments/p2-gate.json \
  --out results/p3-baselines-dev.json --workers 2
.venv/bin/specdx freeze-baselines --from results/p3-baselines-dev.json \
  --out experiments/p3-baselines-prereg --cache work/baseline-llm.jsonl \
  --gate-config experiments/p2-gate.json --gate-result results/p2-gate.json
.venv/bin/specdx study --split test --config experiments/p2-gate.json \
  --baselines experiments/p3-baselines-prereg.json --out results/p3-study.json --workers 2
.venv/bin/specdx report --dev results/p3-baselines-dev.json --test results/p3-study.json \
  --out results/p3-report.md
```

- To try the baselines on the base specification only, with one model: `specdx baselines --split base --config experiments/p2-gate.json --out /tmp/base.json --models gpt-oss:20b`.
- `--models` restricts the LLM arms; the other arms always run.

## 5. Reading the results

- **Table A** gives, per arm:
  - typed and untyped F1 per defect type;
  - average precision per type and Detection@1/3/5, both computed at τ = 0;
  - requests, tokens and duration. A "(bound)" mark means some runs predate exact counting, so the figure is an upper bound.
- **The paired comparison** of SpecDx against each arm gives exact McNemar with Holm correction and the risk difference with a 95% bootstrap CI (10,000 resamples, seed 20261021).
- **Table B** gives SpecDx's F1 at each LLM arm's request budget.
- **INCOMPLETE** after an arm name means it failed on some specifications. They are excluded from its scores and listed.

## 6. Pausing, resuming and failures

- **Pausing.** Pause and resume a detached step with `kill -STOP` / `kill -CONT` on its pid in `work/`. Never start a second SpecDx process on the same database.
- **Resuming.** If a step stops (tunnel dropped, reboot), run the same step again.
  - SpecDx's runs come from `work/runs.jsonl` and the baselines' LLM calls from `work/baseline-llm.jsonl`.
  - Finished work is served from these caches and the rerun writes the same result. Both caches are append-only; never edit them.
- **Outages.** An Ollama outage (5xx, connection failure, a timeout before the host has answered once) is never cached: the step stops and can be resumed. A model timeout or malformed reply is cached as an error and is never a finding.
- **Failures.** A baseline that fails on one specification is marked failed there only. A baseline that fails on the base specification is failed everywhere, because the differential needs the base.

## 7. Licences of the code in use

| Baseline | Code | Licence |
|---|---|---|
| B1 BenchGuard | `benchguard` package, git commit `7d6a52e` | Apache-2.0 |
| B2 Who&When | prompts ported from `mingyin1/Agents_Failure_Attribution` @ `b2bae5c` | MIT |
| B5 | `python-sat` | MIT |
| GLAD, Dawid–Skene | `crowd-kit` | Apache-2.0 |
| B3, B4, JEST | reimplemented from the papers (B4's reference code is BSD-3-Clause-Clear) | — |
| secondary arms | SpecDx's own code | — |
