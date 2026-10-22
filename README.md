# SpecDx

**Diagnosing executable business specifications with populations of LLM interpreters.**

An executable business specification has two parts:
- **T**: the natural-language requirements, policies and knowledge base an executor is told;
- **C**: the executable checks the result is judged by (database invariants, expected outcomes, a golden end state, guardrails).

Systems and agent benchmarks treat such a specification as ground truth, yet it contains defects. SpecDx runs a population of local open-weight LLMs (several model families, Tell variants and seeds) as interpreters of the specification. It then separates *specification defects* from *agent errors* and types each defect:

| Type | Meaning |
|---|---|
| contradiction | no behaviour can satisfy C |
| hidden requirement | C demands something T does not say (C is stricter than T) |
| ambiguity | T admits two or more readings |
| weak check | a wrong behaviour passes every check (C is looser than T) |

Each finding comes with its location (a clause, a cell, or sentences of T) and with witnesses.

Everything runs locally and automatically: there are no hosted APIs and no human labels. The first subject (S1) is the TonomyX canon, CASE-0031: a Vietnamese procure-to-pay case on PostgreSQL.

---

## Contents

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Checking the machine](#checking-the-machine)
7. [Running the experiments](#running-the-experiments)
8. [The `specdx` command line](#the-specdx-command-line)
9. [Outputs](#outputs)
10. [Caches, resuming and moving work](#caches-resuming-and-moving-work)
11. [Troubleshooting](#troubleshooting)
12. [Rules of the repository](#rules-of-the-repository)

---

## How it works

1. **Specification.** The vendored TonomyX canon provides T (requirement documents and knowledge base, extracted into a *Tell corpus*) and C (invariants, expectations, golden state and guardrails, extracted into a clause set). A clause written for one case is *lifted* to the scenario parameters (quantities, discount, number of goods receipts) so that it applies to other scenarios.
2. **Population.** A *capability screen* runs each candidate model on the base specification. A model is admitted when it acts and makes valid tool calls often enough. The population is capped per family and needs at least 2 families and 4 acting models.
3. **Runs.** Each interpreter executes the case's work steps through TonomyX's gated tools on its own PostgreSQL transaction. Every run is cached in an append-only file.
4. **Diagnosis.** Each run is read cell by cell, where a cell is a (table, row, column) value or a clause.
   - An *anchored cell model* treats the specification as a fallible rater. Each cell has a latent set of admissible readings, and each model has a fitted accuracy. Together these separate defects from agent errors.
   - Hitting sets over the violated-clause sets propose contradictions, which are refuted with witness states.
   - Ambiguities are localised to sentences of T by delta debugging.
5. **Active scenarios.** Thompson sampling chooses the next scenario, using boundary values taken from T. Each new scenario starts with a cheap pilot of one model per family, and escalates to the full committee only when the pilot is informative.
6. **Evaluation.** Specification *mutants* carry known defects of each type. They are split into a development set (20 mutants, 5 per type) and a test set (the rest).
   - The **gate** runs SpecDx on the development mutants and decides whether the method may proceed.
   - The **comparison study** then runs SpecDx and seven published baselines on the same specifications, runs and scoring. The baselines' choices are frozen on the development set before the test set is run once.

## Repository layout

| Path | Contents |
|---|---|
| `specdx/spec/` | Tell corpus (PDF → chunks → sentences), clause extraction, lifting, the column model |
| `specdx/scenarios/` | the scenario space, canon scenarios, boundary scenarios taken from T |
| `specdx/orchestrate/` | the parametric CASE-0031 flow on TonomyX's public runtime |
| `specdx/interpreters/` | the population, Tell variants, the LLM executor and harness, the run cache |
| `specdx/observe/` | one run in its own transaction → a behaviour (end state, refusals, claims) |
| `specdx/infer/` | cells and readings, the anchored cell model, hitting sets, typing |
| `specdx/active/` | the main loop, Thompson sampling, refutation, ddmin, run budgets |
| `specdx/mutate/` | mutation operators with ground truths, and the dev/test split |
| `specdx/sim/` | the simulation study (no GPU, no database) |
| `specdx/baselines/` | the comparison baselines and the cached LLM gateway they share |
| `specdx/eval/` | the capability screen, the gate, scoring, the comparison study, statistics, freezing |
| `specdx/cli.py` | the `specdx` command |
| `run_all.sh` | the end-to-end driver (smoke tests, gate, comparison study) |
| `experiments/` | frozen configurations, pre-registrations, hardware profiles, replay scripts |
| `tests/` | unit and database tests (no model is ever called) |
| `vendor/tonomyx/` | TonomyX, vendored byte-identical and never edited |
| `baseline/` | notes on the comparison baselines, plus local reference clones of their code (git-ignored) |
| `work/` | git-ignored: run cache, Tell cache, mutant packs, logs, pid files |
| `results/` | git-ignored: screens, gate and study results, reports |
| `related_work/` | git-ignored: the local literature survey |

## Requirements

| What | Version | Used for |
|---|---|---|
| Python | 3.13 (3.12 or later) | SpecDx and the vendored TonomyX |
| Docker | any recent | PostgreSQL 16 for SpecDx on port 55434 |
| Ollama | 0.30.4 for the official runs | the interpreters; models need native tool calling |
| poppler (`pdftotext`) | any | extracting the Tell corpus from the requirement PDFs, once |
| git, jq, curl | any | setup, freezing configurations, `run_all.sh` |
| GPU | 16 GB for development; 48 GB for the official runs | Ollama only; SpecDx and PostgreSQL can run on a CPU machine |

SpecDx refuses PostgreSQL ports 55432 and 55433, which belong to other projects.

## Installation

```bash
git clone <this repository> specdx && cd specdx
python3.13 -m venv .venv
.venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"          # SpecDx and TonomyX
.venv/bin/python -m pip install -e ".[baselines]"                      # crowd-kit, python-sat, BenchGuard (pinned)
docker compose up -d                                                    # PostgreSQL 16 on localhost:55434 (first time)
.venv/bin/python -m pytest -q                                           # every test; no model is called
.venv/bin/ruff check .
```

After the first `docker compose up -d`, restart the database with `docker start specdx-postgres`. Do not run `docker compose up` again while a run is active.

Optional: check that `vendor/tonomyx` is byte-identical to the pinned TonomyX commit. This needs a TonomyX checkout:

```bash
TONOMYX_SRC=/path/to/TonomyX bash scripts/verify_vendor.sh
```

One-time preparation of the specification (cached under `work/`):

```bash
.venv/bin/specdx build-tell                                 # extract and cache the Tell corpus
.venv/bin/specdx make-variants --model qwen3:14b --count 2  # paraphrase Tell variants 1..2 (frozen once made)
.venv/bin/specdx mutants                                    # build every mutant and list its ground truth
```

## Configuration

SpecDx reads its configuration from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `SPECDX_DATABASE_URL` | `postgresql://tonomyx:tonomyx@localhost:55434/tonomyx` | the SpecDx database; parallel workers add `tonomyx_w0`, `tonomyx_w1`, … on the same server |
| `SPECDX_OLLAMA_URL` | `http://localhost:11434` | one Ollama host, or several separated by commas (worker *i* uses host *i* modulo the count) |
| `SPECDX_WORK` | `work/` in the checkout | run cache, baseline LLM cache, Tell cache, mutant packs |
| `TONOMYX_SRC` | — | a TonomyX checkout, only for the vendor check |

`run_all.sh` also reads these:

| Variable | Default | Meaning |
|---|---|---|
| `WORKERS` | 2 | `--workers` for the long steps; never above the host's `OLLAMA_NUM_PARALLEL` |
| `FOREGROUND` | 0 | 1 runs the long steps in the foreground (`all` sets it) |
| `LOCAL_BUDGET` | 8 | population runs in the local smoke run (0 skips it) |
| `SMOKE_DIR` | a new temporary directory | scratch directory for `local` |
| `GPU_SSH` | — | `user@gpu-host`, so the G3 check can read the host's Ollama log over SSH |
| `OLLAMA_LOG_CMD` | — | instead of `GPU_SSH`: a command printing the host's Ollama log between `$SINCE` and `$UNTIL` |

Each **frozen configuration** in `experiments/` (for example `p2-gate.json`) fixes the following, and nothing is changed by hand after a result has been seen:
- the population (`models_by_family`);
- the Tell variants and seeds;
- the run budgets;
- the development split (`dev_per_type`, `rng_seed`).

## Checking the machine

```bash
.venv/bin/specdx doctor                     # or: specdx doctor --profile gpu48
```

`doctor` checks:
- Python and the packages;
- PostgreSQL;
- the vendored TonomyX;
- every Ollama host: reachable, each candidate model present with native tool calling, and the same digest on every host;
- free disk space and, when `nvidia-smi` is available, VRAM.

It recommends a hardware profile and prints the `ollama pull` commands for missing models. Fix every `fail` line before running; a `warn` line is information. After pulling, load one model and check that `ollama ps` reports `100% GPU`.

| Profile | GPU memory | Candidate models | Suggested `--workers` |
|---|---|---|---:|
| `gpu16` | up to ~22 GB | qwen3:14b, qwen2.5:14b, qwen3:30b-a3b, gpt-oss:20b, mistral-small3.2:24b, granite3.3:8b, llama3.1:8b | 2 |
| `gpu24` | up to ~44 GB | gpu16 plus qwen3:32b, gemma3:27b | 3 |
| `gpu48` | up to ~76 GB | gpu24 plus llama3.3:70b and qwen2.5:72b (q4) | 4 |
| `gpu80` | more | gpu48 plus gpt-oss:120b | 6 |

A profile only widens the candidate list; the capability screen decides admission.

Every Ollama host should run with `OLLAMA_NUM_PARALLEL` at least equal to the number of workers sent to it. On the 48 GB host also set `OLLAMA_MAX_LOADED_MODELS=2` and `OLLAMA_KEEP_ALIVE=-1`, and pull exactly the tags of the population. A request that waits in Ollama's queue counts against the per-call timeout, and a timed-out interpreter call is cached as behaviour for good. When in doubt, use `--workers 1`.

## Running the experiments

`run_all.sh` drives everything. Each step checks the one before it, and re-running a step resumes from the caches.

### 1. Smoke tests on this machine (before any GPU time)

```bash
./run_all.sh local
```

It runs:
- the unit tests and the linter;
- a preview of the population rule;
- the Ollama and digest checks;
- `doctor`;
- an end-to-end SpecDx run on the base specification (budget `LOCAL_BUDGET`);
- every baseline once on that base with one model.

The baseline check fails if an LLM call came close to the context window. Everything is written to a scratch directory, never to `work/`, `results/` or `experiments/`.

### 2. The full pipeline on the GPU host

SpecDx and PostgreSQL run on this machine. Ollama runs on the GPU host, reached through an SSH tunnel:

```bash
ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -L 11435:localhost:11434 user@gpu-host
export SPECDX_OLLAMA_URL=http://localhost:11435
tmux new -s specdx                              # the pipeline takes days
./run_all.sh all                                # or: ./run_all.sh all --yes
```

`all` runs these steps in order, in the foreground:

| Step | What it does | Time |
|---|---|---|
| `g2` | admits the population from the cached capability screen and freezes it into `experiments/p2-gate.json` (committed); exit 3 if fewer than 2 families or 4 acting models | seconds |
| `preflight` | tools, database, the append-only run cache, Ollama version and pinned model digests, workers | seconds |
| `g3` | smoke run of the base specification at budget 60, then its checks: no queued timeout cached as behaviour, every population model has readings and accuracy ≥ 0.5 | hours |
| `gate` | SpecDx on the base and the 20 development mutants; verdict PASSED / NOT PASSED in `results/p2-gate.md`. Before launching, `all` shows the measured seconds per run and asks for confirmation (`--yes` skips the question; answering no stops `all` with exit 0) | days |
| `baselines-dev` | every baseline arm, on every population model, on the base and the development mutants | hours to days |
| `freeze-baselines` | chooses each baseline's model, strategy and threshold on the development set only. Writes and commits `experiments/p3-baselines-prereg.{json,md}`, which records the sha256 of the gate configuration and gate result and a fingerprint of the baseline LLM cache | seconds |
| `study` | SpecDx and the frozen baselines on the test mutants, run once | days |
| `report` | `results/p3-report.md`: tables and statistics for the development set, and for the test set once it exists | seconds |

If the gate is **NOT PASSED**, `all` still runs `baselines-dev` and the development report, freezes nothing, and exits 4: the test set stays locked.

### 3. One step at a time

```bash
./run_all.sh g2 | preflight | g3 | gate [--yes] | baselines-dev | freeze-baselines | study | report
./run_all.sh status        # running steps (pid, log tail), size of both caches, latest results
```

Run singly, the long steps (`gate`, `baselines-dev`, `study`) detach. Each writes a pid file and a log under `work/` (`work/p2-gate.*`, `work/p3-baselines-dev.*`, `work/p3-study.*`), so a closed terminal does not stop it. Pause and resume a detached step with `kill -STOP $(cat work/<step>.pid)` and `kill -CONT $(cat work/<step>.pid)`.

### Exit status of `run_all.sh`

| Code | Meaning |
|---|---|
| 0 | done (or `all` stopped because the gate launch was declined) |
| 1 | a check failed; stop and report |
| 2 | a usage or environment error |
| 3 | the population stop rule (too few families or acting models) |
| 4 | the test study is locked. Causes: the gate is not PASSED; the baseline pre-registration is not committed; the working tree has uncommitted changes; or the gate configuration or gate result changed since the freeze |

### Integrity of the comparison

- **The baselines run on SpecDx's own runs.** Every baseline reads the same specifications and, where it reads runs, the same cached SpecDx runs. The LLM-based baselines (BenchGuard, Who&When) call the same population models with the same digests, through a cached gateway.
- **One scoring code.** Every method is scored by the same matching code, with the same differential against the base specification and the same set of scorable mutants.
- **Choices are frozen on the development set.** Models, strategies and thresholds are chosen there and frozen before the test set runs. `specdx study` accepts only `experiments/p2-gate.json` and `experiments/p3-baselines-prereg.json`, both committed, on a clean tree.
- **Two cost views.** Costs are counted in model requests (tokens and durations wherever recorded). Each method is reported at its native protocol (Table A), and SpecDx is also reported at each LLM baseline's cost, read from its anytime checkpoints (Table B).

The baselines and how they are run are described in [`baseline/README.md`](baseline/README.md).

## The `specdx` command line

| Command | Purpose |
|---|---|
| `specdx doctor [--profile P]` | check the machine, the hosts and the models |
| `specdx build-tell` | extract and cache the Tell corpus |
| `specdx make-variants --model M --count N` | paraphrase Tell variant 0 into variants 1..N |
| `specdx capability MODEL…` | check native tool calling of local models |
| `specdx screen --profile P --out PREFIX [--workers N]` | run the capability screen and choose the population |
| `specdx screen --from SCREEN.json --out PREFIX` | re-apply the admission rules to a cached screen (no model call) |
| `specdx cache-info` | cached runs per model and digest, and per specification |
| `specdx simulate [--sweep S] [--replicates N] --out PREFIX` | the simulation study (no GPU, no database) |
| `specdx mutants` | build every mutant and list its ground truth |
| `specdx run [--variant base\|MUTANT] --config C --out OUT [--workers N]` | SpecDx on one specification |
| `specdx gate --config C --out OUT [--workers N]` | the go/no-go gate on the development mutants |
| `specdx baselines --split dev\|base --config C --out OUT [--models M…] [--workers N]` | every baseline arm on the development set (or the base only) |
| `specdx freeze-baselines --from DEV.json --out PREFIX [--cache CACHE] [--force]` | choose and pin each baseline's arm and threshold |
| `specdx study --split test --config experiments/p2-gate.json --baselines experiments/p3-baselines-prereg.json --out OUT` | the test set (exit 4 when locked) |
| `specdx report --dev DEV.json [--test TEST.json] --out REPORT.md` | tables A and B and the statistics |

`--workers N` runs N workers, each on its own database. A report produced with N workers is byte-identical to the one-worker report over the same cached runs.

## Outputs

| File | Contents |
|---|---|
| `results/p2-screen.{md,json}` | each candidate's act rate and valid-call rate, and the admitted population |
| `results/p2-smoke-base.json` | the G3 smoke run: findings, per-model accuracy, cell posteriors |
| `results/p2-gate.{md,json}`, `results/p2-gate-mutants.jsonl` | the gate verdict, its primary table on scorable mutants, and one line per mutant (findings, matches, run counts) |
| `results/p3-baselines-dev.json` | every baseline arm on the development set: findings, costs, failures, SpecDx checkpoints |
| `experiments/p3-baselines-prereg.{json,md}` | the frozen choices per baseline and the fingerprints they are bound to (committed) |
| `results/p3-study.json` | SpecDx and the frozen baselines on the test set, with the thresholds actually used |
| `results/p3-report.md` | Table A (typed and untyped F1, AP, Detection@1/3/5, costs), the paired comparison (exact McNemar, Holm, bootstrap CIs), Table B (equal cost) |

A gate or study headline is PASSED, NOT PASSED or INCOMPLETE (some specification could not run). A baseline marked INCOMPLETE failed on at least one specification and is scored on the rest.

## Caches, resuming and moving work

- **`work/runs.jsonl`** holds every interpreter run. It is append-only, and `run_all.sh` checks that its first 6505 lines never change.
  - A run key is a digest of the specification, scenario, interpreter, Tell texts, executor settings and the Ollama model digest.
  - A run with the same key is never executed twice.
- **`work/baseline-llm.jsonl`** holds every baseline LLM call, keyed by the model digest and the full request. It is append-only, and its prefix is fingerprinted at the freeze.
  - An Ollama outage is never cached; a model timeout or malformed reply is cached as an error and never counts as a finding.
- **Resume.** After a crash, a reboot or a dropped tunnel, run the same command again. Finished work is served from both caches. The advisory lock refuses a second SpecDx process on the same database.
- **Move.** Copy `work/` to another machine, for example `rsync -a work/ other-host:specdx/work/`. Every run whose model digest matches is reused there; `specdx cache-info` lists what is cached.
- **Keep the results.** `results/` is not tracked by git, so back it up separately. The committed pre-registration binds the gate result and the development results by their sha256.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `another process holds the SpecDx lock` | a SpecDx process (or pytest) is using the database; wait for it or stop it |
| `refusing …: ports 55432, 55433 …` | `SPECDX_DATABASE_URL` points at another project's database; use port 55434 |
| `InfrastructureError: Ollama failed …` | Ollama is down, out of memory or unreachable (tunnel); nothing was cached. Restart it and re-run the same command |
| `all calls timed out … Ollama may be wedged` | restart Ollama; lower `--workers` or `OLLAMA_NUM_PARALLEL` if the GPU is saturated |
| `Ollama does not list model …` / `DigestMismatch` | pull the population's exact tags on every host; a different digest would silently change the model, so the run stops |
| `… differs from the pack this code builds` | a mutant pack in `work/mutants/` was edited or built by other code; move it aside and it is rebuilt |
| `pdftotext` not found | install poppler, or copy `work/tell/` from a machine that built it |
| `nvidia-smi failed: … version mismatch` | the NVIDIA driver was updated without a reboot; reboot the GPU host before any run |
| runs take many minutes each | check that `ollama ps` shows `100% GPU`; a model larger than the GPU runs partly on the CPU |
| `local` fails with `near_ctx` | a baseline prompt plus its output budget reached the 16384-token context; report it before running the GPU steps |
| `run_all.sh study` exits 4 | see the exit-status table: the gate must be PASSED, the pre-registration committed, the tree clean, and the gate files unchanged since the freeze |

## Rules of the repository

- `vendor/` is never edited. Mutants and every scratch file live in `work/`.
- Nothing is imported from the FAA project or from the third-party clones in `baseline/`.
- Only local open-weight models are used, through Ollama.
- No human judgement enters any metric. Every label comes from a mutation operator, from TonomyX's recorded history, or from a published benchmark repair.
- `work/runs.jsonl` and `work/baseline-llm.jsonl` are append-only. Never edit or truncate them.
- Thresholds and choices are frozen before the data they are evaluated on is seen. The test set is run once.
