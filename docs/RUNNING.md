# Running SpecDx P1b

This guide takes a fresh machine to a finished P1b gate: setup, choosing a hardware profile, Ollama on this machine or on remote GPU hosts, the capability screen, the gate, pausing and resuming, moving the run cache, reading the results, and troubleshooting. A Vietnamese quick-start is at the end.

The design is in `docs/superpowers/specs/2026-10-11-specdx-v2-design.md` (a delta on `2026-10-06-specdx-design.md`). Every threshold is pre-registered in `experiments/p1b-preregistration.md`.

## 1. Prerequisites

| What | Version | Why |
|---|---|---|
| Python | 3.13 | SpecDx and the vendored TonomyX |
| Docker | any recent | PostgreSQL 16 for SpecDx, on port 55434 |
| Ollama | 0.30 or later, on this machine or on GPU hosts | the interpreters; models need native tool calling |
| poppler (`pdftotext`) | any | extracts the Tell corpus from the requirement PDFs, once |
| git, jq | any | setup, and freezing the screened population |

SpecDx never uses ports 55432 or 55433: they belong to other projects on the author's machine, and SpecDx refuses them.

## 2. Setup

```bash
git clone <this repository> specdx && cd specdx
python3.13 -m venv .venv
.venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"
docker compose up -d                      # PostgreSQL 16 on localhost:55434
TONOMYX_SRC=/path/to/TonomyX bash scripts/verify_vendor.sh   # optional: needs a TonomyX checkout
.venv/bin/python -m pytest                # every test; no model is called
```

Configuration comes from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `SPECDX_DATABASE_URL` | `postgresql://tonomyx:tonomyx@localhost:55434/tonomyx` | the SpecDx database; workers add databases `tonomyx_w0`, `tonomyx_w1`, … in the same server |
| `SPECDX_OLLAMA_URL` | `http://localhost:11434` | one Ollama host, or several separated by commas |
| `SPECDX_WORK` | `work/` in the checkout | the run cache (`runs.jsonl`), the Tell corpus cache (`tell/`) and the mutant packs (`mutants/`) |
| `TONOMYX_SRC` | `/home/ethan/AILab/TonomyX` | a TonomyX checkout, only for the vendor check |

## 3. Check the machine and choose a profile

```bash
.venv/bin/specdx doctor                   # or: specdx doctor --profile gpu24
```

`doctor` checks Python and the packages, PostgreSQL, the vendor, every Ollama host (reachable, each candidate model present, native tool calling, the same digest on every host), free disk and, when `nvidia-smi` is local, VRAM. It prints a recommended profile and the `ollama pull` commands for missing models. A `fail` line must be fixed first; a `warn` line is information. After pulling, load one model and check that `ollama ps` reports `100% GPU`.

| Profile | GPU memory | Candidates (family: models) | Suggested `--workers` |
|---|---|---|---:|
| gpu16 | up to about 22 GB | qwen: qwen3:14b, qwen2.5:14b, qwen3:30b-a3b · openai: gpt-oss:20b · mistral: mistral-small3.2:24b · ibm: granite3.3:8b · meta: llama3.1:8b | 2 |
| gpu24 | up to about 44 GB | gpu16, plus qwen3:32b and google: gemma3:27b | 3 |
| gpu48 | up to about 76 GB | gpu24, plus llama3.3:70b and qwen2.5:72b (q4) | 4 |
| gpu80 | more | gpu48, plus gpt-oss:120b | 6 |

A profile only widens the list of candidates. The capability screen decides who enters the population, and no rule or threshold depends on the profile. The profiles are in `experiments/profiles/`.

## 4. Ollama on this machine, on a remote host, or on several GPUs

On every Ollama host:

```bash
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_NUM_PARALLEL=2 ollama serve
```

- `--workers` must not exceed the `OLLAMA_NUM_PARALLEL` of the host it is sent to, and two workers serving different models need both models to fit on the GPU together. A request that waits in Ollama's queue (or for a model swap) counts against the 180 s per-call timeout, and a timed-out call is recorded as interpreter behaviour in the run cache, permanently. When in doubt, or when `OLLAMA_NUM_PARALLEL` is unset (it defaults to 1), use `--workers 1`.
- A report made with N workers is byte-identical to the one-worker report over the same cached runs. Fresh runs with `OLLAMA_NUM_PARALLEL` above 1 are not guaranteed to be bit-identical, because the model server batches inference.
- Several hosts: list them all, as in `export SPECDX_OLLAMA_URL=http://gpu1:11434,http://gpu2:11434`. Worker *i* uses host *i* modulo the number of hosts.
- Every host must hold the same build of each model. Pull the same tags everywhere; `doctor` fails when two hosts report different digests, and so does every run.
- SpecDx and PostgreSQL can run on a CPU machine; only Ollama needs the GPU.

## 5. The capability screen

```bash
.venv/bin/specdx screen --profile gpu16 --out results/p1b-screen --workers 1
```

The examples use `--workers 1`. Use `--workers 2` only on a host where `OLLAMA_NUM_PARALLEL` is at least 2 and both models fit on the GPU together (§4).

Each candidate runs the base specification on scenario `happy`, 3 Tell variants × 3 seeds, through the neutral harness. Under spec v3.1 §7 a model is admitted with act rate ≥ 0.3 and valid-call rate ≥ 0.8. The population keeps at most 2 models per family and 4 in all, and needs at least 2 families and at least 4 acting models. With fewer, the command exits with status 3 and prints STOP: the work stops and reports, and the thresholds are never lowered. (P1b screened under v2's act rate 0.6; P2 re-admits that cached screen, §11.)

Freeze the population into the gate configuration:

```bash
jq --slurpfile s results/p1b-screen.json '.models_by_family = $s[0].population' \
  experiments/p1b-gate.json > work/p1b-gate.json && mv work/p1b-gate.json experiments/p1b-gate.json
```

The gate refuses to run on any other population than the one `results/p1b-screen.json` froze.

## 6. The gate

A short smoke run first:

```bash
jq '.budget = 60' experiments/p1b-gate.json > work/smoke.json
.venv/bin/specdx run --variant base --config work/smoke.json --out work/smoke-base.json --workers 1
```

Check the pre-registered θ ordering before launching the gate. The population model with the highest κ must have a θ above the lowest θ of the others:

```bash
jq -r '.kappa as $k | .theta as $t
  | ($k | to_entries | max_by(.value) | .key) as $top
  | ([$t | to_entries[] | select(.key != $top) | .value] | min) as $low
  | "highest κ: \($top), θ \($t[$top]); lowest other θ \($low); ordering "
    + (if $t[$top] > $low then "OK" else "FAILS" end)' work/smoke-base.json
```

On `ordering FAILS`, do not launch the gate: P1b stops and reports.

Then the gate, detached from the terminal so that closing it does not stop the run:

```bash
setsid nohup .venv/bin/specdx gate --config experiments/p1b-gate.json \
  --out results/p1b-gate.json --workers 1 > work/p1b-gate.log 2>&1 < /dev/null &
echo $! > work/p1b-gate.pid
tail -f work/p1b-gate.log                 # one line per executed run
```

It runs the base specification, computes which mutants are scorable from the base runs alone, then runs the 20 development mutants. It writes `results/p1b-gate.json`, `results/p1b-gate.md` and `results/p1b-gate-mutants.jsonl`.

## 7. Pause, resume, and restart

- Pause: `kill -STOP $(cat work/p1b-gate.pid)`. Resume: `kill -CONT $(cat work/p1b-gate.pid)`.
- Never start a second SpecDx process on the same database. The advisory lock refuses it with "another process holds the SpecDx lock"; stop the first one, or wait for it.
- After a crash, a reboot or an Ollama outage, run the same command again. Every finished run is in the cache, so the gate resumes where it stopped. Its budgets are spent as before up to the crash point; runs that were in flight at the crash are executed again and are fresh samples, so they can differ from the lost ones.

## 8. Moving the run cache

The cache is portable. A run key is a digest of the specification, the scenario, the interpreter, the Tell texts, the executor settings and the Ollama model digest. Copy `work/` (or the directory `SPECDX_WORK` names) to the other machine; every run whose model digest matches is reused there.

```bash
rsync -a work/ other-host:specdx/work/
.venv/bin/specdx cache-info               # runs per model and digest, and per specification
```

## 9. Reading the results

| File | What it holds |
|---|---|
| `results/p1b-replay.md` | the offline replay on the P1 cache: κ, θ, the GO criterion, observability per item |
| `results/p1b-screen.md`, `.json` | each candidate's act rate and valid-call rate, and the population |
| `results/p1b-gate.md` | the verdict: the primary table (scorable mutants) decides; the secondary table scores all 20 and counts the excluded ones; the literal typed-against-typed table is for transparency |
| `results/p1b-gate-mutants.jsonl` | one line for the base and one per mutant: findings, matches, run counts, scorability, the Unobservable list, and the base-to-mutant shift of each item's difficulty δ |

The headline is PASSED, NOT PASSED or INCOMPLETE (a mutant could not run). After a second NOT PASSED, P1b stops; a third iteration on the development split needs the author's decision. The test split is never run during P1b.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `another process holds the SpecDx lock` | A SpecDx process (or pytest) uses the database. Wait for it, or stop it. |
| `refusing …: ports 55432, 55433 …` | `SPECDX_DATABASE_URL` points at another project's database. Use port 55434. |
| `InfrastructureError: Ollama failed 5 times` | Ollama is down or out of memory. Nothing was cached; restart Ollama and rerun the same command. |
| `all calls timed out … Ollama may be wedged` | Restart Ollama; lower `--workers` or `OLLAMA_NUM_PARALLEL` if the GPU is saturated. |
| `Ollama does not list model …` | Pull the model on every host (see `specdx doctor`). |
| `model … has different digests across hosts` | Pull the same tag again on every host. |
| `… differs from the pack this code builds` | A mutant pack in `work/mutants/` was edited or built by other code. Move it aside; it is rebuilt. |
| `pdftotext` not found | Install poppler, or copy a `work/tell/` from a machine that built it. |
| `nvidia-smi failed: … Driver/library version mismatch` | The NVIDIA driver was updated without a reboot, and Ollama cannot use the GPU. Reboot the GPU host (or reload the `nvidia` kernel modules) before any run. |
| Runs take many minutes each | Check that `ollama ps` shows `100% GPU` for the loaded model; a model larger than the GPU runs partly on the CPU. |

## 11. P3: SOPBench and τ²-bench

P3 runs every experiment from one script, `./run_all.sh`. Each finished stage writes a Markdown report to `results/p3/<stage>.md` (and its metadata to `<stage>.json`), and **`results/p3/STATUS.md`** is the dashboard: one row per stage with its status, verdict, key number and a link to the report. Stages that later plans implement (P3b, P3c, P3d) show as `pending (plan Px)`.

```bash
./run_all.sh all             # every implemented stage in order; skips stages already done
./run_all.sh local           # one stage
./run_all.sh s3-v4 s3-placebo   # several stages, in order; stops at the first non-zero exit
./run_all.sh status          # rebuild results/p3/STATUS.md
less results/p3/STATUS.md
```

`all` runs, in order, `local`, `fetch-sopbench` (the only stage that uses the network), `replication` and then the four P4a stages. With several stage names, `run_all.sh` checks every name first and then runs them in order, stopping at the first non-zero exit. `all` skips a stage whose recorded exit code is 0 (`specdx p3 status --done <stage>`), and stops at the first non-zero exit. The skip ignores the commit: when a done stage was recorded at another commit than HEAD, `specdx p3 status` (and so `run_all.sh`) prints a warning, and deleting `results/p3/<stage>.json` makes the stage run again. A stage that is not implemented yet prints `<stage>: pending — implemented in plan <Px>` and exits 0. While a stage runs, `work/p3-<stage>.pid` holds its pid and STATUS.md shows `running`; the file is removed when the stage ends, however it ends. `P3_RESULTS` and `P3_WORK` override `results/p3` and `work`.

| Stage | What it does | Plan | State | Report |
|---|---|---|---|---|
| `local` | Local checks and tests | P3a | implemented | `results/p3/local.md` |
| `fetch-sopbench` | Fetch SOPBench data | P3a | implemented | `results/p3/fetch-sopbench.md` |
| `replication` | Offline replication on SOPBench | P3a | implemented | `results/p3/replication.md` |
| `s3-g0v4` | S3 calibration and G0v4 | P4a | implemented | `results/p3/s3-g0v4.md` |
| `s3-witness` | S3 execution witnesses | P4a | implemented | `results/p3/s3-witness.md` |
| `s3-v4` | S3 v4 replication and ablations | P4a | implemented | `results/p3/s3-v4.md` |
| `s3-placebo` | S3 permutation placebo | P4a | implemented | `results/p3/s3-placebo.md` |
| `s3-screen` | S3 screen | P3b | pending | `results/p3/s3-screen.md` |
| `s3-dev` | S3 development runs | P3b | pending | `results/p3/s3-dev.md` |
| `s3-calibrate` | S3 calibration | P3b | pending | `results/p3/s3-calibrate.md` |
| `s3-gate` | S3 gate | P3b | pending | `results/p3/s3-gate.md` |
| `s3-baselines-dev` | S3 baselines on dev | P3b | pending | `results/p3/s3-baselines-dev.md` |
| `s2-screen` | S2 screen | P3c | pending | `results/p3/s2-screen.md` |
| `s2-dev` | S2 development runs | P3c | pending | `results/p3/s2-dev.md` |
| `s2-calibrate` | S2 calibration | P3c | pending | `results/p3/s2-calibrate.md` |
| `s2-gate` | S2 gate | P3c | pending | `results/p3/s2-gate.md` |
| `s2-baselines-dev` | S2 baselines on dev | P3c | pending | `results/p3/s2-baselines-dev.md` |
| `freeze` | Freeze the preregistration | P3b | pending | `results/p3/freeze.md` |
| `s3-test` | S3 test split | P3b | pending | `results/p3/s3-test.md` |
| `s2-test` | S2 test split | P3c | pending | `results/p3/s2-test.md` |
| `s2-real` | S2 real-subject runs | P3c | pending | `results/p3/s2-real.md` |
| `ablations` | Ablations | P3d | pending | `results/p3/ablations.md` |
| `status` | Status dashboard | P3a | implemented | `results/p3/status.md` |

Exit codes: 0 done; 1 a check failed; 2 a usage or environment error; 3 a screen stop; 4 a test split is locked.

Only `fetch-sopbench` uses the network (the pinned SOPBench commit on raw.githubusercontent.com). Nothing in P3a calls Ollama, a hosted model, or the database, and `work/runs.jsonl` is never opened. Every S3 result is labelled `provisional: V3Params calibrated on CASE-0031, recalibrated for S3 in P3b`. Later local SOPBench runs, whose results depend on set iteration order, must be started with `PYTHONHASHSEED=0`.

### P4a stages and smoke mode

The four P4a stages need the published trajectories (`work/sopbench-output`, from `fetch-sopbench`); a stage without them exits 2 and never fetches.

- `s3-g0v4` calibrates one simulated layout per development domain (that domain's check and user-goal groups, raters and difficulty spreads) and runs G0v4.
  - It runs `P4_JOBS` fits in parallel (default: every core), with one BLAS thread per process.
  - Each finished fit is cached in `$P3_WORK/g0v4-cache`, so rerunning the stage after a crash skips the fits already done.
  - On a PASS of the pre-registered run (10 replicates, the full grid, the default seeds, not smoke) it freezes `experiments/p4-s3-v4-params.json`, with the commit and the sha256 of the search JSON. A PASS of any other configuration is not frozen (exit 1). A frozen file that holds other parameters is never overwritten (exit 2).
- `s3-witness` replays every task on SOPBench's strict system.
- `s3-v4` and `s3-placebo` use only the four held-out domains and the frozen parameters.

The placebo verdict is PASS only with at least two evaluable domains (real v4 count above 0) and every evaluable v4 ratio at most 0.2. tau_u comes from the G0v4 simulation, not from theory.

**`./run_all.sh all` includes the official P4a stages** (`s3-g0v4` can run for hours and freezes the v4 parameters on PASS), so run `all` on the 48 GB host only; a G0v4 FAIL (exit 1) also stops `all` before the later stages, and `all` prints a warning to that effect.

Official P4a results come only from the 48 GB host. On this machine run the smoke mode, a quick system check:

```bash
./run_all.sh smoke           # the four P4a stages, shrunk, in order
P4_SMOKE=1 ./run_all.sh s3-witness   # one stage in smoke mode
less results/p3/smoke/STATUS.md
```

Smoke runs 2 G0v4 replicates on a one-point grid, 20 witnessed tasks per domain, the first development domain for `s3-v4` and `s3-placebo` (never a held-out one) and 2 placebo seeds. Reports go to `results/p3/smoke/` (override with `P3_RESULTS`), start with `SMOKE RUN — system check only, not a result`, and have their own `STATUS.md` titled `SMOKE — not results`. A smoke run never freezes parameters (`s3-g0v4` writes `smoke-v4-params.json` in the smoke directory instead, which a smoke `s3-v4` or `s3-placebo` uses when nothing is frozen) and never marks an official stage done. `smoke` runs all four stages even when one fails, prints a summary (stage, exit code, verdict) and exits with the largest stage code, so a crash (2 or more) is never hidden by an expected FAIL (1): the smoke placebo uses one domain, so it always ends FAIL (1). `P4_SMOKE` must be unset, empty, `0` or `1`; any other value exits 2. Smoke mode applies to the P4a stages only, so `P4_SMOKE=1 ./run_all.sh <other stage>` exits 2, pending stages included.

### The official P4a run on the 48 GB host

Official P4a results come only from the 48 GB host. The smoke output under `results/p3/smoke/` is never results: do not cite, freeze or commit it. In order:

```bash
# 0. A fresh host: fetch SOPBench's published trajectories (the only network use).
./run_all.sh fetch-sopbench
# 1. G0v4 on the development domains, P4_JOBS fits at a time (default: every core).
P4_JOBS=N ./run_all.sh s3-g0v4
# 2. Only on PASS: commit the frozen parameters, on that host.
git add experiments/p4-s3-v4-params.json
git commit -m "results: P4a — frozen v4 parameters for S3 (G0v4)"
# 3. The remaining stages, in order; the run stops at the first non-zero exit.
./run_all.sh s3-witness s3-v4 s3-placebo
# 4. Send results/p3/*.md back for review.
```

`./run_all.sh all` does steps 0, 1 and 3 in one go (it also runs `local` and `replication` first, and skips a stage already recorded done); commit the frozen file (step 2) after it. A G0v4 FAIL (exit 1) freezes nothing, and the later stages then exit 2 for want of parameters; send `results/p3/s3-g0v4.md` back instead of continuing. If `s3-g0v4` stops part-way (a crash, or Ctrl-C), run step 1 again: the fits in `work/g0v4-cache` are not redone.

G0v4 cost. The official grid is 6 prior settings (3 `cell_sd` x 2 `lambda_prior`; the 27 threshold settings are scored afterwards at no fit cost) and 10 replicates per arm over 4 arms: (6 + 1 held-out) x 4 x 10 = 280 v4 fits and 40 v3 fits. Each simulated domain is now one development domain, fitted on its own, so a fit is far cheaper than the old pooled one (729 s). The smoke run on this machine (2026-11-25, `P4_JOBS=20`, the first prior setting) measured v4 fits of 18 to 221 s and v3 fits of 20 to 43 s; the slowest were the hotel layout's homogeneous and base arms. Extrapolated with those times, the run costs about 36,000 s of CPU, about 10 CPU-hours, and about 0.6 hours at `P4_JOBS=20`. The stage prints its own estimate on stderr after its first fits (one per kind, arm and layout, which the run keeps) and records it in the report; on the smoke run, timing the base arm only (the estimator before it timed every arm), it estimated 479 s and took 466 s. Other prior settings and fits that do not converge can run longer, so allow up to about 2 hours at `P4_JOBS=20`.

The CASE-0031 steps (`g2`, `preflight`, `g3`, `gate`, `baselines-dev`, `freeze-baselines`, `study`, `report`) are no longer in `run_all.sh`. Their `specdx` commands remain, and §12 and §13 are kept as the record of how they ran.

## 12. Reproducing G0 and G1 (P2a, CPU only)

The records `results/p2-calibration.json`, `results/p2-g0.*`, `results/p2-sim.*` and `results/p2-g1.md` (kept on disk; `results/` is git-ignored since 2026-10-22, so their committed versions are in the history up to `d444736`) were produced by the commands below, with these commits:
- G0 ran on the code of `46beddf` and was recorded in `a97a67c`, together with the calibration, the freeze of `V3Params` into `specdx/infer/anchored.py` and the simulation study. Between `46beddf` and `a97a67c` the only change under `specdx/` is that frozen block, so the simulation and the fit are the same code.
- G1 ran on the code of `a97a67c` and was recorded in `e9aac9b`. Both read the cached P1 run log, whose `md5sum` must be `db1e1823c66df102648c69eb6a8789b3` before and after.

To reproduce, write to a scratch directory outside the repository's `work/` (the run cache; `p2_replay` refuses `--out` under it) and outside `results/` and `experiments/`; the block below makes one with `mktemp -d`, so run it as one block from the repository root. `g0 run` refuses to overwrite the frozen `experiments/p2-g0-params.json` unless `--params-out` is given. The cache is read-only: set `SPECDX_WORK` on the one command that reads it, and never export it.

```bash
out=$(mktemp -d); echo "$out"
SPECDX_WORK=/home/ethan/AILab/SpecDx/work .venv/bin/python -m specdx.sim.calibrate \
  --out "$out/p2-calibration.json"                              # seconds
.venv/bin/python -m specdx.sim.g0 run --calibration "$out/p2-calibration.json" \
  --out "$out/p2-g0" --params-out "$out/p2-g0-params.json"      # about 56 min (3354 s recorded); exit 0 PASS, 1 FAIL, 2 error
.venv/bin/specdx simulate --sweep all --config "$out/p2-calibration.json" \
  --params "$out/p2-g0-params.json" --out "$out/p2-sim" > /dev/null   # about 20 min (1195 s recorded)
SPECDX_WORK=/home/ethan/AILab/SpecDx/work .venv/bin/python experiments/p2_replay.py \
  --out "$out/p2-g1.md"                                         # about 20 s; exit 0 PASS, 1 FAIL, 2 error
```

Each output should equal the committed file: `cmp "$out/p2-calibration.json" results/p2-calibration.json`; for G0, `diff <(jq -S 'del(.seconds, .calibration)' "$out/p2-g0.json") <(jq -S 'del(.seconds, .calibration)' results/p2-g0.json)` (the run time and the calibration path differ by construction); `diff "$out/p2-g1.md" results/p2-g1.md`. The G1 replay uses the frozen `V3Params` in the code, so it is the same at any later commit.

## 13. The comparison study (baselines)

> **Historical.** This section records how the CASE-0031 comparison study was meant to run. The `run_all.sh` steps it names (`g2`, `preflight`, `g3`, `gate`, `baselines-dev`, `freeze-baselines`, `study`, `report`, and `all --yes`) no longer exist: `./run_all.sh` now drives the P3 stages of §11. The `specdx` commands at the end of this section still exist.

Once the gate (§11) has finished, P3 compares SpecDx with the baselines (B1 to B5, the truth-inference arms and JEST, and the gate's secondary arms B2-naive and B3-2PL, untuned). `run_all.sh` drives it; each step checks the one before it.

**Setup.** Install the baselines' dependencies once: `.venv/bin/python -m pip install -e ".[baselines]"` (crowd-kit, python-sat, and BenchGuard pinned at commit `7d6a52e`). The LLM arms reuse the population models frozen in `experiments/p2-gate.json`, through the same `SPECDX_OLLAMA_URL` tunnel, so nothing extra needs to be pulled; a model that is missing, or has another digest than the population runs, stops the run before any call. The clones under `baseline/` are reference copies only: nothing is imported from them.

**The steps, in order:**

| Step | Writes | Notes |
|---|---|---|
| `./run_all.sh baselines-dev` | `results/p3-baselines-dev.json` | every arm, every population model, on the dev split; detached (pid and log `work/p3-baselines-dev.*`) |
| `./run_all.sh freeze-baselines` | `experiments/p3-baselines-prereg.{json,md}` | chooses each baseline's arm and τ on dev only, records the line count and md5 of `work/baseline-llm.jsonl` and the sha256 of `experiments/p2-gate.json` and `results/p2-gate.json`, and commits both files (`results/` is git-ignored, so the gate result and the dev results are bound by their recorded sha256 instead); a committed pre-registration is never replaced |
| `./run_all.sh study` | `results/p3-study.json` | SpecDx and the frozen baselines on the test split (days); detached (`work/p3-study.*`); the result embeds the pre-registration's sha256 and the τ it used per arm |
| `./run_all.sh report` | `results/p3-report.md` | tables A and B and the statistics, for dev, and for test once `results/p3-study.json` exists (with the τ the study embedded) |

`study` exits 4 unless the gate PASSED (`results/p2-gate.json`), `experiments/p3-baselines-prereg.json` is committed, the working tree has no uncommitted changes (spec §3.7), and `experiments/p2-gate.json` and `results/p2-gate.json` still have the sha256 the freeze recorded (spec §9.8). Before launching it checks that the first 6505 lines of `work/runs.jsonl` and the baseline cache lines recorded at the freeze are unchanged. `./run_all.sh all [--yes]` runs g2, preflight, G3, the gate and the steps above one after another in the foreground (`FOREGROUND=1`; run it inside tmux). Before the gate it reports the measured seconds per run and asks; `--yes` skips the question, and answering no stops `all` with exit 0. When the gate is NOT PASSED, `all` runs `baselines-dev` and the dev report, freezes nothing, and exits 4. In the foreground each long step still writes its pid to `work/<step>.pid` while it runs, so `./run_all.sh status` (the gate, baselines-dev and study processes, and the size of both caches) and the guard against a second process see it. `./run_all.sh s2` exits 2: S2 (τ²-bench) is not built yet.

Before any GPU time, `./run_all.sh local` also runs every baseline arm on its smoke base with one model, and fails when an LLM call's prompt plus output budget reached `num_ctx` (16384): each result counts those calls per arm in `near_ctx`.

The commands `run_all.sh` runs, for reference (`study` accepts only these two files):

```bash
SPECDX_WORK=/home/ethan/AILab/SpecDx/work SPECDX_OLLAMA_URL=http://localhost:11435 \
  .venv/bin/specdx baselines --split dev --config experiments/p2-gate.json \
  --out results/p3-baselines-dev.json --workers 2
.venv/bin/specdx freeze-baselines --from results/p3-baselines-dev.json \
  --out experiments/p3-baselines-prereg --cache work/baseline-llm.jsonl \
  --gate-config experiments/p2-gate.json --gate-result results/p2-gate.json
SPECDX_WORK=/home/ethan/AILab/SpecDx/work SPECDX_OLLAMA_URL=http://localhost:11435 \
  .venv/bin/specdx study --split test --config experiments/p2-gate.json \
  --baselines experiments/p3-baselines-prereg.json --out results/p3-study.json --workers 2
.venv/bin/specdx report --dev results/p3-baselines-dev.json --test results/p3-study.json \
  --out results/p3-report.md
```

**Pause and resume.** Pause and resume a detached step as in §7 (`kill -STOP` / `kill -CONT` on the pid in `work/p3-*.pid`), and never start a second process on the same database. If a step stops (the tunnel drops, the machine restarts), re-run the same step: SpecDx's runs come from `work/runs.jsonl` and the baselines' LLM calls from `work/baseline-llm.jsonl`, so the finished work is served from both caches and the rerun writes the same result file. Both caches are append-only.

## Bắt đầu nhanh (tiếng Việt)

1. Cài Python 3.13, Docker, Ollama (0.30 trở lên) và poppler.
2. Cài đặt:
   ```bash
   python3.13 -m venv .venv && .venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"
   docker compose up -d
   .venv/bin/python -m pytest
   ```
3. Kiểm tra máy và chọn cấu hình: `.venv/bin/specdx doctor`. Lệnh này gợi ý một profile (gpu16, gpu24, gpu48, gpu80) và in các lệnh `ollama pull` còn thiếu.
4. Nếu Ollama chạy ở máy GPU khác: trên máy đó chạy `OLLAMA_HOST=0.0.0.0:11434 OLLAMA_NUM_PARALLEL=2 ollama serve`; trên máy này đặt `export SPECDX_OLLAMA_URL=http://may-gpu:11434` (nhiều máy thì cách nhau bằng dấu phẩy). Lưu ý: `--workers` không được vượt quá `OLLAMA_NUM_PARALLEL` của máy Ollama (mặc định 1), vì yêu cầu phải xếp hàng có thể hết thời gian chờ và bị ghi vĩnh viễn vào cache như hành vi của mô hình; nếu không chắc, dùng `--workers 1`.
5. Sàng lọc năng lực: `.venv/bin/specdx screen --profile gpu16 --out results/p1b-screen --workers 1`. Mã thoát 3 nghĩa là STOP: theo spec v3.1 §7 (act rate ≥ 0,3), ít hơn 2 họ mô hình hoặc ít hơn 4 mô hình hành động được nhận, công việc dừng. Các lệnh ở đây dùng `--workers 1`; chỉ dùng `--workers 2` khi máy Ollama có `OLLAMA_NUM_PARALLEL` ≥ 2 và cả hai mô hình cùng vừa trong GPU (mục 4).
6. Ghi quần thể vào cấu hình gate (lệnh `jq` ở mục 5), rồi chạy gate nền: `setsid nohup .venv/bin/specdx gate --config experiments/p1b-gate.json --out results/p1b-gate.json --workers 1 > work/p1b-gate.log 2>&1 < /dev/null &`.
7. Tạm dừng: `kill -STOP <pid>`; chạy tiếp: `kill -CONT <pid>`. Không bao giờ chạy hai tiến trình trên cùng một cơ sở dữ liệu. Nếu bị ngắt, chạy lại đúng lệnh cũ: các lần chạy đã xong được lấy từ bộ nhớ đệm.
8. Kết quả nằm ở `results/p1b-gate.md`.
