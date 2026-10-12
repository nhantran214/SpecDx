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

- `OLLAMA_NUM_PARALLEL` must be at least the number of workers sent to that host; otherwise the workers queue behind each other.
- A report made with N workers is byte-identical to the one-worker report over the same cached runs. Fresh runs with `OLLAMA_NUM_PARALLEL` above 1 are not guaranteed to be bit-identical, because the model server batches inference.
- Several hosts: list them all, as in `export SPECDX_OLLAMA_URL=http://gpu1:11434,http://gpu2:11434`. Worker *i* uses host *i* modulo the number of hosts.
- Every host must hold the same build of each model. Pull the same tags everywhere; `doctor` fails when two hosts report different digests, and so does every run.
- SpecDx and PostgreSQL can run on a CPU machine; only Ollama needs the GPU.

## 5. The capability screen

```bash
.venv/bin/specdx screen --profile gpu16 --out results/p1b-screen --workers 2
```

Each candidate runs the base specification on scenario `happy`, 3 Tell variants × 3 seeds, through the neutral harness. A model is admitted with act rate ≥ 0.6 and valid-call rate ≥ 0.8. The population keeps at most 2 models per family and 4 in all, and needs at least 2 families. With fewer, the command exits with status 3 and prints STOP: P1b stops and reports, and the thresholds are never lowered.

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
.venv/bin/specdx run --variant base --config work/smoke.json --out work/smoke-base.json --workers 2
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
  --out results/p1b-gate.json --workers 2 > work/p1b-gate.log 2>&1 < /dev/null &
echo $! > work/p1b-gate.pid
tail -f work/p1b-gate.log                 # one line per executed run
```

It runs the base specification, computes which mutants are scorable from the base runs alone, then runs the 20 development mutants. It writes `results/p1b-gate.json`, `results/p1b-gate.md` and `results/p1b-gate-mutants.jsonl`.

## 7. Pause, resume, and restart

- Pause: `kill -STOP $(cat work/p1b-gate.pid)`. Resume: `kill -CONT $(cat work/p1b-gate.pid)`.
- Never start a second SpecDx process on the same database. The advisory lock refuses it with "another process holds the SpecDx lock"; stop the first one, or wait for it.
- After a crash, a reboot or an Ollama outage, run the same command again. Every finished run is in the cache, so the gate resumes where it stopped and spends its budgets exactly as before.

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

## Bắt đầu nhanh (tiếng Việt)

1. Cài Python 3.13, Docker, Ollama (0.30 trở lên) và poppler.
2. Cài đặt:
   ```bash
   python3.13 -m venv .venv && .venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"
   docker compose up -d
   .venv/bin/python -m pytest
   ```
3. Kiểm tra máy và chọn cấu hình: `.venv/bin/specdx doctor`. Lệnh này gợi ý một profile (gpu16, gpu24, gpu48, gpu80) và in các lệnh `ollama pull` còn thiếu.
4. Nếu Ollama chạy ở máy GPU khác: trên máy đó chạy `OLLAMA_HOST=0.0.0.0:11434 OLLAMA_NUM_PARALLEL=2 ollama serve`; trên máy này đặt `export SPECDX_OLLAMA_URL=http://may-gpu:11434` (nhiều máy thì cách nhau bằng dấu phẩy).
5. Sàng lọc năng lực: `.venv/bin/specdx screen --profile gpu16 --out results/p1b-screen --workers 2`. Mã thoát 3 nghĩa là STOP: ít hơn 2 họ mô hình đạt, P1b dừng.
6. Ghi quần thể vào cấu hình gate (lệnh `jq` ở mục 5), rồi chạy gate nền: `setsid nohup .venv/bin/specdx gate --config experiments/p1b-gate.json --out results/p1b-gate.json --workers 2 > work/p1b-gate.log 2>&1 < /dev/null &`.
7. Tạm dừng: `kill -STOP <pid>`; chạy tiếp: `kill -CONT <pid>`. Không bao giờ chạy hai tiến trình trên cùng một cơ sở dữ liệu. Nếu bị ngắt, chạy lại đúng lệnh cũ: các lần chạy đã xong được lấy từ bộ nhớ đệm.
8. Kết quả nằm ở `results/p1b-gate.md`.
