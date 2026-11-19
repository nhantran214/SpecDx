# SpecDx P3a — SOPBench offline and the P3 runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SpecDx diagnoses SOPBench (S3) offline from the trajectories SOPBench published, and a new `run_all.sh` runs the P3 stages and writes one Markdown report per stage plus `results/p3/STATUS.md`.

**Architecture:**
- A new package `specdx/subjects/sopbench/` turns SOPBench into the objects SpecDx's inference already consumes: a `TellCorpus`, `Cell`s and `Reading`s. It reads vendored task data and downloaded trajectory JSON only, with no database, Ollama or TonomyX.
- Diagnosis calls the existing domain-agnostic functions `fit_anchored`, `type_cells` and `soft_hitting_sets`.
- A small reporting module writes stage reports and the status dashboard. `specdx p3 …` subcommands use it, and `run_all.sh` calls them.

**Tech Stack:** Python 3.13 (`.venv`), numpy (existing), pytest, ruff, bash, jq. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-19-specdx-s2-s3-design.md`, especially §5.2, §8.4, §13c and §13d (§13d overrides §5.2's cells). This is plan 1 of 4:
- **P3a** (this plan): S3 offline and the runner.
- **P3b:** S3 on the GPU.
- **P3c:** S2, τ²-bench.
- **P3d:** the new baselines, ablations and the type-B table.

## Global Constraints

**Repository**
- `vendor/` is never edited after the vendoring task.
- Nothing is imported from `baseline/` clones, `/home/ethan/AILab/FAA` or `aebench`.
- Code, comments and docs are in English. Ruff: `line-length = 100`, rules E, F, I, UP, B, SIM, target py312.

**SOPBench pin**
- SOPBench is pinned at commit `8790ed7406acbd22270f90bd1c4038de3fc5db8f` (2026-09-13).
- Code is MIT ("The SOPBench Authors"); data is CC BY 4.0.

**Network and runs**
- Network access only in `specdx p3 fetch-sopbench`, which downloads from `https://raw.githubusercontent.com/Leezekun/SOPBench/8790ed7406acbd22270f90bd1c4038de3fc5db8f/<path>`. Never in tests.
- No hosted model API is called. No interpreter run, no Ollama call and no database access happen in this plan.
- `work/runs.jsonl` is never opened. No CASE-0031 run is made. The existing CASE-0031 code and its tests stay and must keep passing.

**Inference**
- `V3Params` is not changed. Every S3 result made with it is labelled `provisional: V3Params calibrated on CASE-0031, recalibrated for S3 in P3b`.
- S3 cells, anchors, readings, clean, V(b) and exclusions are exactly those of design §13d.

**Reports**
- Reports go to `results/p3/<stage>.md`, metadata to `results/p3/<stage>.json`, and the dashboard to `results/p3/STATUS.md`.
- Exit codes: 0 done; 1 a check failed; 2 a usage or environment error; 3 a screen stop; 4 a test split locked.

## Review Focus

1. **A truncated or empty trajectory file.** SOPBench ships files of 2 bytes and 4.7 kB. Expected: excluded at selection by size, and a JSON-invalid file raises `TrajectoryError` naming the file. Never a crash in the middle of a domain. Tested in Tasks 2 and 3.
2. **A trajectory whose records do not line up with the vendored tasks** (different count, or a different `user_prompt` at the same index). Expected: `TrajectoryMismatch`; the file is excluded and listed in the report. Never silently joined to the wrong task. Tested in Task 3.
3. **A model name absent from `MODEL_FAMILIES`.** Expected: selection fails with the unknown names listed, rather than assigning a guessed family. Tested in Task 2.
4. **A domain where every rater agrees with every anchor.** Expected: zero findings, the report still written, and the fit reported with its convergence. Tested in Task 5.
5. **Re-running `run_all.sh all`.** Expected: a stage whose metadata says exit 0 is skipped; a failed stage runs again. Tested in Task 6.

---

## File structure

| File | Responsibility |
|---|---|
| `vendor/sopbench/` | Byte-identical `git archive` of SOPBench `env data reference LICENSE DATA_LICENSE README.md requirements.txt` at the pin |
| `specdx/subjects/__init__.py`, `specdx/subjects/sopbench/__init__.py` | Package markers; `SOPBENCH_COMMIT`, `SOPBENCH_DIR`, `DOMAINS` |
| `specdx/subjects/sopbench/trajectories.py` | File names → selection → download with sha256 manifest |
| `specdx/subjects/sopbench/load.py` | Vendored tasks and trajectory records → `S3Task`, `S3Run` |
| `specdx/subjects/sopbench/cells.py` | Tell corpus, `Cell`s, `Reading`s, V(b) per §13d |
| `specdx/subjects/sopbench/diagnose.py` | Per-domain fit, typing, hitting sets → `Defect`s; population compositions |
| `specdx/report/__init__.py`, `specdx/report/stages.py` | Stage table, report writer, `STATUS.md`, skip rule |
| `specdx/report/p3.py` | The P3 stage functions this plan implements (`local`, `fetch-sopbench`, `replication`) |
| `specdx/cli.py` | New `p3` subcommand group |
| `run_all.sh` | Rewritten for P3 (§13c) |
| `experiments/p3-sopbench-trajectories.json` | Committed manifest of selected files with sha256 |
| `docs/RUNNING.md`, `README.md` | The P3 run guide replaces the P2b section |
| `tests/subjects/…`, `tests/report/…`, `tests/test_docs.py` | Tests |

---

### Task 1: Vendor SOPBench

**Files:**
- Create: `vendor/sopbench/**` (via `git archive`)
- Modify: `vendor/VENDORED.md`, `scripts/verify_vendor.sh`
- Create: `specdx/subjects/__init__.py`, `specdx/subjects/sopbench/__init__.py`
- Test: `tests/subjects/__init__.py`, `tests/subjects/test_sopbench_vendor.py`

**Interfaces:**
- Produces: `SOPBENCH_COMMIT: str` = `"8790ed7406acbd22270f90bd1c4038de3fc5db8f"`; `SOPBENCH_DIR: Path` = `REPO_ROOT / "vendor" / "sopbench"`; `DOMAINS: tuple[str, ...]` = `("bank", "dmv", "healthcare", "hotel", "library", "online_market", "university")`.

- [ ] **Step 1: Write the failing test**

```python
TASKS = {"bank": 134, "dmv": 97, "healthcare": 124, "hotel": 195, "library": 66,
         "online_market": 172, "university": 42}

def test_vendored_sopbench_has_every_domain_with_its_task_count():
    for d in DOMAINS:
        raw = json.loads((SOPBENCH_DIR / "data" / f"{d}_tasks.json").read_text(encoding="utf-8"))
        assert sum(len(v) for v in raw.values()) == TASKS[d]
    assert sum(TASKS.values()) == 830

def test_vendored_sopbench_carries_both_licences_and_no_output():
    assert "MIT" in (SOPBENCH_DIR / "LICENSE").read_text(encoding="utf-8")
    assert "CC BY 4.0" in (SOPBENCH_DIR / "DATA_LICENSE").read_text(encoding="utf-8")
    assert not (SOPBENCH_DIR / "output").exists()

def test_the_pin_is_recorded():
    assert SOPBENCH_COMMIT in (REPO_ROOT / "vendor" / "VENDORED.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run it.** `.venv/bin/python -m pytest tests/subjects/test_sopbench_vendor.py -q -o addopts=""` → FAIL (`ModuleNotFoundError: specdx.subjects`).
- [ ] **Step 3: Vendor.**
  - Clone `https://github.com/Leezekun/SOPBench` into the session scratchpad.
  - Check out `8790ed7…`.
  - Run `git archive 8790ed7 env data reference LICENSE DATA_LICENSE README.md requirements.txt | tar -x -C vendor/sopbench`.
  - Add a `## sopbench` section to `VENDORED.md`, in the format of the `tonomyx` section: source URL, commit, date, the `git archive` command, licences, and "no `output/`: published trajectories are fetched by `specdx p3 fetch-sopbench`".
  - Extend `scripts/verify_vendor.sh` to diff `vendor/sopbench` against a fresh `git archive` of the pin when a clone path is given in `SOPBENCH_SRC`, and to skip with a note otherwise.
  - Create the two `__init__.py` files with the constants above.
- [ ] **Step 4: Run it.** Same command → 3 passed. Then the full suite `-m "not db"` and `ruff check .` → all pass.
- [ ] **Step 5: Commit.** `git add vendor/sopbench vendor/VENDORED.md scripts/verify_vendor.sh specdx/subjects tests/subjects` and `git commit -m "vendor: SOPBench at 8790ed7 (code MIT, data CC BY 4.0), without published trajectories"`.

---

### Task 2: Select and fetch the published trajectories

**Files:**
- Create: `specdx/subjects/sopbench/trajectories.py`
- Test: `tests/subjects/test_sopbench_trajectories.py`

**Interfaces:**
- Consumes: `SOPBENCH_COMMIT`, `DOMAINS` (Task 1).
- Produces:
  - `TrajectoryName(path: str, domain: str, user: str | None, model: str, canonical: str, mode: str, tool_list: str, size: int)` (frozen). `canonical` is `model` with `-it-fireworks` replaced by `-instruct`.
  - `parse_name(path: str, size: int) -> TrajectoryName | None`. It returns None for paths not matching `output/<domain>/[usr_<user>-]ast_<model>-mode_<mode>-dep_full-fmt_structured-tool_<tool>-shuffle_False.json`.
  - `MODEL_FAMILIES: dict[str, str]`, keyed by canonical name.
  - `OPEN_WEIGHTS: frozenset[str]` of canonical names.
  - `UnknownModels(ValueError)`.
  - `TrajectoryError(ValueError)`, raised for an unreadable or non-list trajectory file. Task 3 imports it from here.
  - `select(names: Sequence[TrajectoryName]) -> list[TrajectoryName]`.
  - `fetch(selected: Sequence[TrajectoryName], dest: Path, manifest: Path, *, opener: Callable[[str], bytes]) -> dict`.

**Fixed values**
- `MODEL_FAMILIES`:
  - `openai`: `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-5`, `gpt-5-mini`, `o1`, `o4-mini`, `o4-mini-high`.
  - `anthropic`: `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219`, `claude-3-7-sonnet-20250219-thinking`.
  - `google`: `gemini-1.5-pro`, `gemini-2.0-flash-001`, `gemini-2.0-flash-thinking-exp`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemma-4-e2b-it`, `gemma-4-e4b-it`.
  - `meta`: `llama3.1-8b-instruct`, `llama3.1-70b-instruct`.
  - `qwen`: `qwen2.5-7b-instruct`, `qwen2.5-14b-instruct`, `qwen2.5-32b-instruct`, `qwen2.5-72b-instruct`, `qwen3.5-2b`, `qwen3.5-4b`.
  - `deepseek`: `deepseek-r1`.
- `OPEN_WEIGHTS` = every `gemma-*`, `llama*`, `qwen*` and `deepseek-r1` entry above.

**Selection rule** (in this order):
1. Drop names with a `usr_` prefix, `tool_list != "full"`, `model == "human"`, or `size < 10_000`.
2. Raise `UnknownModels`, listing every remaining canonical name not in `MODEL_FAMILIES`.
3. Keep one file per `(domain, canonical)`. Prefer mode `fc`, then `react`, then `act-only`, then `react-v`; then the non-fireworks name; then the lexically smaller path.

**Fetch**
- Download each file from the raw URL at the pin, using the injected `opener`.
- Check that it parses as a JSON list, and raise `TrajectoryError(path)` otherwise.
- Write it to `dest/<domain>/<file name>`.
- Write the manifest `{"commit", "fetched_at", "files": [{"path", "domain", "model", "canonical", "family", "open_weights", "mode", "size", "sha256"}]}`, sorted by path.
- Return the manifest dict.
- If `dest/<domain>/<file>` already exists with the same sha256 recorded in an existing manifest, skip the download.

- [ ] **Step 1: Write the failing tests.** Use a fake tree with these names:
  - `test_parse_name_reads_model_mode_and_tool_and_canonicalises_fireworks`: for `output/bank/ast_qwen2.5-72b-it-fireworks-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`, `canonical == "qwen2.5-72b-instruct"`, `mode == "fc"`, `tool_list == "full"`, `user is None`.
  - `test_select_drops_user_variants_oracle_tools_humans_and_truncated_files`: a 2-byte file and an `oracle` file are both gone.
  - `test_select_keeps_one_file_per_domain_and_model_preferring_fc_then_non_fireworks`: `fc` beats `react`, and the instruct file beats the fireworks one at equal mode.
  - `test_select_names_every_unknown_model`: `pytest.raises(UnknownModels, match="mystery-1b")`.
  - `test_fetch_writes_files_and_a_sorted_manifest_with_sha256` and `test_fetch_rejects_a_non_list_file`: use a fake `opener` that returns `b"[]"` or `b"{}"`. The second raises `TrajectoryError`.
- [ ] **Step 2: Run them** → FAIL (import error).
- [ ] **Step 3: Implement** the interfaces above in `trajectories.py`, with `urllib.request` as the default `opener` in the CLI only.
- [ ] **Step 4: Run them** → 6 passed. Then the suite and ruff.
- [ ] **Step 5: Commit.** `feat: select and fetch SOPBench's published trajectories with a sha256 manifest`.

---

### Task 3: Load tasks and runs

**Files:**
- Create: `specdx/subjects/sopbench/load.py`
- Test: `tests/subjects/test_sopbench_load.py`, `tests/subjects/fixtures/` (two tiny trajectory files built in the test)

**Interfaces:**
- Consumes: `SOPBENCH_DIR`, `TrajectoryName` (Tasks 1–2).
- Produces:
  - `S3Task(domain: str, index: int, user_goal: str, should_succeed: bool, user_prompt: str)` (frozen). `load_tasks(domain: str, root: Path = SOPBENCH_DIR) -> list[S3Task]` iterates the JSON dict's groups in file order and numbers tasks from 0 across groups. `user_goal` is the group key.
  - `S3Run(domain: str, task: int, model: str, family: str, mode: str, succeeded: bool, db_match: bool, constraints: bool, process: bool, tool_error: bool, prompt: str)` (frozen), with property `run_key -> str` = `f"{model}|{domain}|{task}"`.
  - `LoadStats(records: int, runs: int, missing_evaluation: int)`.
  - `TrajectoryMismatch(ValueError)`. `TrajectoryError` is imported from Task 2.
  - `load_runs(path: Path, name: TrajectoryName, family: str, tasks: Sequence[S3Task]) -> tuple[list[S3Run], LoadStats]`.

**Rules**
- The file is a list with one record per task, in task order.
- Raise `TrajectoryMismatch` when `len(records) != len(tasks)` or `record["task"]["user_prompt"] != tasks[i].user_prompt`.
- Each record uses `interactions[0]` and `evaluations[0]` only.
- A record with no `evaluations` is counted in `missing_evaluation` and skipped.
- The fields map as follows: `succeeded = evaluations[0]["action_successfully_called"]`, `db_match = ["database_match"]`, `constraints = ["constraint_not_violated"]`, `process = ["dirgraph_satisfied"]`, `tool_error = not ["no_tool_call_error"]`, `prompt = interactions[0]["prompt"]`.
- JSON that is unreadable or not a list raises `TrajectoryError(path)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_load_tasks_counts_and_orders_bank_like_the_file`: 134 tasks; `tasks[0].user_goal` equals the first group key; indices are `0..133`.
  - `test_load_runs_maps_evaluation_fields`: build a 2-task fake file from two real vendored library tasks.
  - `test_a_record_without_evaluations_is_skipped_and_counted`.
  - `test_a_shorter_file_or_a_shifted_prompt_raises_mismatch`: two cases.
  - `test_an_empty_file_raises_trajectory_error`.
- [ ] **Step 2: Run them** → FAIL.
- [ ] **Step 3: Implement `load.py`.**
- [ ] **Step 4: Run them** → 5 passed. Then the suite and ruff.
- [ ] **Step 5: Commit.** `feat: load SOPBench tasks and published runs, refusing misaligned files`.

---

### Task 4: Tell corpus, cells and readings (§13d)

**Files:**
- Create: `specdx/subjects/sopbench/cells.py`
- Test: `tests/subjects/test_sopbench_cells.py`

**Interfaces:**
- Consumes: `S3Task`, `S3Run` (Task 3); `Cell`, `Reading`, `CellData`, `SATISFIED`, `VIOLATED` from `specdx.infer.cells`; `Chunk`, `Sentence`, `TellCorpus` from `specdx.spec.tell`.
- Produces:
  - Constants `OUTCOME_VALUES = ("succeeds", "refused")` and `CHECKS = ("db_match", "constraints", "process")`.
  - `cell_id(domain: str, task: int, part: str) -> str` = `f"s3:{domain}:{task}:{part}"`, where `part` is `outcome` or one of `CHECKS`.
  - `tell_for(domain: str, runs: Sequence[S3Run]) -> tuple[TellCorpus, int]`:
    - one `Chunk` per task, with id `f"s3:{domain}:{task}"` and source `"sopbench"`;
    - sentences are the prompt's non-empty stripped lines, with ids `f"{chunk}:{n:02d}"` from 0;
    - the int returned is the number of runs whose prompt differs from the first run's prompt for that task. The first prompt is used.
  - `S3CellData(data: CellData, violated: dict[str, frozenset[str]], excluded_tool_error: int)` (frozen).
  - `cells_for(tasks: Sequence[S3Task], runs: Sequence[S3Run]) -> S3CellData`.

**Rules (§13d)**
- **Cells.** One outcome cell and three check cells per task.
  - Outcome: `kind="value"`, `anchor` = `"succeeds"` if `should_succeed` else `"refused"`, `clause` = its own id, `location` = its id, `group = f"outcome:{user_goal}"`, `steps=()`.
  - Check: `kind="constraint"`, `anchor=SATISFIED`, `clause` = its own id, `location` = its id, `group` = the check name.
- **Exclusions.** Runs with `tool_error` give no readings and are counted.
- **Reading.** `Reading(cell, run.run_key, run.model, run.family, scenario=f"{domain}:{task}", value, clean, variant=0)`.
  - Outcome value is `"succeeds"` if `run.succeeded` else `"refused"`.
  - Check values are `SATISFIED` or `VIOLATED`.
  - `clean = db_match and constraints and process`, the same for all four readings of the run.
- **V(b).** `violated[run_key]` is the set of check-cell ids that the run violated.

- [ ] **Step 1: Write the failing tests.**
  - `test_each_task_has_one_outcome_and_three_check_cells_with_the_13d_anchors`: a should-fail task's outcome anchor is `"refused"`.
  - `test_readings_follow_evaluations_and_clean_needs_all_three_checks`.
  - `test_a_tool_error_run_gives_no_readings_and_is_counted`.
  - `test_violated_lists_only_the_checks_the_run_failed`.
  - `test_tell_has_one_chunk_per_task_and_counts_prompt_disagreements`.
- [ ] **Step 2: Run them** → FAIL.
- [ ] **Step 3: Implement `cells.py`.**
- [ ] **Step 4: Run them** → 5 passed. Then the suite and ruff.
- [ ] **Step 5: Commit.** `feat: S3 cells, readings and Tell corpus from SOPBench runs (design §13d)`.

---

### Task 5: Offline diagnosis per domain, and population compositions

**Files:**
- Create: `specdx/subjects/sopbench/diagnose.py`
- Test: `tests/subjects/test_sopbench_diagnose.py`

**Interfaces:**
- Consumes: `S3CellData` (Task 4); `fit_anchored`, `AnchoredFit`, `V3Params` from `specdx.infer.anchored`; `type_cells`, `CellFinding` from `specdx.infer.typing`; `soft_hitting_sets` from `specdx.infer.hitting`; `Defect` from `specdx.types`; `OPEN_WEIGHTS` (Task 2).
- Produces:
  - `PROVISIONAL = "provisional: V3Params calibrated on CASE-0031, recalibrated for S3 in P3b"`.
  - `S3Diagnosis(domain: str, raters: tuple[str, ...], fit: AnchoredFit, defects: list[Defect])`.
  - `diagnose_domain(domain: str, cd: S3CellData, models: Mapping[str, str], params: V3Params = V3Params()) -> S3Diagnosis`.
  - `COMPOSITIONS = ("all", "open", "hosted")` and `compose(runs: Sequence[S3Run], which: str) -> list[S3Run]`. `open` keeps the canonical names in `OPEN_WEIGHTS`; `hosted` keeps the rest.

**Algorithm**
1. Fit: `fit = fit_anchored(cd.data.cells, cd.data.readings, models=models, params=params)`.
2. Cell findings: for each `f` in `type_cells(cd.data.cells, fit, cd.data.by_cell(), params)`, make `Defect(type=f.type, location=(f.cell,), scenarios=(scenario of the cell,), witnesses=f.witnesses, score=f.score, subtype=f.subtype, note=…)`. The note is `"refutation not attempted (offline)"` for `hidden` and `""` otherwise.
3. Contradictions: per task, run `soft_hitting_sets([(cd.violated[rk], fit.weights.get(rk, 0.0)) for each run of the task with a non-empty V(b)], tau=0.05, max_size=3)`. Each set H with |H| ≥ 2 becomes `Defect("contradiction", tuple(sorted(H)), (scenario,), (), mass)`.
4. Return the defects sorted by score descending, then by `key`.

- [ ] **Step 1: Write the failing tests.** All on hand-built `S3CellData`, as `tests/infer/test_anchored.py` does.
  - `test_agreeing_raters_give_no_defect_and_a_converged_fit` (Review Focus 4).
  - `test_raters_who_all_do_a_forbidden_action_flag_the_outcome_cell_as_hidden`: 6 raters from 3 families, all `"succeeds"` on a `"refused"` anchor, with checks satisfied. A `hidden` defect at that outcome cell, noted offline.
  - `test_a_split_population_flags_ambiguity`: half `"succeeds"`, half `"refused"` across every family, on a cell whose anchor is `"succeeds"`. An `ambiguity` defect.
  - `test_two_checks_violated_together_by_most_runs_give_a_contradiction`.
  - `test_compose_splits_open_and_hosted_raters`.
- [ ] **Step 2: Run them** → FAIL.
- [ ] **Step 3: Implement `diagnose.py`.** If a test shows that `type_cells` needs inputs beyond the hand-built ones, use the inputs `experiments/p2_replay.py:replay` passes. Do not change `specdx/infer/`.
- [ ] **Step 4: Run them** → 5 passed. Then the suite and ruff.
- [ ] **Step 5: Commit.** `feat: offline per-domain diagnosis of SOPBench runs and population compositions`.

---

### Task 6: Stage reports and the status dashboard

**Files:**
- Create: `specdx/report/__init__.py`, `specdx/report/stages.py`
- Test: `tests/report/__init__.py`, `tests/report/test_stages.py`

**Interfaces:**
- Produces:
  - `StageDef(name: str, title: str, plan: str, implemented: bool)` (frozen).
  - `STAGES: tuple[StageDef, ...]`, in §13c's order. Each entry gives `(name, plan, implemented)`:

| Stage | Plan | Implemented |
|---|---|---|
| `local` | P3a | True |
| `fetch-sopbench` | P3a | True |
| `replication` | P3a | True |
| `s3-screen` | P3b | False |
| `s3-dev` | P3b | False |
| `s3-calibrate` | P3b | False |
| `s3-gate` | P3b | False |
| `s3-baselines-dev` | P3b | False |
| `s2-screen` | P3c | False |
| `s2-dev` | P3c | False |
| `s2-calibrate` | P3c | False |
| `s2-gate` | P3c | False |
| `s2-baselines-dev` | P3c | False |
| `freeze` | P3b | False |
| `s3-test` | P3b | False |
| `s2-test` | P3c | False |
| `s2-real` | P3c | False |
| `ablations` | P3d | False |
| `status` | P3a | True |

    Each `title` is a short English phrase.
  - `StageOutcome(exit_code: int, verdict: str, key_number: str, body: str, data: dict | None = None)`. `verdict` is one of `PASS`, `FAIL`, `STOP`, `DONE` or `PENDING`.
  - `run_stage(name: str, fn: Callable[[], StageOutcome], *, command: str, results: Path, subject: str = "", split: str = "") -> int`. It:
    - records the start in UTC, then calls `fn`;
    - on an exception, uses `StageOutcome(2, "FAIL", "", "```\n<last 30 lines of the traceback>\n```")`;
    - writes the report, its metadata and `data` (as `results/<name>.data.json` when not None);
    - rebuilds the status page and returns the exit code.
  - `write_report(...) -> Path`. Its header table has the rows `Started`, `Ended`, `Duration`, `Commit` (short SHA plus `clean`/`dirty` from `git status --porcelain`), `Command`, `Exit code`, `Subject`, `Split`, `Verdict`, then `## Results` and `body`.
  - `already_done(name: str, results: Path) -> bool`: true when `results/<name>.json` exists with `exit_code == 0`.
  - `update_status(results: Path, work: Path) -> Path`. It writes `STATUS.md`: a title, the time it was built, and a table with columns `Stage | Plan | Status | Verdict | Key number | Ended | Duration | Report`.
    - Status is `pending (plan Px)` for an unimplemented stage.
    - `running` when `work/p3-<name>.pid` names a live pid.
    - `done` for exit 0; `failed (exit N)` otherwise; `not run` when there is no metadata.

- [ ] **Step 1: Write the failing tests** (with `tmp_path` and a fake `git`, or with `REPO_ROOT` as cwd).
  - `test_run_stage_writes_header_rows_and_body`: every header row name is present, and the verdict is shown.
  - `test_an_exception_becomes_exit_2_with_the_traceback_in_the_report`.
  - `test_already_done_only_after_exit_0` (Review Focus 5).
  - `test_status_lists_every_stage_in_order_with_pending_for_later_plans`.
  - `test_status_shows_running_for_a_live_pid_file`, using `os.getpid()`.
- [ ] **Step 2: Run them** → FAIL.
- [ ] **Step 3: Implement `stages.py`.**
- [ ] **Step 4: Run them** → 5 passed. Then the suite and ruff.
- [ ] **Step 5: Commit.** `feat: P3 stage reports and results/p3/STATUS.md`.

---

### Task 7: The `specdx p3` commands

**Files:**
- Create: `specdx/report/p3.py`
- Modify: `specdx/cli.py` (a `p3` subparser with sub-subcommands; a branch in `main`)
- Test: `tests/report/test_p3.py`, `tests/test_cli.py` (add cases)

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces the CLI, all with `--results` (default `results/p3`):
  - `specdx p3 fetch-sopbench [--dest work/sopbench-output] [--manifest experiments/p3-sopbench-trajectories.json]`;
  - `specdx p3 replication [--trajectories work/sopbench-output] [--manifest …] [--domains …]`;
  - `specdx p3 local`;
  - `specdx p3 status`.

  Each command calls `run_stage` and returns its exit code.
- `fetch_stage(...) -> StageOutcome`:
  - Lists the pinned tree through `https://api.github.com/repos/Leezekun/SOPBench/git/trees/8790ed7406acbd22270f90bd1c4038de3fc5db8f?recursive=1`, then runs `select` and `fetch`.
  - Verdict `DONE`. The key number is `"<n> files, <m> models, <k> families"`.
  - The body is a table of domain against files, followed by the list of models by family.
- `replication_stage(...) -> StageOutcome`:
  - For each domain:
    - load the tasks and every manifest file;
    - exclude files that raise `TrajectoryMismatch` or `TrajectoryError`, listing them;
    - for each composition in `COMPOSITIONS`, run `cells_for` and `diagnose_domain`.
  - The body starts with the `PROVISIONAL` line.
  - Table 1 has one row per domain and composition: `raters | runs | excluded runs | θ min | θ median | converged | hidden | ambiguity | weak | contradiction`.
  - Table 2 lists the 10 highest-scoring defects per domain from `all`, with the task's `user_goal`.
  - Then come the excluded files and the prompt-disagreement counts.
  - `data` holds every defect and every θ.
  - The key number is `"<defects in all> findings over <domains> domains"`. Verdict `DONE`.
  - Exit 2, verdict `FAIL`, when the manifest or a trajectory directory is missing.
- `local_stage() -> StageOutcome`:
  - Runs `pytest -q -o addopts="" -m "not db"` and `ruff check .` as subprocesses.
  - Runs `replication_stage` on `library` only, when the manifest exists, writing to a temporary results directory. Otherwise it notes "replication smoke skipped: run fetch-sopbench first".
  - Exit 1, verdict `FAIL`, if tests or ruff fail; otherwise `PASS`.

- [ ] **Step 1: Write the failing tests.**
  - `test_replication_stage_on_two_fake_files_writes_both_tables_and_the_provisional_line`.
  - `test_replication_lists_a_misaligned_file_as_excluded` (Review Focus 2).
  - `test_replication_without_a_manifest_fails_with_exit_2`.
  - In `tests/test_cli.py`: `test_p3_subcommands_parse`.
- [ ] **Step 2: Run them** → FAIL.
- [ ] **Step 3: Implement `p3.py` and the CLI branch.**
- [ ] **Step 4: Run them** → pass. Then the suite and ruff.
- [ ] **Step 5: Commit.** `feat: specdx p3 fetch-sopbench, replication, local and status`.

---

### Task 8: Rewrite `run_all.sh` and the run guide for P3

**Files:**
- Modify: `run_all.sh` (rewrite), `specdx/cli.py` (`p3 status --done`), `docs/RUNNING.md`, `README.md`, `tests/test_docs.py`

**Interfaces:**
- Consumes: `specdx p3 …` (Task 7), `STAGES` (Task 6).
- Produces the following `run_all.sh` usage. The header comment lists every stage of `STAGES` with its plan, plus `all` and `status`.
  - `./run_all.sh <stage>` runs one implemented stage through `specdx p3 <stage>`.
  - A stage that is not yet implemented prints `"<stage>: pending — implemented in plan <Px>"`, rebuilds the status page and exits 0.
  - `./run_all.sh all` runs every implemented stage in `STAGES` order. It skips any for which `specdx p3 status --done <stage>` succeeds, which is `already_done`. It stops at the first non-zero exit and then rebuilds the status page.
  - Exit codes are those of Global Constraints.
  - CASE-0031 stages (`g2`, `preflight`, `g3`, `gate`, `baselines-dev`, `freeze-baselines`, `study`, `report`, the `s2` stub) are removed. The `specdx` CLI commands behind them remain.
  - Add to the CLI: `specdx p3 status --done <stage>`, exit 0 if done, 1 otherwise.
- `docs/RUNNING.md`: replace §11 (P2b) with "§11 P3: SOPBench and τ²-bench". It gives the stages table of design §13c, which plan implements each, where the reports go, and `PYTHONHASHSEED=0` for later local SOPBench runs. Keep §12 (reproducing G0/G1) as history.
- `README.md`: change the "Running the experiments" section to point to `run_all.sh all` and `results/p3/STATUS.md`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_docs.py`, replace `test_run_all_offers_the_comparison_steps`:
  - `test_run_all_usage_lists_every_p3_stage`: `bash run_all.sh` exits 2, and stdout names every `StageDef.name`.
  - `test_a_later_plan_stage_is_pending_and_exits_0`: `bash run_all.sh s2-screen` prints `pending — implemented in plan P3c`.
  - `test_run_all_no_longer_offers_case_0031_stages`: `g3`, `gate` and `preflight` are absent from the usage.
  - `test_running_guide_describes_p3`: RUNNING.md contains `P3: SOPBench and τ²-bench` and `results/p3/STATUS.md`.
- [ ] **Step 2: Run them** → FAIL.
- [ ] **Step 3: Rewrite `run_all.sh`.** Keep `set -euo pipefail`, the `step`/`ok`/`warn`/`fail`/`die` helpers, and `check_tools`. Make the doc edits.
- [ ] **Step 4: Run them** → pass. Run the full suite `-m "not db"`, then ruff, then `bash -n run_all.sh`.
- [ ] **Step 5: Commit.** `feat: run_all.sh for P3, one Markdown report per stage; RUNNING and README point to results/p3/STATUS.md`.

---

### Task 9: First real run, on this machine (CPU and network)

**Files:**
- Create: `experiments/p3-sopbench-trajectories.json` (from the fetch)

- [ ] **Step 1: Fetch.**
  - Command: `./run_all.sh fetch-sopbench`.
  - Expected: exit 0, and `results/p3/fetch-sopbench.md` lists 7 domains, at least 20 models and at least 5 families.
  - About 1–2 GB is downloaded to `work/sopbench-output/`.
- [ ] **Step 2: Replicate.**
  - Command: `./run_all.sh replication`.
  - Expected: exit 0, and `results/p3/replication.md` has both tables for 7 domains and 3 compositions, plus the provisional line.
  - Record the wall time and the number of excluded files in the ledger.
- [ ] **Step 3: Check the dashboard.**
  - Command: `./run_all.sh status`, then `cat results/p3/STATUS.md`.
  - Expected: `local`, `fetch-sopbench` and `replication` show their statuses; every other stage shows `pending (plan P3b|P3c|P3d)`.
- [ ] **Step 4: Re-run everything.**
  - Command: `./run_all.sh all`.
  - Expected: `fetch-sopbench` and `replication` are skipped as done, `local` runs, and the exit is 0.
- [ ] **Step 5: Commit the manifest only.** `results/` is not tracked. `git add experiments/p3-sopbench-trajectories.json` and `git commit -m "results: P3a manifest of SOPBench's published trajectories (sha256)"`.
