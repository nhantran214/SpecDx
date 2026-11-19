# Global constraints (from the plan; binding on every task)

## Repository
- `vendor/` is never edited after Task 1.
- Nothing is imported from `baseline/` clones, `/home/ethan/AILab/FAA` or `aebench`.
- Code, comments and docs are in English.
- Ruff: line-length 100, rules E, F, I, UP, B, SIM, target py312.

## SOPBench
- Pinned at commit `8790ed7406acbd22270f90bd1c4038de3fc5db8f` (2026-09-13).
- Code is MIT ("The SOPBench Authors"); data is CC BY 4.0.

## What may run
- Network access is allowed only in `specdx p3 fetch-sopbench`, and never in tests. It may reach only two places:
  - `raw.githubusercontent.com/Leezekun/SOPBench/<pin>/<path>`;
  - the pinned tree API.
- No hosted model API is called.
- There is no interpreter run, Ollama call or database access in this plan.
- `work/runs.jsonl` is never opened.
- No CASE-0031 run is made. The existing CASE-0031 code and its tests stay and must pass.

## V3Params
- `V3Params` is not changed.
- Every S3 result is labelled `provisional: V3Params calibrated on CASE-0031, recalibrated for S3 in P3b`.

## S3 cells (design §13d)
Each task has four cells:

| Cell | Kind | Vocabulary | Anchor | Reading |
|---|---|---|---|---|
| `s3:<domain>:<task>:outcome` | value | `succeeds` / `refused` | `succeeds` if `action_should_succeed` | `succeeds` if `action_successfully_called` |
| `s3:<domain>:<task>:db_match` | constraint | satisfied / violated | satisfied | `database_match` |
| `s3:<domain>:<task>:constraints` | constraint | satisfied / violated | satisfied | `constraint_not_violated` |
| `s3:<domain>:<task>:process` | constraint | satisfied / violated | satisfied | `dirgraph_satisfied` |

- A run is clean when all three check cells are satisfied and its outcome reading equals the anchor: clean = db_match ∧ constraints ∧ process ∧ (outcome reading == anchor). Amended 2026-11-20 (ruling I-6, design §13d), after v3's weak-check rule (v3 design §5.4): a deviant reading at an anchored value cell violates its golden clause.
- V(b) is the run's set of violated check cells.
- Runs with `no_tool_call_error` false give no readings, and are counted.

## Reports
- A stage writes `results/p3/<stage>.md` and its metadata `results/p3/<stage>.json`.
- The dashboard is `results/p3/STATUS.md`.
- `P3_RESULTS` overrides the results directory; `P3_WORK` overrides the work directory.
- Exit codes:
  - 0: done
  - 1: a check failed
  - 2: a usage or environment error
  - 3: a screen stop
  - 4: a test split locked

## Tests
- `.venv/bin/python -m pytest -q -o addopts="" -m "not db"` must pass.
- `.venv/bin/ruff check specdx tests scripts experiments` must pass. This follows the Task 1 ruling: the E501 in `paper/` is out of scope.
