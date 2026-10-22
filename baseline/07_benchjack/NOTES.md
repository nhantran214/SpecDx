# BenchJack: agent red-teaming of benchmark checkers (weak-check witnesses)

**Group:** A. Direct competitor (weak checks) · **Replaces/compares with:** Weak-check detector with witnesses · **SpecDx defect types it can address:** Weak check only

## Papers (keys in `related_work/references.bib`)
- `wang2026androids`: Hao Wang et al. (2026). *Do Androids Dream of Breaking the Game? Systematically Auditing AI Agent Benchmarks with BenchJack*. arXiv (preprint). https://arxiv.org/abs/2605.12673

## Code
- `code/`: github.com/benchjack/benchjack @ `22c554f217357da5b3107c6e3fd7a128cef42d3b 2026-05-25` (shallow clone). Licence: Apache-2.0.

## What it does
A hackability scanner: static tools (Semgrep, Bandit, Hadolint) plus coding agents that audit a benchmark's evaluation and synthesise reward-hacking exploits. Each exploit scores without solving the task. It found 219 flaws across 10 agent benchmarks.

## Running it on SpecDx
- **Input it needs:** A benchmark repository with its grader. For SpecDx: a task with its checks C (SQL invariants, golden state) and access to the gated tools.
- **Adaptation:** For each S2 task and each S1 scenario, ask the agent for a behaviour that passes C while violating a rule of T, framed as 'satisfy the checker without doing the task'. A found exploit is a weak-check witness. Compare with SpecDx's weak checks on the W mutants. Its vulnerability classes V1–V8 mostly concern sandbox and grader hacking (leaked answers, `eval` injection). Only the 'grader too loose' class is in scope; exclude the infrastructure exploits.
- **Models:** Runs through Claude Code or Codex (`server/ai_runner.py`). An offline version needs a local coding agent. Static analysis runs offline.
- **Effort:** Medium–high. Re-target from repository-level hacking to clause-level looseness.

## Caveats
- Adversarial rather than honest executors, which is a different evidence model from SpecDx's.
- No hidden requirements, ambiguity or contradictions.
- It is a preprint (arXiv 2605.12673).
