# BenchGuard: LLM cross-verification of benchmark artifacts

**Group:** A. Direct competitor (whole method) · **Replaces/compares with:** B1 (static LLM review), published form · **SpecDx defect types it can address:** Hidden requirement (an instruction omits a rule the grader enforces); Weak check (the grader accepts a shortcut); Contradiction (the grader rejects a valid solution); Ambiguity

## Papers (keys in `related_work/references.bib`)
- `tu2026who`: Xinming Tu et al. (2026). *Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks*. arXiv (preprint). https://arxiv.org/abs/2604.24955

## Code
- `code/`: github.com/XinmingTu/BenchGuard @ `7d6a52e5627a1a734ecd895e0c73778ab54aca63 2026-10-20` (shallow clone). Licence: Apache-2.0 (LICENSE file; the GitHub API reports NOASSERTION).

## What it does
Uses LLMs to cross-check the instruction, reference solution, grader and environment of execution-based agent benchmarks through structured protocols. Agent traces can optionally be added as evidence. The arXiv preprint is 2604.24955, and the README states it was accepted at COLM 2026.

## Running it on SpecDx
- **Input it needs:** A benchmark directory per task. For SpecDx: T (documents or policy, tool descriptions), C (SQL checks, golden state), and optionally SpecDx run traces.
- **Adaptation:** Package each S1 scenario and each S2 task as a BenchGuard task directory, then run it with and without SpecDx traces attached. Map its findings to the four types and score them against mutants and real defects. The 'with traces' arm tests whether population statistics add value beyond giving one auditor the same evidence.
- **Models:** The main interface is a Claude Code or Codex skill (frontier models). The older Python package (`src/benchguard`) uses litellm and can be pointed at Ollama for an offline run.
- **Effort:** Medium. Port to the Python package with a local model, and write a task packager.

## Caveats
- It is recall-oriented by design (the README describes its findings as a queue for human review), so expect low precision.
- Running it with local models departs from the paper's setting; state this in the threats to validity.
