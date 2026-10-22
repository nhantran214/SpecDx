# Who&When automated failure attribution (all-at-once, step-by-step, binary search)

**Group:** A. Direct competitor (agent error vs defect) · **Replaces/compares with:** Per-run LLM-judge attribution; counterpart of SpecDx's population separation · **SpecDx defect types it can address:** Separating agent errors from specification defects; Hidden requirement (via clause blame)

## Papers (keys in `related_work/references.bib`)
- `zhang2025which`: Shaokun Zhang et al. (2025). *Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems*. ICML (peer-reviewed, CORE2023 A*). https://proceedings.mlr.press/v267/zhang25cq.html

## Code
- `code/`: github.com/mingyin1/Agents_Failure_Attribution @ `b2bae5c5b06d681d04ea5e9b63b7a30525c04925 2026-01-05` (shallow clone). Licence: MIT.

## What it does
Given a failure log, an LLM judge names the responsible agent and the decisive error step, using one of three strategies. Accuracy on the Who&When benchmark is at most 53.5% for the agent and 14.2% for the step.

## Running it on SpecDx
- **Input it needs:** A failing trajectory plus the task. For SpecDx: each run whose end state violates at least one clause, with T and the violated clauses as context.
- **Adaptation:** Add a third verdict, 'specification defect' (the run followed T, and C is wrong), next to 'agent error at step k'. Run the three strategies on every failing run, then aggregate per cell or clause: flag it when most of its failing runs are attributed to the specification. This is the single-judge, per-trace counterpart of SpecDx's population model.
- **Models:** `Automated_FA/inference.py` uses an OpenAI client and can be pointed at Ollama's OpenAI-compatible endpoint, so it runs offline.
- **Effort:** Low–medium. Prompt changes plus a run-log exporter.

## Caveats
- Designed for multi-agent systems, but SpecDx runs are single-agent with gated tools: treat each tool call as a step.
- The reported step accuracy is low, so expect noisy per-run labels; aggregation is the fair comparison.
