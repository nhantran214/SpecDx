# IntellAgent: policy-graph scenario generation and critique

**Group:** C. Component baselines and ablations · **Replaces/compares with:** Alternative scenario source to SpecDx's synthesis (S2); single-critic defect detector · **SpecDx defect types it can address:** Scenario synthesis; weak per-policy verdicts

## Papers (keys in `related_work/references.bib`)
- `levi2025intellagent`: Elad Levi et al. (2025). *IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems*. arXiv (preprint). https://arxiv.org/abs/2501.11067

## Code
- `code/`: github.com/plurai-ai/intellagent @ `5f971e4a3f3e2690ace783d56d20f471c45420c4 2026-09-14` (shallow clone). Licence: Apache-2.0.

## What it does
Decomposes a system prompt or policy into a weighted policy graph, samples multi-policy scenarios together with a database, simulates user–agent dialogues and critiques policy adherence.

## Running it on SpecDx
- **Input it needs:** The policy text T (the τ²-bench airline or retail policy, or the TonomyX documents), tool schemas and a database.
- **Adaptation:** (a) Replace SpecDx's scenario synthesiser with IntellAgent's policy-graph sampler at an equal run budget (ablation of component 4). (b) Use its critique agent's per-policy verdicts over SpecDx runs as a single-judge detector.
- **Models:** Supports local models: the repository has an Ollama provider (`tests/test_ollama_provider.py`, LangChain/LangGraph).
- **Effort:** Medium.

## Caveats
- It assumes the policies are correct; it evaluates agents, not specifications.
- It is a preprint (arXiv 2501.11067).
