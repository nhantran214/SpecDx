# AgentSuite / COBA: component-based auditing of agent benchmarks

**Group:** A. Direct competitor (whole method) · **Replaces/compares with:** B1 + B5 (static LLM review plus rule-based checks) · **SpecDx defect types it can address:** Hidden requirement (ground-truth issues); Weak check (evaluation issues); Ambiguity (user-instruction issues)

## Papers (keys in `related_work/references.bib`)
- `suh2026agentsuite`: Hyewon Suh et al. (2026). *AgentSuite: Toward More Reliable Agent Evaluation with a Component-Based Benchmark Auditing Pipeline*. ICML (peer-reviewed, CORE2023 A*). https://proceedings.mlr.press/v306/suh26a.html

## Code
- `code/`: github.com/Agent-Suite/AgentSuite @ `0f9eac1c1a376a411ad807bd974555055f08e6c5 2026-07-05` (shallow clone). Licence: no licence file (research use only).

## What it does
Splits each agent task into User, Environment, Ground Truth and Evaluation components. It detects validity issues with rule-based detectors plus taxonomy-guided LLM judgement (F1 0.79–0.87 against experts) and found τ²-bench issues that manual verification missed. The pipeline is in `pipeline/`. The repository also vendors τ-bench and τ²-bench.

## Running it on SpecDx
- **Input it needs:** Per task: the user instruction, the environment and database, the ground-truth actions and the evaluation config. SpecDx's S2 (τ²-bench airline and retail) has exactly this shape. S1 needs packaging: instruction = scenario plus T, ground truth = golden state, evaluation = clauses C.
- **Adaptation:** Run COBA on (a) the original τ²-bench tasks, scored against the τ²-Bench-Verified repairs (R-τ²), and (b) the seeded S2 mutants. Map component-level issue labels to SpecDx's types and locations (cell or clause) to compare recall and localisation. It is the strongest *peer-reviewed* automatic auditor on SpecDx's own benchmark.
- **Models:** Uses OpenAI-compatible endpoints through litellm and vLLM (`.env` with `API_KEY` and `BASE_URL`), so Ollama or vLLM serving the SpecDx population can be configured. The rule detectors run offline.
- **Effort:** Medium–high. Conda environment, porting prompts to local models, and a task packager for S1.

## Caveats
- 2.3 GB clone (mostly the vendored `tau2-bench` and `pipeline` data).
- No licence file.
- Expect lower quality with local models than with the paper's frontier models; report both if API access is ever allowed (SpecDx's constraint is local models only).
