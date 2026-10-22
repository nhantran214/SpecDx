# ClarifyGPT: execution-consistency ambiguity detection

**Group:** A. Direct competitor (whole method) · **Replaces/compares with:** Ambiguity detector; single-family ablation of SpecDx; B4 · **SpecDx defect types it can address:** Ambiguity

## Papers (keys in `related_work/references.bib`)
- `mu2024clarifygpt`: Fangwen Mu et al. (2024). *ClarifyGPT: A Framework for Enhancing LLM-Based Code Generation via Requirements Clarification*. FSE (PACMSE) (peer-reviewed, CORE2023 A* (as FSE)). https://doi.org/10.1145/3660810

## Code
- `code/`: github.com/ClarifyGPT/ClarifyGPT @ `543b34b35c5525579d9afee838c992792a3a43dd 2023-10-28` (shallow clone). Licence: no licence file (research use only; do not redistribute).

## What it does
Samples several programs for one requirement, runs them on generated test inputs, and declares the requirement ambiguous when their outputs disagree. It then asks clarifying questions. The detection step is the part to compare.

## Running it on SpecDx
- **Input it needs:** A requirement, a generator LLM and inputs to execute on. In SpecDx: Tell T plus one scenario step, the runs of one model, and the cell readings of their end states.
- **Adaptation:** Per scenario step, treat one model's runs (3 Tell variants × 3 seeds) as ClarifyGPT's n samples and the cells as the test outputs. Flag the step, or the cell, as ambiguous when the readings are not all identical. A pooled-population variant (all families) is a second arm. No anchor is used, so it cannot see hidden requirements, nor a collapse onto one wrong reading.
- **Models:** The released scripts call the OpenAI API (`src/clarify/run_clarify_chatgpt_*.py`). The detection rule needs no LLM once SpecDx runs are cached, so it runs offline on `work/runs.jsonl`.
- **Effort:** Low. Reimplement the consistency check (tens of lines) on the run cache. The repository is the reference for the rule and its prompts.

## Caveats
- The code targets HumanEval and MBPP code generation, not database end states.
- The repository has no licence file.
- The t6 agent did not find an official code link, but the t3 agent found this repository linked from the paper's references. Treat it as the authors' replication package.
