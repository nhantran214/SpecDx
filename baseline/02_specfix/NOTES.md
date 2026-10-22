# SpecFix: ambiguity as the distribution of sampled interpretations, anchored to examples

**Group:** A. Direct competitor (whole method) · **Replaces/compares with:** Ambiguity and hidden requirement (examples as arbiter) · **SpecDx defect types it can address:** Ambiguity; Hidden requirement (partial: it never blames the examples)

## Papers (keys in `related_work/references.bib`)
- `jia2025automated`: Haoxiang Jia et al. (2025). *Automated Repair of Ambiguous Problem Descriptions for LLM-Based Code Generation*. ASE (peer-reviewed, CORE2023 A*). https://doi.org/10.1109/ase63991.2025.00038

## Code
- `code/`: github.com/msv-lab/SpecFix @ `f3a36b35ca4390c6024c7b9f1ed9670a6b5eab8d 2025-10-22` (shallow clone). Licence: no licence file (research use only).

## What it does
Clusters LLM-generated candidate programs by their behaviour on tests (differential testing). It measures ambiguity from the cluster distribution, checks the clusters against input-output examples, and repairs the description by contrastive specification inference.

## Running it on SpecDx
- **Input it needs:** An NL description, an LLM, and input-output examples. In SpecDx: T plus a scenario step, the runs of one model, and C's expected outcomes as the examples.
- **Adaptation:** Cluster one model's runs per scenario step by their end-state cell readings, which play the role of SpecFix's behavioural clustering. Flag (a) steps with two or more clusters as Ambiguity, and (b) steps whose dominant cluster disagrees with C as a Hidden-requirement candidate. Score both against the seeded mutants and the real defects. Optionally run its repair step on T as qualitative output.
- **Models:** `specfix/model.py` uses an OpenAI-compatible client (`base_url`), so it can point at Ollama's OpenAI endpoint (`http://localhost:11434/v1`).
- **Effort:** Medium. Write an adapter from SpecDx runs to its clustering input, or reimplement its detection metric on the run cache.

## Caveats
- It treats the examples as ground truth, so by construction it never reports a weak check or a check-side defect.
- It has no rater model, so model error and specification ambiguity are mixed.
- The repository has no licence file and is 372 MB with its datasets.
