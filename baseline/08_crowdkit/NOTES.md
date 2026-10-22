# Crowd-Kit: Majority Vote, Dawid–Skene, GLAD, MACE, KOS, M-MSR, Wawa

**Group:** B. Statistical aggregation on the reading matrix · **Replaces/compares with:** Simulation baselines (majority vote) plus classic truth inference; ablation 'no set-valued truth, no anchor' · **SpecDx defect types it can address:** Hidden requirement / mis-key (aggregate ≠ d_c); crude Ambiguity (via BinaryRelevance)

## Papers (keys in `related_work/references.bib`)
- `ustalov2024learning`: Dmitry Ustalov et al. (2024). *Learning from Crowds with Crowd-Kit*. JOSS (peer-reviewed). https://doi.org/10.21105/joss.06227
- `dawid1979maximum`: A. P. Dawid et al. (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*. JRSS-C (peer-reviewed). https://doi.org/10.2307/2346806
- `whitehill2009whose`: Jacob Whitehill et al. (2009). *Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise*. NeurIPS (peer-reviewed, CORE2023 A*). https://papers.nips.cc/paper_files/paper/2009/hash/f899139df5e1059396431415e770c6dd-Abstract.html
- `hovy2013learning`: Dirk Hovy et al. (2013). *Learning Whom to Trust with MACE*. NAACL (peer-reviewed, CORE2023 A). https://aclanthology.org/N13-1132/

## Code
- `code/`: github.com/Toloka/crowd-kit @ `c6717c40560cd8e675fc1dda2ab5461295657d07 2025-10-13` (shallow clone). Licence: Apache-2.0 (LICENSE file; the GitHub API reports NOASSERTION).

## What it does
Reference implementations of the classic label-aggregation models. GLAD is the closest ancestor of SpecDx's σ(α_m − β) accuracy model, and Dawid–Skene of its EM fitting.

## Running it on SpecDx
- **Input it needs:** A DataFrame (task, worker, label). Use task = cell id (pooled lifted cell, or per scenario), worker = model (or model × seed), label = normalised reading.
- **Adaptation:** Run every aggregator on the same cell × model readings SpecDx uses (including the simulation study of v3 §8.1). Flag a cell as a hidden-requirement candidate when the aggregate differs from d_c. Optional arm: add the specification as a pseudo-worker carrying d_c, which is closest to SpecDx's 'specification as a fallible rater' view. GLAD vs SpecDx isolates the contribution of set-valued A_c, the anchor and the restricted error model.
- **Models:** None; CPU, offline (`pip install crowd-kit`).
- **Effort:** Low. A DataFrame exporter from `specdx/infer/cells.py` output.

## Caveats
- Single-truth models: they cannot represent A_c = {x, y} or ∅ except through BinaryRelevance (a per-value yes/no hack).
- Weak checks cannot be expressed.
