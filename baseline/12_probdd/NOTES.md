# ProbDD: probabilistic delta debugging

**Group:** C. Component baselines and ablations · **Replaces/compares with:** Alternative to ddmin in SpecDx's ambiguity localisation (v1 §3.7, v3 §5.3) · **SpecDx defect types it can address:** Ambiguity localisation in T

## Papers (keys in `related_work/references.bib`)
- `wang2021probabilistic`: Guancheng Wang et al. (2021). *Probabilistic Delta debugging*. ESEC/FSE (peer-reviewed, CORE2023 A*). https://doi.org/10.1145/3468264.3468625

## Code
- `code/`: github.com/Amocy-Wang/ProbDD @ `b01969ebf47fc35cb9cbd4c074227d84c097794e 2021-08-02` (shallow clone). Licence: GPL-3.0.

## What it does
Replaces ddmin's fixed partitioning with a Bayesian model of how likely each element is to be essential, and deletes the subset with the highest expected gain. It reaches smaller results with fewer tests than ddmin.

## Running it on SpecDx
- **Input it needs:** The sentences of T relevant to step s, and the property test 'the population still splits on cell c'.
- **Adaptation:** Swap ddmin for ProbDD inside the v3 §5.3 oracle and compare oracle calls (each call is a population run) and the size of the minimal sentence set at equal budget (80).
- **Models:** Uses the SpecDx population as the oracle; the ProbDD logic itself needs no model.
- **Effort:** Low. ProbDD is about 50 lines to reimplement in Python; the repository is the reference (C/Python replication package).

## Caveats
- ProbDD assumes a deterministic oracle. The population oracle is noisy, so repeat or threshold each test, and note this.
- GPL-3.0: do not copy code into SpecDx; reimplement from the paper.
