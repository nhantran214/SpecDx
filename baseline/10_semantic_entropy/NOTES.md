# Semantic entropy (disagreement entropy over meaning clusters)

**Group:** B. Statistical aggregation on the reading matrix · **Replaces/compares with:** B4 (inter-family disagreement entropy), published form · **SpecDx defect types it can address:** Ambiguity (high entropy); no typing

## Papers (keys in `related_work/references.bib`)
- `farquhar2024detecting`: Sebastian Farquhar et al. (2024). *Detecting hallucinations in large language models using semantic entropy*. Nature (peer-reviewed). https://doi.org/10.1038/s41586-024-07421-0
- `kuhn2023semantic`: Lorenz Kuhn et al. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*. ICLR (peer-reviewed, CORE2023 A*). https://iclr.cc/virtual/2023/poster/11013

## Code
- `code/`: github.com/jlko/semantic_uncertainty @ `a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4 2024-04-12` (shallow clone). Licence: BSD-3-Clause-Clear.

## What it does
Samples several answers, clusters them by bidirectional entailment (meaning) and computes the entropy over the clusters. High entropy flags confabulations (Nature 2024; ICLR 2023).

## Running it on SpecDx
- **Input it needs:** The sampled outputs per item. In SpecDx: all runs' readings of one cell (or the free-text claims of one step).
- **Adaptation:** For each cell, cluster readings (exact match after v3 §2.5 normalisation for values; NLI clustering for free text such as `apr.noi_dung` or the final claim) and compute discrete semantic entropy, per family and pooled. Flag high-entropy cells as ambiguous. This replaces SpecDx's ad hoc B4 with a published, citable estimator.
- **Models:** Discrete semantic entropy needs only sampled strings, so it works with cached Ollama runs. NLI clustering needs a local entailment model (e.g. a DeBERTa-MNLI checkpoint).
- **Effort:** Low.

## Caveats
- Disagreement-only detectors miss 'semantic collapse', where models agree on one wrong reading (Richter & Papadakis 2026, in `related_work`). This is the main argument for SpecDx's anchor.
- Deng & Tang (a NeurIPS 2026 workshop paper, not yet verified) report AUC 0.47 for raw ensemble disagreement on contested IFRS rules.
