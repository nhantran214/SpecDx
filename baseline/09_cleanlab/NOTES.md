# cleanlab: Confident Learning and CROWDLAB (specification as one annotator)

**Group:** B. Statistical aggregation on the reading matrix · **Replaces/compares with:** Mis-key detector; label-error detection literature · **SpecDx defect types it can address:** Hidden requirement / mis-key

## Papers (keys in `related_work/references.bib`)
- `northcutt2021confident`: Curtis Northcutt et al. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. JAIR (peer-reviewed, CORE2020 A). https://doi.org/10.1613/jair.1.12125
- `goh2022crowdlab`: Hui Wen Goh et al. (2022). *CROWDLAB: Supervised learning to infer consensus labels and quality scores for data with multiple annotators*. NeurIPS-W (other, workshop (not ranked)). https://arxiv.org/abs/2210.06812

## Code
- `code/`: github.com/cleanlab/cleanlab @ `750625747de1b26d8530954f51f0530bd0b51d3c 2026-01-13` (shallow clone). Licence: Apache-2.0 (AGPL-free).

## What it does
Confident Learning estimates the joint distribution of given and true labels to find label errors (JAIR 2021). CROWDLAB infers consensus labels, consensus quality and per-annotator quality for multi-annotator data (NeurIPS 2022 workshop; implemented in `cleanlab.multiannotator`).

## Running it on SpecDx
- **Input it needs:** Per cell, the given label is d_c and `pred_probs` is the population's reading distribution. For CROWDLAB: a cells × interpreters label matrix, with the specification added as one more annotator.
- **Adaptation:** (a) `cleanlab.filter.find_label_issues(labels=d_c, pred_probs=population shares)` flags cells where the population confidently prefers another value: mis-key candidates. (b) `get_label_quality_multiannotator` treats the specification as an annotator. Its annotator-quality score is a global 'specification reliability' figure, and cells where consensus ≠ d_c with high quality are flagged.
- **Models:** None; CPU, offline.
- **Effort:** Low.

## Caveats
- It assumes class-conditional noise and a single true label.
- No ambiguity, weak checks or contradictions.
- CROWDLAB's paper is a workshop paper; Confident Learning (JAIR) is the citation that carries the weight.
