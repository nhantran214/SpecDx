# Active testing: acquisition for label-efficient evaluation

**Group:** C. Component baselines and ablations · **Replaces/compares with:** Alternative to Thompson-EIG scenario selection (v1 §3.5); compare with A4 (random) · **SpecDx defect types it can address:** Efficiency (runs-to-detection), all types

## Papers (keys in `related_work/references.bib`)
- `kossen2021active`: Jannik Kossen et al. (2021). *Active Testing: Sample-Efficient Model Evaluation*. ICML (peer-reviewed, CORE2023 A*). https://proceedings.mlr.press/v139/kossen21a.html

## Code
- `code/`: github.com/jlko/active-testing @ `2807fac9ad91fb12e5814e71b9a26c55df9d50fb 2025-05-22` (shallow clone). Licence: MIT.

## What it does
Chooses which test points to label so that a quantity (risk) is estimated efficiently: acquisition proportional to expected loss under a surrogate, with an unbiased importance-weighted (LURE) estimator.

## Running it on SpecDx
- **Input it needs:** A pool of candidate scenarios or cells, plus a surrogate predicting deviation or defect probability.
- **Adaptation:** Pool = synthesised scenario cells (v1 §3.5). Surrogate = the current anchored-cell posterior. Acquisition q ∝ predicted posterior entropy of the defect label. Run the pilot or committee on the sampled scenario. Compare the area under the detection-versus-runs curve with Thompson-EIG and random selection (A4). Top-two Thompson sampling or IDS (Russo 2020; Russo & Van Roy 2018, in `related_work`) are the theory-backed alternatives the t7 agent recommends.
- **Models:** The SpecDx population; the acquisition code is model-agnostic NumPy.
- **Effort:** Low–medium.

## Caveats
- Designed to estimate an average risk, whereas SpecDx's goal is identification. Use it as an acquisition-function baseline, not as the estimator.
