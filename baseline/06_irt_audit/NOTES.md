# IRT benchmark audit (B3): py-irt 2PL/4PL and the Land & Bikel forced-ceiling indicator

**Group:** B. Statistical aggregation on the reading matrix · **Replaces/compares with:** B3 (to be revised) · **SpecDx defect types it can address:** Hidden requirement / mis-key; partly Ambiguity (items on which no model improves)

## Papers (keys in `related_work/references.bib`)
- `land2026auditing`: Sander Land et al. (2026). *Auditing LLM Benchmarks with Item Response Theory*. arXiv (preprint). https://arxiv.org/abs/2605.30504
- `rodriguez2021evaluation`: Pedro Rodriguez et al. (2021). *Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards?*. ACL (peer-reviewed, CORE2023 A*). https://doi.org/10.18653/v1/2021.acl-long.346
- `lalor2023irt`: John Patrick Lalor et al. (2023). *py-irt: A Scalable Item Response Theory Library for Python*. IJOC (peer-reviewed, CORE2020 A). https://doi.org/10.1287/ijoc.2022.1250

## Code
- `code-py-irt/`: github.com/nd-ball/py-irt @ `6514928ec753771c8858bf16581e9715d6025ed9 2026-06-19` (shallow clone). Licence: MIT.

## What it does
Fits IRT models (subject ability × item difficulty and discrimination) to a correctness matrix. Rodriguez et al. flag annotation errors by negative discrimination. Land & Bikel (EMNLP 2026; arXiv 2605.30504) flag mislabels with a forced-ceiling likelihood ratio from a constrained 4PL, and report that a plain low-discrimination 2PL does worse. Land & Bikel release data only (Hugging Face `Writer/IRT-mislabeled-items`); their method must be reimplemented.

## Running it on SpecDx
- **Input it needs:** A 0/1 matrix: subjects = interpreter runs or models; items = cells (1 if the reading equals the declared value d_c) or clauses (1 if satisfied).
- **Adaptation:** Variant (a): py-irt `2pl`, ranking items with discrimination below 0. This is the current B3. Variant (b): the Land & Bikel indicator, a constrained fit with guessing c_i at the chance floor (1/|V_c| for cells, 0 for clauses) and d_i = 1, compared by likelihood ratio Δℓ_i with a free ceiling d_i. Rank items whose strong interpreters systematically miss d_c. **Recommendation:** replace B3's 'negative discrimination' rule with variant (b) and keep (a) as a secondary arm.
- **Models:** None. CPU or GPU, offline (Pyro/PyTorch).
- **Effort:** Low for (a); medium for (b), which needs a custom Pyro model or about 100 lines of PyTorch.

## Caveats
- It assumes one correct label per item, so it cannot express A_c = {x, y}, ∅ or weak checks.
- With few model families, θ spread may be too small for a stable fit, which is the same issue as SpecDx's P1b finding. Report the fit diagnostics.
