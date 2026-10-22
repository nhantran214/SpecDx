# JEST rule: N+1-version differential testing (the specification as one more version)

**Group:** B. Statistical aggregation on the reading matrix · **Replaces/compares with:** Named, published counterpart of the 'majority vote' and 'anchor-only' simulation baselines · **SpecDx defect types it can address:** Hidden requirement / mis-key; agent-error vs specification-defect separation

## Papers (keys in `related_work/references.bib`)
- `park2021jest`: Jihyeok Park et al. (2021). *JEST: N+1-Version Differential Testing of Both JavaScript Engines and Specification*. ICSE (peer-reviewed, CORE2023 A*). https://doi.org/10.1109/icse43902.2021.00015

## Code


## What it does
JEST (ICSE 2021) generates conformance tests from the ECMAScript specification and runs them on several engines. A failure shared by most engines is attributed to the specification, a minority failure to an engine, and specification bugs are localised statistically (27 bugs found in ES11). It is the closest precedent for SpecDx's 'every competent interpreter deviates at the same place' signal.

## Running it on SpecDx
- **Input it needs:** A deviation matrix: cells × interpreters, where a deviation means the reading differs from d_c.
- **Adaptation:** Reimplement the decision rule: a cell is a specification defect when the share of models (or families) deviating from d_c exceeds a threshold; otherwise the deviations are agent errors. Report it next to the anchor-only and majority-vote baselines of v3 §8.1.
- **Models:** None; offline.
- **Effort:** Very low (a few lines).

## Caveats
- The JEST code (github.com/kaist-plrg/jest) is specific to ECMAScript and its engines, so it was not cloned.
- JEST assumes near-deterministic engines. LLM interpreters are stochastic and of uneven competence, which is exactly the gap SpecDx's accuracy model fills.
