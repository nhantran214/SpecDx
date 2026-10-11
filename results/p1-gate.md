# P1 gate: NOT PASSED

## Verdict

The gate needs 3 wins of 4 (spec §4.8, §11 item 12). Contradiction and hidden: SpecDx's typed F1 must exceed both baselines' untyped F1, since the baselines cannot assign a type. Ambiguity and weak: the baselines report clauses, which can never locate those truths, so SpecDx's typed F1 must reach 0.40.

| type | rule | SpecDx typed | B2 untyped | B3 untyped | SpecDx untyped | SpecDx wins |
|---|---|---:|---:|---:|---:|---|
| contradiction | > baselines' untyped F1 | 0.75 | 0.75 | 0.00 | 0.89 | no |
| hidden | > baselines' untyped F1 | 0.44 | 0.57 | 0.00 | 0.57 | no |
| ambiguity | ≥ 0.40 (baselines cannot locate it) | 0.00 | 0.00 | 0.00 | 0.00 | no |
| weak | ≥ 0.40 (baselines cannot locate it) | 0.00 | 0.00 | 0.00 | 0.00 | no |

## Literal comparison (typed against typed), for transparency only

| type | SpecDx typed | B2 typed | B3 typed | SpecDx wins |
|---|---:|---:|---:|---|
| contradiction | 0.75 | 0.00 | 0.00 | yes |
| hidden | 0.44 | 0.40 | 0.00 | yes |
| ambiguity | 0.00 | 0.00 | 0.00 | no |
| weak | 0.00 | 0.00 | 0.00 | no |
