# PySAT + MARCO: symbolic MUS/MCS enumeration on C alone

**Group:** C. Component baselines and ablations · **Replaces/compares with:** B5 (check-only consistency analysis); ablation of SpecDx step 3 (hard vs soft hitting sets) · **SpecDx defect types it can address:** Contradiction (symbolically expressible only)

## Papers (keys in `related_work/references.bib`)
- `ignatiev2018pysat`: Alexey Ignatiev et al. (2018). *PySAT: A Python Toolkit for Prototyping with SAT Oracles*. SAT (peer-reviewed, CORE2023 A). https://doi.org/10.1007/978-3-319-94144-8_26
- `liffiton2016fast`: Mark H. Liffiton et al. (2016). *Fast, flexible MUS enumeration*. Constraints (peer-reviewed, CORE2020 A). https://doi.org/10.1007/s10601-015-9183-0

## Code
- `code-pysat/`: github.com/pysathq/pysat @ `5c6d801fb1fae958f6504af5692fa26b807c8e76 2026-10-20` (shallow clone). Licence: MIT.
- `code-marco/`: github.com/liffiton/MARCO @ `27e056bd5199ab995cf57473ea17c6a2762ff564 2026-07-30` (shallow clone). Licence: custom (see LICENSE: MARCO plus bundled MiniSat/MUSer2 licences).

## What it does
PySAT provides MUSX (MUS extraction), LBX/MCSls (MCS enumeration), RC2 (MaxSAT) and Hitman (minimal hitting sets). MARCO enumerates MUSes and MCSes anytime over CNF, GCNF or SMT.

## Running it on SpecDx
- **Input it needs:** The clause set C encoded as constraints over a finite scenario-state domain (CNF via PySAT, or SMT for MARCO).
- **Adaptation:** (a) B5: encode the lifted clauses C' over the bounded state domain of a scenario and enumerate MUSes, which are contradiction candidates with no interpreters. (b) Ablation: feed SpecDx's observed violated-clause sets V(b) to Hitman as *hard* sets, against SpecDx's soft, weighted hitting (v1 §3.3).
- **Models:** None; offline.
- **Effort:** Medium for (a): an encoding of SQL invariants and guardrails is needed. Low for (b).

## Caveats
- Only contradictions are visible to C-only analysis; T is ignored.
- Clauses that call PostgreSQL functions (`fn_reconcile`) need a faithful encoding or must be excluded. R-12 is the test case.
