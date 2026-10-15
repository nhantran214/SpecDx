"""B2: the single most competent interpreter, every violated clause flagged (spec §4.5).
The most competent is the model with the highest θ_m of the anchored model (spec v3 §8.3),
which run_gate passes as theta."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence

from specdx.types import Behaviour, Defect


def b2_report(
    behaviours: Sequence[Behaviour],
    violated: Mapping[str, frozenset[str]],
    theta: Mapping[str, float],
    scenario_ids: Collection[str],
) -> list[Defect]:
    if not theta:
        return []
    best = max(sorted(theta), key=lambda m: theta[m])
    mine = [b for b in behaviours
            if b.status == "ok" and b.interpreter.model == best and b.scenario.id in scenario_ids]
    if not mine:
        return []
    counts = Counter(c for b in mine for c in violated[b.run_key])
    out = []
    for clause, n in counts.items():
        hit = [b for b in mine if clause in violated[b.run_key]]
        out.append(Defect("hidden", (clause,), tuple(sorted({b.scenario.id for b in hit})),
                          tuple(b.run_key for b in hit), n / len(mine)))
    return sorted(out, key=lambda d: (-d.score, d.location))
