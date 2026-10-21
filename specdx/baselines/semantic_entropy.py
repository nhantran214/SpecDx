"""B4: discrete semantic entropy (Farquhar et al., Nature 2024; baselines spec §2.7).

Readings are clustered by exact normalised value (cells hold short values, so meaning
clusters are value clusters). A cell's score is the larger of its pooled entropy and its
per-family entropies. Candidates are localised in T with the same ddmin localiser and
budget as SpecDx (v3 §5.3), highest score first until the budget runs out; the oracle
passes while the re-runs still read the cell two or more ways."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from specdx.active.budget import RunBudget
from specdx.active.loop import oracle_interpreters
from specdx.baselines.api import REGISTRY, BaselineInput, BaselineOutput, register
from specdx.cost import Cost, run_cost
from specdx.infer.cells import Reading
from specdx.types import Defect

__all__ = ["REGISTRY", "cell_score", "entropy"]


def entropy(values: Sequence[str]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return -sum((k / n) * math.log(k / n) for k in Counter(values).values())


def cell_score(readings: Sequence[Reading]) -> float:
    pooled = entropy([r.value for r in readings])
    fams: dict[str, list[str]] = {}
    for r in readings:
        fams.setdefault(r.family, []).append(r.value)
    return max([pooled, *(entropy(v) for v in fams.values())])


@dataclass(frozen=True)
class SemanticEntropy:
    name: str = "semantic_entropy"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("ambiguity",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        scored = []
        for cid, rs in inp.data.by_cell().items():
            s = cell_score(rs)
            if s <= 0:
                continue
            cell = inp.data.cells[cid]
            scenario = Counter(r.scenario for r in rs).most_common(1)[0][0]
            note = f"{scenario}|{cell.steps[0] if cell.steps else ''}"
            scored.append((s, cid, Defect("ambiguity", (cell.location,), (scenario,),
                                          tuple(sorted(r.run_key for r in rs)), s, note)))
        scored.sort(key=lambda t: (-t[0], t[1]))
        cfg = inp.config
        budget = RunBudget(0, cfg.localize_budget)
        acted = {r.model for r in inp.data.readings}
        interps = oracle_interpreters(cfg.models_by_family, inp.report.theta, cfg.variants, acted)
        spent: list[Cost] = []
        out: list[Defect] = []
        for _, cid, d in scored:
            def persists(runs, cid=cid) -> bool:
                spent.extend(run_cost(b) for b in runs)
                data = inp.ctx.cells.extract(runs)
                return len({r.value for r in data.by_cell().get(cid, [])}) >= 2

            where = None
            if not budget.exhausted("ddmin") and inp.ctx.localizer is not None:
                where = inp.ctx.localizer(inp.variant, d, interps, budget, persists)
            out.append(Defect(d.type, tuple(where or ()), d.scenarios, d.witnesses, d.score,
                              d.note))
        return BaselineOutput(out, sum(spent, Cost()), extra={"ddmin_runs": budget.ddmin})


register(SemanticEntropy())
