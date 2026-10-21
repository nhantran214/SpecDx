"""GLAD and Dawid–Skene on the cell readings (crowd-kit; baselines spec §2.9).
task = cell, worker = interpreter (model, Tell variant, seed), label = its modal reading.
A cell is a hidden-requirement candidate scored 1 − P(declared value)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from crowdkit.aggregation import GLAD, DawidSkene

from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.baselines.jest_rule import hidden_defect, merge_max
from specdx.baselines.votes import interpreter_votes
from specdx.types import Defect


@dataclass(frozen=True)
class TruthInference:
    name: str
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def _proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.name == "glad":
            return GLAD(n_iter=100, silent=True).fit_predict_proba(df)
        return DawidSkene(n_iter=100).fit_predict_proba(df)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        votes = interpreter_votes(inp.data, inp.report.behaviours)
        anchored = {c: v for c, v in votes.items()
                    if inp.data.cells[c].anchor is not None and inp.data.cells[c].clause}
        rows = [(c, w, lab) for c, by in anchored.items() for w, lab in by.items()]
        if not rows:
            return BaselineOutput([])
        proba = self._proba(pd.DataFrame(rows, columns=["task", "worker", "label"]))
        by_cell = inp.data.by_cell()
        out: list[Defect] = []
        for cid in sorted(anchored):
            cell = inp.data.cells[cid]
            p_anchor = float(proba.loc[cid].get(cell.anchor, 0.0)) if cid in proba.index else 0.0
            score = 1.0 - p_anchor
            if score > 0:
                out.append(hidden_defect(cell, by_cell[cid], score))
        return BaselineOutput(merge_max(out))


register(TruthInference("glad"))
register(TruthInference("dawid_skene"))
