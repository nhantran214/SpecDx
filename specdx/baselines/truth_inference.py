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
        df = pd.DataFrame(rows, columns=["task", "worker", "label"])
        if df["label"].nunique() == 1:  # nothing to infer: the posterior is that label
            proba = pd.DataFrame(1.0, index=sorted(anchored), columns=[df["label"].iloc[0]])
        else:
            proba = self._proba(df)
        by_cell = inp.data.by_cell()
        out: list[Defect] = []
        for cid in sorted(anchored):
            cell = inp.data.cells[cid]
            if cid not in proba.index:
                continue
            row = proba.loc[cid]
            aggregate = min(row.index, key=lambda label: (-row[label], label))
            if aggregate != cell.anchor:  # §2.9: reported when the aggregate differs from d_c
                score = 1.0 - float(row.get(cell.anchor, 0.0))
                out.append(hidden_defect(cell, by_cell[cid], score))
        return BaselineOutput(merge_max(out))


register(TruthInference("glad"))
register(TruthInference("dawid_skene"))
