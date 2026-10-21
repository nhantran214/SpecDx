"""JEST's N+1-version rule (Park et al., ICSE 2021; baselines spec §2.9): a cell is a
specification defect when the share of models whose modal reading deviates from the
specification's declared value reaches τ_JEST. The score is that share."""

from __future__ import annotations

from dataclasses import dataclass

from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.baselines.votes import model_votes
from specdx.infer.cells import Cell, Reading
from specdx.types import Defect


def merge_max(defects: list[Defect]) -> list[Defect]:
    best: dict[tuple[str, tuple[str, ...]], Defect] = {}
    for d in defects:
        if d.key not in best or d.score > best[d.key].score:
            best[d.key] = d
    return sorted(best.values(), key=lambda d: (-d.score, d.location))


def hidden_defect(cell: Cell, readings: list[Reading], score: float) -> Defect:
    """A hidden finding at the cell's clause, witnessed by the readings that deviate from
    the declared value."""
    off = [r for r in readings if r.value != cell.anchor]
    return Defect("hidden", (cell.clause,), tuple(sorted({r.scenario for r in off})),
                  tuple(sorted(r.run_key for r in off)), score)


@dataclass(frozen=True)
class JestRule:
    name: str = "jest"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        out: list[Defect] = []
        by_cell = inp.data.by_cell()
        for cid, votes in model_votes(inp.data).items():
            cell = inp.data.cells[cid]
            if cell.anchor is None or cell.clause is None:
                continue
            share = sum(v != cell.anchor for v in votes.values()) / len(votes)
            if share > 0:
                out.append(hidden_defect(cell, by_cell[cid], share))
        return BaselineOutput(merge_max(out))


register(JestRule())
