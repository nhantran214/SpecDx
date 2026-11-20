"""v4 typing (P4a Task 3; ruling T3-N): v3's finders on v4's posteriors, plus two outcomes
that are not defects, undetermined and shared agent error.

The prior-free statistic log_lr decides and the posterior only types. Per cell:
  1. |log_lr| < tau_u: undetermined, and nothing else.
  2. log_lr <= -tau_u: no defect, so no hidden, ambiguity or weak finding even if the MAP set
     is a defect form; shared error if the {d} set's λ (not the MAP set's) reaches tau_lambda.
  3. log_lr >= tau_u: v3's finders with v4's thresholds, then shared error if the MAP set is
     {d} and λ reaches tau_lambda.
Shared error is reported on anchored cells only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from specdx.infer.anchored_v4 import AnchoredFitV4, CellPosteriorV4, V4Params
from specdx.infer.cells import Cell, Reading
from specdx.infer.typing import (
    CellFinding,
    ambiguity_finding,
    co_occur,
    hidden_finding,
    resolve_calls,
    weak_finding,
)

UNDETERMINED = "undetermined"
SHARED = "shared_error"
DEFECT_TYPES_V4 = ("hidden", "ambiguity", "weak")


def _shared(cell: Cell, post: CellPosteriorV4, params: V4Params) -> CellFinding | None:
    if cell.anchor is None:
        return None
    fit = post.set_fits.get(frozenset({cell.anchor}))
    if fit is None:
        return None
    lam, attractor = fit.lam, fit.attractor
    if lam < params.tau_lambda:
        return None
    return CellFinding(cell.id, SHARED, lam, "", (attractor,) if attractor else ())


def _defects(cell: Cell, post: CellPosteriorV4, mine: Sequence[Reading],
             params: V4Params) -> list[CellFinding]:
    out: list[CellFinding] = []
    found, ambiguity = co_occur(cell, post, mine, hidden_finding(cell, post, mine, params.tau_h),
                                ambiguity_finding(cell, post, mine, params.tau_a))
    if found is not None:
        out.append(found)
    if ambiguity is not None:
        out.append(ambiguity)
    weak = weak_finding(cell, post, mine, params.tau_w)
    kinds = {f.type for f in (found, ambiguity, weak) if f is not None}
    if weak is not None and "weak" in resolve_calls(cell.anchor is not None, kinds):
        out.append(weak)
    return out


def type_cells_v4(cells: Mapping[str, Cell], fit: AnchoredFitV4,
                  by_cell: Mapping[str, Sequence[Reading]],
                  params: V4Params) -> list[CellFinding]:
    """Every finding of the v4 fit, cell by cell in id order (see the module docstring)."""
    out: list[CellFinding] = []
    for cid in sorted(fit.cells):
        cell, post, mine = cells[cid], fit.cells[cid], by_cell.get(cid, [])
        if abs(post.log_lr) < params.tau_u:
            out.append(CellFinding(cid, UNDETERMINED, abs(post.log_lr)))
            continue
        if post.log_lr >= params.tau_u:
            out.extend(_defects(cell, post, mine, params))
            shared = _shared(cell, post, params) if post.map_set == {cell.anchor} else None
        else:
            shared = _shared(cell, post, params)
        if shared is not None:
            out.append(shared)
    return out
