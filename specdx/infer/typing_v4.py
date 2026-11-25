"""v4 typing (P4a Task 3; ruling T3-N): v3's finders on v4's posteriors, plus two outcomes
that are not defects, undetermined and shared agent error.

The prior-free statistic log_lr decides and the posterior only types. Per cell:
  1. |log_lr| < tau_u: undetermined, and nothing else.
  2. log_lr <= -tau_u: no defect, so no hidden, ambiguity or weak finding even if the MAP set
     is a defect form; shared error if the {d} set's λ (not the MAP set's) reaches tau_lambda.
  3. log_lr >= tau_u: v3's finders with v4's thresholds, then shared error if the MAP set is
     {d} and λ reaches tau_lambda.
Shared error is reported on anchored cells with at least three candidate values only. On a
binary cell (|V∖{OTHER}| ≤ 2, all of S3) the error side of {d} is the other value and OTHER,
so the attractor is forced and λ's posterior is a count rule ((n_dev − 2)/(n_dev + 2) under
Beta(1, 3)): it carries no shared-error evidence, so no statistical shared error is emitted
there (ruling F-I1); on S3 shared error comes only from the execution witnesses. A shared
finding names, as witnesses, the runs that read the attractor. A cell can carry both a weak
finding and a shared error, as in v3: the weak check says a clean behaviour deviated, the
shared error that the deviations share one value.

An undetermined cell is also given its would-be type (would_be_type, ruling F-I4, design
§5.2): the type its own posterior gives as if it were determined, so reports can count
undetermined cells by the type they would otherwise have had.
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
WOULD_BE = ("hidden", "ambiguity", "weak", "shared", "none")


def _shared(cell: Cell, post: CellPosteriorV4, mine: Sequence[Reading],
            params: V4Params) -> CellFinding | None:
    if cell.anchor is None or len(post.values) <= 2:  # binary cells: ruling F-I1
        return None
    fit = post.set_fits.get(frozenset({cell.anchor}))
    if fit is None:
        return None
    lam, attractor = fit.lam, fit.attractor
    if lam < params.tau_lambda:
        return None
    witnesses = tuple(sorted(r.run_key for r in mine if r.value == attractor))
    return CellFinding(cell.id, SHARED, lam, "", (attractor,) if attractor else (), witnesses)


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
            shared = _shared(cell, post, mine, params) if post.map_set == {cell.anchor} else None
        else:
            shared = _shared(cell, post, mine, params)
        if shared is not None:
            out.append(shared)
    return out


def would_be_type(cell: Cell, post: CellPosteriorV4, mine: Sequence[Reading],
                  params: V4Params) -> str:
    """The type an undetermined cell's own posterior gives as if it were determined (rule 3
    of the module docstring, whatever its log_lr): its first defect type in WOULD_BE order,
    else "shared" when rule 3 would report a shared error (never on a binary cell), else
    "none"."""
    found = {f.type for f in _defects(cell, post, mine, params)}
    for kind in DEFECT_TYPES_V4:
        if kind in found:
            return kind
    if post.map_set == {cell.anchor} and _shared(cell, post, mine, params) is not None:
        return "shared"
    return "none"
