"""v3.1 typing (spec v3.1 §5).

The cell-level core turns a fit and its thresholds into cell findings: hidden candidates
(§5.1), ambiguities (§5.3) and weak checks (§5.4). The simulation scores these findings
directly; the real diagnosis refutes the hidden candidates and adds multi-clause
contradictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from specdx.infer.anchored import AnchoredFit, CellPosterior, V3Params
from specdx.infer.cells import Cell, Reading

MIS_KEY = "mis-key"
AMBIGUOUS_PICK = "ambiguous-pick"
UNSTATED = "unstated"
SUBTYPES = {"mis": MIS_KEY, "amb": AMBIGUOUS_PICK, "empty": UNSTATED}


@dataclass(frozen=True)
class CellFinding:
    """type: hidden, ambiguity or weak. score: P(A_c ≠ {d_c}), P(|A_c| = 2) or the MAP set's
    posterior. values: an ambiguity's pair, or a weak check's admitted value then the
    deviant values. witnesses: the run keys of the readings behind the finding."""

    cell: str
    type: str
    score: float
    subtype: str = ""
    values: tuple[str, ...] = ()
    witnesses: tuple[str, ...] = ()


def subtype(post: CellPosterior) -> str:
    """§5.1: the hidden finding's subtype follows the most probable defect form. With the
    flat error pool, one model separates the forms (Theorem T2′), so no accuracy contrast
    is needed."""
    forms = post.forms
    return SUBTYPES[max(("mis", "amb", "empty"), key=lambda f: forms[f])]


def hidden_finding(cell: Cell, post: CellPosterior, readings: Sequence[Reading],
                   tau_h: float) -> CellFinding | None:
    """§5.1's candidate: an anchored cell with P(A_c ≠ {d_c}) ≥ τ_H."""
    if cell.anchor is None or post.p_defect < tau_h:
        return None
    witnesses = tuple(sorted({r.run_key for r in readings if r.value != cell.anchor}))
    return CellFinding(cell.id, "hidden", post.p_defect, subtype(post), (), witnesses)


def ambiguity_finding(cell: Cell, post: CellPosterior, readings: Sequence[Reading],
                      tau_a: float) -> CellFinding | None:
    """§5.3: P(|A_c| = 2) ≥ τ_A. No accuracy contrast and no rank-correlation rule: with the
    flat error pool one model identifies a pair (Theorem T2′), and v3's Spearman rule passed
    at about 0% of true pairs (v3.1 §5.3). A cell with one candidate value has no pair."""
    pairs = [s for s in post.sets if len(s) == 2]
    if not pairs or post.p_pair < tau_a:
        return None
    pair = max(pairs, key=lambda s: (post.sets[s], sorted(s)))
    witnesses = tuple(sorted({r.run_key for r in readings if r.value in pair}))
    return CellFinding(cell.id, "ambiguity", post.p_pair, "", tuple(sorted(pair)), witnesses)


def weak_finding(cell: Cell, post: CellPosterior, readings: Sequence[Reading],
                 tau_w: float) -> CellFinding | None:
    """§5.4: a value or event cell whose MAP set is one value x with posterior ≥ τ_W, and
    an acted behaviour that read another value and violates no relevant clause."""
    if cell.kind == "constraint":
        return None
    best = post.map_set
    if len(best) != 1 or post.sets[best] < tau_w:
        return None
    x = next(iter(best))
    deviant = [r for r in readings if r.value != x and r.clean]
    if not deviant:
        return None
    values = (x, *sorted({r.value for r in deviant}))
    return CellFinding(cell.id, "weak", post.sets[best], "", values,
                       tuple(sorted({r.run_key for r in deviant})))


def co_occur(cell: Cell, post: CellPosterior, readings: Sequence[Reading],
             hidden: CellFinding | None,
             ambiguity: CellFinding | None) -> tuple[CellFinding | None, CellFinding | None]:
    """§3.1, §5.1, §5.3 (v1 §2.4): on an anchored cell an ambiguity and its hidden
    requirement are reported together (ruling F1, resolution 11). A reported ambiguity
    brings its hidden candidate, subtype ambiguous-pick, scored max(P(A_c ≠ {d_c}),
    P(|A_c| = 2)); a hidden candidate whose most probable defect form is a pair brings its
    ambiguity. An unanchored cell reports the ambiguity alone."""
    if cell.anchor is None:
        return hidden, ambiguity
    if ambiguity is None and hidden is not None and hidden.subtype == AMBIGUOUS_PICK:
        ambiguity = ambiguity_finding(cell, post, readings, 0.0)
    if ambiguity is not None:
        base = hidden or hidden_finding(cell, post, readings, 0.0)
        assert base is not None
        hidden = CellFinding(cell.id, "hidden", max(post.p_defect, post.p_pair),
                             AMBIGUOUS_PICK, (), base.witnesses)
    return hidden, ambiguity


def resolve_calls(anchored: bool, calls: set[str]) -> set[str]:
    """The cell-level rules every method's raw calls on one cell go through, v3 and the
    simulation's baselines alike. (a) An anchored cell's ambiguity brings its hidden
    requirement (ruling F1). (b) A cell reporting a hidden requirement or an ambiguity
    reports no weak check (resolution 14: weak truths are unanchored cells only)."""
    out = set(calls)
    if anchored and "ambiguity" in out:
        out.add("hidden")
    if out & {"hidden", "ambiguity"}:
        out.discard("weak")
    return out


def type_cells(cells: Mapping[str, Cell], fit: AnchoredFit,
               by_cell: Mapping[str, Sequence[Reading]], params: V3Params) -> list[CellFinding]:
    """Every cell finding of the fit, cell by cell in id order. An ambiguity and its hidden
    requirement co-occur on an anchored cell (`co_occur`). A cell reporting an ambiguity or
    a hidden requirement reports no weak check: its readings are not deviant, and the same
    defect is not typed twice (resolution 14: weak truths are unanchored cells only)."""
    out: list[CellFinding] = []
    for cid in sorted(fit.cells):
        cell, post, mine = cells[cid], fit.cells[cid], by_cell.get(cid, [])
        found, ambiguity = co_occur(cell, post, mine,
                                    hidden_finding(cell, post, mine, params.tau_h),
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

