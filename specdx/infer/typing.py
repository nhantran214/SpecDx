"""v3.1 typing (spec v3.1 §5).

The cell-level core turns a fit and its thresholds into cell findings: hidden candidates
(§5.1), ambiguities (§5.3) and weak checks (§5.4). The simulation scores these findings
directly; the real diagnosis refutes the hidden candidates and adds multi-clause
contradictions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from specdx.active.refute import SYMBOLIC_WITNESS
from specdx.infer.anchored import AnchoredFit, CellPosterior, V3Params
from specdx.infer.cells import Cell, CellData, Reading
from specdx.infer.scores import (
    UNATTEMPTED,
    UNOBSERVABLE,
    Diagnosis,
    ambiguity_key,
    hitting_candidates,
    identity,
    merge_defects,
    observable_clauses,
    refutation_order,
    scenario_groups,
)
from specdx.types import Behaviour, Defect, Refuter

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


# The real diagnosis (§5.1–§5.5): refutation, multi-clause contradictions, observability.

@dataclass
class CellDiagnosis(Diagnosis):
    """findings: every cell finding before refutation. ambiguity_cells: each reported
    ambiguity's identity before localisation (scores.ambiguity_key) → the cell behind it,
    the best scored when several merged: the ddmin oracle re-reads that cell (§5.3).
    spent: the multi-clause candidates refuted by a probe run, or never attempted, that
    found nothing (resolution 8); a cell candidate always yields a finding, so no singleton
    location is spent in v3.1."""

    findings: list[CellFinding] = field(default_factory=list)
    ambiguity_cells: dict[tuple[str, str, tuple[str, ...]], str] = field(default_factory=dict)


def _observable_cell(cell: Cell, readings: Sequence[Reading], observable: frozenset[str]) -> bool:
    """Stage 1 (v2 §3.5): a cell carries evidence when an observable item reads it."""
    return any(f"{r.scenario}|{step}" in observable for r in readings for step in cell.steps)


def _scenarios(readings: Sequence[Reading]) -> tuple[str, ...]:
    return tuple(sorted({r.scenario for r in readings}))


def refutation_scenario(cell: Cell, readings: Sequence[Reading]) -> str:
    """Where a candidate is refuted: its own scenario, or for a pooled cell the scenario
    with the most acted readings outside d_c, ties to the smaller scenario id."""
    off = Counter(r.scenario for r in readings if r.value != cell.anchor)
    every = _scenarios(readings)
    return min(every, key=lambda s: (-off[s], s))


def _note(cell: Cell, readings: Sequence[Reading]) -> str:
    """An ambiguity's or weak check's item, scenario|step: the scenario with the most
    readings (ties to the smaller id) and the cell's first in-scope step."""
    counts = Counter(r.scenario for r in readings)
    return f"{min(counts, key=lambda s: (-counts[s], s))}|{cell.steps[0]}"


def diagnose_v3(
    behaviours: Sequence[Behaviour],
    violated: Mapping[str, frozenset[str]],
    data: CellData,
    fit: AnchoredFit,
    refuter: Refuter,
    *,
    observable: frozenset[str],
    relevant: Mapping[str, frozenset[str]],
    tau: float = 0.05,
    known: frozenset[tuple[str, ...]] = frozenset(),
) -> CellDiagnosis:
    """violated: each run's V(b), restricted to the items where it acted (v2 §3.4).
    observable: the observable items (stage 1). relevant: step id → the clauses relevant to
    it. known: locations the base report holds, refuted last on a mutant (v1 §11 item 13).
    Thresholds come from fit.params. The multi-clause path is v2's (scores.py's shared
    helpers); witnesses are sorted and the refuter sees each group in run-key order, so
    the diagnosis does not depend on the order of the behaviours."""
    params = fit.params
    by_cell = data.by_cell()
    findings = type_cells(data.cells, fit, by_cell, params)
    groups = {sid: sorted(group, key=lambda b: b.run_key)
              for sid, group in scenario_groups(behaviours).items()}

    def witnesses(group: Sequence[Behaviour], targets: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted(b.run_key for b in group if violated[b.run_key] & targets))

    defects: list[Defect] = []
    unobservable: list[Defect] = []
    spent: set[tuple[str, ...]] = set()
    # (score, scenario, targets, the cell finding or None for a multi-clause candidate)
    candidates: list[tuple[float, str, frozenset[str], CellFinding | None]] = []
    for f in findings:
        if f.type != "hidden":
            continue
        cell, mine = data.cells[f.cell], by_cell[f.cell]
        assert cell.clause is not None
        if not _observable_cell(cell, mine, observable):
            unobservable.append(Defect(UNOBSERVABLE, (cell.clause,), _scenarios(mine),
                                       f.witnesses, f.score, "no observable item reads the cell"))
            continue
        candidates.append((f.score, refutation_scenario(cell, mine), frozenset({cell.clause}), f))
    for support, sid, hitting in hitting_candidates(groups, violated, fit.weights, tau):
        if len(hitting) < 2:
            continue  # a single clause is the cell path's: hidden or a contradiction of 1
        if not observable_clauses(hitting, sid, observable, relevant):
            unobservable.append(Defect(UNOBSERVABLE, tuple(sorted(hitting)), (sid,),
                                       witnesses(groups[sid], hitting), support,
                                       "no observable item reads every clause"))
            continue
        candidates.append((support, sid, hitting, None))
    candidates.sort(key=refutation_order(known))
    for score, sid, targets, finding in candidates:
        group = groups[sid]
        location = tuple(sorted(targets))
        result = refuter.refute(targets, group[0].scenario, group)
        failed = not result.refuted and result.attempted
        if finding is not None:
            mine = by_cell[finding.cell]
            if failed:
                defects.append(Defect("contradiction", location, _scenarios(mine),
                                      finding.witnesses, score,
                                      f"unrefuted after {result.attempts} attempts"))
            else:
                note = f"refuted by {result.witness}" if result.refuted else UNATTEMPTED
                defects.append(Defect("hidden", location, _scenarios(mine), finding.witnesses,
                                      score, note, finding.subtype))
        elif failed:
            defects.append(Defect("contradiction", location, (sid,), witnesses(group, targets),
                                  score, f"unrefuted after {result.attempts} attempts"))
        elif not result.refuted or result.witness != SYMBOLIC_WITNESS:
            spent.add(location)  # it cost a probe, or would have, and found nothing
    cells_of: dict[tuple[str, str, tuple[str, ...]], tuple[float, str]] = {}
    for f in findings:
        if f.type == "hidden":
            continue
        cell, mine = data.cells[f.cell], by_cell[f.cell]
        if not _observable_cell(cell, mine, observable):
            continue
        d = Defect(f.type, (cell.location,), _scenarios(mine), f.witnesses, f.score,
                   _note(cell, mine))
        defects.append(d)
        if f.type == "ambiguity":
            key = ambiguity_key(d)
            if key not in cells_of or f.score > cells_of[key][0]:
                cells_of[key] = (f.score, cell.id)
    return CellDiagnosis(merge_defects(defects, key=identity), merge_defects(unobservable), {},
                         frozenset(spent), findings, {k: v for k, (_, v) in cells_of.items()})
