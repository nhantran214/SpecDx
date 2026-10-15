"""The simulation's baselines (spec v3 §8.1): majority vote over models, "deviates from the
anchor ⇒ defect", and v2's stage 2 on joint class keys built from the simulated items."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from specdx.infer.cells import Reading
from specdx.infer.latent import Observation, fit
from specdx.infer.typing import resolve_calls
from specdx.sim.generator import SimSpec

Predictions = set[tuple[str, str]]


def by_cell(spec: SimSpec) -> dict[str, list[Reading]]:
    """Each cell's readings in reading order, every cell of the specification a key (an
    unread cell has none). The baselines, the study and G0 share this one helper."""
    out: dict[str, list[Reading]] = {c: [] for c in spec.cells}
    for r in spec.readings:
        out[r.cell].append(r)
    return out


def _resolved(spec: SimSpec, raw: Predictions) -> Predictions:
    """A baseline's raw (cell, type) calls through the cell-level rules v3 gets from
    `type_cells`: an anchored ambiguity brings its hidden requirement, and a cell reporting
    hidden or ambiguity reports no weak check."""
    kinds: dict[str, set[str]] = {}
    for cid, kind in raw:
        kinds.setdefault(cid, set()).add(kind)
    return {(cid, kind) for cid, ks in kinds.items()
            for kind in resolve_calls(spec.cells[cid].anchor is not None, ks)}


def _mode(values: Sequence[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def majority_vote(spec: SimSpec) -> Predictions:
    """Each model votes its modal reading. Hidden: the plurality of the votes differs from
    the anchor. Ambiguity: the models' votes differ. Weak: every model votes x on a value
    cell, and a clean behaviour read another value."""
    out: Predictions = set()
    for cid, rs in by_cell(spec).items():
        if not rs:
            continue
        cell = spec.cells[cid]
        votes = [_mode([r.value for r in rs if r.model == m])
                 for m in sorted({r.model for r in rs})]
        top = _mode(votes)
        if cell.anchor is not None and top != cell.anchor:
            out.add((cid, "hidden"))
        if len(set(votes)) >= 2:
            out.add((cid, "ambiguity"))
        elif cell.kind != "constraint" and any(r.value != top and r.clean for r in rs):
            out.add((cid, "weak"))
    return _resolved(spec, out)


def anchor_only(spec: SimSpec) -> Predictions:
    """Hidden wherever most readings deviate from the anchor; no other type."""
    return {(cid, "hidden") for cid, rs in by_cell(spec).items()
            if rs and spec.cells[cid].anchor is not None
            and sum(r.value != spec.cells[cid].anchor for r in rs) * 2 > len(rs)}


def stage2_v2(spec: SimSpec, k_max: int = 6, p_single: float = 0.9) -> Predictions:
    """v2's stage 2 (latent.fit, Beta(7, 3)) on joint class keys: a behaviour's class at an
    item is the tuple of its readings there. Typing as v2's scores.py does it, per cell:
    Hidden(c) ≥ 0.5 where some admissible class violates c alone (reads off its anchor);
    an ambiguity at the cells the MAP set's classes differ on when P(|A| ≥ 2) ≥ 0.5; a weak
    check at the cells where a clean class outside the MAP set, read twice or more,
    differs from the MAP set's best class, when its outside mass is ≥ 0.5."""
    rows: dict[tuple[str, str], dict[str, str]] = {}
    who: dict[str, Reading] = {}
    for r in spec.readings:
        item = f"{spec.cells[r.cell].steps[0]}|{r.scenario}"
        rows.setdefault((item, r.run_key), {})[r.cell] = r.value
        who[r.run_key] = r
    classes = {key: repr(sorted(cells.items())) for key, cells in rows.items()}
    obs = [Observation(item, run, who[run].model, who[run].family, classes[(item, run)])
           for item, run in sorted(rows)]
    post = fit(obs, models=dict(spec.models), k_max=k_max, p_single=p_single)
    content = {classes[key]: rows[key] for key in rows}
    out: Predictions = set()
    for item, ip in post.items.items():
        counts = Counter(classes[key] for key in rows if key[0] == item)

        def off(k: str) -> frozenset[str]:
            return frozenset(c for c, v in content[k].items()
                             if spec.cells[c].anchor is not None and v != spec.cells[c].anchor)

        for cid in {c for k in ip.classes for c in content[k]}:
            if spec.cells[cid].anchor is None:
                continue
            hits = {k for k in ip.classes if off(k) == {cid}}
            if sum(p for s, p in ip.set_posterior.items() if s & hits) >= 0.5:
                out.add((cid, "hidden"))
        best = ip.map_set
        if ip.p_ambiguous >= 0.5 and len(best) >= 2:
            members = sorted(best)
            for cid in content[members[0]]:
                if len({content[k].get(cid) for k in members}) >= 2:
                    out.add((cid, "ambiguity"))
            continue
        reference = max(sorted(best), key=lambda k: ip.p_member.get(k, 0.0))
        for k in sorted(counts):
            if k in best or counts[k] < 2 or off(k):
                continue
            outside = (1.0 if k not in ip.classes
                       else sum(p for s, p in ip.set_posterior.items() if k not in s))
            if outside >= 0.5:
                for cid, v in content[k].items():
                    if spec.cells[cid].kind != "constraint" and content[reference].get(cid) != v:
                        out.add((cid, "weak"))
    return _resolved(spec, out)
