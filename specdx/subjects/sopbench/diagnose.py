"""Offline diagnosis of one SOPBench domain, and population compositions (design §13d).

SpecDx's anchored cell model and typing run unchanged over the S3 cells. Nothing executes:
a hidden candidate is never refuted, and is noted so. Every result is provisional, since
V3Params were calibrated on CASE-0031.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from specdx.infer.anchored import AnchoredFit, V3Params, fit_anchored
from specdx.infer.hitting import soft_hitting_sets
from specdx.infer.typing import type_cells
from specdx.subjects.sopbench.cells import S3CellData
from specdx.subjects.sopbench.load import S3Run
from specdx.subjects.sopbench.trajectories import OPEN_WEIGHTS
from specdx.types import Defect

PROVISIONAL = "provisional: V3Params calibrated on CASE-0031, recalibrated for S3 in P3b"
OFFLINE = "refutation not attempted (offline)"
COMPOSITIONS = ("all", "open", "hosted")
TAU = 0.05
MAX_SIZE = 3
DEFAULT_PARAMS = V3Params()  # frozen, so one shared default is safe


@dataclass(frozen=True)
class S3Diagnosis:
    """raters: the models whose runs gave readings, sorted. defects: sorted by score
    descending, then by key."""

    domain: str
    raters: tuple[str, ...]
    fit: AnchoredFit
    defects: list[Defect]


def diagnose_domain(
    domain: str, cd: S3CellData, models: Mapping[str, str], params: V3Params = DEFAULT_PARAMS
) -> S3Diagnosis:
    """Fit the anchored model, type its cells, and add each task's multi-check
    contradictions: the soft minimal hitting sets of size ≥ 2 over every run's V(b), clean
    runs included, weighted by each run's faithful weight, as scores.hitting_candidates
    does for diagnose_v3. Offline no candidate is refuted, so each is reported with the
    hidden findings' note, and its witnesses are the runs whose V(b) meets it.

    Three offline adaptations of diagnose_v3, all ruled: (I1) hitting sets over every run;
    (I2) unrefuted contradictions reported with the offline note; and (M-1) no
    observability stage and no ``_note``. S3 cells have no steps (``steps=()``), so v3's
    observability stage would type every finding UNOBSERVABLE and ``_note`` would fail on
    ``steps[0]``; but every run without a tool error reads all four cells of its task, so
    every cell is observed and the stage has nothing to decide.
    models: model → family for the population (canonical model names)."""
    data = cd.data
    fit = fit_anchored(data.cells, data.readings, models=models, params=params)
    by_cell = data.by_cell()
    defects: list[Defect] = []
    for f in type_cells(data.cells, fit, by_cell, params):
        scenarios = tuple(sorted({r.scenario for r in by_cell[f.cell]}))
        defects.append(Defect(f.type, (f.cell,), scenarios, f.witnesses, f.score,
                              OFFLINE if f.type == "hidden" else "", f.subtype))
    runs_of: dict[str, set[str]] = {}
    for r in data.readings:
        runs_of.setdefault(r.scenario, set()).add(r.run_key)
    for scenario in sorted(runs_of):
        keys = sorted(runs_of[scenario])
        weighted = [(cd.violated[rk], fit.weights.get(rk, 0.0)) for rk in keys]
        for hitting, mass in soft_hitting_sets(weighted, tau=TAU, max_size=MAX_SIZE):
            if len(hitting) >= 2:
                witnesses = tuple(rk for rk in keys if cd.violated[rk] & hitting)
                defects.append(Defect("contradiction", tuple(sorted(hitting)), (scenario,),
                                      witnesses, mass, OFFLINE))
    defects.sort(key=lambda d: (-d.score, d.key))
    raters = tuple(sorted({r.model for r in data.readings}))
    return S3Diagnosis(domain, raters, fit, defects)


def compose(runs: Sequence[S3Run], which: str) -> list[S3Run]:
    """all: every run; open: runs of open-weight models (OPEN_WEIGHTS); hosted: the rest."""
    if which == "all":
        return list(runs)
    if which == "open":
        return [r for r in runs if r.model in OPEN_WEIGHTS]
    if which == "hosted":
        return [r for r in runs if r.model not in OPEN_WEIGHTS]
    raise ValueError(f"unknown composition {which!r}; expected one of {COMPOSITIONS}")
