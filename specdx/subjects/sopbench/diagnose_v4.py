"""v4 diagnosis of one SOPBench domain, with execution witnesses (P4a design §5.2).

The v4 fit and typing run over the S3 cells. An outcome cell the data cannot decide
(undetermined) is routed to the task's execution witness: SOPBench's strict replay of the
task's own constraints. A replay that contradicts the anchor is a self-inconsistent hidden
defect; one that agrees with it makes the agents' deviation a shared error. A replay that
errors, and every undetermined check cell, stay undetermined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from specdx.infer.anchored_v4 import AnchoredFitV4, V4Params, fit_v4
from specdx.infer.typing import CellFinding
from specdx.infer.typing_v4 import SHARED, UNDETERMINED, type_cells_v4
from specdx.subjects.sopbench.cells import S3CellData
from specdx.subjects.sopbench.diagnose import OFFLINE, contradictions
from specdx.subjects.sopbench.witness import WitnessResult, witness
from specdx.types import Defect

WITNESSED = "witnessed"


@dataclass(frozen=True)
class S3DiagnosisV4:
    """raters: the models whose runs gave readings, sorted. defects: sorted by score
    descending, then by key. shared: shared-error findings, determined or witnessed.
    undetermined: cells still undetermined, by kind (outcome, check). witnessed: cells
    resolved by a witness, by result (defect, shared, error). Counting: a witness error is
    counted in witnessed["error"] and also in undetermined["outcome"] (it stays undetermined),
    so the cells accounted for are undetermined + witnessed defect + witnessed shared, never
    + witnessed error. shared holds typed shared findings and witnessed ones, told apart by
    values == (WITNESSED,). cell_outcomes: each cell with readings -> its fate: "defect",
    "shared", "undetermined", "witnessed_defect", "witnessed_shared", "witnessed_error" or
    "none" (a witness error is not also "undetermined" here)."""

    domain: str
    raters: tuple[str, ...]
    fit: AnchoredFitV4
    defects: list[Defect]
    shared: list[CellFinding]
    undetermined: dict[str, int]
    witnessed: dict[str, int]
    cell_outcomes: dict[str, str] = field(default_factory=dict)


def _task_index(cell_id: str) -> int:
    """The task index of s3:<domain>:<task>:<part>."""
    return int(cell_id.rsplit(":", 2)[1])


def diagnose_domain_v4(
    domain: str, cd: S3CellData, models: Mapping[str, str], raw_tasks: Sequence[dict],
    params: V4Params, *, composite: bool = True, profile: bool = True, use_witness: bool = True,
) -> S3DiagnosisV4:
    """Fit v4, type its cells, route undetermined outcome cells to the witness, and add
    the contradictions of the determined data (P3a's hitting sets over fit.weights).
    raw_tasks: the domain's raw tasks, indexed like load_tasks. Each task is witnessed at
    most once. Determined hidden findings carry the OFFLINE note: only witnessed ones were
    refuted. composite, profile: the fit's ablation switches (design §6.4); use_witness=False
    skips the witness (every undetermined cell stays undetermined)."""
    data = cd.data
    fit = fit_v4(data.cells, data.readings, models=models, params=params,
                 composite=composite, profile=profile)
    by_cell = data.by_cell()
    defects: list[Defect] = []
    shared: list[CellFinding] = []
    undetermined = {"outcome": 0, "check": 0}
    witnessed = {"defect": 0, "shared": 0, "error": 0}
    cache: dict[int, WitnessResult] = {}
    outcomes = {c: "none" for c in by_cell}
    for f in type_cells_v4(data.cells, fit, by_cell, params):
        mine = by_cell.get(f.cell, [])
        scenarios = tuple(sorted({r.scenario for r in mine}))
        cell = data.cells[f.cell]
        if f.type == SHARED:
            shared.append(f)
            if outcomes[f.cell] != "defect":
                outcomes[f.cell] = "shared"
        elif f.type != UNDETERMINED:
            outcomes[f.cell] = "defect"
            defects.append(Defect(f.type, (f.cell,), scenarios, f.witnesses, f.score,
                                  OFFLINE if f.type == "hidden" else "", f.subtype))
        elif not f.cell.endswith(":outcome"):
            undetermined["check"] += 1
            outcomes[f.cell] = "undetermined"
        elif not use_witness:
            undetermined["outcome"] += 1
            outcomes[f.cell] = "undetermined"
        else:
            index = _task_index(f.cell)
            if index not in cache:
                cache[index] = witness(domain, raw_tasks[index])
            result = cache[index]
            if result.outcome == "error":
                witnessed["error"] += 1
                undetermined["outcome"] += 1
                outcomes[f.cell] = "witnessed_error"
            elif result.outcome != cell.anchor:
                witnessed["defect"] += 1
                outcomes[f.cell] = "witnessed_defect"
                defects.append(Defect(
                    "hidden", (f.cell,), scenarios, (f"strict replay: {result.calls}",), 1.0,
                    "witnessed: strict replay", "self-inconsistent"))
            else:
                witnessed["shared"] += 1
                outcomes[f.cell] = "witnessed_shared"
                shared.append(CellFinding(
                    f.cell, SHARED, 1.0, values=(WITNESSED,),
                    witnesses=tuple(sorted(r.run_key for r in mine))))
    defects.extend(contradictions(cd, fit.weights))
    defects.sort(key=lambda d: (-d.score, d.key))
    raters = tuple(sorted({r.model for r in data.readings}))
    return S3DiagnosisV4(domain, raters, fit, defects, shared, undetermined, witnessed,
                          outcomes)
