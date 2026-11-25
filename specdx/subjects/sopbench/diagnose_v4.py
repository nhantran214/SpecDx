"""v4 diagnosis of one SOPBench domain, with execution witnesses (P4a design §5.2).

The v4 fit and typing run over the S3 cells. An outcome cell the data cannot decide
(undetermined) whose plurality reading differs from the anchor is routed to the task's
execution witness (ruling W-1): SOPBench's strict replay of the task's own constraints. A
replay that contradicts the anchor is a self-inconsistent hidden defect; one that agrees with
it makes the agents' shared deviation a witnessed shared error. An undetermined outcome cell
whose plurality is the anchor has no shared deviation to witness and stays undetermined, as do
a cell whose replay errors and every undetermined check cell. Every undetermined cell also
gets its would-be type (ruling F-I4).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from specdx.infer.anchored_v4 import AnchoredFitV4, V4Params, fit_v4
from specdx.infer.cells import Reading
from specdx.infer.typing import CellFinding
from specdx.infer.typing_v4 import SHARED, UNDETERMINED, type_cells_v4, would_be_type
from specdx.subjects.sopbench.cells import S3CellData
from specdx.subjects.sopbench.diagnose import OFFLINE, contradictions
from specdx.subjects.sopbench.witness import witness
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
    values == (WITNESSED,). cell_outcomes: every cell of the domain -> its fate: "defect",
    "shared", "undetermined", "witnessed_defect", "witnessed_shared", "witnessed_error" or
    "none" (a witness error is not also "undetermined" here). A cell without readings (every
    run of its task a tool error) is not fitted and appears as "none". would_be: each cell typed
    undetermined by the statistics (witnessed later or not) -> its would-be type, one of
    typing_v4.WOULD_BE (ruling F-I4)."""

    domain: str
    raters: tuple[str, ...]
    fit: AnchoredFitV4
    defects: list[Defect]
    shared: list[CellFinding]
    undetermined: dict[str, int]
    witnessed: dict[str, int]
    cell_outcomes: dict[str, str] = field(default_factory=dict)
    would_be: dict[str, str] = field(default_factory=dict)


def _task_index(cell_id: str) -> int:
    """The task index of s3:<domain>:<task>:<part>."""
    return int(cell_id.rsplit(":", 2)[1])


def _deviates(mine: Sequence[Reading], anchor: str | None) -> bool:
    """Ruling W-1: the cell's plurality reading differs from the anchor (a tie with the anchor
    is not a deviation)."""
    counts = Counter(r.value for r in mine)
    return bool(counts) and counts.get(anchor, 0) < max(counts.values())


def _check_alignment(cells: Mapping, raw_tasks: Sequence[dict]) -> None:
    """The raw tasks are indexed like the cells: every outcome cell's task exists, and where
    a raw task names its user goal, it is the cell's (group outcome:<goal>)."""
    for cid, cell in cells.items():
        if not cid.endswith(":outcome"):
            continue
        index = _task_index(cid)
        if index >= len(raw_tasks):
            raise ValueError(f"raw tasks: {len(raw_tasks)} given, but {cid} needs task "
                             f"{index}")
        goal = raw_tasks[index].get("user_goal")
        if goal is not None and cell.group != f"outcome:{goal}":
            raise ValueError(f"raw tasks are not aligned with the cells: task {index} has "
                             f"goal {goal!r}, {cid} is in group {cell.group!r}")


def diagnose_domain_v4(
    domain: str, cd: S3CellData, models: Mapping[str, str], raw_tasks: Sequence[dict],
    params: V4Params, *, composite: bool = True, profile: bool = True, cell_beta: bool = True,
    use_witness: bool = True,
) -> S3DiagnosisV4:
    """Fit v4, type its cells, route undetermined outcome cells to the witness, and add
    the contradictions of the determined data (P3a's hitting sets over fit.weights).
    raw_tasks: the domain's raw tasks, indexed like load_tasks. Each task is witnessed at
    most once. Determined hidden findings carry the OFFLINE note: only witnessed ones were
    refuted. composite, profile, cell_beta: the fit's ablation switches (design §6.4);
    use_witness=False skips the witness (every undetermined cell stays undetermined). Raises
    ValueError when raw_tasks is not aligned with the cells."""
    data = cd.data
    _check_alignment(data.cells, raw_tasks)
    fit = fit_v4(data.cells, data.readings, models=models, params=params,
                 composite=composite, profile=profile, cell_beta=cell_beta)
    by_cell = data.by_cell()
    defects: list[Defect] = []
    shared: list[CellFinding] = []
    undetermined = {"outcome": 0, "check": 0}
    witnessed = {"defect": 0, "shared": 0, "error": 0}
    outcomes = {c: "none" for c in by_cell}
    would_be: dict[str, str] = {}
    for f in type_cells_v4(data.cells, fit, by_cell, params):
        mine = by_cell.get(f.cell, [])
        scenarios = tuple(sorted({r.scenario for r in mine}))
        cell = data.cells[f.cell]
        if f.type == UNDETERMINED:
            would_be[f.cell] = would_be_type(cell, fit.cells[f.cell], mine, params)
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
        elif not use_witness or not _deviates(mine, cell.anchor):
            undetermined["outcome"] += 1
            outcomes[f.cell] = "undetermined"
        else:
            # One outcome cell per task, so each task is witnessed at most once.
            result = witness(domain, raw_tasks[_task_index(f.cell)])
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
                          outcomes, would_be)
