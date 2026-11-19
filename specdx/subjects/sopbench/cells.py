"""Tell corpus, cells and readings for SOPBench runs (design §13d)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from specdx.infer.cells import SATISFIED, VIOLATED, Cell, CellData, Reading
from specdx.spec.tell import Chunk, Sentence, TellCorpus
from specdx.subjects.sopbench.load import S3Run, S3Task

OUTCOME_VALUES = ("succeeds", "refused")
CHECKS = ("db_match", "constraints", "process")


def cell_id(domain: str, task: int, part: str) -> str:
    return f"s3:{domain}:{task}:{part}"


@dataclass(frozen=True)
class S3CellData:
    data: CellData
    violated: dict[str, frozenset[str]]
    excluded_tool_error: int


def tell_for(domain: str, runs: Sequence[S3Run]) -> tuple[TellCorpus, int]:
    """One chunk per task from the first run's prompt; counts runs whose prompt differs."""
    first: dict[int, str] = {}
    differ = 0
    for run in runs:
        if run.task not in first:
            first[run.task] = run.prompt
        elif run.prompt != first[run.task]:
            differ += 1
    chunks = []
    for task in sorted(first):
        cid = f"s3:{domain}:{task}"
        lines = [ln.strip() for ln in first[task].splitlines() if ln.strip()]
        sentences = tuple(Sentence(f"{cid}:{n:02d}", t) for n, t in enumerate(lines))
        chunks.append(Chunk(cid, "sopbench", sentences))
    return TellCorpus(chunks), differ


def cells_for(tasks: Sequence[S3Task], runs: Sequence[S3Run]) -> S3CellData:
    cells: dict[str, Cell] = {}
    for t in tasks:
        oid = cell_id(t.domain, t.index, "outcome")
        cells[oid] = Cell(
            oid, "value", OUTCOME_VALUES[0] if t.should_succeed else OUTCOME_VALUES[1],
            oid, oid, f"outcome:{t.user_goal}", ())
        for part in CHECKS:
            cid = cell_id(t.domain, t.index, part)
            cells[cid] = Cell(cid, "constraint", SATISFIED, cid, cid, part, ())
    readings: list[Reading] = []
    violated: dict[str, frozenset[str]] = {}
    excluded = 0
    for run in runs:
        if run.tool_error:
            excluded += 1
            continue
        scenario = f"{run.domain}:{run.task}"
        checks = {"db_match": run.db_match, "constraints": run.constraints,
                  "process": run.process}
        clean = all(checks.values())
        values = {"outcome": OUTCOME_VALUES[0] if run.succeeded else OUTCOME_VALUES[1]}
        values |= {p: SATISFIED if ok else VIOLATED for p, ok in checks.items()}
        for part, value in values.items():
            readings.append(Reading(
                cell_id(run.domain, run.task, part), run.run_key, run.model, run.family,
                scenario, value, clean, 0))
        violated[run.run_key] = frozenset(
            cell_id(run.domain, run.task, p) for p, ok in checks.items() if not ok)
    return S3CellData(CellData(cells, readings), violated, excluded)
