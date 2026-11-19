"""Tell corpus, cells and readings for SOPBench runs (design §13d)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
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


def _renders(runs: Sequence[S3Run]) -> dict[int, list[str]]:
    """Each task's distinct prompts, variant 0 first: by decreasing frequency, ties broken by
    the string itself. Prompts are compared as exact strings (on the published runs, no two
    renders of a task differ only in whitespace). Tool-error runs count: the prompt is what
    the agent was given, whatever it then did."""
    counts: dict[int, Counter[str]] = {}
    for run in runs:
        counts.setdefault(run.task, Counter())[run.prompt] += 1
    return {task: sorted(c, key=lambda p: (-c[p], p)) for task, c in counts.items()}


def prompt_variants(runs: Sequence[S3Run]) -> dict[str, int]:
    """run_key → the Tell variant the run saw, decided per run (ruling I-3): within each task
    the most common prompt is variant 0 and every other distinct prompt is 1, 2, … in the
    order of _renders. Published files mix renders, so a model may see several."""
    order = {task: {p: i for i, p in enumerate(ps)} for task, ps in _renders(runs).items()}
    return {run.run_key: order[run.task][run.prompt] for run in runs}


def tell_for(domain: str, runs: Sequence[S3Run]) -> tuple[TellCorpus, int]:
    """One chunk per task from its variant-0 (majority) prompt; also counts the runs that saw
    another variant."""
    renders = _renders(runs)
    differ = sum(1 for v in prompt_variants(runs).values() if v != 0)
    chunks = []
    for task in sorted(renders):
        cid = f"s3:{domain}:{task}"
        lines = [ln.strip() for ln in renders[task][0].splitlines() if ln.strip()]
        sentences = tuple(Sentence(f"{cid}:{n:02d}", t) for n, t in enumerate(lines))
        chunks.append(Chunk(cid, "sopbench", sentences))
    return TellCorpus(chunks), differ


def cells_for(
    tasks: Sequence[S3Task], runs: Sequence[S3Run], variants: Mapping[str, int] | None = None
) -> S3CellData:
    """The §13d cells of every task and one reading per run and cell. A run is clean when its
    three checks are satisfied and its outcome reading equals the anchor (§13d as amended on
    2026-11-20, ruling I-6). variants: run_key → Tell variant; by default prompt_variants of
    ``runs``. Pass the variants of the whole domain when ``runs`` is one composition."""
    if variants is None:
        variants = prompt_variants(runs)
    should = {t.index: t.should_succeed for t in tasks}
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
        clean = all(checks.values()) and run.succeeded == should[run.task]
        values = {"outcome": OUTCOME_VALUES[0] if run.succeeded else OUTCOME_VALUES[1]}
        values |= {p: SATISFIED if ok else VIOLATED for p, ok in checks.items()}
        for part, value in values.items():
            readings.append(Reading(
                cell_id(run.domain, run.task, part), run.run_key, run.model, run.family,
                scenario, value, clean, variants[run.run_key]))
        violated[run.run_key] = frozenset(
            cell_id(run.domain, run.task, p) for p, ok in checks.items() if not ok)
    return S3CellData(CellData(cells, readings), violated, excluded)
