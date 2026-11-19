"""Load SOPBench tasks and published trajectory runs as typed records (design §13d)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from specdx.subjects.sopbench import SOPBENCH_DIR
from specdx.subjects.sopbench.trajectories import TrajectoryError, TrajectoryName


class TrajectoryMismatch(ValueError):
    """A trajectory file is not aligned with the task file."""


@dataclass(frozen=True)
class S3Task:
    """One task of a domain file. user_known and constraint_parameters are kept, as parsed,
    only to check a trajectory record's alignment (they take no part in hashing)."""

    domain: str
    index: int
    user_goal: str
    should_succeed: bool
    user_prompt: str
    user_known: object = field(default=None, hash=False)
    constraint_parameters: object = field(default=None, hash=False)


@dataclass(frozen=True)
class S3Run:
    """One published run. model is the canonical model name (TrajectoryName.canonical), so a
    provider's copy of the same weights is the same rater (ruling I-1)."""

    domain: str
    task: int
    model: str
    family: str
    mode: str
    succeeded: bool
    db_match: bool
    constraints: bool
    process: bool
    tool_error: bool
    prompt: str

    @property
    def run_key(self) -> str:
        return f"{self.model}|{self.domain}|{self.task}"


@dataclass(frozen=True)
class LoadStats:
    records: int
    runs: int
    missing_evaluation: int


def load_tasks(domain: str, root: Path = SOPBENCH_DIR) -> list[S3Task]:
    raw = json.loads((root / "data" / f"{domain}_tasks.json").read_text())
    tasks: list[S3Task] = []
    for goal, group in raw.items():
        for t in group:
            tasks.append(
                S3Task(
                    domain=domain,
                    index=len(tasks),
                    user_goal=goal,
                    should_succeed=bool(t["action_should_succeed"]),
                    user_prompt=t["user_prompt"],
                    user_known=t["user_known"],
                    constraint_parameters=t["constraint_parameters"],
                )
            )
    return tasks


def _aligned(rec_task: dict, task: S3Task) -> bool:
    """A record belongs to its task when every identifying field agrees. user_prompt alone is
    not enough: university tasks 0-6 share one prompt but differ in their label (ruling I-5)."""
    return (rec_task["user_prompt"] == task.user_prompt
            and rec_task["user_goal"] == task.user_goal
            and bool(rec_task["action_should_succeed"]) == task.should_succeed
            and rec_task["user_known"] == task.user_known
            and rec_task["constraint_parameters"] == task.constraint_parameters)


def load_runs(
    path: Path, name: TrajectoryName, family: str, tasks: Sequence[S3Task]
) -> tuple[list[S3Run], LoadStats]:
    try:
        records = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise TrajectoryError(str(path)) from exc
    if not isinstance(records, list):
        raise TrajectoryError(str(path))
    if len(records) != len(tasks):
        raise TrajectoryMismatch(f"{path}: {len(records)} records for {len(tasks)} tasks")
    runs: list[S3Run] = []
    missing = 0
    for i, (rec, task) in enumerate(zip(records, tasks, strict=True)):
        if not _aligned(rec["task"], task):
            raise TrajectoryMismatch(f"{path}: record {i} does not match task {i}")
        if not rec.get("evaluations"):
            missing += 1
            continue
        ev = rec["evaluations"][0]
        runs.append(
            S3Run(
                domain=task.domain,
                task=task.index,
                model=name.canonical,
                family=family,
                mode=name.mode,
                succeeded=bool(ev["action_successfully_called"]),
                db_match=bool(ev["database_match"]),
                constraints=bool(ev["constraint_not_violated"]),
                process=bool(ev["dirgraph_satisfied"]),
                tool_error=not ev["no_tool_call_error"],
                prompt=rec["interactions"][0]["prompt"],
            )
        )
    return runs, LoadStats(records=len(records), runs=len(runs), missing_evaluation=missing)
