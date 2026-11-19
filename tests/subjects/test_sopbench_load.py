import json
from pathlib import Path

import pytest

from specdx.subjects.sopbench import SOPBENCH_DIR
from specdx.subjects.sopbench.load import (
    S3Task,
    TrajectoryMismatch,
    load_runs,
    load_tasks,
)
from specdx.subjects.sopbench.trajectories import TrajectoryError, TrajectoryName


def _name() -> TrajectoryName:
    return TrajectoryName(
        path="output/library/x.json", domain="library", user=None, model="gpt-4o-mini",
        canonical="gpt-4o-mini", mode="fc", tool_list="full", size=1,
    )


def _record(task: S3Task, evaluations: list[dict]) -> dict:
    return {
        "domain": task.domain,
        "setup": {},
        "task": {"user_prompt": task.user_prompt},
        "interactions": [{"prompt": f"P{task.index}", "interaction": [], "database": {}}],
        "evaluations": evaluations,
    }


def _eval(**over) -> dict:
    base = {
        "action_successfully_called": True, "database_match": True,
        "constraint_not_violated": True, "dirgraph_satisfied": True,
        "no_tool_call_error": True,
    }
    return base | over


def _write(tmp_path: Path, payload) -> Path:
    p = tmp_path / "t.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return p


def test_load_tasks_counts_and_orders_bank_like_the_file():
    raw = json.loads((SOPBENCH_DIR / "data" / "bank_tasks.json").read_text())
    tasks = load_tasks("bank")
    assert len(tasks) == 134
    assert tasks[0].user_goal == next(iter(raw))
    assert [t.index for t in tasks] == list(range(134))
    assert all(t.domain == "bank" for t in tasks)


def test_load_runs_maps_evaluation_fields(tmp_path):
    tasks = load_tasks("library")[:2]
    bad = _eval(
        action_successfully_called=False, database_match=False,
        constraint_not_violated=False, dirgraph_satisfied=False, no_tool_call_error=False,
    )
    recs = [_record(tasks[0], [_eval()]), _record(tasks[1], [bad])]
    runs, stats = load_runs(_write(tmp_path, recs), _name(), "openai", tasks)
    assert (stats.records, stats.runs, stats.missing_evaluation) == (2, 2, 0)
    a, b = runs
    assert (a.succeeded, a.db_match, a.constraints, a.process, a.tool_error) == (
        True, True, True, True, False)
    assert (b.succeeded, b.db_match, b.constraints, b.process, b.tool_error) == (
        False, False, False, False, True)
    assert a.run_key == "gpt-4o-mini|library|0"
    assert (a.model, a.family, a.mode, a.prompt, a.task) == (
        "gpt-4o-mini", "openai", "fc", "P0", 0)
    assert b.run_key == "gpt-4o-mini|library|1"


def test_a_record_without_evaluations_is_skipped_and_counted(tmp_path):
    tasks = load_tasks("library")[:2]
    recs = [_record(tasks[0], []), _record(tasks[1], [_eval()])]
    runs, stats = load_runs(_write(tmp_path, recs), _name(), "openai", tasks)
    assert [r.task for r in runs] == [1]
    assert (stats.records, stats.runs, stats.missing_evaluation) == (2, 1, 1)


def test_a_shorter_file_or_a_shifted_prompt_raises_mismatch(tmp_path):
    tasks = load_tasks("library")[:2]
    short = _write(tmp_path, [_record(tasks[0], [_eval()])])
    with pytest.raises(TrajectoryMismatch):
        load_runs(short, _name(), "openai", tasks)
    shifted = _write(tmp_path, [_record(tasks[1], [_eval()]), _record(tasks[0], [_eval()])])
    with pytest.raises(TrajectoryMismatch):
        load_runs(shifted, _name(), "openai", tasks)


def test_an_empty_file_raises_trajectory_error(tmp_path):
    tasks = load_tasks("library")[:2]
    for body in ("", "{}", "not json"):
        with pytest.raises(TrajectoryError):
            load_runs(_write(tmp_path, body), _name(), "openai", tasks)
