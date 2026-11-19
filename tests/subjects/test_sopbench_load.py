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
from specdx.subjects.sopbench.trajectories import (
    TrajectoryError,
    TrajectoryName,
    parse_name,
)


def _name() -> TrajectoryName:
    return TrajectoryName(
        path="output/library/x.json", domain="library", user=None, model="gpt-4o-mini",
        canonical="gpt-4o-mini", mode="fc", tool_list="full", size=1,
    )


def _task_fields(task: S3Task) -> dict:
    return {"user_prompt": task.user_prompt, "user_goal": task.user_goal,
            "action_should_succeed": task.should_succeed, "user_known": task.user_known,
            "constraint_parameters": task.constraint_parameters}


def _record(task: S3Task, evaluations: list[dict], **task_over) -> dict:
    return {
        "domain": task.domain,
        "setup": {},
        "task": _task_fields(task) | task_over,
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


def test_a_run_is_keyed_by_the_canonical_model_name(tmp_path):
    # Ruling I-1: a fireworks file is the same weights as the -instruct model.
    tasks = load_tasks("library")[:1]
    name = parse_name("output/library/ast_llama3.1-70b-it-fireworks-mode_fc-dep_full-"
                      "fmt_structured-tool_full-shuffle_False.json", 1)
    (run,), _ = load_runs(_write(tmp_path, [_record(tasks[0], [_eval()])]), name, "meta",
                          tasks)
    assert run.model == "llama3.1-70b-instruct"
    assert run.run_key == "llama3.1-70b-instruct|library|0"


def test_load_tasks_keeps_the_alignment_fields():
    raw = json.loads((SOPBENCH_DIR / "data" / "university_tasks.json").read_text())
    first = next(iter(raw.values()))[0]
    t = load_tasks("university")[0]
    assert t.user_known == first["user_known"]
    assert t.constraint_parameters == first["constraint_parameters"]


def test_a_shift_inside_a_block_of_identical_prompts_raises_mismatch(tmp_path):
    # Ruling I-5: university tasks 0-6 share one user_prompt; only task 0 should succeed.
    tasks = load_tasks("university")[:7]
    assert len({t.user_prompt for t in tasks}) == 1
    assert [t.should_succeed for t in tasks] == [True] + [False] * 6
    recs = [_record(t, [_eval()]) for t in tasks]
    recs[0], recs[1] = recs[1], recs[0]
    with pytest.raises(TrajectoryMismatch):
        load_runs(_write(tmp_path, recs), _name(), "openai", tasks)


@pytest.mark.parametrize("field", ["action_should_succeed", "user_known",
                                   "constraint_parameters", "user_goal"])
def test_each_alignment_field_is_compared(tmp_path, field):
    tasks = load_tasks("library")[:2]
    recs = [_record(tasks[0], [_eval()]), _record(tasks[1], [_eval()])]
    recs[1]["task"][field] = {"action_should_succeed": not tasks[1].should_succeed,
                              "user_known": {"other": 1},
                              "constraint_parameters": {"other": 1},
                              "user_goal": "other_goal"}[field]
    with pytest.raises(TrajectoryMismatch, match="record 1"):
        load_runs(_write(tmp_path, recs), _name(), "openai", tasks)
