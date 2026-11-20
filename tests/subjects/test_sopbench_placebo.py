"""Permutation placebo for the ability gradient (P4a Task 8), on tiny synthetic domains."""

import math
from collections import Counter

from specdx.infer.anchored import V3Params
from specdx.infer.anchored_v4 import V4Params
from specdx.subjects.sopbench.cells import cells_for
from specdx.subjects.sopbench.load import S3Run, S3Task
from specdx.subjects.sopbench.placebo import (
    PlaceboResult,
    changeable_cells,
    permute_within_tasks,
    placebo,
)

DOMAIN = "bank"
# model -> family; the first models are the strongest
MODELS = {"m0": "a", "m1": "a", "m2": "b", "m3": "b", "m4": "c", "m5": "c", "m6": "d",
          "m7": "d"}


def _run(task: int, model: str, ok: bool = True, db: bool = True, con: bool = True,
         proc: bool = True, err: bool = False) -> S3Run:
    return S3Run(DOMAIN, task, model, MODELS[model], "fc", ok, db, con, proc, err,
                 f"prompt {task}")


def _tasks(n: int) -> list[S3Task]:
    return [S3Task(DOMAIN, i, f"goal{i}", True, f"prompt {i}") for i in range(n)]


def _vector(r: S3Run) -> tuple:
    return (r.succeeded, r.db_match, r.constraints, r.process, r.tool_error)


def _gradient_runs(n_tasks: int = 30) -> list[S3Run]:
    """Every 3rd task is a defect: the stronger half of the models deviates from C there
    (db_match violated) and the weaker half agrees with C. On the other tasks, weaker
    models slip on the constraints check more often, which makes ability identifiable."""
    runs = []
    for t in range(n_tasks):
        for k, m in enumerate(MODELS):
            if t % 3 == 0:
                runs.append(_run(t, m, db=k >= 4))
            else:
                slip = (t * 7 + k * 3) % 8 < max(0, k - 2)
                runs.append(_run(t, m, con=not slip))
    return runs


def test_permutation_keeps_each_task_multiset():
    runs = _gradient_runs(6)
    out = permute_within_tasks(runs, seed=3)
    assert len(out) == len(runs)
    for a, b in zip(runs, out, strict=True):
        assert (a.domain, a.task, a.model, a.family, a.mode, a.prompt) == (
            b.domain, b.task, b.model, b.family, b.mode, b.prompt)
    for t in range(6):
        before = Counter(_vector(r) for r in runs if r.task == t)
        after = Counter(_vector(r) for r in out if r.task == t)
        assert before == after
    assert [_vector(r) for r in out] != [_vector(r) for r in runs]
    assert out == permute_within_tasks(runs, seed=3)


def test_unanimous_tasks_are_not_changeable():
    runs = [_run(0, m) for m in MODELS] + [_run(1, m, db=(m != "m0")) for m in MODELS]
    cd = cells_for(_tasks(2), runs)
    ch = changeable_cells(cd)
    assert ch == frozenset({f"s3:{DOMAIN}:1:db_match"})
    placed = cells_for(_tasks(2), permute_within_tasks(runs, seed=1))
    assert changeable_cells(placed) == ch


def test_tool_error_readings_are_excluded_from_changeability():
    runs = [_run(0, m, db=(m != "m0"), err=(m == "m0")) for m in MODELS]
    assert changeable_cells(cells_for(_tasks(1), runs)) == frozenset()


def test_placebo_destroys_a_planted_gradient():
    runs = _gradient_runs()
    res = placebo(DOMAIN, _tasks(30), runs, MODELS, V4Params(), V3Params(), seeds=range(5))
    assert isinstance(res, PlaceboResult)
    assert res.real_v4 > 0 and res.changeable > 0
    assert len(res.placebo_v4) == len(res.placebo_v3) == 5
    assert res.ratio_v4 <= 0.3
    assert res.ratio_v3 >= 0.8  # v3 is majority-driven: the placebo leaves it standing


def test_ratio_is_nan_when_the_real_count_is_zero():
    runs = [_run(t, m) for t in range(4) for m in MODELS]
    res = placebo(DOMAIN, _tasks(4), runs, MODELS, V4Params(), V3Params(), seeds=range(2))
    assert res.real_v4 == 0 and math.isnan(res.ratio_v4)
    assert res.real_v3 == 0 and math.isnan(res.ratio_v3)
