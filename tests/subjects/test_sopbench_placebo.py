"""Permutation placebo for the ability gradient (P4a Task 8), on tiny synthetic domains."""

import dataclasses
import math
from collections import Counter

import pytest

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


# --- Ruling P-C (final fix wave) and the Task 8 Minors ---

def test_the_count_is_distinct_cells_not_findings(monkeypatch):
    # P-C: a changeable cell with two defect findings (an ambiguity and its hidden) counts once.
    from specdx.infer.typing import CellFinding
    from specdx.subjects.sopbench import placebo as module

    runs = _gradient_runs(6)
    cell = f"s3:{DOMAIN}:0:db_match"
    monkeypatch.setattr(module, "type_cells_v4", lambda cells, fit, by, p: [
        CellFinding(cell, "hidden", 1.0), CellFinding(cell, "ambiguity", 1.0)])
    monkeypatch.setattr(module, "type_cells", lambda cells, fit, by, p: [
        CellFinding(cell, "hidden", 1.0), CellFinding(cell, "ambiguity", 1.0),
        CellFinding(f"s3:{DOMAIN}:3:db_match", "weak", 1.0)])
    res = placebo(DOMAIN, _tasks(6), runs, MODELS, V4Params(), V3Params(), seeds=range(1))
    assert (res.real_v4, res.real_v3) == (1, 2)
    assert (res.placebo_v4, res.placebo_v3) == ([1], [2])


def test_tool_error_runs_keep_their_own_act_vectors():
    # P-C: the permutation runs among a task's runs without a tool error only.
    runs = [_run(t, m, ok=(k % 2 == 0), db=(k % 3 != 0), err=(k in (1, 4)))
            for t in range(4) for k, m in enumerate(MODELS)]
    for seed in range(10):
        out = permute_within_tasks(runs, seed)
        for a, b in zip(runs, out, strict=True):
            if a.tool_error:
                assert _vector(a) == _vector(b)
        for t in range(4):
            clean = [r for r in runs if r.task == t and not r.tool_error]
            moved = [r for r in out if r.task == t and not r.tool_error]
            assert Counter(map(_vector, clean)) == Counter(map(_vector, moved))
            assert all(not r.tool_error for r in moved)


def test_the_permutation_is_keyed_by_domain_and_ignores_the_input_order():
    # P-C: the generator is keyed by (seed, crc32(domain), task), and a task's runs are taken
    # in run_key order, so the shuffle does not depend on how the runs were listed.
    runs = _gradient_runs(9)
    out = permute_within_tasks(runs, seed=4)
    backwards = permute_within_tasks(runs[::-1], seed=4)
    assert {r.run_key: _vector(r) for r in out} == {r.run_key: _vector(r) for r in backwards}
    other = [dataclasses.replace(r, domain="dmv") for r in runs]
    moved = permute_within_tasks(other, seed=4)
    assert [_vector(r) for r in moved] != [_vector(r) for r in out]


def test_different_seeds_give_different_permutations():
    runs = _gradient_runs(9)
    shuffles = {tuple(_vector(r) for r in permute_within_tasks(runs, s)) for s in range(5)}
    assert len(shuffles) == 5


def test_a_task_with_a_single_clean_run_is_left_alone():
    runs = [_run(0, "m0", db=False)] + [_run(0, m, err=True) for m in list(MODELS)[1:]]
    runs += [_run(1, m, db=(k < 4)) for k, m in enumerate(MODELS)]
    out = permute_within_tasks(runs, seed=2)
    assert [_vector(r) for r in out if r.task == 0] == [_vector(r) for r in runs if r.task == 0]


def test_placebo_validates_its_domain_and_tasks():
    runs = _gradient_runs(3)
    for bad_runs, tasks in (
            ([*runs, dataclasses.replace(runs[0], domain="dmv")], _tasks(3)),
            (runs, [*_tasks(3)[:2], S3Task("dmv", 2, "goal2", True, "prompt 2")]),
            (runs, _tasks(2))):
        with pytest.raises(ValueError):
            placebo(DOMAIN, tasks, bad_runs, MODELS, V4Params(), V3Params(), seeds=range(1))


def test_placebo_checks_the_changeable_set_is_kept(monkeypatch):
    from specdx.subjects.sopbench import placebo as module

    def broken(runs, seed):
        return [dataclasses.replace(r, db_match=True) for r in runs]

    monkeypatch.setattr(module, "permute_within_tasks", broken)
    with pytest.raises(AssertionError, match="changeable"):
        placebo(DOMAIN, _tasks(9), _gradient_runs(9), MODELS, V4Params(), V3Params(),
                seeds=range(1))


def test_the_real_cells_are_the_planted_ones():
    # Task 8 Minor: the real v4 count is pinned to the planted cells, not just positive.
    from specdx.infer.anchored_v4 import fit_v4
    from specdx.infer.typing_v4 import type_cells_v4

    runs = _gradient_runs()
    cd = cells_for(_tasks(30), runs)
    fit = fit_v4(cd.data.cells, cd.data.readings, models=MODELS, params=V4Params())
    flagged = {f.cell for f in type_cells_v4(cd.data.cells, fit, cd.data.by_cell(), V4Params())
               if f.type == "hidden"}
    assert flagged and flagged <= {f"s3:{DOMAIN}:{t}:db_match" for t in range(0, 30, 3)}


def test_without_a_gradient_v4_finds_nothing_and_v3_still_does():
    # Task 8 Minor: the no-gradient control. On every 3rd task half the models deviate, but
    # which half does not follow ability.
    runs = []
    for t in range(30):
        for k, m in enumerate(MODELS):
            if t % 3 == 0:
                runs.append(_run(t, m, db=(k % 2 == 0)))
            else:
                slip = (t * 7 + k * 3) % 8 < max(0, k - 2)
                runs.append(_run(t, m, con=not slip))
    res = placebo(DOMAIN, _tasks(30), runs, MODELS, V4Params(), V3Params(), seeds=range(2))
    assert res.real_v4 == 0 and res.real_v3 > 0


def _cut_runs(cut: int) -> list[S3Run]:
    """_gradient_runs with the models from rank `cut` on deviating at the defect tasks."""
    runs = []
    for t in range(30):
        for k, m in enumerate(MODELS):
            if t % 3 == 0:
                runs.append(_run(t, m, db=k >= cut))
            else:
                slip = (t * 7 + k * 3) % 8 < max(0, k - 2)
                runs.append(_run(t, m, con=not slip))
    return runs


def test_intermediate_gradients():
    # Task 8 Minor: intermediate controls. With only the two strongest of eight deviating the
    # gradient is too thin to decide; with six of eight it fires and vanishes under placebo.
    thin = placebo(DOMAIN, _tasks(30), _cut_runs(2), MODELS, V4Params(), V3Params(),
                   seeds=range(1))
    assert thin.real_v4 == 0
    wide = placebo(DOMAIN, _tasks(30), _cut_runs(6), MODELS, V4Params(), V3Params(),
                   seeds=range(3))
    assert wide.real_v4 > 0 and wide.ratio_v4 <= 0.2


def test_a_given_real_fit_is_not_refitted(monkeypatch):
    # Task 10 Minor 6: the stage passes its diagnosis's v4 fit, so the real data is fitted once.
    from specdx.infer.anchored_v4 import fit_v4
    from specdx.subjects.sopbench import placebo as module

    runs = _gradient_runs(9)
    cd = cells_for(_tasks(9), runs)
    real = fit_v4(cd.data.cells, cd.data.readings, models=MODELS, params=V4Params())
    calls = []

    def counted(*a, **k):
        calls.append(1)
        return fit_v4(*a, **k)

    monkeypatch.setattr(module, "fit_v4", counted)
    given = placebo(DOMAIN, _tasks(9), runs, MODELS, V4Params(), V3Params(), seeds=range(2),
                    real_fit=real)
    assert len(calls) == 2
    calls.clear()
    fresh = placebo(DOMAIN, _tasks(9), runs, MODELS, V4Params(), V3Params(), seeds=range(2))
    assert len(calls) == 3
    fields = ("real_v4", "real_v3", "placebo_v4", "placebo_v3", "changeable")
    assert [getattr(fresh, f) for f in fields] == [getattr(given, f) for f in fields]
