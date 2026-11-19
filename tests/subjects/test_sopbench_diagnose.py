"""Offline per-domain diagnosis of SOPBench runs (P3a Task 5). Every scenario runs the
anchored model at the default V3Params (ruling F2): the asserted types are what the model
really does there. The S3CellData come from hand-built tasks and runs through cells_for."""

import pytest

from specdx.subjects.sopbench.cells import S3CellData, cell_id, cells_for
from specdx.subjects.sopbench.diagnose import (
    COMPOSITIONS,
    PROVISIONAL,
    S3Diagnosis,
    compose,
    diagnose_domain,
)
from specdx.subjects.sopbench.load import S3Run, S3Task

DOMAIN = "library"
# Six raters from three families.
RATERS = {
    "gpt-4o": "openai", "gpt-4o-mini": "openai",
    "gemini-2.5-pro": "google", "gemini-2.5-flash": "google",
    "qwen2.5-7b-instruct": "qwen", "qwen2.5-72b-instruct": "qwen",
}


def _task(index: int, should_succeed: bool = True) -> S3Task:
    return S3Task(DOMAIN, index, f"goal{index}", should_succeed, f"prompt {index}")


def _run(task: int, model: str, ok: bool = True, db: bool = True, con: bool = True,
         proc: bool = True) -> S3Run:
    return S3Run(DOMAIN, task, model, RATERS[model], "fc", ok, db, con, proc, False,
                 f"prompt {task}")


def _agreeing(tasks: list[S3Task]) -> list[S3Run]:
    """Every rater reads every task's anchor, with every check satisfied."""
    return [_run(t.index, m, ok=t.should_succeed) for t in tasks for m in RATERS]


def _diagnose(tasks: list[S3Task], runs: list[S3Run]) -> tuple[S3CellData, S3Diagnosis]:
    cd = cells_for(tasks, runs)
    return cd, diagnose_domain(DOMAIN, cd, {r.model: r.family for r in runs})


def test_agreeing_raters_give_no_defect_and_a_converged_fit():
    tasks = [_task(i, should_succeed=i % 2 == 0) for i in range(4)]
    _, dx = _diagnose(tasks, _agreeing(tasks))
    assert dx.domain == DOMAIN
    assert dx.defects == []
    assert dx.fit.converged
    assert dx.raters == tuple(sorted(RATERS))
    assert PROVISIONAL == ("provisional: V3Params calibrated on CASE-0031, "
                           "recalibrated for S3 in P3b")


def test_raters_who_all_do_a_forbidden_action_flag_the_outcome_cell_as_hidden():
    forbidden = _task(0, should_succeed=False)
    others = [_task(i) for i in range(1, 4)]
    runs = [_run(0, m, ok=True) for m in RATERS] + _agreeing(others)
    _, dx = _diagnose([forbidden, *others], runs)
    outcome = cell_id(DOMAIN, 0, "outcome")
    hidden = [d for d in dx.defects if d.key == ("hidden", (outcome,))]
    assert len(hidden) == 1
    d = hidden[0]
    assert d.note == "refutation not attempted (offline)"
    assert d.scenarios == (f"{DOMAIN}:0",)
    assert d.witnesses == tuple(sorted(f"{m}|{DOMAIN}|0" for m in RATERS))
    assert d.score >= 0.5 and d.subtype == "mis-key"
    assert all(x.location == (outcome,) for x in dx.defects)


def test_a_split_population_flags_ambiguity():
    split = _task(0, should_succeed=True)
    # Strengthened under ruling F2: with few agreeing tasks the fit explains the split as
    # the refusers' lower accuracy (no ambiguity at 3 to 11 tasks; a weak check at 3). The
    # raters must first show their accuracy elsewhere; 20 tasks clear the transition at 12.
    others = [_task(i) for i in range(1, 21)]
    # One rater per family succeeds and the other refuses.
    refusers = {"gpt-4o-mini", "gemini-2.5-flash", "qwen2.5-7b-instruct"}
    runs = [_run(0, m, ok=m not in refusers) for m in RATERS] + _agreeing(others)
    _, dx = _diagnose([split, *others], runs)
    outcome = cell_id(DOMAIN, 0, "outcome")
    amb = [d for d in dx.defects if d.key == ("ambiguity", (outcome,))]
    assert len(amb) == 1
    assert amb[0].note == ""
    assert amb[0].witnesses == tuple(sorted(f"{m}|{DOMAIN}|0" for m in RATERS))
    # On an anchored cell the ambiguity brings its hidden requirement (typing.co_occur).
    pick = [d for d in dx.defects if d.key == ("hidden", (outcome,))]
    assert [(d.subtype, d.note) for d in pick] == [
        ("ambiguous-pick", "refutation not attempted (offline)")]


def test_two_checks_violated_together_by_most_runs_give_a_contradiction():
    task = _task(0)
    models = sorted(RATERS)
    # Four runs violate db_match and constraints together; one violates each alone, so
    # neither check alone hits 95% of the mass, and the pair hits all of it.
    runs = [_run(0, m, db=False, con=False) for m in models[:4]]
    runs += [_run(0, models[4], db=False), _run(0, models[5], con=False)]
    others = [_task(i) for i in range(1, 4)]
    _, dx = _diagnose([task, *others], runs + _agreeing(others))
    pair = tuple(sorted((cell_id(DOMAIN, 0, "db_match"), cell_id(DOMAIN, 0, "constraints"))))
    found = [d for d in dx.defects if d.key == ("contradiction", pair)]
    assert len(found) == 1
    assert found[0].scenarios == (f"{DOMAIN}:0",)
    assert found[0].score == pytest.approx(1.0)
    # No executor refutes it offline (ruling I2), and every run of the task meets the pair.
    assert found[0].note == "refutation not attempted (offline)"
    assert found[0].witnesses == tuple(sorted(f"{m}|{DOMAIN}|0" for m in RATERS))
    scores = [d.score for d in dx.defects]
    assert scores == sorted(scores, reverse=True)


def test_clean_runs_dilute_the_hitting_mass_so_disjoint_single_violations_give_none():
    task = _task(0)
    models = sorted(RATERS)
    # Four clean runs and two runs that each violate a different single check: as in
    # diagnose_v3, every run of the task enters the hitting sets (ruling I1), so the pair
    # hits a third of the mass, not all of it.
    runs = [_run(0, m) for m in models[:4]]
    runs += [_run(0, models[4], db=False), _run(0, models[5], con=False)]
    others = [_task(i) for i in range(1, 4)]
    _, dx = _diagnose([task, *others], runs + _agreeing(others))
    assert not [d for d in dx.defects if d.type == "contradiction"]


def test_compose_splits_open_and_hosted_raters():
    runs = [_run(0, m) for m in RATERS]
    assert COMPOSITIONS == ("all", "open", "hosted")
    assert compose(runs, "all") == runs
    assert {r.model for r in compose(runs, "open")} == {
        "qwen2.5-7b-instruct", "qwen2.5-72b-instruct"}
    assert {r.model for r in compose(runs, "hosted")} == {
        "gpt-4o", "gpt-4o-mini", "gemini-2.5-pro", "gemini-2.5-flash"}
    with pytest.raises(ValueError, match="composition"):
        compose(runs, "closed")
