import pytest

from specdx.baselines.mus_c import explore, minimal_hitting_sets


def test_a_planted_pair_conflict_is_found() -> None:
    # state "s0" violates a; repairing a gives "s1", which violates b; repairing b gives s0
    viol = {"s0": frozenset({"a"}), "s1": frozenset({"b"})}
    nxt = {("s0", "a"): ["s1"], ("s1", "b"): ["s0"]}
    sets = explore("s0", viol.__getitem__, lambda s, c: nxt.get((s, c), []), key=str)
    assert minimal_hitting_sets(sets) == [("a", "b")]


def test_a_consistent_specification_has_no_mus() -> None:
    sets = explore("ok", lambda s: frozenset(), lambda s, c: [], key=str)
    assert minimal_hitting_sets(sets) == []


def test_exploration_respects_the_evaluation_budget() -> None:
    count = {"n": 0}

    def viol(s):
        count["n"] += 1
        return frozenset({"a"})

    explore(0, viol, lambda s, c: [s + 1, s + 2], key=str, max_evals=10)
    assert count["n"] <= 10


@pytest.mark.db
def test_b5_runs_on_the_base_specification(db, pack) -> None:
    from types import SimpleNamespace

    from specdx.baselines.api import REGISTRY
    from specdx.orchestrate.work_orders import load_work_orders
    from specdx.spec.clauses import ClauseSet

    clauses = ClauseSet.from_pack(pack, load_work_orders())
    out = REGISTRY["b5_mus"].run(SimpleNamespace(clauses=clauses, golden=db.golden_state()))
    assert out.failure is None
    assert all(d.type == "contradiction" for d in out.defects)
