from types import SimpleNamespace

from specdx.baselines.api import BASELINE_MODULES, REGISTRY, load_all
from specdx.baselines.b2_single import b2_report
from specdx.baselines.b3_irt import b3_report
from specdx.types import Behaviour, InterpreterId, Scenario, State


def _input():
    def run(key, model, scenario):
        return Behaviour(key, InterpreterId(model, "f", 0, 1), Scenario(scenario, 15, 15, 0),
                         State({}))

    runs = [run("k1", "a", "grn12"), run("k2", "a", "grn12"), run("k3", "b", "grn12"),
            run("k4", "a", "other")]
    violated = {"k1": frozenset({"c1"}), "k2": frozenset({"c1", "c2"}),
                "k3": frozenset({"c2"}), "k4": frozenset({"c3"})}
    return SimpleNamespace(report=SimpleNamespace(behaviours=runs, theta={"a": 0.9, "b": 0.4}),
                           violated=violated, ctx=SimpleNamespace(canon={"grn12": None}))


def test_the_secondary_arms_are_registered_untuned_hidden_only() -> None:
    from specdx.eval.freeze import FIXED, UNTUNED

    load_all()
    assert "secondary" in BASELINE_MODULES
    for name in ("b2_naive", "b3_2pl"):
        b = REGISTRY[name]
        assert (b.reports, b.uses_llm, b.strategies) == (("hidden",), False, (None,))
        assert name in UNTUNED and name in FIXED["untuned"]


def test_the_secondary_arms_are_the_gates_own_b2_and_b3() -> None:
    load_all()
    inp = _input()
    b2 = REGISTRY["b2_naive"].run(inp)
    assert b2.defects == b2_report(inp.report.behaviours, inp.violated, inp.report.theta,
                                   {"grn12"})
    assert b2.defects and all(d.scenarios == ("grn12",) for d in b2.defects)  # canon only
    b3 = REGISTRY["b3_2pl"].run(inp)
    assert b3.defects == b3_report(inp.report.behaviours, inp.violated)
    assert b2.failure is None and b3.failure is None


def test_the_secondary_arms_are_charged_the_runs_they_read() -> None:
    from specdx.eval.study import CONSUMES_RUNS

    assert {"b2_naive", "b3_2pl"} <= CONSUMES_RUNS
