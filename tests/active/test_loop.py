from pathlib import Path

from specdx.active.loop import LoopContext, SpecDxConfig, run_specdx
from specdx.infer.items import ItemView
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, RefuteResult, Scenario, State
from specdx.variant import SpecVariant

CANON = {"discount8": Scenario("discount8", 15, 15, 8), "grn12": Scenario("grn12", 15, 12, 0),
         "happy": Scenario("happy", 15, 15, 0)}
MODELS = {"qwen": ("q1", "q2"), "llama": ("l1",)}
GOOD = ("q1", "l1")
VARIANT = SpecVariant("toy", Path("."), TellCorpus([]))


class FakeRunner:
    def __init__(self, plan) -> None:
        self.plan, self.executed, self.calls = plan, 0, []

    def run(self, variant, scenario, interp, suffix=""):
        self.executed += 1
        self.calls.append(scenario.id)
        klass, bad = self.plan(scenario, interp)
        return Behaviour(f"{scenario.id}|{interp.key}", interp, scenario,
                         State({}, {"k": klass, "v": ",".join(sorted(bad))}, {}))


class FakeLens:
    """One item under test plus five calibration items, as a real run has eleven steps:
    competence is learnt from the steps where competent models agree."""

    def violations(self, b):
        return frozenset(x for x in b.state.derived["v"].split(",") if x)

    def views(self, b, violated):
        k = b.state.derived["k"]
        model, family = b.interpreter.model, b.interpreter.family
        out = [ItemView(f"{b.scenario.id}|step", b.run_key, model, family, k, violated,
                        (("x", ((k, ()),)),))]
        for j in range(5):
            cal = "OK" if model in GOOD else f"E{j}-{b.run_key}"
            out.append(ItemView(f"{b.scenario.id}|cal{j}", b.run_key, model, family, cal,
                                frozenset(), (("x", ((cal, ()),)),)))
        return out

    def diff(self, a, b):
        return frozenset({"x.k"}) if a != b else frozenset()


class AlwaysRefute:
    def refute(self, targets, scenario, behaviours):
        return RefuteResult(True, "w", 1)


def _hidden_plan(scenario, interp):
    good = interp.model in GOOD
    if scenario.id == "grn12":
        return ("K", {"c1"}) if good else (f"E{interp.seed}", {f"n{interp.seed}"})
    return ("OK", set()) if good else (f"E{interp.seed}-{scenario.id}", {"noise"})


def _ctx(runner, localizer=None):
    return LoopContext(runner, FakeLens(), CANON, list(CANON.values()), 5,
                       lambda interps: AlwaysRefute(), localizer)


def test_a_planted_hidden_requirement_is_reported_first() -> None:
    report = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)),
                        SpecDxConfig(MODELS, budget=100, synthesize=False))
    top = report.defects[0]
    assert (top.type, top.location) == ("hidden", ("c1",)) and "grn12" in top.scenarios
    assert report.theta["q1"] > report.theta["q2"]


def test_an_agreeing_pilot_on_known_ground_does_not_escalate() -> None:
    runner = FakeRunner(_hidden_plan)
    run_specdx(VARIANT, _ctx(runner), SpecDxConfig(MODELS, budget=100, synthesize=False))
    assert runner.calls.count("discount8") == 9 and runner.calls.count("grn12") == 9
    assert runner.calls.count("happy") == 2


def test_the_budget_is_respected() -> None:
    runner = FakeRunner(_hidden_plan)
    report = run_specdx(VARIANT, _ctx(runner), SpecDxConfig(MODELS, budget=5, synthesize=False))
    assert report.runs <= 5 and runner.executed <= 5


def _split_plan(scenario, interp):
    good = interp.model in GOOD
    if scenario.id == "grn12" and good:
        # Two competent models read the same text two ways, so the pilot already disagrees.
        return ("X" if interp.model == "q1" else "Y", set())
    return ("OK", set()) if good else (f"E{interp.seed}-{scenario.id}", {"n"})


def test_ambiguity_locations_are_replaced_by_the_localiser() -> None:
    report = run_specdx(
        VARIANT, _ctx(FakeRunner(_split_plan), lambda v, d, i: ("A:01",)),
        SpecDxConfig(MODELS, budget=200, synthesize=False),
    )
    assert any(d.type == "ambiguity" and d.location == ("A:01",) for d in report.defects)
