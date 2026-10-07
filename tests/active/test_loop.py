import dataclasses
import json
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

    def run(self, variant, scenario, interp, suffix="", *, kind="population"):
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


class NeverRefute:
    """Every refutation that the probe budget allows fails; past the cap none is attempted."""

    def __init__(self, budget) -> None:
        self.budget = budget

    def refute(self, targets, scenario, behaviours):
        if self.budget.take("probe"):
            return RefuteResult(False, None, 1)
        return RefuteResult(False, None, 0, attempted=False)


def _ctx(runner, localizer=None, lens=None, refuter=None):
    return LoopContext(runner, lens or FakeLens(), CANON, list(CANON.values()), 5,
                       refuter or (lambda interps, budget: AlwaysRefute()), localizer)


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
        VARIANT, _ctx(FakeRunner(_split_plan), lambda v, d, i, b: ("A:01",)),
        SpecDxConfig(MODELS, budget=200, synthesize=False),
    )
    assert any(d.type == "ambiguity" and d.location == ("A:01",) for d in report.defects)


def test_an_unlocalised_ambiguity_keeps_its_type_with_an_empty_location() -> None:
    report = run_specdx(
        VARIANT, _ctx(FakeRunner(_split_plan), lambda v, d, i, b: None),
        SpecDxConfig(MODELS, budget=200, synthesize=False),
    )
    ambiguities = [d for d in report.defects if d.type == "ambiguity"]
    assert ambiguities and all(d.location == () for d in ambiguities)


def test_a_base_noise_ambiguity_is_removed_before_localisation() -> None:
    config = SpecDxConfig(MODELS, budget=200, synthesize=False)
    base = run_specdx(VARIANT, _ctx(FakeRunner(_split_plan)),
                      dataclasses.replace(config, localize=False))
    assert [d for d in base.defects if d.type == "ambiguity"]
    assert base.ambiguity_keys == {("ambiguity", "step", ("x.k",))}
    calls = []

    def localiser(v, d, i, b):
        calls.append(d.note)
        return (f"A:{len(calls):02d}",)  # another run, other sentence ids

    mutant = run_specdx(VARIANT, _ctx(FakeRunner(_split_plan), localiser), config, base=base)
    assert not [d for d in mutant.defects if d.type == "ambiguity"] and calls == []
    alone = run_specdx(VARIANT, _ctx(FakeRunner(_split_plan), localiser), config)
    assert [d.location for d in alone.defects if d.type == "ambiguity"] == [("A:01",)]


class TwoStepLens(FakeLens):
    """The step under test twice: the same split at two steps of one scenario."""

    def views(self, b, violated):
        out = super().views(b, violated)
        return [*out, dataclasses.replace(out[0], item=f"{b.scenario.id}|step2")]


def test_localised_ambiguities_that_share_a_location_are_merged() -> None:
    report = run_specdx(
        VARIANT,
        _ctx(FakeRunner(_split_plan), lambda v, d, i, b: ("A:01",), lens=TwoStepLens()),
        SpecDxConfig(MODELS, budget=200, synthesize=False),
    )
    assert len(report.ambiguity_keys) == 2
    assert [d.location for d in report.defects if d.type == "ambiguity"] == [("A:01",)]


def test_an_unattempted_singleton_is_hidden_and_an_attempted_one_a_contradiction() -> None:
    def run(probe_budget):
        return run_specdx(
            VARIANT, _ctx(FakeRunner(_hidden_plan), refuter=lambda i, b: NeverRefute(b)),
            SpecDxConfig(MODELS, budget=100, synthesize=False, probe_budget=probe_budget),
        )

    capped = run(0)
    top = capped.defects[0]
    assert (top.type, top.location) == ("hidden", ("c1",)) and "budget" in top.note
    assert capped.probe_runs == 0
    attempted = run(10)
    assert (attempted.defects[0].type, attempted.defects[0].location) == (
        "contradiction", ("c1",))
    # one candidate, refuted once per specification however many iterations see it
    assert attempted.probe_runs == 1


def test_the_report_counts_probe_and_ddmin_runs() -> None:
    def localiser(v, d, i, budget):
        for _ in range(3):
            budget.take("ddmin")
        return ("A:01",)

    report = run_specdx(
        VARIANT,
        _ctx(FakeRunner(_split_plan), localiser, refuter=lambda i, b: NeverRefute(b)),
        SpecDxConfig(MODELS, budget=200, synthesize=False, probe_budget=4),
    )
    assert report.ddmin_runs + report.probe_runs <= 4 and report.ddmin_runs >= 1
    out = json.loads(report.to_json())
    assert (out["runs"], out["probe_runs"], out["ddmin_runs"]) == (
        report.runs, report.probe_runs, report.ddmin_runs)
