import dataclasses
import json
from pathlib import Path

from specdx.active.loop import LoopContext, SpecDxConfig, oracle_interpreters, run_specdx
from specdx.infer.act import Act
from specdx.infer.items import ItemView
from specdx.infer.scores import UNATTEMPTED
from specdx.scenarios.tguided import TLiteral
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, Defect, RefuteResult, Scenario, State
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

    def run_batch(self, requests):
        return [self.run(r.variant, r.scenario, r.interp, r.suffix, kind=r.kind) for r in requests]


CLAUSES = frozenset({"c1", "n", "noise", "n1", "n2", "n3"})


class FakeLens:
    """One item under test plus five calibration items, as a real run has eleven steps:
    competence is learnt from the steps where competent models agree. Every behaviour acts
    at every item, and every clause is read at the steps under test."""

    relevant = {"step": CLAUSES, "step2": CLAUSES} | {f"cal{j}": frozenset() for j in range(5)}

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

    def acts(self, b):
        return [Act(v.item, v.run_key, v.model, v.acted) for v in self.views(b, frozenset())]

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


def test_the_base_reports_locations_are_refuted_last_on_a_mutant(monkeypatch) -> None:
    config = SpecDxConfig(MODELS, budget=100, synthesize=False)
    base = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)), config)
    assert ("c1",) in {d.location for d in base.defects}
    assert isinstance(base.spent, frozenset) and ("c1",) not in base.spent
    # A singleton no class violates alone is dropped below τ_H, and the report keeps it as
    # spent: the probe it cost must not be paid again on every mutant.
    def _shared_plan(scenario, interp):
        if scenario.id == "grn12" and interp.model in GOOD:
            return "K", {"c1", "n"}
        return _hidden_plan(scenario, interp)

    dropped = run_specdx(VARIANT, _ctx(FakeRunner(_shared_plan)), config)
    assert not [d for d in dropped.defects if d.type == "hidden"]
    assert dropped.spent == {("c1",), ("n",)}
    assert json.loads(dropped.to_json())["spent"] == [["c1"], ["n"]]  # a reloaded base keeps it
    seen = []
    real = run_specdx.__globals__["diagnose"]

    def recording(*args, known=frozenset(), **kwargs):
        seen.append(known)
        return real(*args, known=known, **kwargs)

    monkeypatch.setitem(run_specdx.__globals__, "diagnose", recording)
    run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)), config, base=base)
    assert seen and all(("c1",) in known for known in seen)
    # Symbolic repair spent no probe on the base, so a mutant needing one probes it early;
    # ambiguity and weak locations are no refutation candidates at all.
    seen.clear()
    noted = dataclasses.replace(base, defects=[
        Defect("contradiction", ("a",), ("s",), (), 1.0),
        Defect("hidden", ("b",), ("s",), (), 1.0, UNATTEMPTED),
        Defect("hidden", ("c",), ("s",), (), 1.0, "refuted by symbolic-repair"),
        Defect("hidden", ("d",), ("s",), (), 1.0, "refuted by grn12|q1|v0|s1"),
        Defect("ambiguity", ("e",), ("s",), (), 1.0),
        Defect("weak", ("f",), ("s",), (), 1.0),
    ], spent=frozenset({("g",)}))  # a singleton dropped below τ_H after it cost a probe
    run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)), config, base=noted)
    assert seen and all(known == {("a",), ("b",), ("d",), ("g",)} for known in seen)


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


def test_unlocalised_ambiguities_at_different_steps_stay_apart() -> None:
    report = run_specdx(
        VARIANT, _ctx(FakeRunner(_split_plan), lambda v, d, i, b: None, lens=TwoStepLens()),
        SpecDxConfig(MODELS, budget=200, synthesize=False),
    )
    unlocalised = [d for d in report.defects if d.type == "ambiguity"]
    assert [d.location for d in unlocalised] == [(), ()]
    assert sorted(d.note.split("|")[1] for d in unlocalised) == ["step", "step2"]


def test_the_report_records_the_ambiguities_the_differential_dropped() -> None:
    config = SpecDxConfig(MODELS, budget=200, synthesize=False)
    base = run_specdx(VARIANT, _ctx(FakeRunner(_split_plan)),
                      dataclasses.replace(config, localize=False))
    mutant = run_specdx(VARIANT, _ctx(FakeRunner(_split_plan), lens=TwoStepLens()), config,
                        base=base)
    assert mutant.ambiguity_keys == {("ambiguity", "step", ("x.k",)),
                                     ("ambiguity", "step2", ("x.k",))}
    assert mutant.ambiguity_dropped == {("ambiguity", "step", ("x.k",))}


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


def _ddmin_localiser(calls, runs=3):
    """Spends `runs` ddmin runs per ambiguity, like the real one; unlocalised if cut short."""
    def localiser(v, d, i, budget):
        calls.append(d.note)
        for _ in range(runs):
            if not budget.take("ddmin"):
                return None
        return ("A:01",)
    return localiser


def _split_and_shared_plan(scenario, interp):
    """grn12: the competent models split two ways and all violate c1, so the scenario has
    both an ambiguity to localise and a candidate to refute."""
    klass, bad = _split_plan(scenario, interp)
    return (klass, {"c1"}) if scenario.id == "grn12" and interp.model in GOOD else (klass, bad)


def test_the_report_counts_probe_and_ddmin_runs() -> None:
    report = run_specdx(
        VARIANT,
        _ctx(FakeRunner(_split_and_shared_plan), _ddmin_localiser([]),
             refuter=lambda i, b: NeverRefute(b)),
        SpecDxConfig(MODELS, budget=200, synthesize=False, probe_budget=4, localize_budget=5),
    )
    assert (report.probe_runs, report.ddmin_runs) == (1, 3)
    out = json.loads(report.to_json())
    assert (out["runs"], out["probe_runs"], out["ddmin_runs"]) == (
        report.runs, report.probe_runs, report.ddmin_runs)


def test_refutation_spending_its_whole_budget_leaves_localisation_its_own() -> None:
    calls: list[str] = []
    report = run_specdx(
        VARIANT,
        _ctx(FakeRunner(_split_and_shared_plan), _ddmin_localiser(calls),
             refuter=lambda i, b: NeverRefute(b)),
        SpecDxConfig(MODELS, budget=200, synthesize=False, probe_budget=1, localize_budget=80),
    )
    assert report.probe_runs == 1  # the probe budget is spent
    assert calls and report.ddmin_runs == 3
    assert [d.location for d in report.defects if d.type == "ambiguity"] == [("A:01",)]


def test_the_localisation_budget_caps_ddmin_in_descending_score_order() -> None:
    def run(localize_budget):
        calls: list[str] = []
        report = run_specdx(
            VARIANT,
            _ctx(FakeRunner(_split_plan), _ddmin_localiser(calls), lens=TwoStepLens()),
            SpecDxConfig(MODELS, budget=200, synthesize=False, localize_budget=localize_budget),
        )
        found = sorted(d.location for d in report.defects if d.type == "ambiguity")
        return report, calls, found

    full, calls, found = run(6)
    assert full.ddmin_runs == 6 and len(calls) == 2 and found == [("A:01",)]
    cut, calls, found = run(4)  # the second localisation runs out after one run
    assert cut.ddmin_runs == 4 and len(calls) == 2 and found == [(), ("A:01",)]
    spent, calls, found = run(3)  # nothing left for the second: it is not even tried
    assert spent.ddmin_runs == 3 and len(calls) == 1 and found == [(), ("A:01",)]
    score = {d.location: d.score for d in spent.defects if d.type == "ambiguity"}
    assert score[("A:01",)] >= score[()]  # the better supported one went first


def test_t_boundary_scenarios_run_after_the_canon_ones_and_before_the_bandit() -> None:
    runner = FakeRunner(_hidden_plan)
    t1, t2 = Scenario("o15-r12-d7-g1", 15, 12, 7), Scenario("o15-r12-d9-g1", 15, 12, 9)
    extra = Scenario("o16-r14-d4-g2", 16, 14, 4, 2)  # a cell no queued scenario tried
    ctx = dataclasses.replace(_ctx(runner), candidates=[*CANON.values(), t1, extra],
                              tboundary=[t1, t2])
    report = run_specdx(VARIANT, ctx, SpecDxConfig(MODELS, budget=200))
    assert report.scenarios[:6] == ["discount8", "grn12", "happy", t1.id, t2.id, extra.id]


def test_a_queued_scenario_whose_signature_was_visited_is_skipped() -> None:
    runner = FakeRunner(_hidden_plan)
    twin = Scenario("twin", 15, 12, 0)  # grn12's parameters under another id
    ctx = dataclasses.replace(_ctx(runner), tboundary=[twin])
    config = SpecDxConfig(MODELS, budget=100, synthesize=False)
    report = run_specdx(VARIANT, ctx, config)
    assert report.scenarios == ["discount8", "grn12", "happy"] and "twin" not in runner.calls
    plain = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)), config)
    assert report.runs == plain.runs  # the skipped twin spent no population budget


def test_the_report_carries_stage_one_the_signals_and_the_literals() -> None:
    literal = TLiteral("discount_pct", 5, "%", "tối đa", "KB", "KB:KB-0003", "KB:KB-0003:03", 9)
    ctx = dataclasses.replace(_ctx(FakeRunner(_hidden_plan)), literals=[literal])
    report = run_specdx(VARIANT, ctx, SpecDxConfig(MODELS, budget=100, synthesize=False))
    assert set(report.kappa) == {"q1", "q2", "l1"}
    assert report.items
    item = report.items["grn12|step"]
    assert item["observable"] and {"delta", "observability", "acted", "s1"} <= set(item)
    out = json.loads(report.to_json())
    assert out["literals"] == [literal.as_dict()] and out["unobservable"] == []
    assert set(out["items"]) == set(report.items) and out["kappa"] == report.kappa
    # the family effect the pre-registration promises: one value per family, in the report
    assert out["family_effect"] == report.family_effect
    assert set(report.family_effect) == {"qwen", "llama"}


class IdleLens(FakeLens):
    """No one acts at the step under test: it is never observable."""

    def views(self, b, violated):
        out = super().views(b, violated)
        return [dataclasses.replace(v, acted=not v.item.endswith("|step")) for v in out]


def test_an_item_nobody_acts_at_is_unobservable_and_types_nothing() -> None:
    report = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan), lens=IdleLens()),
                        SpecDxConfig(MODELS, budget=100, synthesize=False))
    assert not [d for d in report.defects if "c1" in d.location]
    assert report.items["grn12|step"]["observable"] is False


def test_the_diagnosis_unobservable_list_reaches_the_report_and_its_json(monkeypatch) -> None:
    real = run_specdx.__globals__["diagnose"]
    entry = Defect("unobservable", ("c1",), ("grn12",), (), 1.0, "no observable item reads it")

    def wrapped(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), unobservable=[entry])

    monkeypatch.setitem(run_specdx.__globals__, "diagnose", wrapped)
    report = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)),
                        SpecDxConfig(MODELS, budget=100, synthesize=False))
    assert report.unobservable == [entry]
    [row] = json.loads(report.to_json())["unobservable"]
    assert (row["type"], row["location"], row["scenarios"]) == ("unobservable", ["c1"], ["grn12"])


def test_the_ddmin_oracle_reruns_each_competent_model_under_each_variant() -> None:
    seen = []

    def localiser(v, d, interps, budget):
        seen.append(list(interps))
        return ("A:01",)

    report = run_specdx(VARIANT, _ctx(FakeRunner(_split_plan), localiser),
                        SpecDxConfig(MODELS, variants=(0, 1), budget=400, synthesize=False))
    assert [d.location for d in report.defects if d.type == "ambiguity"] == [("A:01",)]
    competent = [m for m in ("q1", "q2", "l1") if report.theta[m] >= 0.5]
    assert sorted(competent) == ["l1", "q1"]
    assert {(i.model, i.variant, i.seed) for i in seen[0]} == {
        (m, v, 1) for m in competent for v in (0, 1)}


def test_oracle_interpreters_are_the_competent_models_at_seed_one() -> None:
    interps = oracle_interpreters({"a": ("m1", "m2"), "b": ("m3",)},
                                  {"m1": 0.6, "m2": 0.4, "m3": 0.9}, (0, 2),
                                  acted={"m1", "m2", "m3"})
    assert [(i.model, i.family, i.variant, i.seed) for i in interps] == [
        ("m3", "b", 0, 1), ("m3", "b", 2, 1), ("m1", "a", 0, 1), ("m1", "a", 2, 1)]


def test_a_model_that_never_acted_is_not_competent_at_its_prior_mode() -> None:
    interps = oracle_interpreters({"a": ("idle", "acting", "weak")},
                                  {"idle": 0.75, "acting": 0.6, "weak": 0.4}, (0,),
                                  acted={"acting", "weak"})
    assert [i.model for i in interps] == ["acting"]  # θ = 0.75 alone is no evidence


class SilentLens(FakeLens):
    """A population that acts nowhere: every step fails or writes nothing."""

    def views(self, b, violated):
        return [dataclasses.replace(v, acted=False) for v in super().views(b, violated)]


def test_a_population_that_never_acts_types_nothing_and_still_reports() -> None:
    report = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan), lens=SilentLens()),
                        SpecDxConfig(MODELS, budget=100, synthesize=False))
    assert report.defects == [] and report.runs > 0
    assert report.theta == {"q1": 0.75, "q2": 0.75, "l1": 0.75}  # the prior's mode, n_m = 0
    assert not any(row["observable"] for row in report.items.values())
    assert json.loads(report.to_json())["defects"] == []
