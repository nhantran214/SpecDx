import json
from pathlib import Path

import httpx
import psycopg
import pytest

from specdx.active.loop import LoopContext
from specdx.eval.matching import MIN_SCORABLE, Prf, gate_verdict
from specdx.eval.pilot import (
    GateConfig,
    check_model,
    delta_shift,
    gate_markdown,
    load_population,
    run_gate,
)
from specdx.infer.act import Act
from specdx.infer.items import ItemView
from specdx.mutate.operators import GroundTruth, Mutant
from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InfrastructureError, RefuteResult, Scenario, State, Target
from specdx.variant import SpecVariant


def test_capability_needs_a_declared_and_a_real_tool_call() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        call = {"function": {"name": "echo", "arguments": {"text": "ok"}}}
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "",
                                                     "tool_calls": [call]}})

    cap = check_model("m", base_url="http://x", transport=httpx.MockTransport(handle))
    assert cap.usable


def test_a_model_without_the_tools_capability_is_not_usable() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": ["completion"]})

    cap = check_model("m", base_url="http://x", transport=httpx.MockTransport(handle))
    assert not cap.declares_tools and not cap.usable


def test_a_model_ollama_does_not_have_is_an_error_not_unusable() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'missing:7b' not found"})

    with pytest.raises(httpx.HTTPStatusError):
        check_model("missing:7b", base_url="http://x", transport=httpx.MockTransport(handle))


def test_a_failing_chat_call_is_an_error_too() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        return httpx.Response(500, json={"error": "llama runner process has terminated"})

    with pytest.raises(httpx.HTTPStatusError):
        check_model("m", base_url="http://x", transport=httpx.MockTransport(handle))


CANON = {"grn12": Scenario("grn12", 15, 12, 0)}


class _Runner:
    def __init__(self, broken: bool) -> None:
        self.broken, self.executed = broken, 0

    def run(self, variant, scenario, interp, suffix="", *, kind="population"):
        self.executed += 1
        key = f"{variant.name}|{interp.key}"
        if variant.name == "outage":
            raise InfrastructureError("Ollama failed 5 times: ConnectError")
        if self.broken:
            return Behaviour(key, interp, scenario, State.empty(), status="error", error="bad SQL")
        bad = "c1" if variant.name.startswith("H1-") else ""
        return Behaviour(key, interp, scenario, State({}, {"v": bad}, {}))


STEP = WorkStep("s", "de", "A", "i", ("x",))


class _Lens:
    """Every behaviour acts at the one step "s" and touches its targets."""

    relevant = {"s": frozenset({"c1"})}
    columns = None

    def acted_steps(self, b):
        return [STEP] if b.status == "ok" else []

    def violations(self, b):
        return frozenset(x for x in b.state.derived.get("v", "").split(",") if x)

    def views(self, b, violated):
        if b.status != "ok":
            return []
        return [ItemView(f"{b.scenario.id}|s", b.run_key, b.interpreter.model,
                         b.interpreter.family, "K", violated, ())]

    def acts(self, b):
        return [Act(v.item, v.run_key, v.model, True) for v in self.views(b, frozenset())]

    def diff(self, a, b):
        return frozenset()


class _Refute:
    def refute(self, targets, scenario, behaviours):
        return RefuteResult(True, "w", 1)


class _Bench:
    base = SpecVariant("base", Path("."), TellCorpus([]))

    def loop_context(self, variant, config):
        return LoopContext(_Runner(variant.name == "broken"), _Lens(), CANON,
                           list(CANON.values()), 5, lambda i, b: _Refute())


def _mutant(mid, kind, location, step="s"):
    return Mutant(mid, mid[:2], "s", SpecVariant(mid, Path("."), TellCorpus([])),
                  GroundTruth(mid, kind, frozenset(location), "clause", (Target("step", step),)))


def _config(tmp_path) -> GateConfig:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"models_by_family": {"q": ["m1"], "l": ["m2"]},
                                "variants": [0], "seeds": [1, 2], "budget": 10,
                                "dev_per_type": 1, "rng_seed": 1}), encoding="utf-8")
    return GateConfig.load(path)


def test_the_gate_configuration_carries_both_run_budgets(tmp_path) -> None:
    assert (_config(tmp_path).specdx().probe_budget,
            _config(tmp_path).specdx().localize_budget) == (100, 80)
    gate = GateConfig.load(Path(__file__).parents[2] / "experiments" / "p1-gate.json")
    assert (gate.specdx().probe_budget, gate.specdx().localize_budget) == (100, 80)


def test_gate_records_a_broken_mutant_and_continues(tmp_path) -> None:
    mutants = [_mutant("H1-00", "hidden", {"c1"}), _mutant("broken", "weak", {"x.y"})]
    result = run_gate(_config(tmp_path), _Bench(), mutants)
    assert "broken" in result.failed and "bad SQL" in result.failed["broken"]
    assert result.scores["specdx"]["hidden"].tp == 1
    assert "| hidden |" in gate_markdown(result)


def test_the_report_shows_the_verdict_and_the_literal_comparison(tmp_path) -> None:
    result = run_gate(_config(tmp_path), _Bench(), [_mutant("H1-00", "hidden", {"c1"})])
    report = gate_markdown(result)
    assert "| > baselines' untyped F1 |" in report
    assert "| ≥ 0.40 (baselines cannot locate it) |" in report
    assert "typed against typed" in report
    assert set(json.loads(result.to_json())["literal_wins"]) == set(result.wins)


def test_every_mutant_and_the_base_are_written_one_per_line(tmp_path) -> None:
    out = tmp_path / "results" / "p1-gate-mutants.jsonl"
    mutants = [_mutant("H1-00", "hidden", {"c1"}), _mutant("broken", "weak", {"x.y"})]
    run_gate(_config(tmp_path), _Bench(), mutants, mutants_out=out)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [x.get("mutant") for x in lines] == [None, "H1-00", "broken"]
    base, hit, broken = lines
    assert base["base"] is True and set(base["defects"]) == {"specdx", "b2", "b3"}
    assert hit["base"] is False and hit["type"] == "hidden" and hit["failure"] is None
    assert hit["truth"] == {"location": ["c1"], "kind": "clause"}
    assert hit["matches"]["specdx"]["typed"] == [["hidden", ["c1"]]]
    assert set(hit["matches"]["b2"]) == {"typed", "untyped"}
    assert hit["defects"]["specdx"][0]["location"] == ["c1"]
    assert set(hit["runs"]) == {"population", "probe", "ddmin", "executed"}
    assert hit["ambiguity_keys"] == [] and hit["ambiguity_dropped"] == []
    assert base["ambiguity_keys"] == []
    assert "bad SQL" in broken["failure"] and broken["matches"] is None


def test_a_failed_mutant_makes_the_gate_incomplete(tmp_path) -> None:
    mutants = [_mutant("H1-00", "hidden", {"c1"}), _mutant("broken", "weak", {"x.y"})]
    result = run_gate(_config(tmp_path), _Bench(), mutants)
    assert not result.passed and result.headline == "INCOMPLETE"
    report = gate_markdown(result)
    assert report.startswith("# P1b gate v2: INCOMPLETE")
    assert "PASSED" not in report and "- `broken`: bad SQL" in report
    assert json.loads(result.to_json())["headline"] == "INCOMPLETE"


def test_an_infrastructure_failure_stops_the_gate(tmp_path) -> None:
    mutants = [_mutant("H1-00", "hidden", {"c1"}), _mutant("outage", "weak", {"x.y"})]
    with pytest.raises(InfrastructureError, match="ConnectError"):
        run_gate(_config(tmp_path), _Bench(), mutants)


@pytest.mark.parametrize("error", [
    psycopg.OperationalError("database offline"),
    psycopg.InterfaceError("connection closed"),
    httpx.ConnectError("connection refused"),
])
def test_database_and_transport_failures_stop_the_gate(tmp_path, error) -> None:
    class _Down(_Bench):
        def loop_context(self, variant, config):
            if variant.name == "down":
                raise error
            return super().loop_context(variant, config)

    with pytest.raises(type(error)):
        run_gate(_config(tmp_path), _Down(), [_mutant("down", "weak", {"x.y"})])


def test_scorability_comes_from_the_base_runs_and_only_scorable_mutants_count(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return gate_verdict(*args, **kwargs)

    monkeypatch.setattr("specdx.eval.pilot.gate_verdict", recorder)
    reached = _mutant("H1-00", "hidden", {"c1"})
    unreached = _mutant("H1-01", "hidden", {"c1"}, step="elsewhere")
    out = tmp_path / "lines.jsonl"
    result = run_gate(_config(tmp_path), _Bench(), [reached, unreached], mutants_out=out)
    assert result.touches == {"H1-00": 4, "H1-01": 0}  # 2 models x 2 seeds on grn12
    assert result.scorable == {"H1-00"}
    assert result.primary["specdx"]["hidden"].tp == 1
    assert result.primary["specdx"]["hidden"].fn == 0  # H1-01 is excluded, not missed
    assert result.scores["specdx"]["hidden"].tp == 2  # the secondary table keeps both
    # The verdict is decided on the primary tables with the scorable counts (spec v2 §4.2).
    # B2 also finds c1 here, so the scores alone would not show the count rule at work.
    (args, kwargs), = calls
    assert args == (result.primary, result.primary_untyped)
    assert kwargs == {"scorable": {"contradiction": 0, "hidden": 1, "ambiguity": 0, "weak": 0},
                      "min_scorable": MIN_SCORABLE}
    assert not result.wins["hidden"]
    base, *mutants = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert base["touches"] == {"H1-00": 4, "H1-01": 0}
    assert base["theta"] and base["kappa"] is not None and base["unobservable"] == []
    assert set(base["items"]) == {"grn12|s"}
    data = json.loads(result.to_json())
    assert set(data["primary"]) == {"typed", "untyped"} and data["touches"] == base["touches"]
    assert data["scorable"] == ["H1-00"]
    assert data["scorable_per_type"]["hidden"] == 1 and data["mutants_per_type"]["hidden"] == 2
    assert [(x["mutant"], x["scorable"]) for x in mutants] == [("H1-00", True), ("H1-01", False)]
    assert all(x["unobservable"] == [] and x["delta_shift"] == {"grn12|s": 0.0} for x in mutants)
    report = gate_markdown(result)
    assert "| hidden | 1 of 2 |" in report and "| H1-01 | hidden | 0 | no |" in report


def test_a_type_with_fewer_than_three_scorable_mutants_is_a_loss() -> None:
    perfect = {t: Prf(tp=3) for t in ("contradiction", "hidden", "ambiguity", "weak")}
    none = {t: Prf(fn=3) for t in perfect}
    typed = {"specdx": perfect, "b2": none, "b3": none}
    passed, wins = gate_verdict(typed, typed)
    assert passed and all(wins.values())
    counts = {"contradiction": 3, "hidden": 2, "ambiguity": 3, "weak": 0}
    passed, wins = gate_verdict(typed, typed, scorable=counts)
    assert not passed and wins == {"contradiction": True, "hidden": False, "ambiguity": True,
                                   "weak": False}


def _gate_file(tmp_path, population, screen=None) -> GateConfig:
    raw = {"models_by_family": population, "variants": [0], "seeds": [1, 2], "budget": 10,
           "dev_per_type": 1, "rng_seed": 1}
    if screen is not None:
        raw["screen"] = str(screen)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return GateConfig.load(path)


def test_the_gate_runs_only_on_the_screened_population(tmp_path) -> None:
    screen = tmp_path / "screen.json"
    screen.write_text(json.dumps({"population": {"q": ["m1"], "l": ["m2"]}, "stop": False}),
                      encoding="utf-8")
    run_gate(_gate_file(tmp_path, {"q": ["m1"], "l": ["m2"]}, screen), _Bench(), [])
    with pytest.raises(ValueError, match="differs from the population"):
        run_gate(_gate_file(tmp_path, {"q": ["m1"]}, screen), _Bench(), [])
    with pytest.raises(ValueError, match="empty"):
        run_gate(_gate_file(tmp_path, {}), _Bench(), [])
    screen.write_text(json.dumps({"population": {"q": ["m1"]}, "stop": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 2 families"):
        load_population(screen)


def test_delta_shift_is_mutant_minus_base_over_the_shared_items() -> None:
    class _R:
        def __init__(self, items):
            self.items = {k: {"delta": v} for k, v in items.items()}

    base = _R({"a": 1.0, "b": 0.5, "only-base": 2.0})
    mutant = _R({"a": 1.5, "b": 0.25, "only-mutant": 3.0})
    assert delta_shift(base, mutant) == {"a": 0.5, "b": -0.25}
