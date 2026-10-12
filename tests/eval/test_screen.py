import json
from pathlib import Path

import pytest

from specdx.eval.pilot import load_population
from specdx.eval.screen import (
    ScreenResult,
    choose_population,
    load_profile,
    run_screen,
    screen_markdown,
)
from specdx.infer.act import Act
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, Scenario, State
from specdx.variant import SpecVariant

BASE = SpecVariant("base", Path("."), TellCorpus([]))
CANON = {"happy": Scenario("happy", 15, 15, 0), "grn12": Scenario("grn12", 15, 12, 0)}
# model -> (acted steps of 9, tool calls, valid calls) for each of its runs
PLAN = {"good": (7, 10, 9), "edge": (6, 10, 8), "sloppy": (9, 10, 7), "idle": (2, 4, 4)}


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int, str]] = []

    def model_digest(self, model: str) -> str:
        if model == "absent":
            raise KeyError(f"Ollama does not list model {model!r}")
        return f"d-{model}"

    def run(self, variant, scenario, interp, suffix="", *, kind="population"):
        self.calls.append((variant.name, scenario.id, interp.variant, interp.seed, kind))
        acted, calls, valid = PLAN[interp.model]
        counts = {"nudges": 0, "parsed_from_text": 0, "tool_calls": calls, "valid_calls": valid}
        return Behaviour(f"{interp.key}", interp, scenario, State({}, {"acted": str(acted)}),
                         harness={"step": counts})

    def run_batch(self, requests):
        return [self.run(r.variant, r.scenario, r.interp, r.suffix, kind=r.kind) for r in requests]


class _Lens:
    def acts(self, b):
        n = int(b.state.derived["acted"])
        return [Act(f"s|{i}", b.run_key, b.interpreter.model, i < n) for i in range(9)]


def test_each_candidate_runs_nine_times_on_happy_and_is_measured() -> None:
    runner = _Runner()
    screen = run_screen(runner, BASE, _Lens(), CANON, {"a": ("good", "edge"), "b": ("sloppy",),
                                                       "c": ("idle", "absent")})
    results = {r.model: r for r in screen.results}
    assert sorted({c[:2] for c in runner.calls}) == [("base", "happy")]
    assert {c[4] for c in runner.calls} == {"screen"} and len(runner.calls) == 36
    assert results["good"].runs == 9 and results["good"].act_rate == pytest.approx(7 / 9)
    assert results["good"].valid_rate == pytest.approx(0.9)
    assert results["edge"].admitted  # 6/9 and 0.8 exactly: the thresholds are inclusive
    assert not results["sloppy"].admitted  # acts, but 0.7 of its calls are valid
    assert not results["idle"].admitted
    assert results["absent"].note == "not installed" and not results["absent"].admitted
    assert screen.population == {"a": ("good", "edge")} and screen.stop


def _r(model, family, act, valid=1.0):
    return ScreenResult(model, family, 9, act, valid)


def test_the_population_keeps_two_per_family_and_four_in_all_in_rank_order() -> None:
    results = [_r("q1", "qwen", 0.9), _r("q2", "qwen", 0.95), _r("q3", "qwen", 0.99),
               _r("o1", "openai", 0.8), _r("m1", "meta", 0.7, 0.9), _r("m2", "meta", 0.7, 0.95),
               _r("g1", "ibm", 0.65)]
    assert choose_population(results) == {"qwen": ("q3", "q2"), "openai": ("o1",),
                                          "meta": ("m2",)}


def test_ties_break_by_valid_call_rate_then_name() -> None:
    results = [_r("b", "x", 0.7), _r("a", "y", 0.7), _r("c", "z", 0.7, 0.9)]
    assert list(choose_population(results)) == ["y", "x", "z"]


def test_the_screen_json_is_what_the_gate_freezes(tmp_path) -> None:
    runner = _Runner()
    screen = run_screen(runner, BASE, _Lens(), CANON, {"a": ("good",), "b": ("edge",)})
    path = tmp_path / "screen.json"
    path.write_text(screen.to_json(), encoding="utf-8")
    assert not screen.stop and load_population(path) == {"a": ("good",), "b": ("edge",)}
    assert json.loads(path.read_text())["rules"]["act_rate_min"] == 0.6
    assert "| good | a | 9 |" in screen_markdown(screen, "test")


def test_a_profile_loads_by_path(tmp_path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"profile": "p", "workers": 2, "ollama_num_parallel": 2,
                                "candidates": {"qwen": ["qwen3:14b"]}}), encoding="utf-8")
    profile = load_profile(str(path))
    assert (profile.name, profile.workers, profile.candidates) == ("p", 2, {"qwen": ("qwen3:14b",)})


def test_the_shipped_profiles_widen_each_other() -> None:
    profiles = [load_profile(name) for name in ("gpu16", "gpu24", "gpu48", "gpu80")]

    def models(p):
        return {(f, m) for f, ms in p.candidates.items() for m in ms}

    assert all(models(a) < models(b) for a, b in zip(profiles, profiles[1:], strict=False))
    assert set(profiles[0].candidates) == {"qwen", "openai", "mistral", "ibm", "meta"}
    assert all(p.workers <= p.ollama_num_parallel for p in profiles)
