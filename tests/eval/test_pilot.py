import json
from pathlib import Path

import httpx

from specdx.active.loop import LoopContext
from specdx.eval.pilot import GateConfig, check_model, gate_markdown, run_gate
from specdx.infer.items import ItemView
from specdx.mutate.operators import GroundTruth, Mutant
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, RefuteResult, Scenario, State
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


CANON = {"grn12": Scenario("grn12", 15, 12, 0)}


class _Runner:
    def __init__(self, broken: bool) -> None:
        self.broken, self.executed = broken, 0

    def run(self, variant, scenario, interp, suffix=""):
        self.executed += 1
        key = f"{variant.name}|{interp.key}"
        if self.broken:
            return Behaviour(key, interp, scenario, State.empty(), status="error", error="bad SQL")
        bad = "c1" if variant.name == "H1-00" else ""
        return Behaviour(key, interp, scenario, State({}, {"v": bad}, {}))


class _Lens:
    def violations(self, b):
        return frozenset(x for x in b.state.derived.get("v", "").split(",") if x)

    def views(self, b, violated):
        if b.status != "ok":
            return []
        return [ItemView(f"{b.scenario.id}|s", b.run_key, b.interpreter.model,
                         b.interpreter.family, "K", violated, ())]

    def diff(self, a, b):
        return frozenset()


class _Refute:
    def refute(self, targets, scenario, behaviours):
        return RefuteResult(True, "w", 1)


class _Bench:
    base = SpecVariant("base", Path("."), TellCorpus([]))

    def loop_context(self, variant, config):
        return LoopContext(_Runner(variant.name == "broken"), _Lens(), CANON,
                           list(CANON.values()), 5, lambda i: _Refute())


def _mutant(mid, kind, location):
    return Mutant(mid, mid[:2], "s", SpecVariant(mid, Path("."), TellCorpus([])),
                  GroundTruth(mid, kind, frozenset(location), "clause"))


def test_gate_records_a_broken_mutant_and_continues(tmp_path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"models_by_family": {"q": ["m1"], "l": ["m2"]},
                                "variants": [0], "seeds": [1], "budget": 10,
                                "dev_per_type": 1, "rng_seed": 1}), encoding="utf-8")
    mutants = [_mutant("H1-00", "hidden", {"c1"}), _mutant("broken", "weak", {"x.y"})]
    result = run_gate(GateConfig.load(path), _Bench(), mutants)
    assert "broken" in result.failed and "bad SQL" in result.failed["broken"]
    assert result.scores["specdx"]["hidden"].tp == 1
    assert "| hidden |" in gate_markdown(result)
