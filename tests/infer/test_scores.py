from specdx.infer.items import ItemView
from specdx.infer.latent import Observation, fit
from specdx.infer.scores import diagnose
from specdx.types import Behaviour, InterpreterId, RefuteResult, Scenario, State

GOOD = ("g1", "g2", "g3", "g4")
BAD = ("b1", "b2")
S = Scenario("s1", 15, 12, 0)


def _proj(name: str):
    return (("x", ((name, (("v", name),)),)),)


def _diff(a, b):
    return frozenset({"x.v"}) if a != b else frozenset()


class _Refuter:
    def __init__(self, refutes: bool) -> None:
        self.refutes = refutes

    def refute(self, targets, scenario, behaviours):
        return RefuteResult(self.refutes, "w" if self.refutes else None, 1)


def _world(outputs):
    """outputs: model -> list of (klass, violated set) per run, on one item."""
    behaviours, violated, views = [], {}, []
    for model, runs in outputs.items():
        for i, (klass, bad) in enumerate(runs):
            key = f"{model}|{i}"
            behaviours.append(Behaviour(key, InterpreterId(model, model[0], 0, i), S, State({})))
            violated[key] = frozenset(bad)
            views.append(ItemView("s1|step", key, model, model[0], klass, frozenset(bad),
                                  _proj(klass)))
    # Calibration items: competent models agree, each weak model errs in its own way.
    clean = [ItemView(f"c{j}|step", f"{m}|c{j}", m, m[0], "OK" if m in GOOD else f"E{j}-{m}",
                      frozenset(), _proj("OK")) for j in range(6) for m in outputs]
    obs = [Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views + clean]
    return behaviours, violated, views + clean, fit(obs)


def test_a_shared_violation_that_repair_refutes_is_hidden() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {f"n{i}"}) for i in range(3)]
                                                    for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Refuter(True), _diff)
    assert defects[0].type == "hidden" and defects[0].location == ("c",)


def test_the_same_pattern_unrefuted_is_a_contradiction() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {"c", f"n{i}"})
                                                           for i in range(3)] for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Refuter(False), _diff)
    assert defects[0].type == "contradiction" and defects[0].location == ("c",)


def test_competent_models_splitting_cleanly_is_ambiguity() -> None:
    outputs = {m: [("X", set()), ("Y", set()), ("X", set()), ("Y", set())] for m in GOOD}
    outputs |= {m: [(f"E{i}", {f"n{i}"}) for i in range(4)] for m in BAD}
    b, v, views, post = _world(outputs)
    found = [d for d in diagnose(b, v, views, post, _Refuter(True), _diff) if d.type == "ambiguity"]
    assert found and found[0].location == ("x.v",) and found[0].note == "s1|step"


def test_a_clean_deviant_class_beside_a_clean_admissible_one_is_weak() -> None:
    outputs = {m: [("A", set())] * 3 for m in GOOD} | {m: [("W", set())] * 3 for m in BAD}
    b, v, views, post = _world(outputs)
    found = [d for d in diagnose(b, v, views, post, _Refuter(True), _diff) if d.type == "weak"]
    assert found and found[0].location == ("x.v",)


def test_no_weak_when_the_admissible_class_itself_violates() -> None:
    outputs = {m: [("A", {"c"})] * 3 for m in GOOD} | {m: [("W", set())] * 3 for m in BAD}
    b, v, views, post = _world(outputs)
    assert not [d for d in diagnose(b, v, views, post, _Refuter(True), _diff) if d.type == "weak"]
