import pytest

from specdx.infer.items import ItemView
from specdx.infer.latent import ItemPosterior, Observation, fit
from specdx.infer.scores import _weak, diagnose, run_weights
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


def test_run_weights_are_admissibility_times_competence() -> None:
    outputs = {m: [("K", set())] * 3 for m in GOOD} | {m: [(f"E{i}", {f"n{i}"}) for i in range(3)]
                                                    for m in BAD}
    _, _, views, post = _world(outputs)
    weights = run_weights(views, post)
    for key, model in (("g1|0", "g1"), ("b1|0", "b1")):
        mine = [v for v in views if v.run_key == key]
        admissible = sum(post.items[v.item].responsibility[key] for v in mine) / len(mine)
        assert weights[key] == pytest.approx(admissible * post.theta[model])
    assert post.theta["g1"] > post.theta["b1"]


def test_a_shared_violation_that_repair_refutes_is_hidden() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {f"n{i}"}) for i in range(3)]
                                                    for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Refuter(True), _diff)
    found = [d for d in defects if d.type == "hidden"]
    assert found and found[0].location == ("c",)


def test_the_same_pattern_unrefuted_is_a_contradiction() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {"c", f"n{i}"})
                                                           for i in range(3)] for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Refuter(False), _diff)
    found = [d for d in defects if d.type == "contradiction"]
    assert found and found[0].location == ("c",)


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


def test_hidden_score_follows_the_posterior() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {f"n{i}"}) for i in range(3)]
                                                    for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Refuter(True), _diff)
    found = [d for d in defects if d.type == "hidden"]
    assert found
    expected_score = sum(p for s, p in post.items["s1|step"].set_posterior.items() if "K" in s)
    assert abs(found[0].score - expected_score) < 1e-9


class _Unattempted:
    """The probe budget ran out: no refutation was attempted."""

    def refute(self, targets, scenario, behaviours):
        return RefuteResult(False, None, 0, attempted=False)


def test_an_unattempted_singleton_is_hidden_not_a_contradiction() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {"c", f"n{i}"})
                                                           for i in range(3)] for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Unattempted(), _diff)
    assert not [d for d in defects if d.type == "contradiction"]
    found = [d for d in defects if d.type == "hidden"]
    assert found and found[0].location == ("c",) and "budget" in found[0].note


def test_an_unattempted_set_is_no_contradiction() -> None:
    outputs = {m: [("A", {"a"})] * 3 for m in GOOD[:2]} | {m: [("B", {"b"})] * 3
                                                          for m in GOOD[2:]}
    outputs |= {m: [(f"E{i}", {"a", "b"}) for i in range(3)] for m in BAD}
    b, v, views, post = _world(outputs)
    attempted = diagnose(b, v, views, post, _Refuter(False), _diff)
    assert [d.location for d in attempted if d.type == "contradiction"] == [("a", "b")]
    skipped = diagnose(b, v, views, post, _Unattempted(), _diff)
    assert not [d for d in skipped if d.location == ("a", "b")]


def test_candidates_are_refuted_in_descending_score_order(monkeypatch) -> None:
    supports = {"low": 0.96, "high": 0.99}
    monkeypatch.setattr(
        "specdx.infer.scores.soft_hitting_sets",
        lambda sets, tau: [(sets[0][0], supports[next(iter(sets[0][0]))])],
    )
    s2 = Scenario("s2", 15, 13, 0)
    behaviours, violated, views = [], {}, []
    for scenario, clause in ((S, "low"), (s2, "high")):
        for m in GOOD:
            key = f"{scenario.id}|{m}"
            behaviours.append(Behaviour(key, InterpreterId(m, m[0], 0, 1), scenario, State({})))
            violated[key] = frozenset({clause})
            views.append(ItemView(f"{scenario.id}|step", key, m, m[0], "K",
                                  frozenset({clause}), _proj("K")))
    post = fit([Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views])
    order = []

    class _Recording:
        def refute(self, targets, scenario, behaviours):
            order.append(next(iter(targets)))
            return RefuteResult(True, "w", 1)

    diagnose(behaviours, violated, views, post, _Recording(), _diff)
    assert order == ["high", "low"]


def test_weak_tie_break_is_deterministic() -> None:
    views = [
        ItemView("item", f"model|{i}", "model", "m", klass, frozenset(),
                 _proj(klass))
        for klass in ("A", "B", "C") for i in range(3)
    ]
    ip = ItemPosterior(
        classes=("A", "B", "C"),
        set_posterior={frozenset({"A"}): 1.0},
        p_member={"A": 1.0, "B": 0.0, "C": 0.0},
        p_ambiguous=0.0,
        responsibility={},
    )
    result = _weak(ip, views, 2)
    assert result == ("B", 1.0)


def test_weak_union_score() -> None:
    views = [
        ItemView("item", f"model|{i}", "model", "m", klass, frozenset(),
                 _proj(klass))
        for klass in ("A", "B", "C") for i in range(3)
    ]
    ip = ItemPosterior(
        classes=("A", "B", "C"),
        set_posterior={
            frozenset({"A"}): 0.5,
            frozenset({"A", "B"}): 0.3,
            frozenset({"A", "C"}): 0.2,
        },
        p_member={"A": 1.0, "B": 0.0, "C": 0.0},
        p_ambiguous=0.0,
        responsibility={},
    )
    result = _weak(ip, views, 2)
    assert result is not None
    klass, score = result
    assert klass == "C"
    assert score == 1.0
