import pytest

from specdx.infer.items import ItemView, observations
from specdx.infer.latent import ItemPosterior, fit
from specdx.infer.scores import (
    UNATTEMPTED,
    UNOBSERVABLE,
    _weak,
    diagnose,
    merge_defects,
    run_weights,
)
from specdx.types import Behaviour, Defect, InterpreterId, RefuteResult, Scenario, State

GOOD = ("g1", "g2", "g3", "g4")
BAD = ("b1", "b2")
S = Scenario("s1", 15, 12, 0)
CLAUSES = frozenset({"a", "b", "c", "x", "n0", "n1", "n2", "n3", "noise", "low", "high"})
RELEVANT = {"step": CLAUSES}


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
    return behaviours, violated, views + clean, fit(observations(views + clean))


def _diagnose(b, v, views, post, refuter, *, observable=None, permutations=200, **kwargs):
    seen = frozenset(x.item for x in views) if observable is None else observable
    return diagnose(b, v, views, post, refuter, _diff, observable=seen, relevant=RELEVANT,
                    permutations=permutations, **kwargs)


def _shared(bad_clause_too: bool = False):
    """Competent models all violate c; weak models scatter."""
    return {m: [("K", {"c"})] * 3 for m in GOOD} | {
        m: [(f"E{i}", {"c", f"n{i}"} if bad_clause_too else {f"n{i}"}) for i in range(3)]
        for m in BAD}


def test_run_weights_are_membership_times_competence_over_acted_items() -> None:
    _, _, views, post = _world(_shared())
    idle = [ItemView("s1|step", "idle|0", "g1", "g", "K", frozenset(), _proj("K"), acted=False)]
    weights = run_weights(views + idle, post)
    for key, model in (("g1|0", "g1"), ("b1|0", "b1")):
        mine = [v for v in views if v.run_key == key]
        expected = sum(post.items[v.item].p_member.get(v.klass, 0.0) for v in mine) / len(mine)
        assert weights[key] == pytest.approx(expected * post.theta[model])
    assert weights["idle|0"] == 0.0
    assert post.theta["g1"] > post.theta["b1"]


def test_a_shared_violation_that_repair_refutes_is_hidden() -> None:
    b, v, views, post = _world(_shared())
    found = [d for d in _diagnose(b, v, views, post, _Refuter(True)).defects
             if d.type == "hidden"]
    assert found and found[0].location == ("c",)


def test_the_same_pattern_unrefuted_is_a_contradiction() -> None:
    b, v, views, post = _world(_shared(bad_clause_too=True))
    found = [d for d in _diagnose(b, v, views, post, _Refuter(False)).defects
             if d.type == "contradiction"]
    assert found and found[0].location == ("c",)


def test_competent_models_splitting_cleanly_is_ambiguity() -> None:
    outputs = {m: [("X", set()), ("Y", set()), ("X", set()), ("Y", set())] for m in GOOD}
    outputs |= {m: [(f"E{i}", {f"n{i}"}) for i in range(4)] for m in BAD}
    b, v, views, post = _world(outputs)
    diagnosis = _diagnose(b, v, views, post, _Refuter(True))
    found = [d for d in diagnosis.defects if d.type == "ambiguity"]
    assert found and found[0].location == ("x.v",) and found[0].note == "s1|step"
    assert diagnosis.signals["s1|step"].s1 >= 0.5


def test_a_clean_deviant_class_beside_a_clean_admissible_one_is_weak() -> None:
    outputs = {m: [("A", set())] * 3 for m in GOOD} | {m: [("W", set())] * 3 for m in BAD}
    b, v, views, post = _world(outputs)
    found = [d for d in _diagnose(b, v, views, post, _Refuter(True)).defects if d.type == "weak"]
    assert found and found[0].location == ("x.v",)


def test_no_weak_when_the_admissible_class_itself_violates() -> None:
    outputs = {m: [("A", {"c"})] * 3 for m in GOOD} | {m: [("W", set())] * 3 for m in BAD}
    b, v, views, post = _world(outputs)
    assert not [d for d in _diagnose(b, v, views, post, _Refuter(True)).defects
                if d.type == "weak"]


def test_hidden_score_follows_the_posterior() -> None:
    b, v, views, post = _world(_shared())
    found = [d for d in _diagnose(b, v, views, post, _Refuter(True)).defects
             if d.type == "hidden"]
    assert found
    expected_score = sum(p for s, p in post.items["s1|step"].set_posterior.items() if "K" in s)
    assert abs(found[0].score - expected_score) < 1e-9


def test_a_singleton_no_admissible_class_violates_alone_is_not_hidden() -> None:
    # Every competent run violates c together with x: Hidden(c) is 0, below τ_H.
    outputs = {m: [("K", {"c", "x"})] * 3 for m in GOOD} | {
        m: [(f"E{i}", {"c", f"n{i}"}) for i in range(3)] for m in BAD}
    b, v, views, post = _world(outputs)
    assert not [d for d in _diagnose(b, v, views, post, _Refuter(True)).defects
                if d.type == "hidden"]


class _MustNotRefute:
    def refute(self, targets, scenario, behaviours):
        raise AssertionError("an unobservable candidate must not be refuted")


def test_a_candidate_no_observable_item_reads_is_unobservable_and_never_refuted() -> None:
    b, v, views, post = _world(_shared())
    diagnosis = _diagnose(b, v, views, post, _MustNotRefute(), observable=frozenset())
    assert diagnosis.defects == [] and diagnosis.signals == {}
    assert [(d.type, d.location) for d in diagnosis.unobservable] == [(UNOBSERVABLE, ("c",))]


def _variant_world():
    """One competent model reads variant 2 another way; everyone else agrees."""
    behaviours, violated, views = [], {}, []
    for m in GOOD:
        for variant in (0, 1, 2):
            for seed in (1, 2, 3):
                key = f"{m}|{variant}|{seed}"
                klass = "Y" if m == "g1" and variant == 2 else "X"
                behaviours.append(Behaviour(key, InterpreterId(m, m[0], variant, seed), S,
                                            State({})))
                violated[key] = frozenset()
                views.append(ItemView("s1|step", key, m, m[0], klass, frozenset(), _proj(klass),
                                      variant=variant))
    clean = [ItemView(f"c{j}|step", f"{m}|c{j}", m, m[0], "OK", frozenset(), _proj("OK"))
             for j in range(6) for m in GOOD]
    return behaviours, violated, views + clean, fit(observations(views + clean))


def test_paraphrase_sensitivity_reports_an_ambiguity_the_set_signal_misses() -> None:
    b, v, views, post = _variant_world()
    diagnosis = _diagnose(b, v, views, post, _Refuter(True), permutations=2000)
    signal = diagnosis.signals["s1|step"]
    assert signal.s1 < 0.5 and signal.significant and signal.p is not None
    found = [d for d in diagnosis.defects if d.type == "ambiguity"]
    assert found and found[0].location == ("x.v",)
    assert found[0].score == pytest.approx(1 - signal.p)


class _Unattempted:
    """The probe budget ran out: no refutation was attempted."""

    def refute(self, targets, scenario, behaviours):
        return RefuteResult(False, None, 0, attempted=False)


def test_an_unattempted_singleton_is_hidden_not_a_contradiction() -> None:
    b, v, views, post = _world(_shared(bad_clause_too=True))
    defects = _diagnose(b, v, views, post, _Unattempted()).defects
    assert not [d for d in defects if d.type == "contradiction"]
    found = [d for d in defects if d.type == "hidden"]
    assert found and found[0].location == ("c",) and "budget" in found[0].note


def test_an_unattempted_set_is_no_contradiction() -> None:
    outputs = {m: [("A", {"a"})] * 3 for m in GOOD[:2]} | {m: [("B", {"b"})] * 3
                                                          for m in GOOD[2:]}
    outputs |= {m: [(f"E{i}", {"a", "b"}) for i in range(3)] for m in BAD}
    b, v, views, post = _world(outputs)
    attempted = _diagnose(b, v, views, post, _Refuter(False)).defects
    assert [d.location for d in attempted if d.type == "contradiction"] == [("a", "b")]
    skipped = _diagnose(b, v, views, post, _Unattempted()).defects
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
    post = fit(observations(views))
    order = []

    class _Recording:
        def refute(self, targets, scenario, behaviours):
            order.append(next(iter(targets)))
            return RefuteResult(True, "w", 1)

    _diagnose(behaviours, violated, views, post, _Recording())
    assert order == ["high", "low"]
    # On a mutant, a location the base report already holds goes last, whatever its score
    order.clear()
    _diagnose(behaviours, violated, views, post, _Recording(), known=frozenset({("high",)}))
    assert order == ["low", "high"]


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


def test_a_merged_hidden_finding_keeps_an_attempted_refutation_note() -> None:
    unattempted = Defect("hidden", ("c",), ("s1",), ("r1",), 0.9, UNATTEMPTED)
    refuted = Defect("hidden", ("c",), ("s2",), ("r2",), 0.5, "refuted by w")
    for pair in ((unattempted, refuted), (refuted, unattempted)):
        (merged,) = merge_defects(pair)
        assert merged.note == "refuted by w" and merged.score == 0.9
        assert merged.scenarios == ("s1", "s2")
