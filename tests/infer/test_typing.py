"""The cell-level typing core (spec v3.1 §5.1, §5.3, §5.4), on hand-built posteriors."""

import pytest

from specdx.active.refute import SYMBOLIC_WITNESS
from specdx.eval.matching import location_matches
from specdx.infer.anchored import AnchoredFit, CellPosterior, V3Params
from specdx.infer.cells import Cell, CellData, Reading
from specdx.infer.scores import UNATTEMPTED
from specdx.infer.typing import (
    AMBIGUOUS_PICK,
    MIS_KEY,
    UNSTATED,
    ambiguity_finding,
    diagnose_v3,
    hidden_finding,
    refutation_scenario,
    type_cells,
    weak_finding,
)
from specdx.mutate.operators import GroundTruth
from specdx.types import Behaviour, InterpreterId, RefuteResult, Scenario, State

P = V3Params(pi=0.1, s_mis=0.5, s_amb=0.25, s_empty=0.25, beta_sd=1.0,
             mu_alpha=0.0, sigma_alpha=1.5, tau_h=0.5, tau_a=0.5, tau_w=0.8)
ANCHORED = Cell("c", "value", "d", "golden:t:K:c", "t.c", "t.c", ("step",))
FREE = Cell("u", "value", None, None, "t.u", "t.u", ("step",))
RULE = Cell("inv:x", "constraint", "satisfied", "inv:x", "inv:x", "inv:x", ("step",))
F = frozenset


def _post(cell: Cell, sets: dict) -> CellPosterior:
    values = tuple(sorted({v for s in sets for v in s}))
    return CellPosterior(cell.id, cell.anchor, values, sets, 0.0, {})


def _fit(alpha: dict[str, float], *posts: CellPosterior) -> AnchoredFit:
    return AnchoredFit(alpha, {}, {}, {p.cell: p for p in posts}, {}, True, 1, P)


def _r(cell: str, model: str, value: str, i: int, clean: bool = False) -> Reading:
    return Reading(cell, f"{model}|{i}", model, model, "s", value, clean)


def test_a_hidden_candidate_needs_an_anchor_and_tau_h() -> None:
    post = _post(ANCHORED, {F({"d"}): 0.3, F({"a"}): 0.6, F(): 0.1})
    readings = [_r("c", "m1", "a", 0), _r("c", "m2", "a", 1), _r("c", "m2", "d", 2)]
    found = hidden_finding(ANCHORED, post, readings, 0.5)
    assert found is not None and found.score == pytest.approx(0.7)
    assert found.subtype == MIS_KEY and found.witnesses == ("m1|0", "m2|1")
    assert hidden_finding(ANCHORED, post, readings, 0.75) is None
    free = _post(FREE, {F({"a"}): 1.0})
    assert hidden_finding(FREE, free, readings, 0.5) is None


def test_the_subtype_follows_the_most_probable_defect_form_even_for_one_model() -> None:
    # Theorem T2′: with the flat error pool one model separates the forms, so the unstated
    # and ambiguous-pick subtypes need no second model.
    one = [_r("c", "m1", "a", 0), _r("c", "m1", "b", 1)]
    silent = _post(ANCHORED, {F({"d"}): 0.1, F({"a"}): 0.2, F(): 0.7})
    assert hidden_finding(ANCHORED, silent, one, 0.5).subtype == UNSTATED
    split = _post(ANCHORED, {F({"d"}): 0.1, F({"d", "a"}): 0.6, F(): 0.3})
    assert hidden_finding(ANCHORED, split, one, 0.5).subtype == AMBIGUOUS_PICK


def test_an_ambiguity_needs_tau_a_and_a_pair_but_no_accuracy_contrast() -> None:
    post = _post(ANCHORED, {F({"d"}): 0.2, F({"d", "a"}): 0.8})
    readings = [_r("c", "m1", "d", 0), _r("c", "m1", "a", 1), _r("c", "m2", "x", 2)]
    found = ambiguity_finding(ANCHORED, post, readings, 0.5)
    assert found is not None and found.values == ("a", "d") and found.score == pytest.approx(0.8)
    assert found.witnesses == ("m1|0", "m1|1")
    alone = [r for r in readings if r.model == "m1"]  # one model: still typed (T2′)
    assert ambiguity_finding(ANCHORED, post, alone, 0.5) is not None
    assert ambiguity_finding(ANCHORED, post, readings, 0.9) is None
    single = _post(ANCHORED, {F({"d"}): 0.9, F(): 0.1})  # one value: no pair to report
    assert ambiguity_finding(ANCHORED, single, readings, 0.0) is None


def test_a_weak_check_is_a_clean_deviant_from_a_confident_single_value() -> None:
    post = _post(FREE, {F({"x"}): 0.9, F(): 0.1})
    readings = [_r("u", "m1", "x", 0, True), _r("u", "m1", "y", 1, True),
                _r("u", "m2", "z", 2, False)]
    found = weak_finding(FREE, post, readings, 0.8)
    assert found is not None and found.values == ("x", "y") and found.witnesses == ("m1|1",)
    assert weak_finding(FREE, post, readings[::2], 0.8) is None  # the deviant is not clean
    assert weak_finding(FREE, post, readings, 0.95) is None
    pair = _post(FREE, {F({"x", "y"}): 0.9, F({"x"}): 0.1})
    assert weak_finding(FREE, pair, readings, 0.5) is None
    rule = _post(RULE, {F({"satisfied"}): 0.95, F(): 0.05})
    assert weak_finding(RULE, rule, [_r("inv:x", "m1", "violated", 0, True)], 0.8) is None


def _params(tau_h: float, tau_a: float, tau_w: float) -> V3Params:
    return V3Params(pi=0.1, s_mis=0.5, s_amb=0.25, s_empty=0.25, beta_sd=1.0,
                    mu_alpha=0.0, sigma_alpha=1.5, tau_h=tau_h, tau_a=tau_a, tau_w=tau_w)


def test_a_cell_reporting_an_ambiguity_reports_no_weak_check() -> None:
    # {x} at 0.55 passes τ_W = 0.5 and the pair at 0.45 passes τ_A = 0.4, with a clean
    # deviant reading y: only the ambiguity rule keeps the weak check out.
    amb = _post(FREE, {F({"x"}): 0.55, F({"x", "y"}): 0.45})
    hidden = _post(ANCHORED, {F({"a"}): 0.95, F({"d"}): 0.05})
    fit = _fit({"m1": 2.0, "m2": 0.0}, amb, hidden)
    by_cell = {"u": [_r("u", "m1", "x", 0, True), _r("u", "m2", "y", 1, True)],
               "c": [_r("c", "m1", "a", 2), _r("c", "m2", "a", 3)]}
    assert weak_finding(FREE, amb, by_cell["u"], 0.5) is not None
    found = type_cells({"u": FREE, "c": ANCHORED}, fit, by_cell, _params(0.5, 0.4, 0.5))
    assert [(f.cell, f.type) for f in found] == [("c", "hidden"), ("u", "ambiguity")]


def test_on_an_anchored_cell_an_ambiguity_and_its_hidden_requirement_co_occur() -> None:
    # §3.1, §5.1, §5.3 (ruling F1). "c": the pair passes τ_A but P(A ≠ {d}) misses τ_H.
    # "e": P(A ≠ {d}) passes τ_H with a pair as the most probable form; the pair misses τ_A.
    # "u": unanchored, so its ambiguity comes alone.
    other = Cell("e", "value", "d", "golden:t:K:e", "t.e", "t.e", ("step",))
    by_pair = _post(ANCHORED, {F({"d"}): 0.45, F({"d", "a"}): 0.4, F({"a"}): 0.15})
    by_form = _post(other, {F({"d"}): 0.05, F({"d", "a"}): 0.34, F({"a"}): 0.31, F(): 0.3})
    free = _post(FREE, {F({"x", "y"}): 0.9, F({"x"}): 0.1})
    by_cell = {"c": [_r("c", "m1", "a", 0), _r("c", "m1", "d", 1)],
               "e": [_r("e", "m1", "a", 2)], "u": [_r("u", "m1", "y", 3)]}
    found = type_cells({"c": ANCHORED, "e": other, "u": FREE},
                       _fit({"m1": 1.0}, by_pair, by_form, free), by_cell,
                       _params(0.9, 0.35, 0.9))
    assert [(f.cell, f.type) for f in found] == [
        ("c", "hidden"), ("c", "ambiguity"), ("e", "hidden"), ("e", "ambiguity"),
        ("u", "ambiguity")]
    hidden = {f.cell: f for f in found if f.type == "hidden"}
    assert hidden["c"].subtype == hidden["e"].subtype == AMBIGUOUS_PICK
    assert hidden["c"].score == pytest.approx(0.55) and hidden["c"].witnesses == ("m1|0",)
    assert hidden["e"].score == pytest.approx(0.95)
    assert type_cells({"c": ANCHORED}, _fit({"m1": 1.0}, by_pair), by_cell,
                      _params(0.9, 0.5, 0.9)) == []


def test_a_cell_reporting_a_hidden_requirement_reports_no_weak_check() -> None:
    # An anchored mis-key cell: MAP {a} != {d}, and the reading of d is clean, so
    # weak_finding alone would type the same defect a second time (resolution 14: weak
    # truths are unanchored cells only).
    post = _post(ANCHORED, {F({"a"}): 0.95, F({"d"}): 0.05})
    readings = [_r("c", "m1", "a", 0), _r("c", "m2", "d", 1, True)]
    assert weak_finding(ANCHORED, post, readings, 0.8) is not None
    found = type_cells({"c": ANCHORED}, _fit({"m1": 1.0, "m2": 1.0}, post),
                       {"c": readings}, _params(0.5, 0.5, 0.8))
    assert [(f.cell, f.type, f.subtype) for f in found] == [("c", "hidden", MIS_KEY)]


# Part 2 (spec v3 §5.1–§5.5): diagnose_v3 on hand-built cells and fits.

S1, S2 = Scenario("s1", 15, 12, 0), Scenario("s2", 15, 15, 0)
CLAUSE = "golden:po:PO-0015:trang_thai"
VALUE = Cell("s1|po:PO-0015:trang_thai", "value", "d", CLAUSE, "po.trang_thai",
             "po.trang_thai", ("step",))


class Recorder:
    """Returns the scripted result for each target set; records every call in order."""

    def __init__(self, results: dict[frozenset[str], RefuteResult] | None = None) -> None:
        self.results, self.calls, self.groups = results or {}, [], []

    def refute(self, targets, scenario, behaviours):
        self.calls.append((tuple(sorted(targets)), scenario.id))
        self.groups.append(tuple(b.run_key for b in behaviours))
        return self.results.get(targets, RefuteResult(True, SYMBOLIC_WITNESS, 1))


def _b(key: str, model: str, scenario: Scenario) -> Behaviour:
    return Behaviour(key, InterpreterId(model, model, 0, 1), scenario, State.empty())


def _case(cells, posts, readings, weights=None):
    fit = AnchoredFit({"m1": 2.0, "m2": 0.0}, {}, {}, {p.cell: p for p in posts},
                      weights or {r.run_key: 0.5 for r in readings}, True, 1, P)
    return CellData({c.id: c for c in cells}, readings), fit


def _hidden_case():
    readings = [Reading(VALUE.id, "b1", "m1", "m1", "s1", "a"),
                Reading(VALUE.id, "b2", "m2", "m2", "s1", "a")]
    post = _post(VALUE, {F({"a"}): 0.9, F({"d"}): 0.1})
    data, fit = _case([VALUE], [post], readings)
    runs = [_b("b1", "m1", S1), _b("b2", "m2", S1)]
    return data, fit, runs, {"b1": F({CLAUSE}), "b2": F({CLAUSE})}


EVERY = F({"s1|step", "s2|step"})
NOTHING: frozenset[tuple[str, ...]] = frozenset()


def _diagnose(data, fit, runs, violated, refuter, observable=EVERY, known=NOTHING):
    relevant = {"step": F({CLAUSE, "inv:x", "inv:y"})}
    return diagnose_v3(runs, violated, data, fit, refuter, observable=observable,
                       relevant=relevant, known=known)


def test_refutation_separates_a_hidden_requirement_from_a_contradiction_of_one() -> None:
    data, fit, runs, violated = _hidden_case()
    refuter = Recorder({F({CLAUSE}): RefuteResult(True, "s1|m1|v0|s100", 2)})
    found = _diagnose(data, fit, runs, violated, refuter)
    [hidden] = found.defects
    assert (hidden.type, hidden.location, hidden.note, hidden.subtype) == (
        "hidden", (CLAUSE,), "refuted by s1|m1|v0|s100", MIS_KEY)
    assert refuter.calls == [((CLAUSE,), "s1")]  # the singleton hitting set defers to the cell
    failed = Recorder({F({CLAUSE}): RefuteResult(False, None, 7)})
    [contradiction] = _diagnose(data, fit, runs, violated, failed).defects
    assert (contradiction.type, contradiction.note) == ("contradiction",
                                                        "unrefuted after 7 attempts")
    capped = Recorder({F({CLAUSE}): RefuteResult(False, None, 0, attempted=False)})
    [default] = _diagnose(data, fit, runs, violated, capped).defects
    assert (default.type, default.note) == ("hidden", UNATTEMPTED)


def test_a_pooled_candidate_is_refuted_where_most_readings_leave_the_anchor() -> None:
    pooled = Cell("golden:po:PO-0015:sl_dat", "value", "{ordered}", "golden:po:PO-0015:sl_dat",
                  "po.sl_dat", "po.sl_dat", ("step",))
    readings = [Reading(pooled.id, "b1", "m1", "m1", "s1", "1"),
                Reading(pooled.id, "b2", "m2", "m2", "s2", "35"),
                Reading(pooled.id, "b3", "m1", "m1", "s2", "{ordered}"),
                Reading(pooled.id, "b4", "m2", "m2", "s1", "{ordered}")]
    data, fit = _case([pooled], [_post(pooled, {F({"1"}): 0.8, F({"{ordered}"}): 0.2})],
                      readings)
    runs = [_b("b1", "m1", S1), _b("b2", "m2", S2), _b("b3", "m1", S2), _b("b4", "m2", S1)]
    refuter = Recorder()
    [d] = _diagnose(data, fit, runs, dict.fromkeys(("b1", "b2", "b3", "b4"), F()),
                    refuter).defects
    assert refuter.calls == [((pooled.id,), "s1")] and d.scenarios == ("s1", "s2")  # a tie
    assert refutation_scenario(pooled, readings[1:]) == "s2"


def test_multi_clause_contradictions_come_from_weighted_hitting_sets() -> None:
    runs = [_b(f"b{i}", "m1", S1) for i in range(4)]
    violated = {"b0": F({"inv:x"}), "b1": F({"inv:x"}), "b2": F({"inv:y"}), "b3": F({"inv:y"})}
    data, fit = _case([], [], [], weights={f"b{i}": 0.5 for i in range(4)})
    pair = F({"inv:x", "inv:y"})
    failed = _diagnose(data, fit, runs, violated, Recorder({pair: RefuteResult(False, None, 4)}))
    assert [(d.type, d.location) for d in failed.defects] == [
        ("contradiction", ("inv:x", "inv:y"))]
    probed = _diagnose(data, fit, runs, violated, Recorder({pair: RefuteResult(True, "s1|p", 3)}))
    assert probed.defects == [] and probed.spent == {("inv:x", "inv:y")}
    repaired = _diagnose(data, fit, runs, violated, Recorder())
    assert repaired.defects == [] and repaired.spent == frozenset()


def test_candidates_go_by_score_and_the_base_reports_locations_last() -> None:
    data, fit, runs, violated = _hidden_case()
    runs += [_b("b3", "m1", S1), _b("b4", "m1", S1)]
    violated |= {"b3": F({"inv:x"}), "b4": F({"inv:y"})}
    fit.weights.update({"b3": 0.5, "b4": 0.5})
    triple = (CLAUSE, "inv:x", "inv:y")  # hits every violation set: support 1.0 > 0.9
    order = Recorder()
    _diagnose(data, fit, runs, violated, order)
    assert [c[0] for c in order.calls] == [triple, (CLAUSE,)]
    later = Recorder()
    _diagnose(data, fit, runs, violated, later, known=F({triple}))
    assert [c[0] for c in later.calls] == [(CLAUSE,), triple]


def test_an_unobservable_candidate_is_listed_and_never_refuted() -> None:
    data, fit, runs, violated = _hidden_case()
    refuter = Recorder()
    found = _diagnose(data, fit, runs, violated, refuter, observable=F())
    assert refuter.calls == [] and found.defects == []
    assert [(d.type, d.location) for d in found.unobservable] == [("unobservable", (CLAUSE,))]


def test_ambiguities_and_weak_checks_use_v2s_location_formats_and_merge_by_step() -> None:
    cells = [Cell(f"{s}|po:PO-0015:trang_thai", "value", None, None, "po.trang_thai",
                  "po.trang_thai", ("step",)) for s in ("s1", "s2")]
    event = Cell("s1|evt:DRAFTED:SO-0031:DE-SM-02", "event", None, None, "evt:DRAFTED:SO-0031",
                 "evt", ("step",))
    readings = [Reading(c.id, f"{c.id}|{m}|{i}", m, m, c.id.split("|")[0], v, True)
                for c in cells for m, vals in (("m1", "xy"), ("m2", "yx")) for i, v in
                enumerate(vals)]
    readings += [Reading(event.id, f"e{i}", "m1", "m1", "s1", "present" if i else "absent", True)
                 for i in range(3)]
    posts = [_post(c, {F({"x", "y"}): 0.9 if c.id.startswith("s1") else 0.7, F({"x"}): 0.1})
             for c in cells] + [_post(event, {F({"present"}): 0.95, F(): 0.05})]
    data, fit = _case([*cells, event], posts, readings)
    runs = [_b(r.run_key, r.model, S1 if r.scenario == "s1" else S2) for r in readings]
    found = _diagnose(data, fit, runs, {r.run_key: F() for r in readings}, Recorder())
    by_type = {d.type: d for d in found.defects}
    amb, weak = by_type["ambiguity"], by_type["weak"]
    assert (amb.location, amb.scenarios, amb.note) == (("po.trang_thai",), ("s1", "s2"),
                                                        "s1|step")
    assert found.ambiguity_cells == {("ambiguity", "step", ("po.trang_thai",)): cells[0].id}
    assert (weak.location, weak.witnesses) == (("evt:DRAFTED:SO-0031",), ("e0",))
    w2 = GroundTruth("W2-00", "weak", F({"evt:DRAFTED:SO-0031"}), "field")
    h1 = GroundTruth("H1-00", "hidden", F({CLAUSE}), "clause")
    assert location_matches(weak, w2)
    data, fit, runs, violated = _hidden_case()
    [hidden] = _diagnose(data, fit, runs, violated, Recorder()).defects
    assert location_matches(hidden, h1)


def test_the_diagnosis_is_deterministic() -> None:
    data, fit, runs, violated = _hidden_case()
    assert (_diagnose(data, fit, runs, violated, Recorder())
            == _diagnose(data, fit, runs[::-1], violated, Recorder()))
    # A multi-clause contradiction: its witnesses and the behaviours the refuter sees do not
    # follow the input order (ruling F19).
    runs = [_b(f"b{i}", "m1", S1) for i in range(4)]
    violated = {"b0": F({"inv:x"}), "b1": F({"inv:x"}), "b2": F({"inv:y"}), "b3": F({"inv:y"})}
    data, fit = _case([], [], [], weights={f"b{i}": 0.5 for i in range(4)})
    pair = F({"inv:x", "inv:y"})
    forward, backward = (Recorder({pair: RefuteResult(False, None, 4)}) for _ in range(2))
    found = _diagnose(data, fit, runs, violated, forward)
    assert found == _diagnose(data, fit, runs[::-1], violated, backward)
    assert found.defects[0].witnesses == ("b0", "b1", "b2", "b3")
    assert forward.groups == backward.groups == [("b0", "b1", "b2", "b3")]
