"""The cell-level typing core (spec v3.1 §5.1, §5.3, §5.4), on hand-built posteriors."""

import pytest

from specdx.infer.anchored import AnchoredFit, CellPosterior, V3Params
from specdx.infer.cells import Cell, Reading
from specdx.infer.typing import (
    AMBIGUOUS_PICK,
    MIS_KEY,
    UNSTATED,
    ambiguity_finding,
    hidden_finding,
    type_cells,
    weak_finding,
)

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

