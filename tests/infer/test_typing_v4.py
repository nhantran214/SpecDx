"""v4 typing (P4a Task 3; ruling T3-N): undetermined and shared error are not defects."""

import dataclasses

import pytest

from specdx.infer.anchored import candidate_values
from specdx.infer.anchored_v4 import AnchoredFitV4, CellPosteriorV4, SetFit, V4Params, profile_cell
from specdx.infer.cells import Cell, Reading
from specdx.infer.typing import AMBIGUOUS_PICK, MIS_KEY, CellFinding
from specdx.infer.typing_v4 import DEFECT_TYPES_V4, SHARED, UNDETERMINED, type_cells_v4

P4 = V4Params()
ANCHORED = Cell("c", "value", "d", "k", "t.c", "t.c", ("step",))
FREE = Cell("u", "value", None, None, "t.u", "t.u", ("step",))
RATERS = {"m1": 3.0, "m2": 2.0, "m3": 0.0, "m4": -1.0}
F = frozenset
VALUES = ("d", "a", "b")
MIS = {m: {"d": d, "a": 19 - d, "b": 1} for m, d in zip(RATERS, (1, 3, 8, 12), strict=True)}
SHARED_COUNTS = {m: {"d": 19 - a, "a": a, "b": 1}
                 for m, a in zip(RATERS, (1, 3, 8, 12), strict=True)}


def _readings(cell: Cell, counts: dict[str, dict[str, int]],
              clean: bool = False) -> list[Reading]:
    out, i = [], 0
    for model, row in counts.items():
        for value, n in row.items():
            for _ in range(n):
                out.append(Reading(cell.id, f"{model}|{i}", model, model, "s", value, clean))
                i += 1
    return out


def _fit(*posts: CellPosteriorV4) -> AnchoredFitV4:
    return AnchoredFitV4(dict(RATERS), {}, {}, {p.cell: p for p in posts}, {}, True, 1, P4)


def _type(cell: Cell, post: CellPosteriorV4, counts: dict,
          clean: bool = False) -> list[CellFinding]:
    by_cell = {cell.id: _readings(cell, counts, clean)}
    return type_cells_v4({cell.id: cell}, _fit(post), by_cell, P4)


def _kinds(found: list[CellFinding]) -> set[str]:
    return {f.type for f in found}


def test_unanimous_cell_is_undetermined_only() -> None:
    counts = {m: {"a": 20} for m in RATERS}
    post = profile_cell(ANCHORED, candidate_values("d", ["a"]), counts, RATERS, 0.0, P4)
    found = _type(ANCHORED, post, counts)
    assert [f.type for f in found] == [UNDETERMINED]
    assert found[0].score == pytest.approx(abs(post.log_lr))
    assert UNDETERMINED not in DEFECT_TYPES_V4 and SHARED not in DEFECT_TYPES_V4


def test_shared_error_cell_is_reported_as_shared_not_hidden() -> None:
    post = profile_cell(ANCHORED, VALUES, SHARED_COUNTS, RATERS, 0.0, P4)
    found = _type(ANCHORED, post, SHARED_COUNTS)
    assert post.log_lr <= -P4.tau_u  # rule (ii) fires, not rule (iii)
    assert [f.type for f in found] == [SHARED]
    assert found[0].values == ("a",) and found[0].score == pytest.approx(post.lam)
    assert post.lam >= P4.tau_lambda


def test_mis_key_cell_is_hidden_with_subtype() -> None:
    post = profile_cell(ANCHORED, VALUES, MIS, RATERS, 0.0, P4)
    found = _type(ANCHORED, post, MIS)
    assert [f.type for f in found] == ["hidden"]
    assert found[0].subtype == MIS_KEY


def test_ambiguity_and_weak_follow_v3_rules_on_determined_cells() -> None:
    base = profile_cell(ANCHORED, VALUES, MIS, RATERS, 0.0, P4)
    pair = dataclasses.replace(base, sets={F({"d", "a"}): 0.8, F({"d"}): 0.2},
                               log_lr=5.0, lam=0.0, attractor=None)
    found = _type(ANCHORED, pair, MIS)
    assert {"ambiguity", "hidden"} <= _kinds(found)
    assert next(f for f in found if f.type == "ambiguity").values == ("a", "d")
    assert next(f for f in found if f.type == "hidden").subtype == AMBIGUOUS_PICK
    # an unanchored cell: a confident single value with a clean deviant is a weak check
    free = dataclasses.replace(base, cell="u", anchor=None, values=("x", "y"),
                               sets={F({"x"}): 0.95, F(): 0.05}, log_lr=5.0, lam=0.0,
                               attractor=None)
    counts = {"m1": {"x": 9, "y": 1}}
    weak = _type(FREE, free, counts, clean=True)
    assert _kinds(weak) == {"weak"}
    # the same cell below tau_u is undetermined only
    quiet = dataclasses.replace(free, log_lr=0.5)
    assert _kinds(_type(FREE, quiet, counts, clean=True)) == {UNDETERMINED}


def test_a_decided_no_defect_cell_emits_no_defect_even_when_the_map_set_is_a() -> None:
    # T3-N rule 2: log_lr <= -tau_u says no defect, whatever the prior-bearing MAP set is.
    base = profile_cell(ANCHORED, VALUES, SHARED_COUNTS, RATERS, 0.0, P4)
    d_fit = SetFit(-10.0, 0.0, 0.7, "a")
    post = dataclasses.replace(base, sets={F({"a"}): 0.8, F({"d"}): 0.2}, lam=0.0,
                               attractor=None, log_lr=-5.0, set_fits={F({"d"}): d_fit})
    assert post.map_set == {"a"}
    found = _type(ANCHORED, post, SHARED_COUNTS)
    assert _kinds(found) == {SHARED}  # no hidden, no ambiguity, no weak
    assert found[0].score == pytest.approx(0.7) and found[0].values == ("a",)
    low = dataclasses.replace(post, set_fits={F({"d"}): SetFit(-10.0, 0.0, 0.2, "a")})
    assert _type(ANCHORED, low, SHARED_COUNTS) == []
    # an unanchored cell never reports shared error
    free = dataclasses.replace(post, cell="u", anchor=None, values=("x", "y"),
                               sets={F({"x"}): 0.9, F({"y"}): 0.1})
    assert _type(FREE, free, {}) == []


def test_profile_cell_exposes_every_set_fit() -> None:
    post = profile_cell(ANCHORED, VALUES, SHARED_COUNTS, RATERS, 0.0, P4)
    assert F({"d"}) in post.set_fits and F() in post.set_fits
    assert post.set_fits[post.map_set].lam == post.lam


# Real posteriors found by a seeded random search over abilities, counts and beta_g.
def _real(counts: dict, abilities: dict, beta_g: float) -> tuple[CellPosteriorV4, list[Reading]]:
    post = profile_cell(ANCHORED, VALUES, counts, abilities, beta_g, P4)
    return post, _readings(ANCHORED, counts)


def _type_real(post: CellPosteriorV4, readings: list[Reading]) -> list[CellFinding]:
    return type_cells_v4({"c": ANCHORED}, _fit(post), {"c": readings}, P4)


def test_rule_iii_reports_shared_when_the_statistic_is_defect_side_but_the_map_is_d() -> None:
    counts = {"m0": {"d": 5}, "m1": {"d": 4, "b": 1}, "m2": {"d": 10}, "m3": {"d": 20},
              "m4": {"d": 5}, "m5": {"d": 1, "b": 4}}
    abilities = dict(zip(counts, (-1.1212, 2.3130, -1.0386, 2.2276, 2.0691, 1.2682),
                         strict=True))
    post, readings = _real(counts, abilities, -0.7745)
    assert post.log_lr >= P4.tau_u and post.map_set == {"d"}
    assert post.lam >= P4.tau_lambda
    found = _type_real(post, readings)
    assert SHARED in _kinds(found)
    assert next(f for f in found if f.type == SHARED).score == pytest.approx(post.lam)


def test_rule_iii_with_a_weak_attractor_reports_no_shared() -> None:
    counts = {"m0": {"d": 16, "a": 1, "b": 3}, "m1": {"d": 17, "a": 1, "b": 2}}
    post, readings = _real(counts, {"m0": 0.3021, "m1": 3.1417}, -0.2956)
    assert post.log_lr >= P4.tau_u and post.map_set == {"d"}
    assert post.lam < P4.tau_lambda
    assert SHARED not in _kinds(_type_real(post, readings))


def test_rule_ii_on_a_real_posterior_whose_map_is_a_defect_form() -> None:
    # The prior-bearing MAP is the pair {d, a} while the prior-free statistic says no defect:
    # the posterior must not type a defect.
    counts = {"m0": {"d": 11, "a": 9}, "m1": {"d": 5}}
    post, readings = _real(counts, {"m0": 1.9371, "m1": 3.5357}, -1.9284)
    assert post.log_lr <= -P4.tau_u and post.map_set == {"d", "a"}
    assert not _kinds(_type_real(post, readings)) & set(DEFECT_TYPES_V4)
