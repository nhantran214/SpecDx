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
