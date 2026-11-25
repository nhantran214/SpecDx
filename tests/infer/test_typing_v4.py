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


# --- Final fix wave: F-I1 (binary cells), F-I4 (would-be types), Task 3 Minors ---

def _s3_check_cell(n_raters: int = 26, failing: int = 8):
    """An S3-shaped binary check cell: 26 raters with α spread over [−1, 3], one reading
    each; the `failing` weakest read violated, the rest satisfied (the anchor)."""
    from specdx.infer.cells import SATISFIED, VIOLATED

    cell = Cell("s3:x:0:db_match", "constraint", SATISFIED, "k", "db_match", "db_match", ())
    abilities = {f"m{i:02d}": -1.0 + 4.0 * i / (n_raters - 1) for i in range(n_raters)}
    counts = {m: {VIOLATED if i < failing else SATISFIED: 1}
              for i, m in enumerate(abilities)}
    values = candidate_values(SATISFIED, [v for row in counts.values() for v in row])
    return cell, abilities, counts, values


def test_a_binary_gradient_cell_is_never_typed_shared() -> None:
    # Ruling F-I1: on a binary cell λ's posterior is (n_dev − 2)/(n_dev + 2) under Beta(1, 3),
    # a count rule, so it carries no shared-error evidence: no statistical SHARED there.
    cell, abilities, counts, values = _s3_check_cell()
    post = profile_cell(cell, values, counts, abilities, 0.0, P4)
    assert len(values) == 2 and post.log_lr <= -P4.tau_u
    assert post.set_fits[F({cell.anchor})].lam >= P4.tau_lambda  # the count rule fires
    readings = _readings(cell, counts)
    fit = AnchoredFitV4(abilities, {}, {}, {cell.id: post}, {}, True, 1, P4)
    found = type_cells_v4({cell.id: cell}, fit, {cell.id: readings}, P4)
    assert SHARED not in _kinds(found)
    assert not _kinds(found) & set(DEFECT_TYPES_V4)  # rule (ii): no defect either


def test_shared_findings_name_the_runs_that_read_the_attractor() -> None:
    # Task 3 Minor 3: a statistical shared error carries its witnesses, the run keys of the
    # readings on the attractor.
    post = profile_cell(ANCHORED, VALUES, SHARED_COUNTS, RATERS, 0.0, P4)
    readings = _readings(ANCHORED, SHARED_COUNTS)
    [found] = _type(ANCHORED, post, SHARED_COUNTS)
    assert found.type == SHARED
    assert found.witnesses == tuple(sorted(r.run_key for r in readings if r.value == "a"))


def test_would_be_type_names_what_an_undetermined_cell_would_have_been() -> None:
    # Ruling F-I4 (design §5.2): an undetermined cell is counted by the type its own posterior
    # gives as if it were determined.
    from specdx.infer.typing_v4 import WOULD_BE, would_be_type

    assert WOULD_BE == ("hidden", "ambiguity", "weak", "shared", "none")
    mis = profile_cell(ANCHORED, VALUES, MIS, RATERS, 0.0, P4)
    quiet = dataclasses.replace(mis, log_lr=0.1)
    readings = _readings(ANCHORED, MIS)
    assert _kinds(type_cells_v4({"c": ANCHORED}, _fit(quiet), {"c": readings}, P4)) == {
        UNDETERMINED}
    assert would_be_type(ANCHORED, quiet, readings, P4) == "hidden"
    shared = dataclasses.replace(profile_cell(ANCHORED, VALUES, SHARED_COUNTS, RATERS, 0.0, P4),
                                 log_lr=-0.1)
    assert would_be_type(ANCHORED, shared, _readings(ANCHORED, SHARED_COUNTS), P4) == "shared"
    unanimous = {m: {"d": 20} for m in RATERS}
    agree = profile_cell(ANCHORED, ("d",), unanimous, RATERS, 0.0, P4)
    assert would_be_type(ANCHORED, agree, _readings(ANCHORED, unanimous), P4) == "none"
    # A binary cell is never a would-be shared error either (F-I1).
    cell, abilities, counts, values = _s3_check_cell()
    post = profile_cell(cell, values, counts, abilities, 0.0, P4)
    assert would_be_type(cell, post, _readings(cell, counts), P4) == "none"


def test_weak_on_a_real_determined_unanchored_posterior() -> None:
    # Task 3 Minor 5: weak from a real posterior, not only a hand-built one.
    raters = {"m1": 3.0, "m2": 2.5, "m3": 2.0}
    counts = {"m1": {"x": 19, "y": 1}, "m2": {"x": 19, "y": 1}, "m3": {"x": 18, "y": 2}}
    post = profile_cell(FREE, ("x", "y"), counts, raters, 0.0, P4)
    readings = _readings(FREE, counts, clean=True)
    assert post.map_set == {"x"}
    decided = dataclasses.replace(post, log_lr=P4.tau_u + 1)
    found = type_cells_v4({"u": FREE}, _fit(decided), {"u": readings}, P4)
    assert _kinds(found) == {"weak"}
    assert found[0].witnesses == tuple(sorted(r.run_key for r in readings if r.value == "y"))
    quiet = dataclasses.replace(post, log_lr=0.0)
    assert _kinds(type_cells_v4({"u": FREE}, _fit(quiet), {"u": readings}, P4)) == {
        UNDETERMINED}
