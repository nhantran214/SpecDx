"""The anchored cell model (spec v3 §3). Every test passes its parameters explicitly: the
defaults are provisional until G0 chooses them."""

import math

import numpy as np
import pytest

from specdx.infer.anchored import (
    EMPTY,
    OTHER,
    SETS,
    V3Params,
    candidate_values,
    cell_posterior,
    set_log_prior,
)
from specdx.infer.cells import Cell, Reading

P = V3Params(pi=0.2, s_mis=0.5, s_amb=0.3, s_empty=0.2, beta_sd=1.0,
             mu_alpha=0.0, sigma_alpha=1.5, tau_h=0.5, tau_a=0.5, tau_w=0.8)
ANCHORED = Cell("c", "value", "a", "golden:t:K:c", "t.c", "t.c", ("step",))
UNANCHORED = Cell("u", "value", None, None, "t.u", "t.u", ("step",))


def _r(cell: str, model: str, value: str, i: int) -> Reading:
    return Reading(cell, f"{model}|{value}|{i}", model, model, "s", value)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_candidate_values_put_the_anchor_first_then_frequency_then_the_string() -> None:
    assert candidate_values("a", ["c", "b", "c", "b", "d"]) == ("a", "b", "c", "d")
    assert candidate_values(None, ["c", "b", "c"]) == ("c", "b")
    many = [f"v{i}" for i in range(9) for _ in range(9 - i)]
    assert candidate_values("a", many) == ("a", "v0", "v1", "v2", "v3", "v4")


def test_there_are_22_candidate_sets_with_the_empty_set_last() -> None:
    assert len(SETS) == 22 and SETS[EMPTY] == ()
    assert sum(len(s) == 1 for s in SETS) == 6 and sum(len(s) == 2 for s in SETS) == 15


def test_the_anchored_prior_splits_pi_over_the_forms_uniformly_within_each() -> None:
    prior = np.exp(set_log_prior(4, True, P))
    by = {s: prior[i] for i, s in enumerate(SETS)}
    assert by[(0,)] == pytest.approx(0.8)
    assert by[(1,)] == by[(3,)] == pytest.approx(0.2 * 0.5 / 3)
    assert by[(0, 1)] == by[(2, 3)] == pytest.approx(0.2 * 0.3 / 6)
    assert by[()] == pytest.approx(0.2 * 0.2)
    assert by[(4,)] == by[(0, 5)] == 0.0 and prior.sum() == pytest.approx(1.0)
    alone = np.exp(set_log_prior(1, True, P))
    assert alone[0] == pytest.approx(0.8 / (0.8 + 0.04))
    assert alone[EMPTY] == pytest.approx(0.04 / (0.8 + 0.04))


def test_the_unanchored_prior_gives_the_single_values_one_minus_pi() -> None:
    prior = np.exp(set_log_prior(3, False, P))
    by = {s: prior[i] for i, s in enumerate(SETS)}
    assert by[(0,)] == by[(2,)] == pytest.approx(0.8 / 3)
    assert by[(0, 2)] == pytest.approx(0.2 * (0.3 / 0.5) / 3)
    assert by[()] == pytest.approx(0.2 * (0.2 / 0.5))
    assert prior.sum() == pytest.approx(1.0)


def test_the_posterior_at_fixed_parameters_is_the_hand_computed_one() -> None:
    readings = [_r("c", "m1", "a", 0), _r("c", "m2", "b", 1), _r("c", "m2", "z", 2)]
    error = {"a": 0.2, "b": 0.3, OTHER: 0.5}
    post = cell_posterior(ANCHORED, ("a", "b"), readings, alpha={"m1": 1.0, "m2": 0.0},
                          beta=0.0, error=error, params=P)
    p1, p2 = _sigmoid(1.0), 0.5
    lik = {
        frozenset({"a"}): p1 * ((1 - p2) * 0.3 / 0.8) * ((1 - p2) * 0.5 / 0.8),
        frozenset({"b"}): ((1 - p1) * 0.2 / 0.7) * p2 * ((1 - p2) * 0.5 / 0.7),
        frozenset({"a", "b"}): (p1 / 2) * (p2 / 2) * ((1 - p2) * 0.5 / 0.5),
        frozenset(): 0.2 * 0.3 * 0.5,
    }
    prior = {frozenset({"a"}): 0.8, frozenset({"b"}): 0.1, frozenset({"a", "b"}): 0.06,
             frozenset(): 0.04}
    z = sum(prior[s] * lik[s] for s in lik)
    assert set(post.sets) == set(lik)  # OTHER is never in a set
    for s in lik:
        assert post.sets[s] == pytest.approx(prior[s] * lik[s] / z, rel=1e-12)
    assert post.p_defect == pytest.approx(1 - post.sets[frozenset({"a"})])
    assert post.forms["amb"] == pytest.approx(post.sets[frozenset({"a", "b"})])


def test_readings_that_all_agree_on_another_value_are_a_mis_key_not_silence() -> None:
    readings = [_r("c", m, "b", i) for m in ("m1", "m2") for i in range(10)]
    post = cell_posterior(ANCHORED, ("a", "b"), readings, alpha={"m1": 2.0, "m2": 0.0},
                          beta=0.0, error={"a": 1 / 3, "b": 1 / 3, OTHER: 1 / 3}, params=P)
    assert post.map_set == {"b"} and post.forms["mis"] > 0.99
    assert post.forms["mis"] > post.forms["empty"]


def test_readings_that_are_all_distinct_put_the_defect_mass_on_silence() -> None:
    readings = [_r("c", "m1", f"c{i}", i) for i in (1, 2, 3)] + [
        _r("c", "m2", f"c{i}", i) for i in (4, 5, 6)]
    values = candidate_values("a", [r.value for r in readings])
    assert values == ("a", "c1", "c2", "c3", "c4", "c5")
    error = dict.fromkeys((*values, OTHER), 1 / 7)
    post = cell_posterior(ANCHORED, values, readings, alpha={"m1": 2.0, "m2": 0.0}, beta=0.0,
                          error=error, params=P)
    assert post.forms["empty"] > 0.5 and post.map_set == frozenset()
    assert max(p for s, p in post.sets.items() if len(s) == 1 and "a" not in s) < 0.1


def test_one_model_separates_all_four_forms_when_the_error_pool_is_flat() -> None:
    # Theorem T2′: with ε = u flat and the model's accuracy known (p = 0.9), each form has its
    # own reading distribution, so a single model needs no accuracy contrast.
    values = ("a", "x", "b", "c")  # the anchor a first; z falls in OTHER
    flat = dict.fromkeys((*values, OTHER), 0.2)
    shapes = {"none": {"a": 36, "x": 1, "b": 1, "c": 1, "z": 1},
              "mis": {"x": 36, "a": 1, "b": 1, "c": 1, "z": 1},
              "amb": {"a": 18, "x": 18, "b": 2, "c": 1, "z": 1},
              "empty": {"a": 8, "x": 8, "b": 8, "c": 8, "z": 8}}
    for form, counts in shapes.items():
        readings = [_r("c", "m1", v, i) for v, n in counts.items() for i in range(n)]
        post = cell_posterior(ANCHORED, values, readings, alpha={"m1": math.log(9.0)}, beta=0.0,
                              error=flat, params=P)
        assert max(post.forms, key=lambda f: post.forms[f]) == form, (form, post.forms)


def test_an_unanchored_cell_has_no_defect_mass_and_counts_single_values_as_none() -> None:
    readings = [_r("u", "m1", "x", i) for i in range(4)]
    post = cell_posterior(UNANCHORED, ("x",), readings, alpha={"m1": 1.0}, beta=0.0,
                          error={"x": 0.5, OTHER: 0.5}, params=P)
    assert post.p_defect == 0.0 and post.map_set == {"x"}
    assert post.forms["mis"] == 0.0 and post.forms["none"] > 0.5
