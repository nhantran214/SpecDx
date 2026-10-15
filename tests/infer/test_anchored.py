"""The anchored cell model (spec v3 §3). Every test passes its parameters explicitly: the
defaults are provisional until G0 chooses them."""

import dataclasses
import math

import numpy as np
import pytest

from specdx.infer.anchored import (
    EMPTY,
    OTHER,
    SETS,
    V3Params,
    _em,
    _encode,
    _flat,
    _log_posterior,
    candidate_values,
    cell_posterior,
    fit_anchored,
    posterior_at,
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


# Part 2 (spec v3 §3.4–§3.5): the fit.

ALPHAS = {"m1": 2.0, "m2": 1.0, "m3": 0.0}
FIT = V3Params(pi=0.2, s_mis=0.4, s_amb=0.3, s_empty=0.3, beta_sd=1.0,
               mu_alpha=1.0, sigma_alpha=1.5, tau_h=0.5, tau_a=0.5, tau_w=0.8)
PLANTED = {"none": ("d",), "mis": ("a",), "amb": ("d", "a"), "empty": ()}
VALUES = ("d", "a", "b", "c", "e")


def _world(seed: int, columns: dict[str, int], reads: int = 24, per_column: int = 4):
    """The v3.1 §3.2 model at T2′'s known pool: columns of anchored cells (one per scenario)
    share a planted form and a difficulty β_g ~ N(0, 0.5²); errors fall on the flat pool u;
    each model reads each cell `reads` times."""
    rng = np.random.default_rng(seed)
    cells: dict[str, Cell] = {}
    readings: list[Reading] = []
    truth: dict[str, str] = {}
    plan = [form for form, n in columns.items() for _ in range(n)]
    for col, form in enumerate(plan):
        admitted = PLANTED[form]
        beta = rng.normal(0.0, 0.5)
        for s in range(per_column):
            c = col * per_column + s
            cell = Cell(f"c{c:03d}", "value", "d", f"golden:t:{c}:x", f"t.x{col}",
                        f"t.x{col}", ("s",))
            cells[cell.id], truth[cell.id] = cell, form
            eps = np.full(len(VALUES), 1.0 / len(VALUES))
            outside = np.array([0.0 if v in admitted else e
                                for v, e in zip(VALUES, eps, strict=True)])
            for m, a in ALPHAS.items():
                p = 1.0 / (1.0 + math.exp(-(a - beta)))
                for i in range(reads):
                    if admitted and rng.random() < p:
                        y = admitted[int(rng.integers(len(admitted)))]
                    else:
                        y = VALUES[int(rng.choice(len(VALUES), p=outside / outside.sum()))]
                    readings.append(Reading(cell.id, f"{m}|{c}|{i}", m, m, "s", y))
    return cells, readings, truth


def _form(fit, cell: str) -> str:
    forms = fit.cells[cell].forms
    return max(forms, key=lambda f: forms[f])


def test_the_fit_recovers_accuracy_and_all_four_forms() -> None:
    hits: dict[str, list[bool]] = {form: [] for form in PLANTED}
    for seed in (0, 1, 2):  # a form belongs to a column, so one seed's errors come in fours
        cells, readings, truth = _world(seed, {"none": 30, "mis": 3, "amb": 3, "empty": 3})
        fit = fit_anchored(cells, readings, models={m: m for m in ALPHAS}, params=FIT)
        assert fit.converged
        for m, a in ALPHAS.items():
            assert abs(fit.theta[m] - 1 / (1 + math.exp(-a))) <= 0.05, (m, fit.theta[m])
        for c, form in truth.items():
            hits[form].append(_form(fit, c) == form)
    for form, got in hits.items():
        assert sum(got) / len(got) >= 0.8, (form, sum(got), len(got))


def test_the_fit_is_deterministic_and_ignores_the_order_of_readings() -> None:
    cells, readings, _ = _world(3, {"none": 8, "mis": 1, "amb": 1, "empty": 1}, reads=6)
    one = fit_anchored(cells, readings, models={m: m for m in ALPHAS}, params=FIT)
    two = fit_anchored(cells, readings[::-1], models={m: m for m in ALPHAS}, params=FIT)
    assert one == two


def test_theta_does_not_collapse_when_most_anchored_cells_are_read_alike_off_the_anchor() -> None:
    # Shaped after the P1b smoke run: most anchored cells are rows no model writes (every
    # reading absent), the rest agree with their anchors. T3's median start sits below
    # the MAP; the fit keeps the start with the highest log posterior.
    cells: dict[str, Cell] = {}
    readings: list[Reading] = []
    for c in range(40):
        cell = Cell(f"c{c:02d}", "value", "d", f"golden:t:{c}:x", "t.x", "t.x", ("s",))
        cells[cell.id] = cell
        for m in ("gpt", "qwen"):
            for i in range(9):
                value = "absent" if c < 26 else ("d" if i < 8 else f"e{i}")
                readings.append(Reading(cell.id, f"{m}|{i}", m, m, "s", value))
    fit = fit_anchored(cells, readings, models={"gpt": "gpt", "qwen": "qwen"}, params=FIT)
    assert min(fit.theta.values()) >= 0.5
    assert fit.start != "t3" and fit.starts[fit.start].score > fit.starts["t3"].score
    assert all(fit.cells[f"c{c:02d}"].p_defect > 0.9 for c in range(26))


def test_non_convergence_is_reported() -> None:
    cells, readings, _ = _world(5, {"none": 3, "mis": 1}, reads=6)
    fit = fit_anchored(cells, readings, models={m: m for m in ALPHAS}, params=FIT, max_iter=1)
    assert not fit.converged and fit.iterations == 1


def test_weights_family_effect_and_a_model_without_readings() -> None:
    cells, readings, _ = _world(11, {"none": 5, "mis": 1}, reads=6)
    families = {"m1": "f", "m2": "f", "m3": "g", "idle": "g"}
    readings = [dataclasses.replace(r, family=families[r.model]) for r in readings]
    fit = fit_anchored(cells, readings, models=families, params=FIT)
    assert fit.alpha["idle"] == pytest.approx(FIT.mu_alpha)
    assert fit.family_effect["f"] == pytest.approx(
        (fit.alpha["m1"] + fit.alpha["m2"]) / 2 - sum(fit.alpha.values()) / 4)
    assert set(fit.weights) == {r.run_key for r in readings}
    assert all(0.0 <= w <= 1.0 for w in fit.weights.values())
    mean = {m: np.mean([w for k, w in fit.weights.items() if k.startswith(m)]) for m in ALPHAS}
    assert mean["m1"] > mean["m3"]


def test_an_empty_input_gives_an_empty_converged_fit() -> None:
    fit = fit_anchored({}, [], models={"m1": "f"}, params=FIT)
    assert fit.cells == {} and fit.converged and fit.iterations == 0
    assert fit.theta["m1"] == pytest.approx(1 / (1 + math.exp(-FIT.mu_alpha)))


def test_posterior_at_the_fitted_parameters_reproduces_the_fit() -> None:
    cells, readings, _ = _world(13, {"none": 3, "amb": 1}, reads=8)
    fit = fit_anchored(cells, readings, models={m: m for m in ALPHAS}, params=FIT)
    cell = cells["c012"]
    again = posterior_at(fit, cell, [r for r in readings if r.cell == cell.id])
    for s, p in fit.cells[cell.id].sets.items():
        assert again.sets[s] == pytest.approx(p, abs=1e-9)


# Fix round 1.


def test_each_em_iteration_does_not_lower_the_log_posterior() -> None:
    # The ε M-step is the exact EM step for the truncated error model: it adds the expected
    # rejected draws that §3.2's renormalisation over V∖A implies.
    cells, readings, _ = _world(0, {"none": 30, "mis": 3, "amb": 3, "empty": 3})
    names = sorted(ALPHAS)
    by_cell: dict[str, list[Reading]] = {}
    for r in readings:
        by_cell.setdefault(r.cell, []).append(r)
    batch = _encode([cells[c] for c in sorted(by_cell)], by_cell, names, FIT)
    prior = _flat(batch)
    start = np.full(len(names), 1.0)
    scores = [_log_posterior(batch, start, np.zeros(int(batch.groups.max()) + 1), prior,
                             prior, FIT)]
    for n in range(1, 31):
        a, b, e, _, _ = _em(batch, prior, start, FIT, 0.0, n)
        scores.append(_log_posterior(batch, a, b, e, prior, FIT))
    drops = [after - before for before, after in zip(scores, scores[1:], strict=False)]
    assert min(drops) >= -1e-9, min(drops)


def test_every_start_reports_its_score_and_whether_it_converged() -> None:
    cells, readings, _ = _world(5, {"none": 3, "mis": 1}, reads=6)
    fit = fit_anchored(cells, readings, models={m: m for m in ALPHAS}, params=FIT, max_iter=1)
    assert set(fit.starts) == {"t3", "alpha=0.0", "alpha=1.0", "alpha=2.0", "alpha=3.0"}
    assert all(not s.converged and s.iterations == 1 for s in fit.starts.values())
    full = fit_anchored(cells, readings, models={m: m for m in ALPHAS}, params=FIT)
    assert all(s.converged for s in full.starts.values())
    assert full.starts[full.start].score == max(s.score for s in full.starts.values())


def test_a_model_whose_family_disagrees_raises() -> None:
    cells, readings, _ = _world(5, {"none": 2}, reads=2)
    models = {m: m for m in ALPHAS}
    clash = readings + [dataclasses.replace(readings[0], run_key="extra", family="other")]
    with pytest.raises(ValueError, match="m1"):
        fit_anchored(cells, clash, models=models, params=FIT)
    with pytest.raises(ValueError, match="m1"):
        fit_anchored(cells, readings, models=models | {"m1": "other"}, params=FIT)
