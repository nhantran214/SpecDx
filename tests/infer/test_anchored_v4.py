"""The v4 per-cell profile likelihood (design v4 §3, §5.1; ruling T1-L). Every brief test uses
V4Params()."""

import dataclasses
import math

import numpy as np
import pytest

from specdx.infer.anchored import (
    SETS,
    V3Params,
    _encode,
    _flat,
    _likelihoods,
    candidate_values,
    fit_anchored,
)
from specdx.infer.anchored_v4 import (
    BETA_BOX,
    CellPosteriorV4,
    SetFit,
    V4Params,
    _data_loglik,
    _encode_counts,
    _profile,
    profile_cell,
    profile_sets,
)
from specdx.infer.cells import Cell, Reading

P4 = V4Params()
ANCHORED = Cell("c", "value", "d", "k", "t.c", "t.c", ("step",))
UNANCHORED = Cell("u", "value", None, None, "t.u", "t.u", ("step",))
RATERS = {"m1": 3.0, "m2": 2.0, "m3": 0.0, "m4": -1.0}


def _counts(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {m: {v: n for v, n in row.items() if n} for m, row in table.items()}


def _rows(d: tuple[int, ...], a: tuple[int, ...], b: tuple[int, ...]) -> dict[str, dict[str, int]]:
    return _counts({m: {"d": d[i], "a": a[i], "b": b[i]} for i, m in enumerate(RATERS)})


# Readings of a grow with ability (a mis-key: a is what the able raters read) or shrink with it
# (a shared error: the weak raters fall into a), 20 readings per rater.
MIS_KEY = _rows(d=(1, 3, 8, 12), a=(18, 16, 10, 6), b=(1, 1, 2, 2))
SHARED = _rows(d=(18, 16, 10, 6), a=(1, 3, 8, 12), b=(1, 1, 2, 2))


# --- v3 likelihood through v3's own private functions (the λ = 0 reduction) ---

def _v3_logliks(cell: Cell, values: tuple[str, ...], counts: dict[str, dict[str, int]],
                alpha: dict[str, float], beta: float) -> dict[tuple[int, ...], float]:
    readings = [Reading(cell.id, f"{m}-{v}-{i}", m, m, "s", v)
                for m, row in counts.items() for v, n in row.items() for i in range(n)]
    models = sorted(alpha)
    batch = _encode([cell], {cell.id: readings}, models, V3Params(), {cell.id: values})
    eps = _flat(batch)
    _, lik = _likelihoods(batch, np.array([alpha[m] for m in models]), np.array([beta]), eps)
    out = {}
    for i, s in enumerate(SETS):
        if all(k < len(values) for k in s):
            out[s] = float(_xlog(batch.counts[0], lik[0, i]).sum())
    return out


def test_profile_reduces_to_v3_likelihood_when_lambda_is_zero_and_beta_is_fixed() -> None:
    values = ("d", "a", "b")
    counts = _counts({"m1": {"d": 5, "a": 2, "b": 1, "z": 1}, "m2": {"d": 3, "a": 3, "zz": 2},
                      "m3": {"a": 4, "b": 2, "d": 1}})
    alpha = {"m1": 1.5, "m2": 0.2, "m3": -0.7}
    beta = 0.3
    v3 = _v3_logliks(ANCHORED, values, counts, alpha, beta)
    models, table, a = _encode_counts(values, counts, alpha, P4)
    assert models == ("m1", "m2", "m3")
    flat = np.full(len(values) + 1, 1.0 / (len(values) + 1))
    assert len(v3) == 7  # three singles, three pairs, ∅
    for s, expected in v3.items():
        got = _data_loglik(table, a, s, beta, 0.0, None, flat if not s else None)
        assert got == pytest.approx(expected, abs=1e-10), s
    # λ = 0 makes the attractor irrelevant.
    assert _data_loglik(table, a, (0,), beta, 0.0, 2, None) == pytest.approx(v3[(0,)], abs=1e-10)


def test_gradient_separates_mis_key_from_shared_error() -> None:
    values = ("d", "a", "b")
    mis = profile_cell(ANCHORED, values, MIS_KEY, RATERS, 0.0, P4)
    assert isinstance(mis, CellPosteriorV4)
    assert mis.map_set == {"a"}
    assert mis.log_lr >= 2.0
    shared = profile_cell(ANCHORED, values, SHARED, RATERS, 0.0, P4)
    assert shared.map_set == {"d"}
    assert shared.attractor == "a"
    assert shared.lam >= 0.5
    assert shared.log_lr <= -P4.tau_u  # decided: no defect, the deviation is a shared error


def test_unanimous_deviation_is_undetermined() -> None:
    counts = {m: {"a": 20} for m in RATERS}
    values = candidate_values("d", ["a"])
    post = profile_cell(ANCHORED, values, counts, RATERS, 0.0, P4)
    assert abs(post.log_lr) < V4Params().tau_u


def test_empty_set_keeps_flat_pool_and_no_attractor() -> None:
    fits = profile_sets(ANCHORED, ("d", "a", "b"), MIS_KEY, RATERS, 0.0, P4)
    empty = fits[frozenset()]
    assert isinstance(empty, SetFit)
    assert empty.attractor is None
    assert empty.lam == 0.0
    assert empty.beta == 0.0  # no reading informs β_c under ∅: its prior mode β_g


# --- the profile is the exact maximum (accuracy 1e-4 in log-likelihood) ---

def _xlog(n: np.ndarray, x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, n * np.log(x), 0.0)


def _grid_max(table: np.ndarray, a: np.ndarray, s: tuple[int, ...], n: int, beta_g: float,
              priors: bool) -> float:
    """An independent brute-force profile of one non-empty set on a fine grid: β_c (step
    8e-4) and λ (step 5e-5), every attractor; with priors, the N(β_g, 1) and Beta(1, 3)
    log densities (V4Params() defaults), else β_c in the box β_g ± BETA_BOX."""
    inside = np.array([k in s for k in range(n + 1)])
    faithful, errors = table[:, inside].sum(axis=1), table[:, ~inside].sum(axis=1)
    width = 8.0 if priors else BETA_BOX
    betas = np.linspace(beta_g - width, beta_g + width, 20001)
    p = 1.0 / (1.0 + np.exp(-(a[None, :] - betas[:, None])))
    beta_part = (_xlog(faithful, p / len(s)) + _xlog(errors, 1.0 - p)).sum(axis=1)
    if priors:
        beta_part = beta_part - (betas - beta_g) ** 2 / 2 - 0.5 * math.log(2 * math.pi)
    lams = np.linspace(0.0, 1.0, 20001)
    slots = [k for k in range(n + 1) if not inside[k]]
    per_slot = table.sum(axis=0)
    best = -math.inf
    for e in [k for k in slots if k < n] or [None]:
        lam_part = np.zeros_like(lams)
        for k in slots:
            prob = (1 - lams) / len(slots) + (lams if k == e else 0.0)
            lam_part = lam_part + _xlog(np.full_like(lams, per_slot[k]), prob)
        if priors:
            lam_part = lam_part + math.log(3.0) + _xlog(np.full_like(lams, 2.0), 1 - lams)
        best = max(best, float(beta_part.max() + lam_part.max()))
    return best


@pytest.mark.parametrize("data", ["mis", "shared"])
def test_the_profile_matches_a_dense_grid_with_and_without_priors(data: str) -> None:
    counts = MIS_KEY if data == "mis" else SHARED
    values = ("d", "a", "b")
    models, table, a = _encode_counts(values, counts, RATERS, P4)
    fits, likes, _ = _profile(ANCHORED, values, counts, RATERS, 0.0, P4)
    for s in [(0,), (1,), (0, 1), (1, 2)]:
        key = frozenset(values[k] for k in s)
        for priors, got in ((True, fits[key].loglik), (False, likes[key])):
            grid = _grid_max(table, a, s, len(values), 0.0, priors)
            assert grid - 1e-9 <= got <= grid + 1e-4, (s, priors, got, grid)


def test_log_lr_is_prior_free() -> None:
    # Ruling T1-L: the set, β and λ priors move the posterior over sets, never log_lr.
    values = ("d", "a", "b")
    base = profile_cell(ANCHORED, values, SHARED, RATERS, 0.0, P4)
    other = profile_cell(ANCHORED, values, SHARED, RATERS, 0.0,
                         dataclasses.replace(P4, pi=0.6, lambda_b=9.0, cell_sd=0.3))
    assert other.log_lr == pytest.approx(base.log_lr, abs=1e-9)
    assert other.sets[frozenset({"d"})] != pytest.approx(base.sets[frozenset({"d"})], abs=1e-6)


def test_the_map_set_carries_its_fit_and_error_distribution() -> None:
    values = ("d", "a", "b")
    post = profile_cell(ANCHORED, values, SHARED, RATERS, 0.0, P4)
    fit = profile_sets(ANCHORED, values, SHARED, RATERS, 0.0, P4)[frozenset({"d"})]
    assert (post.beta, post.lam, post.attractor) == (fit.beta, fit.lam, fit.attractor)
    assert sum(post.sets.values()) == pytest.approx(1.0)
    assert post.error["d"] == 0.0 and sum(post.error.values()) == pytest.approx(1.0)
    assert post.error["a"] == pytest.approx((1 - fit.lam) / 3 + fit.lam)


def test_an_unanchored_cell_compares_pairs_and_silence_with_the_best_single_value() -> None:
    counts = {m: {"x": 20 - n, "y": n} for m, n in zip(RATERS, (1, 2, 6, 8), strict=True)}
    post = profile_cell(UNANCHORED, ("x", "y"), counts, RATERS, 0.0, P4)
    assert post.p_defect == 0.0 and post.map_set == {"x"}
    assert post.log_lr < 0.0
    with pytest.raises(ValueError):
        profile_cell(UNANCHORED, (), counts, RATERS, 0.0, P4)


# --- Review Focus 5: v3 is untouched. Recorded from fit_anchored at 5590afe, before any v4
# code existed (ruling G2). ---

V3_CELLS = {
    "c1": Cell("c1", "value", "d", "k1", "t.x", "t.x", ("step",)),
    "c2": Cell("c2", "value", "d", "k1", "t.x", "t.x", ("step",)),
    "c3": Cell("c3", "value", "p", "k2", "t.y", "t.y", ("step",)),
    "c4": Cell("c4", "value", None, None, "t.z", "t.z", ("step",)),
}
V3_PLAN = {  # cell -> model -> one reading per character
    "c1": {"m1": "ddddda", "m2": "dddaab", "m3": "ddaabb"},
    "c2": {"m1": "aaaaad", "m2": "aaaadd", "m3": "aaddbz"},
    "c3": {"m1": "pppppq", "m2": "ppppqq", "m3": "pppqrs"},
    "c4": {"m1": "xxxxxy", "m2": "xxxxyy", "m3": "xxyyyz"},
}
V3_FAMILY = {"m1": "f1", "m2": "f1", "m3": "f2"}
V3_ALPHA = {"m1": 1.6718409687487967, "m2": 0.7962410360210206, "m3": -0.07798977330784787}
V3_SETS = {
    "c1": {
        (): 0.00039041845504258577,
        ("a",): 9.437670613150439e-07,
        ("a", "b"): 3.5490329796789034e-06,
        ("a", "d"): 0.00074457772902546,
        ("b",): 1.6461173113193003e-08,
        ("b", "d"): 0.00038431382732430065,
        ("d",): 0.9984761807273936,
    },
    "c2": {
        (): 0.00034375472586519357,
        ("a",): 0.9941835733504537,
        ("a", "b"): 0.0011124154223866331,
        ("a", "d"): 0.0032294038405303372,
        ("a", "z"): 0.0011124154223866331,
        ("b",): 2.1664112281075932e-10,
        ("b", "d"): 3.002318922346427e-07,
        ("b", "z"): 2.01094525657091e-09,
        ("d",): 1.78343303656025e-05,
        ("d", "z"): 3.002318922346427e-07,
        ("z",): 2.1664112281075932e-10,
    },
    "c3": {
        (): 3.5491686196653635e-06,
        ("p",): 0.9995580978642552,
        ("p", "q"): 0.0003950198168537442,
        ("p", "r"): 2.1665778406699788e-05,
        ("p", "s"): 2.1665778406699788e-05,
        ("q",): 3.0810452538509755e-10,
        ("q", "r"): 6.420356705683258e-10,
        ("q", "s"): 6.420356705683213e-10,
        ("r",): 6.797814426835286e-14,
        ("r", "s"): 1.1459742496789696e-12,
        ("s",): 6.797814426835286e-14,
    },
    "c4": {
        (): 0.00042318336447353375,
        ("x",): 0.9987851129881828,
        ("x", "y"): 0.0002996898694754961,
        ("x", "z"): 0.00048807170722051525,
        ("y",): 3.390741918374733e-06,
        ("y", "z"): 5.507890063239538e-07,
        ("z",): 5.397227732092484e-10,
    },
}
V3_WEIGHTS = {
    "m1-0": 0.9997030510305754, "m1-1": 0.9997030510305754, "m1-2": 0.9997030510305754,
    "m1-3": 0.9997030510305754, "m1-4": 0.9997030510305754, "m1-5": 0.0011738904932941671,
    "m2-0": 0.9997030510305754, "m2-1": 0.9997030510305754, "m2-2": 0.9997030510305754,
    "m2-3": 0.7499890505919062, "m2-4": 0.0011738904932941671, "m2-5": 0.0010835926913968267,
    "m3-0": 0.9997030510305754, "m3-1": 0.9997030510305754, "m3-2": 0.2510742474505173,
    "m3-3": 0.0011738904932941671, "m3-4": 0.0004564737563497144,
    "m3-5": 0.0005027216652370688,
}


def test_v3_is_untouched() -> None:
    readings = [Reading(cell, f"{model}-{i}", model, V3_FAMILY[model], "s", v)
                for cell, by_model in V3_PLAN.items() for model, values in by_model.items()
                for i, v in enumerate(values)]
    fit = fit_anchored(V3_CELLS, readings, models=V3_FAMILY, params=V3Params())
    assert set(fit.alpha) == set(V3_ALPHA)
    for m, a in V3_ALPHA.items():
        assert fit.alpha[m] == pytest.approx(a, abs=1e-12)
    assert set(fit.cells) == set(V3_SETS)
    for c, sets in V3_SETS.items():
        got = {tuple(sorted(s)): p for s, p in fit.cells[c].sets.items()}
        assert set(got) == set(sets)
        for s, p in sets.items():
            assert got[s] == pytest.approx(p, abs=1e-12), (c, s)
    assert set(fit.weights) == set(V3_WEIGHTS)
    for k, w in V3_WEIGHTS.items():
        assert fit.weights[k] == pytest.approx(w, abs=1e-12)
