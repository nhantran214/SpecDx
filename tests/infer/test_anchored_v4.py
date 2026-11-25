"""The v4 per-cell profile likelihood (design v4 §3, §5.1; ruling T1-L) and the v4 outer fit
(§3.3–§3.5; ruling T2-U′). Every brief test uses V4Params()."""

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
    BOX_MARGIN,
    AnchoredFitV4,
    CellPosteriorV4,
    SetFit,
    V4Params,
    _data_loglik,
    _encode_counts,
    _m_step,
    _Pairs,
    _plug_in,
    _profile,
    _profiled_beta,
    _q_objective,
    fit_v4,
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


_POPULATION = {f"r{i:02d}": float(x)
               for i, x in enumerate(np.random.default_rng(20261120).normal(1.0, 1.0, 30))}


@pytest.mark.parametrize("raters", [RATERS, _POPULATION], ids=["4-raters", "30-raters"])
@pytest.mark.parametrize("beta_g", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_unanimous_is_undetermined_for_any_beta_g(raters: dict[str, float],
                                                  beta_g: float) -> None:
    # Ruling T1-L′: T5 makes a unanimous deviation undecidable whatever β_g is, so log_lr's
    # box is relative to the raters present and log_lr does not depend on β_g.
    counts = {m: {"a": 20} for m in raters}
    values = candidate_values("d", ["a"])
    post = profile_cell(ANCHORED, values, counts, raters, beta_g, P4)
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


def _joint_grid(table: np.ndarray, a: np.ndarray, s: tuple[int, ...], n: int, beta_g: float,
                priors: bool, e: int | None) -> float:
    """An independent brute force of one non-empty set at a fixed attractor e (None: none):
    the full reading likelihood P(k | m, β_c, λ) evaluated on a joint 2-D (β_c, λ) mesh over
    the box [min α − BOX_MARGIN, max α + BOX_MARGIN] × [0, 1], 121 × 101 points, then three
    zooms of 201 × 201 points around the mesh maximum (±2 mesh steps each), so the last steps
    are below 1e-4 in β_c and 1e-5 in λ. With priors, the N(β_g, 1) and Beta(1, 3) log
    densities (V4Params() defaults) are added on the mesh."""
    inside = np.array([k in s for k in range(n + 1)])
    outside = int((~inside).sum())
    lo, hi = float(a.min()) - BOX_MARGIN, float(a.max()) + BOX_MARGIN

    def loglik(betas: np.ndarray, lams: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(a[None, :] - betas[:, None])))  # [B, M]
        hot = np.arange(n + 1) == e
        err = (1 - lams)[:, None] / outside + lams[:, None] * hot[None, :]  # [L, K]
        prob = np.where(inside[None, None, None, :], p[:, None, :, None] / len(s),
                        (1 - p)[:, None, :, None] * err[None, :, None, :])  # [B, L, M, K]
        out = _xlog(table[None, None, :, :], prob).sum(axis=(2, 3))
        if priors:
            out = (out - ((betas - beta_g) ** 2 / 2 + 0.5 * math.log(2 * math.pi))[:, None]
                   + math.log(3.0) + _xlog(np.full_like(lams, 2.0), 1 - lams)[None, :])
        return out

    betas, lams = np.linspace(lo, hi, 121), np.linspace(0.0, 1.0, 101)
    best = -math.inf
    for _ in range(4):
        mesh = loglik(betas, lams)
        i, j = np.unravel_index(int(np.argmax(mesh)), mesh.shape)
        best = max(best, float(mesh[i, j]))
        db, dl = 2 * (betas[1] - betas[0]), 2 * (lams[1] - lams[0])
        betas = np.linspace(max(lo, betas[i] - db), min(hi, betas[i] + db), 201)
        lams = np.linspace(max(0.0, lams[j] - dl), min(1.0, lams[j] + dl), 201)
    return best


def _grid_max(table: np.ndarray, a: np.ndarray, s: tuple[int, ...], n: int, beta_g: float,
              priors: bool) -> float:
    """The joint grid's maximum over every attractor (or none, when only OTHER is outside s)."""
    choices = [k for k in range(n) if k not in s] or [None]
    return max(_joint_grid(table, a, s, n, beta_g, priors, e) for e in choices)


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


# --- Task 2: the v4 outer fit ---

ALPHAS4 = (3.0, 2.0, 1.0, 0.0)


def _shared_data(alphas: tuple[float, ...] = ALPHAS4, cells: int = 200, shared: float = 0.3,
                 per: int = 10, groups: int = 20, seed: int = 20261120,
                 ) -> tuple[dict[str, Cell], list[Reading], dict[str, str]]:
    """A hand-built shared-error arm (ruling G1: Task 4's simulator arm comes later). Each
    anchored cell c has vocabulary d, v1, v2, v3, true set {d} and difficulty β_c = β_g +
    N(0, 1) with β_g ~ N(0, 1) per group. The hardest `shared` share of the cells carry a
    shared error: errors are (1 − λ)·flat + λ·δ_v1 with λ ~ U(0.5, 1); elsewhere errors are
    flat. Each model reads each cell `per` times, every reading its own act group."""
    rng = np.random.default_rng(seed)
    models = {f"m{i}": f"f{i}" for i in range(len(alphas))}
    group_beta = rng.normal(0.0, 1.0, groups)
    betas = group_beta[np.arange(cells) % groups] + rng.normal(0.0, 1.0, cells)
    hard = set(np.argsort(-betas)[:round(shared * cells)].tolist())
    out: dict[str, Cell] = {}
    readings: list[Reading] = []
    for c in range(cells):
        cid, group = f"c{c:03d}", f"g{c % groups:02d}"
        out[cid] = Cell(cid, "value", "d", f"k{c}", group, group, ("step",))
        lam = float(rng.uniform(0.5, 1.0)) if c in hard else 0.0
        err = np.full(3, (1.0 - lam) / 3)
        err[0] += lam
        for (m, family), a in zip(models.items(), alphas, strict=True):
            p = 1.0 / (1.0 + math.exp(-(a - betas[c])))
            for k in range(per):
                value = ("d" if rng.random() < p
                         else ("v1", "v2", "v3")[int(rng.choice(3, p=err))])
                readings.append(Reading(cid, f"{m}|{cid}|{k}", m, family, f"s{k}", value))
    return out, readings, models


def _order(theta: dict[str, float]) -> list[str]:
    return sorted(theta, key=lambda m: -theta[m])


def test_recovers_ability_order_with_shared_errors() -> None:
    cells, readings, models = _shared_data()
    fit = fit_v4(cells, readings, models=models, params=P4)
    assert isinstance(fit, AnchoredFitV4)
    assert _order(fit.theta) == ["m0", "m1", "m2", "m3"]
    assert fit.converged


def _small() -> tuple[dict[str, Cell], list[Reading], dict[str, str]]:
    return _shared_data(alphas=(2.5, 1.0, -0.5), cells=30, per=6, groups=5, seed=7)


def test_composite_weights_do_not_change_single_cell_groups() -> None:
    cells, readings, models = _small()
    assert len({(r.run_key, r.scenario) for r in readings}) == len(readings)
    weighted = fit_v4(cells, readings, models=models, params=P4)
    plain = fit_v4(cells, readings, models=models, params=P4, composite=False)
    assert weighted.alpha == plain.alpha
    assert weighted.weights == plain.weights
    assert weighted.start == plain.start


def test_composite_weights_damp_correlated_groups() -> None:
    cells, readings, models = _small()
    single = fit_v4(cells, readings, models=models, params=P4)
    # Every reading copied into k = 4 identical cells of one difficulty group, keeping its
    # run key and scenario: one act read four times.
    copies = {f"{cid}#{i}": dataclasses.replace(cell, id=f"{cid}#{i}")
              for cid, cell in cells.items() for i in range(4)}
    four = [dataclasses.replace(r, cell=f"{r.cell}#{i}") for r in readings for i in range(4)]
    damped = fit_v4(copies, four, models=models, params=P4)
    for m in models:
        assert damped.alpha[m] == pytest.approx(single.alpha[m], abs=1e-3)
    # Equal up to EM's stopping point (tol 1e-6 on α; the copies may stop one step later).
    assert damped.weights == pytest.approx(single.weights, abs=1e-6)
    # Without the weights the four copies count four times and the abilities move.
    loud = fit_v4(copies, four, models=models, params=P4, composite=False)
    assert max(abs(loud.alpha[m] - single.alpha[m]) for m in models) > 1e-3


def test_fit_reports_convergence_and_start() -> None:
    cells, readings, models = _small()
    fit = fit_v4(cells, readings, models=models, params=P4)
    assert fit.converged and fit.iterations >= 1
    assert set(fit.starts) == {"t3", "alpha=0.0", "alpha=1.0", "alpha=2.0", "alpha=3.0"}
    assert fit.start in fit.starts
    assert fit.starts[fit.start].score == max(s.score for s in fit.starts.values())
    assert fit.starts[fit.start].iterations == fit.iterations
    assert fit.params is P4
    assert set(fit.cells) == set(cells)
    assert all(isinstance(post, CellPosteriorV4) for post in fit.cells.values())
    for m, a in fit.alpha.items():
        assert fit.theta[m] == pytest.approx(1.0 / (1.0 + math.exp(-a)))
    assert set(fit.family_effect) == set(models.values())
    assert set(fit.weights) == {r.run_key for r in readings}
    assert all(0.0 <= w <= 1.0 for w in fit.weights.values())
    # Not converging within max_iter is reported, never silent.
    short = fit_v4(cells, readings, models=models, params=P4, max_iter=1)
    assert not short.converged and short.iterations == 1


def test_all_undetermined_data_still_returns_a_fit() -> None:
    # Anchored cells every model always reads as a (anchor d): undetermined for any α (T5).
    # Ruling T2-U′ keeps them in the updates; the fit still returns, converged, and since
    # every model reads alike no ability is told apart from another.
    models = {"m0": "f0", "m1": "f1", "m2": "f2"}
    cells = {f"u{c}": Cell(f"u{c}", "value", "d", f"ku{c}", "ug", "ug", ("step",))
             for c in range(6)}
    readings = [Reading(cid, f"{m}|{cid}|{k}", m, family, f"s{k}", "a")
                for cid in cells for m, family in models.items() for k in range(5)]
    fit = fit_v4(cells, readings, models=models, params=P4)
    assert fit.converged
    assert all(math.isfinite(a) for a in fit.alpha.values())
    assert max(fit.alpha.values()) - min(fit.alpha.values()) < 1e-9
    assert all(abs(post.log_lr) < P4.tau_u for post in fit.cells.values())
    # EM has more than one fixed point on these data: the start with the best score is the
    # one kept, and the common ability stays within a few prior sds of μ_α.
    assert fit.starts[fit.start].score == max(s.score for s in fit.starts.values())
    assert all(abs(a - P4.mu_alpha) < 3 * P4.sigma_alpha for a in fit.alpha.values())


def _moving_pairs() -> _Pairs:
    rng = np.random.default_rng(11)
    total = rng.integers(1, 6, (12, 3)).astype(float)
    inside = np.floor(total * rng.uniform(0, 1, (12, 3)))
    return _Pairs(inside, total, np.ones(12), np.ones(12), np.arange(12) % 2,
                  np.full(12, -12.0), np.full(12, 12.0))


def test_the_line_search_never_takes_a_step_that_lowers_q(monkeypatch) -> None:
    # Task 2 Minor 5: when no step size down to the floor raises Q, the M-step stops where
    # it is instead of taking the floor-sized step.
    from specdx.infer import anchored_v4

    pairs = _moving_pairs()
    start_a, start_b = np.zeros(3), np.zeros(2)

    def lower_everywhere(p, a, b, params, cell_beta=True):
        return -1.0 if (np.array_equal(a, start_a) and np.array_equal(b, start_b)) else -1.1

    monkeypatch.setattr(anchored_v4, "_q_objective", lower_everywhere)
    alpha, beta = _m_step(pairs, start_a.copy(), start_b.copy(), P4)[:2]
    assert np.array_equal(alpha, start_a) and np.array_equal(beta, start_b)


def test_the_line_search_tolerance_is_relative_to_q(monkeypatch) -> None:
    # Task 10 Minor 1: a drop in Q at float noise (1e-14 of |Q|) is no reason to halve the
    # step: the Newton steps go through and reach the true maximiser.
    from specdx.infer import anchored_v4

    pairs = _moving_pairs()
    want_a, want_b = _m_step(pairs, np.zeros(3), np.zeros(2), P4)[:2]
    calls = [0]

    def noisy(p, a, b, params, cell_beta=True):
        # Every evaluation comes out 1e-8 below the one before: noise, not a real descent.
        calls[0] += 1
        return 1e6 - 1e-8 * calls[0]

    monkeypatch.setattr(anchored_v4, "_q_objective", noisy)
    alpha, beta = _m_step(pairs, np.zeros(3), np.zeros(2), P4)[:2]
    assert alpha == pytest.approx(want_a, abs=1e-6) and beta == pytest.approx(want_b, abs=1e-6)


def test_an_m_step_that_hits_its_newton_cap_is_reported(monkeypatch) -> None:
    # Task 2 Minor 6: the inner Newton loop's cap is never silent.
    from specdx.infer import anchored_v4

    cells, readings, models = _small()
    fit = fit_v4(cells, readings, models=models, params=P4)
    assert fit.m_step_capped == 0
    monkeypatch.setattr(anchored_v4, "_NEWTON_MAX", 1)
    capped = fit_v4(cells, readings, models=models, params=P4, max_iter=3)
    assert capped.m_step_capped > 0


def test_the_fit_exposes_the_group_difficulties() -> None:
    # Task 2 Minor 8: β_g per difficulty group, for a later posterior at the fitted values.
    cells, readings, models = _small()
    fit = fit_v4(cells, readings, models=models, params=P4)
    assert set(fit.beta_g) == {c.group for c in cells.values()}
    post = profile_cell(cells["c000"], fit.cells["c000"].values,
                        _counts_of(readings, "c000"), fit.alpha,
                        fit.beta_g[cells["c000"].group], P4)
    assert post.sets == pytest.approx(fit.cells["c000"].sets, abs=1e-12)


def _counts_of(readings: list[Reading], cid: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in readings:
        if r.cell == cid:
            row = out.setdefault(r.model, {})
            row[r.value] = row.get(r.value, 0) + 1
    return out


def test_the_fit_converges_with_mixed_act_group_sizes() -> None:
    # Task 2 Minor 1: act groups of 1, 2 and 3 readings in one fit (as S3's missing cells
    # give); EM still converges and the abilities keep their order.
    cells, readings, models = _shared_data(alphas=(2.5, 1.0, -0.5), cells=60, per=6,
                                           groups=6, seed=9)
    mixed = [dataclasses.replace(r, run_key=f"{r.model}|g{i // (1 + i % 3)}", scenario="s")
             for i, r in enumerate(readings)]
    assert len({len([1 for r in mixed if (r.run_key, r.scenario) == key])
                for key in {(r.run_key, r.scenario) for r in mixed}}) >= 2
    fit = fit_v4(cells, mixed, models=models, params=P4)
    assert fit.converged
    assert _order(fit.theta) == ["m0", "m1", "m2"]


def test_the_v3_view_maps_every_field_by_name() -> None:
    from specdx.infer.anchored_v4 import _v3_view

    p = V4Params(pi=0.11, s_mis=0.5, s_amb=0.2, s_empty=0.3, beta_sd=1.7, mu_alpha=0.3,
                 sigma_alpha=1.9, tau_h=0.61, tau_a=0.41, tau_w=0.81)
    assert _v3_view(p) == V3Params(pi=0.11, s_mis=0.5, s_amb=0.2, s_empty=0.3, beta_sd=1.7,
                                   mu_alpha=0.3, sigma_alpha=1.9, tau_h=0.61, tau_a=0.41,
                                   tau_w=0.81)


def test_the_m_step_maximises_the_profiled_composite_objective() -> None:
    # The M-step maximises Q(α, β_g) = Σ_j ω_j·max_{β_c}[Σ_m in·log σ(α − β_c) + (tot − in)·
    # log σ(β_c − α) − P_j·(β_c − β_g)²/2] + the α and β_g log priors: its gradient vanishes
    # there, and each β_g is its pairs' profiled β_c shrunk under N(0, beta_sd²).
    rng = np.random.default_rng(3)
    j, m, g = 40, 3, 4
    total = rng.integers(0, 8, (j, m)).astype(float) * rng.choice([0.25, 1.0], (j, 1))
    inside = np.floor(total * rng.uniform(0, 1, (j, m)))
    pairs = _Pairs(inside, total, rng.uniform(0.1, 1.0, j), rng.uniform(0.25, 1.0, j),
                   np.arange(j) % g, np.full(j, -12.0), np.full(j, 12.0))
    alpha, beta = _m_step(pairs, np.zeros(m), np.zeros(g), P4)[:2]
    best = _q_objective(pairs, alpha, beta, P4)
    h = 1e-5
    for i in range(m + g):
        step = np.zeros(m + g)
        step[i] = h
        up = _q_objective(pairs, alpha + step[:m], beta + step[m:], P4)
        down = _q_objective(pairs, alpha - step[:m], beta - step[m:], P4)
        assert (up - down) / (2 * h) == pytest.approx(0.0, abs=1e-5)
        assert up <= best + 1e-12 and down <= best + 1e-12
    star = _profiled_beta(pairs, alpha, beta)
    for k in range(g):
        mine = pairs.group == k
        weight = pairs.weight[mine] * pairs.prec[mine]
        expected = (weight * star[mine]).sum() / (weight.sum() + 1.0 / P4.beta_sd ** 2)
        assert beta[k] == pytest.approx(expected, abs=1e-8)


# --- Task 10: the ablation switches (design §6.4): profile=False and cell_beta=False ---

def _plugged(counts, beta_g: float = 0.0):
    """_profile's fits and likes, then _plug_in's, with the MAP set and the table."""
    values = ("d", "a", "b")
    fits, likes, _ = _profile(ANCHORED, values, counts, RATERS, beta_g, P4)
    post = profile_cell(ANCHORED, values, counts, RATERS, beta_g, P4)
    best = fits[post.map_set]
    _, table, a = _encode_counts(values, counts, RATERS, P4)
    plug_fits, plug_likes = dict(fits), dict(likes)
    _plug_in(values, table, a, beta_g, plug_fits, plug_likes, best, P4)
    return values, fits, likes, plug_fits, plug_likes, post.map_set, best, table, a


def test_plug_in_evaluates_every_set_at_the_map_sets_prior_bearing_point() -> None:
    for counts in (MIS_KEY, SHARED):
        values, fits, likes, pfits, plikes, map_set, best, table, a = _plugged(counts)
        slot = {v: k for k, v in enumerate(values)}
        per_slot = table.sum(axis=0)
        for name in likes:
            if not name:
                assert plikes[name] == likes[name] and pfits[name] == fits[name]  # ∅ keeps
                continue
            s = tuple(sorted(slot[v] for v in name))
            choices = [k for k in range(len(values)) if k not in s]
            e = max(choices, key=lambda k: (per_slot[k], -k)) if choices else None
            live = e is not None and best.lam > 0
            want = _data_loglik(table, a, s, best.beta, best.lam if live else 0.0,
                                e if live else None, None)
            assert plikes[name] == pytest.approx(want, abs=1e-10)
            assert pfits[name].beta == best.beta
        # The MAP set at the prior-bearing point is its profiled posterior, but its likes
        # entry is now the data log-likelihood at (β_map, λ_map, e*), not the prior-free box
        # profile.
        assert pfits[map_set].loglik == pytest.approx(fits[map_set].loglik, abs=1e-9)
        if map_set:
            assert plikes[map_set] != pytest.approx(likes[map_set], abs=1e-9)
        for key, fit in fits.items():
            assert pfits[key].loglik <= fit.loglik + 1e-9  # a feasible point never wins


def test_plug_in_log_lr_is_the_difference_of_the_recorded_likes() -> None:
    values = ("d", "a", "b")
    for counts in (MIS_KEY, SHARED):
        _, _, _, _, plikes, _, _, _, _ = _plugged(counts)
        want = (max(v for k, v in plikes.items() if k != frozenset({"d"}))
                - plikes[frozenset({"d"})])
        plug = profile_cell(ANCHORED, values, counts, RATERS, 0.0, P4, profile=False)
        assert plug.log_lr == pytest.approx(want, abs=1e-10)
        prof = profile_cell(ANCHORED, values, counts, RATERS, 0.0, P4)
        assert plug.log_lr != pytest.approx(prof.log_lr, abs=1e-6)


def test_cell_beta_false_pins_every_beta_c_to_beta_g_and_changes_log_lr() -> None:
    values = ("d", "a", "b")
    full = profile_cell(ANCHORED, values, MIS_KEY, RATERS, 0.7, P4)
    pinned = profile_cell(ANCHORED, values, MIS_KEY, RATERS, 0.7, P4, cell_beta=False)
    assert all(f.beta == 0.7 for f in pinned.set_fits.values())
    assert pinned.log_lr != pytest.approx(full.log_lr, abs=1e-6)
    # On a unanimous deviation the free β_c leaves it undecidable (T5); with β_c ≡ β_g the
    # raters' abilities decide it.
    counts = {m: {"a": 20} for m in RATERS}
    vals = candidate_values("d", ["a"])
    free = profile_cell(ANCHORED, vals, counts, RATERS, 0.0, P4)
    fixed = profile_cell(ANCHORED, vals, counts, RATERS, 0.0, P4, cell_beta=False)
    assert abs(free.log_lr) < P4.tau_u
    assert abs(fixed.log_lr) > P4.tau_u and fixed.log_lr != pytest.approx(free.log_lr)


def test_cell_beta_false_m_step_maximises_the_group_only_objective() -> None:
    # With β_c ≡ β_g, Q(α, β_g) = Σ_j ω_j·h_j(α, β_g) + the log priors: the M-step reaches
    # its maximum (zero gradient, checked by central differences), with no β_c to profile.
    rng = np.random.default_rng(5)
    j, m, g = 40, 3, 4
    total = rng.integers(1, 8, (j, m)).astype(float)
    inside = np.floor(total * rng.uniform(0, 1, (j, m)))
    pairs = _Pairs(inside, total, rng.uniform(0.1, 1.0, j), rng.uniform(0.25, 1.0, j),
                   np.arange(j) % g, np.full(j, -12.0), np.full(j, 12.0))
    alpha, beta = _m_step(pairs, np.zeros(m), np.zeros(g), P4, cell_beta=False)[:2]
    best = _q_objective(pairs, alpha, beta, P4, cell_beta=False)
    h = 1e-5
    for i in range(m + g):
        step = np.zeros(m + g)
        step[i] = h
        up = _q_objective(pairs, alpha + step[:m], beta + step[m:], P4, cell_beta=False)
        down = _q_objective(pairs, alpha - step[:m], beta - step[m:], P4, cell_beta=False)
        assert (up - down) / (2 * h) == pytest.approx(0.0, abs=1e-5)
        assert up <= best + 1e-12 and down <= best + 1e-12
    assert np.array_equal(_profiled_beta(pairs, alpha, beta, cell_beta=False),
                          beta[pairs.group])
    # It is not the free-β_c optimum.
    free_alpha, free_beta = _m_step(pairs, np.zeros(m), np.zeros(g), P4)[:2]
    assert not np.allclose(free_beta, beta, atol=1e-6)


def test_cell_beta_false_changes_the_undetermined_count_on_the_fixture() -> None:
    from specdx.infer.typing_v4 import UNDETERMINED, type_cells_v4

    cells, readings, models = _small()
    by_cell: dict[str, list[Reading]] = {}
    for r in readings:
        by_cell.setdefault(r.cell, []).append(r)

    def undetermined(**kw) -> int:
        fit = fit_v4(cells, readings, models=models, params=P4, **kw)
        return sum(f.type == UNDETERMINED for f in type_cells_v4(cells, fit, by_cell, P4))

    assert undetermined(cell_beta=False) != undetermined()


def test_the_default_switches_reproduce_the_base_fit_exactly() -> None:
    # Literals recorded from base commit 572b0fc (before profile/cell_beta existed).
    cells, readings, models = _small()
    fit = fit_v4(cells, readings, models=models, params=P4)
    assert fit.alpha == {"m0": 2.752707049902913, "m1": 1.6980324389406978,
                         "m2": -0.0017929231459093118}
    assert (fit.iterations, fit.start, fit.converged) == (12, "alpha=1.0", True)
    assert [fit.cells[c].log_lr for c in ("c000", "c001", "c002", "c003", "c004", "c005")] == [
        -2.8882420550695644, -1.8800944278756386, -0.06571421811088385,
        -3.8563828646025033, -2.5628422815512373, -2.0770286386396997]
    assert sum(post.log_lr for post in fit.cells.values()) == -84.8797332049631
    values = ("d", "a", "b")
    mis = profile_cell(ANCHORED, values, MIS_KEY, RATERS, 0.0, P4)
    shared = profile_cell(ANCHORED, values, SHARED, RATERS, 0.0, P4)
    assert mis.log_lr == 54.57134381445822
    assert shared.log_lr == -7.817768566452756
    assert mis.set_fits[frozenset({"d"})].loglik == -122.92784577122895
    assert shared.set_fits[frozenset({"d"})].loglik == -64.6399929872344
    explicit = fit_v4(cells, readings, models=models, params=P4, profile=True, cell_beta=True)
    assert explicit.alpha == fit.alpha


def test_fit_v4_with_plug_in_runs_and_recovers_the_order() -> None:
    cells, readings, models = _shared_data()
    fit = fit_v4(cells, readings, models=models, params=P4, profile=False)
    assert _order(fit.theta) == ["m0", "m1", "m2", "m3"]
    assert set(fit.cells) == set(cells)


def test_fit_v4_with_cell_beta_false_runs_and_recovers_the_order() -> None:
    cells, readings, models = _shared_data()
    fit = fit_v4(cells, readings, models=models, params=P4, cell_beta=False)
    assert _order(fit.theta) == ["m0", "m1", "m2", "m3"]
    assert set(fit.cells) == set(cells)
