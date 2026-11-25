"""The v4 bounds table (scripts/v4_bounds_table.py, docs/theory/specdx-v4-bounds.md §6): the
T5 lower bound on a binary cell, its behaviour in n·Δ², and its chance limit."""

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "v4_bounds_table.py"
MS = (4, 26)
NS = (1, 2, 8)
P_BARS = (0.5, 0.8, 0.95)
SPREADS = (0.25, 0.5, 1.0, 2.0, 3.0)


@pytest.fixture(scope="module")
def bt():
    spec = importlib.util.spec_from_file_location("v4_bounds_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_abilities_are_centred_with_the_requested_population_sd(bt) -> None:
    for m in MS:
        for s in (0.0, 0.7, 3.0):
            a = bt.abilities(m, s)
            assert a.shape == (m,)
            assert abs(float(a.mean())) < 1e-12
            assert float(a.std()) == pytest.approx(s, abs=1e-12)


def test_divergence_is_the_jensen_gap_of_the_log_partition(bt) -> None:
    # −log BC(Bern σ(θ1), Bern σ(θ2)) = ½A(θ1) + ½A(θ2) − A((θ1 + θ2)/2), A(θ) = log(1 + e^θ).
    for t1, t2 in [(0.3, -1.2), (4.0, 2.5), (-6.0, 7.0), (12.0, 11.0)]:
        x, y = 1 / (1 + math.exp(-t1)), 1 / (1 + math.exp(-t2))
        direct = -math.log(math.sqrt(x * y) + math.sqrt((1 - x) * (1 - y)))
        a = np.logaddexp(0.0, [t1, t2, (t1 + t2) / 2])
        assert bt.neg_log_bc(np.array([t1]), np.array([t2]))[0] == pytest.approx(direct,
                                                                                abs=1e-12)
        assert direct == pytest.approx(0.5 * a[0] + 0.5 * a[1] - a[2], abs=1e-12)


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("p_bar", P_BARS)
def test_no_spread_is_exactly_chance(bt, m: int, n: int, p_bar: float) -> None:
    # T5(ii): with every α equal, β_S = 2ᾱ − β_H reproduces H's reading law exactly.
    row = bt.evaluate(m, n, p_bar, 0.0)
    assert row.h_star == pytest.approx(0.0, abs=1e-12)
    assert row.lower == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("n", NS)
def test_the_bound_tends_to_chance_as_the_spread_vanishes(bt, m: int, n: int) -> None:
    spreads = (0.3, 0.1, 0.03, 0.01, 0.001)
    lows = [bt.evaluate(m, n, 0.8, s).lower for s in spreads]
    assert all(a <= b + 1e-12 for a, b in zip(lows, lows[1:], strict=False))
    # Corollary T5.3: ½ − bound ≤ ¼·√(n·M)·sd(α), uniformly in the accuracy level.
    for s, low in zip(spreads, lows, strict=True):
        assert 0.5 - low <= 0.25 * math.sqrt(n * m) * s + 1e-12
    assert lows[-1] > 0.5 - 1e-2


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("p_bar", P_BARS)
def test_the_bound_is_monotone_in_n_and_in_the_spread(bt, m: int, p_bar: float) -> None:
    grid = {(n, s): bt.evaluate(m, n, p_bar, s) for n in NS for s in SPREADS}
    for s in SPREADS:  # more readings, same contrast: n·Δ² grows
        lows = [grid[(n, s)].lower for n in NS]
        assert all(a > b for a, b in zip(lows, lows[1:], strict=False))
    for n in NS:  # more spread, same n: Δ_c grows, and so does the bound's exponent
        rows = [grid[(n, s)] for s in SPREADS]
        assert all(a.delta_c < b.delta_c for a, b in zip(rows, rows[1:], strict=False))
        assert all(a.h_star < b.h_star for a, b in zip(rows, rows[1:], strict=False))
        assert all(a.lower >= b.lower for a, b in zip(rows, rows[1:], strict=False))


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("p_bar", P_BARS)
def test_the_bound_is_a_decreasing_function_of_n_h_and_h_is_sandwiched_by_n_delta_sq(
        bt, m: int, p_bar: float) -> None:
    rows = [bt.evaluate(m, n, p_bar, s) for n in NS for s in SPREADS]
    for r in rows:
        # h* is linear in n (the least-favourable β_S does not depend on n) ...
        assert r.h_star == pytest.approx(r.n * bt.evaluate(m, 1, p_bar, r.spread).h_star,
                                         rel=1e-9)
        # ... the bound is ½(1 − √(1 − e^{−2h*})) ≥ ¼e^{−2h*} ...
        assert r.lower == pytest.approx(bt.chance_bound(r.h_star), abs=1e-15)
        assert r.lower >= 0.25 * math.exp(-2 * r.h_star) - 1e-15
        # ... and n·Δ_c²/4 ≤ h* ≤ n·M·Δ_c²/(8·η²(1 − η)²) (Proposition T5.4), η the
        # smallest accuracy distance to 0 or 1 over H's and the mimic's logits.
        assert r.n * r.delta_c ** 2 / 4 <= r.h_star + 1e-12
        eta = r.eta
        assert r.h_star <= r.n * m * r.delta_c ** 2 / (8 * eta ** 2 * (1 - eta) ** 2) + 1e-12
    # Monotone in n·Δ_c² up to the sandwich's constants: if row i's upper exponent is below
    # row j's lower exponent, row i's bound is strictly above row j's.
    for ri in rows:
        upper = ri.n * m * ri.delta_c ** 2 / (8 * ri.eta ** 2 * (1 - ri.eta) ** 2)
        for rj in rows:
            if upper < rj.n * rj.delta_c ** 2 / 4:
                assert ri.lower > rj.lower


def test_chance_bound_is_strictly_decreasing_and_matches_the_closed_form(bt) -> None:
    hs = [0.0, 1e-6, 1e-3, 0.1, 0.5, 1.0, 3.0, 10.0, 40.0]
    vals = [bt.chance_bound(h) for h in hs]
    assert vals[0] == 0.5
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
    for h, v in zip(hs[:6], vals[:6], strict=True):
        assert v == pytest.approx(0.5 * (1 - math.sqrt(1 - math.exp(-2 * h))), abs=1e-12)


def test_the_least_favourable_beta_matches_a_dense_minimisation(bt) -> None:
    rng = np.random.default_rng(7)
    for _ in range(12):
        m = int(rng.integers(2, 12))
        alpha = rng.normal(0.0, float(rng.uniform(0.2, 2.5)), m)
        beta_h = float(rng.uniform(-3, 3))
        _, h = bt.least_favourable(alpha, beta_h)
        dense = np.linspace(alpha.min() - 45, alpha.max() + 45, 180_001)
        brute = float(bt._h_one(alpha, beta_h, dense).min())
        assert h <= brute + 1e-9
        assert h >= brute - 1e-4  # the dense grid's own resolution


def test_exact_tv_matches_brute_force_over_reading_sequences(bt) -> None:
    alpha, n, beta_h, beta_s = np.array([-0.7, 1.3]), 2, 0.4, -0.2
    x = 1 / (1 + np.exp(-(alpha - beta_h)))
    y = 1 / (1 + np.exp(-(beta_s - alpha)))
    total = 0.0
    for bits in itertools.product((0, 1), repeat=alpha.size * n):
        ph = py = 1.0
        for i, bit in enumerate(bits):
            m = i // n
            ph *= x[m] if bit else 1 - x[m]
            py *= y[m] if bit else 1 - y[m]
        total += abs(ph - py)
    assert bt.exact_tv(alpha, n, beta_h, beta_s) == pytest.approx(0.5 * total, abs=1e-12)


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("n", NS)
def test_the_closed_form_spread_is_necessary(bt, m: int, n: int) -> None:
    for p in P_BARS:
        assert bt.necessary_spread(m, n) <= bt.smallest_spread(m, n, p) + 1e-9


def test_the_variance_lemma_bounds_h_below_by_the_ability_variance(bt) -> None:
    # Proposition T5.4(a′): h*(β_H) ≥ ½·η_H²(1 − η_H)²·Σ_m n_m(α_m − ᾱ)², η_H = min_m
    # min(x_m, 1 − x_m); with the upper bound ⅛·Σ n_m(α_m − ᾱ)² it is two-sided in N·sd².
    for m in MS:
        for n in NS:
            for p in P_BARS:
                for s in SPREADS[1:]:
                    r = bt.evaluate(m, n, p, s)
                    var = n * m * s ** 2
                    k = r.eta_h * (1 - r.eta_h)
                    assert 0.5 * k ** 2 * var <= r.h_star + 1e-12
                    assert r.h_star <= var / 8 + 1e-12
    rng = np.random.default_rng(11)
    for _ in range(300):
        m = int(rng.integers(2, 30))
        alpha = rng.normal(0.0, float(rng.uniform(0.05, 3.0)), m)
        beta_h = float(rng.uniform(-4, 4))
        x = 1 / (1 + np.exp(-(alpha - beta_h)))
        eta_h = float(np.minimum(x, 1 - x).min())
        var = float(((alpha - alpha.mean()) ** 2).sum())
        _, h = bt.least_favourable(alpha, beta_h)
        assert 0.5 * (eta_h * (1 - eta_h)) ** 2 * var <= h + 1e-12
        assert h <= var / 8 + 1e-12


def test_the_least_favourable_beta_lies_in_the_box_on_the_tabulated_grid(bt) -> None:
    # So h*_B = h* there, and Table 1 also bounds the box-restricted local risk.
    for m in MS:
        for p in P_BARS:
            for s in SPREADS:
                assert bt.evaluate(m, 1, p, s).in_box


def test_saturation_drives_the_bound_to_chance_at_any_spread(bt) -> None:
    lows = [bt.evaluate(26, 8, p, 2.0).lower for p in (0.9, 0.99, 0.999, 0.99999)]
    assert all(a < b for a, b in zip(lows, lows[1:], strict=False))
    assert lows[-1] > 0.5 - 0.1


@pytest.mark.parametrize("n", NS)
def test_the_exact_two_point_bound_dominates_the_hellinger_bound(bt, n: int) -> None:
    for s in (0.25, 1.0, 3.0):
        row = bt.evaluate(4, n, 0.8, s, exact=True)
        assert row.exact is not None
        assert row.lower - 1e-12 <= row.exact <= 0.5 + 1e-12


def test_the_monte_carlo_of_the_code_statistic_is_deterministic_and_complete(bt) -> None:
    one = bt.monte_carlo(4, 2, 1.0, 0.8, "H", reps=6, seed=3)
    assert one == bt.monte_carlo(4, 2, 1.0, 0.8, "H", reps=6, seed=3)
    assert set(one) == {"defect", "undetermined", "no_defect"}
    assert sum(one.values()) == pytest.approx(1.0)
    # A strong gradient in the mis-key direction is typed as a defect every time.
    assert bt.monte_carlo(26, 8, 3.0, 0.5, "H", reps=4, seed=0)["defect"] == 1.0
    assert bt.monte_carlo(26, 8, 3.0, 0.5, "S", reps=4, seed=0)["no_defect"] == 1.0


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("p_bar", P_BARS)
def test_the_s_side_bound(bt, m: int, p_bar: float) -> None:
    # T5-S: chance at zero spread (S is then i.i.d. and ∅ with q = 1 − y reproduces it) ...
    assert bt.evaluate_s(m, 1, p_bar, 0.0).lower == pytest.approx(0.5, abs=1e-9)
    for s in SPREADS[1:]:
        rows = [bt.evaluate_s(m, n, p_bar, s) for n in NS]
        # ... strictly decreasing in n, with every exponent linear in n ...
        assert all(a.lower > b.lower for a, b in zip(rows, rows[1:], strict=False))
        one = rows[0]
        for r in rows:
            for key in ("h_a", "h_da", "h_empty"):
                assert getattr(r, key) == pytest.approx(r.n * getattr(one, key), rel=1e-9)
            # ... and every exponent is ≥ n·Δ_c²/4 for S's own contrast (Corollary T6.1).
            alpha = bt.abilities(m, s)
            beta_s = -math.log(p_bar / (1 - p_bar))
            p = 1 / (1 + np.exp(-(alpha - beta_s)))
            delta_c = float(p.max() - p.min())
            for key in ("h_a", "h_da", "h_empty"):
                assert getattr(r, key) >= r.n * delta_c ** 2 / 4 - 1e-12
            assert r.lower == pytest.approx(
                bt.chance_bound(min(r.h_a, r.h_da, r.h_empty)), abs=1e-15)


def test_the_s_side_pair_bound_is_the_box_end_and_the_empty_bound_is_minimal(bt) -> None:
    alpha, beta_s = bt.abilities(6, 1.0), -1.0
    h_a, h_da, h_empty = bt.s_side_exponents(alpha, beta_s)
    y = 1 / (1 + np.exp(-(beta_s - alpha)))
    lo, hi = alpha.min() - bt.BOX_MARGIN, alpha.max() + bt.BOX_MARGIN
    betas = np.linspace(lo, hi, 20_001)
    p = 1 / (1 + np.exp(-(alpha[None, :] - betas[:, None])))
    da = (-0.5 * np.log(p / 2) - np.log(np.sqrt(1 - y) + np.sqrt(y))[None, :]).sum(axis=1)
    assert h_da == pytest.approx(float(da.min()), abs=1e-9)
    qs = np.linspace(1e-6, 1 - 1e-6, 200_001)[:, None]
    emp = -np.log(np.sqrt(qs * (1 - y)) + np.sqrt((1 - qs) * y)).sum(axis=1)
    assert h_empty <= float(emp.min()) + 1e-9
    theta_q = alpha[None, :] - betas[:, None]
    to_a = bt.neg_log_bc(np.broadcast_to(beta_s - alpha, theta_q.shape), theta_q).sum(axis=1)
    assert h_a <= float(to_a.min()) + 1e-9


def test_the_note_tables_equal_the_script_output(bt) -> None:
    note = (Path(__file__).parents[1] / "docs" / "theory" / "specdx-v4-bounds.md").read_text(
        encoding="utf-8")
    for line in (bt.render() + bt.render_mc()).splitlines():
        if line.startswith("|"):
            assert line in note, line


def test_the_table_covers_the_s3_grid(bt) -> None:
    text = bt.render()
    assert "| M | n | p̄ |" in text
    for m in MS:
        for n in NS:
            assert f"| {m} | {n} | 0.80 |" in text
