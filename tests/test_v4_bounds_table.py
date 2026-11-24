"""The v4 bounds table (scripts/v4_bounds_table.py, docs/theory/specdx-v4-bounds.md §6): the
T5 lower bound on a binary cell, its behaviour in n·Δ², and its chance limit."""

import importlib.util
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
    ordered = sorted(rows, key=lambda r: r.h_star)
    assert all(a.lower >= b.lower for a, b in zip(ordered, ordered[1:], strict=False))


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


def test_the_table_covers_the_s3_grid(bt) -> None:
    text = bt.render()
    assert "| M | n | p̄ |" in text
    for m in MS:
        for n in NS:
            assert f"| {m} | {n} | 0.80 |" in text
