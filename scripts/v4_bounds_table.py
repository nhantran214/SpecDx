"""The numerical illustration of docs/theory/specdx-v4-bounds.md §6: T5's lower bound on a
binary S3 cell, evaluated at S3-like population sizes, readings per model and ability spreads.

Setting (the note's §3). M models with known abilities α_m, equally spaced, centred at 0 and
with population standard deviation `spread`; n readings per model at one binary cell with
anchor d and the other value a. Under H (mis-key, A = {a}) model m reads a with probability
x_m = σ(α_m − β_H); under S (shared error, A = {d}, λ = 1, e = a) with probability
y_m = σ(β_S − α_m). β_H is fixed by the accuracy p̄ = σ(ᾱ − β_H) of a model of mean ability;
β_S is the free nuisance that the bound chooses least favourably.

With h(β_S) = n·Σ_m −log BC(Bern x_m, Bern y_m) and h* = inf over β_S, Le Cam's two-point
lemma and the Hellinger bound on TV give, for every test,

    max(P_H(error), P_S(error)) ≥ ½·(1 − √(1 − e^{−2h*}))   (≥ ¼·e^{−2h*}).

For small M the exact two-point bound ½·(1 − TV), at the best β_S found, is also printed.
With --mc, Table 5 adds a Monte Carlo of the code's own log_lr on simulated cells.

Run: .venv/bin/python scripts/v4_bounds_table.py [--mc]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

MS = (4, 26)
NS = (1, 2, 8)
P_BARS = (0.5, 0.8, 0.95)
SPREADS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
TARGET = 0.05  # the error level of the "smallest spread" table
_GRID_MARGIN = 40.0
_GRID_STEP = 0.01
_EXACT_MAX = 2_000_000  # (n + 1)^M support points for the exact TV


@dataclass(frozen=True)
class Row:
    """One evaluation. delta_c: H's accuracy contrast σ(max α − β_H) − σ(min α − β_H).
    h_star: inf over β_S of n·Σ_m −log BC_m; beta_s: its minimiser (±inf when the infimum is
    a limit). lower: ½(1 − √(1 − e^{−2h*})). eta: the smallest of σ(t), 1 − σ(t) over
    |t − logit p̄| ≤ max_m |α_m − ᾱ| (Proposition T5.4's η). exact: ½(1 − TV) at the best β_S
    found, or None when not computed."""

    m: int
    n: int
    p_bar: float
    spread: float
    delta_c: float
    h_star: float
    beta_s: float
    lower: float
    eta: float
    exact: float | None


def abilities(m: int, spread: float) -> np.ndarray:
    """M equally spaced abilities, mean 0, population standard deviation `spread`."""
    if m < 1:
        raise ValueError("m must be positive")
    z = np.arange(m, dtype=float) - (m - 1) / 2
    sd = float(z.std())
    return z * (spread / sd) if sd > 0 else np.zeros(m)


def _log_sigmoid(t: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -t)


def _sigmoid(t: float) -> float:
    return 1.0 / (1.0 + math.exp(-t)) if t >= 0 else math.exp(t) / (1.0 + math.exp(t))


def neg_log_bc(theta_1: np.ndarray, theta_2: np.ndarray) -> np.ndarray:
    """−log of the Bhattacharyya coefficient of Bern(σ(θ1)) and Bern(σ(θ2)), elementwise:
    −log(√(xy) + √((1 − x)(1 − y))), computed in log space. It equals the Jensen gap
    ½A(θ1) + ½A(θ2) − A((θ1 + θ2)/2) of A(θ) = log(1 + e^θ)."""
    up = 0.5 * (_log_sigmoid(theta_1) + _log_sigmoid(theta_2))
    down = 0.5 * (_log_sigmoid(-theta_1) + _log_sigmoid(-theta_2))
    return np.maximum(-np.logaddexp(up, down), 0.0)


def _h_one(alpha: np.ndarray, beta_h: float, beta_s: np.ndarray) -> np.ndarray:
    """Σ_m −log BC_m per reading (n = 1), for each β_S in beta_s."""
    theta_h = (alpha - beta_h)[None, :]
    theta_s = beta_s[:, None] - alpha[None, :]
    return neg_log_bc(np.broadcast_to(theta_h, theta_s.shape), theta_s).sum(axis=1)


def least_favourable(alpha: np.ndarray, beta_h: float) -> tuple[float, float]:
    """(β_S, h₁*): the β_S minimising h per reading, by a grid over [min α − 40, max α + 40]
    refined by golden section, compared with the limits β_S → ±∞ (where y_m → 1 or 0 and
    −log BC_m → −½·log x_m or −½·log(1 − x_m))."""
    grid = np.arange(float(alpha.min()) - _GRID_MARGIN, float(alpha.max()) + _GRID_MARGIN,
                     _GRID_STEP)
    values = _h_one(alpha, beta_h, grid)
    k = int(np.argmin(values))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fa, fb = _h_one(alpha, beta_h, np.array([a, b]))
        if fa <= fb:
            hi = b
        else:
            lo = a
    best_beta = 0.5 * (lo + hi)
    best = float(_h_one(alpha, beta_h, np.array([best_beta]))[0])
    if values[k] < best:
        best_beta, best = float(grid[k]), float(values[k])
    mirror = 2 * float(alpha.mean()) - beta_h  # the pivot γ = ᾱ: exact at zero spread
    at_mirror = float(_h_one(alpha, beta_h, np.array([mirror]))[0])
    if at_mirror < best:
        best_beta, best = mirror, at_mirror
    theta_h = alpha - beta_h
    plus = float(-0.5 * _log_sigmoid(theta_h).sum())
    minus = float(-0.5 * _log_sigmoid(-theta_h).sum())
    if plus < best:
        best_beta, best = math.inf, plus
    if minus < best:
        best_beta, best = -math.inf, minus
    return best_beta, best


def chance_bound(h: float) -> float:
    """½(1 − √(1 − e^{−2h})): Le Cam's ½(1 − TV) with TV ≤ √(1 − BC²) and BC = e^{−h}.
    Written as ½·u/(1 + √(1 − u)), u = e^{−2h}, so that it does not round to 0 at large h."""
    u = math.exp(-2.0 * h)
    return 0.5 * u / (1.0 + math.sqrt(max(-math.expm1(-2.0 * h), 0.0)))


def _binomial_pmf(n: int, p: np.ndarray) -> np.ndarray:
    k = np.arange(n + 1)
    logc = np.array([math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                     for j in k])
    with np.errstate(divide="ignore"):
        lp, lq = np.log(p)[:, None], np.log1p(-p)[:, None]
    terms = logc[None, :] + np.where(k[None, :] > 0, k[None, :] * lp, 0.0) \
        + np.where(n - k[None, :] > 0, (n - k[None, :]) * lq, 0.0)
    return np.exp(terms)


def exact_tv(alpha: np.ndarray, n: int, beta_h: float, beta_s: float) -> float:
    """TV between the two product laws of the per-model counts of a (sufficient for both),
    by enumerating the (n + 1)^M count vectors."""
    if (n + 1) ** alpha.size > _EXACT_MAX:
        raise ValueError("support too large for the exact TV")
    x = 1.0 / (1.0 + np.exp(-(alpha - beta_h)))
    if math.isinf(beta_s):
        y = np.full(alpha.size, 1.0 if beta_s > 0 else 0.0)
    else:
        y = 1.0 / (1.0 + np.exp(-(beta_s - alpha)))
    px, py = np.ones(1), np.ones(1)
    for row_x, row_y in zip(_binomial_pmf(n, x), _binomial_pmf(n, y), strict=True):
        px, py = np.outer(px, row_x).ravel(), np.outer(py, row_y).ravel()
    return float(0.5 * np.abs(px - py).sum())


def evaluate(m: int, n: int, p_bar: float, spread: float, exact: bool = False) -> Row:
    """T5 at one setting (module docstring); exact=True adds the exact two-point bound, at
    the best β_S among the Hellinger minimiser and a ±1 grid (step 0.01) around it."""
    alpha = abilities(m, spread)
    c = math.log(p_bar / (1 - p_bar))
    beta_h = float(alpha.mean()) - c
    beta_s, h_one = least_favourable(alpha, beta_h)
    h_star = n * h_one
    delta_c = _sigmoid(float(alpha.max()) - beta_h) - _sigmoid(float(alpha.min()) - beta_h)
    reach = float(np.abs(alpha - alpha.mean()).max())
    eta = min(min(_sigmoid(t), 1 - _sigmoid(t)) for t in (c - reach, c + reach))
    best = None
    if exact:
        centre = beta_s if math.isfinite(beta_s) else 2 * float(alpha.mean()) - beta_h
        candidates = [beta_s, *np.arange(centre - 1.0, centre + 1.0 + 1e-9, 0.01).tolist()]
        best = max(0.5 * (1 - exact_tv(alpha, n, beta_h, b)) for b in candidates)
    return Row(m, n, p_bar, spread, delta_c, h_star, beta_s, chance_bound(h_star), eta, best)


def smallest_spread(m: int, n: int, p_bar: float, target: float = TARGET,
                    top: float = 20.0) -> float:
    """The smallest sd(α) at which the bound falls to target, by bisection on [0, top]
    (the bound decreases in the spread on the tabulated grid); inf if it never does."""
    if evaluate(m, n, p_bar, top).lower > target:
        return math.inf
    lo, hi = 0.0, top
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if evaluate(m, n, p_bar, mid).lower > target:
            lo = mid
        else:
            hi = mid
    return hi


def necessary_spread(m: int, n: int, target: float = TARGET) -> float:
    """Corollary T5.3's closed form: no test reaches error `target` unless
    sd(α) ≥ √(8·h_δ/(n·M)), h_δ = −½·log(4δ(1 − δ)), at any accuracy level."""
    h_delta = -0.5 * math.log(4 * target * (1 - target))
    return math.sqrt(8 * h_delta / (n * m))


def monte_carlo(m: int, n: int, spread: float, p_bar: float, truth: str, reps: int = 400,
                seed: int = 20261125) -> dict[str, float]:
    """How the code's own statistic (specdx.infer.anchored_v4.profile_cell's log_lr, at
    V4Params' default tau_u) types a binary anchored cell simulated under truth "H" (A = {a},
    errors onto d) or "S" (A = {d}, λ = 1, e = a), with the abilities of `evaluate` and β_c
    set by p̄. Returns the shares typed defect (log_lr ≥ tau_u), undetermined and no defect
    (log_lr ≤ −tau_u). CPU only; deterministic in seed."""
    from specdx.infer.anchored import candidate_values
    from specdx.infer.anchored_v4 import V4Params, profile_cell
    from specdx.infer.cells import Cell

    if truth not in ("H", "S"):
        raise ValueError("truth must be 'H' or 'S'")
    params = V4Params()
    cell = Cell("c", "value", "d", "k", "t.c", "t.c", ("step",))
    alpha = abilities(m, spread)
    beta = float(alpha.mean()) - math.log(p_bar / (1 - p_bar))
    raters = {f"r{i:02d}": float(a) for i, a in enumerate(alpha)}
    rng = np.random.default_rng(seed)
    out = {"defect": 0, "undetermined": 0, "no_defect": 0}
    for _ in range(reps):
        counts: dict[str, dict[str, int]] = {}
        observed: list[str] = []
        for name, a in raters.items():
            faithful = int(rng.binomial(n, _sigmoid(a - beta)))
            n_a = faithful if truth == "H" else n - faithful
            counts[name] = {v: k for v, k in (("a", n_a), ("d", n - n_a)) if k}
            observed += ["a"] * n_a + ["d"] * (n - n_a)
        values = candidate_values("d", observed)
        lr = profile_cell(cell, values, counts, raters, 0.0, params).log_lr
        key = ("defect" if lr >= params.tau_u
               else "no_defect" if lr <= -params.tau_u else "undetermined")
        out[key] += 1
    return {k: v / reps for k, v in out.items()}


def render_mc(reps: int = 400) -> str:
    """Table 5: monte_carlo over M ∈ {4, 26}, n ∈ {1, 8}, sd ∈ {0.5, 1, 2}, p̄ = 0.8."""
    out = [f"### Table 5. The code's log_lr on simulated binary cells (p̄ = 0.8, {reps} cells "
           "each, tau_u = 2)", "",
           "| M | n | sd | T5 bound | H: defect | H: undet. | H: no defect "
           "| S: defect | S: undet. | S: no defect |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MS:
        for n in (1, 8):
            for s in (0.5, 1.0, 2.0):
                h = monte_carlo(m, n, s, 0.8, "H", reps)
                sh = monte_carlo(m, n, s, 0.8, "S", reps)
                cells = [h["defect"], h["undetermined"], h["no_defect"],
                         sh["defect"], sh["undetermined"], sh["no_defect"]]
                out.append(f"| {m} | {n} | {s:g} | {_fmt(evaluate(m, n, 0.8, s).lower)} | "
                           + " | ".join(f"{c:.3f}" for c in cells) + " |")
    return "\n".join(out) + "\n"


def _fmt(x: float) -> str:
    return "∞" if math.isinf(x) else f"{x:.3f}"


def render() -> str:
    """The note's §6 tables, as Markdown."""
    head = "| M | n | p̄ | " + " | ".join(f"sd {s:g}" for s in SPREADS) + " |"
    rule = "|---|---|---|" + "---|" * len(SPREADS)
    out = ["### Table 1. T5 lower bound on the minimax error (Hellinger form)", "", head, rule]
    for m in MS:
        for n in NS:
            for p in P_BARS:
                cells = [_fmt(evaluate(m, n, p, s).lower) for s in SPREADS]
                out.append(f"| {m} | {n} | {p:.2f} | " + " | ".join(cells) + " |")
    out += ["", "### Table 2. Exact two-point bound ½(1 − TV), M = 4", "", head, rule]
    for n in NS:
        for p in P_BARS:
            cells = [_fmt(evaluate(4, n, p, s, exact=True).exact or 0.0) for s in SPREADS]
            out.append(f"| 4 | {n} | {p:.2f} | " + " | ".join(cells) + " |")
    out += ["", "### Table 3. Per-reading exponent h₁* and contrast Δ_c (n = 1)", "",
            "| M | p̄ | quantity | " + " | ".join(f"sd {s:g}" for s in SPREADS) + " |",
            "|---|---|---|" + "---|" * len(SPREADS)]
    for m in MS:
        for p in P_BARS:
            rows = [evaluate(m, 1, p, s) for s in SPREADS]
            out.append(f"| {m} | {p:.2f} | h₁* | "
                       + " | ".join(f"{r.h_star:.4f}" for r in rows) + " |")
            out.append(f"| {m} | {p:.2f} | Δ_c | "
                       + " | ".join(f"{r.delta_c:.3f}" for r in rows) + " |")
    out += ["", f"### Table 4. Smallest sd(α) at which the bound falls to {TARGET}", "",
            "| M | n | closed form (any p̄) | " + " | ".join(f"p̄ = {p:.2f}" for p in P_BARS)
            + " |", "|---|---|---|" + "---|" * len(P_BARS)]
    for m in MS:
        for n in NS:
            cells = [_fmt(smallest_spread(m, n, p)) for p in P_BARS]
            out.append(f"| {m} | {n} | {_fmt(necessary_spread(m, n))} | "
                       + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    print(render(), end="")
    if "--mc" in args:
        print()
        print(render_mc(), end="")


if __name__ == "__main__":
    main()
