"""The v4 per-cell model (design v4 §3, §5.1; ruling T1-L): per-cell difficulty and an
attractor error, beside v3 (specdx/infer/anchored.py), which stays unchanged.

At cell c a reading of model m falls in A_c with probability p_m = σ(α_m − β_c), uniformly
over A_c, where β_c ~ N(β_g, cell_sd²) is the cell's own difficulty. Otherwise it is an
error. For A_c ≠ ∅ the error on the slots outside A_c (the other values and OTHER) is
(1 − λ)·u + λ·δ_e: the flat distribution u on those slots, plus an attractor e (a value,
never OTHER) with weight λ ~ Beta(lambda_a, lambda_b). For A_c = ∅ every reading comes from
v3's flat pool ε_c, pulled towards u_c by κ pseudo-counts, with no attractor.

The posterior over A_c uses a profile per candidate set: maximise over β_c and, for A ≠ ∅,
over (λ, e), or, for ∅, over ε, under their priors, then combine with v3's prior over sets.
The undetermined statistic log_lr is the prior-free profile log-likelihood ratio (§5.1).

Maximisation. Given A the data log-likelihood separates into a β_c part, which depends on
the readings inside and outside A per model, and a (λ, e) part, which depends only on the
error counts per slot. Each part is maximised exactly on its own:
  - β_c: the β part is concave (a logistic log-likelihood, plus a Gaussian log density) with
    a decreasing derivative, so a bracketed Newton iteration finds its root, vectorised
    over all candidate sets at once;
  - λ: after multiplying by λ(1 − λ)(1 + qλ) > 0, the derivative's root is a quadratic's, so
    the maximum is the best of λ = 0, λ = 1 and the quadratic's roots in (0, 1), in closed
    form;
  - e: for fixed λ, moving one error reading from another slot onto e changes the log-
    likelihood by log((1 + qλ)/(1 − λ)) ≥ 0, so the best attractor is the value with the
    most error readings (ties to the earlier slot).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from specdx.infer.anchored import K_MAX, KAPPA, OTHER, SETS, CellPosterior, set_log_prior
from specdx.infer.cells import Cell

# Ruling T1-L′: β_c is maximised over the box [min α − BOX_MARGIN, max α + BOX_MARGIN], α over
# the raters present at the cell, in log_lr and in the posterior alike. The box does not
# depend on β_g, so neither does log_lr (T5: a unanimous deviation is undecidable whatever
# β_g is).
BOX_MARGIN = 10.0
_STEP_TOL = 1e-10
_MAX_ITER = 200


@dataclass(frozen=True)
class V4Params:
    """v3's prior over sets (pi, s_*), the difficulty priors β_g ~ N(0, beta_sd²) and
    β_c ~ N(β_g, cell_sd²), the ability prior, λ ~ Beta(lambda_a, lambda_b), v3's typing
    thresholds, and v4's: tau_u (nats, on |log_lr|) and tau_lambda (the shared-error λ).
    lambda_a, lambda_b ≥ 1 so that λ's posterior mode exists."""

    pi: float = 0.2
    s_mis: float = 0.4
    s_amb: float = 0.3
    s_empty: float = 0.3
    beta_sd: float = 1.0
    cell_sd: float = 1.0
    mu_alpha: float = 1.0
    sigma_alpha: float = 1.0
    lambda_a: float = 1.0
    lambda_b: float = 3.0
    tau_h: float = 0.5
    tau_a: float = 0.3
    tau_w: float = 0.9
    tau_u: float = 2.0
    tau_lambda: float = 0.5

    def __post_init__(self) -> None:
        if self.lambda_a < 1 or self.lambda_b < 1:
            raise ValueError("lambda_a and lambda_b must be at least 1: below 1 the Beta "
                             "density is unbounded and λ has no posterior mode")
        if self.cell_sd <= 0:
            raise ValueError("cell_sd must be positive")


@dataclass(frozen=True)
class SetFit:
    """One candidate set's profiled log posterior: the data log-likelihood plus the log
    densities of β_c under N(β_g, cell_sd²), of λ under its Beta prior (A ≠ ∅) or of ε
    under the flat pool's Dirichlet (∅), at their joint maximum (β_c within BOX_MARGIN of
    the raters' abilities, ruling T1-L′). beta, lam, attractor: the maximiser; attractor
    None when λ = 0 (δ_e carries no weight) and always under ∅."""

    loglik: float
    beta: float
    lam: float
    attractor: str | None


@dataclass(frozen=True)
class CellPosteriorV4(CellPosterior):
    """v3's per-cell posterior (sets, forms, map_set, p_defect and p_pair keep their
    meaning), with beta, lam and attractor from the MAP set's SetFit and error the MAP set's
    error distribution over the values and OTHER (zero on the set). log_lr: the prior-free
    profile log-likelihood ratio of §5.1, the best defect form's maximised likelihood minus
    the best no-defect form's ({d_c} on an anchored cell, any single value otherwise)."""

    lam: float
    attractor: str | None
    log_lr: float


def _xlog(n: float, x: float) -> float:
    """n·log x with 0·log 0 = 0 and −inf for n > 0 at x = 0."""
    if n == 0:
        return 0.0
    return n * math.log(x) if x > 0 else -math.inf


def _encode_counts(
    values: tuple[str, ...], counts: Mapping[str, Mapping[str, int]],
    alpha: Mapping[str, float], params: V4Params,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """The models (sorted), table[m, k]: model m's readings in value slot k (slot n = OTHER,
    for every value outside values), and their abilities (μ_α for a model absent from
    alpha)."""
    if not values:
        raise ValueError("a cell needs at least one candidate value")
    if len(values) > K_MAX or len(set(values)) != len(values) or OTHER in values:
        raise ValueError(f"values must be at most {K_MAX} distinct values, OTHER excluded")
    n = len(values)
    slot = {v: k for k, v in enumerate(values)}
    models = tuple(sorted(counts))
    table = np.zeros((len(models), n + 1))
    for i, m in enumerate(models):
        for v, c in counts[m].items():
            if c < 0:
                raise ValueError(f"negative count for {m!r}, {v!r}")
            table[i, slot.get(v, n)] += c
    return models, table, np.array([alpha.get(m, params.mu_alpha) for m in models])


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def _beta_part(inside: np.ndarray, outside: np.ndarray, a: np.ndarray,
               beta: np.ndarray) -> np.ndarray:
    """Σ_m inside·log p_m + outside·log(1 − p_m) per set, at β per set ([S, M] counts)."""
    eta = a[None, :] - beta[:, None]
    with np.errstate(invalid="ignore"):
        return (np.where(inside > 0, inside * _log_sigmoid(eta), 0.0)
                + np.where(outside > 0, outside * _log_sigmoid(-eta), 0.0)).sum(axis=1)


def _box(present: np.ndarray) -> tuple[float, float]:
    """Ruling T1-L′'s β_c box from the abilities of the raters present at the cell; with no
    reading, nothing depends on β_c and the box is ±BOX_MARGIN."""
    if not present.size:
        return -BOX_MARGIN, BOX_MARGIN
    return float(present.min()) - BOX_MARGIN, float(present.max()) + BOX_MARGIN


def _argmax_beta(inside: np.ndarray, total: np.ndarray, a: np.ndarray, centre: float,
                 prec: float, lo: float, hi: float) -> np.ndarray:
    """argmax over β ∈ [lo, hi] of Σ_m inside·log σ(α − β) + (total − inside)·log σ(β − α)
    − prec·(β − centre)²/2, per set. Its derivative D(β) = Σ_m total·p_m − Σ inside −
    prec·(β − centre) decreases in β: the maximiser is lo when D(lo) ≤ 0, hi when D(hi) ≥
    0, and otherwise D's root, found by Newton steps kept inside a shrinking bracket
    (bisection when a step leaves it)."""
    hits = inside.sum(axis=1)

    def slope(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = 1.0 / (1.0 + np.exp(-(a[None, :] - beta[:, None])))
        return ((total * p).sum(axis=1) - hits - prec * (beta - centre),
                (total * p * (1 - p)).sum(axis=1) + prec)

    size = inside.shape[0]
    low, high = np.full(size, lo), np.full(size, hi)
    at_low, at_high = slope(low)[0] <= 0, slope(high)[0] >= 0
    active = ~(at_low | at_high)
    beta = np.full(size, min(max(centre, lo), hi))
    for _ in range(_MAX_ITER):
        if not active.any():
            break
        d, curv = slope(beta)
        low = np.where(active & (d > 0), beta, low)
        high = np.where(active & (d <= 0), beta, high)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(curv > 0, d / curv, np.inf)
        new = beta + step
        new = np.where(((new > low) & (new < high)) | (step == 0), new, (low + high) / 2)
        moved = np.abs(new - beta)
        beta = np.where(active, new, beta)
        active &= (moved >= _STEP_TOL) & (high - low >= _STEP_TOL)
    return np.where(at_low, lo, np.where(at_high, hi, beta))


def _argmax_lambda(n_e: float, rest: float, q: int, a1: float, b1: float) -> float:
    """argmax over λ ∈ [0, 1] of n_e·log(1 + qλ) + (rest + b1)·log(1 − λ) + a1·log λ. The
    maximum is at 0, at 1, or at a stationary point in (0, 1); there the derivative times
    λ(1 − λ)(1 + qλ) > 0 is the quadratic −q(n_e + r + a1)·λ² + (n_e·q + a1·(q − 1) − r)·λ
    + a1 with r = rest + b1. Ties go to the smaller λ."""
    r = rest + b1
    quad, lin, const = -q * (n_e + r + a1), n_e * q + a1 * (q - 1) - r, a1
    roots: list[float] = []
    if quad != 0:
        disc = lin * lin - 4 * quad * const
        if disc >= 0:
            half = -0.5 * (lin + math.copysign(math.sqrt(disc), lin))
            roots.append(half / quad)
            if half != 0:
                roots.append(const / half)
    elif lin != 0:
        roots.append(-const / lin)

    def h(lam: float) -> float:
        return _xlog(n_e, 1 + q * lam) + _xlog(r, 1 - lam) + _xlog(a1, lam)

    candidates = sorted({0.0, 1.0, *(x for x in roots if 0 < x < 1)})
    return max(candidates, key=lambda lam: (h(lam), -lam))


def _lambda_part(error_counts: list[float], n_e: float, lam: float, has_e: bool) -> float:
    """Σ over the slots outside A of N_k·log of the error probability: (1 − λ)/c + λ on e,
    (1 − λ)/c elsewhere, with c slots. Without an attractor the error is flat (λ unused)."""
    c = len(error_counts)
    rest = sum(error_counts) - n_e
    if not has_e:
        return _xlog(sum(error_counts), 1.0 / c)
    return _xlog(n_e, (1 - lam) / c + lam) + _xlog(rest, (1 - lam) / c)


def _log_beta_density(lam: float, params: V4Params) -> float:
    a, b = params.lambda_a, params.lambda_b
    return (_xlog(a - 1, lam) + _xlog(b - 1, 1 - lam)
            - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))


def _data_loglik(table: np.ndarray, a: np.ndarray, s: tuple[int, ...], beta: float,
                 lam: float, e: int | None, eps: np.ndarray | None) -> float:
    """The data log-likelihood of set s (value slots; () is ∅) at β_c = beta, error weight
    lam on attractor slot e (None: none), or, under ∅, the error distribution eps over the
    slots (OTHER last)."""
    per_slot = table.sum(axis=0)
    if not s:
        assert eps is not None
        return sum(_xlog(float(per_slot[k]), float(eps[k])) for k in range(len(per_slot)))
    mask = np.array([k in s for k in range(table.shape[1])])
    inside = table[:, mask].sum(axis=1)
    outside = table[:, ~mask].sum(axis=1)
    out = float(_beta_part(inside[None, :], outside[None, :], a, np.array([beta]))[0])
    out -= float(inside.sum()) * math.log(len(s))
    errs = [float(per_slot[k]) for k in range(table.shape[1]) if not mask[k]]
    if e is None or lam == 0:
        return out + _lambda_part(errs, 0.0, 0.0, False)
    return out + _lambda_part(errs, float(per_slot[e]), lam, True)


def _empty_fit(per_slot: np.ndarray, beta_g: float, params: V4Params) -> tuple[SetFit, float,
                                                                                 np.ndarray]:
    """∅ keeps v3's flat pool: ε = (N + κ·u)/(N + κ), v3's estimate (the posterior mean under
    Dirichlet(κ·u), the mode under Dirichlet(1 + κ·u)), whose log density under Dirichlet(1
    + κ·u) enters the profile. No reading informs β_c, which sits at its prior mode β_g."""
    k = len(per_slot)
    eps = (per_slot + KAPPA / k) / (per_slot.sum() + KAPPA)
    data = sum(_xlog(float(per_slot[i]), float(eps[i])) for i in range(k))
    # Ledger addendum to T1-L: the Dirichlet(1 + κ·u) kernel, v3's estimator.
    dirichlet = (math.lgamma(k + KAPPA) - k * math.lgamma(1 + KAPPA / k)
                 + sum(KAPPA / k * math.log(float(x)) for x in eps))
    beta_prior = -0.5 * math.log(2 * math.pi * params.cell_sd ** 2)
    return SetFit(data + dirichlet + beta_prior, beta_g, 0.0, None), data, eps


def _profile(
    cell: Cell, values: tuple[str, ...], counts: Mapping[str, Mapping[str, int]],
    alpha: Mapping[str, float], beta_g: float, params: V4Params,
) -> tuple[dict[frozenset[str], SetFit], dict[frozenset[str], float], np.ndarray]:
    """Per candidate set: its SetFit (the profiled log posterior) and its prior-free profile
    log-likelihood (β_c in the box of T1-L′, λ ∈ [0, 1], e free; ∅ at the flat pool's ε,
    which is the Dirichlet's part of what ∅ means and so stays); and ∅'s ε over the slots.
    A set whose outside holds only OTHER has no attractor and λ does not enter its
    likelihood; its profile still takes λ's prior at its mode, as for any set, and reports
    lam = 0 (the attractor's weight, with no attractor)."""
    _, table, a = _encode_counts(values, counts, alpha, params)
    n = len(values)
    per_slot = table.sum(axis=0)
    total = table.sum(axis=1)
    sets = [s for s in SETS if all(k < n for k in s)]
    full = [s for s in sets if s]
    member = np.array([[k in s for k in range(n + 1)] for s in full], dtype=float)
    inside = member @ table.T  # [S, M]
    outside = total[None, :] - inside
    prec = 1.0 / params.cell_sd ** 2
    lo, hi = _box(a[total > 0])
    beta_map = _argmax_beta(inside, total, a, beta_g, prec, lo, hi)
    beta_box = _argmax_beta(inside, total, a, beta_g, 0.0, lo, hi)
    data_map = _beta_part(inside, outside, a, beta_map)
    data_box = _beta_part(inside, outside, a, beta_box)
    beta_prior = (-(beta_map - beta_g) ** 2 * prec / 2
                  - 0.5 * math.log(2 * math.pi * params.cell_sd ** 2))
    fits: dict[frozenset[str], SetFit] = {}
    likes: dict[frozenset[str], float] = {}
    for i, s in enumerate(full):
        hits = float(inside[i].sum()) * math.log(len(s))
        errs = [float(per_slot[k]) for k in range(n + 1) if k not in s]
        choices = [k for k in range(n) if k not in s]
        e = max(choices, key=lambda k: (per_slot[k], -k)) if choices else None
        n_e = float(per_slot[e]) if e is not None else 0.0
        rest = sum(errs) - n_e if e is not None else 0.0
        q = len(choices)
        lam = _argmax_lambda(n_e, rest, q, params.lambda_a - 1, params.lambda_b - 1)
        lam_free = _argmax_lambda(n_e, rest, q, 0.0, 0.0)
        posterior = (float(data_map[i]) - hits + float(beta_prior[i])
                     + _lambda_part(errs, n_e, lam, e is not None)
                     + _log_beta_density(lam, params))
        key = frozenset(values[k] for k in s)
        live = e is not None and lam > 0
        fits[key] = SetFit(posterior, float(beta_map[i]), lam if live else 0.0,
                           values[e] if live and e is not None else None)
        likes[key] = (float(data_box[i]) - hits
                      + _lambda_part(errs, n_e, lam_free, e is not None))
    empty, data, eps = _empty_fit(per_slot, beta_g, params)
    fits[frozenset()] = empty
    likes[frozenset()] = data  # ledger addendum to T1-L: ∅ enters log_lr at its pool ε
    return fits, likes, eps


def profile_sets(
    cell: Cell, values: tuple[str, ...], counts: Mapping[str, Mapping[str, int]],
    alpha: Mapping[str, float], beta_g: float, params: V4Params,
) -> dict[frozenset[str], SetFit]:
    """Each candidate set's SetFit, keyed by its values (∅ as frozenset())."""
    return _profile(cell, values, counts, alpha, beta_g, params)[0]


def profile_cell(
    cell: Cell, values: tuple[str, ...], counts: Mapping[str, Mapping[str, int]],
    alpha: Mapping[str, float], beta_g: float, params: V4Params,
) -> CellPosteriorV4:
    """The posterior over A_c from each set's profiled log posterior and v3's prior over
    sets, with the MAP set's β_c, λ, attractor and error, and §5.1's prior-free log_lr.
    counts[model][value]: the cell's readings, unweighted; values outside values are OTHER.
    A model absent from alpha reads at μ_α."""
    n = len(values)
    anchored = cell.anchor is not None
    if anchored and (not values or values[0] != cell.anchor):
        raise ValueError("an anchored cell's values must start with its anchor")
    fits, likes, eps = _profile(cell, values, counts, alpha, beta_g, params)
    prior = set_log_prior(n, anchored, params)  # V4Params carries v3's pi and s_* fields
    keys = [s for s in SETS if all(k < n for k in s)]
    names = [frozenset(values[k] for k in s) for s in keys]
    logpost = np.array([prior[SETS.index(s)] + fits[name].loglik
                        for s, name in zip(keys, names, strict=True)])
    with np.errstate(invalid="ignore"):
        weights = np.exp(logpost - logpost.max())
    weights /= weights.sum()
    sets = {name: float(w) for name, w in zip(names, weights, strict=True)}
    clean = ([frozenset({values[0]})] if anchored
             else [name for name in names if len(name) == 1])
    log_lr = (max(likes[name] for name in names if name not in clean)
              - max(likes[name] for name in clean))
    base = CellPosterior(cell.id, cell.anchor, values, sets, 0.0, {})
    best = fits[base.map_set]
    slots = (*values, OTHER)
    if base.map_set:
        out = [v for v in slots if v not in base.map_set]
        error = {v: 0.0 for v in slots}
        for v in out:
            error[v] = (1 - best.lam) / len(out) + (best.lam if v == best.attractor else 0.0)
    else:
        error = {v: float(eps[k]) for k, v in enumerate(slots)}
    return CellPosteriorV4(cell.id, cell.anchor, values, sets, best.beta, error, best.lam,
                           best.attractor, float(log_lr))
