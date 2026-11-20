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

The outer fit (fit_v4, §3.3–§3.5) estimates α and the group difficulties β_g by v3's
multi-start EM around the per-cell profiles; see fit_v4 and _m_step.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from specdx.infer.anchored import (
    EM_MAX_ITER,
    EM_TOL,
    K_MAX,
    KAPPA,
    OTHER,
    SETS,
    STARTS,
    CellPosterior,
    StartFit,
    V3Params,
    _encode,
    _t3_alpha,
    set_log_prior,
)
from specdx.infer.cells import Cell, Reading

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
    the best no-defect form's ({d_c} on an anchored cell, any single value otherwise).
    set_fits: every candidate set's SetFit (∅ as frozenset()), so typing can read the {d_c}
    set's λ and attractor when it is not the MAP set (ruling T3-N)."""

    lam: float
    attractor: str | None
    log_lr: float
    set_fits: dict[frozenset[str], SetFit] = field(default_factory=dict)


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


def _argmax_beta(inside: np.ndarray, total: np.ndarray, a: np.ndarray,
                 centre: float | np.ndarray, prec: float | np.ndarray,
                 lo: float | np.ndarray, hi: float | np.ndarray) -> np.ndarray:
    """argmax over β ∈ [lo, hi] of Σ_m inside·log σ(α − β) + (total − inside)·log σ(β − α)
    − prec·(β − centre)²/2, per set (per row; centre, prec, lo and hi may be one per row).
    Its derivative D(β) = Σ_m total·p_m − Σ inside − prec·(β − centre) decreases in β: the
    maximiser is lo when D(lo) ≤ 0, hi when D(hi) ≥ 0, and otherwise D's root, found by
    Newton steps kept inside a shrinking bracket (bisection when a step leaves it)."""
    hits = inside.sum(axis=1)

    def slope(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = 1.0 / (1.0 + np.exp(-(a[None, :] - beta[:, None])))
        return ((total * p).sum(axis=1) - hits - prec * (beta - centre),
                (total * p * (1 - p)).sum(axis=1) + prec)

    size = inside.shape[0]
    low, high = np.full(size, lo), np.full(size, hi)
    at_low, at_high = slope(low)[0] <= 0, slope(high)[0] >= 0
    active = ~(at_low | at_high)
    beta = np.full(size, np.clip(centre, lo, hi), dtype=float)
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
    return _posterior(cell, values, counts, alpha, beta_g, params)[0]


def _posterior(
    cell: Cell, values: tuple[str, ...], counts: Mapping[str, Mapping[str, int]],
    alpha: Mapping[str, float], beta_g: float, params: V4Params,
) -> tuple[CellPosteriorV4, float]:
    """profile_cell's posterior and the cell's log marginal, log Σ_A P(A)·exp(SetFit.loglik):
    its profiled log posterior with A summed out, which the outer fit's starts compare."""
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
    top = float(logpost.max())
    with np.errstate(invalid="ignore"):
        weights = np.exp(logpost - top)
    marginal = top + math.log(float(weights.sum()))
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
                           best.attractor, float(log_lr), fits), marginal


# --- The outer fit (design v4 §3.3–§3.5; ruling T2-U′: every cell enters the updates) ---

_NEWTON_TOL = 1e-10
_NEWTON_MAX = 50


@dataclass
class AnchoredFitV4:
    """v3's AnchoredFit fields with their meanings (alpha, theta = σ(α), family_effect,
    cells, weights, converged, iterations, params, start, starts), with v4's per-cell
    posteriors. weights: w_b per run key, the mean faithful probability over the cells the
    behaviour read, as in v3. starts: each start's StartFit, its score the composite
    profiled log posterior (_score)."""

    alpha: dict[str, float]
    theta: dict[str, float]
    family_effect: dict[str, float]
    cells: dict[str, CellPosteriorV4]
    weights: dict[str, float]
    converged: bool
    iterations: int
    params: V4Params
    start: str = "t3"
    starts: dict[str, StartFit] = field(default_factory=dict)


@dataclass(frozen=True)
class _Pairs:
    """The M-step's data, one row j per (cell c, non-empty candidate set A).
    inside[j, m], total[j, m]: model m's composite-weighted readings at c that fall in A, and
    all of them; weight: post(A) at c; prec: w̄_c/cell_sd², with w̄_c the mean composite
    weight of c's readings; group: c's difficulty group; lo, hi: c's β_c box (T1-L′)."""

    inside: np.ndarray
    total: np.ndarray
    weight: np.ndarray
    prec: np.ndarray
    group: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def _profiled_beta(pairs: _Pairs, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """β*_j: each row's β_c maximising its composite term at (α, β_g) (see _m_step)."""
    if not pairs.weight.size:
        return np.zeros(0)
    return _argmax_beta(pairs.inside, pairs.total, alpha, beta[pairs.group], pairs.prec,
                        pairs.lo, pairs.hi)


def _q_objective(pairs: _Pairs, alpha: np.ndarray, beta: np.ndarray,
                 params: V4Params) -> float:
    """The M-step's objective Q(α, β_g) (see _m_step), up to constants."""
    prior = (float(((alpha - params.mu_alpha) ** 2).sum()) / (2 * params.sigma_alpha ** 2)
             + float((beta ** 2).sum()) / (2 * params.beta_sd ** 2))
    if not pairs.weight.size:
        return -prior
    star = _profiled_beta(pairs, alpha, beta)
    data = _beta_part(pairs.inside, pairs.total - pairs.inside, alpha, star)
    shrink = pairs.prec * (star - beta[pairs.group]) ** 2 / 2
    return float((pairs.weight * (data - shrink)).sum()) - prior


def _m_step(pairs: _Pairs, alpha: np.ndarray, beta: np.ndarray,
            params: V4Params) -> tuple[np.ndarray, np.ndarray]:
    """The α and β_g update: the exact maximiser of

        Q(α, β_g) = Σ_j ω_j·max_{β_c ∈ box_j} h_j(α, β_c, β_g) + log N(α; μ_α, σ_α²)
                    + log N(β_g; 0, beta_sd²),
        h_j = Σ_m [in_jm·log σ(α_m − β_c) + (tot_jm − in_jm)·log σ(β_c − α_m)]
              − P_j·(β_c − β_g)²/2,

    with ω_j = post(A) and P_j = w̄_c/cell_sd² (_Pairs), the box fixed at the E-step's α.
    When every reading of a cell weighs the same, Q is, up to a constant, the EM minoriser
    of _score's objective at the current point (Jensen over A with post(A) held fixed, then
    β_c re-profiled, which only raises it), so an EM iteration cannot lower that score;
    otherwise it is the same step on the composite objective. A reading is faithful
    exactly when it lies in A (errors live outside A), so the faithful probability r =
    Σ_A post(A)·1[y ∈ A] enters as these posterior-weighted successes, and under ∅ neither
    α nor β_g enters the likelihood, so ∅ contributes no row. Each h_j is jointly concave
    in (α, β_c, β_g), so its partial maximum over β_c in a box is concave in (α, β_g) and
    so is Q: Newton steps with step halving reach the unique maximum.

    The gradient follows by the envelope theorem at β*_j: ∂Q/∂α_m = Σ_j ω_j·(in_jm −
    tot_jm·p_jm) − (α_m − μ_α)/σ_α² and ∂Q/∂β_g = Σ_{j∈g} ω_j·P_j·(β*_j − β_g) − β_g/beta_sd².
    The Hessian profiles β_c out: with W_jm = tot_jm·p_jm(1 − p_jm) and D_j = Σ_m W_jm + P_j,
    row j adds ω_j·[−diag(W_j, P_j) + u_j·u_jᵀ/D_j], u_j = (W_j, P_j), when β*_j is inside its
    box (and only the diagonal when the box binds). The β_g block is diagonal, so the step is
    solved through its Schur complement, as in v3. At the maximum ∂Q/∂β_g = 0 reads β_g =
    Σ_{j∈g} ω_j·P_j·β*_j / (Σ_{j∈g} ω_j·P_j + 1/beta_sd²): the group's cell difficulties,
    shrunk under the prior."""
    groups = beta.size
    for _ in range(_NEWTON_MAX):
        star = _profiled_beta(pairs, alpha, beta)
        w = pairs.weight
        p = 1.0 / (1.0 + np.exp(-(alpha[None, :] - star[:, None])))
        info = pairs.total * p * (1 - p)
        free = (star > pairs.lo) & (star < pairs.hi)
        coup = np.where(free, w / (info.sum(axis=1) + pairs.prec), 0.0)
        g_a = ((w[:, None] * (pairs.inside - pairs.total * p)).sum(axis=0)
               - (alpha - params.mu_alpha) / params.sigma_alpha ** 2)
        g_b = (np.bincount(pairs.group, w * pairs.prec * (star - beta[pairs.group]), groups)
               - beta / params.beta_sd ** 2)
        h_aa = (-np.diag((w[:, None] * info).sum(axis=0)) + (info * coup[:, None]).T @ info
                - np.eye(alpha.size) / params.sigma_alpha ** 2)
        h_ba = np.zeros((groups, alpha.size))
        np.add.at(h_ba, pairs.group, info * (pairs.prec * coup)[:, None])
        h_bb = (np.bincount(pairs.group, -w * pairs.prec + coup * pairs.prec ** 2, groups)
                - 1 / params.beta_sd ** 2)
        h_ab = h_ba.T
        schur = h_aa - (h_ab / h_bb) @ h_ba
        d_a = np.linalg.solve(schur, -g_a + (h_ab / h_bb) @ g_b)
        d_b = (-g_b - h_ba @ d_a) / h_bb
        before = _q_objective(pairs, alpha, beta, params)
        t = 1.0
        while t > 1e-8 and _q_objective(pairs, alpha + t * d_a, beta + t * d_b,
                                        params) < before - 1e-12:
            t /= 2
        alpha, beta = alpha + t * d_a, beta + t * d_b
        if max(float(np.max(np.abs(t * d_a), initial=0.0)),
               float(np.max(np.abs(t * d_b), initial=0.0))) < _NEWTON_TOL:
            break
    return alpha, beta


@dataclass
class _Data:
    """The fit's fixed inputs. Per cell: its values (v3's candidate values), its readings'
    unweighted counts (profile_cell's input), which models read it, its difficulty group and
    its mean composite weight. Per (cell, non-empty set) row: the composite-weighted counts
    inside the set and in all, and the set's key. reads: each reading as (run key, cell,
    value), for the behaviour weights."""

    cells: list[Cell]
    names: list[str]
    values: list[tuple[str, ...]]
    counts: list[dict[str, dict[str, int]]]
    present: np.ndarray
    group: np.ndarray
    mean_weight: np.ndarray
    row_cell: np.ndarray
    row_key: list[frozenset[str]]
    inside: np.ndarray
    total: np.ndarray
    reads: list[tuple[str, int, str]]


def _v3_view(params: V4Params) -> V3Params:
    """v3's parameters inside V4Params: what _encode (the prior over sets) and _t3_alpha (μ_α)
    read."""
    return V3Params(params.pi, params.s_mis, params.s_amb, params.s_empty, params.beta_sd,
                    params.mu_alpha, params.sigma_alpha, params.tau_h, params.tau_a,
                    params.tau_w)


def _prepare(used: list[Cell], by_cell: Mapping[str, Sequence[Reading]], names: list[str],
             values: list[tuple[str, ...]], composite: bool) -> _Data:
    """§3.3: a reading's act group is (run key, scenario); in a group of k readings each
    weighs 1/k, in the α and β_g update only (composite=False: every reading weighs 1)."""
    size = Counter((r.run_key, r.scenario) for c in used for r in by_cell[c.id])
    index = {m: i for i, m in enumerate(names)}
    group_names = sorted({c.group for c in used})
    counts: list[dict[str, dict[str, int]]] = []
    present = np.zeros((len(used), len(names)), dtype=bool)
    mean_weight = np.ones(len(used))
    row_cell: list[int] = []
    row_key: list[frozenset[str]] = []
    inside: list[np.ndarray] = []
    total: list[np.ndarray] = []
    reads: list[tuple[str, int, str]] = []
    for c, cell in enumerate(used):
        v = values[c]
        slot = {value: k for k, value in enumerate(v)}
        table = np.zeros((len(names), len(v) + 1))
        mine: dict[str, dict[str, int]] = {}
        weights = []
        for r in by_cell[cell.id]:
            w = 1.0 / size[(r.run_key, r.scenario)] if composite else 1.0
            weights.append(w)
            table[index[r.model], slot.get(r.value, len(v))] += w
            row = mine.setdefault(r.model, {})
            row[r.value] = row.get(r.value, 0) + 1
            present[c, index[r.model]] = True
            reads.append((r.run_key, c, r.value))
        counts.append(mine)
        mean_weight[c] = float(np.mean(weights))
        for s in SETS:
            if s and all(k < len(v) for k in s):
                row_cell.append(c)
                row_key.append(frozenset(v[k] for k in s))
                inside.append(table[:, list(s)].sum(axis=1))
                total.append(table.sum(axis=1))
    m = len(names)
    return _Data(
        used, names, values, counts, present,
        np.array([group_names.index(c.group) for c in used], dtype=int), mean_weight,
        np.array(row_cell, dtype=int), row_key,
        np.array(inside).reshape(-1, m), np.array(total).reshape(-1, m), reads)


def _e_step(data: _Data, alpha: np.ndarray, beta: np.ndarray, params: V4Params,
            ) -> tuple[list[CellPosteriorV4], np.ndarray]:
    """Every cell's profiled posterior and log marginal at (α, β_g), its readings
    unweighted."""
    abilities = dict(zip(data.names, alpha.tolist(), strict=True))
    posts: list[CellPosteriorV4] = []
    marginals = np.zeros(len(data.cells))
    for c, cell in enumerate(data.cells):
        post, marginals[c] = _posterior(cell, data.values[c], data.counts[c], abilities,
                                        float(beta[data.group[c]]), params)
        posts.append(post)
    return posts, marginals


def _pairs(data: _Data, posts: Sequence[CellPosteriorV4], alpha: np.ndarray,
           params: V4Params) -> _Pairs:
    """The M-step's rows: every non-empty candidate set of every cell, undetermined cells
    included (ruling T2-U′), weighted by its posterior mass (rows of zero mass dropped),
    with the cell's β_c box at α."""
    weight = np.array([posts[c].sets[key]
                       for c, key in zip(data.row_cell.tolist(), data.row_key, strict=True)])
    boxes = np.array([_box(alpha[data.present[c]]) for c in range(len(data.cells))])
    keep = weight > 0
    rows = data.row_cell[keep]
    return _Pairs(data.inside[keep], data.total[keep], weight[keep],
                  data.mean_weight[rows] / params.cell_sd ** 2, data.group[rows],
                  boxes[rows, 0], boxes[rows, 1])


def _score(data: _Data, marginals: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
           params: V4Params) -> float:
    """The composite profiled log posterior the starts are compared on: Σ_c w̄_c·log Σ_A
    P(A)·exp(SetFit.loglik) over every cell, plus the α and β_g log priors (constants
    dropped, as in v3). With every act group of size one it is the profiled log posterior."""
    return (float((data.mean_weight * marginals).sum())
            - float(((alpha - params.mu_alpha) ** 2).sum()) / (2 * params.sigma_alpha ** 2)
            - float((beta ** 2).sum()) / (2 * params.beta_sd ** 2))


def _em(data: _Data, alpha: np.ndarray, params: V4Params, tol: float, max_iter: int,
        ) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """EM from one start, every β_g = 0: α, β_g, whether the largest change in α and β_g fell
    below tol, and the iterations used. Every cell enters every update (ruling T2-U′:
    leaving out cells by their observed |log_lr| selects on the outcome and widened α)."""
    beta = np.zeros(int(data.group.max()) + 1)
    for iterations in range(1, max_iter + 1):
        posts, _ = _e_step(data, alpha, beta, params)
        new_alpha, new_beta = _m_step(_pairs(data, posts, alpha, params), alpha, beta, params)
        moved = max(float(np.max(np.abs(new_alpha - alpha), initial=0.0)),
                    float(np.max(np.abs(new_beta - beta), initial=0.0)))
        alpha, beta = new_alpha, new_beta
        if moved < tol:
            return alpha, beta, True, iterations
    return alpha, beta, False, max_iter


def _setup(
    cells: Mapping[str, Cell], readings: Sequence[Reading], models: Mapping[str, str],
    params: V4Params, composite: bool,
) -> tuple[dict[str, str], _Data | None, np.ndarray]:
    """model → family (models, completed and checked against the readings), the fit's
    inputs (None without readings) and T3's median start (v3's _t3_alpha), over the
    models in sorted order."""
    family = dict(models)
    by_cell: dict[str, list[Reading]] = {}
    for r in sorted(readings, key=lambda x: (x.cell, x.run_key)):
        if family.setdefault(r.model, r.family) != r.family:
            raise ValueError(f"model {r.model!r} has conflicting families: "
                             f"{family[r.model]!r} and {r.family!r}")
        by_cell.setdefault(r.cell, []).append(r)
    names = sorted(family)
    used = [cells[c] for c in sorted(by_cell)]
    if not used:
        return family, None, np.full(len(names), params.mu_alpha)
    batch = _encode(used, by_cell, names, _v3_view(params))
    data = _prepare(used, by_cell, names, batch.values, composite)
    return family, data, _t3_alpha(batch, _v3_view(params))


def fit_v4(
    cells: Mapping[str, Cell], readings: Sequence[Reading], *, models: Mapping[str, str],
    params: V4Params, tol: float = EM_TOL, max_iter: int = EM_MAX_ITER,
    composite: bool = True,
) -> AnchoredFitV4:
    """α and β_g by v3's multi-start EM (§3.5) around v4's per-cell profiles, deterministic.
    Each iteration profiles every cell at the current (α, β_g), its readings unweighted
    (§3.3), then updates α and β_g together (_m_step) on every cell, undetermined ones
    included (ruling T2-U′), each reading weighted 1/k in its act group (run key,
    scenario) of k readings. EM runs from T3's median start and from each common α in
    STARTS; the fixed point with the best _score is kept (ties to the earlier start).
    models: model → family for the whole population; a model without readings keeps
    α = μ_α. Cells without readings are left out. composite=False weighs every reading 1
    (the no-composite-weights ablation, §6.4)."""
    family, data, t3 = _setup(cells, readings, models, params, composite)
    names = sorted(family)
    alpha = np.full(len(names), params.mu_alpha)
    converged, iterations, start = True, 0, "t3"
    scores: dict[str, StartFit] = {}
    posts: list[CellPosteriorV4] = []
    reads: list[tuple[str, int, str]] = []
    if data is not None:
        reads = data.reads
        starts = [("t3", t3)] + [(f"alpha={a}", np.full(len(names), a)) for a in STARTS]
        best = -np.inf
        for label, alpha0 in starts:
            a, b, ok, n = _em(data, alpha0, params, tol, max_iter)
            fitted, marginals = _e_step(data, a, b, params)
            score = _score(data, marginals, a, b, params)
            scores[label] = StartFit(score, ok, n)
            if not posts or score > best:
                best, start = score, label
                alpha, converged, iterations = a, ok, n
                posts = fitted
    sums: dict[str, list[float]] = {}
    for key, c, value in reads:
        sets = posts[c].sets
        sums.setdefault(key, []).append(sum(p for s, p in sets.items() if value in s))
    alphas = {name: float(alpha[i]) for i, name in enumerate(names)}
    mean = sum(alphas.values()) / len(alphas) if alphas else 0.0
    effect = {f: sum(alphas[m] for m in names if family[m] == f)
              / sum(1 for m in names if family[m] == f) - mean
              for f in sorted(set(family.values()))}
    return AnchoredFitV4(
        alphas, {m: float(1 / (1 + np.exp(-a))) for m, a in alphas.items()}, effect,
        {post.cell: post for post in posts},
        {k: sum(v) / len(v) for k, v in sorted(sums.items())}, converged, iterations, params,
        start, scores)
