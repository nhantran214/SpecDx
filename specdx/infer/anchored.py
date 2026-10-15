"""The anchored cell model (spec v3.1 §3): each cell's latent entailment set A_c.

A reading of model m at cell c falls in A_c with probability p_{m,c} = σ(α_m − β_{g(c)}),
uniformly over A_c, where the difficulty β is shared by the cell's group g(c) (its value
column, or its clause). Otherwise the reading is drawn from the cell's error distribution
ε_c restricted to the values outside A_c, and ε_c is pulled towards the flat pool u_c. When
A_c = ∅ every reading comes from ε_c. θ is accuracy, so error mass never sits on the truth;
and since silence predicts near-flat readings, a sharp split identifies a pair and a sharp
mode a value, from one model (Theorem T2′).

Computation is vectorised over cells: each cell's candidate values occupy value slots
0..K_MAX-1 (slot 0 is d_c on an anchored cell) and slot K_MAX is the OTHER bucket.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from specdx.infer.cells import Cell, Reading

K_MAX = 6  # spec v1 §3.10, kept by v3 §9
OTHER = "__other__"
EM_TOL = 1e-6  # spec v3 §3.4
EM_MAX_ITER = 200
KAPPA = 10.0  # spec v3.1 §3.3: ε_c ~ Dirichlet(κ·u_c), u_c flat over V_c; fixed, not G0's
FORMS: tuple[str, ...] = ("none", "mis", "amb", "empty")
# Every candidate set over the value slots: each single value, each pair, and ∅ last
# (6 + 15 + 1 = 22). OTHER (slot K_MAX) is never in a set.
SETS: tuple[tuple[int, ...], ...] = (
    *((i,) for i in range(K_MAX)), *combinations(range(K_MAX), 2), ())
EMPTY = len(SETS) - 1
_IN = np.array([[k in s for k in range(K_MAX + 1)] for s in SETS])
_SIZE = np.array([len(s) for s in SETS], dtype=float)


# --- G0 frozen parameters: begin ---
PARAMS_FROZEN = False


@dataclass(frozen=True)
class V3Params:
    """PROVISIONAL defaults, not chosen by G0 (spec v3.1 §9). Task 13 of the P2a plan
    replaces this block with the values G0's selection rule picks. Until then every test
    whose outcome depends on a value passes its parameters explicitly."""

    pi: float = 0.1  # π: prior mass of the defect forms on an anchored cell
    s_mis: float = 0.5  # form shares s_mis + s_amb + s_empty = 1
    s_amb: float = 0.25
    s_empty: float = 0.25
    beta_sd: float = 1.0  # τ of §3.3: β_g ~ N(0, τ²) per difficulty group
    mu_alpha: float = 0.0  # α_m ~ N(μ_α, σ_α²)
    sigma_alpha: float = 1.5
    tau_h: float = 0.5  # hidden candidate: P(A_c ≠ {d_c}) ≥ τ_H
    tau_a: float = 0.5  # ambiguity: P(|A_c| = 2) ≥ τ_A
    tau_w: float = 0.8  # weak check: MAP set {x} with posterior ≥ τ_W
# --- G0 frozen parameters: end ---


def candidate_values(
    anchor: str | None, observed: Sequence[str], k_max: int = K_MAX
) -> tuple[str, ...]:
    """§3.1: d_c first when anchored, then the most frequent observed values, ties broken by
    the normalised string, up to k_max in all. Every other value falls in OTHER."""
    counts = Counter(observed)
    ranked = sorted(counts, key=lambda v: (-counts[v], v))
    head = [anchor] if anchor is not None else []
    return tuple([*head, *(v for v in ranked if v != anchor)][:k_max])


def set_log_prior(n: int, anchored: bool, params: V3Params) -> np.ndarray:
    """§3.1's prior over SETS for a cell with n candidate values: log mass, −inf for a set
    that uses an empty slot. Anchored: {d_c} gets 1 − π and each defect form f gets π·s_f,
    uniform over its sets. Unanchored: the single values share 1 − π, the pairs and ∅ share
    π in the ratio s_amb : s_∅. A form with no set (one value has no pair) is dropped and
    the rest renormalised."""
    pairs = n * (n - 1) // 2
    rest = params.s_amb + params.s_empty
    weights = np.zeros(len(SETS))
    for i, s in enumerate(SETS):
        if any(k >= n for k in s):
            continue
        if anchored:
            if s == (0,):
                weights[i] = 1 - params.pi
            elif len(s) == 1:
                weights[i] = params.pi * params.s_mis / (n - 1)
            elif len(s) == 2:
                weights[i] = params.pi * params.s_amb / pairs
            else:
                weights[i] = params.pi * params.s_empty
        elif len(s) == 1:
            weights[i] = (1 - params.pi) / n
        elif rest > 0:
            share = params.s_amb / pairs if len(s) == 2 else params.s_empty
            weights[i] = params.pi * share / rest
    with np.errstate(divide="ignore"):
        return np.log(weights / weights.sum())


@dataclass(frozen=True)
class CellPosterior:
    """values: V_c without OTHER (d_c first when anchored). sets: the posterior over the
    candidate sets, ∅ as frozenset(). error: ε_c over the values and OTHER."""

    cell: str
    anchor: str | None
    values: tuple[str, ...]
    sets: dict[frozenset[str], float]
    beta: float
    error: dict[str, float]

    @property
    def forms(self) -> dict[str, float]:
        """§3.5's form masses: none ({d_c}), mis (another single value), amb (a pair), empty
        (∅). On an unanchored cell every single value counts as none."""
        out = dict.fromkeys(FORMS, 0.0)
        for s, p in self.sets.items():
            if not s:
                out["empty"] += p
            elif len(s) == 2:
                out["amb"] += p
            elif self.anchor is None or s == {self.anchor}:
                out["none"] += p
            else:
                out["mis"] += p
        return out

    @property
    def map_set(self) -> frozenset[str]:
        return max(self.sets, key=lambda s: (self.sets[s], -len(s), sorted(s)))

    @property
    def p_defect(self) -> float:
        """P(A_c ≠ {d_c}); 0 on an unanchored cell, which C does not declare."""
        if self.anchor is None:
            return 0.0
        return 1.0 - self.sets.get(frozenset({self.anchor}), 0.0)

    @property
    def p_pair(self) -> float:
        return self.forms["amb"]


@dataclass
class _Batch:
    """Cells and their readings as arrays. A reading's likelihood depends only on its value
    slot and its model, so the readings enter as counts[c, k, m]: model m's readings of
    cell c that fall in slot k (K_MAX = OTHER). keys: each cell's readings as (run key,
    slot, model index), to give each behaviour its faithful probabilities back. groups: each
    cell's difficulty group (Cell.group), as an index."""

    cells: list[Cell]
    values: list[tuple[str, ...]]
    counts: np.ndarray
    keys: list[list[tuple[str, int, int]]]
    logprior: np.ndarray
    support: np.ndarray
    groups: np.ndarray


def _encode(
    cells: Sequence[Cell], by_cell: Mapping[str, Sequence[Reading]], models: Sequence[str],
    params: V3Params, values: Mapping[str, tuple[str, ...]] | None = None,
) -> _Batch:
    index = {name: i for i, name in enumerate(models)}
    counts = np.zeros((len(cells), K_MAX + 1, len(models)))
    logprior = np.zeros((len(cells), len(SETS)))
    support = np.zeros((len(cells), K_MAX + 1), dtype=bool)
    vals: list[tuple[str, ...]] = []
    keys: list[list[tuple[str, int, int]]] = []
    for c, cell in enumerate(cells):
        mine = by_cell[cell.id]
        v = (values or {}).get(cell.id) or candidate_values(cell.anchor, [r.value for r in mine])
        slot = {value: k for k, value in enumerate(v)}
        own = [(r.run_key, slot.get(r.value, K_MAX), index[r.model]) for r in mine]
        for _, k, m in own:
            counts[c, k, m] += 1
        logprior[c] = set_log_prior(len(v), cell.anchor is not None, params)
        support[c, :len(v)] = True
        support[c, K_MAX] = True
        vals.append(v)
        keys.append(own)
    names = sorted({cell.group for cell in cells})
    groups = np.array([names.index(cell.group) for cell in cells], dtype=int)
    return _Batch(list(cells), vals, counts, keys, logprior, support, groups)


def _likelihoods(batch: _Batch, alpha: np.ndarray, beta: np.ndarray,
                 eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """§3.2 for every cell c, set s, value slot k and model m: the faithful part of
    P(y = k | A = s) and the whole of it."""
    p = 1.0 / (1.0 + np.exp(-(alpha[None, :] - beta[:, None])))
    inside = _IN[None, :, :, None]
    nonempty = (_SIZE > 0)[None, :, None, None]
    faith = np.where(inside & nonempty,
                     p[:, None, None, :] / np.maximum(_SIZE, 1.0)[None, :, None, None], 0.0)
    outside = 1.0 - eps @ _IN.T.astype(float)
    scale = np.where(nonempty, 1.0 - p[:, None, None, :], 1.0)
    err = np.where(inside, 0.0, scale * eps[:, None, :, None] / outside[:, :, None, None])
    return faith, faith + err


def _log_joint(batch: _Batch, lik: np.ndarray) -> np.ndarray:
    """log P(A_c = A) + Σ over the cell's readings of log P(y | A), as [c, A]: −inf where a
    set cannot produce a reading. The E-step and the EM starts' comparison share it."""
    read = batch.counts[:, None, :, :] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = np.where(read, batch.counts[:, None, :, :] * np.log(lik), 0.0).sum(axis=(2, 3))
    return batch.logprior + loglik


def _e_step(
    batch: _Batch, alpha: np.ndarray, beta: np.ndarray, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """§3.4's E-step: the posterior over SETS per cell, and the faithful probability
    r = Σ_A post(A)·P(faithful | y, A) of a reading in slot k by model m, as r[c, k, m]."""
    faith, lik = _likelihoods(batch, alpha, beta, eps)
    logpost = _log_joint(batch, lik)
    with np.errstate(divide="ignore", invalid="ignore"):
        top = logpost.max(axis=1, keepdims=True)
        post = np.exp(logpost - top)
        post /= post.sum(axis=1, keepdims=True)
        share = np.where(lik > 0, faith / lik, 0.0)
    r = (post[:, :, None, None] * share).sum(axis=1)
    return post, r


def _posterior(batch: _Batch, c: int, post: np.ndarray, beta: float,
               eps: np.ndarray) -> CellPosterior:
    v = batch.values[c]
    sets = {frozenset(v[k] for k in s): float(post[c, i])
            for i, s in enumerate(SETS) if all(k < len(v) for k in s)}
    error = {value: float(eps[c, k]) for k, value in enumerate(v)} | {OTHER: float(eps[c, K_MAX])}
    cell = batch.cells[c]
    return CellPosterior(cell.id, cell.anchor, v, sets, beta, error)


def cell_posterior(
    cell: Cell, values: tuple[str, ...], readings: Sequence[Reading], *,
    alpha: Mapping[str, float], beta: float, error: Mapping[str, float], params: V3Params,
) -> CellPosterior:
    """The posterior over A_c at fixed parameters: the E-step for one cell. values: V_c
    (readings outside it are OTHER); error: ε_c over values and OTHER. A model absent from
    alpha reads at the prior mean μ_α."""
    models = sorted({r.model for r in readings} | set(alpha))
    batch = _encode([cell], {cell.id: readings}, models, params, {cell.id: values})
    a = np.array([alpha.get(name, params.mu_alpha) for name in models])
    eps = np.zeros((1, K_MAX + 1))
    for k, value in enumerate(values):
        eps[0, k] = error.get(value, 0.0)
    eps[0, K_MAX] = error.get(OTHER, 0.0)
    eps /= eps.sum()
    post, _ = _e_step(batch, a, np.array([beta]), eps)
    return _posterior(batch, 0, post, beta, eps)


_NEWTON_TOL = 1e-10
_NEWTON_MAX = 50
# EM starts besides T3's median: one common α for every model, on the logit scale. On the
# base data more than half of the anchored cells are read alike by every model and differ
# from d_c (apr case rows no tool writes; golden event actors no interpreter writes), past
# T3's breakdown point, and EM from T3's start alone stops ~200 nats below the MAP.
STARTS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class StartFit:
    """One EM start's outcome: its final log posterior, whether it met the tolerance, and
    the iterations it used (a start that hit max_iter is visible here, not only the best)."""

    score: float
    converged: bool
    iterations: int


@dataclass
class AnchoredFit:
    """§3.5's outputs. alpha and theta = σ(α) per model (θ: the accuracy at a cell of
    median difficulty); family_effect: each family's mean logit θ minus the population's
    (v2 §3.3); cells: the per-cell posteriors; weights: w_b per run key, the mean faithful
    probability over the cells the behaviour read; a behaviour that read no cell is absent,
    and consumers default it to 0. converged is False
    when EM stopped at max_iter (§3.4: reported, never silent). start: the EM start whose
    fixed point has the highest log posterior (t3, or a common α such as "alpha=2.0");
    starts: each start's StartFit (final log posterior, whether it converged, iterations)."""

    alpha: dict[str, float]
    theta: dict[str, float]
    family_effect: dict[str, float]
    cells: dict[str, CellPosterior]
    weights: dict[str, float]
    converged: bool
    iterations: int
    params: V3Params
    start: str = "t3"
    starts: dict[str, StartFit] = field(default_factory=dict)


def _flat(batch: _Batch) -> np.ndarray:
    """u_c, the flat pool over each cell's slots (its candidate values and OTHER). Value
    labels mean nothing across columns, so a flat profile is the only pool beyond the
    column; the strong pull keeps ∅ from fitting a sharp split or mode (v3.1 §3.3)."""
    flat = batch.support.astype(float)
    return flat / flat.sum(axis=1, keepdims=True)


def _t3_alpha(batch: _Batch, params: V3Params) -> np.ndarray:
    """Theorem T3's median estimator: α_m = median over the anchored cells m read of
    logit P̂_{m,c}(d_c), with P̂ = (hits + ½)/(n + 1) so no logit is infinite; μ_α for a
    model that read no anchored cell."""
    anchored = np.array([cell.anchor is not None for cell in batch.cells], dtype=bool)
    n = batch.counts.sum(axis=1)
    share = (batch.counts[:, 0, :] + 0.5) / (n + 1.0)
    out = []
    for m in range(batch.counts.shape[2]):
        mine = anchored & (n[:, m] > 0)
        logits = np.log(share[mine, m] / (1 - share[mine, m]))
        out.append(float(np.median(logits)) if mine.any() else params.mu_alpha)
    return np.array(out)


def _objective(alpha: np.ndarray, beta: np.ndarray, rsum: np.ndarray, trials: np.ndarray,
               params: V3Params) -> float:
    eta = alpha[None, :] - beta[:, None]
    fit = (rsum * eta - trials * np.logaddexp(0.0, eta)).sum()
    prior = (((alpha - params.mu_alpha) ** 2).sum() / (2 * params.sigma_alpha ** 2)
             + (beta ** 2).sum() / (2 * params.beta_sd ** 2))
    return float(fit - prior)


def _newton(rsum: np.ndarray, trials: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
            params: V3Params) -> tuple[np.ndarray, np.ndarray]:
    """§3.4's M-step for α and β: the penalised logistic objective with the faithful
    probabilities as fractional successes. rsum[g, m] sums them over model m's readings of
    the cells of difficulty group g; trials[g, m] counts those readings weighted by each
    cell's q_c = P(A_c ≠ ∅), since under ∅ no reading is a trial of accuracy. Newton steps
    through the Schur complement of the diagonal β block, halved while they would lower the
    objective."""
    for _ in range(_NEWTON_MAX):
        p = 1.0 / (1.0 + np.exp(-(alpha[None, :] - beta[:, None])))
        resid = rsum - trials * p
        w = trials * p * (1 - p)
        g_a = resid.sum(axis=0) - (alpha - params.mu_alpha) / params.sigma_alpha ** 2
        g_b = -resid.sum(axis=1) - beta / params.beta_sd ** 2
        h_aa = -w.sum(axis=0) - 1 / params.sigma_alpha ** 2
        h_bb = -w.sum(axis=1) - 1 / params.beta_sd ** 2
        h_ab = w.T
        schur = np.diag(h_aa) - (h_ab / h_bb) @ h_ab.T
        d_a = np.linalg.solve(schur, -g_a + (h_ab / h_bb) @ g_b)
        d_b = (-g_b - h_ab.T @ d_a) / h_bb
        before = _objective(alpha, beta, rsum, trials, params)
        t = 1.0
        while t > 1e-8 and _objective(alpha + t * d_a, beta + t * d_b, rsum, trials,
                                      params) < before - 1e-12:
            t /= 2
        alpha, beta = alpha + t * d_a, beta + t * d_b
        if max(float(np.max(np.abs(t * d_a), initial=0.0)),
               float(np.max(np.abs(t * d_b), initial=0.0))) < _NEWTON_TOL:
            break
    return alpha, beta


def _error_update(batch: _Batch, post: np.ndarray, r: np.ndarray, eps: np.ndarray,
                  prior: np.ndarray) -> np.ndarray:
    """§3.4's M-step for ε_c, the exact EM step of the truncated error model (§3.2 draws an
    error from ε renormalised over V∖A). Each error reading is a draw from ε after rejecting
    draws that fell inside A, so besides the expected error counts (a reading outside A, or
    any reading under ∅) the complete data hold the expected rejected draws: under A ≠ ∅,
    each of its N_err,A error readings rejects ε(k)/(1 − ε(A)) draws on each slot k in A,
    weighted by post(A). Then ε_c is the mode of the Dirichlet κ·u_c posterior over all
    draws, (counts + rejected + κ·u_c) / (total + κ), which keeps every value's error mass
    positive (the literal mode of Dirichlet(κ·u) is 0 wherever κ·u + count < 1); the support
    mask keeps unused slots at zero and OTHER positive."""
    counts = (batch.counts * (1.0 - r)).sum(axis=2)
    reads = batch.counts.sum(axis=2)
    outside = 1.0 - eps @ _IN.T.astype(float)
    n_err = reads.sum(axis=1, keepdims=True) - reads @ _IN.T.astype(float)
    nonempty = (_SIZE > 0)[None, :]
    reject = np.where(nonempty, post * n_err / outside, 0.0) @ _IN.astype(float)
    out = (counts + eps * reject + KAPPA * prior) * batch.support
    return out / out.sum(axis=1, keepdims=True)


def _log_posterior(batch: _Batch, alpha: np.ndarray, beta: np.ndarray, eps: np.ndarray,
                   prior: np.ndarray, params: V3Params) -> float:
    """log P(readings | α, β, ε) with every A_c summed out, plus the log priors of α, β (one
    per group) and ε (up to constants): what the EM starts are compared on."""
    _, lik = _likelihoods(batch, alpha, beta[batch.groups], eps)
    joint = _log_joint(batch, lik)
    with np.errstate(divide="ignore", invalid="ignore"):
        top = joint.max(axis=1, keepdims=True)
        marginal = float((top[:, 0] + np.log(np.exp(joint - top).sum(axis=1))).sum())
        dirichlet = float(np.where(batch.support, KAPPA * prior * np.log(eps), 0.0).sum())
    return (marginal + dirichlet
            - float(((alpha - params.mu_alpha) ** 2).sum()) / (2 * params.sigma_alpha ** 2)
            - float((beta ** 2).sum()) / (2 * params.beta_sd ** 2))


def _em(batch: _Batch, prior: np.ndarray, alpha: np.ndarray, params: V3Params, tol: float,
        max_iter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """EM from one start, with every β_g = 0 and ε_c = u_c (§3.4): α, β per group, ε,
    whether the largest parameter change fell below tol, and the iterations used."""
    groups = int(batch.groups.max()) + 1
    beta = np.zeros(groups)
    eps = prior.copy()
    reads = batch.counts.sum(axis=1)
    for iterations in range(1, max_iter + 1):
        post, r = _e_step(batch, alpha, beta[batch.groups], eps)
        rsum = np.zeros((groups, len(alpha)))
        trials = np.zeros((groups, len(alpha)))
        np.add.at(rsum, batch.groups, (batch.counts * r).sum(axis=1))
        np.add.at(trials, batch.groups, (1.0 - post[:, EMPTY])[:, None] * reads)
        new_alpha, new_beta = _newton(rsum, trials, alpha, beta, params)
        new_eps = _error_update(batch, post, r, eps, prior)
        moved = max(float(np.max(np.abs(new_alpha - alpha), initial=0.0)),
                    float(np.max(np.abs(new_beta - beta), initial=0.0)),
                    float(np.max(np.abs(new_eps - eps), initial=0.0)))
        alpha, beta, eps = new_alpha, new_beta, new_eps
        if moved < tol:
            return alpha, beta, eps, True, iterations
    return alpha, beta, eps, False, max_iter


def fit_anchored(
    cells: Mapping[str, Cell], readings: Sequence[Reading], *, models: Mapping[str, str],
    params: V3Params, tol: float = EM_TOL, max_iter: int = EM_MAX_ITER,
) -> AnchoredFit:
    """MAP by EM (§3.4), deterministic. EM runs from T3's median start and from each common
    α in STARTS (§3.4's multi-start), and the fixed point with the highest log posterior is
    kept (ties to the earlier start). models: model → family for the whole population; a
    model without readings keeps α = μ_α. Cells without readings are left out."""
    family = dict(models)
    by_cell: dict[str, list[Reading]] = {}
    for r in sorted(readings, key=lambda x: (x.cell, x.run_key)):
        if family.setdefault(r.model, r.family) != r.family:
            raise ValueError(f"model {r.model!r} has conflicting families: "
                             f"{family[r.model]!r} and {r.family!r}")
        by_cell.setdefault(r.cell, []).append(r)
    names = sorted(family)
    used = [cells[c] for c in sorted(by_cell)]
    batch = _encode(used, by_cell, names, params)
    prior = _flat(batch)
    alpha = np.full(len(names), params.mu_alpha)
    beta, eps = np.zeros(len(set(batch.groups.tolist()))), prior
    converged, iterations, start = True, 0, "t3"
    scores: dict[str, StartFit] = {}
    if used:
        starts = [("t3", _t3_alpha(batch, params))] + [
            (f"alpha={a}", np.full(len(names), a)) for a in STARTS]
        best = -np.inf
        for label, alpha0 in starts:
            a, b, e, ok, n = _em(batch, prior, alpha0, params, tol, max_iter)
            score = _log_posterior(batch, a, b, e, prior, params)
            scores[label] = StartFit(score, ok, n)
            if score > best:
                best, start = score, label
                alpha, beta, eps, converged, iterations = a, b, e, ok, n
    cell_beta = beta[batch.groups]
    post, r = _e_step(batch, alpha, cell_beta, eps)
    sums: dict[str, list[float]] = {}
    for c, keys in enumerate(batch.keys):
        for key, k, m in keys:
            sums.setdefault(key, []).append(float(r[c, k, m]))
    alphas = {name: float(alpha[i]) for i, name in enumerate(names)}
    mean = sum(alphas.values()) / len(alphas) if alphas else 0.0
    effect = {f: sum(alphas[m] for m in names if family[m] == f)
              / sum(1 for m in names if family[m] == f) - mean
              for f in sorted(set(family.values()))}
    return AnchoredFit(
        alphas, {m: float(1 / (1 + np.exp(-a))) for m, a in alphas.items()}, effect,
        {cell.id: _posterior(batch, c, post, float(cell_beta[c]), eps)
         for c, cell in enumerate(used)},
        {k: sum(v) / len(v) for k, v in sorted(sums.items())}, converged, iterations, params,
        start, scores)


def posterior_at(fit: AnchoredFit, cell: Cell, readings: Sequence[Reading]) -> CellPosterior:
    """P(A_c | readings) with α, β_{g(c)}, ε_c and V_c held at their fitted values: the
    ddmin oracle's test on re-run readings (§5.3)."""
    fitted = fit.cells[cell.id]
    return cell_posterior(cell, fitted.values, readings, alpha=fit.alpha, beta=fitted.beta,
                          error=fitted.error, params=fit.params)
