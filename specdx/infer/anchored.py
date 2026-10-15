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
from dataclasses import dataclass
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
