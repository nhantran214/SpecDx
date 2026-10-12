"""S2, paraphrase sensitivity (spec v2 §3.6): does a competent model's class depend on
which paraphrase of the Tell text it read?

The statistic is the plug-in conditional mutual information I(K; V | M) between class K
and Tell variant V, stratified by model M. Its null distribution comes from permuting V
within each model's stratum, which keeps every model's variant counts and class counts.
Items of one specification are corrected together with Benjamini–Hochberg.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from specdx.infer.items import ItemView

COMPETENT = 0.5  # spec v2 §3.10
PERMUTATIONS = 2000
MIN_BEHAVIOURS = 6
BH_Q = 0.1
# Permuted statistics within this distance of the observed one count as reaching it.
_TIE = 1e-12


@dataclass(frozen=True)
class Reading:
    """One acted behaviour of a competent model at one item."""

    model: str
    variant: int
    klass: str


def eligible(readings: Sequence[Reading], min_behaviours: int = MIN_BEHAVIOURS) -> bool:
    return (len(readings) >= min_behaviours and len({r.klass for r in readings}) >= 2
            and len({r.variant for r in readings}) >= 2)


def _codes(values: Sequence) -> np.ndarray:
    index = {v: i for i, v in enumerate(sorted(set(values)))}
    return np.array([index[v] for v in values], dtype=np.int64)


def _mutual_information(variants: np.ndarray, klass: np.ndarray) -> np.ndarray:
    """I(K; V) for each row of variants (rows: permutations), in nats."""
    rows, n = variants.shape
    nv, nk = int(variants.max()) + 1, int(klass.max()) + 1
    cells = variants * nk + klass[None, :]
    counts = np.zeros((rows, nv * nk))
    np.add.at(counts, (np.repeat(np.arange(rows), n), cells.ravel()), 1.0)
    joint = counts.reshape(rows, nv, nk)
    pv = joint.sum(axis=2, keepdims=True)
    pk = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint / n * np.log(joint * n / (pv * pk)), 0.0)
    return terms.sum(axis=(1, 2))


def _seed(seed: int | str, item: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{item}".encode()).digest()[:8], "big")


def permutation_test(
    readings: Sequence[Reading], *, permutations: int = PERMUTATIONS, seed: int | str = 0,
    item: str = "",
) -> tuple[float, float]:
    """(I(K; V | M), p) with p = (1 + #{I_perm ≥ I_obs}) / (R + 1)."""
    # Sort readings canonically so p-value does not depend on arrival order.
    readings = sorted(readings, key=lambda r: (r.model, r.variant, r.klass))
    rng = np.random.default_rng(_seed(seed, item))
    strata: dict[str, list[Reading]] = defaultdict(list)
    for r in readings:
        strata[r.model].append(r)
    total = len(readings)
    observed = 0.0
    permuted = np.zeros(permutations)
    for model in sorted(strata):
        mine = strata[model]
        variants = _codes([r.variant for r in mine])
        klass = _codes([r.klass for r in mine])
        weight = len(mine) / total
        observed += weight * float(_mutual_information(variants[None, :], klass)[0])
        shuffled = rng.permuted(np.tile(variants, (permutations, 1)), axis=1)
        permuted += weight * _mutual_information(shuffled, klass)
    reached = int((permuted >= observed - _TIE).sum())
    return observed, (1 + reached) / (permutations + 1)


def benjamini_hochberg(pvalues: Mapping[str, float], q: float = BH_Q) -> frozenset[str]:
    """The items whose test is significant at false discovery rate q."""
    ranked = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    cut = 0
    for rank, (_, p) in enumerate(ranked, start=1):
        if p * len(ranked) <= rank * q + 1e-12:
            cut = rank
    return frozenset(item for item, _ in ranked[:cut])


def paraphrase_sensitivity(
    views: Iterable[ItemView],
    theta: Mapping[str, float],
    items: Iterable[str],
    *,
    competent: float = COMPETENT,
    permutations: int = PERMUTATIONS,
    min_behaviours: int = MIN_BEHAVIOURS,
    seed: int | str = 0,
) -> dict[str, tuple[float, float]]:
    """(statistic, p) for every eligible item among `items`, from the acted views of models
    with θ ≥ competent."""
    wanted = set(items)
    readings: dict[str, list[Reading]] = defaultdict(list)
    for v in views:
        if v.acted and v.item in wanted and theta.get(v.model, 0.0) >= competent:
            readings[v.item].append(Reading(v.model, v.variant, v.klass))
    return {
        item: permutation_test(readings[item], permutations=permutations, seed=seed, item=item)
        for item in sorted(readings) if eligible(readings[item], min_behaviours)
    }
