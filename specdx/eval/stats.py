"""Paired statistics of the comparison study (v1 §4.7; baselines spec §3.6)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from specdx.eval.matching import match_findings

RESAMPLES, SEED = 10_000, 20261021


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def detected(result, arm: str, tau: float) -> dict[str, bool]:
    """Per scorable mutant the arm reported on, whether a typed finding hit its truth."""
    from specdx.eval.study import _reports

    reports, base = _reports(result, arm, tau, primary=True)
    return {k: bool(match_findings(ds, result.truths[k], base, typed=True)[1])
            for k, ds in reports.items()}


def paired(result, arm_a: str, arm_b: str, tau_a: float, tau_b: float) -> dict[str, float]:
    da, db = detected(result, arm_a, tau_a), detected(result, arm_b, tau_b)
    keys = sorted(set(da) & set(db))
    b = sum(da[k] and not db[k] for k in keys)
    c = sum(db[k] and not da[k] for k in keys)
    n = len(keys) or 1
    return {"b": b, "c": c, "n": len(keys), "risk_difference": (b - c) / n,
            "p": mcnemar_exact(b, c)}


def bootstrap_ci(result, arm: str, tau: float, resamples: int = RESAMPLES,
                 seed: int = SEED) -> tuple[float, float]:
    hits = np.array([float(v) for _, v in sorted(detected(result, arm, tau).items())])
    if hits.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = hits[rng.integers(0, hits.size, size=(resamples, hits.size))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def paired_bootstrap_ci(result, arm_a: str, arm_b: str, tau_a: float, tau_b: float,
                        resamples: int = RESAMPLES, seed: int = SEED) -> tuple[float, float]:
    """95% bootstrap CI of the paired risk difference (a minus b) over the mutants both arms
    reported on, resampling mutant pairs with replacement."""
    da, db = detected(result, arm_a, tau_a), detected(result, arm_b, tau_b)
    keys = sorted(set(da) & set(db))
    if not keys:
        return (0.0, 0.0)
    diff = np.array([float(da[k]) - float(db[k]) for k in keys])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(resamples, diff.size))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
