"""B3: 2PL item response theory over runs × clauses (after arXiv:2605.30504).

A clause whose discrimination is negative is one that stronger runs fail more often: the
signature of a mis-keyed item. B3 sees only satisfied/violated, so a clause that does not
apply to a run's scenario counts as satisfied; that is the baseline's documented limit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from specdx.types import Behaviour, Defect


def fit_2pl(
    y: np.ndarray, *, iters: int = 2000, lr: float = 0.05, l2: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m = y.shape
    theta, a, b = np.zeros(n), np.ones(m), np.zeros(m)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a[None, :] * (theta[:, None] - b[None, :]))))
        r = y - p
        g_theta = (r * a[None, :]).sum(axis=1) - l2 * theta
        g_a = (r * (theta[:, None] - b[None, :])).sum(axis=0) - l2 * (a - 1.0)
        g_b = (-r * a[None, :]).sum(axis=0) - l2 * b
        theta += lr * g_theta / max(m, 1)
        a += lr * g_a / max(n, 1)
        b += lr * g_b / max(n, 1)
        theta -= theta.mean()
    return theta, a, b


def b3_report(
    behaviours: Sequence[Behaviour], violated: Mapping[str, frozenset[str]]
) -> list[Defect]:
    ok = [b for b in behaviours if b.status == "ok"]
    clauses = sorted({c for b in ok for c in violated[b.run_key]})
    if len(ok) < 2 or not clauses:
        return []
    y = np.array([[0.0 if c in violated[b.run_key] else 1.0 for c in clauses] for b in ok])
    _, a, _ = fit_2pl(y)
    out = []
    for j, clause in enumerate(clauses):
        if a[j] < 0:
            hit = [b for b in ok if clause in violated[b.run_key]]
            out.append(Defect("hidden", (clause,), tuple(sorted({b.scenario.id for b in hit})),
                              tuple(b.run_key for b in hit), float(-a[j])))
    return sorted(out, key=lambda d: (-d.score, d.location))
