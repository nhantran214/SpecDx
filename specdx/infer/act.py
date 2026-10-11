"""Stage 1 of the act→read model (spec v2 §3.1–§3.2): whether an interpreter acted.

An item is a (scenario, step) pair of an interpreter step (kind de or persona). An
interpreter acted there when the step completed and, if the step's actor may draft
tables, the end state holds a row of the step's slice that the starting state did not
hold. A Rasch model P(act) = σ(κ_m − δ_s) separates a model's capability to act (κ) from
an item's difficulty (δ). Observability O_s is the population's mean probability of
acting at s; only observable items carry evidence about the specification.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from tonomyx.canon.pack import CanonPack

from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.clauses import DONE_STATUSES
from specdx.spec.columns import Cells, Projection

INTERPRETED: frozenset[str] = frozenset({"de", "persona"})
RASCH_L2 = 1.0  # spec v2 §3.10
OMEGA = 0.3
M_MIN = 3

SliceRows = frozenset[tuple[str, str, Cells]]


@dataclass(frozen=True)
class Act:
    """Whether one behaviour (a run) acted at one item."""

    item: str
    run_key: str
    model: str
    acted: bool


def drafting_steps(pack: CanonPack, steps: Sequence[WorkStep]) -> frozenset[str]:
    """The interpreter steps whose actor may draft tables (draft_tables in de_scope).
    Personas decide gates with the human tools and draft nothing."""
    drafts = {r["ma_de"]: [t for t in r["draft_tables"].split("|") if t]
              for r in pack.tables.get("de_scope", [])}
    return frozenset(s.id for s in steps if s.kind in INTERPRETED and drafts.get(s.actor))


def slice_rows(projection: Projection) -> SliceRows:
    """The rows of a step's slice, each as (table, key, compared cells)."""
    return frozenset((table, key, cells) for table, rows in projection for key, cells in rows)


def is_act(status: str, drafts: bool, now: SliceRows, before: SliceRows) -> bool:
    """Spec v2 §3.1. A completed step is done or needs_gate (v1 §11 item 7)."""
    if status not in DONE_STATUSES:
        return False
    return not drafts or bool(now - before)


@dataclass(frozen=True)
class Rasch:
    """kappa per model, delta per item (mean 0), O_s per item, acted behaviours per item."""

    kappa: dict[str, float]
    delta: dict[str, float]
    observability: dict[str, float]
    acted: dict[str, int]

    def observable(self, item: str, *, omega: float = OMEGA, m_min: int = M_MIN) -> bool:
        return self.observability.get(item, 0.0) >= omega and self.acted.get(item, 0) >= m_min

    def observable_items(self, *, omega: float = OMEGA, m_min: int = M_MIN) -> frozenset[str]:
        return frozenset(i for i in self.delta if self.observable(i, omega=omega, m_min=m_min))


def _objective(x: np.ndarray, n: np.ndarray, y: np.ndarray, m: int, lam: float) -> float:
    eta = x[:m, None] - x[None, m:]
    return float((y * eta - n * np.logaddexp(0.0, eta)).sum() - lam / 2 * (x @ x))


def fit_rasch(
    acts: Iterable[Act], *, lam: float = RASCH_L2, tol: float = 1e-6, max_iter: int = 100
) -> Rasch:
    """Penalised maximum likelihood (L2 penalty lam on κ and δ) with mean δ fixed at 0:
    constrained Newton steps, halved while they would lower the objective."""
    data = list(acts)
    if not data:
        return Rasch({}, {}, {}, {})
    models = sorted({a.model for a in data})
    items = sorted({a.item for a in data})
    mi = {name: i for i, name in enumerate(models)}
    si = {name: j for j, name in enumerate(items)}
    m, s = len(models), len(items)
    n = np.zeros((m, s))
    y = np.zeros((m, s))
    for a in data:
        n[mi[a.model], si[a.item]] += 1
        y[mi[a.model], si[a.item]] += 1 if a.acted else 0
    x = np.zeros(m + s)
    # The constraint row: the δ entries sum to zero, and every step keeps that sum.
    kkt = np.zeros((m + s + 1, m + s + 1))
    kkt[m:m + s, -1] = 1.0
    kkt[-1, m:m + s] = 1.0
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(x[:m, None] - x[None, m:])))
        r = y - n * p
        w = n * p * (1 - p)
        grad = np.concatenate([r.sum(axis=1), -r.sum(axis=0)]) - lam * x
        hess = np.zeros((m + s, m + s))
        hess[:m, :m] = -np.diag(w.sum(axis=1))
        hess[m:, m:] = -np.diag(w.sum(axis=0))
        hess[:m, m:] = w
        hess[m:, :m] = w.T
        hess -= lam * np.eye(m + s)
        kkt[:m + s, :m + s] = hess
        step = np.linalg.solve(kkt, np.concatenate([-grad, [0.0]]))[:m + s]
        before = _objective(x, n, y, m, lam)
        t = 1.0
        while t > 1e-6 and _objective(x + t * step, n, y, m, lam) < before - 1e-12:
            t /= 2
        x = x + t * step
        if float(np.max(np.abs(t * step))) < tol:
            break
    kappa = {name: float(x[i]) for name, i in mi.items()}
    delta = {name: float(x[m + j]) for name, j in si.items()}
    prob = 1.0 / (1.0 + np.exp(-(x[:m, None] - x[None, m:])))
    observability = {name: float(prob[:, j].mean()) for name, j in si.items()}
    acted = {name: int(y[:, j].sum()) for name, j in si.items()}
    return Rasch(kappa, delta, observability, acted)


def act_rate(acts: Iterable[Act]) -> Mapping[str, float]:
    """The fraction of a model's (behaviour, item) pairs where it acted."""
    total: dict[str, int] = {}
    hits: dict[str, int] = {}
    for a in acts:
        total[a.model] = total.get(a.model, 0) + 1
        hits[a.model] = hits.get(a.model, 0) + (1 if a.acted else 0)
    return {m: hits[m] / total[m] for m in sorted(total)}
