"""B3: the forced-ceiling IRT mislabel indicator (Land & Bikel, "Auditing benchmark labels
with IRT", arXiv 2605.30504; baselines spec §2.6). Subjects are interpreters (model, Tell
variant, seed); items are anchored cells, scored 1 when an interpreter's modal reading equals
the declared value. θ_j is the logit of the subject's overall accuracy, standardised.

Model (paper §3.2, Eq. 1, p. 3): P_ij = c_i + (d_i − c_i)·σ(a_i(θ_j − b_i)). Score (paper
§3.3, p. 4): Δℓ_i = ℓ_i(d_i = 0) − ℓ_i(d_i = 1), where (a_i, b_i, c_i) are refitted under
each forced ceiling and θ is held fixed. Δℓ_i > 0 means a zero ceiling explains the pattern
better: stronger subjects are *less* likely to match the declared value, the signature of a
mis-keyed item.

Departures from the task brief, which had Δℓ = ℓ(free d ∈ [c, 1]) − ℓ(d = 1) with a fixed
chance floor c (the brief asked for a check against the paper, and the paper differs):
  * The contrast is d = 0 against d = 1, not free against forced. A free d ≥ c cannot express
    a response curve that falls with ability, and the brief's score is never negative.
  * c_i is free in [0, 1] (p. 3: "c_i, d_i ∈ [0, 1]"; Appendix B.1, p. 15: fixing c_i to
    chance "results in the largest degradation"). `c` is kept as an optional override.
  * The box is the paper's, a ∈ [1/15, 15] and b ∈ [−2, 2] (p. 3), on a deterministic grid
    in place of L-BFGS-B; the paper's flat-region and local-minimum concerns (§3.3) are what
    a grid avoids.
  * ℓ is averaged over responses (the paper: per respondent; the same for binary data).
Δℓ is signed, so only cells with Δℓ > 0 are reported. θ comes from every anchored cell, not
from the paper's two-stage preliminary fit on items with r > 0.2 (a simplification)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.baselines.jest_rule import hidden_defect, merge_max
from specdx.baselines.votes import interpreter_votes
from specdx.types import Defect

A_GRID = np.geomspace(1 / 15, 15, 9)
B_GRID = np.linspace(-2.0, 2.0, 17)
C_GRID = np.linspace(0.0, 1.0, 11)
EPS = 1e-6


def _best(y: np.ndarray, n: np.ndarray, theta: np.ndarray, floors: np.ndarray,
          d: float) -> float:
    """max over the (a, b, c) grid of the log-likelihood with the ceiling forced to d."""
    a = A_GRID[:, None, None]
    b = B_GRID[None, :, None]
    s = (1.0 / (1.0 + np.exp(-a * (theta[None, None, :] - b)))).reshape(-1, theta.size)
    c = floors[:, None, None]
    p = np.clip(c + (d - c) * s[None], EPS, 1 - EPS)
    ll = (y * np.log(p) + (n - y) * np.log1p(-p)).sum(-1)
    return float(ll.max())


def delta_loglik(y: np.ndarray, n: np.ndarray, theta: np.ndarray,
                 c: float | None = None) -> float:
    """Δℓ = ℓ(d = 0) − ℓ(d = 1) per response, for y successes out of n trials at ability
    theta. c fixes the lower asymptote; by default it is fitted (paper §3.2)."""
    floors = C_GRID if c is None else np.array([c])
    total = float(n.sum())
    return (_best(y, n, theta, floors, 0.0) - _best(y, n, theta, floors, 1.0)) / total


@dataclass(frozen=True)
class IrtAudit:
    name: str = "irt_audit"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        votes = interpreter_votes(inp.data, inp.report.behaviours)
        items = [c for c in sorted(votes)
                 if inp.data.cells[c].anchor is not None and inp.data.cells[c].clause]
        subjects = sorted({w for c in items for w in votes[c]})
        if len(items) < 2 or len(subjects) < 2:
            return BaselineOutput([])
        y = np.array([[float(votes[c].get(s) == inp.data.cells[c].anchor) for c in items]
                      for s in subjects])
        seen = np.array([[float(s in votes[c]) for c in items] for s in subjects])
        acc = (y.sum(1) + 0.5) / (seen.sum(1) + 1.0)
        theta = np.log(acc / (1 - acc))
        if np.ptp(theta) < 1e-9:  # equal ability: no ability scale to relate items to
            return BaselineOutput([])
        theta = (theta - theta.mean()) / theta.std()
        by_cell = inp.data.by_cell()
        out: list[Defect] = []
        for j, cid in enumerate(items):
            mask = seen[:, j] > 0
            if mask.sum() < 2:
                continue
            score = delta_loglik(y[mask, j], seen[mask, j], theta[mask])
            if score > 0:
                out.append(hidden_defect(inp.data.cells[cid], by_cell[cid], score))
        return BaselineOutput(merge_max(out))


register(IrtAudit())
