"""The simulation's metrics (spec v3 §8.1): typed F1, accuracy recovery, calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from specdx.sim.generator import Truth

TYPES: tuple[str, ...] = ("hidden", "ambiguity", "weak")  # contradiction is not simulated


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp, self.fp, self.fn = self.tp + other.tp, self.fp + other.fp, self.fn + other.fn

    @property
    def f1(self) -> float:
        """2tp / (2tp + fp + fn), exactly (matching.exact_f1), as a float."""
        denominator = 2 * self.tp + self.fp + self.fn
        return float(Fraction(2 * self.tp, denominator)) if denominator else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "f1": self.f1}


def typed_counts(predicted: Iterable[tuple[str, str]],
                 truth: Mapping[str, Truth]) -> dict[str, Counts]:
    """predicted: (cell, type) pairs. Each type is scored over cells: a predicted cell that
    is a truth of that type is a TP, any other an FP, a missed truth an FN."""
    found = set(predicted)
    out: dict[str, Counts] = {}
    for t in TYPES:
        wanted = {c for c, x in truth.items() if getattr(x, t)}
        mine = {c for c, kind in found if kind == t}
        out[t] = Counts(len(mine & wanted), len(mine - wanted), len(wanted - mine))
    return out


def theta_error(estimated: Mapping[str, float], true: Mapping[str, float]) -> float:
    """Mean |θ̂_m − θ_m| over the models."""
    return sum(abs(estimated[m] - true[m]) for m in true) / len(true)


def calibration(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> tuple[float, float]:
    """Expected calibration error over equal-width bins, and the Brier score, of posterior
    defect probabilities against the planted truth."""
    if not pairs:
        return 0.0, 0.0
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mine = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if mine:
            gap = sum(p for p, _ in mine) / len(mine) - sum(y for _, y in mine) / len(mine)
            ece += len(mine) / len(pairs) * abs(gap)
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    return ece, brier
