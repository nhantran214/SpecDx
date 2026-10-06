"""Information-directed scenario selection with sequential escalation (spec §3.5)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from specdx.infer.items import ItemView
from specdx.scenarios.space import cell_of
from specdx.types import Scenario

_EDGE = 1e-6


def binary_kl(p: float, q: float) -> float:
    p = min(max(p, _EDGE), 1 - _EDGE)
    q = min(max(q, _EDGE), 1 - _EDGE)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def information_gain(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    """Realised information: how far the defect scores moved, summed over hypotheses."""
    return sum(binary_kl(after.get(k, 0.0), before.get(k, 0.0)) for k in set(before) | set(after))


@dataclass
class NormalGamma:
    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0

    def update(self, x: float) -> None:
        mu, kappa = self.mu, self.kappa
        self.mu = (kappa * mu + x) / (kappa + 1)
        self.kappa = kappa + 1
        self.alpha += 0.5
        self.beta += kappa * (x - mu) ** 2 / (2 * (kappa + 1))

    def sample(self, rng: random.Random) -> float:
        tau = rng.gammavariate(self.alpha, 1 / self.beta)
        return rng.gauss(self.mu, 1 / math.sqrt(self.kappa * tau))


class CellBandit:
    def __init__(self, cap_pct: int) -> None:
        self.cap_pct = cap_pct
        self.cells: dict[str, NormalGamma] = defaultdict(NormalGamma)
        self.tried: set[str] = set()

    def update(self, scenario: Scenario, reward: float) -> None:
        cell = cell_of(scenario, self.cap_pct)
        self.cells[cell].update(reward)
        self.tried.add(cell)

    def _open(
        self, candidates: Iterable[Scenario], visited: set[tuple[int, ...]]
    ) -> dict[str, list[Scenario]]:
        open_: dict[str, list[Scenario]] = defaultdict(list)
        for s in candidates:
            if s.signature() not in visited:
                open_[cell_of(s, self.cap_pct)].append(s)
        return open_

    def choose(
        self, candidates: Sequence[Scenario], visited: set[tuple[int, ...]], rng: random.Random
    ) -> Scenario | None:
        open_ = self._open(candidates, visited)
        if not open_:
            return None
        untried = sorted(c for c in open_ if c not in self.tried)
        if untried:
            cell = rng.choice(untried)
        else:
            draws = {c: self.cells[c].sample(rng) for c in sorted(open_)}
            cell = max(draws, key=lambda c: draws[c])
        return rng.choice(open_[cell])

    def exhausted(
        self, candidates: Sequence[Scenario], visited: set[tuple[int, ...]], eps: float
    ) -> bool:
        open_ = self._open(candidates, visited)
        if not open_:
            return True
        if any(c not in self.tried for c in open_):
            return False
        return max(self.cells[c].mu for c in open_) < eps


def should_escalate(
    pilot_views: Sequence[ItemView],
    pilot_sets: Sequence[frozenset[str]],
    seen_sets: set[frozenset[str]],
) -> bool:
    """Run the full committee only when the pilot shows something new or disagrees."""
    if any(s not in seen_sets for s in pilot_sets):
        return True
    classes: dict[str, set[str]] = defaultdict(set)
    for v in pilot_views:
        classes[v.item].add(v.klass)
    return any(len(k) >= 2 for k in classes.values())
