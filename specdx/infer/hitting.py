"""Soft minimal hitting sets over sampled correction sets (spec §3.3).

Each behaviour b satisfies C minus V(b), so V(b) is a correction set. Every MUS is a
minimal hitting set of the minimal correction sets (Liffiton & Sakallah, 2008). With noisy
samples, a set H qualifies when it hits at least (1 - tau) of the weighted mass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations


def soft_hitting_sets(
    weighted_sets: Sequence[tuple[frozenset[str], float]],
    *,
    tau: float = 0.05,
    max_size: int = 3,
    min_mass: float = 0.01,
) -> list[tuple[frozenset[str], float]]:
    total = sum(w for _, w in weighted_sets)
    if total <= 0:
        return []
    mass: Counter[str] = Counter()
    for violated, weight in weighted_sets:
        for clause in violated:
            mass[clause] += weight
    universe = sorted(c for c, m in mass.items() if m / total >= min_mass)
    found: list[tuple[frozenset[str], float]] = []
    for size in range(1, max_size + 1):
        for combo in combinations(universe, size):
            candidate = frozenset(combo)
            if any(previous <= candidate for previous, _ in found):
                continue
            covered = sum(w for violated, w in weighted_sets if violated & candidate) / total
            if covered >= 1 - tau:
                found.append((candidate, covered))
    return sorted(found, key=lambda x: (-x[1], len(x[0]), sorted(x[0])))
