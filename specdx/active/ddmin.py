"""Delta debugging (Zeller & Hildebrandt 2002) to localise ambiguity in T (spec §3.7)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from specdx.spec.tell import TellCorpus


def ddmin[T](
    items: Sequence[T], fails: Callable[[Sequence[T]], bool], *, max_tests: int | None = None
) -> list[T]:
    """A minimal subset on which `fails` still holds; anytime under max_tests."""
    tests = 0

    def check(subset: Sequence[T]) -> bool:
        nonlocal tests
        tests += 1
        return fails(subset)

    def out_of_budget() -> bool:
        return max_tests is not None and tests >= max_tests

    current = list(items)
    if not current or not check(current):
        return []
    n = 2
    while len(current) >= 2 and not out_of_budget():
        size = math.ceil(len(current) / n)
        chunks = [current[i:i + size] for i in range(0, len(current), size)]
        reduced = False
        for chunk in chunks:
            if out_of_budget():
                break
            if check(chunk):
                current, n, reduced = chunk, 2, True
                break
        if not reduced:
            for chunk in chunks:
                if out_of_budget():
                    break
                complement = [x for x in current if x not in chunk]
                if complement and check(complement):
                    current, n, reduced = complement, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(current):
                break
            n = min(len(current), 2 * n)
    return current


def localize_ambiguity(
    corpus: TellCorpus,
    candidates: Sequence[str],
    split_persists: Callable[[TellCorpus], bool],
    *,
    max_tests: int | None = None,
) -> list[str]:
    """The fewest candidate sentences that, kept, still let interpreters split."""
    pool = list(candidates)

    def fails(keep: Sequence[str]) -> bool:
        kept = set(keep)
        return split_persists(corpus.without([s for s in pool if s not in kept]))

    return ddmin(pool, fails, max_tests=max_tests)
