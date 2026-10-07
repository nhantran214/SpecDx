"""Delta debugging (Zeller & Hildebrandt 2002) to localise ambiguity in T (spec §3.7)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from specdx.spec.tell import TellCorpus


class OutOfBudget(Exception):
    """No further test can be run; an oracle may raise it too."""


def ddmin[T](
    items: Sequence[T], fails: Callable[[Sequence[T]], bool], *, max_tests: int | None = None
) -> list[T] | None:
    """A 1-minimal subset on which `fails` holds, or None when there is none to report:
    the empty set already fails, the whole does not fail, or the tests run out (max_tests,
    or the oracle raising OutOfBudget) before the result is 1-minimal."""
    tests = 0
    seen: dict[tuple, bool] = {}

    def check(subset: Sequence[T]) -> bool:
        nonlocal tests
        key = tuple(subset)
        if key in seen:
            return seen[key]
        if max_tests is not None and tests >= max_tests:
            raise OutOfBudget
        tests += 1
        result = fails(subset)
        seen[key] = result
        return result

    try:
        current = list(items)
        if check([]) or not check(current):
            return None
        n = 2
        while len(current) >= 2:
            size = math.ceil(len(current) / n)
            chunks = [current[i:i + size] for i in range(0, len(current), size)]
            reduced = False
            for chunk in chunks:
                if check(chunk):
                    current, n, reduced = chunk, 2, True
                    break
            if not reduced and n > 2:
                for chunk in chunks:
                    complement = [x for x in current if x not in chunk]
                    if complement and check(complement):
                        current, n, reduced = complement, max(n - 1, 2), True
                        break
            if not reduced:
                if n >= len(current):
                    break
                n = min(len(current), 2 * n)
        return current
    except OutOfBudget:
        return None


def localize_ambiguity(
    corpus: TellCorpus,
    candidates: Sequence[str],
    split_persists: Callable[[TellCorpus], bool],
    *,
    max_tests: int | None = None,
) -> list[str] | None:
    """The fewest candidate sentences that, kept, still let interpreters split; None when
    the split does not depend on them or the tests run out first (unlocalised)."""
    pool = list(candidates)

    def fails(keep: Sequence[str]) -> bool:
        kept = set(keep)
        return split_persists(corpus.without([s for s in pool if s not in kept]))

    return ddmin(pool, fails, max_tests=max_tests)
