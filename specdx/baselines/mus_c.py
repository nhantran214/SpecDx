"""B5: contradictions in C alone, with no interpreters (baselines spec §9.3; v1 §4.5 B5).

From the canon's golden end state (materialised by PostgreSQL), explore up to two rounds
of single-clause repairs (`propose`), at most 200 clause-set evaluations. Each explored
state's violation set is a correction set; the minimal hitting sets of size ≤ 3 of those
sets (PySAT Hitman, smallest first) are the MUS candidates, reported as contradictions."""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

from pysat.examples.hitman import Hitman

from specdx.active.refute import propose
from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.types import Defect, State

MAX_EVALS, DEPTH, MAX_SIZE = 200, 2, 3


def explore[S](start: S, violations: Callable[[S], frozenset[str]],
               neighbours: Callable[[S, str], list[S]], key: Callable[[S], Hashable],
               max_evals: int = MAX_EVALS, depth: int = DEPTH) -> list[frozenset[str]]:
    first = violations(start)
    evals, seen, sets = 1, {key(start)}, [first]
    frontier = [(start, first)]
    for _ in range(depth):
        nxt = []
        for state, bad in frontier:
            for cid in sorted(bad):
                for cand in neighbours(state, cid):
                    k = key(cand)
                    if k in seen:
                        continue
                    if evals >= max_evals:
                        return sets
                    seen.add(k)
                    evals += 1
                    v = violations(cand)
                    sets.append(v)
                    nxt.append((cand, v))
        frontier = nxt
    return sets


def minimal_hitting_sets(sets: Sequence[frozenset[str]],
                         max_size: int = MAX_SIZE) -> list[tuple[str, ...]]:
    if not sets or any(not s for s in sets):  # a consistent state exists: no MUS
        return []
    out: list[tuple[str, ...]] = []
    with Hitman(bootstrap_with=[sorted(s) for s in sets], htype="sorted") as h:
        for hs in h.enumerate():
            if len(hs) > max_size:
                break
            out.append(tuple(sorted(hs)))
    return sorted(set(out), key=lambda t: (len(t), t))


def _key(state: State) -> str:
    return json.dumps([state.tables, state.derived, state.steps], sort_keys=True,
                      ensure_ascii=False)


@dataclass(frozen=True)
class MusOnC:
    name: str = "b5_mus"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("contradiction",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        if inp.golden is None:
            return BaselineOutput([], failure="no golden state (needs PostgreSQL)")
        clauses, scenario = inp.clauses, inp.clauses.anchor
        sets = explore(inp.golden, lambda s: clauses.violations(s, scenario),
                       lambda s, cid: propose(clauses[cid], s, scenario), _key)
        hs = minimal_hitting_sets(sets)
        return BaselineOutput([Defect("contradiction", h, (scenario.id,), (), 1.0 / len(h))
                               for h in hs], extra={"explored": len(sets)})


register(MusOnC())
