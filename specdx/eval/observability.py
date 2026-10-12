"""Observability-aware scoring (spec v2 §4.1): which mutants the population can score.

A mutant is scorable when, in the base specification's runs, at least k acted behaviours
touch its observed_at targets. A behaviour touches a row or an event when it acted at an
item whose step's slice covers it and its end state holds it; it touches a step when it
acted at that step. The scorable set is computed once, from base runs only, and is the
same for SpecDx, B2 and B3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.columns import ColumnModel
from specdx.types import Behaviour, State, Target

SCORABLE_K = 3  # spec v2 §3.10


class ActingLens(Protocol):
    columns: ColumnModel

    def acted_steps(self, behaviour: Behaviour) -> list[WorkStep]: ...


def _covers(step: WorkStep, table: str, key: str) -> bool:
    return any(entry == table or entry == f"{table}={key}" for entry in step.slice)


def touches(target: Target, step: WorkStep, state: State, columns: ColumnModel) -> bool:
    """Whether a behaviour that acted at this step touches the target."""
    if target.kind == "step":
        return target.name == step.id
    if target.kind == "event":
        return _covers(step, "evt", target.key) and any(
            target.name in ("", e.get("kind")) and e.get("ref") == target.key
            for e in state.tables.get("evt", []))
    if target.kind == "row":
        keys = columns.keys[target.name]
        return _covers(step, target.name, target.key) and any(
            ":".join(str(r.get(k, "")) for k in keys) == target.key
            for r in state.tables.get(target.name, []))
    raise ValueError(f"unknown target kind {target.kind!r}")


def touch_counts(
    targets: Mapping[str, Sequence[Target]], behaviours: Sequence[Behaviour], lens: ActingLens
) -> dict[str, int]:
    """For each mutant id, how many base behaviours touch at least one of its targets."""
    acted = {b.run_key: lens.acted_steps(b) for b in behaviours if b.status == "ok"}
    counts: dict[str, int] = {}
    for mid, wanted in targets.items():
        counts[mid] = sum(
            1 for b in behaviours if b.status == "ok"
            and any(touches(t, s, b.state, lens.columns) for s in acted[b.run_key]
                    for t in wanted)
        )
    return counts


def scorable(counts: Mapping[str, int], k: int = SCORABLE_K) -> frozenset[str]:
    return frozenset(mid for mid, n in counts.items() if n >= k)
