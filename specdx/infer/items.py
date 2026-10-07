"""Per-step views of a behaviour: the class an interpreter's output falls in at each step."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.clauses import Clause, ClauseSet, rows_read
from specdx.spec.columns import ColumnModel, Projection
from specdx.types import Behaviour


@dataclass(frozen=True)
class ItemView:
    item: str
    run_key: str
    model: str
    family: str
    klass: str
    violated: frozenset[str]
    projection: Projection


def _keyed(step: WorkStep, table: str) -> set[str] | None:
    """None when the step owns the whole table; else the keys it owns (maybe none)."""
    keys: set[str] = set()
    for entry in step.slice:
        name, _, key = entry.partition("=")
        if name == table:
            if not key:
                return None
            keys.add(key)
    return keys


def _owns(step: WorkStep, table: str, key: str | None) -> bool:
    """Whether the step's slice owns these rows: the whole table, or the keyed row."""
    keys = _keyed(step, table)
    return keys is None or (key is not None and key in keys)


def _relevant(step: WorkStep, clause: Clause) -> bool:
    if clause.kind == "step":
        return bool(clause.payload["step"] == step.id)
    if clause.kind == "vocabulary":
        return False
    if clause.kind == "golden_event":
        keys = _keyed(step, "evt")
        return keys is None or clause.payload["event"][1] in keys
    if clause.kind == "golden":
        keys = _keyed(step, clause.payload["table"])
        ident = ":".join(str(v) for v in clause.payload["key"].values())
        return keys is None or ident in keys
    if clause.kind == "invariant":
        inv = clause.payload["invariant"]
        keys = _keyed(step, inv["table"])
        if keys is None:
            return True
        return bool(keys & {str(v) for v in (inv.get("where") or {}).values()})
    if clause.kind == "blank":
        keys = _keyed(step, clause.payload["table"])
        return keys is None
    if clause.kind in ("expectation", "reconcile", "guard_pay"):
        # Cross-cutting: only the steps owning the rows read, never mere table overlap.
        return any(_owns(step, table, key) for table, key in rows_read(clause))
    raise ValueError(f"unknown clause kind {clause.kind!r}")


class SpecLens:
    def __init__(
        self, clauses: ClauseSet, steps: Sequence[WorkStep], columns: ColumnModel
    ) -> None:
        self.clauses, self.steps, self.columns = clauses, tuple(steps), columns
        self.relevant = {
            s.id: frozenset(c.id for c in clauses if _relevant(s, c)) for s in self.steps
        }

    def violations(self, behaviour: Behaviour) -> frozenset[str]:
        if behaviour.status != "ok":
            return frozenset()
        return self.clauses.violations(behaviour.state, behaviour.scenario)

    def views(self, behaviour: Behaviour, violated: frozenset[str]) -> list[ItemView]:
        if behaviour.status != "ok":
            return []
        out: list[ItemView] = []
        for step in self.steps:
            mine = violated & self.relevant[step.id]
            projection = self.columns.project(behaviour.state.tables, step.slice)
            status = behaviour.state.steps.get(step.id, "")
            klass = hashlib.sha1(repr((projection, sorted(mine), status)).encode()).hexdigest()[:16]
            out.append(ItemView(
                item=f"{behaviour.scenario.id}|{step.id}", run_key=behaviour.run_key,
                model=behaviour.interpreter.model, family=behaviour.interpreter.family,
                klass=klass, violated=mine, projection=projection,
            ))
        return out

    def diff(self, a: Projection, b: Projection) -> frozenset[str]:
        return self.columns.diff(a, b)
