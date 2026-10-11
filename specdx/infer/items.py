"""Per-step views of a behaviour: the class an interpreter's output falls in at each step.

Items are the interpreter steps of a scenario (spec v2 §3.1). A class is the projection
of the end state onto the step's slice under ≡_S (v1 §2.2), without the step's status
(v2 §3.3): a failed step is a capability failure (stage 1), not a reading of T.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from specdx.infer.act import INTERPRETED, Act, SliceRows, is_act, slice_rows
from specdx.infer.latent import Observation
from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.clauses import Clause, ClauseSet, rows_read
from specdx.spec.columns import ColumnModel, Projection, Tables
from specdx.types import Behaviour


@dataclass(frozen=True)
class ItemView:
    """acted: whether the behaviour acted at this item (stage 1); only acted views enter
    stage 2. variant: the Tell variant the interpreter read (for paraphrase sensitivity)."""

    item: str
    run_key: str
    model: str
    family: str
    klass: str
    violated: frozenset[str]
    projection: Projection
    acted: bool = True
    variant: int = 0


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
    """drafting: the steps whose actor may draft tables (act.drafting_steps). initial: the
    tables every run starts from (the background rows); a drafting step acts only when its
    slice ends with a row these did not hold."""

    def __init__(
        self,
        clauses: ClauseSet,
        steps: Sequence[WorkStep],
        columns: ColumnModel,
        *,
        drafting: frozenset[str] = frozenset(),
        initial: Tables | None = None,
    ) -> None:
        self.clauses, self.columns, self.drafting = clauses, columns, drafting
        self.steps = tuple(s for s in steps if s.kind in INTERPRETED)
        self.relevant = {
            s.id: frozenset(c.id for c in clauses if _relevant(s, c)) for s in self.steps
        }
        self._before: dict[str, SliceRows] = {
            s.id: slice_rows(columns.project(initial or {}, s.slice)) for s in self.steps
        }

    def _acted(self, behaviour: Behaviour, step: WorkStep, projection: Projection) -> bool:
        if behaviour.status != "ok":
            return False
        return is_act(behaviour.state.steps.get(step.id, ""), step.id in self.drafting,
                      slice_rows(projection), self._before[step.id])

    def acts(self, behaviour: Behaviour) -> list[Act]:
        """Stage-1 data: one act indicator per interpreter step. A run that failed as a
        whole acted nowhere."""
        return [
            Act(f"{behaviour.scenario.id}|{step.id}", behaviour.run_key,
                behaviour.interpreter.model,
                self._acted(behaviour, step,
                            self.columns.project(behaviour.state.tables, step.slice)))
            for step in self.steps
        ]

    def violations(self, behaviour: Behaviour) -> frozenset[str]:
        if behaviour.status != "ok":
            return frozenset()
        return self.clauses.violations(behaviour.state, behaviour.scenario)

    def views(self, behaviour: Behaviour, violated: frozenset[str]) -> list[ItemView]:
        if behaviour.status != "ok":
            return []
        out: list[ItemView] = []
        for step in self.steps:
            projection = self.columns.project(behaviour.state.tables, step.slice)
            klass = hashlib.sha1(repr(projection).encode()).hexdigest()[:16]
            out.append(ItemView(
                item=f"{behaviour.scenario.id}|{step.id}", run_key=behaviour.run_key,
                model=behaviour.interpreter.model, family=behaviour.interpreter.family,
                klass=klass, violated=violated & self.relevant[step.id], projection=projection,
                acted=self._acted(behaviour, step, projection),
                variant=behaviour.interpreter.variant,
            ))
        return out

    def diff(self, a: Projection, b: Projection) -> frozenset[str]:
        return self.columns.diff(a, b)


def observations(views: Iterable[ItemView]) -> list[Observation]:
    """Stage-2 data (spec v2 §3.3): acted views only; the others contribute nothing."""
    return [Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views if v.acted]


def acted_violations(views: Iterable[ItemView]) -> frozenset[str]:
    """V(b) of spec v2 §3.4: the violated clauses relevant to an item where b acted. A
    clause relevant only where b did not act is missing for b, not violated."""
    return frozenset().union(*(v.violated for v in views if v.acted))


def items_reading(
    clause: str, scenario_id: str, relevant: Mapping[str, frozenset[str]]
) -> list[str]:
    """The items of one scenario whose step the clause is relevant to."""
    return [f"{scenario_id}|{step}" for step, ids in relevant.items() if clause in ids]
