"""Refutation of a set of clauses on one scenario (spec §3.6).

A witness must satisfy every target clause and add no violation beyond those its starting
point already had. Finding one is proof (sound); failing to find one is only evidence.
Symbolic repair edits states directly but keeps scenario facts (receipts, environment
rows) untouched, and every candidate is checked against the real database, guardrails
included.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol

from tonomyx.flow.case_0031 import CASE_ID
from tonomyx.flow.handlers import PO, SO, VINV

from specdx.active.budget import RunBudget
from specdx.db import PackDatabase
from specdx.observe.state import ERROR
from specdx.spec.clauses import PAID, Clause, ClauseSet, describe
from specdx.spec.lifting import instantiate
from specdx.types import Behaviour, InterpreterId, RefuteResult, Scenario, State
from specdx.variant import SpecVariant

REPAIR_BASIS = "specdx repair"
REPAIR_AT = "2026-09-01 02:00:00+00"
GOAL = "Yêu cầu bổ sung: kết quả cuối cùng phải thỏa mọi điều sau: {items}."
PROTECTED_TABLES = ("cases", "insp", "ncr", "ship")
# The witness of a refutation that symbolic repair found, at no probe cost.
SYMBOLIC_WITNESS = "symbolic-repair"


class RunnerLike(Protocol):
    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = "",
        *, kind: str = "population",
    ) -> Behaviour: ...


def _tables(state: State) -> dict[str, list[dict[str, str]]]:
    return {t: [dict(r) for r in rows] for t, rows in state.tables.items()}


def _event(kind: str, ref: str, actor: str) -> dict[str, str]:
    return {"case_id": CASE_ID, "kind": kind, "ref": ref, "actor": actor,
            "basis": REPAIR_BASIS, "at": REPAIR_AT}


def _rows_key(rows, columns=None):
    return sorted(
        tuple(sorted((k, str(v)) for k, v in r.items() if columns is None or k in columns))
        for r in rows
    )


def _facts_kept(candidate: State, start: State, scenario: Scenario) -> bool:
    """Repair never edits scenario facts: environment rows stay as they started, and the
    receipts either stay as they started or become exactly the scenario's deliveries.
    """
    for table in PROTECTED_TABLES:
        if _rows_key(candidate.tables.get(table, [])) != _rows_key(start.tables.get(table, [])):
            return False
    cols = ("ma_grn", "po", "sl_nhan")
    now = _rows_key(candidate.tables.get("grn", []), cols)
    if now == _rows_key(start.tables.get("grn", []), cols):
        return True
    deliveries = [(g, PO, str(n)) for g, n in scenario.deliveries()]
    truth = _rows_key([{"ma_grn": g, "po": po, "sl_nhan": s} for g, po, s in deliveries], cols)
    return now == truth


def _evt_count(inv: dict, state: State) -> list[State]:
    """A count invariant over events: append one event matching its filter (kind, ref,
    and so on) when there are too few, or drop the last matching one when too many."""
    where = {k: str(v) for k, v in (inv.get("where") or {}).items()}
    want = int(inv["assert"]["count"])
    tables = _tables(state)
    events = tables.setdefault("evt", [])
    matching = [i for i, e in enumerate(events)
                if all(str(e.get(k, "")).strip() == v for k, v in where.items())]
    if len(matching) < want:
        if "kind" not in where:
            return []
        events.append({**_event(where["kind"], where.get("ref", ""), "AI"), **where})
    elif len(matching) > want:
        del events[matching[-1]]
    else:
        return []
    return [dataclasses.replace(state, tables=tables)]


def propose(clause: Clause, state: State, scenario: Scenario) -> list[State]:
    """Candidate edits that would make this one clause true, if any edit can."""
    p = instantiate(clause.payload, scenario)
    replace = dataclasses.replace
    if clause.kind == "golden":
        tables = _tables(state)
        rows = [r for r in tables.get(p["table"], [])
                if all(r.get(k) == v for k, v in p["key"].items())]
        if not rows:
            return []
        for r in rows:
            r[p["column"]] = str(p["value"])
        return [replace(state, tables=tables)]
    if clause.kind == "invariant":
        inv = p["invariant"]
        if inv["table"] == "evt" and "count" in inv["assert"]:
            return _evt_count(inv, state)
        cond = inv["assert"].get("all", inv["assert"])
        if "col" not in cond:
            return []
        if "eq" in cond:
            values = [str(cond["eq"])]
        elif "in" in cond:
            values = [str(v) for v in cond["in"]][:2]
        elif cond.get("is_blank"):
            values = [""]
        else:
            return []
        out = []
        for value in values:
            tables = _tables(state)
            hit = False
            for r in tables.get(inv["table"], []):
                where = inv.get("where") or {}
                if all(str(r.get(k, "")).strip() == str(w) for k, w in where.items()):
                    r[cond["col"]] = value
                    hit = True
            if hit:
                out.append(replace(state, tables=tables))
        return out
    if clause.kind == "golden_event":
        tables = _tables(state)
        tables.setdefault("evt", []).append(_event(*p["event"]))
        return [replace(state, tables=tables)]
    if clause.kind == "expectation":
        targets = p["members"] or [p["eq"]]
        out = []
        for value in targets:
            tables = _tables(state)
            if p["key"] == "three_way":
                rows = [r for r in tables.get("vinv", []) if r["ma_vinv"] == VINV]
                if not rows:
                    return []
                rows[0]["doi_soat"] = value
            elif p["key"] == "apr_0102":
                tables["evt"] = [e for e in tables.get("evt", [])
                                 if not (e["kind"] == "FLAGGED" and e["ref"] == SO)]
                if value == "over_cap":
                    tables["evt"].append(_event("FLAGGED", SO, "AI"))
            else:
                return []
            out.append(replace(state, tables=tables))
        return out
    if clause.kind == "blank":
        tables = _tables(state)
        for r in tables.get(p["table"], []):
            r[p["column"]] = ""
        return [replace(state, tables=tables)]
    if clause.kind == "step":
        return [replace(state, steps={**state.steps, p["step"]: "done"})]
    if clause.kind == "reconcile":
        tables = _tables(state)
        changed = False
        for r in tables.get("vinv", []):
            value = state.derived.get(f"fn_reconcile:{r['po']}")
            if value and value != ERROR:
                r["doi_soat"] = value
                changed = True
        return [replace(state, tables=tables)] if changed else []
    if clause.kind == "guard_pay":
        tables = _tables(state)
        for r in tables.get("pay", []):
            if r["trang_thai"] == PAID:
                r["trang_thai"] = "Chờ duyệt"
        return [replace(state, tables=tables)]
    return []


class SymbolicRepair:
    def __init__(
        self, db: PackDatabase, variant: SpecVariant, clauses: ClauseSet, *, budget: int = 200
    ) -> None:
        self.db, self.variant, self.clauses, self.budget = db, variant, clauses, budget

    def _cost(
        self, state: State, scenario: Scenario, targets: frozenset[str], allowed: frozenset[str]
    ) -> int:
        bad = self.clauses.violations(state, scenario)
        return len(bad & targets) + len(bad - allowed - targets)

    def search(
        self, targets: frozenset[str], scenario: Scenario, starts: Sequence[State]
    ) -> State | None:
        self.db.use(self.variant.pack)
        spent = 0
        ordered = sorted(starts, key=lambda s: len(self.clauses.violations(s, scenario) & targets))
        for start in ordered:
            allowed = self.clauses.violations(start, scenario) - targets
            current = start
            cost = self._cost(current, scenario, targets, allowed)
            while spent < self.budget:
                if cost == 0:
                    spent += 1
                    real = self.db.materialize(current)
                    if (real is not None and self._cost(real, scenario, targets, allowed) == 0 and
                        _facts_kept(real, start, scenario)):
                        return real
                    break
                bad = self.clauses.violations(current, scenario)
                focus = sorted((bad & targets) | (bad - allowed))
                proposals = [
                    s for cid in focus for s in propose(self.clauses[cid], current, scenario)
                ]
                best: State | None = None
                best_cost = cost
                for candidate in proposals:
                    if spent >= self.budget:
                        break
                    if not _facts_kept(candidate, start, scenario):
                        spent += 1
                        continue
                    spent += 1
                    c = self._cost(candidate, scenario, targets, allowed)
                    if c < best_cost:
                        best, best_cost = candidate, c
                if best is None:
                    break
                current, cost = best, best_cost
        return None


class DirectedProbe:
    def __init__(
        self,
        runner: RunnerLike,
        variant: SpecVariant,
        clauses: ClauseSet,
        interpreters: Sequence[InterpreterId],
        *,
        attempts: int = 3,
        width: int = 3,
        budget: RunBudget | None = None,
    ) -> None:
        self.runner, self.variant, self.clauses = runner, variant, clauses
        self.interpreters = list(interpreters)
        self.attempts, self.width, self.budget = attempts, width, budget

    @property
    def out_of_budget(self) -> bool:
        return self.budget is not None and self.budget.exhausted("probe")

    def search(
        self, targets: frozenset[str], scenario: Scenario, allowed: frozenset[str]
    ) -> tuple[Behaviour | None, int]:
        """A witness behaviour or None, and how many probe runs were requested."""
        items = "; ".join(describe(self.clauses[c], scenario) for c in sorted(targets))
        suffix = GOAL.format(items=items)
        runs = 0
        for interp in self.interpreters[: self.width]:
            for attempt in range(self.attempts):
                if self.budget is not None and not self.budget.take("probe"):
                    return None, runs
                runs += 1
                b = self.runner.run(
                    self.variant, scenario, dataclasses.replace(interp, seed=100 + attempt),
                    suffix=suffix, kind="probe",
                )
                if b.status != "ok":
                    continue
                bad = self.clauses.violations(b.state, scenario)
                if not (bad & targets) and bad <= allowed:
                    return b, runs
        return None, runs


class CombinedRefuter:
    """Repair first (cheap), then a directed probe. Memoised per scenario and targets."""

    def __init__(
        self, clauses: ClauseSet, repair: SymbolicRepair | None, probe: DirectedProbe | None
    ) -> None:
        self.clauses, self.repair, self.probe = clauses, repair, probe
        self._memo: dict[tuple[tuple[int, ...], frozenset[str]], RefuteResult] = {}

    def refute(
        self, targets: frozenset[str], scenario: Scenario, behaviours: Sequence[Behaviour]
    ) -> RefuteResult:
        key = (scenario.signature(), targets)
        if key in self._memo:
            return self._memo[key]
        starts = [b.state for b in behaviours if b.status == "ok"]
        allowed: frozenset[str] = frozenset().union(
            *(self.clauses.violations(s, scenario) for s in starts)
        ) - targets
        attempts = 0
        result: RefuteResult | None = None
        if self.repair is not None and starts:
            attempts += 1
            if self.repair.search(targets, scenario, starts) is not None:
                result = RefuteResult(True, SYMBOLIC_WITNESS, attempts)
        if result is None and self.probe is not None:
            found, probed = self.probe.search(targets, scenario, allowed)
            attempts += probed
            if found is not None:
                result = RefuteResult(True, found.run_key, attempts)
            elif probed == 0 and self.probe.out_of_budget:
                # The cap stopped the probe before it started: no refutation was attempted.
                result = RefuteResult(False, None, attempts, attempted=False)
        if result is None:
            result = RefuteResult(False, None, attempts)
        self._memo[key] = result
        return result
