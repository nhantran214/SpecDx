"""The Check side C of an executable business specification (spec §2.1, §11)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tonomyx.canon.invariants import evaluate as evaluate_invariant
from tonomyx.canon.invariants import is_blank
from tonomyx.canon.pack import CanonPack
from tonomyx.db.bootstrap import OPERATIONAL_TABLES
from tonomyx.flow.handlers import PO, SO, VINV

from specdx.db import golden_tables
from specdx.orchestrate.work_orders import WorkStep
from specdx.scenarios.space import ANCHOR, canon_scenarios
from specdx.spec.columns import ColumnClass, ColumnModel
from specdx.spec.lifting import (
    column_type,
    instantiate,
    lift_expectations,
    lift_invariant,
    lift_value,
)
from specdx.types import STRUCTURAL, Scenario, State

GOLDEN_TABLES: tuple[str, ...] = (
    "cases", "so", "soline", "inv", "po", "grn", "insp", "ncr", "ship", "vinv", "pay", "apr",
)
# The canon labels one seeded event as illustrative (it belongs to discount8).
ILLUSTRATIVE_PREFIX = "(Minh họa"
DONE_STATUSES = frozenset({"done", "needs_gate"})
PAID = "Đã chi"
APPROVED = "Đã duyệt"
NEEDS_CHECK = "Cần kiểm"
_EXPECTATION_TABLES: dict[str, frozenset[str]] = {
    "apr_0102": frozenset({"evt"}),
    "three_way": frozenset({"vinv"}),
    "pay_units": frozenset({"grn"}),
    "open_units": frozenset({"po", "grn"}),
}
_TRUE = frozenset({"t", "true", "1"})
_FALSE = frozenset({"f", "false", "0"})


@dataclass(frozen=True)
class Clause:
    id: str
    kind: str
    scope: str  # "all" | "lifted" | "anchored"
    anchor: str | None
    tables: frozenset[str]
    description: str
    payload: dict[str, Any] = field(default_factory=dict)


def master_tables(pack: CanonPack) -> dict[str, list[dict[str, str]]]:
    return {t: rows for t, rows in pack.tables.items() if t not in OPERATIONAL_TABLES}


def _same(a: str, b: str) -> bool:
    la, lb = a.strip().lower(), b.strip().lower()
    if la in _TRUE | _FALSE and lb in _TRUE | _FALSE:
        return (la in _TRUE) == (lb in _TRUE)
    return a.strip() == b.strip()


def extract_clauses(
    pack: CanonPack,
    canon: Mapping[str, Scenario],
    steps: Sequence[WorkStep],
    columns: ColumnModel,
    anchor: str = ANCHOR,
) -> list[Clause]:
    anchor_s = canon[anchor]
    out: list[Clause] = []
    for inv in pack.invariants:
        lifted, ok = lift_invariant(inv, anchor_s, pack)
        out.append(Clause(
            f"inv:{inv['id']}", "invariant", "lifted" if ok else "anchored",
            None if ok else anchor, frozenset({inv["table"]}), str(inv["desc"]),
            {"invariant": lifted},
        ))
    gold = golden_tables(pack)
    for table in GOLDEN_TABLES:
        keys = columns.keys[table]
        for row in gold[table]:
            if table == "apr" and is_blank(row.get("case_id")):
                continue
            ident = ":".join(row[k] for k in keys)
            for col, value in row.items():
                if col in keys or is_blank(value):
                    continue
                if columns.classes[(table, col)] is not ColumnClass.LOCKED:
                    continue
                placeholder = lift_value(value, column_type(pack, table, col), anchor_s)
                out.append(Clause(
                    f"golden:{table}:{ident}:{col}", "golden",
                    "lifted" if placeholder else "anchored", None if placeholder else anchor,
                    frozenset({table}), f"{table} {ident}: {col} = {value}",
                    {"table": table, "key": {k: row[k] for k in keys}, "column": col,
                     "value": placeholder or value},
                ))
    for row in gold["evt"]:
        if row["basis"].startswith(ILLUSTRATIVE_PREFIX):
            continue
        out.append(Clause(
            f"event:{row['kind']}:{row['ref']}:{row['actor']}", "golden_event", "anchored",
            anchor, frozenset({"evt"}), f"sự kiện {row['kind']} {row['ref']} ({row['actor']})",
            {"event": [row["kind"], row["ref"], row["actor"]]},
        ))
    expects = {name: dict(pack.scenarios[name]["expect"]) for name in canon}
    for rule in lift_expectations(canon, expects):
        out.append(Clause(
            rule.id, "expectation", rule.scope, rule.anchor,
            _EXPECTATION_TABLES.get(rule.key, frozenset()), f"kỳ vọng {rule.key}",
            {"key": rule.key, "eq": rule.eq, "members": list(rule.members)},
        ))
    for table, cols in sorted(pack.non_canon_columns.items()):
        if table not in OPERATIONAL_TABLES:
            continue
        for col in sorted(cols):
            out.append(Clause(
                f"blank:{table}.{col}", "blank", "all", None, frozenset({table}),
                f"{table}.{col} để trống", {"table": table, "column": col},
            ))
    for step in steps:
        out.append(Clause(
            f"step:{step.id}", "step", "all", None, step.tables,
            f"bước {step.id} hoàn thành", {"step": step.id},
        ))
    out.append(Clause("reconcile:consistency", "reconcile", "all", None,
                      frozenset({"vinv", "po", "grn"}), "vinv.doi_soat = fn_reconcile(po)", {}))
    out.append(Clause("guard:pay", "guard_pay", "all", None, frozenset({"pay", "apr", "vinv"}),
                      "không chi khi chưa duyệt hoặc đối soát còn Cần kiểm", {}))
    out.append(Clause("vocab:evt", "vocabulary", "all", None, frozenset({"evt"}),
                      "evt.kind thuộc danh mục canon", {"kinds": sorted(pack.event_kinds)}))
    return out


def derive_outcome(state: State) -> dict[str, str]:
    """The scenarios.json vocabulary, read from an end state."""
    t = state.tables
    flagged = any(
        e["kind"] == "FLAGGED" and e["ref"] == SO
        and not e.get("basis", "").startswith(ILLUSTRATIVE_PREFIX)
        for e in t.get("evt", [])
    )
    vinv = next((r for r in t.get("vinv", []) if r["ma_vinv"] == VINV), None)
    received = sum(int(r["sl_nhan"]) for r in t.get("grn", [])
                   if r["po"] == PO and not is_blank(r["sl_nhan"]))
    po = next((r for r in t.get("po", []) if r["ma_po"] == PO), None)
    ordered = int(po["sl_dat"]) if po is not None and not is_blank(po["sl_dat"]) else 0
    return {
        "apr_0102": "over_cap" if flagged else "normal",
        "three_way": vinv["doi_soat"] if vinv is not None else "",
        "pay_units": str(received),
        "open_units": str(ordered - received),
    }


# The rows each cross-cutting clause reads, as (table, key), key None for every row of
# the table. Only steps whose work-order slice owns them are answerable for the clause.
# The expectation rows mirror derive_outcome above.
Rows = tuple[tuple[str, str | None], ...]
EXPECTATION_ROWS: dict[str, Rows] = {
    "apr_0102": (("evt", SO),),
    "three_way": (("vinv", VINV),),
    "pay_units": (("grn", None),),
    "open_units": (("po", PO), ("grn", None)),
}


def rows_read(clause: Clause) -> Rows:
    if clause.kind == "expectation":
        key = clause.payload["key"]
        if key not in EXPECTATION_ROWS:
            raise ValueError(f"expectation {key!r} declares no rows: add it to EXPECTATION_ROWS")
        return EXPECTATION_ROWS[key]
    if clause.kind == "reconcile":
        return (("vinv", None),)
    if clause.kind == "guard_pay":
        return (("pay", None),)
    raise ValueError(f"{clause.kind!r} clauses are not cross-cutting")


def evaluate(
    clause: Clause, state: State, scenario: Scenario,
    master: Mapping[str, list[dict[str, str]]],
) -> bool | None:
    """True or False on this state; None when the clause cannot be judged on it."""
    p = instantiate(clause.payload, scenario)
    t = state.tables
    if clause.kind == "invariant":
        return evaluate_invariant(p["invariant"], {**master, **t}).passed
    if clause.kind == "golden":
        rows = [r for r in t.get(p["table"], []) if all(r.get(k) == v for k, v in p["key"].items())]
        return bool(rows) and _same(str(rows[0].get(p["column"], "")), str(p["value"]))
    if clause.kind == "golden_event":
        kind, ref, actor = p["event"]
        return any(e["kind"] == kind and e["ref"] == ref and e["actor"] == actor
                   for e in t.get("evt", []))
    if clause.kind == "expectation":
        value = derive_outcome(state)[p["key"]]
        return value in p["members"] if p["members"] else value == p["eq"]
    if clause.kind == "blank":
        return all(is_blank(r.get(p["column"])) for r in t.get(p["table"], []))
    if clause.kind == "step":
        status = state.steps.get(p["step"])
        return None if status is None else status in DONE_STATUSES
    if clause.kind == "reconcile":
        rows = t.get("vinv", [])
        if not rows:
            return None
        return all(state.derived.get(f"fn_reconcile:{r['po']}") == r["doi_soat"] for r in rows)
    if clause.kind == "guard_pay":
        approved = {r["hoso_ref"] for r in t.get("apr", []) if r["trang_thai"] == APPROVED}
        checks = {r["ma_vinv"]: r["doi_soat"] for r in t.get("vinv", [])}
        return all(
            r["ma_pay"] in approved and checks.get(r["ma_vinv"]) != NEEDS_CHECK
            for r in t.get("pay", []) if r["trang_thai"] == PAID
        )
    if clause.kind == "vocabulary":
        return all(e["kind"] in p["kinds"] for e in t.get("evt", []))
    raise ValueError(f"unknown clause kind {clause.kind!r}")


def describe(clause: Clause, scenario: Scenario) -> str:
    """Describe a clause using scenario-instantiated values.

    Returns scenario-specific text shown to interpreters.
    Vietnamese by design.
    """
    p = instantiate(clause.payload, scenario)
    if clause.kind == "golden" and clause.scope == "lifted":
        ident = ":".join(str(v) for v in p["key"].values())
        return f"{p['table']} {ident}: {p['column']} = {p['value']}"
    if clause.kind == "invariant" and clause.scope == "lifted":
        inv = p["invariant"]
        cond = inv["assert"].get("all", inv["assert"])
        if "col" not in cond or "eq" not in cond:
            return clause.description
        where = inv.get("where") or {}
        where_str = ", ".join(f"{k} = {v}" for k, v in where.items())
        col = cond["col"]
        eq = cond["eq"]
        if where_str:
            return f"{inv['table']} {where_str}: {col} = {eq}"
        return f"{inv['table']}: {col} = {eq}"
    if clause.kind == "expectation":
        if p.get("eq"):
            return f"kỳ vọng {p['key']} = {p['eq']}"
        members = p.get("members", [])
        if members:
            return f"kỳ vọng {p['key']} thuộc {{{', '.join(str(m) for m in members)}}}"
        return clause.description
    return clause.description


class ClauseSet:
    """Every clause of one specification, with the rule for where each applies."""

    def __init__(
        self,
        clauses: Sequence[Clause],
        master: Mapping[str, list[dict[str, str]]],
        canon: Mapping[str, Scenario],
        anchor: str = ANCHOR,
    ) -> None:
        self._clauses = list(clauses)
        self._by_id = {c.id: c for c in self._clauses}
        self.master = master
        self.canon = dict(canon)
        self.anchor = canon[anchor]

    @classmethod
    def from_pack(cls, pack: CanonPack, steps: Sequence[WorkStep]) -> ClauseSet:
        canon = canon_scenarios(pack)
        clauses = extract_clauses(pack, canon, steps, ColumnModel.of(pack))
        return cls(clauses, master_tables(pack), canon)

    def __iter__(self) -> Iterator[Clause]:
        return iter(self._clauses)

    def __len__(self) -> int:
        return len(self._clauses)

    def __getitem__(self, clause_id: str) -> Clause:
        return self._by_id[clause_id]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def applies(self, clause: Clause, scenario: Scenario) -> bool:
        if clause.scope == "all":
            return True
        if clause.scope == "lifted":
            return all(scenario.params()[p] == self.anchor.params()[p] for p in STRUCTURAL)
        assert clause.anchor is not None
        return scenario.signature() == self.canon[clause.anchor].signature()

    def violations(self, state: State, scenario: Scenario) -> frozenset[str]:
        return frozenset(
            c.id for c in self._clauses
            if self.applies(c, scenario) and evaluate(c, state, scenario, self.master) is False
        )
