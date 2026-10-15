"""Anchored cells (spec v3 §2): the unit of evidence.

A cell is one (table, row, column) value, one golden event, or one cross-cutting clause,
read at the items whose step may write it. Each carries the specification's own declared
value as its anchor, when C declares one. Only behaviours that acted at an in-scope item
read a cell, and each reads it at most once.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from tonomyx.canon.pack import CanonPack

from specdx.db import golden_tables, is_background
from specdx.infer.items import SpecLens
from specdx.spec.clauses import GOLDEN_TABLES, Clause, evaluate
from specdx.spec.columns import ColumnClass, ColumnModel
from specdx.spec.lifting import column_type, lift_value
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, Scenario

ABSENT = "absent"
PRESENT = "present"
SATISFIED = "satisfied"
VIOLATED = "violated"
# The cross-cutting clauses read as constraint cells (§2.2). Golden values and events are
# value cells; the event vocabulary is relevant to no step (items._relevant).
CONSTRAINT_KINDS: frozenset[str] = frozenset(
    {"invariant", "expectation", "blank", "reconcile", "guard_pay", "step"})
_PLACEHOLDER = re.compile(r"^\{([a-z_]+)\}$")
_NUMERAL = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
_TRUE = frozenset({"t", "true", "1"})
_FALSE = frozenset({"f", "false", "0"})
_NUMERIC_TYPES = frozenset({"int", "bigint", "numeric"})


def normalise(value: str, column_type: str = "text") -> str:
    """§2.5, in order: Unicode NFC, trim, inner whitespace collapsed, casefold, numerals
    canonicalised (35, 35.0 and 35,0 are 35). No semantic mapping: draft and Nháp stay
    distinct. A bool column's t/f (the golden CSVs) and true/false (end states read through
    the typed database) are one value, as C's own comparison already treats them."""
    text = " ".join(unicodedata.normalize("NFC", value).split()).casefold()
    if column_type == "bool":
        if text in _TRUE:
            return "true"
        if text in _FALSE:
            return "false"
    if column_type in _NUMERIC_TYPES and _NUMERAL.match(text):
        number = Decimal(text.replace(",", "."))
        return "0" if number == 0 else format(number.normalize(), "f")
    return text


def tell_texts(corpus: TellCorpus) -> list[str]:
    """Every sentence of T, normalised once for token search."""
    return [normalise(corpus.sentence(s)) for s in corpus.sentence_ids()]


def occurs(value: str, texts: Iterable[str]) -> bool:
    """Whether the normalised value occurs in a normalised text as a whole token: neither
    preceded nor followed by a word character or a hyphen, so SOL-1 is not found in SOL-10
    and PO-0015 is not found in PO-0015-01."""
    needle = normalise(value)
    if not needle:
        return False
    # The boundary class [\w-] excludes only the ASCII hyphen, on purpose.
    pattern = re.compile(rf"(?<![\w-]){re.escape(needle)}(?![\w-])")
    return any(pattern.search(text) for text in texts)


def _case_rows(pack: CanonPack, table: str) -> list[dict[str, str]]:
    return [r for r in golden_tables(pack).get(table, []) if not is_background(table, r)]


def executor_generated(
    pack: CanonPack, columns: ColumnModel, corpus: TellCorpus
) -> frozenset[str]:
    """§2.6: the tables whose canon key T never states. A key is executor-generated when
    none of its golden values occurs anywhere in T; the test reads T only. Events have no
    key of their own (they are identified by kind, ref and actor)."""
    texts = tell_texts(corpus)
    out: set[str] = set()
    for table in sorted(golden_tables(pack)):
        keys = columns.keys.get(table, ())
        rows = _case_rows(pack, table)
        if table == "evt" or not keys or not rows:
            continue
        # The composite-key join is latent: only evt has a composite key and it is excluded.
        if not any(occurs(":".join(r[k] for k in keys), texts) for r in rows):
            out.add(table)
    return frozenset(out)


def match_columns(
    table: str, pack: CanonPack, columns: ColumnModel, generated: frozenset[str]
) -> tuple[str, ...]:
    """The columns that identify a row: its key, or for an executor-generated key the
    remaining LOCKED foreign-key columns (soline rows match on so and ma_sp)."""
    keys = columns.keys[table]
    if table not in generated:
        return keys
    return tuple(c.name for c in pack.columns[table]
                 if c.fk and c.name not in keys
                 and columns.classes[(table, c.name)] is ColumnClass.LOCKED)


def identity(row: Mapping[str, str], match: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalise(str(row.get(c, ""))) for c in match)


@dataclass(frozen=True)
class TableRows:
    """How one table's rows are named. golden: a golden row's identity → its canon key
    (SOLINE-0031-1). A row matching no golden row is named by its normalised identity."""

    table: str
    keys: tuple[str, ...]
    match: tuple[str, ...]
    golden: dict[tuple[str, ...], str]

    def name(self, row: Mapping[str, str]) -> tuple[str, bool]:
        """The row's name and whether it matches a golden row."""
        ident = identity(row, self.match)
        if ident in self.golden:
            return self.golden[ident], True
        return "/".join(ident), False


def table_rows(
    pack: CanonPack, columns: ColumnModel, corpus: TellCorpus, tables: Iterable[str]
) -> dict[str, TableRows]:
    generated = executor_generated(pack, columns, corpus)
    out: dict[str, TableRows] = {}
    for table in tables:
        keys = columns.keys[table]
        match = match_columns(table, pack, columns, generated)
        golden = {identity(r, match): ":".join(r[k] for k in keys)
                  for r in _case_rows(pack, table)}
        out[table] = TableRows(table, keys, match, golden)
    return out


def lift_reading(value: str, ctype: str, scenario: Scenario, anchor: str) -> str:
    """§2.7: a pooled cell's reading in the lifted vocabulary: the placeholder lift_value
    gives in this scenario, else the text. When the value equals several parameters at once
    (happy and discount8 order and receive 15), lift_value gives none; the reading is then
    the anchor's own placeholder if the anchor's parameter is among them, so "equals this
    scenario's value of the anchor's parameter" is one reading in every scenario."""
    if value == ABSENT:
        return value
    placeholder = lift_value(value, ctype, scenario)
    if placeholder is not None:
        return placeholder
    m = _PLACEHOLDER.match(anchor)
    if (m and _NUMERAL.match(value) and "." not in value
            and int(value) == scenario.params().get(m.group(1))):
        return anchor
    return value


@dataclass(frozen=True)
class Cell:
    """kind: value, event or constraint. anchor: d_c, normalised (a lifted placeholder for a
    pooled value cell), or None when C declares nothing. clause: the clause d_c comes from,
    the location of a hidden finding and the target of its refutation. location: the
    ambiguity and weak-check location (table.column, evt:KIND:REF, or the clause id).
    group: the difficulty group g(c) of v3.1 §3.2: a value cell's column (pooled over rows
    and scenarios), an event cell's kind (evt:KIND), a constraint cell's clause. steps: the
    in-scope interpreter steps, in work-order order."""

    id: str
    kind: str
    anchor: str | None
    clause: str | None
    location: str
    group: str
    steps: tuple[str, ...]


@dataclass(frozen=True)
class Reading:
    """One acted behaviour's reading of one cell. clean: the behaviour violates no clause
    relevant at some in-scope item where it acted (the weak-check test of §5.4)."""

    cell: str
    run_key: str
    model: str
    family: str
    scenario: str
    value: str
    clean: bool = False
    variant: int = 0


@dataclass(frozen=True)
class CellData:
    cells: dict[str, Cell]
    readings: list[Reading]

    def by_cell(self) -> dict[str, list[Reading]]:
        out: dict[str, list[Reading]] = {c: [] for c in self.cells}
        for r in self.readings:
            out[r.cell].append(r)
        return out


def draft_tables(pack: CanonPack) -> dict[str, frozenset[str]]:
    """Each digital employee's draft scope (draft_tables in de_scope.csv)."""
    return {r["ma_de"]: frozenset(t for t in r["draft_tables"].split("|") if t)
            for r in pack.tables.get("de_scope", [])}


# One acted behaviour of a scenario: the behaviour, the steps where it acted, and the acted
# steps where it violates no relevant clause.
_Acted = tuple[Behaviour, frozenset[str], frozenset[str]]


class CellLens:
    """Extracts cells and readings from behaviours, on top of the v2 lens (items, acts,
    relevance) and the column model. corpus: the specification's own T, which decides the
    executor-generated keys."""

    def __init__(self, lens: SpecLens, pack: CanonPack, corpus: TellCorpus) -> None:
        self.lens, self.pack, self.clauses = lens, pack, lens.clauses
        self.columns = lens.columns
        self.rows = table_rows(pack, self.columns, corpus,
                               [t for t in GOLDEN_TABLES if t in self.columns.keys])
        drafts = draft_tables(pack)
        self.drafters: dict[str, tuple[str, ...]] = {}
        for step in lens.steps:
            for table in sorted(drafts.get(step.actor, frozenset())):
                self.drafters[table] = (*self.drafters.get(table, ()), step.id)
        self.golden: dict[tuple[str, str, str], Clause] = {}
        self.events: dict[tuple[str, str, str], Clause] = {}
        for c in self.clauses:
            if c.kind == "golden":
                ident = ":".join(str(v) for v in c.payload["key"].values())
                self.golden[(c.payload["table"], ident, c.payload["column"])] = c
            elif c.kind == "golden_event":
                self.events[_triple(*c.payload["event"])] = c
        self.constraints = [c for c in self.clauses if c.kind in CONSTRAINT_KINDS]
        self.constraint_steps = {
            c.id: tuple(s.id for s in lens.steps if c.id in lens.relevant[s.id])
            for c in self.constraints}

    def scope(self, table: str, key: str) -> tuple[str, ...]:
        """§2.4: the steps that may write the row. A drafted table is written by the steps
        whose actor drafts it (draft_tables). A table no step drafts (apr, evt) is written
        through tools; its row is written by the steps whose slice keys it (apr=APR-0102,
        evt=SO-0031)."""
        if table in self.drafters:
            return self.drafters[table]
        wanted = normalise(key)
        return tuple(s.id for s in self.lens.steps
                     if any(e.partition("=")[0] == table and e.partition("=")[2]
                            and normalise(e.partition("=")[2]) == wanted for e in s.slice))

    def value_columns(self, table: str) -> list[str]:
        """The LOCKED columns that are read: neither the key nor the matching columns."""
        rows = self.rows[table]
        return [c.name for c in self.pack.columns[table]
                if self.columns.classes[(table, c.name)] is ColumnClass.LOCKED
                and c.name not in rows.keys and c.name not in rows.match]

    def extract(self, behaviours: Sequence[Behaviour]) -> CellData:
        groups: dict[str, list[_Acted]] = {}
        for b in sorted(behaviours, key=lambda x: x.run_key):
            if b.status != "ok":
                continue
            acted = frozenset(s.id for s in self.lens.acted_steps(b))
            if not acted:
                continue
            bad = self.lens.violations(b)
            clean = frozenset(s for s in acted if not bad & self.lens.relevant[s])
            groups.setdefault(b.scenario.id, []).append((b, acted, clean))
        cells: dict[str, Cell] = {}
        readings: list[Reading] = []
        for sid in sorted(groups):
            self._values(groups[sid], cells, readings)
            self._events(groups[sid], cells, readings)
        self._constraints(groups, cells, readings)
        readings.sort(key=lambda r: (r.cell, r.run_key))
        return CellData(dict(sorted(cells.items())), readings)

    def _values(self, group: list[_Acted], cells: dict[str, Cell],
                readings: list[Reading]) -> None:
        scenario = group[0][0].scenario
        for table, rows in sorted(self.rows.items()):
            names: dict[str, bool] = {name: True for name in rows.golden.values()}
            held: list[dict[str, list[Mapping[str, str]]]] = []
            for b, _, _ in group:
                mine: dict[str, list[Mapping[str, str]]] = {}
                for row in b.state.tables.get(table, []):
                    if is_background(table, row):
                        continue
                    name, gold = rows.name(row)
                    mine.setdefault(name, []).append(row)
                    names.setdefault(name, gold)
                held.append(mine)
            for name in sorted(names):
                steps = self.scope(table, name)
                if not steps:
                    continue
                for col in self.value_columns(table):
                    ctype = column_type(self.pack, table, col)
                    clause = self.golden.get((table, name, col)) if names[name] else None
                    if clause is not None and not self.clauses.applies(clause, scenario):
                        clause = None
                    lifted = clause is not None and clause.scope == "lifted"
                    anchor = None if clause is None else normalise(
                        str(clause.payload["value"]), ctype)
                    cell = Cell(
                        clause.id if lifted and clause is not None
                        else f"{scenario.id}|{table}:{name}:{col}",
                        "value", anchor, None if clause is None else clause.id,
                        f"{table}.{col}", f"{table}.{col}", steps)
                    for (b, acted, clean), mine in zip(group, held, strict=True):
                        here = acted.intersection(steps)
                        if not here:
                            continue
                        found = mine.get(name, [])
                        value = (min(normalise(str(r.get(col, "")), ctype) for r in found)
                                 if found else ABSENT)
                        if lifted and anchor is not None:
                            value = lift_reading(value, ctype, scenario, anchor)
                        readings.append(_reading(cell.id, b, value, bool(here & clean)))
                        cells[cell.id] = cell

    def _events(self, group: list[_Acted], cells: dict[str, Cell],
                readings: list[Reading]) -> None:
        scenario = group[0][0].scenario
        shown: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        for t, c in self.events.items():
            kind, ref, actor = c.payload["event"]
            shown[t] = (str(kind), str(ref), str(actor))
        held: list[frozenset[tuple[str, str, str]]] = []
        for b, _, _ in group:
            mine = set()
            for e in b.state.tables.get("evt", []):
                raw = (str(e.get("kind", "")), str(e.get("ref", "")), str(e.get("actor", "")))
                t = _triple(*raw)
                mine.add(t)
                shown[t] = min(shown.get(t, raw), raw)
            held.append(frozenset(mine))
        for t in sorted(shown):
            kind, ref, actor = shown[t]
            steps = self.scope("evt", ref)
            if not steps:
                continue
            clause = self.events.get(t)
            if clause is not None and not self.clauses.applies(clause, scenario):
                clause = None
            cell = Cell(f"{scenario.id}|evt:{kind}:{ref}:{actor}", "event",
                        None if clause is None else PRESENT,
                        None if clause is None else clause.id, f"evt:{kind}:{ref}",
                        f"evt:{kind}", steps)
            for (b, acted, clean), mine in zip(group, held, strict=True):
                here = acted.intersection(steps)
                if here:
                    value = PRESENT if t in mine else ABSENT
                    readings.append(_reading(cell.id, b, value, bool(here & clean)))
                    cells[cell.id] = cell

    def _constraints(self, groups: Mapping[str, list[_Acted]], cells: dict[str, Cell],
                     readings: list[Reading]) -> None:
        for clause in self.constraints:
            steps = self.constraint_steps[clause.id]
            if not steps:
                continue
            for sid in sorted(groups):
                for b, acted, clean in groups[sid]:
                    here = acted.intersection(steps)
                    if not here or not self.clauses.applies(clause, b.scenario):
                        continue
                    verdict = evaluate(clause, b.state, b.scenario, self.clauses.master)
                    if verdict is None:
                        continue
                    readings.append(_reading(clause.id, b, SATISFIED if verdict else VIOLATED,
                                             bool(here & clean)))
                    cells[clause.id] = Cell(clause.id, "constraint", SATISFIED, clause.id,
                                            clause.id, clause.id, steps)


def _triple(kind: str, ref: str, actor: str) -> tuple[str, str, str]:
    return normalise(str(kind)), normalise(str(ref)), normalise(str(actor))


def _reading(cell: str, b: Behaviour, value: str, clean: bool) -> Reading:
    i = b.interpreter
    return Reading(cell, b.run_key, i.model, i.family, b.scenario.id, value, clean, i.variant)
