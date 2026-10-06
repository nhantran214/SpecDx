"""Spec-induced equivalence ≡_S (spec §2.2): which columns separate two behaviours."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from tonomyx.canon.invariants import is_blank
from tonomyx.canon.pack import CanonPack


class ColumnClass(StrEnum):
    LOCKED = "locked"
    BLANK = "blank"
    IGNORED = "ignored"


# Free text, as TonomyX's own golden comparison excludes it.
FREE_TEXT: frozenset[tuple[str, str]] = frozenset({("apr", "noi_dung"), ("evt", "basis")})
_IGNORED_TYPES = frozenset({"serial", "timestamp"})
# Any value in a blank column: the specification says only "this must stay empty".
FILLED = "<filled>"

Cells = tuple[tuple[str, str], ...]
Projection = tuple[tuple[str, tuple[tuple[str, Cells], ...]], ...]
Tables = Mapping[str, Sequence[Mapping[str, str]]]


def _parse(entries: Iterable[str]) -> dict[str, set[str] | None]:
    """'table' keeps every row; 'table=KEY' keeps only those keys (refs, for evt)."""
    wanted: dict[str, set[str] | None] = {}
    for entry in entries:
        table, _, key = entry.partition("=")
        if not key:
            wanted[table] = None
        elif wanted.get(table, set()) is not None:
            wanted.setdefault(table, set()).add(key)  # type: ignore[union-attr]
    return wanted


@dataclass(frozen=True)
class ColumnModel:
    classes: dict[tuple[str, str], ColumnClass]
    keys: dict[str, tuple[str, ...]]

    @classmethod
    def of(cls, pack: CanonPack) -> ColumnModel:
        blank = pack.non_canon_columns
        classes: dict[tuple[str, str], ColumnClass] = {}
        keys: dict[str, tuple[str, ...]] = {}
        for table, columns in pack.columns.items():
            keys[table] = tuple(c.name for c in columns if c.pk)
            for c in columns:
                if c.name in blank.get(table, frozenset()):
                    classes[(table, c.name)] = ColumnClass.BLANK
                elif c.type in _IGNORED_TYPES or (table, c.name) in FREE_TEXT:
                    classes[(table, c.name)] = ColumnClass.IGNORED
                else:
                    classes[(table, c.name)] = ColumnClass.LOCKED
        return cls(classes, keys)

    def _row(self, table: str, row: Mapping[str, str]) -> tuple[str, Cells]:
        if table == "evt":
            key = f"{row.get('kind', '')}:{row.get('ref', '')}"
            return key, (("actor", row.get("actor", "")), ("case_id", row.get("case_id", "")))
        key = ":".join(str(row.get(k, "")) for k in self.keys[table])
        cells: list[tuple[str, str]] = []
        for column in sorted(row):
            cls = self.classes.get((table, column))
            if cls is ColumnClass.LOCKED:
                cells.append((column, str(row[column])))
            elif cls is ColumnClass.BLANK:
                cells.append((column, "" if is_blank(row[column]) else FILLED))
        return key, tuple(cells)

    def project(self, tables: Tables, entries: Iterable[str]) -> Projection:
        out: list[tuple[str, tuple[tuple[str, Cells], ...]]] = []
        for table, keys in sorted(_parse(entries).items()):
            rows: list[tuple[str, Cells]] = []
            for row in tables.get(table, []):
                selector = row.get("ref", "") if table == "evt" else ":".join(
                    str(row.get(k, "")) for k in self.keys[table]
                )
                if keys is None or selector in keys:
                    rows.append(self._row(table, row))
            out.append((table, tuple(sorted(rows))))
        return tuple(out)

    def diff(self, a: Projection, b: Projection) -> frozenset[str]:
        """The fields on which two projections disagree."""
        fields: set[str] = set()
        tables_a, tables_b = dict(a), dict(b)
        for table in set(tables_a) | set(tables_b):
            rows_a = Counter(tables_a.get(table, ()))
            rows_b = Counter(tables_b.get(table, ()))
            changed = (rows_a - rows_b) + (rows_b - rows_a)
            if table == "evt":
                fields |= {f"evt:{key}" for key, _ in changed}
                continue
            by_key_a = {k: dict(c) for k, c in tables_a.get(table, ())}
            by_key_b = {k: dict(c) for k, c in tables_b.get(table, ())}
            for key, _ in changed:
                ca, cb = by_key_a.get(key, {}), by_key_b.get(key, {})
                for column in set(ca) | set(cb):
                    if ca.get(column) != cb.get(column):
                        fields.add(f"{table}.{column}")
        return frozenset(fields)
