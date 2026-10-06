"""Schema-driven table access.

There is deliberately no hand-written repository per table: the column list comes
from the canon pack's schema.json, so adding a table needs no code here. Table and
column names are validated against that schema before they reach SQL, which is
also what makes composing identifiers safe.

There is no delete. The canon has no operation that removes a business row.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import psycopg
from psycopg import sql

from tonomyx.canon.pack import CanonPack


class UnknownTable(KeyError):
    """The table is not declared in the canon schema."""


class UnknownColumn(KeyError):
    """The column is not declared for that table in the canon schema."""


class AppendOnlyViolation(RuntimeError):
    """An attempt to modify a row in an append-only table."""


class Repository:
    """Read and write canon tables through their declared schema."""

    def __init__(self, conn: psycopg.Connection[dict[str, Any]], pack: CanonPack) -> None:
        self.conn = conn
        self.pack = pack

    def columns(self, table: str) -> list[str]:
        if table not in self.pack.columns:
            raise UnknownTable(table)
        return [c.name for c in self.pack.columns[table]]

    def _validate(self, table: str, names: list[str]) -> list[str]:
        known = set(self.columns(table))
        for name in names:
            if name not in known:
                raise UnknownColumn(f"{name} is not a column of {table}")
        return names

    def _reject_append_only(self, table: str) -> None:
        if table in self.pack.append_only:
            raise AppendOnlyViolation(
                f"{table} is append-only: rows can be inserted but never changed"
            )

    def select(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected = self._validate(table, list(fields) if fields else self.columns(table))
        clause = dict(where or {})
        self._validate(table, list(clause))
        query = sql.SQL("SELECT {fields} FROM {table}").format(
            fields=sql.SQL(", ").join(sql.Identifier(c) for c in selected),
            table=sql.Identifier(table),
        )
        if clause:
            query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(
                sql.SQL("{} = %s").format(sql.Identifier(c)) for c in clause
            )
        rows = self.conn.execute(query, list(clause.values())).fetchall()
        return [dict(r) for r in rows]

    def select_one(self, table: str, where: dict[str, Any]) -> dict[str, Any] | None:
        rows = self.select(table, where=where)
        return rows[0] if rows else None

    def insert(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate(table, list(payload))
        query = sql.SQL("INSERT INTO {table} ({cols}) VALUES ({vals}) RETURNING *").format(
            table=sql.Identifier(table),
            cols=sql.SQL(", ").join(sql.Identifier(c) for c in payload),
            vals=sql.SQL(", ").join(sql.Placeholder() * len(payload)),
        )
        row = self.conn.execute(query, list(payload.values())).fetchone()
        assert row is not None  # RETURNING * on a successful INSERT always yields a row
        return dict(row)

    def update(
        self, table: str, key: dict[str, Any], changes: dict[str, Any]
    ) -> dict[str, Any]:
        # Check before touching the database so the refusal is the same whether or
        # not a connection is healthy, and so the reason is legible.
        self._reject_append_only(table)
        self._validate(table, list(key))
        self._validate(table, list(changes))
        if not changes:
            raise ValueError("update needs at least one column to change")
        query = sql.SQL("UPDATE {table} SET {sets} WHERE {cond} RETURNING *").format(
            table=sql.Identifier(table),
            sets=sql.SQL(", ").join(
                sql.SQL("{} = %s").format(sql.Identifier(c)) for c in changes
            ),
            cond=sql.SQL(" AND ").join(
                sql.SQL("{} = %s").format(sql.Identifier(c)) for c in key
            ),
        )
        row = self.conn.execute(
            query, [*changes.values(), *key.values()]
        ).fetchone()
        if row is None:
            raise KeyError(f"no row in {table} matching {key}")
        return dict(row)

    def as_text(self, table: str) -> list[dict[str, str]]:
        """Read a table with every cell rendered as text.

        The invariant evaluator compares strings, because the canon states its
        invariants against CSV text. Rendering here keeps that one comparison
        rule for both sources.
        """
        return [
            {k: _as_text(v) for k, v in row.items()} for row in self.select(table)
        ]


def _as_text(value: Any) -> str:
    """Render a database value the way the canon CSVs write it."""
    if value is None:
        return ""
    if isinstance(value, bool):
        # The canon accepts "true"/"false" as well as "t"/"f".
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    return str(value)


def tables_as_text(repo: Repository) -> dict[str, list[dict[str, str]]]:
    """Every canon table as text, ready for the invariant evaluator."""
    return {table: repo.as_text(table) for table in repo.pack.columns}
