"""One PostgreSQL, loaded with whichever specification is under test.

Every run starts from master data plus background rows, the operational rows that do
not belong to CASE-0031 (spec §11, item 2). seed_operational.sql is never loaded: the
golden end state is read from the pack's CSVs instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from tonomyx.canon.invariants import is_blank
from tonomyx.canon.pack import CanonPack
from tonomyx.config import Settings
from tonomyx.db.bootstrap import OPERATIONAL_TABLES, drop_everything, resync_sequences

from specdx.config import DATABASE_URL, settings_for
from specdx.observe.state import capture_state
from specdx.types import State

MASTER_LOAD_ORDER: tuple[str, ...] = ("ddl.sql", "guardrails.sql", "seed_master.sql")
BACKGROUND_TABLES: tuple[str, ...] = ("cand", "att", "esg")
_CAST = {
    "text": "text", "int": "int", "bigint": "bigint", "numeric": "numeric",
    "bool": "boolean", "date": "date", "timestamp": "timestamptz",
}
_MARKER = "specdx_marker"

Tables = Mapping[str, Sequence[Mapping[str, str]]]


def pack_digest(pack: Path) -> str:
    """A hash of every file in the pack: any edit makes it a different specification."""
    h = hashlib.sha256()
    for path in sorted(p for p in pack.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(pack)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def golden_tables(pack: CanonPack) -> dict[str, list[dict[str, str]]]:
    """The canon's end state for the anchor scenario, straight from its CSVs."""
    return {t: [dict(r) for r in pack.tables.get(t, [])] for t in OPERATIONAL_TABLES}


def is_background(table: str, row: Mapping[str, str]) -> bool:
    return table in BACKGROUND_TABLES or (table == "apr" and is_blank(row.get("case_id")))


def split_background(tables: Tables) -> tuple[dict[str, list[dict[str, str]]], ...]:
    """Split rows into (background, case) by the rule in is_background."""
    background: dict[str, list[dict[str, str]]] = {}
    case: dict[str, list[dict[str, str]]] = {}
    for table, rows in tables.items():
        for row in rows:
            target = background if is_background(table, row) else case
            target.setdefault(table, []).append(dict(row))
    return background, case


def write_rows(conn: psycopg.Connection[dict[str, Any]], pack: CanonPack, tables: Tables) -> None:
    """Insert text rows in foreign-key order, casting each cell to its schema type."""
    for table in OPERATIONAL_TABLES:
        columns = [c for c in pack.columns[table] if c.type != "serial"]
        for row in tables.get(table, []):
            present = [c for c in columns if c.name in row]
            query = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({vals})").format(
                t=sql.Identifier(table),
                cols=sql.SQL(", ").join(sql.Identifier(c.name) for c in present),
                vals=sql.SQL(", ").join(sql.SQL("%s::" + _CAST[c.type]) for c in present),
            )
            conn.execute(query, [None if is_blank(row[c.name]) else row[c.name] for c in present])


def _marker(conn: psycopg.Connection[dict[str, Any]]) -> str | None:
    try:
        row = conn.execute(f"SELECT digest FROM {_MARKER}").fetchone()
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return None
    return None if row is None else str(row["digest"])


class PackDatabase:
    """The SpecDx database and the specification currently loaded in it."""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.url = database_url
        self._digest: str | None = None
        self._settings: Settings | None = None
        self._pack: CanonPack | None = None
        self._background: dict[str, list[dict[str, str]]] = {}

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            raise RuntimeError("no specification loaded: call use() first")
        return self._settings

    @property
    def pack(self) -> CanonPack:
        if self._pack is None:
            raise RuntimeError("no specification loaded: call use() first")
        return self._pack

    def use(self, pack_dir: Path) -> Settings:
        """Load a specification's schema, guardrails and master data, unless already loaded."""
        pack_dir = pack_dir.resolve()
        digest = pack_digest(pack_dir)
        if digest != self._digest:
            self._digest = None
            self._settings = None
            self._pack = None
            self._background = {}
            with psycopg.connect(self.url, row_factory=dict_row) as conn:
                if _marker(conn) != digest:
                    drop_everything(conn)
                    for name in MASTER_LOAD_ORDER:
                        path = pack_dir / "targets" / "postgres" / name
                        conn.execute(path.read_text(encoding="utf-8"))
                    resync_sequences(conn)
                    conn.execute(f"CREATE TABLE {_MARKER} (digest text)")
                    conn.execute(f"INSERT INTO {_MARKER} VALUES (%s)", (digest,))
                    conn.commit()
            self._digest = digest
            self._settings = settings_for(pack_dir, self.url)
            self._pack = CanonPack(pack_dir)
            self._background, _ = split_background(golden_tables(self._pack))
        return self.settings

    @contextmanager
    def isolated(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """A connection on the run's starting state. Everything is rolled back afterwards."""
        conn = psycopg.connect(self.url, row_factory=dict_row)
        try:
            names = sql.SQL(", ").join(sql.Identifier(t) for t in OPERATIONAL_TABLES)
            conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(names))
            write_rows(conn, self.pack, self._background)
            yield conn
        finally:
            conn.rollback()
            conn.close()

    def materialize(self, state: State) -> State | None:
        """Write a state's case rows into a fresh run and read it back, or None if refused."""
        _, case = split_background(state.tables)
        with self.isolated() as conn:
            try:
                with conn.transaction():
                    write_rows(conn, self.pack, case)
            except (psycopg.OperationalError, psycopg.InterfaceError):
                raise
            except psycopg.DatabaseError:
                return None
            return capture_state(conn, self.pack, state.steps)

    def golden_state(self) -> State:
        """The canon's end state, materialised so that derived values are real."""
        state = self.materialize(State(golden_tables(self.pack)))
        if state is None:
            raise RuntimeError("the database refuses the canon's own golden state")
        return state
