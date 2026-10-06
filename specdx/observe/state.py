"""Read a run's end state out of the database, as text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import psycopg
from tonomyx.canon.pack import CanonPack
from tonomyx.db.bootstrap import OPERATIONAL_TABLES
from tonomyx.db.repo import Repository

from specdx.types import State

# What fn_reconcile "returns" when it raises (two receipts make its scalar subquery fail).
ERROR = "ERROR"


def reconcile_value(conn: psycopg.Connection[dict[str, Any]], po: str) -> str:
    """The canon's own fn_reconcile verdict for one order, or ERROR if it cannot answer."""
    try:
        with conn.transaction():
            row = conn.execute("SELECT fn_reconcile(%s) AS r", (po,)).fetchone()
    except psycopg.Error:
        return ERROR
    return "" if row is None or row["r"] is None else str(row["r"])


def capture_state(
    conn: psycopg.Connection[dict[str, Any]],
    pack: CanonPack,
    steps: Mapping[str, str] | None = None,
) -> State:
    """Every operational and audit table as text, plus fn_reconcile for every order."""
    repo = Repository(conn, pack)
    tables = {t: repo.as_text(t) for t in OPERATIONAL_TABLES}
    orders = {r["ma_po"] for r in tables["po"]} | {r["po"] for r in tables["vinv"] if r["po"]}
    derived = {f"fn_reconcile:{po}": reconcile_value(conn, po) for po in sorted(orders)}
    return State(tables=tables, derived=derived, steps=dict(steps or {}))
