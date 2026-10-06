"""Load the canon pack's PostgreSQL projection.

The load order comes from canon.meta.json, key v2_advanced.thu_tu_nap. The
comment at the top of guardrails.sql states a different, older order; it is a
leftover from v1 and is deliberately ignored. Guardrails must exist before the
operational seed runs, so that the canon data proves itself valid under them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from tonomyx.config import Settings

LOAD_ORDER: tuple[str, ...] = (
    "ddl.sql",
    "guardrails.sql",
    "seed_master.sql",
    "seed_operational.sql",
)

# Operational and audit tables, in the order seed_operational.sql truncates them.
OPERATIONAL_TABLES: tuple[str, ...] = (
    "cases", "so", "soline", "inv", "po", "grn", "insp", "ncr",
    "ship", "vinv", "pay", "cand", "att", "esg", "apr", "evt",
)


def _sql_dir(settings: Settings) -> Path:
    return settings.canon_pack / "targets" / "postgres"


def _run_file(conn: psycopg.Connection[dict[str, Any]], settings: Settings, name: str) -> None:
    conn.execute((_sql_dir(settings) / name).read_text(encoding="utf-8"))


def drop_everything(conn: psycopg.Connection[dict[str, Any]]) -> None:
    """Reset the database to empty so bootstrap is repeatable."""
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    conn.commit()


def resync_sequences(conn: psycopg.Connection[dict[str, Any]]) -> None:
    """Align every serial column's sequence with the rows already in its table.

    A sequence-backed column only advances its sequence on a default-value
    INSERT. Seed data loaded with explicit ids (as seed_operational.sql does for
    evt) leaves the sequence wherever TRUNCATE ... RESTART IDENTITY put it, so
    the next id-less insert can collide with an id the seed already used. This
    finds every such column via pg_get_serial_sequence instead of naming one, so
    a future table with its own serial column is covered without touching this
    function.

    The candidate name passed to pg_get_serial_sequence is schema-qualified
    (table_schema || '.' || table_name), not bare table_name: PostgreSQL does
    not guarantee that a WHERE clause's AND operands are evaluated left to
    right, so the table_schema = 'public' filter cannot be relied on to run
    before pg_get_serial_sequence sees a row from another schema (such as
    information_schema's own views) — an unqualified name for one of those can
    fail to resolve on the search_path and raise, rather than simply returning
    NULL. A qualified name always resolves to the exact relation and stays
    correct under any evaluation order.
    """
    owners = conn.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND pg_get_serial_sequence(table_schema || '.' || table_name, column_name)
              IS NOT NULL
        """
    ).fetchall()
    for owner in owners:
        table, column = owner["table_name"], owner["column_name"]
        row = conn.execute(
            sql.SQL("SELECT MAX({column}) AS m FROM {table}").format(
                column=sql.Identifier(column), table=sql.Identifier(table)
            )
        ).fetchone()
        max_id = row["m"] if row is not None else None
        if max_id is None:
            # Empty table: the next generated value must still be 1.
            conn.execute("SELECT setval(pg_get_serial_sequence(%s, %s), 1, false)", (table, column))
        else:
            conn.execute(
                "SELECT setval(pg_get_serial_sequence(%s, %s), %s, true)", (table, column, max_id)
            )


def bootstrap(conn: psycopg.Connection[dict[str, Any]], settings: Settings) -> None:
    """Run the four canon SQL files in the order canon.meta.json mandates."""
    for name in LOAD_ORDER:
        _run_file(conn, settings, name)
    resync_sequences(conn)
    conn.commit()


def seed_operational(conn: psycopg.Connection[dict[str, Any]], settings: Settings) -> None:
    """Restore operational and audit tables to the canon state.

    The file starts with its own TRUNCATE, so this is idempotent. The seed
    inserts rows with explicit ids, which never advances a sequence on its own
    (see resync_sequences), so the sequences are resynced here too.
    """
    _run_file(conn, settings, "seed_operational.sql")
    resync_sequences(conn)
    conn.commit()


def truncate_operational(conn: psycopg.Connection[dict[str, Any]]) -> None:
    """Clear operational and audit tables, keeping master data seeded.

    This is the starting point the flow must run from: master data only. TRUNCATE
    does not fire the BEFORE UPDATE OR DELETE trigger that protects evt, which is
    why the canon's own seed file uses it too.
    """
    conn.execute(
        "TRUNCATE " + ", ".join(OPERATIONAL_TABLES) + " RESTART IDENTITY CASCADE"
    )
    conn.commit()
