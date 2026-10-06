"""Resolve which rows of a table belong to one case, via the canon's own foreign keys.

Some tables carry case_id directly. Others reach a case only through a parent --
soline through so, grn and ship through po, insp through grn, ncr through insp,
pay through vinv. Rather than hand-coding each of those paths (or guessing them
from a column name), this walks the "fk" metadata schema.json already declares for
every column, so a table added later needs only its foreign key declared there;
nothing here changes.
"""

from __future__ import annotations

from typing import Any

from tonomyx.db.repo import Repository

_CASE_ID_COLUMN = "case_id"


def _has_case_id(repo: Repository, table: str) -> bool:
    return any(c.name == _CASE_ID_COLUMN for c in repo.pack.columns[table])


def _fk_edges(repo: Repository, table: str) -> list[tuple[str, str]]:
    """Every (column, target table) pair this table's schema declares as a foreign key."""
    return [(c.name, c.fk.split(".")[0]) for c in repo.pack.columns[table] if c.fk is not None]


def _primary_key(repo: Repository, table: str) -> str:
    for column in repo.pack.columns[table]:
        if column.pk:
            return column.name
    raise ValueError(f"{table} declares no primary key in the canon schema")


def _first_hop_toward_case(
    repo: Repository, table: str, seen: frozenset[str]
) -> tuple[str, str] | None:
    """The first (fk_column, parent_table) step on a path from `table` to a table
    that carries case_id, or None when no declared foreign key leads there."""
    if table in seen:
        return None
    seen = seen | {table}
    for column, target in _fk_edges(repo, table):
        if _has_case_id(repo, target) or _first_hop_toward_case(repo, target, seen) is not None:
            return column, target
    return None


def case_scoped_rows(repo: Repository, table: str, case_id: str) -> list[dict[str, Any]]:
    """Every row of `table` that belongs to one case.

    A table that carries case_id is filtered directly. One that does not is
    filtered by following its foreign keys, however many hops, to a table that
    does. A table with no path to a case at all (none in CASE_TABLES today) comes
    back whole, matching the behaviour before any case-scoped filtering existed.
    """
    if _has_case_id(repo, table):
        return repo.select(table, where={"case_id": case_id})
    hop = _first_hop_toward_case(repo, table, frozenset())
    if hop is None:
        return repo.select(table)
    column, parent_table = hop
    parent_keys = {
        row[_primary_key(repo, parent_table)]
        for row in case_scoped_rows(repo, parent_table, case_id)
    }
    return [row for row in repo.select(table) if row.get(column) in parent_keys]
