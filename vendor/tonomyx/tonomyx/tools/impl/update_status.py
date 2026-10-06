"""update_status — move a record between states the canon allows.

Two gates apply. First the transition must be one the canon recognises; a table or
a status we have no rule for is refused rather than waved through. Second, a status
that carries an approval gate needs that approval already decided — which the
database also enforces with a trigger, deliberately covering the same ground.
"""

from __future__ import annotations

from typing import Any

from tonomyx.db.repo import Repository
from tonomyx.policy.rules import APPROVED, allow_side_effect
from tonomyx.tools.registry import ToolContext, ToolDenied, require_tool

_STATUS_COLUMN = "trang_thai"

# Legal transitions, per table: from a status, to the statuses it may become.
STATUS_FLOW: dict[str, dict[str, frozenset[str]]] = {
    "so": {"Chờ duyệt": frozenset({"Đã duyệt", "Từ chối"})},
    "po": {"Chờ duyệt": frozenset({"Đã duyệt", "Từ chối"})},
    "inv": {"Chờ duyệt": frozenset({"Đã duyệt", "Từ chối"})},
    "vinv": {"Chờ duyệt": frozenset({"Đã duyệt", "Từ chối"})},
    "pay": {"Chờ duyệt": frozenset({"Đã chi", "Từ chối"})},
}

# Statuses that may only be reached once the linked approval says so.
GATED_STATUSES: dict[str, frozenset[str]] = {
    "so": frozenset({"Đã duyệt"}),
    "po": frozenset({"Đã duyệt"}),
    "pay": frozenset({"Đã chi"}),
}


def linked_approval(repo: Repository, record_ref: str) -> dict[str, Any] | None:
    """The approval that stands in front of one record, via apr.hoso_ref.

    The schema allows more than one apr row to target the same hoso_ref (nothing
    stops create_approval from being called twice with different ma_apr), and the
    database's own fn_guard_pay trigger decides with "any approved approval
    suffices" (counting approved rows), not "the first row wins". This mirrors
    that rule: an approved row is returned if one exists, and the choice is a
    stable sort on ma_apr in both branches so a repeated call — or a different
    physical row order — never changes the answer.
    """
    rows = repo.select("apr", where={"hoso_ref": record_ref})
    if not rows:
        return None
    approved = [row for row in rows if row[_STATUS_COLUMN] == APPROVED]
    candidates = approved or rows
    return min(candidates, key=lambda row: str(row["ma_apr"]))


def _primary_key_value(key: dict[str, Any]) -> str:
    if len(key) != 1:
        raise ToolDenied(f"update_status needs exactly one key column, got {sorted(key)}")
    return str(next(iter(key.values())))


def update_status(
    ctx: ToolContext, *, table: str, key: dict[str, Any], new_status: str
) -> dict[str, Any]:
    """Move one record to a new status, or explain why it may not move."""
    require_tool(ctx, "update_status")
    # Validate the key shape once, up front, so every path below — gated or not —
    # gets it. Do not rely on repo.update's AND-composed WHERE clause to make an
    # extra key column harmless; that is an implementation detail elsewhere.
    key_value = _primary_key_value(key)

    flow = STATUS_FLOW.get(table)
    if flow is None:
        raise ToolDenied(f"no status flow is declared for {table}")

    current = ctx.repo.select_one(table, where=key)
    if current is None:
        raise ToolDenied(f"no row in {table} matching {key}")
    old_status = str(current[_STATUS_COLUMN])

    allowed = flow.get(old_status, frozenset())
    if new_status not in allowed:
        raise ToolDenied(
            f"{table}: {old_status!r} may not become {new_status!r}; "
            f"allowed: {sorted(allowed)}"
        )

    if new_status in GATED_STATUSES.get(table, frozenset()):
        approval = linked_approval(ctx.repo, key_value)
        decision = allow_side_effect(approval[_STATUS_COLUMN] if approval else None)
        if not decision.allowed:
            raise ToolDenied(f"{table} {key}: {decision.reason}")

    row = ctx.repo.update(table, key=key, changes={_STATUS_COLUMN: new_status})
    return {"table": table, "row": row, "from": old_status, "to": new_status}
