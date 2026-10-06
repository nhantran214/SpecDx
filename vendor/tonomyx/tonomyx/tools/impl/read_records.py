"""read_records — read rows from a table the task token allows."""

from __future__ import annotations

from typing import Any

from tonomyx.tools.registry import ToolContext, ToolDenied, require_tool


def read_records(
    ctx: ToolContext,
    *,
    table: str,
    filter: dict[str, Any] | None = None,  # noqa: A002 - the canon catalog names it "filter"
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Return rows in scope. Refuses before reading anything it may not read."""
    require_tool(ctx, "read_records")
    if table not in ctx.token.read:
        raise ToolDenied(f"{ctx.token.sub} may not read {table}: not in task token scope")
    return {"table": table, "rows": ctx.repo.select(table, where=filter, fields=fields)}
