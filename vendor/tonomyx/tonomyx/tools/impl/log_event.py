"""log_event — append one row to the audit trail.

The vocabulary of event kinds lives in the canon pack and in an evt_kind table the
database enforces with a trigger. Checking here as well means a caller gets a
legible refusal rather than a SQL exception; the trigger is what holds when nobody
comes through here at all.
"""

from __future__ import annotations

from typing import Any

from tonomyx.tools.registry import ToolContext, ToolDenied, require_tool


def log_event(
    ctx: ToolContext, *, case_id: str, kind: str, ref: str, actor: str, basis: str
) -> dict[str, Any]:
    """Record what happened, why, and on whose authority."""
    require_tool(ctx, "log_event")
    if kind not in ctx.pack.event_kinds:
        raise ToolDenied(
            f"event kind {kind!r} is not in the canon registry: "
            f"{sorted(ctx.pack.event_kinds)}"
        )
    row = ctx.repo.insert(
        "evt",
        {
            "case_id": case_id,
            "kind": kind,
            "ref": ref,
            "actor": actor,
            "basis": basis,
            "at": ctx.clock.now(),
        },
    )
    return {"row": row}
