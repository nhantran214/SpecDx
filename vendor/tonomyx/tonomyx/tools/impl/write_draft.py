"""write_draft — create a draft row, never a finished one.

The canon's whole posture is draft first: a digital employee prepares, a human
decides. A draft that arrives already approved would skip the gate, so those
statuses are refused here regardless of which table is being written.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from tonomyx.policy.rules import check_discount
from tonomyx.tools.registry import ToolContext, ToolDenied, require_tool

# Canon status values a draft may never be created with.
FINAL_STATUSES: frozenset[str] = frozenset({"Đã duyệt", "Đã chi", "Đã gửi", "Đã phát hành"})

# The column that carries a discount rate in the canon schema.
_DISCOUNT_COLUMN = "chiet_khau"

# The column that carries a status in the canon schema.
_STATUS_COLUMN = "trang_thai"

# Keys write_draft uses inside the dict it hands the idempotency store. These are
# this tool's own wrapping, not part of the IdempotencyStore protocol, which only
# ever sees an opaque dict[str, Any].
_ROW_FIELD = "row"
_FINGERPRINT_FIELD = "fingerprint"


def _fingerprint(payload: dict[str, Any]) -> str:
    """A stable fingerprint of a flat row of scalars, for replay comparison."""
    return json.dumps(payload, sort_keys=True, default=str)


def write_draft(
    ctx: ToolContext, *, table: str, payload: dict[str, Any], idempotency_key: str
) -> dict[str, Any]:
    """Insert one draft row, or return what this key already produced."""
    require_tool(ctx, "write_draft")
    if table not in ctx.token.draft:
        raise ToolDenied(f"{ctx.token.sub} may not draft into {table}: not in task token scope")

    status = payload.get(_STATUS_COLUMN)
    if status in FINAL_STATUSES:
        raise ToolDenied(
            f"a draft may not be created with status {status!r}: that status needs a decision"
        )

    rate = payload.get(_DISCOUNT_COLUMN)
    if rate is not None:
        decision = check_discount(Decimal(str(rate)), ctx.policy.discount_cap())
        if not decision.allowed:
            raise ToolDenied(decision.reason)

    fingerprint = _fingerprint(payload)
    seen = ctx.idempotency.seen(idempotency_key)
    if seen is not None:
        if seen[_FINGERPRINT_FIELD] != fingerprint:
            raise ToolDenied(
                f"idempotency key {idempotency_key!r} was already used with a different payload"
            )
        return {"table": table, "row": seen[_ROW_FIELD], "replayed": True}

    row = ctx.repo.insert(table, payload)
    ctx.idempotency.record(idempotency_key, {_ROW_FIELD: row, _FINGERPRINT_FIELD: fingerprint})
    return {"table": table, "row": row, "replayed": False}
