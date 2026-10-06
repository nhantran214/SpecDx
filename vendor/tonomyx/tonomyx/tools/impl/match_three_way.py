"""match_three_way — reconcile a purchase order against what arrived.

The canon's own SQL function fn_reconcile compares po.sl_dat with grn.sl_nhan and
nothing else: the vendor invoice carries an amount but no unit count, so it is
billed at the ordered quantity until the receipt says otherwise. This function
mirrors fn_reconcile exactly, which tests/tools/test_match_three_way.py asserts
directly against it — including the case fn_reconcile cannot answer at all.

fn_reconcile's body reads grn through a scalar subquery, (SELECT sl_nhan FROM grn
WHERE po=p_po), which PostgreSQL raises CardinalityViolation on if more than one
row comes back. ddl.sql puts a primary key on grn.ma_grn only — there is no
uniqueness constraint on grn.po — so two receipts against one purchase order is a
legal row set, not a hypothetical. When the lookup here is by purchase order (no
grn argument given) and more than one receipt is found, this function refuses
with ToolDenied rather than summing the rows or picking one: silently summing
would let two receipts that happen to add up to the ordered quantity return a
reconciliation-matched verdict on exactly the input the database cannot answer,
which would then clear a payment fn_reconcile itself would have refused to speak
to. A system that needs multi-receipt reconciliation has to change the canon's
SQL first; this module does not get to invent an answer fn_reconcile does not
have.

When grn is given, the lookup is by primary key and returns at most one row, so
that path is never ambiguous. Zero receipts is a well-defined state where nothing
arrived: grn_units 0, verdict NEEDS_CHECK, and fn_reconcile agrees (NULL <> sl_dat).

The tool reads only. Recording the outcome on the vendor invoice is a separate,
gated write.
"""

from __future__ import annotations

from typing import Any

from tonomyx.tools.registry import ToolContext, ToolDenied, require_tool

MATCHED = "Khớp"
NEEDS_CHECK = "Cần kiểm"


def match_three_way(
    ctx: ToolContext, *, po: str, grn: str | None = None, vinv: str | None = None
) -> dict[str, Any]:
    """Compare ordered against received and report the difference."""
    require_tool(ctx, "match_three_way")
    order = ctx.repo.select_one("po", where={"ma_po": po})
    if order is None:
        raise KeyError(f"no purchase order {po}")
    ordered = int(order["sl_dat"])

    receipts = ctx.repo.select("grn", where={"ma_grn": grn} if grn else {"po": po})
    if grn is None and len(receipts) > 1:
        raise ToolDenied(
            f"purchase order {po} has {len(receipts)} receipts: fn_reconcile's scalar "
            "subquery cannot answer this, so neither can this function"
        )
    received = sum(int(r["sl_nhan"]) for r in receipts)

    return {
        "result": MATCHED if ordered == received else NEEDS_CHECK,
        "po_units": ordered,
        "grn_units": received,
        "diff": ordered - received,
        "vinv": vinv,
    }
