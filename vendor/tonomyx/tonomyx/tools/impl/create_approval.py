"""create_approval — open a gate and stop.

Every approval starts pending. The amount stays blank unless a caller supplies one:
the canon leaves apr.gia_tri blank on all four of its approvals, because the source
books establish no money.
"""

from __future__ import annotations

from typing import Any

from tonomyx.policy.rules import APPROVAL_AUTHORITY, PENDING
from tonomyx.tools.registry import ToolContext, ToolDenied

# The canon's label for each approval type, matching the phrasing in 19_apr.csv.
APPROVAL_LABELS: dict[str, str] = {
    "quote": "Báo giá cần duyệt (P2)",
    "po": "Đơn mua cần duyệt (P6, maker-checker)",
    "payment": "Đề nghị chi cần duyệt (P5)",
    "escalation": "Offer vượt khung (TH0)",
}

# Trust gate level of a human-decided approval.
_HUMAN_GATE_LEVEL = "L3"

# Fields that identify what gate ma_apr names. A repeat call with the same ma_apr
# must agree with the stored row on all of these, or it is not a replay — it is a
# different gate wearing the same id, and decide_approval would route it to the
# wrong persona by reading loai off the stored row.
_IDENTITY_FIELDS: tuple[str, ...] = ("loai", "hoso_ref", "case_id", "gia_tri")


def _identity_mismatches(
    requested: dict[str, Any], existing: dict[str, Any]
) -> list[str]:
    """Identity fields on which a fresh request and the stored row disagree."""
    return [f for f in _IDENTITY_FIELDS if requested[f] != existing[f]]


def create_approval(
    ctx: ToolContext,
    *,
    type: str,  # noqa: A002 - the canon catalog names this input "type"
    record_ref: str,
    case_id: str | None,
    amount: int | None = None,
    noi_dung: str = "",
    ma_apr: str | None = None,
) -> dict[str, Any]:
    """Create one pending approval referring to one record."""
    if "create_approval" not in ctx.token.tools:
        raise ToolDenied(f"{ctx.token.sub} may not call create_approval: not in task token")
    if type not in APPROVAL_AUTHORITY:
        raise ToolDenied(f"unknown approval type: {type}")

    key = ma_apr or f"APR-{record_ref}"
    payload: dict[str, Any] = {
        "ma_apr": key,
        "loai": APPROVAL_LABELS[type],
        "noi_dung": noi_dung or f"{record_ref} cần {APPROVAL_LABELS[type]}",
        "case_id": case_id,
        "hoso_ref": record_ref,
        "nguoi_tao": ctx.token.sub,
        "gia_tri": amount,
        "muc_tu_hanh": _HUMAN_GATE_LEVEL,
        "trang_thai": PENDING,
        "created_at": ctx.clock.now(),
    }

    existing = ctx.repo.select_one("apr", where={"ma_apr": key})
    if existing is not None:
        mismatches = _identity_mismatches(payload, existing)
        if mismatches:
            details = ", ".join(
                f"{field}: requested {payload[field]!r}, stored {existing[field]!r}"
                for field in mismatches
            )
            raise ToolDenied(f"{key} already exists with a different {details}")
        return {"row": existing, "replayed": True}
    return {"row": ctx.repo.insert("apr", payload), "replayed": False}
