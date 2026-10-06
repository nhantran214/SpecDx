"""Python port of the canon pack's targets/tonomyx/policy.rego.

We do not run OPA. The .rego file stays the source of record, and
tests/policy/test_rego_conformance.py parses it to prove this module covers every
rule in it.

Every rule fails closed: an approval type we do not recognise, a persona we do not
recognise, or a missing linked approval is a refusal, never a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

DISCOUNT_CAP_KEY = "NG-CK-0005"
DEFAULT_DISCOUNT_CAP = Decimal("0.05")

# Canon status values. These stay Vietnamese because they are data, not code.
APPROVED = "Đã duyệt"
REJECTED = "Từ chối"
PENDING = "Chờ duyệt"

# The persona that owns each approval type, from the four allow_decide rules.
APPROVAL_AUTHORITY: dict[str, str] = {
    "quote": "P2",
    "po": "P6",
    "payment": "P5",
    "escalation": "P1",
}

# apr.loai is free Vietnamese text in the canon while policy.rego keys are English.
# Map by the phrase the canon puts at the head of each label. Order matters only in
# that the first match wins; the phrases do not overlap.
_APPROVAL_TYPE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Báo giá", "quote"),
    ("Đơn mua", "po"),
    ("Đề nghị chi", "payment"),
    ("Offer vượt khung", "escalation"),
)


@dataclass(frozen=True)
class Decision:
    """A yes or a no with the sentence that explains it."""

    allowed: bool
    reason: str = ""


def check_discount(rate: Decimal, cap: Decimal | None) -> Decision:
    """Refuse a discount above the policy cap.

    The canon leaves several policy thresholds blank on purpose; for this one the
    rego rule supplies 0.05 as the fallback.
    """
    effective = cap if cap is not None else DEFAULT_DISCOUNT_CAP
    if rate > effective:
        return Decision(False, f"KB-0003: max 5%, requested {rate}")
    return Decision(True)


def allow_decide(apr_type: str, persona: str) -> Decision:
    """Only the persona that owns an approval type may decide it."""
    owner = APPROVAL_AUTHORITY.get(apr_type)
    if owner is None:
        return Decision(False, f"unknown approval type: {apr_type}")
    if owner != persona:
        return Decision(False, f"{apr_type} approvals belong to {owner}, not {persona}")
    return Decision(True)


def allow_side_effect(linked_apr_status: str | None) -> Decision:
    """A side effect outside the system needs its gate already approved."""
    if linked_apr_status == APPROVED:
        return Decision(True)
    shown = linked_apr_status if linked_apr_status else "missing"
    return Decision(False, f"linked approval is {shown!r}, not {APPROVED!r}")


def approval_type_of(loai: str) -> str | None:
    """Translate a canon approval label into the policy type, or nothing."""
    for marker, apr_type in _APPROVAL_TYPE_MARKERS:
        if loai.startswith(marker):
            return apr_type
    return None
