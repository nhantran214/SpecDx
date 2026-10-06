"""Read policy thresholds out of the control table.

The policy table is versioned on (ma_nguong, version); the highest version wins,
matching the canon's fn_policy SQL function.
"""

from __future__ import annotations

from decimal import Decimal

from tonomyx.db.repo import Repository
from tonomyx.policy.rules import DEFAULT_DISCOUNT_CAP, DISCOUNT_CAP_KEY


class PolicyStore:
    """Thresholds as the database currently holds them."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def value(self, ma_nguong: str) -> Decimal | None:
        """The latest value of one threshold, or nothing when the canon leaves it blank."""
        rows = self.repo.select("policy", where={"ma_nguong": ma_nguong})
        if not rows:
            return None
        latest = max(rows, key=lambda r: int(r["version"]))
        value = latest["gia_tri"]
        return Decimal(str(value)) if value is not None else None

    def discount_cap(self) -> Decimal:
        """The discount cap, falling back the way policy.rego does.

        `or DEFAULT_DISCOUNT_CAP` would be wrong here: Decimal("0") is a legitimate,
        deliberately configured cap ("no discounts at all") and is falsy, so it must
        be told apart from a genuinely unconfigured threshold (None), which is the
        only case the fallback belongs to.
        """
        value = self.value(DISCOUNT_CAP_KEY)
        return value if value is not None else DEFAULT_DISCOUNT_CAP
