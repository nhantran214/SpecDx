"""The cap on directed-probe and ddmin runs for one specification (spec §11).

Every requested run counts, whether the Runner serves it from the cache or executes it,
so a resumed study spends its budget exactly as the original run did.
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS: tuple[str, ...] = ("probe", "ddmin")


@dataclass
class ProbeBudget:
    limit: int
    probe: int = 0
    ddmin: int = 0

    @property
    def used(self) -> int:
        return self.probe + self.ddmin

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def take(self, kind: str) -> bool:
        """Count one requested run of this kind; False, counting nothing, once the cap is hit."""
        if kind not in KINDS:
            raise ValueError(f"unknown run kind {kind!r}")
        if self.exhausted:
            return False
        setattr(self, kind, getattr(self, kind) + 1)
        return True
