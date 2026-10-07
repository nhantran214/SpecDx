"""The caps on directed-probe and ddmin runs for one specification (spec §11).

Refutation (directed probes) and localisation (ddmin) each have their own cap, so a
refuter that spends its budget never leaves localisation without one. Every requested
run counts, whether the Runner serves it from the cache or executes it, so a resumed
study spends its budgets exactly as the original run did.
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS: tuple[str, ...] = ("probe", "ddmin")


@dataclass
class RunBudget:
    probe_limit: int
    ddmin_limit: int
    probe: int = 0
    ddmin: int = 0

    def _check(self, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown run kind {kind!r}")

    def exhausted(self, kind: str) -> bool:
        self._check(kind)
        used: int = getattr(self, kind)
        limit: int = getattr(self, f"{kind}_limit")
        return used >= limit

    def take(self, kind: str) -> bool:
        """Count one requested run of this kind; False, counting nothing, once its cap is hit."""
        if self.exhausted(kind):
            return False
        setattr(self, kind, getattr(self, kind) + 1)
        return True
