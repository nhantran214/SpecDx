"""The gate's secondary arms, run and reported in the comparison (baselines spec §2.1, §2.6,
§9.8): B2-naive (the single most competent interpreter, every violated clause flagged) and
B3-2PL (negative 2PL discrimination). They wrap the gate's own b2_report and b3_report,
unchanged, and are untuned: τ is 0, the gate's own rule."""

from __future__ import annotations

from dataclasses import dataclass

from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.baselines.b2_single import b2_report
from specdx.baselines.b3_irt import b3_report


@dataclass(frozen=True)
class B2Naive:
    name: str = "b2_naive"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        return BaselineOutput(b2_report(inp.report.behaviours, inp.violated, inp.report.theta,
                                        set(inp.ctx.canon)))


@dataclass(frozen=True)
class B3TwoPL:
    name: str = "b3_2pl"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        return BaselineOutput(b3_report(inp.report.behaviours, inp.violated))


register(B2Naive())
register(B3TwoPL())
