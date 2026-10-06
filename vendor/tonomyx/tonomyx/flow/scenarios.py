"""Run the canon pack's scenarios and compare against what it expects.

Both the parameters and the expectations come from data/scenarios.json. Adding a
fourth scenario means editing that file, not this module.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from tonomyx.canon.pack import CanonPack
from tonomyx.de.runtime import DeRuntime
from tonomyx.flow.case_0031 import CaseOutcome, CaseParams, run_case

# scenarios.json states the discount as a percentage; policy works in fractions.
_PERCENT = Decimal("100")


def params_for(pack: CanonPack, name: str, *, use_illustrative: bool = False) -> CaseParams:
    """The levers this scenario pulls.

    use_illustrative is opt-in and off by default: the canon invariants only hold
    when every amount stays blank, so a caller must ask for the non-canon layer
    explicitly rather than get it by accident.
    """
    scenario = pack.scenarios[name]
    return CaseParams(
        discount_pct=Decimal(str(scenario["intake_discount_pct"])) / _PERCENT,
        grn_received=int(scenario["grn_received"]),
        use_illustrative_prices=use_illustrative,
    )


def expectations_for(pack: CanonPack, name: str) -> dict[str, Any]:
    """What the pack says this scenario must produce."""
    return dict(pack.scenarios[name]["expect"])


def run_scenario(runtime: DeRuntime, name: str, *, use_illustrative: bool = False) -> CaseOutcome:
    """Run one named scenario end to end."""
    return run_case(runtime, params_for(runtime.pack, name, use_illustrative=use_illustrative))


def check(outcome: CaseOutcome, expected: dict[str, Any]) -> list[str]:
    """Every way the run differs from what was expected. Empty means it matched.

    Walks every key the pack declares, not a fixed set written down here — a pack
    that names a key the outcome has no field for is reported as a difference
    rather than crashing, because an expectation that cannot be checked has not
    been met.
    """
    actual = {
        "apr_0102": outcome.apr_0102,
        "three_way": outcome.three_way,
        "pay_units": outcome.pay_units,
        "open_units": outcome.open_units,
    }
    problems: list[str] = []
    for key, value in expected.items():
        if key not in actual:
            known = ", ".join(sorted(actual))
            problems.append(
                f"{key}: expected {value!r}, but the outcome has no such field "
                f"(it carries: {known})"
            )
        elif actual[key] != value:
            problems.append(f"{key}: expected {value!r}, got {actual[key]!r}")
    # A step that did not complete is a problem regardless of what the pack
    # declares: an incomplete run must never be reported back as an empty list.
    for failure in outcome.failed_steps:
        problems.append(f"step did not complete: {failure}")
    return problems
