"""The scenario space Π for S1 (spec §2.1, §3.5)."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from tonomyx.canon.pack import CanonPack
from tonomyx.flow.handlers import PO
from tonomyx.policy.rules import DISCOUNT_CAP_KEY

from specdx.types import Scenario

# The canon scenario whose end state the invariants and the golden state describe.
ANCHOR = "grn12"


def canon_scenarios(pack: CanonPack) -> dict[str, Scenario]:
    """The canon's own scenarios. The ordered quantity is the canon's purchase order."""
    ordered = int(next(r for r in pack.tables["po"] if r["ma_po"] == PO)["sl_dat"])
    return {
        name: Scenario(
            id=name,
            ordered=ordered,
            received=int(s["grn_received"]),
            discount_pct=int(s["intake_discount_pct"]),
        )
        for name, s in pack.scenarios.items()
    }


def discount_cap_pct(pack: CanonPack) -> int:
    """The discount cap as a whole percentage, from the policy table."""
    row = next(r for r in pack.tables["policy"] if r["ma_nguong"] == DISCOUNT_CAP_KEY)
    return round(float(row["gia_tri"]) * 100)


def boundary_domain(
    pack: CanonPack, bound: Iterable[tuple[str, int]] = ()
) -> dict[str, list[int]]:
    """Canon values plus the values on either side of every literal that matters: the
    lifted clauses' literals, and each (parameter, value) bound from T (spec v2 §3.7).
    enumerate_scenarios keeps only the physical combinations; discounts stay in 0..100."""
    canon = canon_scenarios(pack)
    anchor = canon[ANCHOR]
    ordered = anchor.ordered
    cap = discount_cap_pct(pack)
    domain = {
        "ordered": {ordered - 1, ordered, ordered + 1},
        "received": (
            {s.received for s in canon.values()}
            | {ordered - 1, ordered, ordered + 1}
            | {anchor.received - 1, anchor.received + 1}
        ),
        "discount_pct": {s.discount_pct for s in canon.values()} | {cap - 1, cap, cap + 1},
        "grn_count": {1, 2},
    }
    for parameter, value in bound:
        for v in (value - 1, value, value + 1):
            if v >= 0 and (parameter != "discount_pct" or v <= 100):
                domain[parameter].add(v)
    return {k: sorted(v) for k, v in domain.items()}


def enumerate_scenarios(domain: dict[str, list[int]]) -> list[Scenario]:
    """Every physical combination: at most one unit over-delivered, one unit per receipt."""
    out: list[Scenario] = []
    for o, r, d, g in itertools.product(
        domain["ordered"], domain["received"], domain["discount_pct"], domain["grn_count"]
    ):
        if r > o + 1 or r < g:
            continue
        out.append(Scenario(f"o{o}-r{r}-d{d}-g{g}", o, r, d, g))
    return out


def cell_of(scenario: Scenario, cap_pct: int) -> str:
    """The relational cell a scenario falls in: what is compared with what, not the values."""
    if scenario.received < scenario.ordered:
        rel = "lt"
    elif scenario.received == scenario.ordered:
        rel = "eq"
    else:
        rel = "gt"
    if scenario.discount_pct == 0:
        disc = "zero"
    elif scenario.discount_pct <= cap_pct:
        disc = "le"
    else:
        disc = "gt"
    return f"recv-{rel}|disc-{disc}|grn-{scenario.grn_count}"
