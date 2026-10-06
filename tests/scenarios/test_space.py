from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.scenarios.space import (
    boundary_domain,
    canon_scenarios,
    cell_of,
    discount_cap_pct,
    enumerate_scenarios,
)
from specdx.types import Scenario

PACK = CanonPack(BASE_PACK)


def test_the_canon_scenarios_carry_the_canon_parameters() -> None:
    canon = canon_scenarios(PACK)
    assert canon["grn12"] == Scenario("grn12", 15, 12, 0)
    assert canon["happy"] == Scenario("happy", 15, 15, 0)
    assert canon["discount8"] == Scenario("discount8", 15, 15, 8)


def test_the_discount_cap_is_read_from_the_policy_table() -> None:
    assert discount_cap_pct(PACK) == 5


def test_the_boundary_domain_brackets_every_literal() -> None:
    assert boundary_domain(PACK) == {
        "ordered": [14, 15, 16],
        "received": [11, 12, 13, 14, 15, 16],
        "discount_pct": [0, 4, 5, 6, 8],
        "grn_count": [1, 2],
    }


def test_enumeration_keeps_only_physical_scenarios() -> None:
    scenarios = enumerate_scenarios(boundary_domain(PACK))
    assert all(s.received <= s.ordered + 1 and s.received >= s.grn_count for s in scenarios)
    assert any(s.received == s.ordered + 1 for s in scenarios)
    assert len({s.id for s in scenarios}) == len(scenarios)


def test_cells_group_by_relation_not_by_value() -> None:
    assert cell_of(Scenario("a", 15, 12, 0), 5) == cell_of(Scenario("b", 16, 11, 0), 5)
    assert cell_of(Scenario("a", 15, 15, 8), 5) == "recv-eq|disc-gt|grn-1"
    assert cell_of(Scenario("a", 15, 16, 4, 2), 5) == "recv-gt|disc-le|grn-2"
