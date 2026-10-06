from specdx.scenarios.space import canon_scenarios
from specdx.spec.lifting import (
    ExpectationRule,
    instantiate,
    lift_expectations,
    lift_invariant,
    lift_value,
)
from specdx.types import Scenario

ANCHOR = Scenario("grn12", 15, 12, 0)


def test_only_unique_integer_literals_lift() -> None:
    assert lift_value("15", "int", ANCHOR) == "{ordered}"
    assert lift_value("12", "int", ANCHOR) == "{received}"
    assert lift_value("20", "int", ANCHOR) is None
    assert lift_value("15", "text", ANCHOR) is None
    assert lift_value("Đã duyệt", "text", ANCHOR) is None


def test_exactly_two_canon_invariants_lift(pack) -> None:
    lifted = {}
    for inv in pack.invariants:
        new, ok = lift_invariant(inv, ANCHOR, pack)
        if ok:
            lifted[inv["id"]] = new["assert"]["eq"]
    assert lifted == {"po-15": "{ordered}", "grn-12": "{received}"}


def test_lifting_never_mutates_the_pack(pack) -> None:
    before = [dict(i["assert"]) for i in pack.invariants]
    for inv in pack.invariants:
        lift_invariant(inv, ANCHOR, pack)
    assert [dict(i["assert"]) for i in pack.invariants] == before


def test_instantiate_fills_placeholders_recursively() -> None:
    s = Scenario("x", 16, 14, 4)
    assert instantiate({"a": "{ordered}", "b": ["{received}", "x"], "c": 3}, s) == {
        "a": "16", "b": ["14", "x"], "c": 3
    }


def test_expectations_lift_per_the_rules_in_spec_section_11(pack) -> None:
    canon = canon_scenarios(pack)
    expects = {name: dict(pack.scenarios[name]["expect"]) for name in canon}
    rules = {r.id: r for r in lift_expectations(canon, expects)}
    assert rules["expect:pay_units"] == ExpectationRule(
        "expect:pay_units", "pay_units", "lifted", None, eq="{received}"
    )
    assert rules["expect:three_way:in"].members == ("Cần kiểm", "Khớp")
    assert rules["expect:apr_0102:in"].members == ("normal", "over_cap")
    assert rules["expect:open_units:grn12"].eq == "3"
    assert rules["expect:three_way:happy"].anchor == "happy"
    assert len(rules) == 12
