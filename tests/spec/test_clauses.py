import pytest

from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet, derive_outcome
from specdx.types import Scenario, State

STEPS = load_work_orders()


@pytest.fixture(scope="module")
def clauses(pack) -> ClauseSet:
    return ClauseSet.from_pack(pack, STEPS)


def _done(state: State) -> State:
    return State(state.tables, state.derived, {s.id: "done" for s in STEPS})


def test_clause_families_and_counts(clauses) -> None:
    kinds = [c.kind for c in clauses]
    assert kinds.count("invariant") == 42
    assert kinds.count("expectation") == 12
    assert kinds.count("step") == 11
    assert kinds.count("golden_event") == 10
    assert {"reconcile:consistency", "guard:pay", "vocab:evt", "blank:pay.so_tien"} <= clauses.ids
    assert len(clauses.ids) == len(list(clauses))


@pytest.mark.db
def test_the_canon_end_state_violates_nothing_on_its_own_scenario(clauses, golden) -> None:
    grn12 = Scenario("grn12", 15, 12, 0)
    assert clauses.violations(_done(golden), grn12) == frozenset()


@pytest.mark.db
def test_the_canon_end_state_is_wrong_for_happy(clauses, golden) -> None:
    happy = Scenario("happy", 15, 15, 0)
    v = clauses.violations(_done(golden), happy)
    assert {"expect:three_way:happy", "expect:pay_units", "golden:grn:GRN-0015:sl_nhan",
            "inv:grn-12"} <= v
    assert "expect:three_way:grn12" not in v


@pytest.mark.db
def test_a_changed_status_violates_its_golden_clause(clauses, golden) -> None:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["so"][0]["trang_thai"] = "Đã duyệt"
    v = clauses.violations(_done(State(tables, golden.derived)), Scenario("grn12", 15, 12, 0))
    assert v == frozenset({"golden:so:SO-0031:trang_thai"})


def test_lifted_clauses_do_not_apply_to_two_receipts(clauses) -> None:
    po = clauses["golden:po:PO-0015:sl_dat"]
    assert po.scope == "lifted"
    assert clauses.applies(po, Scenario("x", 16, 13, 0))
    assert not clauses.applies(po, Scenario("x", 16, 13, 0, grn_count=2))


def test_anchored_clauses_follow_parameters_not_names(clauses) -> None:
    rule = clauses["expect:three_way:happy"]
    assert clauses.applies(rule, Scenario("o15-r15-d0-g1", 15, 15, 0))
    assert not clauses.applies(rule, Scenario("grn12", 15, 12, 0))


def test_the_illustrative_flag_does_not_count_as_over_cap() -> None:
    state = State({"evt": [{"kind": "FLAGGED", "ref": "SO-0031", "actor": "AI",
                            "basis": "(Minh họa, không canon) x"}],
                   "grn": [], "po": [], "vinv": []})
    assert derive_outcome(state)["apr_0102"] == "normal"


def test_a_missing_step_is_not_applicable_and_an_error_violates(clauses) -> None:
    rule = clauses["step:DE-FN-02:reconcile"]
    grn12 = Scenario("grn12", 15, 12, 0)
    assert rule.id not in clauses.violations(State({}, {}, {}), grn12)
    assert rule.id in clauses.violations(State({}, {}, {rule.payload["step"]: "error"}), grn12)
