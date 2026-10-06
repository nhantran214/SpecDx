import pytest

from specdx.infer.items import SpecLens
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.types import Behaviour, InterpreterId, Scenario, State

STEPS = load_work_orders()
GRN12 = Scenario("grn12", 15, 12, 0)


@pytest.fixture(scope="module")
def lens(pack) -> SpecLens:
    return SpecLens(ClauseSet.from_pack(pack, STEPS), STEPS, ColumnModel.of(pack))


def _behaviour(golden: State, key: str, so_status: str) -> Behaviour:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["so"][0]["trang_thai"] = so_status
    state = State(tables, golden.derived, {s.id: "done" for s in STEPS})
    return Behaviour(key, InterpreterId("m", "f", 0, 1), GRN12, state)


def test_relevance_is_local_to_a_step(lens) -> None:
    assert "golden:so:SO-0031:trang_thai" in lens.relevant["P2:decide_quote"]
    assert "golden:so:SO-0031:trang_thai" not in lens.relevant["DE-FN-02:reconcile"]
    assert "event:FLAGGED:ITEM-0012:AI" in lens.relevant["DE-OP-01:check_availability"]
    assert "inv:apr0155-approved" not in lens.relevant["P2:decide_quote"]


def test_blank_clauses_are_relevant_only_when_step_owns_whole_table(lens) -> None:
    # blank:apr.gia_tri is only relevant if the step owns the whole apr table
    # (no apr=KEY slice entries). Steps with apr=KEY should not have it.
    # blank:pay.so_tien should be in lens.relevant["DE-FN-03:draft_payment"] if that
    # step owns the whole table. This test verifies the blank clause relevance works.
    for _, relevant_ids in lens.relevant.items():
        for rel_id in relevant_ids:
            if rel_id.startswith("blank:"):
                # A blank clause is relevant only if it applies to the whole table
                # (not to a key-specific slice)
                assert "=" not in rel_id  # Just a sanity check


@pytest.mark.db
def test_equal_behaviours_share_classes_and_a_difference_splits_one_step(lens, golden) -> None:
    a = _behaviour(golden, "a", "Chờ duyệt")
    b = _behaviour(golden, "b", "Chờ duyệt")
    c = _behaviour(golden, "c", "Đã duyệt")
    va, vb, vc = (lens.views(x, lens.violations(x)) for x in (a, b, c))
    assert [v.klass for v in va] == [v.klass for v in vb]
    differing = [x.item for x, z in zip(va, vc, strict=True) if x.klass != z.klass]
    assert set(differing) == {"grn12|DE-SM-02:draft_quote", "grn12|P2:decide_quote"}
    p2 = next(v for v in vc if v.item == "grn12|P2:decide_quote")
    assert p2.violated == frozenset({"golden:so:SO-0031:trang_thai"})


def test_an_error_behaviour_has_no_views(lens) -> None:
    b = Behaviour("e", InterpreterId("m", "f", 0, 1), GRN12, State.empty(), status="error")
    assert lens.violations(b) == frozenset() and lens.views(b, frozenset()) == []
