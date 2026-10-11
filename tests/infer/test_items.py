import pytest

from specdx.db import golden_tables, split_background
from specdx.infer.act import drafting_steps
from specdx.infer.items import (
    ItemView,
    SpecLens,
    acted_violations,
    items_reading,
    observations,
)
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import Clause, ClauseSet, rows_read
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


def test_blank_clauses_belong_only_to_steps_owning_the_whole_table(lens) -> None:
    assert all("blank:apr.gia_tri" not in ids for ids in lens.relevant.values())
    assert "blank:pay.so_tien" in lens.relevant["DE-FN-03:draft_payment"]
    assert "blank:pay.so_tien" not in lens.relevant["DE-SM-02:draft_quote"]


def test_cross_cutting_clauses_belong_to_the_steps_owning_the_rows_they_read(lens) -> None:
    def owners(clause_id: str) -> list[str]:
        assert clause_id in lens.clauses.ids, clause_id
        return [step for step, ids in lens.relevant.items() if clause_id in ids]

    for name in ("discount8", "grn12", "happy", "in"):
        assert owners(f"expect:apr_0102:{name}") == ["DE-SM-02:draft_quote"]
        assert owners(f"expect:three_way:{name}") == ["DE-FN-02:reconcile"]
    assert owners("guard:pay") == ["DE-FN-03:draft_payment"]
    assert owners("reconcile:consistency") == ["DE-FN-02:reconcile"]
    assert owners("expect:pay_units") == ["DE-OP-02:record_goods_receipt"]
    for name in ("discount8", "grn12", "happy"):
        # po is owned by its drafter and its approver, grn by the receipt step
        assert owners(f"expect:open_units:{name}") == [
            "DE-FN-04:draft_purchase_order", "P6:decide_po", "DE-OP-02:record_goods_receipt"
        ]


def test_an_expectation_with_undeclared_rows_is_an_error() -> None:
    clause = Clause("expect:new_key:grn12", "expectation", "anchored", "grn12",
                    frozenset(), "kỳ vọng new_key", {"key": "new_key", "eq": "1", "members": []})
    with pytest.raises(ValueError, match="new_key"):
        rows_read(clause)


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


INTERPRETED_STEPS = [s.id for s in STEPS if s.kind != "env"]
NON_DRAFTING = {"DE-OP-01:check_availability", "P2:decide_quote", "P6:decide_po"}


@pytest.fixture(scope="module")
def acting_lens(pack) -> SpecLens:
    background, _ = split_background(golden_tables(pack))
    return SpecLens(ClauseSet.from_pack(pack, STEPS), STEPS, ColumnModel.of(pack),
                    drafting=drafting_steps(pack, STEPS), initial=background)


def _run(pack, key: str, tables=None, statuses=None) -> Behaviour:
    steps = {s.id: "done" for s in STEPS} | (statuses or {})
    state = State(golden_tables(pack) if tables is None else tables, {}, steps)
    return Behaviour(key, InterpreterId("m", "f", 2, 1), GRN12, state)


def _acted(lens, b) -> dict[str, bool]:
    return {v.item.split("|")[1]: v.acted for v in lens.views(b, frozenset())}


def test_items_are_the_interpreter_steps_and_classes_ignore_the_status(acting_lens, pack) -> None:
    done = acting_lens.views(_run(pack, "a"), frozenset())
    failed = acting_lens.views(_run(pack, "b", statuses={"P2:decide_quote": "error"}),
                               frozenset())
    assert [v.item for v in done] == [f"grn12|{s}" for s in INTERPRETED_STEPS]
    assert [v.klass for v in done] == [v.klass for v in failed]
    assert all(v.variant == 2 for v in done)


def test_acting_needs_completion_and_a_new_row_in_a_drafting_steps_slice(acting_lens, pack) -> None:
    assert all(_acted(acting_lens, _run(pack, "g")).values())
    empty = _acted(acting_lens, _run(pack, "e", tables={}))
    assert {s for s, acted in empty.items() if acted} == NON_DRAFTING
    errored = _acted(acting_lens, _run(pack, "x", statuses={"DE-SM-02:draft_quote": "error"}))
    assert [s for s, acted in errored.items() if not acted] == ["DE-SM-02:draft_quote"]


def test_a_run_that_failed_as_a_whole_acted_nowhere(acting_lens) -> None:
    b = Behaviour("e", InterpreterId("m", "f", 0, 1), GRN12, State.empty(), status="error")
    acts = acting_lens.acts(b)
    assert len(acts) == len(INTERPRETED_STEPS) and not any(a.acted for a in acts)
    assert acting_lens.views(b, frozenset()) == []


def test_stage_two_and_violation_sets_keep_only_acted_views() -> None:
    views = [ItemView("s|a", "r", "m", "f", "K1", frozenset({"c1"}), (), acted=True),
             ItemView("s|b", "r", "m", "f", "K2", frozenset({"c2"}), (), acted=False)]
    assert [o.item for o in observations(views)] == ["s|a"]
    assert acted_violations(views) == {"c1"}
    assert items_reading("c1", "s", {"a": frozenset({"c1"}), "b": frozenset()}) == ["s|a"]
