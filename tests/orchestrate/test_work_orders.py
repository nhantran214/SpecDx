from specdx.orchestrate.work_orders import (
    TEMPLATE_NAMES,
    load_work_orders,
    render_inputs,
    template_values,
)
from specdx.types import Scenario


def test_the_template_names_are_what_a_scenario_fills_in() -> None:
    names = set(template_values(Scenario("x", 15, 12, 0)))
    assert names == TEMPLATE_NAMES
    assert names == {"ordered", "received", "discount_pct", "grn_count", "deliveries"}


def test_the_work_order_has_eleven_steps_in_canon_order() -> None:
    steps = load_work_orders()
    assert [s.id for s in steps][:3] == [
        "DE-SM-02:draft_quote", "DE-OP-01:check_availability", "P2:decide_quote"
    ]
    assert len(steps) == 11
    assert {s.kind for s in steps} == {"de", "persona", "env"}


def test_inputs_are_rendered_from_the_scenario() -> None:
    steps = {s.id: s for s in load_work_orders()}
    s = Scenario("x", 16, 13, 4, grn_count=2)
    assert render_inputs(steps["DE-SM-02:draft_quote"], s)["lines"] == "ITEM-0005:20,ITEM-0012:16"
    assert render_inputs(steps["DE-SM-02:draft_quote"], s)["discount_pct"] == "4"
    receipt = render_inputs(steps["DE-OP-02:record_goods_receipt"], s)
    assert receipt == {"ma_po": "PO-0015", "received": "13",
                       "deliveries": "GRN-0015:7,GRN-0016:6"}


def test_slice_tables_drop_the_keys() -> None:
    steps = {s.id: s for s in load_work_orders()}
    assert steps["DE-SM-02:draft_quote"].tables == frozenset({"so", "soline", "apr", "evt"})
