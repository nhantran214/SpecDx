from datetime import UTC, datetime

import pytest
from tonomyx.contracts.clock import FixedClock
from tonomyx.contracts.envelope import ResultEnvelope, TaskEnvelope, denied
from tonomyx.de.runtime import build_runtime
from tonomyx.tools.registry import ToolContext

from specdx.observe.state import capture_state
from specdx.orchestrate.flow import run_case
from specdx.orchestrate.reference import reference_executor
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import canon_scenarios
from specdx.spec.clauses import GOLDEN_TABLES, ILLUSTRATIVE_PREFIX, ClauseSet
from specdx.spec.columns import ColumnModel

pytestmark = pytest.mark.db
STEPS = load_work_orders()
CLOCK = FixedClock(datetime(2026, 9, 1, 2, 0, tzinfo=UTC))


def _run(db, scenario, executor):
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        case = run_case(runtime, scenario, STEPS)
        return case, capture_state(conn, db.pack, case.statuses)


def _chain(state):
    events = sorted(state.tables["evt"], key=lambda e: int(e["id"]))
    return [(e["kind"], e["ref"], e["actor"]) for e in events
            if not e["basis"].startswith(ILLUSTRATIVE_PREFIX)]


@pytest.mark.parametrize("name", ["grn12", "happy", "discount8"])
def test_the_reference_flow_satisfies_every_clause(db, pack, name) -> None:
    scenario = canon_scenarios(pack)[name]
    case, state = _run(db, scenario, reference_executor())
    assert set(case.statuses.values()) <= {"done", "needs_gate"}
    assert ClauseSet.from_pack(pack, STEPS).violations(state, scenario) == frozenset()


def test_the_reference_flow_reproduces_the_golden_state(db, pack, golden) -> None:
    _, state = _run(db, canon_scenarios(pack)["grn12"], reference_executor())
    model = ColumnModel.of(pack)
    assert model.project(state.tables, GOLDEN_TABLES) == model.project(golden.tables, GOLDEN_TABLES)
    assert _chain(state) == _chain(golden)


class _Without:
    """The reference flow with one intent refused."""

    def __init__(self, intent: str) -> None:
        self.inner = reference_executor()
        self.intent = intent

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        if task.intent == self.intent:
            return denied(task, "removed for the test")
        return self.inner.run(task, ctx)


def test_a_failed_step_does_not_poison_later_steps(db, pack) -> None:
    case, state = _run(db, canon_scenarios(pack)["grn12"], _Without("record_goods_receipt"))
    assert case.statuses["DE-OP-02:record_goods_receipt"] == "denied"
    assert case.statuses["ENV:inspection"] == "error"
    assert case.statuses["DE-FN-03:draft_payment"] == "needs_gate"
    assert [r["ma_pay"] for r in state.tables["pay"]] == ["PAY-0031"]
