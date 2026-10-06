import dataclasses

import pytest

from specdx.active.refute import CombinedRefuter, DirectedProbe, SymbolicRepair
from specdx.config import BASE_PACK
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import Clause, ClauseSet
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InterpreterId, Scenario, State
from specdx.variant import SpecVariant

pytestmark = pytest.mark.db
STEPS = load_work_orders()
GRN12 = Scenario("grn12", 15, 12, 0)


@pytest.fixture
def clauses(pack) -> ClauseSet:
    return ClauseSet.from_pack(pack, STEPS)


@pytest.fixture
def variant(pack) -> SpecVariant:
    return SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))


def _done(golden: State, cells: dict[tuple[str, str], str] | None = None) -> State:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    for (table, column), value in (cells or {}).items():
        tables[table][0][column] = value
    return State(tables, dict(golden.derived), {s.id: "done" for s in STEPS})


def test_a_hidden_requirement_is_refuted_by_repair(db, clauses, variant, golden) -> None:
    start = _done(golden, {("so", "trang_thai"): "Đã duyệt"})
    target = frozenset({"golden:so:SO-0031:trang_thai"})
    assert clauses.violations(start, GRN12) == target
    found = SymbolicRepair(db, variant, clauses).search(target, GRN12, [start])
    assert found is not None and found.tables["so"][0]["trang_thai"] == "Chờ duyệt"


def test_reconciliation_over_two_receipts_cannot_be_repaired(db, clauses, variant, golden) -> None:
    two = Scenario("x", 15, 12, 0, grn_count=2)
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["grn"] = [{"ma_grn": "GRN-0015", "po": "PO-0015", "sl_nhan": "6"},
                     {"ma_grn": "GRN-0016", "po": "PO-0015", "sl_nhan": "6"}]
    start = db.materialize(State(tables, {}, {s.id: "done" for s in STEPS}))
    target = frozenset({"reconcile:consistency"})
    assert "reconcile:consistency" in clauses.violations(start, two)
    assert SymbolicRepair(db, variant, clauses).search(target, two, [start]) is None


def test_a_conflicting_pair_cannot_be_repaired_without_new_violations(
    db, clauses, variant, golden
) -> None:
    extra = Clause(
        "inv:mut", "invariant", "anchored", "grn12", frozenset({"po"}), "PO đặt 18",
        {"invariant": {"id": "mut", "desc": "x", "table": "po", "where": {"ma_po": "PO-0015"},
                       "assert": {"col": "sl_dat", "eq": "18"}}},
    )
    mutant = ClauseSet([*clauses, extra], clauses.master, clauses.canon)
    start = _done(golden)
    assert mutant.violations(start, GRN12) == frozenset({"inv:mut"})
    repair = SymbolicRepair(db, variant, mutant)
    assert repair.search(frozenset({"inv:mut"}), GRN12, [start]) is None


def test_a_state_the_database_refuses_is_no_witness(db, clauses, variant, golden) -> None:
    bogus = Clause("event:BOGUS:SO-0031:AI", "golden_event", "anchored", "grn12",
                   frozenset({"evt"}), "bogus", {"event": ["BOGUS", "SO-0031", "AI"]})
    mutant = ClauseSet([*clauses, bogus], clauses.master, clauses.canon)
    target = frozenset({bogus.id})
    assert SymbolicRepair(db, variant, mutant).search(target, GRN12, [_done(golden)]) is None


class _FakeRunner:
    def __init__(self, state: State) -> None:
        self.state, self.calls = state, 0

    def run(self, variant, scenario, interp, suffix=""):
        self.calls += 1
        assert "Yêu cầu bổ sung" in suffix
        return Behaviour(f"probe{self.calls}", interp, scenario, self.state)


def test_the_probe_refutes_and_results_are_memoised(clauses, variant, golden) -> None:
    runner = _FakeRunner(_done(golden))
    interp = InterpreterId("m", "f", 0, 1)
    probe = DirectedProbe(runner, variant, clauses, [interp])
    refuter = CombinedRefuter(clauses, None, probe)
    start = Behaviour("s", interp, GRN12, _done(golden, {("so", "trang_thai"): "Đã duyệt"}))
    target = frozenset({"golden:so:SO-0031:trang_thai"})
    first = refuter.refute(target, GRN12, [start])
    again = refuter.refute(target, dataclasses.replace(GRN12, id="other"), [start])
    assert first.refuted and first.witness == "probe1"
    assert again == first and runner.calls == 1
