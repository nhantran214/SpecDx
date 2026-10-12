import dataclasses

import pytest
from tonomyx.canon.pack import CanonPack

from specdx.active.budget import RunBudget
from specdx.active.refute import (
    CombinedRefuter,
    DirectedProbe,
    SymbolicRepair,
    _facts_kept,
    propose,
)
from specdx.config import BASE_PACK
from specdx.mutate.operators import all_mutants
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

    def run(self, variant, scenario, interp, suffix="", *, kind="population"):
        self.calls += 1
        assert "Yêu cầu bổ sung" in suffix and kind == "probe"
        return Behaviour(f"probe{self.calls}", interp, scenario, self.state)

    def run_batch(self, requests):
        return [self.run(r.variant, r.scenario, r.interp, r.suffix, kind=r.kind) for r in requests]


def test_the_probe_budget_caps_directed_probe_runs(clauses, variant) -> None:
    runner = _FakeRunner(State.empty())  # every probe fails: nothing satisfies the targets
    budget = RunBudget(probe_limit=4, ddmin_limit=80)
    probe = DirectedProbe(runner, variant, clauses, [InterpreterId("m", "f", 0, 1)],
                          budget=budget)
    refuter = CombinedRefuter(clauses, None, probe)
    first = refuter.refute(frozenset({"golden:so:SO-0031:trang_thai"}), GRN12, [])
    assert not first.refuted and first.attempted and runner.calls == 3
    second = refuter.refute(frozenset({"golden:po:PO-0015:trang_thai"}), GRN12, [])
    assert second.attempted and runner.calls == 4
    third = refuter.refute(frozenset({"golden:inv:INV-0031:trang_thai"}), GRN12, [])
    assert not third.refuted and not third.attempted
    assert runner.calls == 4 and budget.probe == 4 and budget.ddmin == 0
    assert budget.exhausted("probe") and not budget.exhausted("ddmin")


def test_the_probe_refutes_and_results_are_memoised(clauses, variant, golden) -> None:
    runner = _FakeRunner(_done(golden))
    interp = InterpreterId("m", "f", 0, 1)
    probe = DirectedProbe(runner, variant, clauses, [interp])
    refuter = CombinedRefuter(clauses, None, probe)
    start = Behaviour("s", interp, GRN12, _done(golden, {("so", "trang_thai"): "Đã duyệt"}))
    target = frozenset({"golden:so:SO-0031:trang_thai"})
    first = refuter.refute(target, GRN12, [start])
    again = refuter.refute(target, dataclasses.replace(GRN12, id="other"), [start])
    # all three attempts are dispatched together; the first in order is the witness
    assert first.refuted and first.witness == "probe1"
    assert again == first and runner.calls == 3


def test_repair_cannot_rewrite_scenario_facts(db, clauses, variant, golden) -> None:
    two = Scenario("x", 15, 12, 0, grn_count=2)
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["grn"] = [{"ma_grn": "GRN-0015", "po": "PO-0015", "sl_nhan": "6"},
                     {"ma_grn": "GRN-0016", "po": "PO-0015", "sl_nhan": "6"}]
    start = db.materialize(State(tables, {}, {s.id: "done" for s in STEPS}))
    extra = Clause(
        "inv:mut-grn", "invariant", "all", None, frozenset({"grn"}),
        "GRN-0015 sl_nhan eq 7",
        {"invariant": {"id": "mut-grn", "desc": "x", "table": "grn",
                       "where": {"ma_grn": "GRN-0015"},
                       "assert": {"col": "sl_nhan", "eq": "7"}}},
    )
    mutant = ClauseSet([*clauses, extra], clauses.master, clauses.canon)
    target = frozenset({"inv:mut-grn"})
    assert SymbolicRepair(db, variant, mutant).search(target, two, [start]) is None


def test_repair_can_fix_so_when_grn_starts_diverged(db, clauses, variant, golden) -> None:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["grn"] = [{"ma_grn": "GRN-0015", "po": "PO-0015", "sl_nhan": "10"}]
    start = db.materialize(State(tables, {}, {s.id: "done" for s in STEPS}))
    assert start is not None
    modified = _done(start, {("so", "trang_thai"): "Đã duyệt"})
    target = frozenset({"golden:so:SO-0031:trang_thai"})
    found = SymbolicRepair(db, variant, clauses).search(target, GRN12, [modified])
    assert found is not None
    assert found.tables["grn"][0]["sl_nhan"] == "10"


@pytest.fixture(scope="module")
def h2_mutants(pack, tmp_path_factory):
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))
    found = [m for m in all_mutants(base, [], tmp_path_factory.mktemp("work"))
             if m.operator == "H2"]
    assert [m.id for m in found] == ["H2-00", "H2-01"]
    return found


def test_repair_refutes_the_real_h2_mutants_with_one_event(db, golden, h2_mutants) -> None:
    for m in h2_mutants:
        mutant = ClauseSet.from_pack(CanonPack(m.variant.pack), STEPS)
        target = frozenset(m.truth.location)
        start = _done(golden)
        assert mutant.violations(start, GRN12) == target, m.id
        found = SymbolicRepair(db, m.variant, mutant).search(target, GRN12, [start])
        assert found is not None, m.id
        kind, ref = m.site.split(":")
        assert [e["ref"] for e in found.tables["evt"] if e["kind"] == kind] == [ref], m.id
        assert mutant.violations(found, GRN12) == frozenset(), m.id


def test_an_evt_count_proposal_appends_or_removes_one_matching_event() -> None:
    clause = Clause("inv:x", "invariant", "anchored", "grn12", frozenset({"evt"}), "x",
                    {"invariant": {"id": "x", "desc": "x", "table": "evt",
                                   "where": {"kind": "PAID", "ref": "PAY-0031"},
                                   "assert": {"count": 1}}})
    paid = {"case_id": "CASE-0031", "kind": "PAID", "ref": "PAY-0031", "actor": "AI",
            "basis": "b", "at": "2026-09-01 02:00:00+00"}
    other = {**paid, "kind": "DRAFTED"}
    [added] = propose(clause, State({"evt": [other]}), GRN12)
    assert [(e["kind"], e["ref"]) for e in added.tables["evt"]] == [
        ("DRAFTED", "PAY-0031"), ("PAID", "PAY-0031")]
    [removed] = propose(clause, State({"evt": [paid, other, dict(paid)]}), GRN12)
    assert [(e["kind"], e["ref"]) for e in removed.tables["evt"]] == [
        ("PAID", "PAY-0031"), ("DRAFTED", "PAY-0031")]
    assert propose(clause, State({"evt": [paid]}), GRN12) == []


def test_facts_kept_with_protected_table_rows() -> None:
    start = State({"ncr": [{"ma_ncr": "NCR-001", "mota": "test"}]})
    candidate = State({"ncr": [{"ma_ncr": "NCR-001", "mota": "test"}]})
    scenario = Scenario("test", 15, 12, 0)
    assert _facts_kept(candidate, start, scenario)
    changed = State({"ncr": [{"ma_ncr": "NCR-001", "mota": "changed"}]})
    assert not _facts_kept(changed, start, scenario)
