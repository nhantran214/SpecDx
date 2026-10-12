import pytest

from specdx.config import BASE_PACK
from specdx.db import golden_tables, split_background
from specdx.eval.observability import scorable, touch_counts, touches
from specdx.infer.act import drafting_steps
from specdx.infer.items import SpecLens
from specdx.mutate.operators import all_mutants, dev_split
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InterpreterId, Scenario, State, Target
from specdx.variant import SpecVariant

STEPS = load_work_orders()
GRN12 = Scenario("grn12", 15, 12, 0)
STEP = {s.id: s for s in STEPS}


@pytest.fixture(scope="module")
def lens(pack) -> SpecLens:
    background, _ = split_background(golden_tables(pack))
    return SpecLens(ClauseSet.from_pack(pack, STEPS), STEPS, ColumnModel.of(pack),
                    drafting=drafting_steps(pack, STEPS), initial=background)


@pytest.fixture(scope="module")
def dev(pack, tmp_path_factory):
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))
    return dev_split(all_mutants(base, [], tmp_path_factory.mktemp("work")), 5, 20261006)


def _runs(pack, n: int, tables=None) -> list[Behaviour]:
    state = State(golden_tables(pack) if tables is None else tables, {},
                  {s.id: "done" for s in STEPS})
    return [Behaviour(f"r{i}", InterpreterId("m", "f", 0, i), GRN12, state) for i in range(n)]


def test_every_mutant_declares_what_its_truth_needs(dev) -> None:
    assert all(m.truth.observed_at for m in dev)
    observed = {m.id: m.truth.observed_at for m in dev}
    assert observed["H1-02"] == (Target("row", "inv", "INV-0031"),)
    assert observed["W2-01"] == (Target("event", "FLAGGED", "ITEM-0012"),)
    assert observed["C2-00"] == (Target("row", "po", "PO-0015"),)
    assert Target("step", "P2:decide_quote") in observed["A3-00"]
    assert observed["W1-04"] == (Target("row", "vinv", "VINV-0015"),)
    assert observed["H2-00"] == (Target("event", "", "PAY-0031"),)  # any event about PAY-0031


def test_a_target_is_touched_only_from_a_step_whose_slice_covers_it(pack) -> None:
    state, columns = State(golden_tables(pack)), ColumnModel.of(pack)
    po = Target("row", "po", "PO-0015")
    assert touches(po, STEP["P6:decide_po"], state, columns)
    assert not touches(po, STEP["DE-SM-02:draft_quote"], state, columns)
    assert not touches(po, STEP["P6:decide_po"], State({}), columns)  # the row is not there
    flag = Target("event", "FLAGGED", "ITEM-0012")
    assert touches(flag, STEP["DE-OP-01:check_availability"], state, columns)
    assert touches(Target("step", "P2:decide_quote"), STEP["P2:decide_quote"], State({}), columns)


def test_golden_runs_make_every_development_mutant_scorable(pack, lens, dev) -> None:
    counts = touch_counts({m.id: m.truth.observed_at for m in dev}, _runs(pack, 3), lens)
    assert counts == {m.id: 3 for m in dev} and scorable(counts) == {m.id for m in dev}
    two = touch_counts({m.id: m.truth.observed_at for m in dev}, _runs(pack, 2), lens)
    assert scorable(two) == frozenset()  # fewer than k = 3 behaviours


def test_runs_that_write_nothing_reach_only_the_steps_that_draft_nothing(pack, lens, dev) -> None:
    counts = touch_counts({m.id: m.truth.observed_at for m in dev}, _runs(pack, 3, {}), lens)
    assert scorable(counts) == {"A1-01", "A2-01", "A2-02", "A3-00"}
