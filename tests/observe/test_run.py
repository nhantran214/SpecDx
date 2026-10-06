import psycopg
import pytest

from specdx.config import BASE_PACK
from specdx.interpreters.cache import RunCache
from specdx.observe.run import Runner
from specdx.orchestrate.reference import reference_executor
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import canon_scenarios
from specdx.spec.clauses import ClauseSet
from specdx.spec.tell import TellCorpus
from specdx.types import InterpreterId, Scenario
from specdx.variant import SpecVariant

pytestmark = pytest.mark.db
STEPS = load_work_orders()
REF = InterpreterId("reference", "reference", 0, 1)


@pytest.fixture
def variant(pack) -> SpecVariant:
    return SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))


def test_a_run_is_executed_once_then_served_from_cache(db, pack, variant, tmp_path) -> None:
    runner = Runner(db, RunCache(tmp_path / "r.jsonl"), lambda i, c, s: reference_executor(),
                    STEPS, model_digest=lambda m: "d")
    grn12 = canon_scenarios(pack)["grn12"]
    first = runner.run(variant, grn12, REF)
    assert first.status == "ok"
    assert ClauseSet.from_pack(pack, STEPS).violations(first.state, grn12) == frozenset()
    renamed = runner.run(variant, Scenario("o15-r12-d0-g1", 15, 12, 0), REF)
    assert runner.executed == 1
    assert renamed.scenario.id == "o15-r12-d0-g1" and renamed.state == first.state


def test_a_crashing_factory_yields_an_error_behaviour(db, pack, variant, tmp_path) -> None:
    def broken(i, c, s):
        raise RuntimeError("no model")

    runner = Runner(db, RunCache(tmp_path / "r.jsonl"), broken, STEPS, model_digest=lambda m: "d")
    b = runner.run(variant, canon_scenarios(pack)["grn12"], REF)
    assert b.status == "error" and "no model" in b.error


def test_a_database_outage_is_not_cached(db, pack, variant, tmp_path) -> None:
    def db_fails(i, c, s):
        raise psycopg.OperationalError("database offline")

    cache = RunCache(tmp_path / "r.jsonl")
    runner = Runner(db, cache, db_fails, STEPS, model_digest=lambda m: "d")
    with pytest.raises(psycopg.OperationalError):
        runner.run(variant, canon_scenarios(pack)["grn12"], REF)
    assert len(cache) == 0
