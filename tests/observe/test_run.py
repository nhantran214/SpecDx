import logging

import httpx
import psycopg
import pytest
from tonomyx.tools.catalog import build_registry

from specdx.config import BASE_PACK
from specdx.interpreters.cache import RunCache
from specdx.interpreters.executor import SpecInterpreterExecutor
from specdx.interpreters.prompt import load_variants
from specdx.observe.run import Runner
from specdx.orchestrate.reference import reference_executor
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import canon_scenarios
from specdx.spec.clauses import ClauseSet
from specdx.spec.tell import TellCorpus
from specdx.types import InfrastructureError, InterpreterId, Scenario
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


def test_a_different_context_digest_never_reuses_a_cached_run(db, pack, variant, tmp_path) -> None:
    cache = RunCache(tmp_path / "r.jsonl")
    grn12 = canon_scenarios(pack)["grn12"]

    def runner(context: str) -> Runner:
        return Runner(db, cache, lambda i, c, s: reference_executor(), STEPS,
                      model_digest=lambda m: "d", context_digest=context)

    first, second, same_as_first = runner("tell-a"), runner("tell-b"), runner("tell-a")
    key_a = first.run(variant, grn12, REF).run_key
    key_b = second.run(variant, grn12, REF).run_key
    assert key_a != key_b and second.executed == 1
    same_as_first.run(variant, grn12, REF)
    assert same_as_first.executed == 0


def test_each_executed_run_is_logged_once(db, pack, variant, tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="specdx")
    runner = Runner(db, RunCache(tmp_path / "r.jsonl"), lambda i, c, s: reference_executor(),
                    STEPS, model_digest=lambda m: "d")
    grn12 = canon_scenarios(pack)["grn12"]
    runner.run(variant, grn12, REF, kind="probe")
    runner.run(variant, grn12, REF, kind="probe")  # served from the cache: not logged
    lines = [r.getMessage() for r in caplog.records if r.name == "specdx.observe.run"]
    assert len(lines) == 1
    assert all(part in lines[0] for part in ("base", "grn12", REF.key, "probe", "ok", " s"))


def _ollama(handle):
    """An interpreter factory whose Ollama is a MockTransport; sleeps are recorded, not slept."""
    sleeps: list[float] = []

    def factory(interp, corpus, suffix):
        return SpecInterpreterExecutor(
            interp.model, build_registry(), corpus, load_variants(),
            transport=httpx.MockTransport(handle), sleep=sleeps.append,
        )

    return factory, sleeps


def test_an_ollama_outage_raises_and_is_never_cached(db, pack, variant, tmp_path) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    factory, sleeps = _ollama(handle)
    cache = RunCache(tmp_path / "r.jsonl")
    runner = Runner(db, cache, factory, STEPS, model_digest=lambda m: "d")
    with pytest.raises(InfrastructureError, match="ConnectError"):
        runner.run(variant, canon_scenarios(pack)["grn12"], InterpreterId("m", "f", 0, 1))
    assert len(cache) == 0 and not (tmp_path / "r.jsonl").exists()
    assert sleeps == [2.0, 4.0, 8.0, 16.0] and len(calls) == 5


def test_a_server_error_that_recovers_is_retried_and_cached(db, pack, variant, tmp_path) -> None:
    replies = [httpx.Response(503, json={"error": "llama runner process has terminated"})]

    def handle(request: httpx.Request) -> httpx.Response:
        if replies:
            return replies.pop(0)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Xong."}})

    factory, sleeps = _ollama(handle)
    cache = RunCache(tmp_path / "r.jsonl")
    runner = Runner(db, cache, factory, STEPS, model_digest=lambda m: "d")
    b = runner.run(variant, canon_scenarios(pack)["grn12"], InterpreterId("m", "f", 0, 1))
    assert b.status == "ok" and len(cache) == 1 and sleeps == [2.0]
