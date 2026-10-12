import json
import random
import time
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from specdx.active.loop import run_specdx
from specdx.eval.pilot import GateConfig
from specdx.interpreters.cache import RunCache
from specdx.observe.run import RunRequest
from specdx.orchestrate.reference import reference_executor
from specdx.parallel import ParallelRunner, worker_url
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InfrastructureError, InterpreterId, Scenario, State
from specdx.variant import SpecVariant
from specdx.workbench import Workbench
from tests.test_end_to_end import Deviant

VARIANT = SpecVariant("v", Path("."), TellCorpus([]))
S = Scenario("s", 15, 12, 0)


class _Worker:
    """Runs take a random while; every execution is logged in the order it starts."""

    model_digest = staticmethod(lambda model: f"d-{model}")

    def __init__(self, started: list[str], rng: random.Random, broken: str = "") -> None:
        self.started, self.rng, self.broken = started, rng, broken

    def key(self, variant, scenario, interp, suffix=""):
        return f"{scenario.id}|{interp.key}|{suffix}"

    def execute(self, request, key):
        self.started.append(key)
        time.sleep(self.rng.random() / 50)
        if request.interp.model == self.broken:
            raise InfrastructureError("Ollama failed 5 times")
        return Behaviour(key, request.interp, request.scenario, State({}))


def _requests(*models: str) -> list[RunRequest]:
    return [RunRequest(VARIANT, S, InterpreterId(m, "f", 0, i)) for i, m in enumerate(models)]


def test_worker_databases_live_in_the_same_server() -> None:
    info = conninfo_to_dict(worker_url("postgresql://u:p@localhost:55434/tonomyx", 2))
    assert (info["dbname"], info["port"], info["host"]) == ("tonomyx_w2", "55434", "localhost")


def test_a_batch_comes_back_in_submission_order_and_is_cached_by_one_writer(tmp_path) -> None:
    started: list[str] = []
    cache = RunCache(tmp_path / "runs.jsonl")
    runner = ParallelRunner([_Worker(started, random.Random(i)) for i in range(3)], cache)
    requests = _requests("b", "a", "b", "c", "a", "c")
    out = runner.run_batch(requests)
    assert [b.interpreter for b in out] == [r.interp for r in requests]
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["run_key"] for x in lines] == [b.run_key for b in out]
    again = runner.run_batch(requests[:2] + requests[:1])
    assert runner.executed == 6 and [b.run_key for b in again] == [
        out[0].run_key, out[1].run_key, out[0].run_key]


def test_one_worker_executes_the_batch_grouped_by_model(tmp_path) -> None:
    started: list[str] = []
    runner = ParallelRunner([_Worker(started, random.Random(0))], RunCache(tmp_path / "r.jsonl"))
    runner.run_batch(_requests("b", "a", "b", "a"))
    assert [key.split("|")[1] for key in started] == ["a", "a", "b", "b"]


def test_an_infrastructure_failure_stops_the_batch_and_is_never_cached(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    runner = ParallelRunner([_Worker([], random.Random(i), broken="x") for i in range(2)],
                            RunCache(path))
    with pytest.raises(InfrastructureError):
        runner.run_batch(_requests("a", "x", "b"))
    cached = [json.loads(x)["interpreter"]["model"] for x in path.read_text().splitlines()]
    assert cached == ["a"]  # collected before the failure; nothing after it is written


CONFIG = GateConfig(
    models_by_family={"ref": ("reference",), "dev": ("deviant",)},
    variants=(0,), seeds=(1, 2), budget=12, dev_per_type=1, rng_seed=1, probe_budget=6,
    localize_budget=6,
)


def _report(work_dir: Path, workers: int) -> tuple[str, int]:
    bench = Workbench.open(
        work_dir, workers=workers, model_digest=lambda m: f"digest-{m}",
        factory=lambda i, c, s: Deviant() if i.model == "deviant" else reference_executor(),
    )
    config = CONFIG.specdx()
    report = run_specdx(bench.base, bench.loop_context(bench.base, config), config)
    return report.to_json(), report.executed


@pytest.mark.db
def test_three_workers_give_a_byte_identical_report(db, tmp_path) -> None:
    one, executed_one = _report(tmp_path / "one", 1)
    three, executed_three = _report(tmp_path / "three", 3)
    assert one == three and executed_one == executed_three > 0
    with psycopg.connect(db.url) as conn:
        names = {r[0] for r in conn.execute("SELECT datname FROM pg_database").fetchall()}
    assert {"tonomyx_w0", "tonomyx_w1", "tonomyx_w2"} <= names


@pytest.mark.db
def test_workers_take_the_ollama_hosts_round_robin(db, tmp_path) -> None:
    bench = Workbench.open(tmp_path, workers=3, hosts=["http://h0:11434", "http://h1:11434"],
                           model_digest=lambda m: m)
    interp = InterpreterId("m", "f", 0, 1)
    urls = [w.factory(interp, bench.base.corpus, "").base_url for w in bench.runner.workers]
    assert urls == ["http://h0:11434", "http://h1:11434", "http://h0:11434"]
