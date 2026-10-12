"""Parallel workers (spec v2 §5.1): the independent runs of one loop step, at once.

Worker i owns database <name>_w<i> in the same PostgreSQL server, created on demand, with
its own PackDatabase and advisory lock, so per-run isolation is unchanged. The caller
charges budgets before it dispatches a batch. Runs are dispatched grouped by model, to
limit model swaps on the GPU, and collected in submission order; this runner is the run
cache's only writer. A report is therefore byte-identical to a sequential run's.
"""

from __future__ import annotations

import dataclasses
import queue
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from specdx.interpreters.cache import RunCache
from specdx.observe.run import RunRequest
from specdx.types import Behaviour, InterpreterId, Scenario
from specdx.variant import SpecVariant


class Worker(Protocol):
    model_digest: Callable[[str], str]

    def key(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = ""
    ) -> str: ...

    def execute(self, request: RunRequest, key: str) -> Behaviour: ...


def worker_url(url: str, index: int) -> str:
    """The database URL of worker `index`: the same server, database <name>_w<index>."""
    name = conninfo_to_dict(url).get("dbname") or "tonomyx"
    return make_conninfo(url, dbname=f"{name}_w{index}")


def ensure_database(url: str, name: str) -> None:
    """Create database `name` in url's server unless it exists (url's own database is the
    one connected to)."""
    with psycopg.connect(url, autocommit=True) as conn:
        found = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if found is None:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


class ParallelRunner:
    """The Runner interface over several workers."""

    def __init__(self, workers: Sequence[Worker], cache: RunCache) -> None:
        if not workers:
            raise ValueError("a ParallelRunner needs at least one worker")
        self.workers, self.cache = list(workers), cache
        self.model_digest = self.workers[0].model_digest
        self.executed = 0
        self._free: queue.Queue[Worker] = queue.Queue()
        for worker in self.workers:
            self._free.put(worker)
        self._pool = ThreadPoolExecutor(max_workers=len(self.workers),
                                        thread_name_prefix="specdx-worker")

    def _execute(self, request: RunRequest, key: str) -> Behaviour:
        worker = self._free.get()
        try:
            return worker.execute(request, key)
        finally:
            self._free.put(worker)

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = "",
        *, kind: str = "population",
    ) -> Behaviour:
        return self.run_batch([RunRequest(variant, scenario, interp, suffix, kind)])[0]

    def run_batch(self, requests: Sequence[RunRequest]) -> list[Behaviour]:
        keys = [self.workers[0].key(r.variant, r.scenario, r.interp, r.suffix) for r in requests]
        found: dict[str, Behaviour] = {}
        todo: dict[str, RunRequest] = {}
        for request, key in zip(requests, keys, strict=True):
            cached = self.cache.get(key)
            if cached is not None:
                found[key] = cached
            elif key not in todo:
                todo[key] = request
        # sorted() is stable: within a model, runs keep their submission order.
        dispatch = sorted(todo, key=lambda k: todo[k].interp.model)
        futures = {k: self._pool.submit(self._execute, todo[k], k) for k in dispatch}
        for key in todo:  # submission order: one writer, the same cache file every time
            try:
                behaviour = futures[key].result()
            except BaseException:
                for future in futures.values():
                    future.cancel()
                raise
            self.cache.put(behaviour)
            self.executed += 1
            found[key] = behaviour
        return [dataclasses.replace(found[k], scenario=r.scenario, interpreter=r.interp)
                for r, k in zip(requests, keys, strict=True)]

    def close(self) -> None:
        self._pool.shutdown(wait=True)
