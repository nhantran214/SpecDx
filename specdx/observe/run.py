"""One interpreter, one scenario, one specification: a cached Behaviour."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import psycopg
from tonomyx.contracts.clock import FixedClock
from tonomyx.de.executor import DeExecutor
from tonomyx.de.runtime import build_runtime

from specdx.db import PackDatabase
from specdx.interpreters.cache import RunCache, run_key
from specdx.observe.state import capture_state
from specdx.orchestrate.flow import run_case
from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InterpreterId, Scenario, State
from specdx.variant import SpecVariant

ExecutorFactory = Callable[[InterpreterId, TellCorpus, str], DeExecutor]
RUN_CLOCK = datetime(2026, 9, 1, 2, 0, tzinfo=UTC)


class Runner:
    def __init__(
        self,
        db: PackDatabase,
        cache: RunCache,
        factory: ExecutorFactory,
        steps: Sequence[WorkStep],
        *,
        model_digest: Callable[[str], str] | None = None,
    ) -> None:
        self.db, self.cache, self.factory, self.steps = db, cache, factory, tuple(steps)
        self.model_digest = model_digest or (lambda model: model)
        self._steps_digest = hashlib.sha256(repr(self.steps).encode()).hexdigest()
        self.executed = 0

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = ""
    ) -> Behaviour:
        key = run_key(
            variant.digest(), self._steps_digest,
            json.dumps(scenario.params(), sort_keys=True), interp.key,
            suffix, self.model_digest(interp.model),
        )
        cached = self.cache.get(key)
        if cached is not None:
            return dataclasses.replace(cached, scenario=scenario)
        self.executed += 1
        try:
            settings = self.db.use(variant.pack)
            executor = self.factory(interp, variant.corpus, suffix)
            with self.db.isolated() as conn:
                runtime = build_runtime(conn, settings, executor=executor,
                                        clock=FixedClock(RUN_CLOCK))
                case = run_case(runtime, scenario, self.steps)
                state = capture_state(conn, runtime.pack, case.statuses)
            behaviour = Behaviour(
                run_key=key, interpreter=interp, scenario=scenario, state=state,
                refused=tuple(case.refused), claims=dict(case.claims),
            )
        except (psycopg.OperationalError, psycopg.InterfaceError):
            raise
        except Exception as exc:  # noqa: BLE001 - a failed run is data, never dropped
            behaviour = Behaviour(
                run_key=key, interpreter=interp, scenario=scenario, state=State.empty(),
                status="error", error=f"{type(exc).__name__}: {exc}",
            )
        self.cache.put(behaviour)
        return behaviour
