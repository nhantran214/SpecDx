"""One interpreter, one scenario, one specification: a cached Behaviour."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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
from specdx.types import Behaviour, InfrastructureError, InterpreterId, Scenario, State
from specdx.variant import SpecVariant

ExecutorFactory = Callable[[InterpreterId, TellCorpus, str], DeExecutor]
RUN_CLOCK = datetime(2026, 9, 1, 2, 0, tzinfo=UTC)
# Bump on any interpreter-visible change to prompts, flow or executor behaviour: runs
# cached under an older version are then never reused.
RUN_SCHEMA_VERSION = 3  # 3: the neutral harness (spec v2 §2.1)
log = logging.getLogger(__name__)


def context_digest(
    tell: Mapping[str, Any], settings: Mapping[str, Any], *, schema: int | None = None
) -> str:
    """The part of every run key that covers what interpreters are told (the Tell
    variants), how they are run (executor settings) and the code (RUN_SCHEMA_VERSION).
    schema: an older RUN_SCHEMA_VERSION, to find the runs an older version cached."""
    version = RUN_SCHEMA_VERSION if schema is None else schema
    payload = {"schema": version, "tell": tell, "executor": settings}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class RunRequest:
    """One run a caller wants: what run_batch takes. kind (population, probe, ddmin or
    screen) only labels the log line; it is not in the run key."""

    variant: SpecVariant
    scenario: Scenario
    interp: InterpreterId
    suffix: str = ""
    kind: str = "population"


class Runner:
    def __init__(
        self,
        db: PackDatabase,
        cache: RunCache,
        factory: ExecutorFactory,
        steps: Sequence[WorkStep],
        *,
        model_digest: Callable[[str], str] | None = None,
        context_digest: str = "",
    ) -> None:
        self.db, self.cache, self.factory, self.steps = db, cache, factory, tuple(steps)
        self.model_digest = model_digest or (lambda model: model)
        self.context_digest = context_digest
        self._steps_digest = hashlib.sha256(repr(self.steps).encode()).hexdigest()
        self.executed = 0

    def key(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = ""
    ) -> str:
        return run_key(
            variant.digest(), self._steps_digest, self.context_digest,
            json.dumps(scenario.params(), sort_keys=True), interp.key,
            suffix, self.model_digest(interp.model),
        )

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = "",
        *, kind: str = "population",
    ) -> Behaviour:
        key = self.key(variant, scenario, interp, suffix)
        cached = self.cache.get(key)
        if cached is not None:
            # The key holds the parameters, not the scenario's name or the family: the
            # caller's own scenario and interpreter are the ones that ran here.
            return dataclasses.replace(cached, scenario=scenario, interpreter=interp)
        self.executed += 1
        behaviour = self.execute(RunRequest(variant, scenario, interp, suffix, kind), key)
        self.cache.put(behaviour)
        return behaviour

    def run_batch(self, requests: Sequence[RunRequest]) -> list[Behaviour]:
        """One after another, in order: the sequential form of ParallelRunner.run_batch."""
        return [self.run(r.variant, r.scenario, r.interp, r.suffix, kind=r.kind)
                for r in requests]

    def execute(self, request: RunRequest, key: str) -> Behaviour:
        """Make one run on this Runner's database, outside the cache: the caller stores it.
        Infrastructure failures are raised, never returned as behaviour."""
        variant, scenario, interp, suffix = (request.variant, request.scenario, request.interp,
                                             request.suffix)
        digest = self.model_digest(interp.model)
        started = time.monotonic()
        executor: DeExecutor | None = None
        try:
            settings = self.db.use(variant.pack)
            executor = self.factory(interp, variant.corpus, suffix)
            if hasattr(executor, "infra_failure"):
                executor.infra_failure = None
            with self.db.isolated() as conn:
                runtime = build_runtime(conn, settings, executor=executor,
                                        clock=FixedClock(RUN_CLOCK))
                case = run_case(runtime, scenario, self.steps)
                # DeRuntime turns an Ollama outage into a step error: read the side channel.
                failure = getattr(executor, "infra_failure", None)
                if failure:
                    raise InfrastructureError(failure)
                state = capture_state(conn, runtime.pack, case.statuses)
            behaviour = Behaviour(
                run_key=key, interpreter=interp, scenario=scenario, state=state,
                refused=tuple(case.refused), claims=dict(case.claims), model_digest=digest,
                harness=dict(case.harness), spec=variant.name,
            )
        except (psycopg.OperationalError, psycopg.InterfaceError, InfrastructureError):
            raise
        except Exception as exc:  # noqa: BLE001 - a failed run is data, never dropped
            failure = getattr(executor, "infra_failure", None)
            if failure:  # Ollama had already failed: whatever broke next is not behaviour
                raise InfrastructureError(failure) from exc
            behaviour = Behaviour(
                run_key=key, interpreter=interp, scenario=scenario, state=State.empty(),
                status="error", error=f"{type(exc).__name__}: {exc}", model_digest=digest,
                spec=variant.name,
            )
        log.info("run %s %s %s %s %s %.1f s", variant.name, scenario.id, interp.key,
                 request.kind, behaviour.status, time.monotonic() - started)
        return behaviour
