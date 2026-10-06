"""Run CASE-0031 for any scenario through TonomyX's public runtime.

The vendored run_case takes two parameters and hard-codes the rest; this runner reads
the work order instead. Digital-employee steps go through DeRuntime.run, so every gate
applies. Persona steps go through the same executor with a human context. Environment
steps reuse TonomyX's own recorders, which write the canon's fixed facts.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope, TaskScope
from tonomyx.de.runtime import DeRuntime
from tonomyx.flow.case_0031 import CASE_ID, _record_inspection, _record_shipment, open_case

from specdx.orchestrate.work_orders import WorkStep, render_inputs
from specdx.types import Scenario


@dataclass
class CaseRun:
    statuses: dict[str, str] = field(default_factory=dict)
    claims: dict[str, str] = field(default_factory=dict)
    refused: list[tuple[str, str]] = field(default_factory=list)


def _environment(runtime: DeRuntime, step: WorkStep, scenario: Scenario) -> tuple[str, str]:
    try:
        with runtime.repo.conn.transaction():
            if step.intent == "inspection":
                _record_inspection(runtime, scenario.received)
            elif step.intent == "shipment":
                _record_shipment(runtime)
            else:
                raise ValueError(f"unknown environment step {step.intent!r}")
    except Exception as exc:  # noqa: BLE001 - a failed step is recorded, never swallowed
        return ResultStatus.ERROR.value, f"{type(exc).__name__}: {exc}"
    return ResultStatus.DONE.value, ""


def _persona(runtime: DeRuntime, step: WorkStep, inputs: dict[str, str]) -> ResultEnvelope:
    ctx = runtime.human_context(step.actor)
    task = TaskEnvelope(
        task_id=f"H-{uuid.uuid4().hex[:12]}",
        case_id=CASE_ID,
        de=step.actor,
        intent=step.intent,
        inputs=dict(inputs),
        scope=TaskScope(tools=sorted(ctx.token.tools), read=sorted(ctx.token.read)),
        idempotency_key=f"{step.actor}:{step.intent}:{CASE_ID}",
        deadline=runtime.clock.now() + timedelta(seconds=300),
    )
    try:
        with runtime.repo.conn.transaction():
            return runtime.executor.run(task, ctx)
    except Exception as exc:  # noqa: BLE001 - same contract as DeRuntime.run
        return ResultEnvelope(
            task_id=task.task_id, case_id=CASE_ID, status=ResultStatus.ERROR,
            next_hint=f"{type(exc).__name__}: {exc}",
        )


def run_case(runtime: DeRuntime, scenario: Scenario, steps: Sequence[WorkStep]) -> CaseRun:
    run = CaseRun()
    open_case(runtime.repo)
    for step in steps:
        if step.kind == "env":
            status, hint = _environment(runtime, step, scenario)
        else:
            inputs = render_inputs(step, scenario)
            envelope = (
                runtime.run(step.actor, step.intent, CASE_ID, dict(inputs))
                if step.kind == "de"
                else _persona(runtime, step, inputs)
            )
            status, hint = envelope.status.value, envelope.next_hint
            run.refused += [
                (step.id, str(call.get("name", "")))
                for call in envelope.usage.get("tool_calls", [])
                if call.get("outcome") != "ok"
            ]
        run.statuses[step.id] = status
        run.claims[step.id] = hint
    return run
