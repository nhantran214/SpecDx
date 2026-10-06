"""The scripted reference flow: TonomyX's handlers plus the two persona decisions.

Used only to prove the runner reproduces the canon. It is never an interpreter.
"""

from __future__ import annotations

from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope
from tonomyx.de.executor import ScriptedExecutor
from tonomyx.flow.handlers import PO, register_handlers
from tonomyx.tools.impl.decide_approval import decide_approval
from tonomyx.tools.impl.log_event import log_event
from tonomyx.tools.impl.update_status import update_status
from tonomyx.tools.registry import ToolContext


def _decide(ctx: ToolContext, task: TaskEnvelope, basis: str) -> dict[str, object]:
    apr = str(task.inputs["ma_apr"])
    decide_approval(
        ctx, apr=apr, decision=str(task.inputs["quyet_dinh"]), reason="",
        idempotency_key=f"decide:{apr}",
    )
    return log_event(
        ctx, case_id=task.case_id, kind="APPROVED", ref=apr, actor="Người", basis=basis
    )["row"]


def decide_quote(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """P2 approves the quote. The canon leaves SO-0031 pending afterwards."""
    event = _decide(ctx, task, "P2 duyệt báo giá")
    return ResultEnvelope(
        task_id=task.task_id, case_id=task.case_id, status=ResultStatus.DONE, events=[event]
    )


def decide_po(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """P6 approves the purchase order, which then advances."""
    event = _decide(ctx, task, f"P6 duyệt {PO} (maker-checker) trước khi phát hành")
    update_status(ctx, table="po", key={"ma_po": PO}, new_status="Đã duyệt")
    return ResultEnvelope(
        task_id=task.task_id, case_id=task.case_id, status=ResultStatus.DONE, events=[event]
    )


def reference_executor() -> ScriptedExecutor:
    executor = ScriptedExecutor()
    register_handlers(executor)
    executor.register("P2", "decide_quote", decide_quote)
    executor.register("P6", "decide_po", decide_po)
    return executor
