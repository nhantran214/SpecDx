"""Task and Result envelopes — the contract for every digital-employee call.

Field names and the required set are taken from the canon pack's
targets/tonomyx/task-envelope.schema.json and result-envelope.schema.json.
tests/contracts/test_envelope.py validates what we produce against those files,
so the two cannot drift apart silently.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskScope(BaseModel):
    """The least privilege a task runs under, copied from the DE's de_scope row."""

    tools: list[str] = Field(default_factory=list)
    read: list[str] = Field(default_factory=list)
    draft: list[str] = Field(default_factory=list)
    kb_domain: str = ""


class TaskEnvelope(BaseModel):
    """One unit of work handed to a digital employee."""

    task_id: str
    case_id: str
    de: str
    intent: str
    inputs: dict[str, Any]
    scope: TaskScope
    idempotency_key: str
    deadline: datetime


class ResultStatus(StrEnum):
    """The four outcomes the canon allows."""

    DONE = "done"
    NEEDS_GATE = "needs_gate"
    DENIED = "denied"
    ERROR = "error"


class ResultEnvelope(BaseModel):
    """What a digital employee hands back, including what it could not do."""

    task_id: str
    case_id: str
    status: ResultStatus
    drafts: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    next_hint: str = ""


def denied(task: TaskEnvelope, reason: str) -> ResultEnvelope:
    """Refuse a task, saying why. Never a silent success."""
    return ResultEnvelope(
        task_id=task.task_id,
        case_id=task.case_id,
        status=ResultStatus.DENIED,
        next_hint=reason,
    )
