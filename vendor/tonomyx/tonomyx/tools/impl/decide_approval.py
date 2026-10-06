"""decide_approval — a human decides one gate.

Three things are checked, all failing closed. There must be a user token, because a
digital employee may never decide its own gate. The persona must own the approval
type, which is the rule policy.rego states and this port repeats. And the approval
must still be pending, so a decision is never quietly overwritten.

The signature takes one approval, never a list. The canon forbids batching over-cap
decisions, and the simplest way to honour that is to have no batch path at all.

A replay must match what was actually decided. write_draft solves the same problem
by fingerprinting its payload and refusing a mismatch; this follows that shape,
fingerprinting the approval id, the decision and the deciding persona, so an
idempotency key already spent on one decision can never be reused to hand back a
different approval's outcome.
"""

from __future__ import annotations

import json
from typing import Any

from tonomyx.policy.rules import (
    APPROVED,
    PENDING,
    REJECTED,
    allow_decide,
    approval_type_of,
)
from tonomyx.tools.registry import ToolContext, ToolDenied, require_tool

DECISIONS: frozenset[str] = frozenset({APPROVED, REJECTED})

# Keys this tool uses inside the dict it hands the idempotency store. These are
# this tool's own wrapping, not part of the IdempotencyStore protocol, which only
# ever sees an opaque dict[str, Any].
_ROW_FIELD = "row"
_FINGERPRINT_FIELD = "fingerprint"


def _fingerprint(apr: str, decision: str, persona: str) -> str:
    """A stable fingerprint of what is being decided, for replay comparison."""
    return json.dumps({"apr": apr, "decision": decision, "persona": persona}, sort_keys=True)


def decide_approval(
    ctx: ToolContext, *, apr: str, decision: str, reason: str, idempotency_key: str
) -> dict[str, Any]:
    """Approve or reject one pending approval."""
    require_tool(ctx, "decide_approval")
    if ctx.user_token is None:
        raise ToolDenied("decide_approval needs a user token: a person must decide the gate")
    if not isinstance(apr, str):
        raise ToolDenied("decide_approval takes one approval at a time, never a batch")
    if decision not in DECISIONS:
        raise ToolDenied(f"unknown decision {decision!r}; expected one of {sorted(DECISIONS)}")

    fingerprint = _fingerprint(apr, decision, ctx.user_token.persona)
    seen = ctx.idempotency.seen(idempotency_key)
    if seen is not None:
        if seen[_FINGERPRINT_FIELD] != fingerprint:
            raise ToolDenied(
                f"idempotency key {idempotency_key!r} was already used for a different "
                "approval, decision or persona"
            )
        return {"row": seen[_ROW_FIELD], "replayed": True}

    row = ctx.repo.select_one("apr", where={"ma_apr": apr})
    if row is None:
        raise ToolDenied(f"no approval {apr}")
    if row["trang_thai"] != PENDING:
        raise ToolDenied(f"{apr} is already {row['trang_thai']!r} and may not be decided again")

    apr_type = approval_type_of(str(row["loai"]))
    if apr_type is None:
        raise ToolDenied(f"{apr}: approval label {row['loai']!r} maps to no policy type")

    verdict = allow_decide(apr_type, ctx.user_token.persona)
    if not verdict.allowed:
        raise ToolDenied(f"{apr}: {verdict.reason}")

    updated = ctx.repo.update("apr", key={"ma_apr": apr}, changes={"trang_thai": decision})
    ctx.idempotency.record(idempotency_key, {_ROW_FIELD: updated, _FINGERPRINT_FIELD: fingerprint})
    return {"row": updated, "replayed": False, "reason": reason}
