"""The tool registry: one gate every tool call passes through.

The nine names come from the canon pack's tool-catalog.json. Registering a name
outside that list is refused, so the action surface cannot grow by accident;
widening it is a deliberate edit to the canon projection, not a code change here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tonomyx.canon.pack import CanonPack
from tonomyx.contracts.clock import Clock
from tonomyx.contracts.token import TaskToken, UserToken
from tonomyx.db.repo import Repository
from tonomyx.policy.store import PolicyStore
from tonomyx.tools.ports import IdempotencyStore, MailSender

CANON_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_records",
        "write_draft",
        "update_status",
        "create_approval",
        "decide_approval",
        "log_event",
        "search_kb",
        "send_email",
        "match_three_way",
    }
)


class ToolDenied(PermissionError):
    """The call was refused by a gate."""


class ToolNotFound(KeyError):
    """No implementation is registered under that name."""


@dataclass(frozen=True)
class ToolContext:
    """Everything a tool is allowed to reach."""

    repo: Repository
    pack: CanonPack
    policy: PolicyStore
    token: TaskToken
    clock: Clock
    idempotency: IdempotencyStore
    user_token: UserToken | None = None
    mail: MailSender | None = None


def require_tool(ctx: ToolContext, name: str) -> None:
    """Refuse a call before it does anything else, if the token does not carry it.

    The catalog's gating column (tool-catalog.json) describes each tool's business
    gate, not whether a worker may call the tool at all -- de_scope.tools is what
    answers that question. Every one of the nine canon tools calls this first, so
    the answer is enforced at every real call site, not only at ToolRegistry.invoke,
    which no production code path actually goes through.
    """
    if name not in ctx.token.tools:
        raise ToolDenied(f"{ctx.token.sub} may not call {name}: not in task token")


ToolFn = Callable[..., dict[str, Any]]


class ToolRegistry:
    """Maps a canon tool name to its implementation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if name not in CANON_TOOL_NAMES:
            raise ValueError(f"{name} is not one of the nine canon tools")
        self._tools[name] = fn

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def get(self, name: str) -> ToolFn:
        """The implementation registered under a name, for introspection only.

        Calling it directly bypasses the token check in invoke(); use invoke() to run it.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(name) from None

    def invoke(self, name: str, ctx: ToolContext, **kwargs: Any) -> dict[str, Any]:
        if name not in ctx.token.tools:
            raise ToolDenied(f"{ctx.token.sub} may not call {name}: not in task token")
        try:
            fn = self._tools[name]
        except KeyError:
            raise ToolNotFound(name) from None
        return fn(ctx, **kwargs)
