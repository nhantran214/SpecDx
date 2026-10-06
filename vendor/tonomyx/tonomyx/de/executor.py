"""How a digital employee's work actually gets done.

The rest of the system talks to this protocol and never to a particular runtime, so
swapping the engine changes nothing else. Three implementations ship:

ScriptedExecutor runs deterministic handlers and is what the tests use, because an
acceptance suite that depends on a language model proves nothing repeatable.

NemoClawExecutor keeps NemoClaw's original design. NemoClaw is a sandbox stack, not
a multi-agent framework: it exposes no SDK, no REST and no gRPC, only a host CLI. So
the integration is a subprocess call and a JSON parse, and it lives behind this one
seam.

OllamaExecutor is the local-model path.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope
from tonomyx.tools.registry import ToolContext, ToolDenied, ToolNotFound, ToolRegistry
from tonomyx.tools.schema import tool_schemas

Handler = Callable[[ToolContext, TaskEnvelope], ResultEnvelope]


class DeExecutor(Protocol):
    """Anything that can carry out one task envelope."""

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope: ...


def _error(task: TaskEnvelope, reason: str) -> ResultEnvelope:
    return ResultEnvelope(
        task_id=task.task_id,
        case_id=task.case_id,
        status=ResultStatus.ERROR,
        next_hint=reason,
    )


class ScriptedExecutor:
    """Deterministic handlers, keyed by digital employee and intent."""

    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], Handler] = {}

    def register(self, de: str, intent: str, handler: Handler) -> None:
        self._handlers[(de, intent)] = handler

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        handler = self._handlers.get((task.de, task.intent))
        if handler is None:
            return ResultEnvelope(
                task_id=task.task_id,
                case_id=task.case_id,
                status=ResultStatus.DENIED,
                next_hint=f"{task.de} has no handler for intent {task.intent!r}",
            )
        return handler(ctx, task)


Runner = Callable[..., subprocess.CompletedProcess[str]]


class NemoClawExecutor:
    """Run one agent turn inside a NemoClaw sandbox through its host CLI.

    The argv follows NemoClaw's documented non-interactive contract
    (docs/reference/commands.mdx upstream): `nemoclaw <sandbox> agent` forwards to
    `openclaw agent` inside the sandbox, which needs a target selector (--agent),
    takes the message with -m, and with --json prints OpenClaw's own turn JSON.

    The agent inside the sandbox holds OpenClaw's tools, not the nine gated tools
    here. Whatever it replies, it has written nothing through them, so a reply is
    never reported as work done: the result is an error that keeps the reply for
    inspection. Doing real work through NemoClaw needs the gated tools exposed to
    the sandbox first, as the MCP tool service the canon's stack projection names.
    """

    def __init__(
        self,
        sandbox: str,
        *,
        agent: str = "main",
        timeout: int = 120,
        runner: Runner = subprocess.run,
    ) -> None:
        self.sandbox = sandbox
        self.agent = agent
        self.timeout = timeout
        self.runner = runner

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        argv = [
            "nemoclaw", self.sandbox, "agent",
            "--agent", self.agent,
            "--json",
            "--timeout", str(self.timeout),
            "-m", _task_message(task),
        ]
        try:
            completed = self.runner(argv, capture_output=True, text=True, check=False)
        except OSError as exc:
            return _error(task, f"could not start nemoclaw: {exc}")
        if completed.returncode != 0:
            return _error(task, completed.stderr.strip() or "nemoclaw exited non-zero")
        reply = completed.stdout.strip()
        return _error(
            task,
            "the NemoClaw sandbox agent cannot reach the gated tools, so its reply is not "
            f"work done; expose the nine tools to the sandbox first. Reply: {reply}"[
                :_HINT_LIMIT
            ],
        )


def _task_message(task: TaskEnvelope) -> str:
    """The task as one Vietnamese message for an agent outside this process."""
    return (
        f"Bạn là nhân viên số {task.de}. Nhiệm vụ: {task.intent}. Vụ việc: {task.case_id}. "
        f"Dữ liệu vào: {json.dumps(task.inputs, ensure_ascii=False)}."
    )


@dataclass(frozen=True)
class Playbook:
    """What a correct outcome for one intent is, for a model-backed executor.

    steps is the instruction a model is given. requires names the tools that must
    have succeeded at least once before the task counts as finished: a model that
    stops short is reminded, and one that stays short is reported as an error, never
    as done. A tool whose call was refused does not count.
    """

    steps: str
    requires: tuple[str, ...] = ()

    def missing(self, succeeded: frozenset[str]) -> tuple[str, ...]:
        return tuple(name for name in self.requires if name not in succeeded)


class OllamaExecutor:
    """Run a task with a local Ollama model that works through the gated tools.

    The model is offered only the tools in the task token, described by schemas
    derived from the tools themselves. Each call it makes goes through
    ToolRegistry.invoke, so every gate a scripted handler meets applies here too,
    and each call runs in its own savepoint, so a call that fails inside PostgreSQL
    rolls back alone and the model can correct course.

    The result envelope is assembled from what the tools actually did. A model that
    says it drafted a quote but never called write_draft reports no drafts, and a
    model that never makes a successful call at all reports an error: claiming work
    is not doing it.

    Prompts stay Vietnamese: the business vocabulary, the knowledge base and the
    status values are all Vietnamese, and translating them would make the model
    reason about words the data does not use.
    """

    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        *,
        base_url: str = "http://localhost:11434",
        max_steps: int = 12,
        timeout: float = 180.0,
        think: bool | None = None,
        transport: httpx.BaseTransport | None = None,
        playbooks: Mapping[tuple[str, str], Playbook] | None = None,
        max_reminders: int = 2,
    ) -> None:
        self.model = model
        self.playbooks = dict(playbooks or {})
        self.max_reminders = max_reminders
        self.tools = tools
        self.base_url = base_url
        self.max_steps = max_steps
        self.timeout = timeout
        self.think = think
        self.transport = transport

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        schemas = tool_schemas(self.tools, ctx.pack, names=ctx.token.tools)
        playbook = self.playbooks.get((task.de, task.intent))
        reminders_left = self.max_reminders
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(
                task, ctx, playbook.steps if playbook else None
            )},
            {"role": "user", "content": f"Thực hiện nhiệm vụ {task.intent} cho {task.case_id}."},
        ]
        record = _Record()
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
                for step in range(1, self.max_steps + 1):
                    reply = self._chat(client, messages, schemas)
                    messages.append(reply)
                    calls = reply.get("tool_calls") or []
                    if not calls:
                        missing = playbook.missing(record.succeeded_names()) if playbook else ()
                        if missing and reminders_left > 0:
                            reminders_left -= 1
                            messages.append(_reminder(missing))
                            continue
                        if missing:
                            return record.envelope(
                                task, self.model, step, ResultStatus.ERROR,
                                f"{self.model} stopped without a successful call to: "
                                f"{', '.join(missing)}",
                            )
                        return record.finish(task, self.model, step, _plain(reply))
                    for call in calls:
                        messages.append(self._execute(call, ctx, record))
        except httpx.HTTPError as exc:
            return _error(task, f"ollama call failed: {exc}")
        return record.envelope(
            task, self.model, self.max_steps, ResultStatus.ERROR,
            f"{self.model} did not finish within {self.max_steps} steps",
        )

    def _chat(
        self,
        client: httpx.Client,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": schemas,
            "stream": False,
            "options": {"temperature": 0},
        }
        if self.think is not None:
            body["think"] = self.think
        response = client.post(f"{self.base_url}/api/chat", json=body)
        response.raise_for_status()
        message: dict[str, Any] = response.json()["message"]
        return message

    def _execute(self, call: dict[str, Any], ctx: ToolContext, record: _Record) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        args: dict[str, Any] = {}
        try:
            args = _arguments(function.get("arguments"))
            invented = _invented_numbers(name, args, ctx.pack.non_canon_columns)
            if invented:
                raise ToolDenied(
                    f"the canon leaves {', '.join(invented)} blank: a model may not "
                    "supply a value there; pass null"
                )
            # A savepoint per call: a call that fails inside PostgreSQL must not
            # leave the transaction aborted for every call after it.
            with ctx.repo.conn.transaction():
                result = self.tools.invoke(name, ctx, **args)
            record.ok(name, args, result)
            payload: Any = result
        except ToolDenied as exc:
            record.failed(name, args, "denied")
            payload = {"error": "denied", "tool": name, "reason": str(exc)}
        except ToolNotFound:
            record.failed(name, args, "unknown")
            payload = {"error": "unknown tool", "tool": name}
        except Exception as exc:  # noqa: BLE001 - the model is told what broke and may retry
            record.failed(name, args, "error")
            payload = {"error": type(exc).__name__, "tool": name, "reason": str(exc)}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        return {"role": "tool", "tool_name": name, "content": content[:_TOOL_REPLY_LIMIT]}


# Longest tool reply handed back to the model, in characters.
_TOOL_REPLY_LIMIT = 4000

# Longest model summary kept as the envelope's next_hint, in characters.
_HINT_LIMIT = 500

# Which envelope list a successful call's row belongs to.
_ENVELOPE_FIELD: dict[str, str] = {
    "write_draft": "drafts",
    "create_approval": "approvals",
    "log_event": "events",
}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Tool arguments that land in a canon column other than through a draft payload.
_ARGUMENT_COLUMNS: dict[tuple[str, str], tuple[str, str]] = {
    ("create_approval", "amount"): ("apr", "gia_tri"),
}


def _invented_numbers(
    name: str, args: dict[str, Any], blank: dict[str, frozenset[str]]
) -> list[str]:
    """Every canon-blank column this call would fill with a value.

    The illustrative-price layer may write amounts, but only when an operator turns
    it on and only through orchestration. A language model never supplies a number
    the canon leaves blank: zero is a value, not blank.
    """
    found: list[str] = []
    table, payload = args.get("table"), args.get("payload")
    if isinstance(table, str) and isinstance(payload, dict):
        found += [
            f"{table}.{col}" for col, value in payload.items()
            if value is not None and col in blank.get(table, frozenset())
        ]
    for (tool, arg), (target, col) in _ARGUMENT_COLUMNS.items():
        if tool == name and args.get(arg) is not None and col in blank.get(target, frozenset()):
            found.append(f"{target}.{col}")
    return found


class _Record:
    """What the tools actually did during one run."""

    def __init__(self) -> None:
        self.lists: dict[str, list[dict[str, Any]]] = {
            "drafts": [], "approvals": [], "events": [], "citations": [],
        }
        self.calls: list[dict[str, Any]] = []

    def ok(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.calls.append({"name": name, "args": args, "outcome": "ok"})
        field = _ENVELOPE_FIELD.get(name)
        if field is not None and not result.get("replayed"):
            self.lists[field].append(result["row"])
        for passage in result.get("passages", []):
            self.lists["citations"].append(
                {"kb_id": passage["kb_id"], "version": passage["version"]}
            )

    def failed(self, name: str, args: dict[str, Any], outcome: str) -> None:
        self.calls.append({"name": name, "args": args, "outcome": outcome})

    def succeeded(self) -> bool:
        return any(c["outcome"] == "ok" for c in self.calls)

    def succeeded_names(self) -> frozenset[str]:
        return frozenset(c["name"] for c in self.calls if c["outcome"] == "ok")

    def finish(self, task: TaskEnvelope, model: str, steps: int, summary: str) -> ResultEnvelope:
        if not self.succeeded():
            return self.envelope(
                task, model, steps, ResultStatus.ERROR,
                f"{model} finished with no tool call that succeeded: {summary}",
            )
        status = ResultStatus.NEEDS_GATE if self.lists["approvals"] else ResultStatus.DONE
        return self.envelope(task, model, steps, status, summary)

    def envelope(
        self, task: TaskEnvelope, model: str, steps: int, status: ResultStatus, hint: str
    ) -> ResultEnvelope:
        return ResultEnvelope(
            task_id=task.task_id,
            case_id=task.case_id,
            status=status,
            drafts=self.lists["drafts"],
            approvals=self.lists["approvals"],
            events=self.lists["events"],
            citations=self.lists["citations"],
            usage={"model": model, "steps": steps, "tool_calls": self.calls},
            next_hint=hint[:_HINT_LIMIT],
        )


def _arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as an object or, from some models, as JSON text."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if not isinstance(raw, dict):
        raise TypeError(f"tool arguments must be an object, got {type(raw).__name__}")
    return raw


def _reminder(missing: tuple[str, ...]) -> dict[str, Any]:
    """Tell a model that stopped early exactly what its task still requires."""
    return {
        "role": "user",
        "content": (
            "Chưa xong. Nhiệm vụ này bắt buộc phải gọi thành công: "
            f"{', '.join(missing)}. Hãy làm tiếp theo hướng dẫn nghiệp vụ."
        ),
    }


def _plain(reply: dict[str, Any]) -> str:
    return _THINK_BLOCK.sub("", str(reply.get("content") or "")).strip()


def _system_prompt(task: TaskEnvelope, ctx: ToolContext, playbook: str | None) -> str:
    """The standing instructions, in Vietnamese, built from the canon and the token."""
    roles = {row["ma_de"]: row["vai"] for row in ctx.pack.tables.get("de", [])}

    def describe(tables: frozenset[str]) -> str:
        if not tables:
            return "(không có)"
        return "; ".join(
            f"{t}({', '.join(c.name for c in ctx.pack.columns[t])})"
            for t in sorted(tables) if t in ctx.pack.columns
        )

    return (
        f"Bạn là nhân viên số {task.de} — {roles.get(task.de, 'không rõ vai')} — "
        "trong Hệ Doanh nghiệp Tự hành.\n"
        "Quy tắc bắt buộc:\n"
        "1. Chỉ hành động bằng các công cụ được cấp. Việc không làm qua công cụ "
        "thì coi như chưa làm.\n"
        "2. Mọi thứ bạn ghi là bản nháp. Không bao giờ tự duyệt; không đặt trạng thái "
        "'Đã duyệt', 'Đã chi' hay 'Đã gửi'.\n"
        "3. Không bịa số tiền: để null mọi cột tiền.\n"
        "4. Nếu công cụ từ chối, đọc lý do rồi điều chỉnh; không lặp lại đúng lời gọi đó.\n"
        "5. Khi xong, trả lời một câu tóm tắt bằng tiếng Việt và không gọi thêm công cụ.\n"
        f"Vụ việc: {task.case_id}. Nhiệm vụ: {task.intent}. "
        f"Dữ liệu vào: {json.dumps(task.inputs, ensure_ascii=False)}.\n"
        f"Bảng được đọc: {describe(ctx.token.read)}.\n"
        f"Bảng được soạn nháp: {describe(ctx.token.draft)}.\n"
        f"Hướng dẫn nghiệp vụ: {playbook or 'Không có hướng dẫn riêng cho nhiệm vụ này.'}"
    )
