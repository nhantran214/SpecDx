"""An interpreter: a local model that reads T through search_spec and acts through gated tools.

It follows OllamaExecutor's contract: the envelope comes from what the tools did, and every
call runs in its own savepoint. It differs in what the model is told: no playbook and no
business rule in the prompt, only the Tell corpus behind search_spec.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope
from tonomyx.de.executor import _arguments, _invented_numbers, _plain, _Record
from tonomyx.tools.registry import ToolContext, ToolDenied, ToolNotFound, ToolRegistry
from tonomyx.tools.schema import tool_schemas

from specdx.config import OLLAMA_URL
from specdx.interpreters.prompt import TellVariants, system_prompt
from specdx.spec.tell import TellCorpus

SEARCH_SPEC = "search_spec"
SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_SPEC,
        "description": (
            "Tra cứu tài liệu nghiệp vụ (tài liệu BA, mô tả demo, SAD, KB, nguyên tắc canon). "
            "Trả về các đoạn liên quan nhất."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
_REPLY_LIMIT = 4000


class SpecInterpreterExecutor:
    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        corpus: TellCorpus,
        variants: TellVariants,
        *,
        variant: int = 0,
        seed: int = 1,
        temperature: float = 0.7,
        base_url: str = OLLAMA_URL,
        max_steps: int = 16,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
        suffix: str = "",
    ) -> None:
        self.model, self.tools, self.corpus, self.variants = model, tools, corpus, variants
        self.variant, self.seed, self.temperature = variant, seed, temperature
        self.base_url, self.max_steps, self.timeout = base_url, max_steps, timeout
        self.transport, self.suffix = transport, suffix

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        schemas = [*tool_schemas(self.tools, ctx.pack, names=ctx.token.tools), SEARCH_SCHEMA]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(
                self.variants, self.variant, pack=ctx.pack, task=task,
                read=ctx.token.read, draft=ctx.token.draft, suffix=self.suffix,
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
                        return record.finish(task, self.model, step, _plain(reply))
                    for call in calls:
                        messages.append(self._execute(call, ctx, record))
        except httpx.HTTPError as exc:
            return ResultEnvelope(
                task_id=task.task_id, case_id=task.case_id, status=ResultStatus.ERROR,
                next_hint=f"ollama call failed: {exc}",
            )
        return record.envelope(
            task, self.model, self.max_steps, ResultStatus.ERROR,
            f"{self.model} did not finish within {self.max_steps} steps",
        )

    def _chat(
        self, client: httpx.Client, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model, "messages": messages, "tools": schemas, "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        if self.model.startswith("qwen3"):
            body["think"] = False
        response = client.post(f"{self.base_url}/api/chat", json=body)
        response.raise_for_status()
        message: dict[str, Any] = response.json()["message"]
        return message

    def _execute(self, call: dict[str, Any], ctx: ToolContext, record: _Record) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        args: dict[str, Any] = {}
        payload: Any
        try:
            args = _arguments(function.get("arguments"))
            if name == SEARCH_SPEC:
                hits = self.corpus.search(str(args.get("query", "")), k=5)
                payload = {"passages": [{"id": c.id, "text": c.text} for c in hits]}
                record.calls.append({"name": name, "args": args, "outcome": "ok"})
            else:
                invented = _invented_numbers(name, args, ctx.pack.non_canon_columns)
                if invented:
                    raise ToolDenied(
                        f"the canon leaves {', '.join(invented)} blank: pass null instead"
                    )
                with ctx.repo.conn.transaction():
                    result = self.tools.invoke(name, ctx, **args)
                record.ok(name, args, result)
                payload = result
        except ToolDenied as exc:
            record.failed(name, args, "denied")
            payload = {"error": "denied", "tool": name, "reason": str(exc)}
        except ToolNotFound:
            record.failed(name, args, "unknown")
            payload = {"error": "unknown tool", "tool": name}
        except Exception as exc:  # noqa: BLE001 - the model is told what broke
            record.failed(name, args, "error")
            payload = {"error": type(exc).__name__, "tool": name, "reason": str(exc)}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        return {"role": "tool", "tool_name": name, "content": content[:_REPLY_LIMIT]}
