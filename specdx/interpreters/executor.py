"""An interpreter: a local model that reads T through search_spec and acts through gated tools.

It follows OllamaExecutor's contract: the envelope comes from what the tools did, and every
call runs in its own savepoint. It differs in what the model is told: no playbook and no
business rule in the prompt, only the Tell corpus behind search_spec.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope
from tonomyx.de.executor import OllamaExecutor, _arguments, _plain, _Record
from tonomyx.tools.registry import ToolContext, ToolRegistry
from tonomyx.tools.schema import tool_schemas

from specdx.config import OLLAMA_URL
from specdx.interpreters.prompt import TellVariants, system_prompt
from specdx.spec.tell import TellCorpus

SEARCH_SPEC = "search_spec"
TEMPERATURE = 0.7
MAX_STEPS = 16
TIMEOUT = 180.0
SEARCH_K = 5
# whole passages only, never a sliced JSON
SEARCH_REPLY_LIMIT = 12000
# Ollama's default context would silently drop history once search replies are this long
NUM_CTX = 16384
_REPLY_LIMIT = 4000
# Pauses between the five attempts of one chat call when Ollama itself fails.
RETRY_PAUSES: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
log = logging.getLogger(__name__)
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


def interpreter_settings() -> dict[str, float | int]:
    """Every executor setting an interpreter's behaviour depends on, as now configured.
    The run key covers them (spec §6.5), so changing one never reuses a cached run."""
    return {"temperature": TEMPERATURE, "max_steps": MAX_STEPS, "timeout": TIMEOUT,
            "num_ctx": NUM_CTX, "search_k": SEARCH_K, "search_reply_limit": SEARCH_REPLY_LIMIT}


def _ollama_failed(exc: httpx.HTTPError) -> bool:
    """Ollama itself failed (unreachable, crashed, out of memory). A read timeout is not
    such a failure: it is the model being slow, which is interpreter behaviour."""
    if isinstance(exc, httpx.ReadTimeout):
        return False
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class SpecInterpreterExecutor:
    """infra_failure is set when Ollama kept failing. DeRuntime swallows the error into a
    step result, so the Runner reads this flag to keep the run out of the cache."""

    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        corpus: TellCorpus,
        variants: TellVariants,
        *,
        variant: int = 0,
        seed: int = 1,
        temperature: float = TEMPERATURE,
        base_url: str = OLLAMA_URL,
        max_steps: int = MAX_STEPS,
        timeout: float = TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        suffix: str = "",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model, self.tools, self.corpus, self.variants = model, tools, corpus, variants
        self.variant, self.seed, self.temperature = variant, seed, temperature
        self.base_url, self.max_steps, self.timeout = base_url, max_steps, timeout
        self.transport, self.suffix, self.sleep = transport, suffix, sleep
        self.infra_failure: str | None = None
        self._gated = OllamaExecutor(model, tools)

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        if self.infra_failure is not None:
            # The run is void already; do not wait out the retries again on every step.
            return ResultEnvelope(
                task_id=task.task_id, case_id=task.case_id, status=ResultStatus.ERROR,
                next_hint=f"not attempted: {self.infra_failure}",
            )
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
            "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": NUM_CTX},
        }
        if self.model.startswith("qwen3"):
            body["think"] = False
        attempts = len(RETRY_PAUSES) + 1
        for attempt in range(1, attempts + 1):
            try:
                response = client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
                message: dict[str, Any] = response.json()["message"]
                return message
            except httpx.HTTPError as exc:
                if not _ollama_failed(exc):
                    raise
                reason = f"{type(exc).__name__}: {exc}"
                if attempt == attempts:
                    self.infra_failure = (
                        f"Ollama failed {attempts} times for {self.model}; last: {reason}"
                    )
                    log.error("%s", self.infra_failure)
                    raise
                pause = RETRY_PAUSES[attempt - 1]
                log.warning("Ollama call for %s failed (%s); retry %d of %d in %.0f s",
                            self.model, reason, attempt, attempts - 1, pause)
                self.sleep(pause)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _execute(self, call: dict[str, Any], ctx: ToolContext, record: _Record) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        # Route gated tools directly to vendored executor before parsing arguments
        if name != SEARCH_SPEC:
            return self._gated._execute(call, ctx, record)
        # For search_spec, parse arguments and handle within this method
        args: dict[str, Any] = {}
        try:
            args = _arguments(function.get("arguments"))
            hits = self.corpus.search(str(args.get("query", "")), k=SEARCH_K)
            # Build payload passage by passage, adding only if it fits within SEARCH_REPLY_LIMIT
            passages: list[dict[str, Any]] = []
            for i, hit in enumerate(hits):
                passage = {"id": hit.id, "text": hit.text}
                # Always include the top passage, even if it exceeds limit
                if i == 0:
                    payload_test = {"passages": [passage]}
                    serialized = json.dumps(
                        payload_test, ensure_ascii=False, default=str
                    )
                    if len(serialized) > SEARCH_REPLY_LIMIT:
                        # Trim the text to fit with "…"; include "…" in the length check
                        while len(passage["text"]) > 0:
                            passage["text"] = passage["text"][:-1]
                            passage["text"] += "…"
                            payload_test = {"passages": [passage]}
                            serialized = json.dumps(
                                payload_test, ensure_ascii=False, default=str
                            )
                            if len(serialized) <= SEARCH_REPLY_LIMIT:
                                break
                            # Remove "…" and try again
                            passage["text"] = passage["text"][:-1]
                    passages.append(passage)
                else:
                    # Try adding this passage
                    payload_test = {"passages": passages + [passage]}
                    serialized = json.dumps(payload_test, ensure_ascii=False, default=str)
                    if len(serialized) <= SEARCH_REPLY_LIMIT:
                        passages.append(passage)
                    else:
                        # Stop adding more passages
                        break
            payload = {"passages": passages}
            record.calls.append({"name": name, "args": args, "outcome": "ok"})
            content = json.dumps(payload, ensure_ascii=False, default=str)
            return {"role": "tool", "tool_name": name, "content": content}
        except Exception as exc:  # noqa: BLE001 - the model is told what broke
            record.failed(name, args, "error")
            payload = {"error": type(exc).__name__, "tool": name, "reason": str(exc)}
            content = json.dumps(payload, ensure_ascii=False, default=str)
            return {"role": "tool", "tool_name": name, "content": content[:_REPLY_LIMIT]}
