"""One door to Ollama for the LLM-calling baselines (baselines spec §4.2).

Every call is keyed by sha256(schema, model digest, request body), so a rerun is served
from the append-only cache and a changed model can never hit an old answer. A model error
(a read timeout once the host has answered, an HTTP error about the model's own output, an
unusable 200 body) is cached as status "error": counted, never a finding, never retried.
An outage is never cached (spec §9.8): the executor's classification (`_ollama_failed`:
unreachable, crashed or out-of-memory runner) raises InfrastructureError, and so does a read
timeout before this gateway has had any ok reply from the host (a wedged Ollama).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from specdx.cost import Cost
from specdx.interpreters.executor import NUM_CTX, _describe, _ollama_failed
from specdx.types import InfrastructureError

SCHEMA = 1


class DigestMismatch(RuntimeError):
    """The model is not pulled, or its digest differs from the population runs'."""


@dataclass(frozen=True)
class Reply:
    text: str
    cost: Cost
    status: str
    error: str = ""
    cached: bool = False


class Gateway:
    def __init__(self, base_url: str, cache_path: Path,
                 pinned: Mapping[str, str] | None = None, *,
                 transport: httpx.BaseTransport | None = None, timeout: float = 600.0) -> None:
        self.base_url, self.cache_path = base_url.rstrip("/"), cache_path
        self.pinned = dict(pinned or {})
        self.transport, self.timeout = transport, timeout
        self.total = Cost()
        self._digests: dict[str, str] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        self._answered = False  # a live ok reply came from the host (cache hits do not count)
        if cache_path.exists():
            self._load(cache_path)

    def _load(self, path: Path) -> None:
        # Split on b"\n" only: str.splitlines would also split on U+2028, U+2029 and U+0085.
        data, start, lines = path.read_bytes(), 0, []
        for raw in data.split(b"\n"):
            if raw.strip():
                lines.append((start, raw))
            start += len(raw) + 1
        for i, (offset, raw) in enumerate(lines):
            try:
                rec = json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                if i == len(lines) - 1:  # a hard kill mid-append leaves a partial last line:
                    os.truncate(path, offset)  # drop only it, so the next append starts clean
                    return
                raise ValueError(f"{path}: unparseable record on line {i + 1}") from exc
            self._cache[rec["key"]] = rec
        if data and not data.endswith(b"\n"):  # killed between the record and its newline
            with path.open("ab") as fh:
                fh.write(b"\n")

    def _client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport, timeout=self.timeout)

    def digest(self, model: str) -> str:
        if model not in self._digests:
            try:
                with self._client() as c:
                    resp = c.get(f"{self.base_url}/api/tags")
                    resp.raise_for_status()
                    tags = resp.json()
            except httpx.HTTPError as exc:
                raise InfrastructureError(f"cannot list models at {self.base_url}: {exc}") from exc
            found = {m["name"]: m["digest"] for m in tags.get("models", [])}
            if model not in found:
                raise DigestMismatch(f"{model} is not pulled at {self.base_url}")
            if model in self.pinned and self.pinned[model] != found[model]:
                raise DigestMismatch(f"{model} has digest {found[model]}, the population runs "
                                     f"used {self.pinned[model]}")
            self._digests[model] = found[model]
        return self._digests[model]

    def chat(self, model: str, messages: Sequence[Mapping[str, str]], *,
             options: Mapping[str, Any] | None = None, max_tokens: int | None = None) -> Reply:
        digest = self.digest(model)
        opts: dict[str, Any] = {"num_ctx": NUM_CTX, **(options or {})}
        if max_tokens:
            opts["num_predict"] = max_tokens
        body: dict[str, Any] = {"model": model, "messages": [dict(m) for m in messages],
                                "stream": False, "options": opts}
        if model.startswith("qwen3"):  # as the interpreters call it (executor._chat)
            body["think"] = False
        key = hashlib.sha256(json.dumps({"schema": SCHEMA, "digest": digest, "body": body},
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self._cache:
            rec = self._cache[key]
            cost = Cost(**rec["cost"])
            self.total = self.total + cost
            return Reply(rec["text"], cost, rec["status"], rec.get("error", ""), cached=True)
        try:
            with self._client() as c:
                resp = c.post(f"{self.base_url}/api/chat", json=body)
                resp.raise_for_status()
                data = resp.json()
            cost = Cost(1, int(data.get("prompt_eval_count") or 0),
                        int(data.get("eval_count") or 0),
                        int(data.get("total_duration") or 0) // 1_000_000)
            rec = {"text": str(data["message"].get("content") or ""), "status": "ok", "error": ""}
        except httpx.ReadTimeout as exc:
            if not self._answered:  # no reply yet from this host: wedged, not the model
                raise InfrastructureError(
                    f"Ollama at {self.base_url} timed out before any reply ({exc}): "
                    "it may be wedged") from exc
            cost = Cost(requests=1, errors=1)
            rec = {"text": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        except httpx.HTTPError as exc:
            if _ollama_failed(exc):  # unreachable, crashed or OOM runner: not data
                raise InfrastructureError(
                    f"Ollama at {self.base_url} failed: {_describe(exc)}") from exc
            cost = Cost(requests=1, errors=1)
            rec = {"text": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        except (ValueError, KeyError, AttributeError, TypeError) as exc:  # 200 but unusable body
            cost = Cost(requests=1, errors=1)
            rec = {"text": "", "status": "error", "error": f"malformed reply: {exc!r}"}
        else:
            self._answered = True
        rec.update({"key": key, "model": model, "digest": digest, "cost": cost.as_dict()})
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self._cache[key] = rec
        self.total = self.total + cost
        return Reply(rec["text"], cost, rec["status"], rec["error"])
