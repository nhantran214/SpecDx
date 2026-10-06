"""Ports the tools depend on, so adapters can be swapped without touching them."""

from __future__ import annotations

from typing import Any, Protocol


class IdempotencyStore(Protocol):
    """Remembers what an idempotency key already produced."""

    def seen(self, key: str) -> dict[str, Any] | None: ...

    def record(self, key: str, result: dict[str, Any]) -> None: ...


class InMemoryIdempotencyStore:
    """Good enough for one process; swap in a shared store to scale out."""

    def __init__(self) -> None:
        self._results: dict[str, dict[str, Any]] = {}

    def seen(self, key: str) -> dict[str, Any] | None:
        return self._results.get(key)

    def record(self, key: str, result: dict[str, Any]) -> None:
        self._results[key] = result


class MailSender(Protocol):
    """Anything that can deliver a message. Implemented in tonomyx/tools/impl."""

    def send(self, to: str, subject: str, body: str) -> None: ...
