"""Time as a port, so tests never depend on the wall clock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """Anything that can say what time it is, in UTC."""

    def now(self) -> datetime: ...


class SystemClock:
    """The real clock. Used everywhere outside tests."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """A clock that only moves when a test moves it."""

    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, seconds: float) -> None:
        self._at = self._at + timedelta(seconds=seconds)
