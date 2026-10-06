"""Value types shared by every SpecDx unit."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

# Scenario parameters (spec §2.1). grn_count changes which rows exist, so it is structural.
PARAMETERS: tuple[str, ...] = ("ordered", "received", "discount_pct", "grn_count")
STRUCTURAL: frozenset[str] = frozenset({"grn_count"})
DEFECT_TYPES: tuple[str, ...] = ("contradiction", "hidden", "ambiguity", "weak")

FIRST_GRN = "GRN-0015"
SECOND_GRN = "GRN-0016"


@dataclass(frozen=True)
class Scenario:
    """One point of the scenario space."""

    id: str
    ordered: int
    received: int
    discount_pct: int
    grn_count: int = 1

    def params(self) -> dict[str, int]:
        return {
            "ordered": self.ordered,
            "received": self.received,
            "discount_pct": self.discount_pct,
            "grn_count": self.grn_count,
        }

    def signature(self) -> tuple[int, ...]:
        """The parameters alone: two scenarios with equal signatures are the same scenario."""
        return tuple(self.params()[p] for p in PARAMETERS)

    def deliveries(self) -> list[tuple[str, int]]:
        """How the received units arrive: one receipt, or two when grn_count is 2."""
        if self.grn_count == 1:
            return [(FIRST_GRN, self.received)]
        if self.grn_count == 2:
            first = self.received - self.received // 2
            return [(FIRST_GRN, first), (SECOND_GRN, self.received // 2)]
        raise ValueError(f"grn_count must be 1 or 2, got {self.grn_count}")


@dataclass(frozen=True)
class State:
    """A database end state as text, plus values only the database can compute."""

    tables: dict[str, list[dict[str, str]]]
    derived: dict[str, str] = field(default_factory=dict)
    steps: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> State:
        return cls(tables={})


@dataclass(frozen=True)
class InterpreterId:
    """One member of the population: a model read through one Tell variant with one seed."""

    model: str
    family: str
    variant: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}|v{self.variant}|s{self.seed}"


@dataclass(frozen=True)
class Behaviour:
    """What one interpreter did on one scenario."""

    run_key: str
    interpreter: InterpreterId
    scenario: Scenario
    state: State
    refused: tuple[tuple[str, str], ...] = ()
    claims: dict[str, str] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Behaviour:
        raw = json.loads(text)
        return cls(
            run_key=raw["run_key"],
            interpreter=InterpreterId(**raw["interpreter"]),
            scenario=Scenario(**raw["scenario"]),
            state=State(**raw["state"]),
            refused=tuple((str(a), str(b)) for a, b in raw["refused"]),
            claims=dict(raw["claims"]),
            status=raw["status"],
            error=raw["error"],
        )


@dataclass(frozen=True)
class Defect:
    """One typed finding about the specification."""

    type: str
    location: tuple[str, ...]
    scenarios: tuple[str, ...]
    witnesses: tuple[str, ...]
    score: float
    note: str = ""

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return (self.type, self.location)


@dataclass(frozen=True)
class RefuteResult:
    """Whether a state satisfying the targets was found (spec §3.6)."""

    refuted: bool
    witness: str | None
    attempts: int


class Refuter(Protocol):
    """Anything that can try to satisfy a set of clauses on one scenario."""

    def refute(
        self, targets: frozenset[str], scenario: Scenario, behaviours: Sequence[Behaviour]
    ) -> RefuteResult: ...
