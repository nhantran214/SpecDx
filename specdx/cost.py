"""What a method spent, in model requests (baselines spec §9.1).

A request is one chat call to Ollama. Runs made since the executor counts requests carry
exact figures; older cached runs keep only tool_calls and nudges per step, and a step can
have made at most tool_calls + nudges + 1 requests, so their cost is an upper bound. The
bound overstates SpecDx's cost, which is the conservative direction for the comparison.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from specdx.types import Behaviour

COUNTERS: tuple[str, ...] = ("requests", "prompt_tokens", "eval_tokens", "duration_ms")


@dataclass(frozen=True)
class Cost:
    requests: int = 0
    prompt_tokens: int = 0
    eval_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0
    exact: bool = True

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.requests + other.requests, self.prompt_tokens + other.prompt_tokens,
                    self.eval_tokens + other.eval_tokens, self.duration_ms + other.duration_ms,
                    self.errors + other.errors, self.exact and other.exact)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run_cost(b: Behaviour) -> Cost:
    errors = int(b.status == "error")
    steps = list(b.harness.values())
    if steps and all("requests" in s for s in steps):
        totals = {k: sum(int(s.get(k, 0)) for s in steps) for k in COUNTERS}
        return Cost(**totals, errors=errors, exact=True)
    bound = sum(int(s.get("tool_calls", 0)) + int(s.get("nudges", 0)) + 1 for s in steps)
    return Cost(requests=bound, errors=errors, exact=False)


def behaviours_cost(behaviours: Sequence[Behaviour], extra_runs: int = 0) -> Cost:
    """The runs' cost, plus extra_runs (probe and ddmin runs a report only counts) charged
    at the mean requests per run, rounded up."""
    total = sum((run_cost(b) for b in behaviours), Cost())
    if extra_runs and behaviours:
        per = math.ceil(total.requests / len(behaviours))
        total = total + Cost(requests=per * extra_runs, exact=False)
    return total
