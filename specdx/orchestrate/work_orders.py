"""The CASE-0031 work order, as data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from specdx.types import Scenario

WORK_ORDERS = Path(__file__).with_name("work_orders.json")
_WHOLE = re.compile(r"^\{(\w+)\}$")


@dataclass(frozen=True)
class WorkStep:
    id: str
    kind: str  # "de" | "persona" | "env"
    actor: str
    intent: str
    slice: tuple[str, ...]
    inputs: dict[str, str] = field(default_factory=dict)

    @property
    def tables(self) -> frozenset[str]:
        return frozenset(entry.partition("=")[0] for entry in self.slice)


def load_work_orders(path: Path = WORK_ORDERS) -> tuple[WorkStep, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        WorkStep(
            id=s["id"], kind=s["kind"], actor=s["actor"], intent=s["intent"],
            slice=tuple(s["slice"]), inputs=dict(s.get("inputs", {})),
        )
        for s in raw["steps"]
    )


def render_inputs(step: WorkStep, scenario: Scenario) -> dict[str, str]:
    values = {k: str(v) for k, v in scenario.params().items()}
    values["deliveries"] = ",".join(f"{grn}:{n}" for grn, n in scenario.deliveries())
    out: dict[str, str] = {}
    for key, template in step.inputs.items():
        whole = _WHOLE.match(template)
        out[key] = values[whole.group(1)] if whole else template.format(**values)
    return out
