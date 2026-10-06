"""The capability pilot (spec §4.2) and the go/no-go gate (spec §4.8)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from specdx.active.loop import LoopContext, SpecDxConfig, run_specdx
from specdx.baselines.b2_single import b2_report
from specdx.baselines.b3_irt import b3_report
from specdx.config import OLLAMA_URL
from specdx.eval.matching import Prf, evaluate_reports, gate_verdict
from specdx.mutate.operators import GroundTruth, Mutant
from specdx.types import DEFECT_TYPES, Defect
from specdx.variant import SpecVariant

ECHO = {"type": "function", "function": {
    "name": "echo", "description": "Trả lại văn bản.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                   "required": ["text"]}}}


@dataclass(frozen=True)
class Capability:
    model: str
    declares_tools: bool
    calls_tool: bool

    @property
    def usable(self) -> bool:
        return self.declares_tools and self.calls_tool


def check_model(
    model: str, *, base_url: str = OLLAMA_URL, transport: httpx.BaseTransport | None = None
) -> Capability:
    with httpx.Client(transport=transport, timeout=300.0) as client:
        show = client.post(f"{base_url}/api/show", json={"model": model}).json()
        if "tools" not in show.get("capabilities", []):
            return Capability(model, False, False)
        body: dict[str, Any] = {
            "model": model, "stream": False, "tools": [ECHO], "options": {"temperature": 0},
            "messages": [{"role": "user", "content": "Gọi công cụ echo với text là 'ok'."}],
        }
        if model.startswith("qwen3"):
            body["think"] = False
        message = client.post(f"{base_url}/api/chat", json=body).json().get("message", {})
    calls = message.get("tool_calls") or []
    return Capability(model, True, any((c.get("function") or {}).get("name") == "echo"
                                       for c in calls))


@dataclass(frozen=True)
class GateConfig:
    models_by_family: dict[str, tuple[str, ...]]
    variants: tuple[int, ...]
    seeds: tuple[int, ...]
    budget: int
    dev_per_type: int
    rng_seed: int

    @classmethod
    def load(cls, path: Path) -> GateConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            models_by_family={f: tuple(ms) for f, ms in raw["models_by_family"].items()},
            variants=tuple(raw["variants"]), seeds=tuple(raw["seeds"]), budget=int(raw["budget"]),
            dev_per_type=int(raw["dev_per_type"]), rng_seed=int(raw["rng_seed"]),
        )

    def specdx(self) -> SpecDxConfig:
        return SpecDxConfig(self.models_by_family, variants=self.variants, seeds=self.seeds,
                            budget=self.budget, rng_seed=self.rng_seed)


class BenchLike(Protocol):
    base: SpecVariant

    def loop_context(self, variant: SpecVariant, config: SpecDxConfig) -> LoopContext: ...


@dataclass
class GateResult:
    scores: dict[str, dict[str, Prf]]
    untyped: dict[str, dict[str, Prf]]
    passed: bool
    wins: dict[str, bool]
    failed: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        def table(s: dict[str, dict[str, Prf]]) -> dict[str, Any]:
            return {m: {t: p.as_dict() for t, p in by.items()} for m, by in s.items()}
        return json.dumps({"passed": self.passed, "wins": self.wins, "failed": self.failed,
                           "typed": table(self.scores), "untyped": table(self.untyped)},
                          ensure_ascii=False, indent=2)


def _baselines(report_behaviours, ctx: LoopContext, theta, canon_ids):
    violated = {b.run_key: ctx.lens.violations(b) for b in report_behaviours}
    return (b2_report(report_behaviours, violated, theta, canon_ids),
            b3_report(report_behaviours, violated))


def run_gate(config: GateConfig, bench: BenchLike, mutants: Sequence[Mutant]) -> GateResult:
    cfg = config.specdx()
    base_ctx = bench.loop_context(bench.base, cfg)
    base = run_specdx(bench.base, base_ctx, cfg)
    canon_ids = set(base_ctx.canon)
    base_b2, base_b3 = _baselines(base.behaviours, base_ctx, base.theta, canon_ids)
    reports: dict[str, dict[str, list[Defect]]] = {"specdx": {}, "b2": {}, "b3": {}}
    truths: dict[str, GroundTruth] = {}
    failed: dict[str, str] = {}
    for m in mutants:
        try:
            ctx = bench.loop_context(m.variant, cfg)
            report = run_specdx(m.variant, ctx, cfg)
        except Exception as exc:  # noqa: BLE001 - one broken mutant must not stop the gate
            failed[m.id] = f"{type(exc).__name__}: {exc}"
            continue
        if report.behaviours and all(b.status == "error" for b in report.behaviours):
            failed[m.id] = report.behaviours[0].error
            continue
        truths[m.id] = m.truth
        reports["specdx"][m.id] = report.defects
        reports["b2"][m.id], reports["b3"][m.id] = _baselines(
            report.behaviours, ctx, report.theta, canon_ids
        )
    bases = {"specdx": base.defects, "b2": base_b2, "b3": base_b3}
    scores = {n: evaluate_reports(reports[n], truths, bases[n]) for n in reports}
    untyped = {n: evaluate_reports(reports[n], truths, bases[n], typed=False) for n in reports}
    passed, wins = gate_verdict(scores)
    return GateResult(scores, untyped, passed, wins, failed)


def gate_markdown(result: GateResult) -> str:
    lines = [
        f"# P1 gate: {'PASSED' if result.passed else 'NOT PASSED'}",
        "",
        "Typed F1 (verdict) and untyped F1 (location only) per defect type.",
        "",
        "| type | SpecDx | B2 | B3 | SpecDx untyped | B2 untyped | B3 untyped | SpecDx wins |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in DEFECT_TYPES:
        lines.append(
            f"| {t} | {result.scores['specdx'][t].f1:.2f} | {result.scores['b2'][t].f1:.2f} | "
            f"{result.scores['b3'][t].f1:.2f} | {result.untyped['specdx'][t].f1:.2f} | "
            f"{result.untyped['b2'][t].f1:.2f} | {result.untyped['b3'][t].f1:.2f} | "
            f"{'yes' if result.wins[t] else 'no'} |"
        )
    if result.failed:
        lines += ["", "## Mutants that could not run", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(result.failed.items())]
    return "\n".join(lines) + "\n"
