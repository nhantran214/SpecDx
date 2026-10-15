"""The capability pilot (spec §4.2) and the go/no-go gate (spec §4.8; v2 §4.1–§4.2).

Gate v2 runs on the population the capability screen froze. Before any mutant runs, it
computes from the base runs which mutants the population can score (eval.observability).
The primary table scores those only and decides the verdict; a type with fewer than three
scorable mutants is a loss. The secondary table scores all mutants, as P1 did.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import psycopg

from specdx.active.loop import LoopContext, Report, SpecDxConfig, run_specdx
from specdx.baselines.b2_single import b2_report
from specdx.baselines.b3_irt import b3_report
from specdx.config import OLLAMA_URL
from specdx.eval.matching import (
    ABSOLUTE_BAR,
    MIN_SCORABLE,
    Prf,
    evaluate_reports,
    gate_rule,
    gate_verdict,
    literal_wins,
    match_findings,
)
from specdx.eval.observability import scorable as scorable_ids
from specdx.eval.observability import touch_counts
from specdx.infer.anchored import V3Params
from specdx.mutate.operators import GroundTruth, Mutant
from specdx.types import DEFECT_TYPES, Defect, InfrastructureError
from specdx.variant import SpecVariant

# Faults of the machinery, not of a mutant: the gate stops rather than score around them.
INFRASTRUCTURE_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.OperationalError, psycopg.InterfaceError, httpx.TransportError, InfrastructureError,
)

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
    """An HTTP error (a model Ollama does not have, a crashed runner) is raised: only a
    model that answers can be judged unusable."""
    with httpx.Client(transport=transport, timeout=300.0) as client:
        response = client.post(f"{base_url}/api/show", json={"model": model})
        response.raise_for_status()
        show = response.json()
        if "tools" not in show.get("capabilities", []):
            return Capability(model, False, False)
        body: dict[str, Any] = {
            "model": model, "stream": False, "tools": [ECHO], "options": {"temperature": 0},
            "messages": [{"role": "user", "content": "Gọi công cụ echo với text là 'ok'."}],
        }
        if model.startswith("qwen3"):
            body["think"] = False
        reply = client.post(f"{base_url}/api/chat", json=body)
        reply.raise_for_status()
        message = reply.json().get("message", {})
    calls = message.get("tool_calls") or []
    return Capability(model, True, any((c.get("function") or {}).get("name") == "echo"
                                       for c in calls))


def load_population(path: Path) -> dict[str, tuple[str, ...]]:
    """The population a capability screen froze (its "population"), refusing a screen that
    ended in the stop rule."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("stop"):
        raise ValueError(f"{path}: the screen admitted fewer than 2 families or 4 models; "
                         "the work stops")
    return {f: tuple(ms) for f, ms in raw["population"].items()}


@dataclass(frozen=True)
class GateConfig:
    """screen: the capability screen's JSON; when set, the population must be the one it
    froze (spec v2 §2.2)."""

    models_by_family: dict[str, tuple[str, ...]]
    variants: tuple[int, ...]
    seeds: tuple[int, ...]
    budget: int
    dev_per_type: int
    rng_seed: int
    probe_budget: int = SpecDxConfig.probe_budget
    localize_budget: int = SpecDxConfig.localize_budget
    screen: Path | None = None
    # Never read from the JSON: the gate runs on the frozen V3Params (spec v3 §9); tests pass
    # their own.
    params: V3Params = V3Params()

    @classmethod
    def load(cls, path: Path) -> GateConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            models_by_family={f: tuple(ms) for f, ms in raw["models_by_family"].items()},
            variants=tuple(raw["variants"]), seeds=tuple(raw["seeds"]), budget=int(raw["budget"]),
            dev_per_type=int(raw["dev_per_type"]), rng_seed=int(raw["rng_seed"]),
            probe_budget=int(raw.get("probe_budget", SpecDxConfig.probe_budget)),
            localize_budget=int(raw.get("localize_budget", SpecDxConfig.localize_budget)),
            screen=Path(raw["screen"]) if raw.get("screen") else None,
        )

    def check_population(self) -> None:
        """The gate runs only on a non-empty population, the screen's when one is named."""
        if not self.models_by_family or not any(self.models_by_family.values()):
            raise ValueError("models_by_family is empty: run `specdx screen` and freeze its "
                             "population into the gate configuration first")
        if self.screen is not None and load_population(self.screen) != self.models_by_family:
            raise ValueError(f"models_by_family differs from the population {self.screen} "
                             "froze; the gate runs on the screened population only")

    def specdx(self) -> SpecDxConfig:
        return SpecDxConfig(self.models_by_family, variants=self.variants, seeds=self.seeds,
                            budget=self.budget, rng_seed=self.rng_seed,
                            probe_budget=self.probe_budget, localize_budget=self.localize_budget,
                            params=self.params)


class BenchLike(Protocol):
    base: SpecVariant

    def loop_context(self, variant: SpecVariant, config: SpecDxConfig) -> LoopContext: ...


@dataclass
class GateResult:
    """scores and untyped: the secondary tables, over every mutant that ran. primary and
    primary_untyped: the same over the scorable mutants only, which decide the verdict.
    touches: per mutant, the base behaviours that touch its observed_at targets."""

    scores: dict[str, dict[str, Prf]]
    untyped: dict[str, dict[str, Prf]]
    passed: bool
    wins: dict[str, bool]
    failed: dict[str, str] = field(default_factory=dict)
    literal_wins: dict[str, bool] = field(default_factory=dict)
    primary: dict[str, dict[str, Prf]] = field(default_factory=dict)
    primary_untyped: dict[str, dict[str, Prf]] = field(default_factory=dict)
    touches: dict[str, int] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)

    @property
    def scorable(self) -> frozenset[str]:
        return scorable_ids(self.touches)

    def counts(self, only_scorable: bool) -> dict[str, int]:
        """Mutants per type, all of them or the scorable ones."""
        out = {t: 0 for t in DEFECT_TYPES}
        for mid, t in self.types.items():
            if not only_scorable or mid in self.scorable:
                out[t] += 1
        return out

    @property
    def headline(self) -> str:
        """No verdict is given while any mutant failed to run."""
        if self.failed:
            return "INCOMPLETE"
        return "PASSED" if self.passed else "NOT PASSED"

    def to_json(self) -> str:
        def table(s: dict[str, dict[str, Prf]]) -> dict[str, Any]:
            return {m: {t: p.as_dict() for t, p in by.items()} for m, by in s.items()}
        return json.dumps({"headline": self.headline, "passed": self.passed, "wins": self.wins,
                           "literal_wins": self.literal_wins, "failed": self.failed,
                           "primary": {"typed": table(self.primary),
                                       "untyped": table(self.primary_untyped)},
                           "typed": table(self.scores), "untyped": table(self.untyped),
                           "touches": self.touches, "scorable": sorted(self.scorable),
                           "scorable_per_type": self.counts(True),
                           "mutants_per_type": self.counts(False)},
                          ensure_ascii=False, indent=2)


def _baselines(report_behaviours, ctx: LoopContext, theta, canon_ids):
    violated = {b.run_key: ctx.lens.violations(b) for b in report_behaviours}
    return (b2_report(report_behaviours, violated, theta, canon_ids),
            b3_report(report_behaviours, violated))


def _defects(defects: Sequence[Defect]) -> list[dict[str, Any]]:
    return [asdict(d) for d in defects]


def _runs(report: Report) -> dict[str, int]:
    return {"population": report.runs, "probe": report.probe_runs,
            "ddmin": report.ddmin_runs, "executed": report.executed}


def _keys(keys: frozenset[tuple[str, str, tuple[str, ...]]]) -> list[list[Any]]:
    """Ambiguity identities (type, step, fields), sorted, as JSON."""
    return [[t, step, list(fields)] for t, step, fields in sorted(keys)]


def delta_shift(base: Report, report: Report) -> dict[str, float]:
    """Each item's difficulty δ on the mutant minus on the base (spec v2 §3.9): a shift
    shows where the mutant changed who acts, which abstention assumes it does not."""
    return {item: row["delta"] - base.items[item]["delta"]
            for item, row in sorted(report.items.items()) if item in base.items}


def _mutant_line(
    m: Mutant,
    found: dict[str, list[Defect]] | None,
    bases: dict[str, list[Defect]],
    report: Report | None,
    failure: str | None,
    touched: int = 0,
    is_scorable: bool = False,
    base: Report | None = None,
) -> dict[str, Any]:
    matches = None
    if found is not None:
        matches = {
            method: {
                ("typed" if typed else "untyped"): [
                    [d.type, list(d.location)]
                    for d in match_findings(found[method], m.truth, bases[method],
                                            typed=typed)[1]
                ]
                for typed in (True, False)
            }
            for method in found
        }
    return {
        "base": False, "mutant": m.id, "operator": m.operator, "site": m.site,
        "type": m.truth.type,
        "truth": {"location": sorted(m.truth.location), "kind": m.truth.location_kind},
        "defects": None if found is None else {k: _defects(v) for k, v in found.items()},
        "matches": matches,
        "runs": None if report is None else _runs(report),
        # before the differential, and the base noise it removed (spec §11 item 9)
        "ambiguity_keys": None if report is None else _keys(report.ambiguity_keys),
        "ambiguity_dropped": None if report is None else _keys(report.ambiguity_dropped),
        "touches": touched, "scorable": is_scorable,
        "unobservable": None if report is None else _defects(report.unobservable),
        "theta": None if report is None else report.theta,
        "alpha": None if report is None else report.alpha,
        "kappa": None if report is None else report.kappa,
        "family_effect": None if report is None else report.family_effect,
        "converged": None if report is None else report.converged,
        "iterations": None if report is None else report.iterations,
        "items": None if report is None else report.items,
        "cells": None if report is None else report.cells,
        "literals": None if report is None else [x.as_dict() for x in report.literals],
        "delta_shift": None if report is None or base is None else delta_shift(base, report),
        "failure": failure,
    }


def run_gate(
    config: GateConfig,
    bench: BenchLike,
    mutants: Sequence[Mutant],
    *,
    mutants_out: Path | None = None,
) -> GateResult:
    """mutants_out, if given, receives one JSON line for the base and one per mutant,
    written as each finishes."""
    config.check_population()
    cfg = config.specdx()
    base_ctx = bench.loop_context(bench.base, cfg)
    # The base report is the differential's reference: its ambiguities are compared on
    # step and fields, before localisation, so localising them would be wasted runs.
    base = run_specdx(bench.base, base_ctx, dataclasses.replace(cfg, localize=False))
    # Scorability comes from the base runs alone, before any mutant runs (spec v2 §4.1).
    touched = touch_counts({m.id: m.truth.observed_at for m in mutants}, base.behaviours,
                           base_ctx.lens)
    keep = scorable_ids(touched)
    canon_ids = set(base_ctx.canon)
    base_b2, base_b3 = _baselines(base.behaviours, base_ctx, base.theta, canon_ids)
    bases = {"specdx": base.defects, "b2": base_b2, "b3": base_b3}
    reports: dict[str, dict[str, list[Defect]]] = {"specdx": {}, "b2": {}, "b3": {}}
    truths: dict[str, GroundTruth] = {}
    failed: dict[str, str] = {}
    lines = None
    if mutants_out is not None:
        mutants_out.parent.mkdir(parents=True, exist_ok=True)
        lines = mutants_out.open("w", encoding="utf-8")

    def write(line: dict[str, Any]) -> None:
        if lines is not None:
            lines.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
            lines.flush()

    try:
        write({"base": True, "variant": base.variant,
               "defects": {k: _defects(v) for k, v in bases.items()}, "runs": _runs(base),
               "ambiguity_keys": _keys(base.ambiguity_keys),
               "unobservable": _defects(base.unobservable), "theta": base.theta,
               "alpha": base.alpha, "kappa": base.kappa, "family_effect": base.family_effect,
               "converged": base.converged, "iterations": base.iterations,
               "items": base.items, "cells": base.cells,
               "literals": [x.as_dict() for x in base.literals], "touches": touched})
        for m in mutants:
            report: Report | None = None
            try:
                ctx = bench.loop_context(m.variant, cfg)
                report = run_specdx(m.variant, ctx, cfg, base=base)
            except INFRASTRUCTURE_ERRORS:
                raise
            except Exception as exc:  # noqa: BLE001 - one broken mutant must not stop the gate
                failed[m.id] = f"{type(exc).__name__}: {exc}"
                write(_mutant_line(m, None, bases, report, failed[m.id], touched[m.id],
                                   m.id in keep, base))
                continue
            if report.behaviours and all(b.status == "error" for b in report.behaviours):
                failed[m.id] = report.behaviours[0].error
                write(_mutant_line(m, None, bases, report, failed[m.id], touched[m.id],
                                   m.id in keep, base))
                continue
            truths[m.id] = m.truth
            reports["specdx"][m.id] = report.defects
            reports["b2"][m.id], reports["b3"][m.id] = _baselines(
                report.behaviours, ctx, report.theta, canon_ids
            )
            found = {n: reports[n][m.id] for n in reports}
            write(_mutant_line(m, found, bases, report, None, touched[m.id], m.id in keep, base))
    finally:
        if lines is not None:
            lines.close()
    scores = {n: evaluate_reports(reports[n], truths, bases[n]) for n in reports}
    untyped = {n: evaluate_reports(reports[n], truths, bases[n], typed=False) for n in reports}

    def primary_of(typed: bool) -> dict[str, dict[str, Prf]]:
        return {n: evaluate_reports({k: v for k, v in reports[n].items() if k in keep},
                                    truths, bases[n], typed=typed) for n in reports}

    result = GateResult(scores, untyped, False, {}, failed, literal_wins(scores),
                        primary_of(True), primary_of(False), touched,
                        {m.id: m.truth.type for m in mutants})
    passed, wins = gate_verdict(result.primary, result.primary_untyped,
                                scorable=result.counts(True), min_scorable=MIN_SCORABLE)
    return dataclasses.replace(result, passed=passed and not failed, wins=wins)


def gate_markdown(result: GateResult) -> str:
    def f1(table: dict[str, dict[str, Prf]], method: str, t: str) -> str:
        return f"{table[method][t].f1:.2f}"

    def win(w: dict[str, bool], t: str) -> str:
        return "yes" if w.get(t) else "no"

    lines = [f"# SpecDx gate v3: {result.headline}", ""]
    if result.failed:
        lines += [f"{len(result.failed)} mutant(s) could not run, so no verdict is given.", ""]
    scorable_n, all_n = result.counts(True), result.counts(False)
    lines += [
        "## Verdict (primary table: scorable mutants)",
        "",
        "The gate needs 3 wins of 4 (spec §4.8, §11 item 12, v2 §4.2, v3 §8.3). Contradiction and "
        "hidden: SpecDx's typed F1 must exceed both baselines' untyped F1, since the "
        "baselines cannot assign a type. Ambiguity and weak: the baselines report clauses, "
        "which can never locate those truths, so SpecDx's typed F1 must reach "
        f"{ABSOLUTE_BAR:.2f}. A type with fewer than {MIN_SCORABLE} scorable mutants is a "
        "loss.",
        "",
        "| type | scorable | rule | SpecDx typed | B2 untyped | B3 untyped | SpecDx untyped | "
        "SpecDx wins |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for t in DEFECT_TYPES:
        lines.append(
            f"| {t} | {scorable_n[t]} of {all_n[t]} | {gate_rule(t)} | "
            f"{f1(result.primary, 'specdx', t)} | {f1(result.primary_untyped, 'b2', t)} | "
            f"{f1(result.primary_untyped, 'b3', t)} | "
            f"{f1(result.primary_untyped, 'specdx', t)} | {win(result.wins, t)} |"
        )
    lines += [
        "",
        "## Secondary table: every mutant",
        "",
        "| type | excluded (not scorable) | SpecDx typed | B2 untyped | B3 untyped | "
        "SpecDx untyped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in DEFECT_TYPES:
        lines.append(
            f"| {t} | {all_n[t] - scorable_n[t]} | {f1(result.scores, 'specdx', t)} | "
            f"{f1(result.untyped, 'b2', t)} | {f1(result.untyped, 'b3', t)} | "
            f"{f1(result.untyped, 'specdx', t)} |"
        )
    lines += [
        "",
        "## Literal comparison (typed against typed, every mutant), for transparency only",
        "",
        "| type | SpecDx typed | B2 typed | B3 typed | SpecDx wins |",
        "|---|---:|---:|---:|---|",
    ]
    for t in DEFECT_TYPES:
        lines.append(
            f"| {t} | {f1(result.scores, 'specdx', t)} | {f1(result.scores, 'b2', t)} | "
            f"{f1(result.scores, 'b3', t)} | {win(result.literal_wins, t)} |"
        )
    lines += ["", "## Scorability (base runs touching each mutant's truth targets)", "",
              "| mutant | type | base behaviours touching | scorable |", "|---|---|---:|---|"]
    lines += [f"| {mid} | {result.types[mid]} | {n} | "
              f"{'yes' if mid in result.scorable else 'no'} |"
              for mid, n in sorted(result.touches.items())]
    if result.failed:
        lines += ["", "## Mutants that could not run", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(result.failed.items())]
    return "\n".join(lines) + "\n"
