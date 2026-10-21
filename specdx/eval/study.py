"""The comparison study (baselines spec §3, §4.1): SpecDx and every baseline arm on one
split, scored with the gate's own matching. A split's base is the differential's reference
for every arm. LLM arms on the dev split are (baseline, model[, strategy]) for each
population model (decision 6); on the test split, the frozen arm per baseline."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from specdx.active.loop import SpecDxConfig, run_specdx  # noqa: F401 - SpecDxConfig for tests
from specdx.baselines.api import REGISTRY, BaselineInput, BaselineOutput, above, load_all
from specdx.baselines.llm import DigestMismatch
from specdx.cost import Cost, behaviours_cost
from specdx.eval.matching import Prf, evaluate_reports, match_findings
from specdx.eval.observability import scorable as scorable_ids
from specdx.eval.observability import touch_counts
from specdx.eval.pilot import INFRASTRUCTURE_ERRORS
from specdx.infer.cells import draft_tables
from specdx.mutate.operators import GroundTruth
from specdx.spec.clauses import ClauseSet
from specdx.types import DEFECT_TYPES, Defect

CHECKPOINTS: tuple[int, ...] = (25, 50, 100, 150, 200)
# Outages stop the step (spec §5.3); a rerun resumes from the run and baseline caches.
STOPS: tuple[type[BaseException], ...] = (*INFRASTRUCTURE_ERRORS, DigestMismatch)
load_all()  # every baseline registers before arms are listed


@dataclass(frozen=True)
class Arm:
    name: str
    model: str | None = None
    strategy: str | None = None

    @property
    def label(self) -> str:
        out = self.name + (f"@{self.model}" if self.model else "")
        return out + (f"/{self.strategy}" if self.strategy else "")


def dev_arms(models_by_family: Mapping[str, Sequence[str]]) -> list[Arm]:
    models = sorted(m for ms in models_by_family.values() for m in ms)
    arms: list[Arm] = []
    for name in sorted(REGISTRY):
        b = REGISTRY[name]
        for strategy in b.strategies:
            if b.uses_llm:
                arms += [Arm(name, m, strategy) for m in models]
            else:
                arms.append(Arm(name, None, strategy))
    return arms


def _defect(raw: Mapping[str, Any]) -> Defect:
    return Defect(raw["type"], tuple(raw["location"]), tuple(raw["scenarios"]),
                  tuple(raw["witnesses"]), float(raw["score"]), raw.get("note", ""),
                  raw.get("subtype", ""))


@dataclass
class SplitResult:
    """failures: per arm label, the specifications it could not report on, with the reason.
    A failed (arm, specification) is left out of that arm's scores."""

    split: str
    truths: dict[str, GroundTruth]
    scorable: set[str]
    defects: dict[str, dict[str, list[Defect]]]
    costs: dict[str, dict[str, dict[str, Any]]]
    failures: dict[str, dict[str, str]]
    checkpoints: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "split": self.split,
            "truths": {k: {"type": t.type, "location": sorted(t.location),
                           "kind": t.location_kind} for k, t in sorted(self.truths.items())},
            # observed_at (scorability metadata) is not kept: scorable is stored instead.
            "scorable": sorted(self.scorable),
            "defects": {a: {s: [asdict(d) for d in ds] for s, ds in sorted(by.items())}
                        for a, by in sorted(self.defects.items())},
            "costs": {a: dict(sorted(by.items())) for a, by in sorted(self.costs.items())},
            "failures": {a: dict(sorted(by.items())) for a, by in sorted(self.failures.items())},
            "checkpoints": dict(sorted(self.checkpoints.items())),
        }, ensure_ascii=False, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SplitResult:
        raw = json.loads(text)
        return cls(raw["split"],
                   {k: GroundTruth(k, t["type"], frozenset(t["location"]), t["kind"])
                    for k, t in raw["truths"].items()},
                   set(raw["scorable"]),
                   {a: {s: [_defect(d) for d in ds] for s, ds in by.items()}
                    for a, by in raw["defects"].items()},
                   raw["costs"], raw["failures"], raw.get("checkpoints", {}))


def _reports(result: SplitResult, arm: str, tau: float, primary: bool):
    by = result.defects[arm]
    keep = (lambda k: k in result.scorable) if primary else (lambda k: True)
    reports = {k: above(v, tau) for k, v in by.items() if k != "base" and k in result.truths
               and keep(k) and k not in result.failures.get(arm, {})}
    return reports, above(by.get("base", []), tau)


def score(result: SplitResult, arm: str, tau: float = 0.0, primary: bool = True,
          typed: bool = True) -> dict[str, Prf]:
    reports, base = _reports(result, arm, tau, primary)
    return evaluate_reports(reports, result.truths, base, typed=typed)


def rank_metrics(result: SplitResult, arm: str, tau: float = 0.0,
                 ks: Sequence[int] = (1, 3, 5)) -> dict[str, dict[str, float]]:
    """One truth per mutant, so AP is the reciprocal rank of the first typed hit."""
    reports, base = _reports(result, arm, tau, primary=True)
    per: dict[str, list[int | None]] = {t: [] for t in DEFECT_TYPES}
    for k, ds in reports.items():
        truth = result.truths[k]
        new, hits = match_findings(ds, truth, base, typed=True)
        rank = next((i + 1 for i, d in enumerate(new) if d in hits), None)
        per[truth.type].append(rank)
    out = {}
    for t, ranks in per.items():
        n = len(ranks) or 1
        row = {"ap": sum(1 / r for r in ranks if r) / n}
        row.update({f"detect@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks})
        out[t] = row
    return out


def equal_cost(result: SplitResult, arm: str) -> dict[str, list[Defect]]:
    """Per specification, SpecDx at its last checkpoint within the arm's requests there."""
    out: dict[str, list[Defect]] = {}
    for spec, cps in result.checkpoints.items():
        budget = int(result.costs.get(arm, {}).get(spec, {}).get("requests", 0))
        fit = [c for c in cps if int(c["cost"]["requests"]) <= budget]
        out[spec] = [_defect(d) for d in fit[-1]["defects"]] if fit else []
    return out


def make_input(bench, variant, report, ctx, cfg, gateway) -> BaselineInput:
    """The variant's pack is loaded first, so golden_state is that variant's own (B5)."""
    from tonomyx.canon.pack import CanonPack

    pack = CanonPack(variant.pack)
    bench.db.use(variant.pack)
    return BaselineInput(
        variant=variant, report=report, data=ctx.cells.extract(report.behaviours),
        violated={b.run_key: ctx.lens.violations(b) for b in report.behaviours},
        clauses=ClauseSet.from_pack(pack, bench.steps), ctx=ctx, config=cfg, steps=bench.steps,
        statements=bench.variants.statements, drafts=draft_tables(pack),
        golden=bench.db.golden_state(), gateway=gateway)


def _specdx_failure(report) -> str | None:
    """As in the gate: a report whose every run errored is a failed specification."""
    bs = report.behaviours
    return bs[0].error if bs and all(b.status == "error" for b in bs) else None


def run_split(bench, config, mutants, arms: Sequence[Arm], gateway, out_lines: Path,
              split: str) -> SplitResult:
    """SpecDx (from the run cache, with checkpoints) then every arm, per specification.
    One JSON line per specification is written to out_lines as it finishes.

    Infrastructure errors and digest mismatches propagate (spec §5.3). Any other error
    fails that (arm, specification) only; a mutant SpecDx cannot run fails every arm on it,
    since the baselines read SpecDx's runs."""
    cfg = dataclasses.replace(config.specdx(), checkpoints=CHECKPOINTS)
    base_cfg = dataclasses.replace(cfg, localize=False)
    base_ctx = bench.loop_context(bench.base, cfg)
    base = run_specdx(bench.base, base_ctx, base_cfg)
    touched = touch_counts({m.id: m.truth.observed_at for m in mutants}, base.behaviours,
                           base_ctx.lens)
    result = SplitResult(split, {m.id: m.truth for m in mutants}, set(scorable_ids(touched)),
                         {"specdx": {}}, {"specdx": {}}, {}, {})

    def fail(label: str, spec: str, reason: str) -> None:
        result.failures.setdefault(label, {})[spec] = reason

    out_lines.parent.mkdir(parents=True, exist_ok=True)
    with out_lines.open("w", encoding="utf-8") as fh:
        for spec, variant, report, ctx in [("base", bench.base, base, base_ctx)] + [
                (m.id, m.variant, None, None) for m in mutants]:
            failure = None
            if report is None:
                try:
                    ctx = bench.loop_context(variant, cfg)
                    report = run_specdx(variant, ctx, cfg, base=base)
                    failure = _specdx_failure(report)
                except STOPS:
                    raise
                except Exception as exc:  # noqa: BLE001 - one broken mutant, as in the gate
                    failure = f"{type(exc).__name__}: {exc}"
            if failure is not None:
                result.defects["specdx"][spec] = []
                result.costs["specdx"][spec] = (Cost() if report is None else behaviours_cost(
                    report.behaviours, report.probe_runs + report.ddmin_runs)).as_dict()
                result.checkpoints[spec] = []
                fail("specdx", spec, failure)
                for arm in arms:
                    result.defects.setdefault(arm.label, {})[spec] = []
                    result.costs.setdefault(arm.label, {})[spec] = Cost().as_dict()
                    fail(arm.label, spec, f"no SpecDx report: {failure}")
                fh.write(json.dumps({"spec": spec, "done": True, "failed": True}) + "\n")
                fh.flush()
                continue
            result.defects["specdx"][spec] = list(report.defects)
            result.costs["specdx"][spec] = behaviours_cost(
                report.behaviours, report.probe_runs + report.ddmin_runs).as_dict()
            result.checkpoints[spec] = report.checkpoints
            inp = make_input(bench, variant, report, ctx, cfg, gateway)
            for arm in arms:
                b = REGISTRY[arm.name]
                try:
                    out: BaselineOutput = b.run(inp, model=arm.model, strategy=arm.strategy)
                except STOPS:
                    raise
                except Exception as exc:  # noqa: BLE001 - one arm, one spec
                    out = BaselineOutput([], failure=f"{type(exc).__name__}: {exc}")
                result.defects.setdefault(arm.label, {})[spec] = out.defects
                result.costs.setdefault(arm.label, {})[spec] = out.cost.as_dict()
                if out.failure:
                    fail(arm.label, spec, out.failure)
            fh.write(json.dumps({"spec": spec, "done": True}) + "\n")
            fh.flush()
    return result
