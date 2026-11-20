"""G0v4 (P4a design §6.1): choose and freeze v4's free parameters by simulation.

The grid and the selection rule are fixed here, before G0v4 runs. The free parameters are
cell_sd and the λ prior (the PRIOR settings: they change the fit, so each is fitted once per
replicate and arm) and tau_u, tau_h and tau_a (the THRESHOLD settings: they only change
typing, so they are scored on the same fits). tau_w = 0.9 and tau_lambda = 0.5 are fixed;
π, the form shares, β's prior and the α prior are v3's frozen values.

Arms: base (the ordinary arm), shared, homogeneous and saturated. The criteria (§6.1) are
judged on the selection replicates (seeds base ... base + n - 1) and reported on the held-out
ones (base + n ... base + 2n - 1):
  1. the hidden false-positive rate among determined cells on the shared arm is at most 0.1
     (v3's rate is reported beside it);
  2. on the ordinary arm, hidden F1 and ambiguity F1 on determined cells are each at least
     v3's minus 0.05 (the undetermined rate is reported);
  3. on the homogeneous and saturated arms, at least 80% of the planted defects and shared
     cells are undetermined.
Scoring: an undetermined cell is neither a prediction nor a miss. It is excluded from TP, FP
and FN, and is reported as an undetermined rate.
Selection: among the candidates that pass every criterion, the best mean typed F1 across
arms; ties go to the larger tau_u, then to grid order. If none passes the verdict is FAIL,
the best candidate by the same rule is reported, with the criteria it fails.

Usage (official runs happen on the 48 GB host; freeze only after a PASS):
  python -m specdx.sim.g0v4 run --config PATH --out results/p4/g0v4 [--replicates 10]
  python -m specdx.sim.g0v4 freeze RESULT.json [--path experiments/p4-s3-v4-params.json]
Exit status of run: 0 PASS, 1 FAIL, 2 error.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from specdx.infer.anchored import AnchoredFit, V3Params, fit_anchored
from specdx.infer.anchored_v4 import AnchoredFitV4, V4Params, fit_v4
from specdx.infer.typing import CellFinding, type_cells
from specdx.infer.typing_v4 import UNDETERMINED, type_cells_v4
from specdx.sim.baselines import by_cell
from specdx.sim.generator import SimConfig, SimSpec, generate
from specdx.sim.metrics import TYPES, Counts, typed_counts
from specdx.sim.study import load_config

CELL_SD: tuple[float, ...] = (0.5, 1.0, 2.0)
LAMBDA_PRIOR: tuple[tuple[float, float], ...] = ((1.0, 3.0), (1.0, 1.0))
TAU_U: tuple[float, ...] = (1.0, 2.0, 3.0)
TAU_H: tuple[float, ...] = (0.5, 0.7, 0.9)
TAU_A: tuple[float, ...] = (0.3, 0.5, 0.7)
TAU_W = 0.9
TAU_LAMBDA = 0.5
ARMS: tuple[str, ...] = ("base", "shared", "homogeneous", "saturated")
REPLICATES = 10
SHARED_FP_MAX = 0.1
F1_MARGIN = 0.05
UNDETERMINED_MIN = 0.8
PLANTED_ARMS: tuple[str, ...] = ("homogeneous", "saturated")
PASS, FAIL = "PASS", "FAIL"
PARAMS_FILE = Path(__file__).resolve().parents[2] / "experiments" / "p4-s3-v4-params.json"
_EPS = 1e-9


def prior_settings(grid: dict[str, Sequence[Any]] | None = None) -> list[V4Params]:
    """Every setting of the priors, in grid order, with the first thresholds. v3's frozen
    values fill every other field."""
    g, v3 = grid or {}, V3Params()
    return [V4Params(pi=v3.pi, s_mis=v3.s_mis, s_amb=v3.s_amb, s_empty=v3.s_empty,
                     beta_sd=v3.beta_sd, mu_alpha=v3.mu_alpha, sigma_alpha=v3.sigma_alpha,
                     cell_sd=sd, lambda_a=a, lambda_b=b, tau_h=g.get("tau_h", TAU_H)[0],
                     tau_a=g.get("tau_a", TAU_A)[0], tau_w=TAU_W, tau_u=g.get("tau_u", TAU_U)[0],
                     tau_lambda=TAU_LAMBDA)
            for sd, (a, b) in itertools.product(g.get("cell_sd", CELL_SD),
                                                g.get("lambda_prior", LAMBDA_PRIOR))]


def thresholds(grid: dict[str, Sequence[Any]] | None = None
               ) -> list[tuple[float, float, float]]:
    """(tau_u, tau_h, tau_a) in grid order."""
    g = grid or {}
    return list(itertools.product(g.get("tau_u", TAU_U), g.get("tau_h", TAU_H),
                                  g.get("tau_a", TAU_A)))


@dataclass
class Scored:
    """One specification's findings scored. counts: per defect type, on determined cells
    only. undetermined: the cells typed undetermined. types: cell → the defect types
    predicted on it."""

    counts: dict[str, Counts]
    undetermined: set[str]
    types: dict[str, set[str]]


def score_findings(spec: SimSpec, found: Iterable[CellFinding]) -> Scored:
    """An undetermined cell is neither a prediction nor a miss: it is dropped from the
    predictions and from the truth before TP, FP and FN are counted."""
    found = list(found)
    undetermined = {f.cell for f in found if f.type == UNDETERMINED}
    types: dict[str, set[str]] = {}
    for f in found:
        if f.type in TYPES and f.cell not in undetermined:
            types.setdefault(f.cell, set()).add(f.type)
    truth = {c: t for c, t in spec.truth.items() if c not in undetermined}
    counts = typed_counts(((c, t) for c, ts in types.items() for t in ts), truth)
    return Scored(counts, undetermined, types)


@dataclass
class ArmTally:
    """The counts of one arm over its replicates, for one parameter setting."""

    counts: dict[str, Counts] = field(default_factory=lambda: {t: Counts() for t in TYPES})
    cells: int = 0
    undetermined: int = 0
    shared_determined: int = 0
    shared_hidden: int = 0
    planted: int = 0
    planted_undetermined: int = 0

    def add(self, spec: SimSpec, scored: Scored) -> None:
        for t, c in scored.counts.items():
            self.counts[t].add(c)
        self.cells += len(spec.cells)
        self.undetermined += len(scored.undetermined)
        for cid, truth in spec.truth.items():
            if truth.shared and cid not in scored.undetermined:
                self.shared_determined += 1
                self.shared_hidden += "hidden" in scored.types.get(cid, ())
            if truth.hidden or truth.ambiguity or truth.weak or truth.shared:
                self.planted += 1
                self.planted_undetermined += cid in scored.undetermined

    @property
    def f1(self) -> dict[str, float]:
        return {t: self.counts[t].f1 for t in TYPES}

    @property
    def shared_rate(self) -> float:
        return self.shared_hidden / self.shared_determined if self.shared_determined else 0.0

    @property
    def planted_rate(self) -> float:
        return self.planted_undetermined / self.planted if self.planted else 1.0

    @property
    def undetermined_rate(self) -> float:
        return self.undetermined / self.cells if self.cells else 0.0


def metrics_of(tallies: dict[str, ArmTally]) -> dict[str, Any]:
    """The numbers the criteria read, from the per-arm tallies."""
    base, shared = tallies["base"], tallies["shared"]
    return {"shared_fp": shared.shared_rate, "shared_determined": shared.shared_determined,
            "hidden_f1": base.f1["hidden"], "ambiguity_f1": base.f1["ambiguity"],
            "undetermined": base.undetermined_rate,
            "planted_undetermined": {a: tallies[a].planted_rate for a in PLANTED_ARMS},
            "planted": {a: tallies[a].planted for a in PLANTED_ARMS}}


def judge(metrics: dict[str, Any], v3: dict[str, float]) -> dict[str, Any]:
    """The three criteria of §6.1 (inclusive bounds, to 1e-9)."""
    return {"1": metrics["shared_fp"] <= SHARED_FP_MAX + _EPS,
            "2": {"hidden": metrics["hidden_f1"] >= v3["hidden_f1"] - F1_MARGIN - _EPS,
                  "ambiguity": metrics["ambiguity_f1"] >= v3["ambiguity_f1"] - F1_MARGIN - _EPS},
            "3": {a: metrics["planted_undetermined"][a] >= UNDETERMINED_MIN - _EPS
                  for a in PLANTED_ARMS}}


def failed(criteria: dict[str, Any]) -> list[str]:
    """The names of the criteria that fail (2 and 3 by part, as `2.hidden`)."""
    out = []
    for key, value in criteria.items():
        if isinstance(value, dict):
            out += [f"{key}.{k}" for k, ok in value.items() if not ok]
        elif not value:
            out.append(key)
    return out


def verdict(criteria: dict[str, Any]) -> str:
    return PASS if not failed(criteria) else FAIL


@dataclass
class Candidate:
    """params: the prior setting with its thresholds. objective: the mean typed F1 over the
    defect types, averaged over the arms with equal weight. passes: every criterion holds
    on the selection replicates."""

    params: V4Params
    objective: float
    arms: dict[str, dict[str, float]]
    passes: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)
    undetermined: dict[str, float] = field(default_factory=dict)
    nonconverged: dict[str, int] = field(default_factory=dict)


def choose(candidates: Sequence[Candidate]) -> Candidate:
    """Candidates that pass first; then the highest objective (to 1e-9); ties to the larger
    tau_u, then to the earlier in grid order."""
    ranked = sorted(enumerate(candidates), key=lambda x: (
        not x[1].passes, -round(x[1].objective, 9), -x[1].params.tau_u, x[0]))
    return ranked[0][1]


def make_specs(base: SimConfig, replicates: int, offset: int = 0) -> dict[str, list[SimSpec]]:
    return {arm: [generate(dataclasses.replace(base, arm=arm, seed=base.seed + offset + i))
                  for i in range(replicates)] for arm in ARMS}


def fit_arms(specs: dict[str, list[SimSpec]], prior: V4Params
             ) -> dict[str, list[AnchoredFitV4]]:
    return {arm: [fit_v4(s.cells, s.readings, models=s.models, params=prior)
                  for s in specs[arm]] for arm in ARMS}


def tally_at(specs: dict[str, list[SimSpec]], fits: dict[str, list[AnchoredFitV4]],
             params: V4Params, readings: dict[str, list[dict[str, list[Any]]]]
             ) -> dict[str, ArmTally]:
    tallies = {arm: ArmTally() for arm in ARMS}
    for arm in ARMS:
        for spec, fit, mine in zip(specs[arm], fits[arm], readings[arm], strict=True):
            found = type_cells_v4(spec.cells, fit, mine, params)
            tallies[arm].add(spec, score_findings(spec, found))
    return tallies


def objective_of(tallies: dict[str, ArmTally]) -> float:
    return sum(sum(t.f1.values()) / len(TYPES) for t in tallies.values()) / len(tallies)


def v3_reference(specs: dict[str, list[SimSpec]], v3: V3Params | None = None
                 ) -> dict[str, Any]:
    """v3 on the same specifications: hidden and ambiguity F1 on the ordinary arm, and the
    share of the shared arm's shared cells it types hidden (v3 never abstains)."""
    v3 = v3 or V3Params()
    counts = {t: Counts() for t in TYPES}
    hit = total = 0
    for arm in ("base", "shared"):
        for spec in specs[arm]:
            fit: AnchoredFit = fit_anchored(spec.cells, spec.readings, models=spec.models,
                                            params=v3)
            found = type_cells(spec.cells, fit, by_cell(spec), v3)
            if arm == "base":
                for t, c in typed_counts(((f.cell, f.type) for f in found), spec.truth).items():
                    counts[t].add(c)
            else:
                hidden = {f.cell for f in found if f.type == "hidden"}
                shared = [c for c, t in spec.truth.items() if t.shared]
                total += len(shared)
                hit += sum(c in hidden for c in shared)
    return {"hidden_f1": counts["hidden"].f1, "ambiguity_f1": counts["ambiguity"].f1,
            "weak_f1": counts["weak"].f1, "shared_hidden": hit / total if total else 0.0,
            "shared_cells": total}


def _readings(specs: dict[str, list[SimSpec]]) -> dict[str, list[dict[str, list[Any]]]]:
    return {arm: [by_cell(s) for s in specs[arm]] for arm in ARMS}


def search(base: SimConfig, replicates: int = REPLICATES,
           grid: dict[str, Sequence[Any]] | None = None, progress: Any = None
           ) -> tuple[Candidate, list[Candidate], dict[str, Any]]:
    """Fit every prior setting once per replicate and arm, type at every threshold setting
    on the same fits, judge the criteria, and choose. Returns the chosen candidate, all of
    them and v3's reference numbers."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    specs = make_specs(base, replicates)
    mine = _readings(specs)
    v3 = v3_reference(specs)
    candidates: list[Candidate] = []
    for n, prior in enumerate(prior_settings(grid)):
        fits = fit_arms(specs, prior)
        nonconverged = {arm: sum(not f.converged for f in fits[arm]) for arm in ARMS}
        for tau_u, tau_h, tau_a in thresholds(grid):
            params = dataclasses.replace(prior, tau_u=tau_u, tau_h=tau_h, tau_a=tau_a)
            tallies = tally_at(specs, fits, params, mine)
            metrics = metrics_of(tallies)
            criteria = judge(metrics, v3)
            candidates.append(Candidate(
                params, objective_of(tallies), {a: t.f1 for a, t in tallies.items()},
                not failed(criteria), metrics, criteria,
                {a: t.undetermined_rate for a, t in tallies.items()}, dict(nonconverged)))
        if progress is not None:
            progress(n + 1)
    return choose(candidates), candidates, v3


def held_out(base: SimConfig, params: V4Params, replicates: int) -> dict[str, Any]:
    """The chosen parameters on the next `replicates` seeds, reported and not judged."""
    specs = make_specs(base, replicates, offset=replicates)
    fits = fit_arms(specs, params)
    tallies = tally_at(specs, fits, params, _readings(specs))
    v3 = v3_reference(specs)
    metrics = metrics_of(tallies)
    return {"metrics": metrics, "v3": v3, "criteria": judge(metrics, v3),
            "arms": {a: t.f1 for a, t in tallies.items()},
            "undetermined": {a: t.undetermined_rate for a, t in tallies.items()},
            "nonconverged": {a: sum(not f.converged for f in fits[a]) for a in ARMS}}


def _yes(ok: bool) -> str:
    return "yes" if ok else "no"


def _criteria_lines(metrics: dict[str, Any], v3: dict[str, Any],
                    criteria: dict[str, Any]) -> list[str]:
    pu = metrics["planted_undetermined"]
    return [
        "| criterion | value | bar | holds |", "|---|---:|---:|---|",
        f"| 1. hidden false-positive rate on shared cells, determined "
        f"(n = {metrics['shared_determined']}) | {metrics['shared_fp']:.3f} "
        f"(v3: {v3['shared_hidden']:.3f}) | <= {SHARED_FP_MAX} | {_yes(criteria['1'])} |",
        f"| 2. ordinary hidden F1 | {metrics['hidden_f1']:.3f} (v3: {v3['hidden_f1']:.3f}) "
        f"| >= v3 - {F1_MARGIN} | {_yes(criteria['2']['hidden'])} |",
        f"| 2. ordinary ambiguity F1 | {metrics['ambiguity_f1']:.3f} "
        f"(v3: {v3['ambiguity_f1']:.3f}) | >= v3 - {F1_MARGIN} "
        f"| {_yes(criteria['2']['ambiguity'])} |",
        f"| 3. homogeneous: planted and shared cells undetermined | {pu['homogeneous']:.3f} "
        f"| >= {UNDETERMINED_MIN} | {_yes(criteria['3']['homogeneous'])} |",
        f"| 3. saturated: planted and shared cells undetermined | {pu['saturated']:.3f} "
        f"| >= {UNDETERMINED_MIN} | {_yes(criteria['3']['saturated'])} |",
        "", f"Undetermined rate on the ordinary arm: {metrics['undetermined']:.3f}."]


def render(result: dict[str, Any]) -> str:
    chosen, hold = result["chosen"], result["held_out"]
    lines = ["<!-- generated by python -m specdx.sim.g0v4 -->", "",
             f"# G0v4: {result['verdict']}", "",
             f"Replicates per arm: {result['replicates']} (selection) and "
             f"{result['replicates']} (held out). Grid: {result['settings']} prior settings x "
             f"{result['thresholds']} threshold settings. Runtime: {result['seconds']:.0f} s.",
             "", "## Chosen parameters" + ("" if result["verdict"] == PASS else
                                           " (best candidate; no candidate passes)"), "",
             "| parameter | value |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in chosen["params"].items()]
    lines += ["", f"Objective (mean typed F1 over arms): {chosen['objective']:.4f}."]
    if result["failed"]:
        lines += ["", "Criteria failed: " + ", ".join(result["failed"]) + "."]
    lines += ["", "## Criteria on the selection replicates", ""]
    lines += _criteria_lines(chosen["metrics"], result["v3"], result["criteria"])
    lines += ["", "## Typed F1 on determined cells, and undetermined rate, per arm", "",
              "| arm | hidden | ambiguity | weak | undetermined |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {arm} | {f['hidden']:.3f} | {f['ambiguity']:.3f} | {f['weak']:.3f} "
              f"| {chosen['undetermined'][arm]:.3f} |" for arm, f in chosen["arms"].items()]
    lines += ["", "## Held-out replicates (reported, not judged)", ""]
    lines += _criteria_lines(hold["metrics"], hold["v3"], hold["criteria"])
    lines += ["", "## Fits that did not converge (selection, chosen prior setting)", "",
              ", ".join(f"{a}: {n}" for a, n in chosen["nonconverged"].items()) or "none"]
    return "\n".join(lines) + "\n"


def params_dict(p: V4Params) -> dict[str, float]:
    return dataclasses.asdict(p)


def run_g0v4(config: SimConfig, out: Path, *, replicates: int = REPLICATES,
             grid: dict[str, Sequence[Any]] | None = None) -> int:
    """Run G0v4 and write `out`.json and `out`.md. Returns 0 PASS, 1 FAIL, 2 error. It
    never freezes: `freeze_v4` is called on a PASS by the caller."""
    try:
        started = time.monotonic()
        total = len(prior_settings(grid))
        chosen, candidates, v3 = search(
            config, replicates, grid,
            progress=lambda n: print(f"G0v4: {n}/{total} prior settings", file=sys.stderr))
        hold = held_out(config, chosen.params, replicates)
        result = {
            "verdict": PASS if chosen.passes else FAIL, "replicates": replicates,
            "settings": total, "thresholds": len(thresholds(grid)),
            "seconds": time.monotonic() - started, "base": dataclasses.asdict(config),
            "chosen": {"params": params_dict(chosen.params), "objective": chosen.objective,
                       "arms": chosen.arms, "metrics": chosen.metrics,
                       "undetermined": chosen.undetermined,
                       "nonconverged": chosen.nonconverged},
            "criteria": chosen.criteria, "failed": failed(chosen.criteria), "v3": v3,
            "held_out": hold,
            "candidates": [{"params": params_dict(x.params), "objective": x.objective,
                            "arms": x.arms, "passes": x.passes, "metrics": x.metrics,
                            "criteria": x.criteria, "undetermined": x.undetermined,
                            "nonconverged": x.nonconverged} for x in candidates]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(render(result), encoding="utf-8")
        return 0 if chosen.passes else 1
    except Exception:
        traceback.print_exc()
        return 2


def freeze_v4(params: V4Params, path: Path = PARAMS_FILE) -> None:
    """Write the chosen parameters (experiments/p4-s3-v4-params.json). Called on PASS only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params_dict(params)}, indent=2) + "\n",
                    encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="specdx.sim.g0v4")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--replicates", type=int, default=REPLICATES)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("result", type=Path)
    freeze.add_argument("--path", type=Path, default=PARAMS_FILE)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        code = run_g0v4(load_config(args.config), args.out, replicates=args.replicates)
        if code == 0:
            print("PASS: freeze with `python -m specdx.sim.g0v4 freeze`", file=sys.stderr)
        return code
    data = json.loads(args.result.read_text(encoding="utf-8"))
    if data["verdict"] != PASS:
        print("refusing to freeze: the verdict is not PASS", file=sys.stderr)
        return 1
    freeze_v4(V4Params(**data["chosen"]["params"]), args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
