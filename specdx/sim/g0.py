"""G0 (spec v3.1 §8.2, §9): choose the free parameters by simulation, then check G0.

The grid and the selection rule are fixed here, before G0 runs. For each setting of the
priors (π and the form shares, τ, μ_α and σ_α; κ is fixed at 10) the anchored model is
fitted on REPLICATES synthetic specifications per arm (the calibrated configuration, A3
violated, A5 violated); each threshold setting (τ_H, τ_A, τ_W) is then scored on the same
fits. The objective is the mean typed F1 over hidden requirement, ambiguity and weak
check, averaged over the three arms with equal weight. Ties go to the more conservative
setting: higher P(|A| = 1) on an anchored cell (1 − π + π·s_mis), then smaller π, then
grid order.

At the chosen setting G0 checks §8.2's criteria: (i) mean |θ̂ − θ| ≤ 0.05; (ii) typed F1
≥ 0.6 for hidden requirement and ≥ 0.4 for ambiguity (the weak check's F1 is reported
against 0.4 for information only: spec v3.1 §13 item 15); (iii) at the
calibrated point, under the A3 and A5 arms, and at the population's own M and Δ, v3.1 is
never below the best baseline by more than 0.05 on any type (weak included), and at the
calibrated point it beats the best baseline on ambiguity by more than 0.05. If G0 fails, the
work stops.
The criteria are judged on the selection replicates; the next replicates' seeds give an
out-of-sample report for information only (ruling F22). Non-converged fits are counted in
the search and at every criteria point and reported (§3.4, ruling F2).

Usage:
  python -m specdx.sim.g0 run --calibration PATH --out results/p2-g0 [--params-out PATH]
  python -m specdx.sim.g0 freeze PATH        # rewrite V3Params from G0's chosen values
  python -m specdx.sim.g0 table              # V3Params as the pre-registration's table
Exit status of run: 0 PASS, 1 FAIL, 2 error (an unexpected exception included). While
PARAMS_FROZEN is true, run refuses to overwrite experiments/p2-g0-params.json: give
--params-out to reproduce G0 elsewhere.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specdx.infer.anchored import PARAMS_FROZEN, AnchoredFit, V3Params, fit_anchored
from specdx.infer.typing import (
    AMBIGUOUS_PICK,
    CellFinding,
    ambiguity_finding,
    hidden_finding,
    resolve_calls,
    weak_finding,
)
from specdx.sim.baselines import by_cell
from specdx.sim.generator import SimConfig, SimSpec, generate
from specdx.sim.metrics import TYPES, Counts, theta_error, typed_counts
from specdx.sim.study import (
    BASELINES,
    MARGIN,
    METHODS,
    REGIME_ACCURACY,
    REGIME_DELTA,
    load_config,
    regime_config,
    score,
)

# The grid (§9: "written in the simulation module before G0 runs").
PI: tuple[float, ...] = (0.05, 0.1, 0.2)
SHARES: tuple[tuple[float, float, float], ...] = ((0.6, 0.2, 0.2), (0.4, 0.3, 0.3),
                                                  (0.2, 0.4, 0.4))
BETA_SD: tuple[float, ...] = (1.0, 2.0)
ALPHA_PRIOR: tuple[tuple[float, float], ...] = ((0.0, 1.5), (1.0, 1.0))
TAU_H: tuple[float, ...] = (0.5, 0.7, 0.9)
TAU_A: tuple[float, ...] = (0.3, 0.5, 0.7)
TAU_W: tuple[float, ...] = (0.6, 0.8, 0.9)
ARMS: tuple[str, ...] = ("base", "a3", "a5")
REPLICATES = 10  # §8.1: at least 10 per arm at the calibrated point
# G0's criteria (§8.2).
THETA_ERROR_MAX = 0.05
F1_MIN: dict[str, float] = {"hidden": 0.6, "ambiguity": 0.4, "weak": 0.4}
# Which bars gate (ii): the weak bar is reported, never decisive (spec v3.1 §8.2, §13 item 15).
F1_GATED: tuple[str, ...] = ("hidden", "ambiguity")
AMBIGUITY_WIN = 0.05  # (iii): at the calibrated point v3.1 beats the best baseline by more
PASS, FAIL = "PASS", "FAIL"
PARAMS_FILE = Path(__file__).resolve().parents[2] / "experiments" / "p2-g0-params.json"
ANCHORED = Path(__file__).resolve().parents[1] / "infer" / "anchored.py"
_BLOCK = re.compile(
    r"# --- G0 frozen parameters: begin ---\n.*?# --- G0 frozen parameters: end ---", re.DOTALL)


def prior_settings(grid: dict[str, Sequence[Any]] | None = None) -> list[V3Params]:
    """Every setting of the priors, in grid order, with the first thresholds."""
    g = grid or {}
    out = []
    for pi, shares, beta_sd, (mu, sigma) in itertools.product(
            g.get("pi", PI), g.get("shares", SHARES), g.get("beta_sd", BETA_SD),
            g.get("alpha_prior", ALPHA_PRIOR)):
        out.append(V3Params(pi=pi, s_mis=shares[0], s_amb=shares[1], s_empty=shares[2],
                            beta_sd=beta_sd, mu_alpha=mu, sigma_alpha=sigma,
                            tau_h=TAU_H[0], tau_a=TAU_A[0], tau_w=TAU_W[0]))
    return out


def thresholds(grid: dict[str, Sequence[Any]] | None = None) -> list[tuple[float, float, float]]:
    g = grid or {}
    return list(itertools.product(g.get("tau_h", TAU_H), g.get("tau_a", TAU_A),
                                  g.get("tau_w", TAU_W)))


def conservative(p: V3Params) -> tuple[float, float]:
    """The tie-break: higher P(|A| = 1) on an anchored cell, then smaller π."""
    return (-(1 - p.pi + p.pi * p.s_mis), p.pi)


@dataclass
class Candidate:
    """nonconverged: per arm, the fits of this prior setting that did not converge (§3.4:
    reported, never silent); the threshold settings of one prior setting share them."""

    params: V3Params
    objective: float
    arms: dict[str, dict[str, float]]
    nonconverged: dict[str, int] = dataclasses.field(default_factory=dict)


def potential(spec: SimSpec, fit: AnchoredFit) -> list[CellFinding]:
    """Every finding the typing core makes at zero thresholds: each threshold setting then
    keeps a subset of them, without typing the cells again."""
    readings = by_cell(spec)
    out: list[CellFinding] = []
    for cid in sorted(fit.cells):
        cell, post, mine = spec.cells[cid], fit.cells[cid], readings[cid]
        for found in (hidden_finding(cell, post, mine, 0.0),
                      ambiguity_finding(cell, post, mine, 0.0),
                      weak_finding(cell, post, mine, 0.0)):
            if found is not None:
                out.append(found)
    return out


def counts_at(spec: SimSpec, found: Sequence[CellFinding], tau_h: float, tau_a: float,
               tau_w: float) -> dict[str, Counts]:
    """typing.type_cells at these thresholds, from the zero-threshold findings. On an
    anchored cell an ambiguity and its hidden requirement co-occur (the first half of
    typing.co_occur here; the rest is typing.resolve_calls, which type_cells shares): a
    hidden candidate whose form is a pair brings its ambiguity, and a cell reporting a
    hidden requirement or an ambiguity reports no weak check."""
    by_cell_found: dict[str, dict[str, CellFinding]] = {}
    for f in found:
        by_cell_found.setdefault(f.cell, {})[f.type] = f
    kept: list[tuple[str, str]] = []
    for cid in sorted(by_cell_found):
        fs, anchored = by_cell_found[cid], spec.cells[cid].anchor is not None
        hidden, pair, weak = fs.get("hidden"), fs.get("ambiguity"), fs.get("weak")
        hidden = hidden if hidden is not None and hidden.score >= tau_h else None
        ambiguity = pair if pair is not None and pair.score >= tau_a else None
        if anchored and ambiguity is None and hidden is not None and (
                hidden.subtype == AMBIGUOUS_PICK):
            ambiguity = pair  # at threshold 0, as co_occur asks for it
        calls = {f.type for f in (hidden, ambiguity) if f is not None}
        if weak is not None and weak.score >= tau_w:
            calls.add("weak")
        kept += [(cid, t) for t in sorted(resolve_calls(anchored, calls))]
    return typed_counts(kept, spec.truth)


def search(base: SimConfig, replicates: int = REPLICATES,
           grid: dict[str, Sequence[Any]] | None = None,
           progress: Any = None) -> tuple[Candidate, list[Candidate]]:
    """Fit every prior setting on every arm and replicate, score every threshold setting,
    and choose by the selection rule. Returns the chosen candidate and all of them."""
    specs = {arm: [generate(dataclasses.replace(base, arm=arm, seed=base.seed + i))
                   for i in range(replicates)] for arm in ARMS}
    candidates: list[Candidate] = []
    for n, prior in enumerate(prior_settings(grid)):
        found: dict[str, list[list[CellFinding]]] = {arm: [] for arm in ARMS}
        nonconverged = dict.fromkeys(ARMS, 0)
        for arm in ARMS:
            for s in specs[arm]:
                fit = fit_anchored(s.cells, s.readings, models=s.models, params=prior)
                nonconverged[arm] += not fit.converged
                found[arm].append(potential(s, fit))
        for tau_h, tau_a, tau_w in thresholds(grid):
            params = dataclasses.replace(prior, tau_h=tau_h, tau_a=tau_a, tau_w=tau_w)
            arms: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                total = {t: Counts() for t in TYPES}
                for spec, mine in zip(specs[arm], found[arm], strict=True):
                    for t, c in counts_at(spec, mine, tau_h, tau_a, tau_w).items():
                        total[t].add(c)
                arms[arm] = {t: total[t].f1 for t in TYPES}
            objective = sum(sum(a.values()) / len(TYPES) for a in arms.values()) / len(ARMS)
            candidates.append(Candidate(params, objective, arms, dict(nonconverged)))
        if progress is not None:
            progress(n + 1)
    return choose(candidates), candidates


def choose(candidates: Sequence[Candidate]) -> Candidate:
    """The selection rule: the highest objective (to 1e-9); ties to the more conservative
    setting, then to the earlier one in grid order."""
    ranked = sorted(enumerate(candidates), key=lambda x: (
        -round(x[1].objective, 9), *conservative(x[1].params), x[0]))
    return ranked[0][1]


def population_point(base: SimConfig) -> tuple[int, float, float]:
    """The regime-map point of the population itself (§8.2 (iii)): its number of models,
    and the grid accuracy and spread nearest its mean accuracy and its accuracy range."""
    accuracy = [1 / (1 + math.exp(-a)) for a in base.alphas]
    mean, spread = sum(accuracy) / len(accuracy), max(accuracy) - min(accuracy)
    return (len(accuracy), min(REGIME_ACCURACY, key=lambda a: (abs(a - mean), a)),
            min(REGIME_DELTA, key=lambda d: (abs(d - spread), d)))


def evaluations(base: SimConfig) -> dict[str, SimConfig]:
    """Where criterion (iii) is judged: the calibrated point, its A3 and A5 arms, and the
    population's own point of the regime map."""
    m, accuracy, delta = population_point(base)
    return {"calibrated": dataclasses.replace(base, arm="base"),
            "a3": dataclasses.replace(base, arm="a3"),
            "a5": dataclasses.replace(base, arm="a5"),
            f"own M={m} acc={accuracy} Δ={delta}": regime_config(base, m, accuracy, delta)}


def holds(f1: dict[str, dict[str, float]]) -> dict[str, bool]:
    """Per type: v3.1 is not below the best baseline by more than MARGIN."""
    return {t: f1["v3"][t] - max(f1[b][t] for b in BASELINES) >= -MARGIN for t in TYPES}


def check(base: SimConfig, params: V3Params, replicates: int = REPLICATES,
          offset: int = 0) -> dict[str, Any]:
    """G0's criteria (§8.2) at the chosen parameters, on seeds base.seed + offset + i. G0
    judges them on the selection replicates (offset 0, ruling F22); `run` adds the next
    `replicates` seeds as an out-of-sample report, for information only.
    nonconverged: per evaluation point, the fits that did not converge (§3.4)."""
    where: dict[str, dict[str, dict[str, float]]] = {}
    errors: list[float] = []
    nonconverged: dict[str, int] = {}
    for name, config in evaluations(base).items():
        totals = {m: {t: Counts() for t in TYPES} for m in METHODS}
        nonconverged[name] = 0
        for i in range(replicates):
            spec = generate(dataclasses.replace(config, seed=base.seed + offset + i))
            fit = fit_anchored(spec.cells, spec.readings, models=spec.models, params=params)
            nonconverged[name] += not fit.converged
            for m, by in score(spec, fit, params).counts.items():
                for t, c in by.items():
                    totals[m][t].add(c)
            if name == "calibrated":
                errors.append(theta_error(fit.theta, spec.theta))
        where[name] = {m: {t: totals[m][t].f1 for t in TYPES} for m in METHODS}
    calibrated = where["calibrated"]
    error = sum(errors) / len(errors)
    best_ambiguity = max(calibrated[b]["ambiguity"] for b in BASELINES)
    return {"theta_error": error, "f1": where,
            "i": error <= THETA_ERROR_MAX,
            "ii": {t: calibrated["v3"][t] >= F1_MIN[t] for t in F1_GATED},
            "ii_weak_information": calibrated["v3"]["weak"] >= F1_MIN["weak"],
            "iii": {name: holds(f1) for name, f1 in where.items()},
            "ambiguity_win": calibrated["v3"]["ambiguity"] - best_ambiguity > AMBIGUITY_WIN,
            "nonconverged": nonconverged}


def verdict(criteria: dict[str, Any]) -> str:
    """PASS when (i), (ii) on hidden and ambiguity, and every part of (iii) hold. The weak
    check's F1 against its bar (`ii_weak_information`) never changes the verdict (§13 item
    15)."""
    ok = (criteria["i"] and all(criteria["ii"].values()) and criteria["ambiguity_win"]
          and all(all(by.values()) for by in criteria["iii"].values()))
    return PASS if ok else FAIL


def params_dict(p: V3Params) -> dict[str, float]:
    return dataclasses.asdict(p)


def _yes(ok: bool) -> str:
    return "yes" if ok else "no"


def render(result: dict[str, Any]) -> str:
    chosen = result["chosen"]
    c = result["criteria"]
    lines = ["<!-- generated by python -m specdx.sim.g0 -->", "",
             f"# G0: {result['verdict']}", "",
             f"Calibration: `{result['calibration']}`. Replicates per arm: "
             f"{result['replicates']}. Grid: {result['settings']} prior settings x "
             f"{result['thresholds']} threshold settings. Runtime: {result['seconds']:.0f} s.",
             "", "## Chosen parameters (selection rule of spec v3.1 §9)", "",
             "| parameter | value |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in chosen["params"].items()]
    lines += ["", f"Objective (mean typed F1, arms equally weighted): {chosen['objective']:.4f}.",
              "", "| arm | hidden | ambiguity | weak |", "|---|---:|---:|---:|"]
    lines += [f"| {arm} | {f['hidden']:.3f} | {f['ambiguity']:.3f} | {f['weak']:.3f} |"
              for arm, f in chosen["arms"].items()]
    nc = result["nonconverged"]
    lines += ["", f"Non-converged fits (§3.4): {nc['search']} of {nc['search_fits']} in the "
              f"search; at the chosen setting {nc['chosen']}; at the criteria points "
              f"{c['nonconverged']}; out of sample {result['out_of_sample']['nonconverged']}."]
    lines += ["", "## Criteria (spec v3.1 §8.2, on the selection replicates)", "",
              *_criteria_lines(c)]
    o = result["out_of_sample"]
    lines += [f"## Out of sample (information only, not a criterion): seeds base + "
              f"{result['replicates']}…{2 * result['replicates'] - 1}", "",
              f"The same criteria would give {verdict(o)}.", "", *_criteria_lines(o)]
    return "\n".join(lines)


def _criteria_lines(c: dict[str, Any]) -> list[str]:
    calibrated = c["f1"]["calibrated"]
    lines = [f"- (i) mean abs θ error {c['theta_error']:.4f} ≤ {THETA_ERROR_MAX}: "
             f"{_yes(c['i'])}"]
    lines += [f"- (ii) {t} F1 {calibrated['v3'][t]:.3f} ≥ {F1_MIN[t]}: {_yes(c['ii'][t])}"
              for t in F1_GATED]
    lines += [f"- (ii) weak F1 {calibrated['v3']['weak']:.3f} against {F1_MIN['weak']}: "
              f"{_yes(c['ii_weak_information'])} (information only, spec v3.1 §13 item 15: it "
              "does not gate the verdict)"]
    lines += [f"- (iii) {name}: not below the best baseline by more than {MARGIN}: "
              + ", ".join(f"{t} {_yes(ok)}" for t, ok in by.items())
              for name, by in c["iii"].items()]
    lines += [f"- (iii) ambiguity win at the calibrated point (> {AMBIGUITY_WIN} over the best "
              f"baseline): {_yes(c['ambiguity_win'])}", ""]
    for name, f1 in c["f1"].items():
        lines += [f"### {name}", "", "| method | hidden | ambiguity | weak |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {m} | {f['hidden']:.3f} | {f['ambiguity']:.3f} | {f['weak']:.3f} |"
                  for m, f in f1.items()]
        lines.append("")
    return lines


def run(calibration: Path, out: Path, *, replicates: int = REPLICATES,
        grid: dict[str, Sequence[Any]] | None = None, params_out: Path = PARAMS_FILE) -> str:
    started = time.monotonic()
    base = load_config(calibration)
    total = len(prior_settings(grid))
    chosen, candidates = search(
        base, replicates, grid,
        progress=lambda n: print(f"G0: {n}/{total} prior settings", file=sys.stderr))
    criteria = check(base, chosen.params, replicates)
    out_of_sample = check(base, chosen.params, replicates, offset=replicates)
    per_prior = {dataclasses.replace(x.params, tau_h=0.0, tau_a=0.0, tau_w=0.0): x.nonconverged
                 for x in candidates}
    nonconverged = {"search": sum(sum(n.values()) for n in per_prior.values()),
                    "search_fits": len(per_prior) * len(ARMS) * replicates,
                    "chosen": chosen.nonconverged}
    result = {"verdict": verdict(criteria), "calibration": str(calibration),
              "replicates": replicates, "settings": total, "thresholds": len(thresholds(grid)),
              "seconds": time.monotonic() - started, "base": dataclasses.asdict(base),
              "chosen": {"params": params_dict(chosen.params), "objective": chosen.objective,
                         "arms": chosen.arms},
              "criteria": criteria, "nonconverged": nonconverged,
              "out_of_sample": out_of_sample,
              "candidates": [{"params": params_dict(x.params), "objective": x.objective,
                              "arms": x.arms, "nonconverged": x.nonconverged}
                             for x in candidates]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(render(result), encoding="utf-8")
    params_out.parent.mkdir(parents=True, exist_ok=True)
    params_out.write_text(json.dumps({"verdict": result["verdict"],
                                      "params": params_dict(chosen.params)}, indent=2) + "\n",
                          encoding="utf-8")
    return str(result["verdict"])


def frozen_block(params: V3Params) -> str:
    """V3Params with G0's values as its defaults, for specdx/infer/anchored.py."""
    fields = "\n".join(f"    {f.name}: float = {getattr(params, f.name)!r}"
                       for f in dataclasses.fields(V3Params))
    return ("# --- G0 frozen parameters: begin ---\nPARAMS_FROZEN = True\n\n\n"
            "@dataclass(frozen=True)\nclass V3Params:\n"
            '    """The values G0 chose by the selection rule of spec v3.1 §9, frozen in the '
            'v3\n    pre-registration (experiments/p2-preregistration.md)."""\n\n'
            f"{fields}\n# --- G0 frozen parameters: end ---")


# The pre-registration's names for V3Params' fields (spec v3.1 §3.1, §3.3, §5).
NAMES: dict[str, str] = {
    "pi": "π", "s_mis": "s_mis", "s_amb": "s_amb", "s_empty": "s_∅", "beta_sd": "τ (β prior sd)",
    "mu_alpha": "μ_α", "sigma_alpha": "σ_α", "tau_h": "τ_H", "tau_a": "τ_A", "tau_w": "τ_W",
}


def params_table(params: V3Params) -> str:
    """The chosen values as the v3 pre-registration states them."""
    rows = [f"| {NAMES[f.name]} | `{f.name}` | {getattr(params, f.name)!r} |"
            for f in dataclasses.fields(V3Params)]
    return "\n".join(["| parameter | field | value |", "|---|---|---:|", *rows])


def freeze(path: Path, target: Path = ANCHORED) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    params = V3Params(**raw["params"])
    text = target.read_text(encoding="utf-8")
    if not _BLOCK.search(text):
        raise ValueError(f"{target} has no G0 frozen parameters block")
    target.write_text(_BLOCK.sub(lambda _: frozen_block(params), text), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--calibration", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--replicates", type=int, default=REPLICATES)
    r.add_argument("--params-out", type=Path, default=None,
                   help="where to write the chosen parameters (required once G0's are frozen)")
    f = sub.add_parser("freeze")
    f.add_argument("params", type=Path)
    sub.add_parser("table")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.params)
        return 0
    if args.command == "table":
        print(params_table(V3Params()))
        return 0
    if args.params_out is None and PARAMS_FROZEN:
        print(f"error: G0's parameters are frozen; refusing to overwrite {PARAMS_FILE}. "
              "Give --params-out PATH to reproduce G0 elsewhere.", file=sys.stderr)
        return 2
    try:
        outcome = run(args.calibration, args.out, replicates=args.replicates,
                      params_out=args.params_out or PARAMS_FILE)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a crash must not share FAIL's exit status
        print(f"error: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(outcome)
    return {PASS: 0, FAIL: 1}[outcome]


if __name__ == "__main__":
    raise SystemExit(main())
