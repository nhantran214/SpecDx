"""The simulation study (spec v3.1 §8.1): replicates, sweeps, the regime map, the report.

A replicate generates one synthetic specification, fits the anchored model, types its
cells with the same core the real diagnosis uses (infer.typing), and scores v3 and the
three baselines against the planted truth.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from specdx.infer.anchored import AnchoredFit, V3Params, fit_anchored
from specdx.infer.typing import type_cells
from specdx.sim.baselines import anchor_only, by_cell, majority_vote, stage2_v2
from specdx.sim.generator import SimConfig, SimSpec, generate
from specdx.sim.metrics import TYPES, Counts, calibration, theta_error, typed_counts

METHODS: tuple[str, ...] = ("v3", "majority", "anchor_only", "v2_stage2")
BASELINES: tuple[str, ...] = METHODS[1:]
F1_TARGET = 0.6  # "runs needed to reach a given F1": the G0 bar for hidden
# The regime map of spec v3.1 §8.1: mean cell accuracy × accuracy spread × models, judged
# against the best baseline per type with a margin.
REGIME_ACCURACY: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 0.95)
REGIME_DELTA: tuple[float, ...] = (0.0, 0.1, 0.2)
REGIME_MODELS: tuple[int, ...] = (2, 4)
MARGIN = 0.05


def min_replicates(sweep: str) -> int:
    """§8.1's minimum replicates per point: 5 on the regime map, 10 elsewhere (at least 10
    per arm at the calibrated point). Fewer is reported as exploratory."""
    return 5 if sweep == "regime" else 10


@dataclass
class Result:
    """One replicate, or several merged. counts: method → type → counts. pairs: P(A_c ≠
    {d_c}) and the hidden truth for every anchored cell (calibration)."""

    counts: dict[str, dict[str, Counts]]
    theta_errors: list[float]
    pairs: list[tuple[float, bool]]
    converged: list[bool]
    reads: list[float]
    gaps: list[float]
    sizes: list[float]
    accuracy: list[float] = dataclasses.field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """n: mean readings per model and cell; delta: the accuracy gap of the two most
        accurate models; t4a: Theorem T4(a)'s bound. accuracy: the mean true accuracy at a
        cell of median difficulty; chance: T2′'s chance line |A|/K for |A| = 1 and 2, with
        K the mean number of candidate values; t4c: T4(c)'s bound at that accuracy."""
        ece, brier = calibration(self.pairs)
        n, gap, size = _mean(self.reads), _mean(self.gaps), _mean(self.sizes)
        p, k = _mean(self.accuracy), max(2, round(size))
        return {"methods": {m: {t: c.as_dict() for t, c in by.items()}
                            for m, by in self.counts.items()},
                "theta_error": _mean(self.theta_errors), "ece": ece, "brier": brier,
                "converged": _mean([float(c) for c in self.converged]), "n": n, "delta": gap,
                "t4a": min(1.0, 4 * size * math.exp(-n * gap ** 2 / 32)),
                "accuracy": p, "chance": [1 / k, 2 / k],
                "t4c": min(1.0, 2 * k * math.exp(-n * gamma_min(k, p) ** 2 / 2))}


def gamma_min(k: int, p: float) -> float:
    """T2′/T4(c): the smallest sup-norm distance between the reading distributions that
    two candidate sets (one value, a pair or ∅, over k values) predict at accuracy p under
    the flat error pool. It is 0 where p equals some |A|/k: T2′'s chance line."""

    def predicted(a: tuple[int, ...]) -> list[float]:
        if not a:
            return [1 / k] * k
        return [p / len(a) if y in a else (1 - p) / (k - len(a)) for y in range(k)]

    sets = [(), *((i,) for i in range(k)), *combinations(range(k), 2)]
    shapes = [predicted(a) for a in sets]
    return min(max(abs(x - y) for x, y in zip(u, v, strict=True))
               for u, v in combinations(shapes, 2))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def merge(results: Sequence[Result]) -> Result:
    counts = {m: {t: Counts() for t in TYPES} for m in METHODS}
    for r in results:
        for m, by in r.counts.items():
            for t, c in by.items():
                counts[m][t].add(c)
    return Result(counts, [x for r in results for x in r.theta_errors],
                  [x for r in results for x in r.pairs], [x for r in results for x in r.converged],
                  [x for r in results for x in r.reads], [x for r in results for x in r.gaps],
                  [x for r in results for x in r.sizes], [x for r in results for x in r.accuracy])


def v3_counts(spec: SimSpec, fit: AnchoredFit, params: V3Params) -> dict[str, Counts]:
    """v3's typed counts at the given thresholds (no refit: G0 varies τ_H, τ_A, τ_W)."""
    found = type_cells(spec.cells, fit, by_cell(spec), params)
    return typed_counts(((f.cell, f.type) for f in found), spec.truth)


def score(spec: SimSpec, fit: AnchoredFit, params: V3Params, baselines: bool = True) -> Result:
    counts = {"v3": v3_counts(spec, fit, params)}
    if baselines:
        counts |= {"majority": typed_counts(majority_vote(spec), spec.truth),
                   "anchor_only": typed_counts(anchor_only(spec), spec.truth),
                   "v2_stage2": typed_counts(stage2_v2(spec), spec.truth)}
    pairs = [(fit.cells[c].p_defect, spec.truth[c].hidden) for c in fit.cells
             if spec.cells[c].anchor is not None]
    true = spec.theta
    ranked = sorted(true.values(), reverse=True)
    reads = [len(rs) / len({r.model for r in rs}) for rs in by_cell(spec).values() if rs]
    return Result(counts, [theta_error(fit.theta, true)], pairs, [fit.converged],
                  [_mean(reads)], [ranked[0] - ranked[1] if len(ranked) > 1 else 0.0],
                  [_mean([len(p.values) + 1 for p in fit.cells.values()])], [_mean(ranked)])


def replicate(config: SimConfig, params: V3Params,
              baselines: bool = True) -> tuple[Result, AnchoredFit, SimSpec]:
    spec = generate(config)
    fit = fit_anchored(spec.cells, spec.readings, models=spec.models, params=params)
    return score(spec, fit, params, baselines), fit, spec


def _spread(alphas: tuple[float, ...], scale: float) -> tuple[float, ...]:
    mean = sum(alphas) / len(alphas)
    return tuple(mean + scale * (a - mean) for a in alphas)


def _logit(p: float) -> float:
    p = min(max(p, 0.51), 0.995)
    return math.log(p / (1 - p))


def regime_config(base: SimConfig, models: int, accuracy: float, delta: float) -> SimConfig:
    """The base configuration with `models` models whose accuracies at a cell of median
    difficulty are evenly spread by `delta` around `accuracy` (all α shifted); act rates are
    the base's, in its order."""
    acc = [accuracy + delta * (0.5 - i / (models - 1)) for i in range(models)]
    act = list(base.act) + [base.act[-1]] * max(0, models - len(base.act))
    return dataclasses.replace(base, alphas=tuple(_logit(p) for p in acc),
                               act=tuple(act[:models]))


def regime_points(base: SimConfig) -> list[tuple[str, SimConfig]]:
    return [(f"M={m} acc={a} Δ={d}", regime_config(base, m, a, d))
            for m in REGIME_MODELS for a in REGIME_ACCURACY for d in REGIME_DELTA]


def versus(f1: dict[str, dict[str, float]], margin: float = MARGIN) -> dict[str, str]:
    """Per type: W when v3's F1 beats the best baseline's by more than the margin, L when
    it is below it by more than the margin, else T."""
    out = {}
    for t in TYPES:
        gap = f1["v3"][t] - max(f1[m][t] for m in BASELINES)
        out[t] = "W" if gap > margin else ("L" if gap < -margin else "T")
    return out


def _models(base: SimConfig, m: int) -> SimConfig:
    alphas, act = list(base.alphas), list(base.act)
    while len(alphas) < m:
        alphas.append(min(alphas) - 0.5)
        act.append(act[-1])
    return dataclasses.replace(base, alphas=tuple(alphas[:m]), act=tuple(act[:m]))


Sweep = Callable[[SimConfig], list[tuple[str, SimConfig]]]
# §8.1's sweeps, each varying one factor from the base (calibrated) configuration.
SWEEPS: dict[str, Sweep] = {
    "delta": lambda b: [(f"spread x{s}", dataclasses.replace(b, alphas=_spread(b.alphas, s)))
                        for s in (0.0, 0.5, 1.0, 1.5, 2.0)],
    "models": lambda b: [(f"M={m}", _models(b, m)) for m in (2, 3, 4)],
    "n": lambda b: [(f"seeds={k}", dataclasses.replace(b, seeds=k)) for k in (1, 2, 3, 5, 8)],
    "tau_true": lambda b: [(f"tau={t}", dataclasses.replace(b, beta_sd=t))
                           for t in (0.25, 0.5, 1.0, 2.0)],
    "defect_rate": lambda b: [(f"rate={r}", dataclasses.replace(b, defect_rate=r))
                              for r in (0.05, 0.15, 0.3)],
    "error_conc": lambda b: [(f"conc={c}", dataclasses.replace(b, error_conc=c))
                             for c in (0.2, 1.0, 5.0)],
    "arms": lambda b: [(arm, dataclasses.replace(b, arm=arm)) for arm in ("base", "a3", "a5")]
    + [("within", dataclasses.replace(b, beta_within=0.5))],
    "regime": regime_points,
}


def run_sweep(name: str, base: SimConfig, params: V3Params,
              replicates: int) -> list[dict[str, Any]]:
    """Each point of one sweep, merged over `replicates` seeds (base.seed, base.seed + 1, …)."""
    out = []
    for label, config in SWEEPS[name](base):
        runs = [replicate(dataclasses.replace(config, seed=base.seed + i), params)[0]
                for i in range(replicates)]
        summary = merge(runs).summary()
        f1 = {m: {t: summary["methods"][m][t]["f1"] for t in TYPES} for m in METHODS}
        out.append({"point": label, "config": config_dict(config)} | summary
                   | {"versus": versus(f1)})
    return out


def config_dict(config: SimConfig) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, tuple) else v
            for k, v in dataclasses.asdict(config).items()}


def load_config(path: Path) -> SimConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = raw.get("config", raw)
    return SimConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def load_params(path: Path) -> V3Params:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return V3Params(**raw.get("params", raw))


def _f1s(point: dict[str, Any], method: str) -> str:
    return " / ".join(f"{point['methods'][method][t]['f1']:.2f}" for t in TYPES)


def sweep_markdown(sweeps: dict[str, list[dict[str, Any]]], params: V3Params,
                   replicates: Mapping[str, int]) -> str:
    lines = ["<!-- generated by specdx simulate -->", "", "# SpecDx v3.1 simulation study", "",
             f"Parameters: {dataclasses.asdict(params)}.",
             "F1 cells read hidden / ambiguity / weak. T4(a) is Theorem T4's bound "
             "4|V|·exp(−nΔ²/32) on the two-model contrast rule, with n the mean readings per "
             "model and cell and Δ the accuracy gap of the two most accurate models. p̄ is the "
             "mean accuracy at a cell of median difficulty; T2′ identifies a set A from one "
             "model unless p equals its chance line |A|/K (shown for |A| = 1 and 2, K the mean "
             "number of candidate values), and T4(c) is the minimum-distance rule's bound "
             "2K·exp(−nγ_min²/2) at p̄. The last column compares v3.1 with the best baseline "
             f"per type (W, T or L, margin {MARGIN}).", ""]
    for name, points in sweeps.items():
        n, least = replicates[name], min_replicates(name)
        label = (f" Exploratory: below §8.1's minimum of {least} replicates per point."
                 if n < least else "")
        lines += [f"## Sweep: {name}", "", f"Replicates per point: {n}.{label}", "",
                  "| point | n | Δ | T4(a) | p̄ | T2′ chance | T4(c) | v3 | majority | "
                  "anchor-only | v2 stage 2 | mean abs θ error | ECE | converged | "
                  "v3 vs best (H A W) |",
                  "|---|---:|---:|---:|---:|---|---:|---|---|---|---|---:|---:|---:|---|"]
        for p in points:
            chance = " / ".join(f"{c:.2f}" for c in p["chance"])
            lines.append(f"| {p['point']} | {p['n']:.1f} | {p['delta']:.3f} | {p['t4a']:.3f} | "
                         f"{p['accuracy']:.3f} | {chance} | {p['t4c']:.3f} | "
                         f"{_f1s(p, 'v3')} | {_f1s(p, 'majority')} | {_f1s(p, 'anchor_only')} | "
                         f"{_f1s(p, 'v2_stage2')} | {p['theta_error']:.3f} | {p['ece']:.3f} | "
                         f"{p['converged']:.2f} | {' '.join(p['versus'][t] for t in TYPES)} |")
        if name == "n":
            for t in TYPES:
                reached = [p["n"] for p in points if p["methods"]["v3"][t]["f1"] >= F1_TARGET]
                where = f"{min(reached):.1f}" if reached else "not reached"
                lines.append(f"\nReadings per model and cell for v3 {t} F1 ≥ {F1_TARGET}: "
                             f"{where}.")
        lines.append("")
    return "\n".join(lines) + "\n"
