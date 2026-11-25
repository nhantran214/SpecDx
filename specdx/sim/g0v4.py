"""G0v4 (P4a design §6.1): choose and freeze v4's free parameters by simulation.

The grid and the selection rule are fixed here, before G0v4 runs. The free parameters are
cell_sd and the λ prior (the PRIOR settings: they change the fit, so each is fitted once per
replicate and arm) and tau_u, tau_h and tau_a (the THRESHOLD settings: they only change
typing, so they are scored on the same fits). tau_w = 0.9 and tau_lambda = 0.5 are fixed;
π, the form shares, β's prior and the α prior are v3's frozen values.

Layouts (ruling F-C1). The simulator copies the S3 layout of each development domain
(SimConfig.goals: the check groups and one outcome group per user goal, that domain's raters
and difficulty spreads). Replicate r of a replicate set uses layout r mod (number of layouts),
and each simulated domain is fitted on its own, as the real fits are per domain.

Arms: base (the ordinary arm), shared, homogeneous and saturated. The criteria (§6.1) are
judged on the selection replicates (seeds base ... base + n - 1) and reported on the held-out
ones (base + n ... base + 2n - 1):
  1. the Wilson 95% upper bound of the hidden false-positive rate among determined shared
     cells on the shared arm, pooled over the replicates, is at most 0.1 (ruling G0-V; v3's
     rate is reported beside it; no determined shared cell: not evaluable, so it fails);
  2. on the ordinary arm, hidden F1 and ambiguity F1 on determined cells are each at least
     v3's minus 0.05 (the undetermined rate is reported). When v3's ambiguity F1 is below
     0.05 the ambiguity part is satisfied by construction, so it is reported as non-binding
     and neither passes nor fails the gate (ruling F-I2b);
  3. on the homogeneous and saturated arms, at least 80% of the planted defects and shared
     cells are undetermined (an arm with none planted is not evaluable, so it fails).
Scoring for the criteria: an undetermined cell is neither a prediction nor a miss. It is
excluded from TP, FP and FN, and is reported as an undetermined rate.
Selection (ruling F-I2a): among the candidates that pass every criterion, the best objective,
the coverage-aware F1 (every planted cell left undetermined counts as a false negative, so
abstaining is never free), averaged over the defect types and then over the arms with equal
weight; ties (to 1e-9) go to the smaller tau_u, then to grid order. If none passes the
verdict is FAIL, and the best candidate by the same rule is reported with the criteria it
fails. A fit that did not converge, or whose M-step hit its Newton cap, is reported, not
gated.

Running (ruling F-I8). Every fit runs in one process pool (forkserver, with the BLAS threads
pinned to 1 before numpy loads in a worker; ruling F-I7) and leaves a small typed summary in
a cache directory when one is given, keyed by the sha256 of the fit's configuration and of the
code that produces it, so a rerun after a crash skips what is done. The output does not depend
on the number of jobs or on the cache. The cost estimate times the first fit of each kind,
arm and layout (fits that the run keeps) and simulates the pool's two barriers.

Freezing (ruling F-I5). The search JSON records the replicates, the grid, the seeds and
whether the run was a smoke run. freeze_result refuses unless the verdict is PASS, the run is
not a smoke run, the replicates, grid and seeds are the pre-registered ones; it never
overwrites a frozen file that holds other parameters, and it records the git commit and the
search JSON's sha256 beside the parameters.

Usage (official runs happen on the 48 GB host; freeze only after a PASS):
  python -m specdx.sim.g0v4 run --config PATH --out results/p4/g0v4 [--jobs N] [--cache DIR]
  python -m specdx.sim.g0v4 freeze RESULT.json [--path experiments/p4-s3-v4-params.json]
Exit status of run: 0 PASS, 1 FAIL, 2 error; of freeze: 0 frozen, 1 refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import itertools
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from specdx.infer.anchored import V3Params, fit_anchored
from specdx.infer.anchored_v4 import V4Params, fit_v4
from specdx.infer.typing import CellFinding, type_cells
from specdx.infer.typing_v4 import UNDETERMINED, type_cells_v4
from specdx.sim.baselines import by_cell
from specdx.sim.generator import SimConfig, SimSpec, generate
from specdx.sim.metrics import TYPES, Counts, typed_counts

CELL_SD: tuple[float, ...] = (0.5, 1.0, 2.0)
LAMBDA_PRIOR: tuple[tuple[float, float], ...] = ((1.0, 3.0), (1.0, 1.0))
TAU_U: tuple[float, ...] = (1.0, 2.0, 3.0)
TAU_H: tuple[float, ...] = (0.5, 0.7, 0.9)
TAU_A: tuple[float, ...] = (0.3, 0.5, 0.7)
TAU_W = 0.9
TAU_LAMBDA = 0.5
ARMS: tuple[str, ...] = ("base", "shared", "homogeneous", "saturated")
REPLICATES = 10
DEFAULT_SEED = 0
SHARED_FP_MAX = 0.1
F1_MARGIN = 0.05
NON_BINDING_V3_F1 = 0.05  # ruling F-I2b: below this v3 F1, criterion 2's part is non-binding
UNDETERMINED_MIN = 0.8
PLANTED_ARMS: tuple[str, ...] = ("homogeneous", "saturated")
PASS, FAIL = "PASS", "FAIL"
NON_BINDING = "non-binding"
PARAMS_FILE = Path(__file__).resolve().parents[2] / "experiments" / "p4-s3-v4-params.json"
CACHE_DIR = "g0v4-cache"
SUMMARY_VERSION = "g0v4-summary-1"
BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_EPS = 1e-9
WILSON_Z = 1.96


def wilson_upper(k: int, n: int, z: float = WILSON_Z) -> float:
    """The upper end of the Wilson score interval for k successes in n trials (n >= 1)."""
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + spread) / (1 + z * z / n))


def shared_upper(hidden: int, determined: int) -> float | None:
    """Criterion 1's statistic: the Wilson upper bound of hidden among determined shared cells,
    None (not evaluable) when no shared cell is determined."""
    return wilson_upper(hidden, determined) if determined else None


# --- the grid ---

def full_grid() -> dict[str, list[Any]]:
    """The pre-registered grid, as the search JSON records it."""
    return _grid_json(None)


def _grid_json(grid: Mapping[str, Sequence[Any]] | None) -> dict[str, list[Any]]:
    g = grid or {}
    axes = {"cell_sd": CELL_SD, "lambda_prior": LAMBDA_PRIOR, "tau_u": TAU_U, "tau_h": TAU_H,
            "tau_a": TAU_A}
    return {k: [list(v) if isinstance(v, tuple) else v for v in g.get(k, default)]
            for k, default in axes.items()}


def prior_settings(grid: Mapping[str, Sequence[Any]] | None = None) -> list[V4Params]:
    """Every setting of the priors, in grid order. v3's frozen values fill every other field.
    The thresholds carried here are placeholders (each axis's first value): a fit does not
    read them, and every threshold setting is typed on the same fit."""
    g, v3 = grid or {}, V3Params()
    return [V4Params(pi=v3.pi, s_mis=v3.s_mis, s_amb=v3.s_amb, s_empty=v3.s_empty,
                     beta_sd=v3.beta_sd, mu_alpha=v3.mu_alpha, sigma_alpha=v3.sigma_alpha,
                     cell_sd=sd, lambda_a=a, lambda_b=b, tau_h=g.get("tau_h", TAU_H)[0],
                     tau_a=g.get("tau_a", TAU_A)[0], tau_w=TAU_W, tau_u=g.get("tau_u", TAU_U)[0],
                     tau_lambda=TAU_LAMBDA)
            for sd, (a, b) in itertools.product(g.get("cell_sd", CELL_SD),
                                                g.get("lambda_prior", LAMBDA_PRIOR))]


def thresholds(grid: Mapping[str, Sequence[Any]] | None = None
               ) -> list[tuple[float, float, float]]:
    """(tau_u, tau_h, tau_a) in grid order."""
    g = grid or {}
    return list(itertools.product(g.get("tau_u", TAU_U), g.get("tau_h", TAU_H),
                                  g.get("tau_a", TAU_A)))


# --- scoring ---

@dataclass
class Scored:
    """One specification's findings scored. counts: per defect type, on determined cells
    only (the criteria). coverage: per defect type over every cell, an undetermined planted
    cell a false negative (the selection objective, ruling F-I2a). undetermined: the cells
    typed undetermined. types: cell → the defect types predicted on it."""

    counts: dict[str, Counts]
    undetermined: set[str]
    types: dict[str, set[str]]
    coverage: dict[str, Counts] = field(default_factory=dict)


def score_findings(spec: SimSpec, found: Iterable[CellFinding]) -> Scored:
    """An undetermined cell is neither a prediction nor a miss for `counts`: it is dropped
    from the predictions and from the truth before TP, FP and FN are counted. For `coverage`
    only the predictions are dropped, so a planted undetermined cell is a miss."""
    found = list(found)
    undetermined = {f.cell for f in found if f.type == UNDETERMINED}
    types: dict[str, set[str]] = {}
    for f in found:
        if f.type in TYPES and f.cell not in undetermined:
            types.setdefault(f.cell, set()).add(f.type)
    predicted = [(c, t) for c, ts in types.items() for t in ts]
    truth = {c: t for c, t in spec.truth.items() if c not in undetermined}
    return Scored(typed_counts(predicted, truth), undetermined, types,
                  typed_counts(predicted, spec.truth))


def summarise(spec: SimSpec, scored: Scored) -> dict[str, Any]:
    """The numbers one arm's tally needs from one scored specification (JSON-safe)."""
    shared_determined = shared_hidden = planted = planted_undetermined = 0
    for cid, truth in spec.truth.items():
        if truth.shared and cid not in scored.undetermined:
            shared_determined += 1
            shared_hidden += "hidden" in scored.types.get(cid, ())
        if truth.hidden or truth.ambiguity or truth.weak or truth.shared:
            planted += 1
            planted_undetermined += cid in scored.undetermined
    return {"counts": {t: [c.tp, c.fp, c.fn] for t, c in scored.counts.items()},
            "coverage": {t: [c.tp, c.fp, c.fn] for t, c in scored.coverage.items()},
            "cells": len(spec.cells), "undetermined": len(scored.undetermined),
            "shared_determined": shared_determined, "shared_hidden": shared_hidden,
            "planted": planted, "planted_undetermined": planted_undetermined}


@dataclass
class ArmTally:
    """The counts of one arm over its replicates, for one parameter setting."""

    counts: dict[str, Counts] = field(default_factory=lambda: {t: Counts() for t in TYPES})
    coverage: dict[str, Counts] = field(default_factory=lambda: {t: Counts() for t in TYPES})
    cells: int = 0
    undetermined: int = 0
    shared_determined: int = 0
    shared_hidden: int = 0
    planted: int = 0
    planted_undetermined: int = 0

    def add(self, summary: Mapping[str, Any]) -> None:
        for name in ("counts", "coverage"):
            for t, (tp, fp, fn) in summary[name].items():
                getattr(self, name)[t].add(Counts(tp, fp, fn))
        for name in ("cells", "undetermined", "shared_determined", "shared_hidden", "planted",
                     "planted_undetermined"):
            setattr(self, name, getattr(self, name) + summary[name])

    @property
    def f1(self) -> dict[str, float]:
        return {t: self.counts[t].f1 for t in TYPES}

    @property
    def coverage_f1(self) -> dict[str, float]:
        return {t: self.coverage[t].f1 for t in TYPES}

    @property
    def shared_rate(self) -> float:
        return self.shared_hidden / self.shared_determined if self.shared_determined else 0.0

    @property
    def planted_rate(self) -> float:
        return self.planted_undetermined / self.planted if self.planted else 1.0

    @property
    def undetermined_rate(self) -> float:
        return self.undetermined / self.cells if self.cells else 0.0


def metrics_of(tallies: Mapping[str, ArmTally]) -> dict[str, Any]:
    """The numbers the criteria read, from the per-arm tallies."""
    base, shared = tallies["base"], tallies["shared"]
    return {"shared_fp": shared.shared_rate, "shared_determined": shared.shared_determined,
            "shared_hidden": shared.shared_hidden,
            "shared_fp_upper": shared_upper(shared.shared_hidden, shared.shared_determined),
            "hidden_f1": base.f1["hidden"], "ambiguity_f1": base.f1["ambiguity"],
            "undetermined": base.undetermined_rate,
            "planted_undetermined": {a: tallies[a].planted_rate for a in PLANTED_ARMS},
            "planted": {a: tallies[a].planted for a in PLANTED_ARMS}}


def judge(metrics: Mapping[str, Any], v3: Mapping[str, float]) -> dict[str, Any]:
    """The three criteria of §6.1 (inclusive bounds, to 1e-9; rulings G0-V and F-I2b). A
    criterion part is True (holds), False (fails) or None (non-binding: v3's ambiguity F1 is
    below 0.05, so the part holds by construction and neither passes nor fails the gate)."""
    upper = shared_upper(metrics["shared_hidden"], metrics["shared_determined"])
    ambiguity: bool | None = None
    if v3["ambiguity_f1"] >= NON_BINDING_V3_F1:
        ambiguity = metrics["ambiguity_f1"] >= v3["ambiguity_f1"] - F1_MARGIN - _EPS
    return {"1": upper is not None and upper <= SHARED_FP_MAX + _EPS,
            "2": {"hidden": metrics["hidden_f1"] >= v3["hidden_f1"] - F1_MARGIN - _EPS,
                  "ambiguity": ambiguity},
            "3": {a: metrics["planted"][a] > 0
                  and metrics["planted_undetermined"][a] >= UNDETERMINED_MIN - _EPS
                  for a in PLANTED_ARMS}}


def failed(criteria: Mapping[str, Any]) -> list[str]:
    """The names of the criteria that fail (2 and 3 by part, as `2.hidden`); a non-binding
    part (None) is not a failure."""
    out = []
    for key, value in criteria.items():
        if isinstance(value, dict):
            out += [f"{key}.{k}" for k, ok in value.items() if ok is False]
        elif not value:
            out.append(key)
    return out


def non_binding(criteria: Mapping[str, Any]) -> list[str]:
    """The criterion parts reported as non-binding (ruling F-I2b)."""
    return [f"{key}.{k}" for key, value in criteria.items() if isinstance(value, dict)
            for k, ok in value.items() if ok is None]


def verdict(criteria: Mapping[str, Any]) -> str:
    return PASS if not failed(criteria) else FAIL


@dataclass
class Candidate:
    """params: the prior setting with its thresholds. objective: the coverage-aware F1 (a
    planted undetermined cell is a miss), mean over the defect types, then over the arms with
    equal weight (ruling F-I2a). arms: typed F1 on determined cells per arm. passes: every
    binding criterion holds on the selection replicates. undetermined: the undetermined rate
    per arm (the ordinary arm's is reported per candidate)."""

    params: V4Params
    objective: float
    arms: dict[str, dict[str, float]]
    passes: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)
    undetermined: dict[str, float] = field(default_factory=dict)
    nonconverged: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    capped: dict[str, int] = field(default_factory=dict)


def choose(candidates: Sequence[Candidate]) -> Candidate:
    """Candidates that pass first; then the highest objective (ties within 1e-9); ties to the
    smaller tau_u (ruling F-I2a), then to the earlier in grid order."""
    pool = [i for i, c in enumerate(candidates) if c.passes] or list(range(len(candidates)))
    best = max(candidates[i].objective for i in pool)
    tied = [i for i in pool if candidates[i].objective >= best - _EPS]
    return candidates[min(tied, key=lambda i: (candidates[i].params.tau_u, i))]


def objective_of(tallies: Mapping[str, ArmTally]) -> float:
    return sum(sum(t.coverage_f1.values()) / len(TYPES) for t in tallies.values()) / len(tallies)


# --- fits, as tasks a process pool runs ---

@dataclass(frozen=True)
class FitTask:
    """One fit: kind "v4" (typed at every threshold setting given) or "v3"; config: the
    layout with its arm and seed; layout: the layout's index."""

    kind: str
    layout: int
    config: SimConfig
    prior: V4Params | None = None
    thresholds: tuple[tuple[float, float, float], ...] = ()


def run_task(task: FitTask) -> dict[str, Any]:
    """Generate the specification, fit it and summarise the typed findings (no spec or fit
    leaves the worker). seconds: the time this took."""
    started = time.monotonic()
    spec = generate(task.config)
    mine = by_cell(spec)
    if task.kind == "v4":
        assert task.prior is not None
        fit = fit_v4(spec.cells, spec.readings, models=spec.models, params=task.prior)
        out: dict[str, Any] = {"converged": fit.converged, "capped": fit.m_step_capped,
                               "thresholds": []}
        for tau_u, tau_h, tau_a in task.thresholds:
            params = dataclasses.replace(task.prior, tau_u=tau_u, tau_h=tau_h, tau_a=tau_a)
            found = type_cells_v4(spec.cells, fit, mine, params)
            out["thresholds"].append(summarise(spec, score_findings(spec, found)))
    else:
        v3 = V3Params()
        fit3 = fit_anchored(spec.cells, spec.readings, models=spec.models, params=v3)
        found = type_cells(spec.cells, fit3, mine, v3)
        hidden = {f.cell for f in found if f.type == "hidden"}
        shared = [c for c, t in spec.truth.items() if t.shared]
        counts = typed_counts(((f.cell, f.type) for f in found), spec.truth)
        out = {"counts": {t: [c.tp, c.fp, c.fn] for t, c in counts.items()},
               "shared": len(shared), "hit": sum(c in hidden for c in shared)}
    out["seconds"] = time.monotonic() - started
    return out


def _sources() -> list[Path]:
    from specdx.infer import anchored, anchored_v4, cells, typing, typing_v4
    from specdx.sim import baselines, generator, metrics
    return [Path(m.__file__) for m in (anchored, anchored_v4, cells, typing, typing_v4,
                                       baselines, generator, metrics)]


_CODE_VERSION: str | None = None


def code_version() -> str:
    """The sha256 of the code a fit's summary depends on, and of the summary format."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        h = hashlib.sha256(SUMMARY_VERSION.encode())
        for path in _sources():
            h.update(path.read_bytes())
        _CODE_VERSION = h.hexdigest()
    return _CODE_VERSION


def task_key(task: FitTask) -> str:
    """The cache key: sha256 of the task's configuration and of code_version()."""
    payload = {"kind": task.kind, "config": dataclasses.asdict(task.config),
               "prior": dataclasses.asdict(task.prior) if task.prior else None,
               "thresholds": [list(t) for t in task.thresholds], "code": code_version()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_read(cache: Path | None, task: FitTask) -> dict[str, Any] | None:
    if cache is None:
        return None
    path = cache / f"{task_key(task)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_write(cache: Path | None, task: FitTask, result: Mapping[str, Any]) -> None:
    if cache is None:
        return
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{task_key(task)}.json"
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, path)


def pin_blas_threads() -> None:
    """One BLAS/OpenMP thread per process, so N jobs do not oversubscribe the cores (ruling
    J-1). Set before the pool starts: its forkserver and workers read it when numpy loads."""
    for name in BLAS_THREADS:
        os.environ[name] = "1"


def open_pool(jobs: int) -> Executor | None:
    """A forkserver process pool of `jobs` workers (ruling F-I7: no fork of a threaded
    parent), or None for jobs <= 1 (everything runs in this process)."""
    if jobs <= 1:
        return None
    pin_blas_threads()
    return ProcessPoolExecutor(max_workers=jobs,
                               mp_context=multiprocessing.get_context("forkserver"))


def execute(tasks: Sequence[FitTask], pool: Executor | None, cache: Path | None = None,
            progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    """Every task's summary, in input order: from the cache when it is there, else run (in
    the pool when there is one) and cached as soon as it finishes."""
    results: list[dict[str, Any] | None] = [_cache_read(cache, t) for t in tasks]
    todo = [i for i, r in enumerate(results) if r is None]
    done = len(tasks) - len(todo)

    def finished(i: int, result: dict[str, Any]) -> None:
        nonlocal done
        results[i] = result
        _cache_write(cache, tasks[i], result)
        done += 1
        if progress is not None:
            progress(done, len(tasks))

    if pool is None:
        for i in todo:
            finished(i, run_task(tasks[i]))
    else:
        futures = {pool.submit(run_task, tasks[i]): i for i in todo}
        for future in as_completed(futures):
            finished(futures[future], future.result())
    return [r for r in results if r is not None]


# --- the search ---

def _layouts(config: SimConfig | Sequence[SimConfig]) -> tuple[SimConfig, ...]:
    layouts = (config,) if isinstance(config, SimConfig) else tuple(config)
    if not layouts:
        raise ValueError("G0v4 needs at least one layout")
    return layouts


def _spec_config(layouts: Sequence[SimConfig], arm: str, i: int, offset: int,
                 ) -> tuple[int, SimConfig]:
    """Replicate i of the set starting at `offset`: layout i mod L, seed layout seed +
    offset + i."""
    k = i % len(layouts)
    return k, dataclasses.replace(layouts[k], arm=arm, seed=layouts[k].seed + offset + i)


def seeds_of(layouts: Sequence[SimConfig], replicates: int) -> dict[str, list[int]]:
    return {"selection": [_spec_config(layouts, ARMS[0], i, 0)[1].seed
                          for i in range(replicates)],
            "held_out": [_spec_config(layouts, ARMS[0], i, replicates)[1].seed
                         for i in range(replicates)]}


def v4_tasks(layouts: Sequence[SimConfig], prior: V4Params, replicates: int, offset: int,
             cuts: Sequence[tuple[float, float, float]]) -> list[FitTask]:
    """One v4 fit per (arm, replicate), arms in ARMS order."""
    return [FitTask("v4", *_spec_config(layouts, arm, i, offset), prior, tuple(cuts))
            for arm in ARMS for i in range(replicates)]


def v3_tasks(layouts: Sequence[SimConfig], replicates: int, offset: int) -> list[FitTask]:
    """v3 on the base and shared arms (v3 never abstains)."""
    return [FitTask("v3", *_spec_config(layouts, arm, i, offset))
            for arm in ("base", "shared") for i in range(replicates)]


def v3_reference(results: Sequence[Mapping[str, Any]], replicates: int) -> dict[str, Any]:
    """v3's numbers from v3_tasks' summaries: hidden and ambiguity F1 on the ordinary arm,
    and the share of the shared arm's shared cells it types hidden."""
    counts = {t: Counts() for t in TYPES}
    for r in results[:replicates]:
        for t, (tp, fp, fn) in r["counts"].items():
            counts[t].add(Counts(tp, fp, fn))
    total = sum(r["shared"] for r in results[replicates:])
    hit = sum(r["hit"] for r in results[replicates:])
    return {"hidden_f1": counts["hidden"].f1, "ambiguity_f1": counts["ambiguity"].f1,
            "weak_f1": counts["weak"].f1, "shared_hidden": hit / total if total else 0.0,
            "shared_cells": total}


def _tallies(results: Sequence[Mapping[str, Any]], replicates: int, cut: int,
             ) -> dict[str, ArmTally]:
    tallies = {arm: ArmTally() for arm in ARMS}
    for n, r in enumerate(results):
        tallies[ARMS[n // replicates]].add(r["thresholds"][cut])
    return tallies


def _fit_flags(results: Sequence[Mapping[str, Any]], replicates: int, key: str,
               ) -> dict[str, int]:
    return {arm: sum(int(not r["converged"]) if key == "converged" else int(r[key] > 0)
                     for r in results[n * replicates:(n + 1) * replicates])
            for n, arm in enumerate(ARMS)}


def candidates_of(priors: Sequence[V4Params], cuts: Sequence[tuple[float, float, float]],
                  results: Sequence[Sequence[Mapping[str, Any]]], v3: Mapping[str, Any],
                  replicates: int) -> list[Candidate]:
    """One candidate per (prior, threshold) in grid order, from each prior's v4 summaries."""
    out: list[Candidate] = []
    for prior, fits in zip(priors, results, strict=True):
        nonconverged = _fit_flags(fits, replicates, "converged")
        capped = _fit_flags(fits, replicates, "capped")
        for cut, (tau_u, tau_h, tau_a) in enumerate(cuts):
            params = dataclasses.replace(prior, tau_u=tau_u, tau_h=tau_h, tau_a=tau_a)
            tallies = _tallies(fits, replicates, cut)
            metrics = metrics_of(tallies)
            criteria = judge(metrics, v3)
            out.append(Candidate(
                params, objective_of(tallies), {a: t.f1 for a, t in tallies.items()},
                not failed(criteria), metrics, criteria,
                {a: t.undetermined_rate for a, t in tallies.items()}, dict(nonconverged),
                {a: t.coverage_f1 for a, t in tallies.items()}, dict(capped)))
    return out


def _makespan(durations: Iterable[float], jobs: int) -> float:
    """The pool's wall time for these tasks in submission order (each goes to the first free
    worker), ending at a barrier."""
    free = [0.0] * max(1, jobs)
    for d in durations:
        heapq.heapreplace(free, free[0] + d)
    return max(free)


def estimate_cost(calibration: Sequence[tuple[FitTask, Mapping[str, Any]]], wall: float,
                  rest: Sequence[FitTask], held: Sequence[FitTask], jobs: int,
                  cached: int) -> dict[str, Any]:
    """Extrapolate the run from the calibration fits (the first fit of each kind, arm and
    layout in use, run first and kept: fit times differ by arm as much as by layout): their
    wall time, then the pool's two barriers, the rest of the selection fits and the held-out
    fits of the chosen prior, each simulated in submission order over `jobs` workers (fewer
    tasks than jobs leave workers idle). A fit without a timing of its own takes the slowest
    of its kind. Fits that do not converge, or other prior settings, can still run longer."""
    times: dict[str, float] = {}
    for task, result in calibration:
        times[_timing_key(task)] = float(result["seconds"])

    def cost(task: FitTask) -> float:
        slowest = max((v for k, v in times.items() if k.startswith(task.kind + "/")),
                      default=0.0)
        return times.get(_timing_key(task), slowest)

    phase1 = _makespan((cost(t) for t in rest), jobs)
    phase2 = _makespan((cost(t) for t in held), jobs)
    n4 = sum(t.kind == "v4" for t in (*rest, *held)) + sum(t.kind == "v4" for t, _ in calibration)
    n3 = sum(t.kind == "v3" for t in rest) + sum(t.kind == "v3" for t, _ in calibration)
    return {"v4_fit_s": {k[3:]: v for k, v in sorted(times.items()) if k.startswith("v4/")},
            "v3_fit_s": {k[3:]: v for k, v in sorted(times.items()) if k.startswith("v3/")},
            "calibration_s": wall, "selection_s": phase1, "held_out_s": phase2,
            "v4_fits": n4, "v3_fits": n3, "cached": cached, "jobs": jobs,
            "total_s": wall + phase1 + phase2}


def _timing_key(task: FitTask) -> str:
    return f"{task.kind}/{task.config.arm}/layout {task.layout}"


def _calibration_split(tasks: Sequence[FitTask], used: int) -> tuple[list[int], list[int]]:
    """The indices of the calibration tasks (the first fit of each kind, arm and layout in
    use) and of the rest, both in input order."""
    first: set[int] = set()
    seen: set[str] = set()
    for i, t in enumerate(tasks):
        key = _timing_key(t)
        if key not in seen and t.layout < used:
            seen.add(key)
            first.add(i)
    return sorted(first), [i for i in range(len(tasks)) if i not in first]


def search(layouts: SimConfig | Sequence[SimConfig], replicates: int = REPLICATES,
           grid: Mapping[str, Sequence[Any]] | None = None, jobs: int = 1,
           cache: Path | None = None, pool: Executor | None = None,
           log: Callable[[str], None] | None = None,
           ) -> tuple[Candidate, list[Candidate], dict[str, Any], dict[str, Any]]:
    """Fit every prior setting once per replicate and arm, type at every threshold setting
    on the same fits, judge the criteria, choose, and fit the chosen prior on the held-out
    replicates. Returns the chosen candidate, all of them, v3's reference numbers and the
    held-out report (with the cost estimate under "estimate")."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    say = log or (lambda text: None)
    layouts = _layouts(layouts)
    priors, cuts = prior_settings(grid), thresholds(grid)
    selection = [t for p in priors for t in v4_tasks(layouts, p, replicates, 0, cuts)]
    references = v3_tasks(layouts, replicates, 0) + v3_tasks(layouts, replicates, replicates)
    tasks = selection + references
    cached = sum(_cache_read(cache, t) is not None for t in tasks)
    first, rest = _calibration_split(tasks, min(replicates, len(layouts)))
    started = time.monotonic()
    calibration = execute([tasks[i] for i in first], pool, cache)
    wall = time.monotonic() - started
    held_guess = v4_tasks(layouts, priors[0], replicates, replicates, cuts[:1])
    estimate = estimate_cost(list(zip([tasks[i] for i in first], calibration, strict=True)),
                             wall, [tasks[i] for i in rest
                                    if _cache_read(cache, tasks[i]) is None],
                             held_guess, jobs, cached)
    say(f"G0v4: estimated {estimate['total_s']:.0f} s ({estimate['total_s'] / 3600:.2f} h) "
        f"with {jobs} job(s); {cached} of {len(tasks)} fits cached")
    results: list[dict[str, Any]] = [{}] * len(tasks)
    for i, r in zip(first, calibration, strict=True):
        results[i] = r
    done = execute([tasks[i] for i in rest], pool, cache,
                   progress=lambda n, total: say(f"G0v4: {n}/{total} fits"))
    for i, r in zip(rest, done, strict=True):
        results[i] = r
    per = len(ARMS) * replicates
    v4_results = [results[n * per:(n + 1) * per] for n in range(len(priors))]
    v3_results = results[len(selection):]
    v3 = v3_reference(v3_results[:2 * replicates], replicates)
    candidates = candidates_of(priors, cuts, v4_results, v3, replicates)
    chosen = choose(candidates)
    held = v4_tasks(layouts, chosen.params, replicates, replicates,
                    [(chosen.params.tau_u, chosen.params.tau_h, chosen.params.tau_a)])
    hold = execute(held, pool, cache,
                   progress=lambda n, total: say(f"G0v4: held-out {n}/{total} fits"))
    tallies = _tallies(hold, replicates, 0)
    v3_held = v3_reference(v3_results[2 * replicates:], replicates)
    metrics = metrics_of(tallies)
    report = {"metrics": metrics, "v3": v3_held, "criteria": judge(metrics, v3_held),
              "arms": {a: t.f1 for a, t in tallies.items()},
              "coverage": {a: t.coverage_f1 for a, t in tallies.items()},
              "undetermined": {a: t.undetermined_rate for a, t in tallies.items()},
              "nonconverged": _fit_flags(hold, replicates, "converged"),
              "capped": _fit_flags(hold, replicates, "capped"), "estimate": estimate}
    return chosen, candidates, v3, report


# --- the report ---

def _yes(ok: bool | None) -> str:
    return "non-binding (v3 F1 < 0.05)" if ok is None else ("yes" if ok else "no")


def _criteria_lines(metrics: Mapping[str, Any], v3: Mapping[str, Any],
                    criteria: Mapping[str, Any]) -> list[str]:
    pu, planted = metrics["planted_undetermined"], metrics["planted"]
    upper = metrics.get("shared_fp_upper")
    bound = "not evaluable" if upper is None else f"Wilson upper {upper:.3f}"
    lines = [
        "| criterion | value | bar | holds |", "|---|---:|---:|---|",
        f"| 1. hidden false-positive rate on shared cells, determined "
        f"(n = {metrics['shared_determined']}) | {metrics['shared_fp']:.3f}, {bound} "
        f"(v3: {v3['shared_hidden']:.3f}) | upper bound <= {SHARED_FP_MAX} "
        f"| {_yes(criteria['1'])} |",
        f"| 2. ordinary hidden F1 | {metrics['hidden_f1']:.3f} (v3: {v3['hidden_f1']:.3f}) "
        f"| >= v3 - {F1_MARGIN} | {_yes(criteria['2']['hidden'])} |",
        f"| 2. ordinary ambiguity F1 | {metrics['ambiguity_f1']:.3f} "
        f"(v3: {v3['ambiguity_f1']:.3f}) | >= v3 - {F1_MARGIN} "
        f"| {_yes(criteria['2']['ambiguity'])} |",
        f"| 3. homogeneous: planted and shared cells undetermined "
        f"(n = {planted['homogeneous']}) | {pu['homogeneous']:.3f} "
        f"| >= {UNDETERMINED_MIN} | {_yes(criteria['3']['homogeneous'])} |",
        f"| 3. saturated: planted and shared cells undetermined "
        f"(n = {planted['saturated']}) | {pu['saturated']:.3f} "
        f"| >= {UNDETERMINED_MIN} | {_yes(criteria['3']['saturated'])} |",
        "", f"Undetermined rate on the ordinary arm: {metrics['undetermined']:.3f}."]
    if criteria["2"]["ambiguity"] is None:
        lines += ["", f"Criterion 2.ambiguity is non-binding: v3's ambiguity F1 is below "
                      f"{NON_BINDING_V3_F1}, so the bar (v3 - {F1_MARGIN}) is met by "
                      "construction. It neither passes nor fails the gate and is no "
                      "evidence for v4 (ruling F-I2b)."]
    return lines


def _fmt_seconds(est: Mapping[str, Any], key: str) -> str:
    return ", ".join(f"{k}: {v:.1f} s" for k, v in est[key].items()) or "none"


def render(result: Mapping[str, Any]) -> str:
    chosen, hold = result["chosen"], result["held_out"]
    lines = ["<!-- generated by python -m specdx.sim.g0v4 -->", "",
             f"# G0v4: {result['verdict']}", "",
             f"Replicates per arm: {result['replicates']} (selection) and "
             f"{result['replicates']} (held out), over {len(result.get('layouts', [1]))} "
             f"layout(s). Grid: {result['settings']} prior settings x "
             f"{result['thresholds']} threshold settings. Runtime: {result['seconds']:.0f} s.",
             "", "## Chosen parameters" + ("" if result["verdict"] == PASS else
                                           " (best candidate; no candidate passes)"), "",
             "| parameter | value |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in chosen["params"].items()]
    lines += ["", f"Objective (coverage-aware F1, mean over types and arms; an undetermined "
                  f"planted cell is a miss): {chosen['objective']:.4f}."]
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
    if failed(hold["criteria"]):
        lines += ["", "WARNING: the chosen candidate fails on the held-out replicates: "
                  + ", ".join(failed(hold["criteria"])) + "."]
    est = result.get("estimate")
    if est:
        lines += ["", "## Cost estimate (from the first fits, before the rest of the run)", "",
                  f"One v4 fit: {_fmt_seconds(est, 'v4_fit_s')}; one v3 fit: "
                  f"{_fmt_seconds(est, 'v3_fit_s')}. {est['v4_fits']} v4 and "
                  f"{est['v3_fits']} v3 fits ({est['cached']} cached), {est['jobs']} job(s): "
                  f"calibration {est['calibration_s']:.0f} s, selection "
                  f"{est['selection_s']:.0f} s and held out {est['held_out_s']:.0f} s, "
                  f"estimated {est['total_s']:.0f} s ({est['total_s'] / 3600:.2f} h); "
                  f"measured {result['seconds']:.0f} s."]
    lines += ["", "## Fits that did not converge or hit the Newton cap (selection, chosen "
                  "prior setting; reported, not gated)", "",
              "Not converged: " + (", ".join(f"{a}: {n}" for a, n in
                                              chosen["nonconverged"].items()) or "none") + ".",
              "M-step at its Newton cap: " + (", ".join(f"{a}: {n}" for a, n in
                                                      chosen.get("capped", {}).items())
                                              or "none") + "."]
    if any(chosen["nonconverged"].values()):
        lines += ["", "WARNING: some fits of the chosen prior setting did not converge."]
    lines += ["", "## Every candidate", "",
              "| cell_sd | lambda prior | tau_u | tau_h | tau_a | objective | passes "
              "| ordinary undetermined |", "|---:|---|---:|---:|---:|---:|---|---:|"]
    for c in result["candidates"]:
        p = c["params"]
        lines.append(f"| {p['cell_sd']} | ({p['lambda_a']}, {p['lambda_b']}) | {p['tau_u']} "
                     f"| {p['tau_h']} | {p['tau_a']} | {c['objective']:.4f} "
                     f"| {_yes(c['passes'])} | {c['undetermined']['base']:.3f} |")
    return "\n".join(lines) + "\n"


def params_dict(p: V4Params) -> dict[str, float]:
    return dataclasses.asdict(p)


def run_g0v4(config: SimConfig | Sequence[SimConfig], out: Path, *,
             replicates: int = REPLICATES, grid: Mapping[str, Sequence[Any]] | None = None,
             jobs: int = 1, cache: Path | None = None, smoke: bool = False,
             names: Sequence[str] | None = None) -> int:
    """Run G0v4 on the layouts (one SimConfig or one per development domain; names: their
    domains, for the report) and write `out`.json and `out`.md. Returns 0 PASS, 1 FAIL, 2
    error. It never freezes: freeze_result is called on a PASS by the caller. jobs: processes
    for the fits, all in one pool; cache: the fit cache directory (None: no cache). The
    output does not depend on jobs or on the cache (apart from the timing fields)."""
    try:
        started = time.monotonic()
        layouts = _layouts(config)
        with _PoolScope(jobs) as pool:
            chosen, candidates, v3, hold = search(
                layouts, replicates, grid, jobs, cache, pool,
                log=lambda text: print(text, file=sys.stderr))
        estimate = hold.pop("estimate")
        result = {
            "verdict": PASS if chosen.passes else FAIL, "replicates": replicates,
            "grid": _grid_json(grid), "seeds": seeds_of(layouts, replicates), "smoke": smoke,
            "settings": len(prior_settings(grid)), "thresholds": len(thresholds(grid)),
            "seconds": time.monotonic() - started, "estimate": estimate, "jobs": jobs,
            "layouts": [dataclasses.asdict(c) for c in layouts],
            "layout_names": list(names) if names else None, "code": code_version(),
            "chosen": {"params": params_dict(chosen.params), "objective": chosen.objective,
                       "arms": chosen.arms, "coverage": chosen.coverage,
                       "metrics": chosen.metrics, "undetermined": chosen.undetermined,
                       "nonconverged": chosen.nonconverged, "capped": chosen.capped},
            "criteria": chosen.criteria, "failed": failed(chosen.criteria),
            "non_binding": non_binding(chosen.criteria), "v3": v3, "held_out": hold,
            "candidates": [{"params": params_dict(x.params), "objective": x.objective,
                            "arms": x.arms, "coverage": x.coverage, "passes": x.passes,
                            "metrics": x.metrics, "criteria": x.criteria,
                            "undetermined": x.undetermined, "nonconverged": x.nonconverged,
                            "capped": x.capped} for x in candidates]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(render(result), encoding="utf-8")
        return 0 if chosen.passes else 1
    except Exception:  # noqa: BLE001 - exit 2 is the stage's error contract; the cache keeps work
        traceback.print_exc()
        return 2


class _PoolScope:
    """open_pool as a context manager that shuts the pool down however the run ends."""

    def __init__(self, jobs: int) -> None:
        self.pool = open_pool(jobs)

    def __enter__(self) -> Executor | None:
        return self.pool

    def __exit__(self, *exc: object) -> None:
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)


# --- freezing ---

def freeze_problems(result: Mapping[str, Any]) -> list[str]:
    """Why a search result may not be frozen (ruling F-I5); empty when it may."""
    problems = []
    if result.get("verdict") != PASS:
        problems.append("the verdict is not PASS")
    if result.get("smoke") is not False:
        problems.append("it is a smoke run, or records no smoke marker")
    if result.get("replicates") != REPLICATES:
        problems.append(f"replicates {result.get('replicates')} is not the pre-registered "
                        f"{REPLICATES}")
    if result.get("grid") != full_grid():
        problems.append("the grid is not the pre-registered one")
    layouts = result.get("layouts") or []
    want = {"selection": list(range(DEFAULT_SEED, DEFAULT_SEED + REPLICATES)),
            "held_out": list(range(DEFAULT_SEED + REPLICATES, DEFAULT_SEED + 2 * REPLICATES))}
    if result.get("seeds") != want or any(c.get("seed") != DEFAULT_SEED for c in layouts):
        problems.append("the seeds are not the default ones")
    return problems


def _git_head() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=PARAMS_FILE.parents[1], check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, cwd=PARAMS_FILE.parents[1], check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty.strip() else "")


def freeze_result(result_path: Path, path: Path = PARAMS_FILE,
                  head: Callable[[], str] = _git_head) -> tuple[bool, str]:
    """Freeze the chosen parameters of a search JSON to `path`, with provenance (the commit,
    from `head`, and the search JSON's sha256). Refuses (False, why) on any freeze_problems, and
    never overwrites a frozen file that holds other parameters; one that holds the same
    parameters is left as it is."""
    raw = Path(result_path).read_bytes()
    result = json.loads(raw)
    problems = freeze_problems(result)
    if problems:
        return False, "refusing to freeze: " + "; ".join(problems)
    params = params_dict(V4Params(**result["chosen"]["params"]))
    path = Path(path)
    if path.exists():
        held = json.loads(path.read_text(encoding="utf-8")).get("params")
        if held != params:
            return False, f"refusing to overwrite {path}: it holds other parameters"
        return True, f"{path} already holds these parameters; left as it is"
    payload = {"params": params,
               "provenance": {"commit": head(), "search": str(result_path),
                              "search_sha256": hashlib.sha256(raw).hexdigest()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True, f"frozen to {path}"


def load_layouts(path: Path) -> tuple[SimConfig, ...]:
    """A config file: one SimConfig (its fields, or under "config") or {"layouts": [...]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["layouts"] if "layouts" in raw else [raw.get("config", raw)]
    return tuple(SimConfig(**{k: tuple(v) if isinstance(v, list) else v
                              for k, v in item.items()}) for item in items)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="specdx.sim.g0v4")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--replicates", type=int, default=REPLICATES)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--cache", type=Path, default=None)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("result", type=Path)
    freeze.add_argument("--path", type=Path, default=PARAMS_FILE)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        code = run_g0v4(load_layouts(args.config), args.out, replicates=args.replicates,
                        jobs=args.jobs, cache=args.cache)
        if code == 0:
            print("PASS: freeze with `python -m specdx.sim.g0v4 freeze`", file=sys.stderr)
        return code
    ok, message = freeze_result(args.result, args.path)
    print(message, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
