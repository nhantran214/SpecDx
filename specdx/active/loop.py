"""The SpecDx main loop (spec §3.8; v2 §3.8; v3 §5–§6).

Per iteration: pilot, escalation, stage 1 (act), then the anchored cell model on the
cells read so far, then the v3 diagnosis. Escalation still compares the acted views' class
keys (v2 §3.8); the evidence about the specification comes from cells only.
"""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from specdx.active.budget import RunBudget
from specdx.active.refute import SYMBOLIC_WITNESS
from specdx.active.search import CellBandit, information_gain, should_escalate
from specdx.cost import behaviours_cost
from specdx.infer.act import Act, Rasch, fit_rasch
from specdx.infer.anchored import AnchoredFit, V3Params, fit_anchored, posterior_at
from specdx.infer.cells import CellData
from specdx.infer.items import ItemView, acted_violations
from specdx.infer.scores import ambiguity_key, identity, merge_defects
from specdx.infer.typing import CellDiagnosis, diagnose_v3
from specdx.interpreters.population import committee, pilot
from specdx.observe.run import RunRequest
from specdx.scenarios.tguided import TLiteral
from specdx.spec.columns import Projection
from specdx.types import Behaviour, Defect, InterpreterId, Refuter, RefuteResult, Scenario
from specdx.variant import SpecVariant


class RunnerLike(Protocol):
    executed: int

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = "",
        *, kind: str = "population",
    ) -> Behaviour: ...

    def run_batch(self, requests: Sequence[RunRequest]) -> list[Behaviour]: ...


class LensLike(Protocol):
    relevant: Mapping[str, frozenset[str]]

    def violations(self, behaviour: Behaviour) -> frozenset[str]: ...

    def views(self, behaviour: Behaviour, violated: frozenset[str]) -> list[ItemView]: ...

    def acts(self, behaviour: Behaviour) -> list[Act]: ...

    def diff(self, a: Projection, b: Projection) -> frozenset[str]: ...


class CellsLike(Protocol):
    def extract(self, behaviours: Sequence[Behaviour]) -> CellData: ...


# The ddmin oracle's statistical test on one test's re-runs (spec v3 §5.3).
Persists = Callable[[Sequence[Behaviour]], bool]
# The localised sentence ids, or None when the ambiguity stays unlocalised.
Localizer = Callable[
    [SpecVariant, Defect, Sequence[InterpreterId], RunBudget, Persists], tuple[str, ...] | None
]
AmbiguityKey = tuple[str, str, tuple[str, ...]]


@dataclass(frozen=True)
class SpecDxConfig:
    models_by_family: dict[str, tuple[str, ...]]
    variants: tuple[int, ...] = (0,)
    seeds: tuple[int, ...] = (1, 2, 3)
    budget: int = 300
    tau: float = 0.05
    eps: float = 0.01
    k_max: int = 6
    rng_seed: int = 20261006
    synthesize: bool = True
    localize: bool = True
    localize_tests: int = 40
    # Runs requested per specification, cached or not (§11): directed probes for
    # refutation, and ddmin runs for localisation, each under its own cap.
    probe_budget: int = 100
    localize_budget: int = 80
    # The anchored model's priors and thresholds (spec v3 §9).
    params: V3Params = V3Params()
    # Anytime checkpoints (baselines spec §4.4, §9.2): population-run counts at which the
    # diagnosis so far is recorded. Output only: never read by any decision.
    checkpoints: tuple[int, ...] = ()


@dataclass
class LoopContext:
    """tboundary: the T-boundary scenarios, queued after the canon ones; literals: the T
    literals they come from, reported per specification (spec v2 §3.7). cells: the cell
    lens the v3 diagnosis reads (infer.cells.CellLens); the loop refuses to run without."""

    runner: RunnerLike
    lens: LensLike
    canon: dict[str, Scenario]
    candidates: list[Scenario]
    cap_pct: int
    refuter_factory: Callable[[Sequence[InterpreterId], RunBudget], Refuter]
    localizer: Localizer | None = None
    tboundary: list[Scenario] = field(default_factory=list)
    literals: list[TLiteral] = field(default_factory=list)
    cells: CellsLike | None = None


@dataclass
class Report:
    """runs counts population runs; probe_runs and ddmin_runs the requested runs charged to
    the probe and localisation budgets. ambiguity_keys are the ambiguities found, before
    the differential and localisation; ambiguity_dropped those the differential removed.
    The v2 additions (spec v2 §3.8): κ per model; per item δ, O_s and whether it is
    observable; the Unobservable list; the T literals. The v3 additions (spec v3 §3.5,
    §5.5): θ_m (theta) and α_m (alpha) of the anchored model, the family effect on the
    logit scale, the per-cell posterior summaries, and whether EM converged (converged,
    iterations, the winning start) — non-convergence is reported, never silent."""

    variant: str
    defects: list[Defect]
    runs: int
    executed: int
    scenarios: list[str]
    theta: dict[str, float]
    behaviours: list[Behaviour] = field(default_factory=list, repr=False)
    probe_runs: int = 0
    ddmin_runs: int = 0
    ambiguity_keys: frozenset[AmbiguityKey] = frozenset()
    ambiguity_dropped: frozenset[AmbiguityKey] = frozenset()  # base noise, removed
    # Multi-clause candidate locations that cost a probe run, or would have, and found
    # nothing (resolution 8; CellDiagnosis.spent).
    spent: frozenset[tuple[str, ...]] = frozenset()
    kappa: dict[str, float] = field(default_factory=dict)
    family_effect: dict[str, float] = field(default_factory=dict)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    unobservable: list[Defect] = field(default_factory=list)
    literals: list[TLiteral] = field(default_factory=list)
    alpha: dict[str, float] = field(default_factory=dict)
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0
    fit_start: str = ""
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        out = {"variant": self.variant, "defects": [asdict(d) for d in self.defects],
             "unobservable": [asdict(d) for d in self.unobservable],
             "runs": self.runs, "probe_runs": self.probe_runs, "ddmin_runs": self.ddmin_runs,
             "executed": self.executed, "scenarios": self.scenarios, "theta": self.theta,
             "alpha": self.alpha, "kappa": self.kappa, "family_effect": self.family_effect,
             "converged": self.converged, "iterations": self.iterations,
             "fit_start": self.fit_start, "items": self.items, "cells": self.cells,
             "literals": [x.as_dict() for x in self.literals],
             "spent": [list(loc) for loc in sorted(self.spent)]}
        if self.checkpoints:
            out["checkpoints"] = self.checkpoints
        return json.dumps(out, ensure_ascii=False, indent=2)


def item_table(rasch: Rasch) -> dict[str, dict[str, Any]]:
    """Per item, stage 1: δ, O_s, acted behaviours, observable."""
    return {item: {"delta": rasch.delta[item], "observability": rasch.observability[item],
                   "acted": rasch.acted[item], "observable": rasch.observable(item)}
            for item in sorted(rasch.delta)}


def cell_table(data: CellData, fit: AnchoredFit) -> dict[str, dict[str, Any]]:
    """Per cell (spec v3.1 §3.5, §5.5): its anchor, the readings per model, β_{g(c)}, the
    posterior over A_c (sets: at most 22 [sorted set, p] pairs, most probable first), the
    form masses, the MAP set and its posterior."""
    counts: dict[str, dict[str, int]] = {}
    for r in data.readings:
        mine = counts.setdefault(r.cell, {})
        mine[r.model] = mine.get(r.model, 0) + 1
    out: dict[str, dict[str, Any]] = {}
    for cid, post in sorted(fit.cells.items()):
        best = post.map_set
        sets = sorted(([sorted(s), p] for s, p in post.sets.items()),
                      key=lambda x: (-x[1], x[0]))
        out[cid] = {"anchor": post.anchor, "readings": dict(sorted(counts[cid].items())),
                    "beta": post.beta, "sets": sets, "forms": post.forms, "map": sorted(best),
                    "map_posterior": post.sets[best]}
    return out


class _Memoised:
    """One refutation per scenario and target set for the whole specification: the loop
    builds a fresh refuter every iteration, and repeated requests would drain the budget."""

    def __init__(self, refuter: Refuter, memo: dict) -> None:
        self.refuter, self.memo = refuter, memo

    def refute(
        self, targets: frozenset[str], scenario: Scenario, behaviours: Sequence[Behaviour]
    ) -> RefuteResult:
        key = (scenario.signature(), targets)
        if key not in self.memo:
            self.memo[key] = self.refuter.refute(targets, scenario, behaviours)
        result: RefuteResult = self.memo[key]
        return result


def ranked_interpreters(
    models_by_family: dict[str, tuple[str, ...]],
    theta: dict[str, float],
    variants: tuple[int, ...],
) -> list[InterpreterId]:
    """Every model once, most competent first: the order the refuter tries them in."""
    models = [(m, f) for f, ms in models_by_family.items() for m in ms]
    models.sort(key=lambda mf: -theta.get(mf[0], 0.0))
    return [InterpreterId(m, f, variants[0], 1) for m, f in models]


def oracle_interpreters(
    models_by_family: dict[str, tuple[str, ...]],
    theta: dict[str, float],
    variants: tuple[int, ...],
    acted: Collection[str],
) -> list[InterpreterId]:
    """The ddmin oracle's re-runs (spec v3 §5.3): one per model with at least one acted
    behaviour in the specification and per Tell variant, at seed 1, highest θ_m first. No θ
    threshold: a model that never acted cannot yield a reading, whatever its θ."""
    models = [(m, f) for f, ms in models_by_family.items() for m in ms if m in acted]
    models.sort(key=lambda mf: -theta.get(mf[0], 0.0))
    return [InterpreterId(m, f, v, 1) for m, f in models for v in variants]


def run_specdx(
    variant: SpecVariant, ctx: LoopContext, config: SpecDxConfig, *, base: Report | None = None
) -> Report:
    """base: the base specification's report. Ambiguities it already holds (same step and
    fields) are dropped before localisation, so only the mutant's own are localised, and
    candidates at locations it already holds are refuted last (§11)."""
    if ctx.cells is None:
        raise ValueError("the v3 loop reads cells: the loop context has no cell lens")
    cells = ctx.cells
    rng = random.Random(config.rng_seed)
    budget = RunBudget(config.probe_budget, config.localize_budget)
    refutations: dict = {}
    bandit = CellBandit(ctx.cap_pct)
    # Canon scenarios, then T-boundary scenarios, then the bandit (spec v2 §3.8).
    queue = [ctx.canon[name] for name in sorted(ctx.canon)] + list(ctx.tboundary)
    visited: set[tuple[int, ...]] = set()
    behaviours: list[Behaviour] = []
    violated: dict[str, frozenset[str]] = {}
    views: list[ItemView] = []
    acts: list[Act] = []
    seen_sets: set[frozenset[str]] = set()
    population = {m: f for f, ms in config.models_by_family.items() for m in ms}
    data = CellData({}, [])
    model = fit_anchored({}, [], models=population, params=config.params)
    rasch = fit_rasch([])
    diagnosis = CellDiagnosis([])
    defects: list[Defect] = []
    spent: frozenset[tuple[str, ...]] = frozenset()
    previous: dict[str, float] = {}
    order: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    runs = 0
    executed_before = ctx.runner.executed
    members = committee(config.models_by_family, config.variants, config.seeds)
    # Base locations whose candidates spent probes, or would: contradictions, and hidden
    # findings refuted by a probe run or never attempted. A base finding that symbolic
    # repair refuted cost no probe; if it needs one on a mutant, the mutant changed it.
    known = (frozenset(d.location for d in base.defects
                       if d.type == "contradiction"
                       or (d.type == "hidden" and d.note != f"refuted by {SYMBOLIC_WITNESS}"))
             | base.spent if base is not None else frozenset())

    while runs < config.budget:
        if queue:
            scenario = queue.pop(0)
            if scenario.signature() in visited:
                continue
        elif config.synthesize and not bandit.exhausted(ctx.candidates, visited, config.eps):
            chosen = bandit.choose(ctx.candidates, visited, rng)
            if chosen is None:
                break
            scenario = chosen
        else:
            break
        visited.add(scenario.signature())
        order.append(scenario.id)

        # Each batch is charged to the budget before it is dispatched (spec v2 §5.1).
        pilot_ids = pilot(config.models_by_family, model.theta if model.cells else None)
        batch = pilot_ids[:config.budget - runs]
        runs += len(batch)
        fresh = ctx.runner.run_batch([RunRequest(variant, scenario, i) for i in batch])
        seen = {b.run_key: ctx.lens.views(b, ctx.lens.violations(b)) for b in fresh}
        pilot_views = [v for b in fresh for v in seen[b.run_key]]
        ok_sets = [acted_violations(seen[b.run_key]) for b in fresh if b.status == "ok"]
        if should_escalate(pilot_views, ok_sets, seen_sets):
            batch = [i for i in members if i not in pilot_ids][:config.budget - runs]
            runs += len(batch)
            fresh += ctx.runner.run_batch([RunRequest(variant, scenario, i) for i in batch])

        for b in fresh:
            mine = ctx.lens.views(b, ctx.lens.violations(b))
            violated[b.run_key] = acted_violations(mine)
            if b.status == "ok":
                seen_sets.add(violated[b.run_key])
            behaviours.append(b)
            views.extend(mine)
            acts.extend(ctx.lens.acts(b))
        rasch = fit_rasch(acts)
        data = cells.extract(behaviours)
        model = fit_anchored(data.cells, data.readings, models=population,
                             params=config.params)
        refuter = _Memoised(ctx.refuter_factory(
            ranked_interpreters(config.models_by_family, model.theta, config.variants),
            budget,
        ), refutations)
        diagnosis = diagnose_v3(
            behaviours, violated, data, model, refuter,
            observable=rasch.observable_items(), relevant=ctx.lens.relevant,
            tau=config.tau, known=known,
        )
        defects, spent = diagnosis.defects, diagnosis.spent
        for at in config.checkpoints:
            if runs >= at and all(c["at"] != at for c in checkpoints):
                kept = [d for d in defects if d.type != "ambiguity" or base is None
                        or ambiguity_key(d) not in base.ambiguity_keys]
                checkpoints.append({"at": at, "runs": runs,
                                    "cost": behaviours_cost(behaviours).as_dict(),
                                    "defects": [asdict(d) for d in kept]})
        scores = {repr(identity(d)): d.score for d in defects}
        bandit.update(scenario, information_gain(previous, scores))
        previous = scores

    found = frozenset(ambiguity_key(d) for d in defects if d.type == "ambiguity")
    dropped = found & base.ambiguity_keys if base is not None else frozenset()
    defects = [d for d in defects if d.type != "ambiguity" or ambiguity_key(d) not in dropped]
    if config.localize and ctx.localizer is not None:
        interps = oracle_interpreters(config.models_by_family, model.theta, config.variants,
                                      {v.model for v in views if v.acted})
        # Best supported first, until the localisation budget runs out; from then on every
        # ambiguity is unlocalised: an empty location, which matches no ground truth.
        located: dict[int, tuple[str, ...]] = {}
        for i in sorted((i for i, d in enumerate(defects) if d.type == "ambiguity"),
                        key=lambda i: -defects[i].score):
            cell = diagnosis.ambiguity_cells[ambiguity_key(defects[i])]
            persists = _persists(cells, data, model, cell)
            # The oracle passes only while P(|A_c| = 2) ≥ τ_A, so an ambiguity reported
            # below it (F1 co-occurrence, direction B) could never be localised.
            localisable = model.cells[cell].p_pair >= config.params.tau_a
            sentences = (None if budget.exhausted("ddmin") or not localisable
                         else ctx.localizer(variant, defects[i], interps, budget, persists))
            located[i] = tuple(sentences or ())
        # Localised defects that now share type and location merge; unlocalised ones keep
        # their identity from before localisation (step and fields), so they stay apart.
        merge_key: dict[int, Hashable] = {}
        out: list[Defect] = []
        for i, d in enumerate(defects):
            new = dataclasses.replace(d, location=located[i]) if i in located else d
            merge_key[id(new)] = ambiguity_key(d) if i in located and not located[i] else new.key
            out.append(new)
        defects = merge_defects(out, key=lambda d: merge_key[id(d)])

    return Report(variant.name, defects, runs, ctx.runner.executed - executed_before, order,
                  dict(model.theta), behaviours, probe_runs=budget.probe,
                  ddmin_runs=budget.ddmin, ambiguity_keys=found, ambiguity_dropped=dropped,
                  spent=spent, kappa=dict(rasch.kappa),
                  family_effect=dict(model.family_effect), items=item_table(rasch),
                  unobservable=diagnosis.unobservable, literals=list(ctx.literals),
                  alpha=dict(model.alpha), cells=cell_table(data, model),
                  converged=model.converged, iterations=model.iterations,
                  fit_start=model.start, checkpoints=checkpoints)


def _persists(cells: CellsLike, data: CellData, model: AnchoredFit, cell: str) -> Persists:
    """Spec v3 §5.3: a ddmin test passes while P(|A_c| = 2), computed on the re-run
    readings of the ambiguity's cell with α, β and ε held at their fitted values, stays at
    or above τ_A."""
    def persists(runs: Sequence[Behaviour]) -> bool:
        readings = [r for r in cells.extract(runs).readings if r.cell == cell]
        post = posterior_at(model, data.cells[cell], readings)
        return post.p_pair >= model.params.tau_a

    return persists
