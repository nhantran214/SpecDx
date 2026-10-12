"""The SpecDx main loop (spec §3.8)."""

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
from specdx.infer.act import Act, Rasch, fit_rasch
from specdx.infer.items import ItemView, acted_violations, observations
from specdx.infer.latent import Posterior, fit
from specdx.infer.scores import (
    Diagnosis,
    ambiguity_key,
    diagnose,
    identity,
    merge_defects,
)
from specdx.infer.sensitivity import COMPETENT
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


# The localised sentence ids, or None when the ambiguity stays unlocalised.
Localizer = Callable[
    [SpecVariant, Defect, Sequence[InterpreterId], RunBudget], tuple[str, ...] | None
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
    ambig_threshold: float = 0.5
    weak_threshold: float = 0.5
    k_max: int = 6
    p_single: float = 0.9
    rng_seed: int = 20261006
    synthesize: bool = True
    localize: bool = True
    localize_tests: int = 40
    # Runs requested per specification, cached or not (§11): directed probes for
    # refutation, and ddmin runs for localisation, each under its own cap.
    probe_budget: int = 100
    localize_budget: int = 80


@dataclass
class LoopContext:
    """tboundary: the T-boundary scenarios, queued after the canon ones; literals: the T
    literals they come from, reported per specification (spec v2 §3.7)."""

    runner: RunnerLike
    lens: LensLike
    canon: dict[str, Scenario]
    candidates: list[Scenario]
    cap_pct: int
    refuter_factory: Callable[[Sequence[InterpreterId], RunBudget], Refuter]
    localizer: Localizer | None = None
    tboundary: list[Scenario] = field(default_factory=list)
    literals: list[TLiteral] = field(default_factory=list)


@dataclass
class Report:
    """runs counts population runs; probe_runs and ddmin_runs the requested runs charged to
    the probe and localisation budgets. ambiguity_keys are the ambiguities found, before
    the differential and localisation; ambiguity_dropped those the differential removed.
    The v2 additions (spec v2 §3.8): κ per model; per item δ, O_s, whether it is
    observable, and its ambiguity signals S1, S2, p; the Unobservable list; the T literals.
    family_effect is the model-family effect on the logit scale (spec v2 §3.3)."""

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
    # Singleton locations that spent probes and were dropped below τ_H (Diagnosis.spent).
    spent: frozenset[tuple[str, ...]] = frozenset()
    kappa: dict[str, float] = field(default_factory=dict)
    family_effect: dict[str, float] = field(default_factory=dict)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    unobservable: list[Defect] = field(default_factory=list)
    literals: list[TLiteral] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"variant": self.variant, "defects": [asdict(d) for d in self.defects],
             "unobservable": [asdict(d) for d in self.unobservable],
             "runs": self.runs, "probe_runs": self.probe_runs, "ddmin_runs": self.ddmin_runs,
             "executed": self.executed, "scenarios": self.scenarios, "theta": self.theta,
             "kappa": self.kappa, "family_effect": self.family_effect, "items": self.items,
             "literals": [x.as_dict() for x in self.literals],
             "spent": [list(loc) for loc in sorted(self.spent)]},
            ensure_ascii=False, indent=2,
        )


def item_table(rasch: Rasch, diagnosis: Diagnosis) -> dict[str, dict[str, Any]]:
    """Per item: stage 1 (δ, O_s, acted behaviours, observable) and, when observable, the
    ambiguity signals (S1, S2 and its p-value, BH significance)."""
    out: dict[str, dict[str, Any]] = {}
    for item in sorted(rasch.delta):
        row: dict[str, Any] = {"delta": rasch.delta[item],
                               "observability": rasch.observability[item],
                               "acted": rasch.acted[item], "observable": rasch.observable(item)}
        signal = diagnosis.signals.get(item)
        if signal is not None:
            row |= {"s1": signal.s1, "s2": signal.s2, "p": signal.p,
                    "significant": signal.significant}
        out[item] = row
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
    competent: float = COMPETENT,
) -> list[InterpreterId]:
    """The ddmin oracle's re-runs (spec v2 §3.6): one per competent model and Tell variant,
    at seed 1, most competent first. Competent is θ ≥ 0.5 with an acted behaviour: a
    prior-only θ (no acted behaviour) is not evidence of competence."""
    models = [(m, f) for f, ms in models_by_family.items() for m in ms
              if m in acted and theta.get(m, 0.0) >= competent]
    models.sort(key=lambda mf: -theta.get(mf[0], 0.0))
    return [InterpreterId(m, f, v, 1) for m, f in models for v in variants]


def run_specdx(
    variant: SpecVariant, ctx: LoopContext, config: SpecDxConfig, *, base: Report | None = None
) -> Report:
    """base: the base specification's report. Ambiguities it already holds (same step and
    fields) are dropped before localisation, so only the mutant's own are localised, and
    candidates at locations it already holds are refuted last (§11)."""
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
    posterior = Posterior({}, {}, {})
    rasch = fit_rasch([])
    diagnosis = Diagnosis([])
    defects: list[Defect] = []
    spent: frozenset[tuple[str, ...]] = frozenset()
    previous: dict[str, float] = {}
    order: list[str] = []
    runs = 0
    executed_before = ctx.runner.executed
    members = committee(config.models_by_family, config.variants, config.seeds)
    population = {m: f for f, ms in config.models_by_family.items() for m in ms}
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
        pilot_ids = pilot(config.models_by_family, posterior.theta or None)
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
        posterior = fit(observations(views), models=population, k_max=config.k_max,
                        p_single=config.p_single)
        refuter = _Memoised(ctx.refuter_factory(
            ranked_interpreters(config.models_by_family, posterior.theta, config.variants),
            budget,
        ), refutations)
        diagnosis = diagnose(
            behaviours, violated, views, posterior, refuter, ctx.lens.diff,
            observable=rasch.observable_items(), relevant=ctx.lens.relevant,
            tau=config.tau, ambig_threshold=config.ambig_threshold,
            weak_threshold=config.weak_threshold, rng_seed=config.rng_seed, known=known,
        )
        defects, spent = diagnosis.defects, diagnosis.spent
        scores = {repr(identity(d)): d.score for d in defects}
        bandit.update(scenario, information_gain(previous, scores))
        previous = scores

    found = frozenset(ambiguity_key(d) for d in defects if d.type == "ambiguity")
    dropped = found & base.ambiguity_keys if base is not None else frozenset()
    defects = [d for d in defects if d.type != "ambiguity" or ambiguity_key(d) not in dropped]
    if config.localize and ctx.localizer is not None:
        interps = oracle_interpreters(config.models_by_family, posterior.theta, config.variants,
                                      {v.model for v in views if v.acted})
        # Best supported first, until the localisation budget runs out; from then on every
        # ambiguity is unlocalised: an empty location, which matches no ground truth.
        located: dict[int, tuple[str, ...]] = {}
        for i in sorted((i for i, d in enumerate(defects) if d.type == "ambiguity"),
                        key=lambda i: -defects[i].score):
            sentences = (None if budget.exhausted("ddmin")
                         else ctx.localizer(variant, defects[i], interps, budget))
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
                  dict(posterior.theta), behaviours, probe_runs=budget.probe,
                  ddmin_runs=budget.ddmin, ambiguity_keys=found, ambiguity_dropped=dropped,
                  spent=spent, kappa=dict(rasch.kappa),
                  family_effect=dict(posterior.family_effect), items=item_table(rasch, diagnosis),
                  unobservable=diagnosis.unobservable, literals=list(ctx.literals))
