"""The SpecDx main loop (spec §3.8)."""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from specdx.active.budget import ProbeBudget
from specdx.active.search import CellBandit, information_gain, should_escalate
from specdx.infer.items import ItemView
from specdx.infer.latent import Observation, Posterior, fit
from specdx.infer.scores import ambiguity_key, diagnose, identity, merge_defects
from specdx.interpreters.population import committee, pilot
from specdx.spec.columns import Projection
from specdx.types import Behaviour, Defect, InterpreterId, Refuter, RefuteResult, Scenario
from specdx.variant import SpecVariant


class RunnerLike(Protocol):
    executed: int

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = "",
        *, kind: str = "population",
    ) -> Behaviour: ...


class LensLike(Protocol):
    def violations(self, behaviour: Behaviour) -> frozenset[str]: ...

    def views(self, behaviour: Behaviour, violated: frozenset[str]) -> list[ItemView]: ...

    def diff(self, a: Projection, b: Projection) -> frozenset[str]: ...


# The localised sentence ids, or None when the ambiguity stays unlocalised.
Localizer = Callable[
    [SpecVariant, Defect, Sequence[InterpreterId], ProbeBudget], tuple[str, ...] | None
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
    # Directed-probe and ddmin runs requested per specification, cached or not (§11).
    probe_budget: int = 100


@dataclass
class LoopContext:
    runner: RunnerLike
    lens: LensLike
    canon: dict[str, Scenario]
    candidates: list[Scenario]
    cap_pct: int
    refuter_factory: Callable[[Sequence[InterpreterId], ProbeBudget], Refuter]
    localizer: Localizer | None = None


@dataclass
class Report:
    """runs counts population runs; probe_runs and ddmin_runs the requested runs that the
    probe budget charged. ambiguity_keys are the ambiguities found, before localisation."""

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

    def to_json(self) -> str:
        return json.dumps(
            {"variant": self.variant, "defects": [asdict(d) for d in self.defects],
             "runs": self.runs, "probe_runs": self.probe_runs, "ddmin_runs": self.ddmin_runs,
             "executed": self.executed, "scenarios": self.scenarios, "theta": self.theta},
            ensure_ascii=False, indent=2,
        )


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
    """Every model once, most competent first: who refutes and who localises."""
    models = [(m, f) for f, ms in models_by_family.items() for m in ms]
    models.sort(key=lambda mf: -theta.get(mf[0], 0.0))
    return [InterpreterId(m, f, variants[0], 1) for m, f in models]


def run_specdx(
    variant: SpecVariant, ctx: LoopContext, config: SpecDxConfig, *, base: Report | None = None
) -> Report:
    """base: the base specification's report. Ambiguities it already holds (same step and
    fields) are dropped before localisation, so only the mutant's own are localised (§11)."""
    rng = random.Random(config.rng_seed)
    budget = ProbeBudget(config.probe_budget)
    refutations: dict = {}
    bandit = CellBandit(ctx.cap_pct)
    queue = [ctx.canon[name] for name in sorted(ctx.canon)]
    visited: set[tuple[int, ...]] = set()
    behaviours: list[Behaviour] = []
    violated: dict[str, frozenset[str]] = {}
    views: list[ItemView] = []
    seen_sets: set[frozenset[str]] = set()
    posterior = Posterior({}, {}, {})
    defects: list[Defect] = []
    previous: dict[str, float] = {}
    order: list[str] = []
    runs = 0
    executed_before = ctx.runner.executed
    members = committee(config.models_by_family, config.variants, config.seeds)

    while runs < config.budget:
        if queue:
            scenario = queue.pop(0)
        elif config.synthesize and not bandit.exhausted(ctx.candidates, visited, config.eps):
            chosen = bandit.choose(ctx.candidates, visited, rng)
            if chosen is None:
                break
            scenario = chosen
        else:
            break
        visited.add(scenario.signature())
        order.append(scenario.id)

        pilot_ids = pilot(config.models_by_family, posterior.theta or None)
        fresh: list[Behaviour] = []
        for interp in pilot_ids:
            if runs >= config.budget:
                break
            fresh.append(ctx.runner.run(variant, scenario, interp))
            runs += 1
        sets = [ctx.lens.violations(b) for b in fresh]
        pilot_views = [v for b, s in zip(fresh, sets, strict=True) for v in ctx.lens.views(b, s)]
        ok_sets = [s for b, s in zip(fresh, sets, strict=True) if b.status == "ok"]
        if should_escalate(pilot_views, ok_sets, seen_sets):
            for interp in members:
                if interp in pilot_ids:
                    continue
                if runs >= config.budget:
                    break
                fresh.append(ctx.runner.run(variant, scenario, interp))
                runs += 1

        for b in fresh:
            v = ctx.lens.violations(b)
            violated[b.run_key] = v
            if b.status == "ok":
                seen_sets.add(v)
            behaviours.append(b)
            views.extend(ctx.lens.views(b, v))
        posterior = fit(
            [Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views],
            k_max=config.k_max, p_single=config.p_single,
        )
        refuter = _Memoised(ctx.refuter_factory(
            ranked_interpreters(config.models_by_family, posterior.theta, config.variants),
            budget,
        ), refutations)
        defects = diagnose(
            behaviours, violated, views, posterior, refuter, ctx.lens.diff,
            tau=config.tau, ambig_threshold=config.ambig_threshold,
            weak_threshold=config.weak_threshold,
        )
        scores = {repr(identity(d)): d.score for d in defects}
        bandit.update(scenario, information_gain(previous, scores))
        previous = scores

    found = frozenset(ambiguity_key(d) for d in defects if d.type == "ambiguity")
    if base is not None:
        defects = [d for d in defects
                   if d.type != "ambiguity" or ambiguity_key(d) not in base.ambiguity_keys]
    if config.localize and ctx.localizer is not None:
        interps = ranked_interpreters(config.models_by_family, posterior.theta, config.variants)
        localised: list[Defect] = []
        for d in defects:
            if d.type == "ambiguity":
                # Unlocalised: an empty location, which matches no ground truth.
                sentences = ctx.localizer(variant, d, interps, budget)
                d = dataclasses.replace(d, location=tuple(sentences or ()))
            localised.append(d)
        defects = merge_defects(localised)

    return Report(variant.name, defects, runs, ctx.runner.executed - executed_before, order,
                  dict(posterior.theta), behaviours, probe_runs=budget.probe,
                  ddmin_runs=budget.ddmin, ambiguity_keys=found)
