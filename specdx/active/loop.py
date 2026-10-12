"""The SpecDx main loop (spec §3.8)."""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from specdx.active.budget import RunBudget
from specdx.active.refute import SYMBOLIC_WITNESS
from specdx.active.search import CellBandit, information_gain, should_escalate
from specdx.infer.act import Act, fit_rasch
from specdx.infer.items import ItemView, acted_violations, observations
from specdx.infer.latent import Posterior, fit
from specdx.infer.scores import ambiguity_key, diagnose, identity, merge_defects
from specdx.interpreters.population import committee, pilot
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
    the differential and localisation; ambiguity_dropped those the differential removed."""

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
    fields) are dropped before localisation, so only the mutant's own are localised, and
    candidates at locations it already holds are refuted last (§11)."""
    rng = random.Random(config.rng_seed)
    budget = RunBudget(config.probe_budget, config.localize_budget)
    refutations: dict = {}
    bandit = CellBandit(ctx.cap_pct)
    queue = [ctx.canon[name] for name in sorted(ctx.canon)]
    visited: set[tuple[int, ...]] = set()
    behaviours: list[Behaviour] = []
    violated: dict[str, frozenset[str]] = {}
    views: list[ItemView] = []
    acts: list[Act] = []
    seen_sets: set[frozenset[str]] = set()
    posterior = Posterior({}, {}, {})
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
        seen = {b.run_key: ctx.lens.views(b, ctx.lens.violations(b)) for b in fresh}
        pilot_views = [v for b in fresh for v in seen[b.run_key]]
        ok_sets = [acted_violations(seen[b.run_key]) for b in fresh if b.status == "ok"]
        if should_escalate(pilot_views, ok_sets, seen_sets):
            for interp in members:
                if interp in pilot_ids:
                    continue
                if runs >= config.budget:
                    break
                fresh.append(ctx.runner.run(variant, scenario, interp))
                runs += 1

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
        interps = ranked_interpreters(config.models_by_family, posterior.theta, config.variants)
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
                  spent=spent)
