"""From behaviours and the act→read posterior to typed, located defects (spec v2 §3.4–§3.6).

Only observable items carry evidence (stage 1). A candidate reading any clause that no
observable item reads is Unobservable: it is reported, never typed and never scored.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from specdx.infer.hitting import soft_hitting_sets
from specdx.infer.items import ItemView, items_reading
from specdx.infer.latent import ItemPosterior, Posterior
from specdx.infer.sensitivity import (
    BH_Q,
    COMPETENT,
    MIN_BEHAVIOURS,
    PERMUTATIONS,
    benjamini_hochberg,
    paraphrase_sensitivity,
)
from specdx.spec.columns import Projection
from specdx.types import Behaviour, Defect, Refuter

Diff = Callable[[Projection, Projection], frozenset[str]]
TAU_H = 0.5  # spec v2 §3.10: a hidden finding needs Hidden(c) ≥ τ_H
# The kind of an Unobservable entry: not a defect type, never in DEFECT_TYPES.
UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class ItemSignal:
    """An observable item's ambiguity signals: S1 = P(|A_s| ≥ 2); S2's statistic and
    p-value (None when the item was not eligible); whether S2 is significant after BH."""

    s1: float
    s2: float | None
    p: float | None
    significant: bool


@dataclass
class Diagnosis:
    defects: list[Defect]
    unobservable: list[Defect] = field(default_factory=list)
    signals: dict[str, ItemSignal] = field(default_factory=dict)


def run_weights(views: Iterable[ItemView], posterior: Posterior) -> dict[str, float]:
    """Spec v2 §3.3: an acted behaviour weighs P([b] ∈ A_s) × θ_m. A run's weight is the
    mean over the items where it acted; a run that acted nowhere weighs 0."""
    acc: dict[str, list[float]] = defaultdict(list)
    for v in views:
        acc.setdefault(v.run_key, [])
        if not v.acted:
            continue
        item = posterior.items.get(v.item)
        member = item.p_member.get(v.klass, 0.0) if item is not None else 0.0
        acc[v.run_key].append(member * posterior.theta.get(v.model, 0.0))
    return {k: (sum(w) / len(w) if w else 0.0) for k, w in acc.items()}


def ambiguity_key(d: Defect) -> tuple[str, str, tuple[str, ...]]:
    """An ambiguity's identity before localisation: its step and the fields its admissible
    classes differ on (its note is "scenario|step"). Localisation replaces the fields with
    sentence ids that vary between runs, so differentials compare on this key (§11)."""
    return (d.type, d.note.split("|", 1)[-1], d.location)


def identity(d: Defect) -> tuple:
    """What makes two findings one: type and location, and the step for an ambiguity
    that is not localised yet."""
    return ambiguity_key(d) if d.type == "ambiguity" else d.key


# The note of a singleton typed hidden only because the probe budget ran out before any
# refutation attempt: its type is a default, not evidence (§11).
UNATTEMPTED = "refutation not attempted: probe budget exhausted"


def merge_defects(
    defects: Iterable[Defect], key: Callable[[Defect], Hashable] = lambda d: d.key
) -> list[Defect]:
    merged: dict[Hashable, Defect] = {}
    for d in defects:
        k = key(d)
        old = merged.get(k)
        if old is None:
            merged[k] = d
            continue
        best = d if d.score > old.score else old
        note = best.note
        if UNATTEMPTED in (old.note, d.note) and old.note != d.note:
            # A refutation attempted in any scenario is evidence; keep its note (§11).
            note = d.note if old.note == UNATTEMPTED else old.note
        merged[k] = Defect(
            d.type, d.location,
            tuple(sorted(set(old.scenarios) | set(d.scenarios))),
            tuple(sorted(set(old.witnesses) | set(d.witnesses))),
            best.score, note,
        )
    return sorted(merged.values(), key=lambda d: (-d.score, d.type, d.location, d.note))


def _scenario_of(item: str) -> str:
    return item.split("|", 1)[0]


def _hidden_score(
    clause: str, items: Iterable[str], posterior: Posterior,
    by_item: Mapping[str, list[ItemView]],
) -> float | None:
    """Spec §3.4: the posterior mass of admissible sets holding a class whose only
    step-relevant violation is this clause, maximised over the scenario's items."""
    scores = []
    for item in items:
        ip = posterior.items.get(item)
        views = by_item.get(item, [])
        if ip is None or not any(clause in v.violated for v in views):
            continue
        hits = {v.klass for v in views if v.violated == frozenset({clause})}
        scores.append(sum(p for s, p in ip.set_posterior.items() if s & hits))
    return max(scores) if scores else None


def _observable(
    clause: str, scenario_id: str, observable: frozenset[str],
    relevant: Mapping[str, frozenset[str]],
) -> bool:
    return any(item in observable for item in items_reading(clause, scenario_id, relevant))


def _weak(ip: ItemPosterior, views: Sequence[ItemView], min_runs: int) -> tuple[str, float] | None:
    failing = {v.klass for v in views if v.violated}
    map_set = ip.map_set
    if failing & map_set:
        return None  # an admissible output itself fails a check: that is not a weak check
    counts = Counter(v.klass for v in views)
    eligible = sorted(
        k for k in set(counts) - failing
        if counts[k] >= min_runs and k not in map_set
    )
    if not eligible:
        return None

    def outside(k: str) -> float:
        if k not in ip.classes:
            return 1.0
        return sum(p for s, p in ip.set_posterior.items() if k not in s)

    if any(k not in ip.classes for k in eligible):
        score = 1.0
    else:
        score = sum(p for s, p in ip.set_posterior.items() if any(k not in s for k in eligible))
    chosen = sorted(eligible, key=lambda k: (-outside(k), -counts[k], k))[0]
    return chosen, score


def _reference(ip: ItemPosterior) -> str:
    """The clean admissible class a weak deviant is compared with: the MAP set's most
    probable member (_weak has checked that the whole MAP set is clean)."""
    return max(sorted(ip.map_set), key=lambda k: ip.p_member.get(k, 0.0))


def diagnose(
    behaviours: Sequence[Behaviour],
    violated: Mapping[str, frozenset[str]],
    views: Sequence[ItemView],
    posterior: Posterior,
    refuter: Refuter,
    diff: Diff,
    *,
    observable: frozenset[str],
    relevant: Mapping[str, frozenset[str]],
    tau: float = 0.05,
    ambig_threshold: float = 0.5,
    weak_threshold: float = 0.5,
    weak_min_runs: int = 2,
    hidden_threshold: float = TAU_H,
    competent: float = COMPETENT,
    permutations: int = PERMUTATIONS,
    s2_min_behaviours: int = MIN_BEHAVIOURS,
    bh_q: float = BH_Q,
    rng_seed: int = 0,
    known: frozenset[tuple[str, ...]] = frozenset(),
) -> Diagnosis:
    """violated: each run's V(b), already restricted to the items where it acted (§3.4).
    observable: the observable items (stage 1). relevant: step id → the clauses relevant
    to it. known: locations the base specification's report already holds. On a mutant
    they are refuted last, so base noise does not spend the probe budget meant for the
    mutant's own candidates."""
    defects: list[Defect] = []
    unobservable: list[Defect] = []
    weights = run_weights(views, posterior)
    by_item: dict[str, list[ItemView]] = defaultdict(list)
    for v in views:
        if v.acted:
            by_item[v.item].append(v)
    groups: dict[str, list[Behaviour]] = defaultdict(list)
    for b in behaviours:
        if b.status == "ok":
            groups[b.scenario.id].append(b)
    # Every scenario's candidates, refuted in descending score order, locations the base
    # report does not hold first: the probe budget goes to the best supported new
    # candidates first (§11).
    candidates: list[tuple[float, str, frozenset[str]]] = []
    for sid, group in groups.items():
        sets = [(violated[b.run_key], weights.get(b.run_key, 0.0)) for b in group]
        candidates += [(support, sid, hitting)
                       for hitting, support in soft_hitting_sets(sets, tau=tau)]
    candidates.sort(key=lambda c: (tuple(sorted(c[2])) in known, -c[0], c[1], len(c[2]),
                                   sorted(c[2])))
    for support, sid, hitting in candidates:
        group = groups[sid]
        witnesses = tuple(b.run_key for b in group if violated[b.run_key] & hitting)
        location = tuple(sorted(hitting))
        if not all(_observable(c, sid, observable, relevant) for c in hitting):
            # No refutation either: nothing the population does can tell its type (§3.9).
            unobservable.append(Defect(UNOBSERVABLE, location, (sid,), witnesses, support,
                                       "no observable item reads every clause"))
            continue
        result = refuter.refute(hitting, group[0].scenario, group)
        if not result.refuted and result.attempted:
            defects.append(Defect("contradiction", location, (sid,), witnesses, support,
                                  f"unrefuted after {result.attempts} attempts"))
        elif len(hitting) == 1:
            # Refuted, or never attempted because the probe budget ran out: a
            # contradiction needs a failed refutation attempt (§11). Hidden(c) must reach
            # τ_H: a candidate no admissible class supports is no hidden requirement.
            clause = next(iter(hitting))
            scenario_items = sorted(i for i in by_item if _scenario_of(i) == sid)
            score = _hidden_score(clause, scenario_items, posterior, by_item) or 0.0
            if score >= hidden_threshold:
                note = f"refuted by {result.witness}" if result.refuted else UNATTEMPTED
                defects.append(Defect("hidden", location, (sid,), witnesses, score, note))
        # A refuted set is dropped; so is an unattempted one, which is no contradiction.
    tested = paraphrase_sensitivity(
        views, posterior.theta, [i for i in posterior.items if i in observable],
        competent=competent, permutations=permutations, min_behaviours=s2_min_behaviours,
        seed=rng_seed,
    )
    significant = benjamini_hochberg({i: p for i, (_, p) in tested.items()}, bh_q)
    signals: dict[str, ItemSignal] = {}
    for item, ip in sorted(posterior.items.items()):
        if item not in observable:
            continue
        stat, p = tested.get(item, (None, None))
        signals[item] = ItemSignal(ip.p_ambiguous, stat, p, item in significant)
        mine = by_item.get(item, [])
        projection = {v.klass: v.projection for v in mine}
        ranked = sorted(ip.p_member, key=lambda k: -ip.p_member[k])
        if (ip.p_ambiguous >= ambig_threshold or item in significant) and len(ranked) >= 2:
            members = sorted(ip.map_set) if len(ip.map_set) >= 2 else ranked[:2]
            fields = frozenset().union(
                *(diff(projection[a], projection[b]) for a, b in combinations(members, 2))
            )
            if fields:
                witnesses = tuple(v.run_key for v in mine if v.klass in members)
                score = max(ip.p_ambiguous, 1 - p if p is not None else 0.0)
                defects.append(Defect("ambiguity", tuple(sorted(fields)), (_scenario_of(item),),
                                      witnesses, score, note=item))
        weak = _weak(ip, mine, weak_min_runs)
        if weak is not None and weak[1] >= weak_threshold:
            klass, score = weak
            fields = diff(projection[klass], projection[_reference(ip)])
            if fields:
                witnesses = tuple(v.run_key for v in mine if v.klass == klass)
                defects.append(Defect("weak", tuple(sorted(fields)), (_scenario_of(item),),
                                      witnesses, score, note=item))
    return Diagnosis(merge_defects(defects, key=identity), merge_defects(unobservable), signals)
