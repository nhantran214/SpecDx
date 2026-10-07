"""From behaviours and a posterior to typed, located defects (spec §3.3–§3.4)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import combinations

from specdx.infer.hitting import soft_hitting_sets
from specdx.infer.items import ItemView
from specdx.infer.latent import ItemPosterior, Posterior
from specdx.spec.columns import Projection
from specdx.types import Behaviour, Defect, Refuter

Diff = Callable[[Projection, Projection], frozenset[str]]


def run_weights(views: Iterable[ItemView], posterior: Posterior) -> dict[str, float]:
    """How faithful each run looks, averaged over its items."""
    acc: dict[str, list[float]] = defaultdict(list)
    for v in views:
        item = posterior.items.get(v.item)
        if item is not None:
            acc[v.run_key].append(item.responsibility.get(v.run_key, 0.0))
    return {k: sum(r) / len(r) for k, r in acc.items()}


def ambiguity_key(d: Defect) -> tuple[str, str, tuple[str, ...]]:
    """An ambiguity's identity before localisation: its step and the fields its admissible
    classes differ on (its note is "scenario|step"). Localisation replaces the fields with
    sentence ids that vary between runs, so differentials compare on this key (§11)."""
    return (d.type, d.note.split("|", 1)[-1], d.location)


def identity(d: Defect) -> tuple:
    """What makes two findings one: type and location, and the step for an ambiguity
    that is not localised yet."""
    return ambiguity_key(d) if d.type == "ambiguity" else d.key


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
        merged[k] = Defect(
            d.type, d.location,
            tuple(sorted(set(old.scenarios) | set(d.scenarios))),
            tuple(sorted(set(old.witnesses) | set(d.witnesses))),
            best.score, best.note,
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


def _weak(ip: ItemPosterior, views: Sequence[ItemView], min_runs: int) -> tuple[str, float] | None:
    violated = {v.klass: v.violated for v in views}
    map_set = ip.map_set
    if any(violated.get(k) for k in map_set):
        return None  # the admissible output itself fails a check: that is not a weak check
    counts = Counter(v.klass for v in views)
    eligible = sorted(
        k for k in {v.klass for v in views if not v.violated}
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


def diagnose(
    behaviours: Sequence[Behaviour],
    violated: Mapping[str, frozenset[str]],
    views: Sequence[ItemView],
    posterior: Posterior,
    refuter: Refuter,
    diff: Diff,
    *,
    tau: float = 0.05,
    ambig_threshold: float = 0.5,
    weak_threshold: float = 0.5,
    weak_min_runs: int = 2,
) -> list[Defect]:
    defects: list[Defect] = []
    weights = run_weights(views, posterior)
    by_item: dict[str, list[ItemView]] = defaultdict(list)
    for v in views:
        by_item[v.item].append(v)
    groups: dict[str, list[Behaviour]] = defaultdict(list)
    for b in behaviours:
        if b.status == "ok":
            groups[b.scenario.id].append(b)
    # Every scenario's candidates, refuted in descending score order: the probe budget
    # goes to the best supported candidates first (§11).
    candidates: list[tuple[float, str, frozenset[str]]] = []
    for sid, group in groups.items():
        sets = [(violated[b.run_key], weights.get(b.run_key, 0.0)) for b in group]
        candidates += [(support, sid, hitting)
                       for hitting, support in soft_hitting_sets(sets, tau=tau)]
    candidates.sort(key=lambda c: (-c[0], c[1], len(c[2]), sorted(c[2])))
    for support, sid, hitting in candidates:
        group = groups[sid]
        result = refuter.refute(hitting, group[0].scenario, group)
        witnesses = tuple(b.run_key for b in group if violated[b.run_key] & hitting)
        location = tuple(sorted(hitting))
        if not result.refuted and result.attempted:
            defects.append(Defect("contradiction", location, (sid,), witnesses, support,
                                  f"unrefuted after {result.attempts} attempts"))
        elif len(hitting) == 1:
            # Refuted, or never attempted because the probe budget ran out: a
            # contradiction needs a failed refutation attempt (§11).
            clause = next(iter(hitting))
            scenario_items = sorted({v.item for v in views if _scenario_of(v.item) == sid})
            score = _hidden_score(clause, scenario_items, posterior, by_item)
            if score is None:
                score = support
            note = (f"refuted by {result.witness}" if result.refuted
                    else "refutation not attempted: probe budget exhausted")
            defects.append(Defect("hidden", location, (sid,), witnesses, score, note))
        # A refuted set is dropped; so is an unattempted one, which is no contradiction.
    for item, ip in posterior.items.items():
        mine = by_item.get(item, [])
        projection = {v.klass: v.projection for v in mine}
        ranked = sorted(ip.p_member, key=lambda k: -ip.p_member[k])
        if ip.p_ambiguous >= ambig_threshold and len(ranked) >= 2:
            members = sorted(ip.map_set) if len(ip.map_set) >= 2 else ranked[:2]
            fields = frozenset().union(
                *(diff(projection[a], projection[b]) for a, b in combinations(members, 2))
            )
            if fields:
                witnesses = tuple(v.run_key for v in mine if v.klass in members)
                defects.append(Defect("ambiguity", tuple(sorted(fields)), (_scenario_of(item),),
                                      witnesses, ip.p_ambiguous, note=item))
        weak = _weak(ip, mine, weak_min_runs)
        if weak is not None and weak[1] >= weak_threshold and ranked:
            klass, score = weak
            fields = diff(projection[klass], projection[ranked[0]])
            if fields:
                witnesses = tuple(v.run_key for v in mine if v.klass == klass)
                defects.append(Defect("weak", tuple(sorted(fields)), (_scenario_of(item),),
                                      witnesses, score, note=item))
    return merge_defects(defects, key=identity)
