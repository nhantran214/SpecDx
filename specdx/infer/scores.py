"""From behaviours and a posterior to typed, located defects (spec §3.3–§3.4)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

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


def merge_defects(defects: Iterable[Defect]) -> list[Defect]:
    merged: dict[tuple[str, tuple[str, ...]], Defect] = {}
    for d in defects:
        old = merged.get(d.key)
        if old is None:
            merged[d.key] = d
            continue
        best = d if d.score > old.score else old
        merged[d.key] = Defect(
            d.type, d.location,
            tuple(sorted(set(old.scenarios) | set(d.scenarios))),
            tuple(sorted(set(old.witnesses) | set(d.witnesses))),
            best.score, best.note,
        )
    return sorted(merged.values(), key=lambda d: (-d.score, d.type, d.location))


def _scenario_of(item: str) -> str:
    return item.split("|", 1)[0]


def _weak(ip: ItemPosterior, views: Sequence[ItemView], min_runs: int) -> tuple[str, float] | None:
    violated = {v.klass: v.violated for v in views}
    if any(violated.get(k) for k in ip.map_set):
        return None  # the admissible output itself fails a check: that is not a weak check
    counts = Counter(v.klass for v in views)
    best: tuple[str, float] | None = None
    for klass in {v.klass for v in views if not v.violated}:
        if counts[klass] < min_runs or klass in ip.map_set:
            continue
        score = (
            sum(p for s, p in ip.set_posterior.items() if klass not in s)
            if klass in ip.classes else 1.0
        )
        if best is None or score > best[1]:
            best = (klass, score)
    return best


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
    groups: dict[str, list[Behaviour]] = defaultdict(list)
    for b in behaviours:
        if b.status == "ok":
            groups[b.scenario.id].append(b)
    for sid, group in groups.items():
        sets = [(violated[b.run_key], weights.get(b.run_key, 0.0)) for b in group]
        for hitting, support in soft_hitting_sets(sets, tau=tau):
            result = refuter.refute(hitting, group[0].scenario, group)
            witnesses = tuple(b.run_key for b in group if violated[b.run_key] & hitting)
            location = tuple(sorted(hitting))
            if not result.refuted:
                defects.append(Defect("contradiction", location, (sid,), witnesses, support,
                                      f"unrefuted after {result.attempts} attempts"))
            elif len(hitting) == 1:
                defects.append(Defect("hidden", location, (sid,), witnesses, support,
                                      f"refuted by {result.witness}"))
    by_item: dict[str, list[ItemView]] = defaultdict(list)
    for v in views:
        by_item[v.item].append(v)
    for item, ip in posterior.items.items():
        mine = by_item.get(item, [])
        projection = {v.klass: v.projection for v in mine}
        ranked = sorted(ip.p_member, key=lambda k: -ip.p_member[k])
        if ip.p_ambiguous >= ambig_threshold and len(ranked) >= 2:
            fields = diff(projection[ranked[0]], projection[ranked[1]])
            if fields:
                witnesses = tuple(v.run_key for v in mine if v.klass in ranked[:2])
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
    return merge_defects(defects)
