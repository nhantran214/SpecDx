"""Stage 2 of the act→read model: a set-valued latent-class model (spec v1 §3.2, v2 §3.3).

Each item (a scenario step) has an unknown SET A of admissible output classes. A model
with competence θ emits an admissible class with probability θ (uniformly within A),
and otherwise a class drawn from the item's error distribution ε. Competence is shared
across items, which is what separates a specification defect (every competent model
deviates at the same place) from an agent error (scattered).

ε is re-estimated in every EM iteration from the outputs not explained as faithful. P1
used the smoothed distribution of all outputs instead; then an item where everyone agrees
is explained as well by ε as by competence, the likelihood favours θ → 0, and EM drifts
there (the P1 inversion).

Only acted behaviours are observations here (items.observations). θ has a Beta(α, β)
prior and the M-step is its MAP, θ_m = (S_m + α − 1) / (n_m + α + β − 2), with S_m the
expected number of the model's admissible behaviours and n_m its acted ones: a model that
never acts keeps the prior mode and cannot drag the others' θ.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

RARE = "__rare__"
_EDGE = 1e-4
ALPHA, BETA = 7.0, 3.0  # spec v2 §3.10: θ ~ Beta(7, 3)


@dataclass(frozen=True)
class Observation:
    item: str
    run_key: str
    model: str
    family: str
    klass: str


@dataclass
class ItemPosterior:
    """error: the item's error distribution ε over its classes and RARE."""

    classes: tuple[str, ...]
    set_posterior: dict[frozenset[str], float]
    p_member: dict[str, float]
    p_ambiguous: float
    responsibility: dict[str, float]
    error: dict[str, float] = field(default_factory=dict)

    @property
    def map_set(self) -> frozenset[str]:
        return max(
            self.set_posterior,
            key=lambda s: (self.set_posterior[s], -len(s), tuple(sorted(s))),
        )


@dataclass
class Posterior:
    """family_effect: each family's mean logit θ minus the population's (reported; the
    Beta prior, not the family, anchors θ)."""

    theta: dict[str, float]
    family_effect: dict[str, float]
    items: dict[str, ItemPosterior]


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _log_sum_exp(values: Sequence[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def _subset_prior(m: int, p_single: float) -> dict[int, float]:
    """Prior mass of one subset of each size: singletons share p_single."""
    if m == 1:
        return {1: 1.0}
    weights = {s: 0.5 ** (s - 1) for s in range(2, m + 1)}
    total = sum(weights.values())
    prior = {1: p_single / m}
    for size, w in weights.items():
        prior[size] = (1 - p_single) * (w / total) / math.comb(m, size)
    return prior


def _top(obs: Sequence[Observation], k_max: int) -> tuple[str, ...]:
    counts = Counter(o.klass for o in obs)
    return tuple(k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k_max])


def _error_update(
    obs: Sequence[Observation], top: tuple[str, ...], responsibility: Mapping[str, float]
) -> dict[str, float]:
    """The M-step for ε: what the outputs not explained as faithful look like, smoothed by
    one per class. Starting from every output (all responsibilities 0) gives the plain
    smoothed empirical distribution."""
    mass: dict[str, float] = {k: 1.0 for k in (*top, RARE)}
    for o in obs:
        k = o.klass if o.klass in top else RARE
        mass[k] += 1.0 - responsibility.get(o.run_key, 0.0)
    total = sum(mass.values())
    return {k: v / total for k, v in mass.items()}


def _item_posterior(
    obs: Sequence[Observation],
    theta: dict[str, float],
    k_max: int,
    p_single: float,
    eps: Mapping[str, float] | None = None,
) -> ItemPosterior:
    top = _top(obs, k_max)
    m = len(top)
    if eps is None:
        eps = _error_update(obs, top, {})
    labelled = [(o, o.klass if o.klass in top else RARE) for o in obs]
    prior = _subset_prior(m, p_single)
    subsets = [frozenset(c) for size in range(1, m + 1) for c in combinations(top, size)]
    log_post: dict[frozenset[str], float] = {}
    for subset in subsets:
        lp = math.log(prior[len(subset)])
        for o, k in labelled:
            th = theta[o.model]
            faith = th / len(subset) if k in subset else 0.0
            lp += math.log(faith + (1 - th) * eps[k])
        log_post[subset] = lp
    z = _log_sum_exp(list(log_post.values()))
    post = {s: math.exp(v - z) for s, v in log_post.items()}
    p_member = {k: sum(p for s, p in post.items() if k in s) for k in top}
    p_ambiguous = sum(p for s, p in post.items() if len(s) >= 2)
    responsibility: dict[str, float] = {}
    for o, k in labelled:
        th = theta[o.model]
        r = 0.0
        for s, p in post.items():
            faith = th / len(s) if k in s else 0.0
            r += p * faith / (faith + (1 - th) * eps[k])
        responsibility[o.run_key] = r
    return ItemPosterior(top, post, p_member, p_ambiguous, responsibility, dict(eps))


def _family_effect(theta: dict[str, float], family: dict[str, str]) -> dict[str, float]:
    logits = {m: _logit(min(max(th, _EDGE), 1 - _EDGE)) for m, th in theta.items()}
    mu = sum(logits.values()) / len(logits)
    out: dict[str, float] = {}
    for fam in sorted(set(family.values())):
        members = [logits[m] for m in sorted(theta) if family[m] == fam]
        out[fam] = sum(members) / len(members) - mu
    return out


def fit(
    observations: Sequence[Observation],
    *,
    models: Mapping[str, str] | None = None,
    k_max: int = 6,
    p_single: float = 0.9,
    alpha: float = ALPHA,
    beta: float = BETA,
    iters: int = 50,
    tol: float = 1e-6,
    theta0: float = 0.7,
) -> Posterior:
    """models: model → family of the whole population. A listed model without any acted
    behaviour still gets θ, the prior's mode (n_m = 0)."""
    family = dict(models or {}) | {o.model: o.family for o in observations}
    if not family:
        return Posterior({}, {}, {})
    by_item: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_item[o.item].append(o)
    theta = {m: theta0 for m in sorted(family)}
    eps: dict[str, dict[str, float]] = {}
    for _ in range(iters):
        items = {i: _item_posterior(obs, theta, k_max, p_single, eps.get(i))
                 for i, obs in by_item.items()}
        new_eps = {i: _error_update(obs, items[i].classes, items[i].responsibility)
                   for i, obs in by_item.items()}
        moved = max((abs(new_eps[i][k] - items[i].error[k]) for i in new_eps for k in new_eps[i]),
                    default=0.0)
        eps = new_eps
        admissible: dict[str, float] = defaultdict(float)
        acted: Counter[str] = Counter()
        for o in observations:
            admissible[o.model] += items[o.item].responsibility[o.run_key]
            acted[o.model] += 1
        new = {
            m: min(max((admissible[m] + alpha - 1) / (acted[m] + alpha + beta - 2), _EDGE),
                   1 - _EDGE)
            for m in theta
        }
        delta = max(abs(new[m] - theta[m]) for m in theta)
        theta = new
        if max(delta, moved) < tol:
            break
    items = {i: _item_posterior(obs, theta, k_max, p_single, eps.get(i))
             for i, obs in by_item.items()}
    return Posterior(theta, _family_effect(theta, family), items)
