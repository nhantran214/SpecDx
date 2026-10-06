"""Set-valued latent-class model (spec §3.2).

Each item (a scenario step) has an unknown SET A of admissible output classes. A model
with competence θ emits an admissible class with probability θ (uniformly within A),
and otherwise a class drawn from the item's smoothed empirical error distribution ε.
Competence is shared across items, which is what separates a specification defect (every
competent model deviates at the same place) from an agent error (scattered).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

RARE = "__rare__"
_EDGE = 1e-4


@dataclass(frozen=True)
class Observation:
    item: str
    run_key: str
    model: str
    family: str
    klass: str


@dataclass
class ItemPosterior:
    classes: tuple[str, ...]
    set_posterior: dict[frozenset[str], float]
    p_member: dict[str, float]
    p_ambiguous: float
    responsibility: dict[str, float]

    @property
    def map_set(self) -> frozenset[str]:
        return max(
            self.set_posterior,
            key=lambda s: (self.set_posterior[s], -len(s), tuple(sorted(s))),
        )


@dataclass
class Posterior:
    theta: dict[str, float]
    family_effect: dict[str, float]
    items: dict[str, ItemPosterior]


def _sigmoid(x: float) -> float:
    p = 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))
    return min(max(p, _EDGE), 1 - _EDGE)


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


def _item_posterior(
    obs: Sequence[Observation], theta: dict[str, float], k_max: int, p_single: float
) -> ItemPosterior:
    counts = Counter(o.klass for o in obs)
    top = tuple(k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k_max])
    n, m = len(obs), len(top)
    eps = {k: (counts[k] + 1) / (n + m + 1) for k in top}
    eps[RARE] = (n - sum(counts[k] for k in top) + 1) / (n + m + 1)
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
    return ItemPosterior(top, post, p_member, p_ambiguous, responsibility)


def fit(
    observations: Sequence[Observation],
    *,
    k_max: int = 6,
    p_single: float = 0.9,
    sigma_v: float = 1.0,
    sigma_u: float = 1.0,
    iters: int = 50,
    tol: float = 1e-6,
    theta0: float = 0.7,
) -> Posterior:
    if not observations:
        return Posterior({}, {}, {})
    by_item: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_item[o.item].append(o)
    family = {o.model: o.family for o in observations}
    mu = _logit(theta0)
    eta = {m: mu for m in family}
    effect = {f: 0.0 for f in set(family.values())}
    for _ in range(iters):
        theta = {m: _sigmoid(e) for m, e in eta.items()}
        items = {i: _item_posterior(obs, theta, k_max, p_single) for i, obs in by_item.items()}
        rs: dict[str, list[float]] = defaultdict(list)
        for o in observations:
            rs[o.model].append(items[o.item].responsibility[o.run_key])
        new_eta: dict[str, float] = {}
        for model, values in rs.items():
            center = mu + effect[family[model]]
            e = eta[model]
            for _ in range(20):
                p = _sigmoid(e)
                grad = sum(values) - len(values) * p - (e - center) / sigma_v**2
                hess = -len(values) * p * (1 - p) - 1 / sigma_v**2
                step = grad / hess
                e -= step
                if abs(step) < 1e-9:
                    break
            new_eta[model] = e
        for fam in effect:
            members = [m for m in new_eta if family[m] == fam]
            effect[fam] = sigma_u**2 * sum(new_eta[m] - mu for m in members) / (
                sigma_u**2 * len(members) + sigma_v**2
            )
        mu = sum(new_eta[m] - effect[family[m]] for m in new_eta) / len(new_eta)
        delta = max(abs(_sigmoid(new_eta[m]) - _sigmoid(eta[m])) for m in new_eta)
        eta = new_eta
        if delta < tol:
            break
    theta = {m: _sigmoid(e) for m, e in eta.items()}
    items = {i: _item_posterior(obs, theta, k_max, p_single) for i, obs in by_item.items()}
    return Posterior(theta, effect, items)
