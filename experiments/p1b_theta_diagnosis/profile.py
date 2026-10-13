# ruff: noqa
"""Profile log posterior over a common θ, ε and A handled as in latent.py (ε by EM with θ held)."""
import sys; sys.path.insert(0, sys.argv[1])
import math
from collections import defaultdict
from common import *
from specdx.infer import latent as L

d = load()
ln, _ = lens(d["steps"])
which = sys.argv[2]
pop = SMOKE_POP if which == "smoke" else P1_POP
bs = [b for _, b in d[which]]
views, acts, violated, rasch, post = build(bs, ln, pop)
obs = observations(views)
by_item = defaultdict(list)
for o in obs: by_item[o.item].append(o)

def item_loglik(ob, theta, eps, k_max=6, p_single=0.9):
    top = L._top(ob, k_max); m = len(top)
    prior = L._subset_prior(m, p_single)
    from itertools import combinations
    subsets = [frozenset(c) for size in range(1, m + 1) for c in combinations(top, size)]
    vals = []
    for s in subsets:
        lp = math.log(prior[len(s)])
        for o in ob:
            k = o.klass if o.klass in top else L.RARE
            th = theta[o.model]
            faith = th / len(s) if k in s else 0.0
            lp += math.log(faith + (1 - th) * eps[k])
        vals.append(lp)
    return L._log_sum_exp(vals)

def objective(theta, inner=150):
    eps = {}
    for _ in range(inner):
        items = {i: L._item_posterior(ob, theta, 6, 0.9, eps.get(i)) for i, ob in by_item.items()}
        eps = {i: L._error_update(ob, items[i].classes, items[i].responsibility) for i, ob in by_item.items()}
    ll = sum(item_loglik(ob, theta, eps[i]) for i, ob in by_item.items())
    dirichlet = sum(math.log(v) for e in eps.values() for v in e.values())
    beta = sum(6 * math.log(t) + 2 * math.log(1 - t) for t in theta.values())
    resp_unanimous = [items[i].responsibility for i in items if len(items[i].classes) == 1]
    r_un = sum(sum(r.values()) for r in resp_unanimous) / max(sum(len(r) for r in resp_unanimous), 1)
    return ll, dirichlet, beta, ll + dirichlet + beta, r_un

grid = [float(x) for x in sys.argv[3].split(",")]
print("| common θ | log-lik | Dirichlet(ε) term | Beta(θ) term | log posterior | mean responsibility at unanimous items |")
print("|---:|---:|---:|---:|---:|---:|")
for g in grid:
    th = {m: g for m in pop}
    ll, dr, be, tot, r = objective(th)
    print(f"| {g:.3f} | {ll:.1f} | {dr:.1f} | {be:.1f} | {tot:.1f} | {r:.3f} |")
th = dict(post.theta)
ll, dr, be, tot, r = objective(th)
print(f"| fit() MAP {', '.join(f'{v:.3f}' for v in th.values())} | {ll:.1f} | {dr:.1f} | {be:.1f} | {tot:.1f} | {r:.3f} |")
