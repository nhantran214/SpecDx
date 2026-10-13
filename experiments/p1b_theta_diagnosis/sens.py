# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import Counter, defaultdict
from itertools import combinations
from common import *
from specdx.infer.latent import fit, Observation
from trace import traced_fit
d = load()
ln, _ = lens(d["steps"])
bs = [b for _, b in d["smoke"]]
views, acts, violated, rasch, post = build(bs, ln, SMOKE_POP)
acted = [v for v in views if v.acted]
obs = observations(views)
print("== start-point sensitivity (fit with theta0)")
for t0 in (0.95, 0.9, 0.7, 0.5, 0.3, 0.1):
    p = fit(obs, models=SMOKE_POP, k_max=6, p_single=0.9, theta0=t0, iters=500)
    print(f"theta0 {t0}: " + ", ".join(f"{m} {p.theta[m]:.4f}" for m in sorted(p.theta)))
print("== 50-iteration cap reached?")
th, items, n_it, _ = traced_fit(obs, SMOKE_POP, log=False)
print("iterations used", n_it)
print("== item subsets")
draft = ln.drafting
def sub(pred, label):
    o = [x for x in obs if pred(x.item.split('|')[1], x.item)]
    p = fit(o, models=SMOKE_POP, k_max=6, p_single=0.9)
    print(f"{label}: n={len(o)} items={len({x.item for x in o})} " + ", ".join(f"{m} {p.theta[m]:.3f}" for m in sorted(p.theta)))
sub(lambda s, i: s in draft, "drafting steps only")
sub(lambda s, i: s not in draft, "non-drafting steps only (P2, P6, DE-OP-01)")
sub(lambda s, i: s == "DE-OP-01:check_availability", "DE-OP-01 only (always empty projection, unanimous)")
sub(lambda s, i: s in ("DE-OP-01:check_availability", "DE-OP-02:record_goods_receipt", "DE-FN-03:draft_payment"), "the three least fragmented steps")
print("== within-model agreement by variant relation")
by = defaultdict(list)
for v in acted: by[v.item].append(v)
seed = {b.run_key: b.interpreter.seed for b in bs}
same = Counter(); tot = Counter()
for i, vs in by.items():
    for a, b in combinations(vs, 2):
        if a.model != b.model: continue
        rel = "same variant" if a.variant == b.variant else "different variant"
        tot[(a.model, rel)] += 1; same[(a.model, rel)] += a.klass == b.klass
for k in sorted(tot): print(k, f"{same[k]}/{tot[k]} = {same[k]/tot[k]:.3f}")
