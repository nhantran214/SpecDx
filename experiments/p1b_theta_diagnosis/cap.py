# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import Counter, defaultdict
from common import *
d = load()
ln, _ = lens(d["steps"])
which = sys.argv[2]
pop = SMOKE_POP if which == "smoke" else P1_POP
bs = [b for _, b in d[which]]
views, acts, violated, rasch, post = build(bs, ln, pop)
by = defaultdict(list)
for v in views:
    if v.acted: by[v.item].append(v)
for m in sorted(pop):
    rates = []
    for i, vs in by.items():
        mine = [v for v in vs if v.model == m]
        if len(mine) < 3: continue
        ms = post.items[i].map_set
        rates.append((sum(v.klass in ms for v in mine) / len(mine), len(mine), i))
    rates.sort()
    k = len(rates)
    print(m, f"items with >=3 acted: {k}; share of the model's behaviours in the item's MAP class: min {rates[0][0]:.3f}, "
          f"q25 {rates[k//4][0]:.3f}, median {rates[k//2][0]:.3f}, max {rates[-1][0]:.3f}; "
          f"items with share <= 0.2: {sum(r <= 0.2 for r, _, _ in rates)}; theta {post.theta[m]:.3f}")
    print("   lowest:", [(f"{r:.2f}", n, i) for r, n, i in rates[:6]])
