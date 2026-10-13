# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[0], sys.argv[1], "smoke"]
from collections import Counter, defaultdict
from itertools import combinations
import normalise as N  # reuses its views and transforms (prints its table first)
ln = N.ln
def apply(v, fns):
    step = v.item.split("|", 1)[1]
    p = v.projection
    for f in fns: p = f(p, step)
    return p
for label, fns in (("as-is", []), ("after T4", N.PIPE[4][1])):
    by = defaultdict(list)
    for v in N.views: by[v.item].append((v, apply(v, fns)))
    fieldc = Counter(); pairs = 0; onlyc = Counter()
    for i, vs in by.items():
        for (a, pa), (b, pb) in combinations(vs, 2):
            if pa == pb: continue
            pairs += 1
            f = ln.columns.diff(pa, pb)
            # evt fields: group by kind
            g = frozenset(x if not x.startswith("evt:") else "evt:" + x.split(":")[1] for x in f)
            for x in g: fieldc[x] += 1
            if len(g) == 1: onlyc[next(iter(g))] += 1
            if not g: onlyc["(no field: duplicate rows only)"] += 1
    print(f"\n== {label}: {pairs} disagreeing same-item pairs; field differs in how many of them (top 25)")
    for k, n in fieldc.most_common(25): print(f"  {k}: {n} ({n/pairs:.1%})")
    print(f"  -- pairs separated by exactly one field (top 15)")
    for k, n in onlyc.most_common(15): print(f"  {k}: {n}")
