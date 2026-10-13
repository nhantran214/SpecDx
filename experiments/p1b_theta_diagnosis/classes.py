# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import Counter, defaultdict
from itertools import combinations
from common import *
d = load()
ln, _ = lens(d["steps"])
which = sys.argv[2]
pop = SMOKE_POP if which == "smoke" else P1_POP
bs = [b for _, b in d[which]]
views, acts, violated, rasch, post = build(bs, ln, pop)
by_item = defaultdict(list)
for v in views:
    if v.acted:
        by_item[v.item].append(v)
models = sorted(pop)
print(f"| item | obs | n | classes | top class share | " + " | ".join(f"{m[:10]} n/classes/top" for m in models) + " | shared classes | pair agree (within) | pair agree (between) |")
tot = Counter()
rows = []
for item in sorted(by_item):
    vs = by_item[item]
    n = len(vs)
    cnt = Counter(v.klass for v in vs)
    per = {m: Counter(v.klass for v in vs if v.model == m) for m in models}
    shared = set.intersection(*[set(c) for c in per.values() if c]) if sum(bool(c) for c in per.values()) >= 2 else set()
    # pairwise agreement rates
    w_same = w_tot = b_same = b_tot = 0
    for a, b in combinations(vs, 2):
        if a.model == b.model:
            w_tot += 1; w_same += a.klass == b.klass
        else:
            b_tot += 1; b_same += a.klass == b.klass
    tot["w_same"] += w_same; tot["w_tot"] += w_tot; tot["b_same"] += b_same; tot["b_tot"] += b_tot
    tot["items"] += 1; tot["n"] += n; tot["classes"] += len(cnt)
    tot["singleton_items"] += (len(cnt) == n)
    tot["unanimous"] += (len(cnt) == 1)
    tot["shared_any"] += bool(shared)
    top = cnt.most_common(1)[0][1]
    tot["top"] += top
    cells = []
    for m in models:
        c = per[m]
        cells.append(f"{sum(c.values())}/{len(c)}/{c.most_common(1)[0][1] if c else 0}")
    obs = "y" if rasch.observable(item) else "n"
    print(f"| {item} | {obs} | {n} | {len(cnt)} | {top}/{n} | " + " | ".join(cells) + f" | {len(shared)} | {w_same}/{w_tot} | {b_same}/{b_tot} |")
print()
print("TOTAL items", tot["items"], "acted", tot["n"], "distinct classes summed", tot["classes"],
      "items where every acted view is its own class", tot["singleton_items"],
      "unanimous items", tot["unanimous"], "items with a class shared across models", tot["shared_any"])
print("pairwise within-model agreement", tot["w_same"], "/", tot["w_tot"], "=", round(tot["w_same"]/max(tot["w_tot"],1),3))
print("pairwise between-model agreement", tot["b_same"], "/", tot["b_tot"], "=", round(tot["b_same"]/max(tot["b_tot"],1),3))
print("modal-class share overall", tot["top"], "/", tot["n"], "=", round(tot["top"]/tot["n"],3))
