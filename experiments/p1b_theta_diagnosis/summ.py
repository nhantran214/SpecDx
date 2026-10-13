# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
import random
from collections import Counter, defaultdict
from common import *
from specdx.infer.latent import fit
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
    rows = []
    for i, vs in by.items():
        mine = [v.klass for v in vs if v.model == m]
        if len(mine) >= 3:
            c = Counter(mine)
            rows.append((len(mine), len(c), c.most_common(1)[0][1]))
    if not rows: continue
    k = len(rows)
    print(f"{m}: items with >=3 acted {k}; mean acted {sum(r[0] for r in rows)/k:.2f}; mean distinct classes {sum(r[1] for r in rows)/k:.2f}; "
          f"items where its own modal class holds >50% of its runs: {sum(r[2]*2 > r[0] for r in rows)}; items where every run is its own class: {sum(r[1]==r[0] for r in rows)}")
ms = sorted(pop)
if len(ms) == 2:
    both = same = 0
    for i, vs in by.items():
        cs = [Counter(v.klass for v in vs if v.model == m) for m in ms]
        if all(sum(c.values()) >= 3 for c in cs):
            both += 1
            same += cs[0].most_common(1)[0][0] == cs[1].most_common(1)[0][0]
    print(f"items where both models have >=3 acted: {both}; same modal class: {same}")
if which == "smoke":
    rng = random.Random(1)
    groups = defaultdict(list)
    for b in bs: groups[(b.interpreter.model, b.scenario.id)].append(b)
    vb = defaultdict(list)
    for v in views: vb[v.run_key].append(v)
    wins = 0; ths = []
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    for r in range(R):
        obs = []
        for g, members in groups.items():
            for j in range(len(members)):
                b = rng.choice(members)
                for v in vb[b.run_key]:
                    if v.acted:
                        obs.append(observations([v])[0].__class__(v.item, f"{v.run_key}#{r}#{j}", v.model, v.family, v.klass))
        p = fit(obs, models=pop, k_max=6, p_single=0.9)
        ths.append((p.theta["gpt-oss:20b"], p.theta["qwen3:30b-a3b"]))
        wins += p.theta["gpt-oss:20b"] > p.theta["qwen3:30b-a3b"]
    g = sorted(t[0] for t in ths); q = sorted(t[1] for t in ths)
    print(f"bootstrap over runs (R={R}): P(theta_gpt > theta_qwen) = {wins/R:.2f}; gpt 90% interval [{g[int(.05*R)]:.3f}, {g[int(.95*R)-1]:.3f}]; qwen [{q[int(.05*R)]:.3f}, {q[int(.95*R)-1]:.3f}]; max theta over all reps {max(max(t) for t in ths):.3f}")
