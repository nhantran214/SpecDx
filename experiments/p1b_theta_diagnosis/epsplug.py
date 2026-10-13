# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import defaultdict
from common import *
from specdx.infer import latent as L
d = load()
ln, _ = lens(d["steps"])
for which, pop in (("smoke", SMOKE_POP), ("p1", P1_POP)):
    bs = [b for _, b in d[which]]
    views, acts, violated, rasch, post = build(bs, ln, pop)
    obs = observations(views)
    by = defaultdict(list)
    for o in obs: by[o.item].append(o)
    diffs = []; tv = []
    for i, ob in by.items():
        ip = post.items[i]
        plug = L._error_update(ob, ip.classes, {})
        diffs.append(max(abs(plug[k] - ip.error[k]) for k in plug))
        tv.append(0.5 * sum(abs(plug[k] - ip.error[k]) for k in plug))
    S = sum(sum(post.items[i].responsibility.values()) for i in by)
    print(f"{which}: acted {len(obs)}; total responsibility {S:.1f} ({S/len(obs):.1%} faithful, {1-S/len(obs):.1%} attributed to eps); "
          f"TV distance between fitted eps and the P1 plug-in (smoothed empirical): mean {sum(tv)/len(tv):.3f}, max {max(tv):.3f}")
    # P1 plug-in fit: eps fixed to smoothed empirical, theta by the same M-step
    theta = {m: 0.7 for m in pop}
    for _ in range(200):
        items = {i: L._item_posterior(ob, theta, 6, 0.9, None) for i, ob in by.items()}
        adm = defaultdict(float); n = defaultdict(int)
        for o in obs: adm[o.model] += items[o.item].responsibility[o.run_key]; n[o.model] += 1
        theta = {m: min(max((adm[m] + 6) / (n[m] + 8), 1e-4), 1 - 1e-4) for m in theta}
    print(f"   with eps frozen at the P1 plug-in instead: theta {', '.join(f'{m} {t:.3f}' for m, t in sorted(theta.items()))}")
