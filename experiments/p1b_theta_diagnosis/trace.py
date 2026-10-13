# ruff: noqa
"""Trace stage-2 EM exactly as specdx.infer.latent.fit runs it, logging per iteration."""
import sys; sys.path.insert(0, sys.argv[1])
import math
from collections import Counter, defaultdict
from common import *
from specdx.infer import latent as L

def traced_fit(observations, models, k_max=6, p_single=0.9, alpha=7.0, beta=3.0, iters=50,
               tol=1e-6, theta0=0.7, log=True):
    family = dict(models) | {o.model: o.family for o in observations}
    by_item = defaultdict(list)
    for o in observations:
        by_item[o.item].append(o)
    theta = {m: theta0 for m in sorted(family)}
    eps = {}
    hist = []
    for it in range(iters):
        items = {i: L._item_posterior(obs, theta, k_max, p_single, eps.get(i)) for i, obs in by_item.items()}
        new_eps = {i: L._error_update(obs, items[i].classes, items[i].responsibility) for i, obs in by_item.items()}
        moved = max(abs(new_eps[i][k] - items[i].error[k]) for i in new_eps for k in new_eps[i])
        eps = new_eps
        adm = defaultdict(float); acted = Counter()
        for o in observations:
            adm[o.model] += items[o.item].responsibility[o.run_key]; acted[o.model] += 1
        new = {m: min(max((adm[m] + alpha - 1) / (acted[m] + alpha + beta - 2), L._EDGE), 1 - L._EDGE) for m in theta}
        delta = max(abs(new[m] - theta[m]) for m in theta)
        hist.append((it, dict(theta), dict(adm), dict(acted), new))
        if log and (it < 12 or it % 5 == 0):
            print(f"iter {it:2d} theta_in " + ", ".join(f"{m[:12]} {theta[m]:.4f}" for m in sorted(theta))
                  + " | S_m " + ", ".join(f"{adm[m]:.2f}" for m in sorted(theta))
                  + " | n_m " + ", ".join(f"{acted[m]}" for m in sorted(theta))
                  + " | theta_out " + ", ".join(f"{new[m]:.4f}" for m in sorted(theta))
                  + f" | eps moved {moved:.4f}")
        theta = new
        if max(delta, moved) < tol:
            break
    items = {i: L._item_posterior(obs, theta, k_max, p_single, eps.get(i)) for i, obs in by_item.items()}
    return theta, items, it + 1, by_item

if __name__ == "__main__":
    d = load()
    ln, _ = lens(d["steps"])
    which = sys.argv[2]
    pop = SMOKE_POP if which == "smoke" else P1_POP
    bs = [b for _, b in d[which]]
    views, acts, violated, rasch, post = build(bs, ln, pop)
    obs = observations(views)
    theta, items, n_it, by_item = traced_fit(obs, pop)
    print("converged after", n_it, "iterations; theta", {m: round(t, 4) for m, t in theta.items()},
          "fit() gives", {m: round(t, 4) for m, t in post.theta.items()})
    # per-item MAP set, responsibilities, eps
    print()
    print("| item | n | classes | top-6 counts | MAP set (size) | P(MAP) | count in MAP | eps on MAP class | eps RARE | mean resp (MAP-class obs) | sum resp |")
    tot_resp = Counter(); tot_map = 0; tot_n = 0
    for i in sorted(items):
        ip = items[i]; ob = by_item[i]
        cnt = Counter(o.klass for o in ob)
        ms = ip.map_set
        inmap = [o for o in ob if o.klass in ms]
        r_in = [ip.responsibility[o.run_key] for o in inmap]
        sresp = sum(ip.responsibility[o.run_key] for o in ob)
        tot_map += len(inmap); tot_n += len(ob)
        for o in ob: tot_resp[o.model] += ip.responsibility[o.run_key]
        eps_map = ", ".join(f"{ip.error[k]:.3f}" for k in sorted(ms))
        print(f"| {i} | {len(ob)} | {len(cnt)} | {[cnt[k] for k in ip.classes]} | {len(ms)} | {ip.set_posterior[ms]:.3f} | {len(inmap)} | {eps_map} | {ip.error[L.RARE]:.3f} | {sum(r_in)/max(len(r_in),1):.3f} | {sresp:.2f} |")
    print("obs in MAP classes", tot_map, "of", tot_n, "; S_m", {m: round(v, 2) for m, v in tot_resp.items()})
