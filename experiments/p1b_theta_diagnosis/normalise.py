# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
import hashlib
from collections import Counter, defaultdict
from itertools import combinations
from common import *
from specdx.infer.latent import Observation, fit

d = load()
ln, _ = lens(d["steps"])
which = sys.argv[2]
pop = SMOKE_POP if which == "smoke" else P1_POP
actor = {s.id: s.actor for s in ln.steps}
kind = {s.id: s.kind for s in ln.steps}
bs = [b for _, b in d[which]]
views = []
for b in bs:
    views.extend(v for v in ln.views(b, frozenset()) if v.acted)

DRAFT = {"draft", "nháp"}

def t_case(proj, step):
    return tuple((t, tuple(sorted((k.strip().casefold(), tuple((c, x.strip().casefold()) for c, x in cells)) for k, cells in rows))) for t, rows in proj)

def t_surrogate(proj, step):
    out = []
    for t, rows in proj:
        if t == "soline":
            rows = tuple(sorted((f"{dict(cells).get('so','')}:{dict(cells).get('ma_sp','')}",
                                 tuple((c, x) for c, x in cells if c != "ma_soline")) for k, cells in rows))
        out.append((t, rows))
    return tuple(out)

def t_status(proj, step):
    out = []
    for t, rows in proj:
        rows = tuple(sorted((k, tuple((c, ("nháp" if c == "trang_thai" and x in DRAFT else x)) for c, x in cells)) for k, cells in rows))
        out.append((t, rows))
    return tuple(out)

def t_ownevt(proj, step):
    out = []
    for t, rows in proj:
        if t == "evt":
            rows = tuple(r for r in rows if dict(r[1]).get("actor", "") == actor[step].casefold())
        out.append((t, rows))
    return tuple(out)

def t_persona(proj, step):
    if kind[step] != "persona":
        return proj
    out = []
    for t, rows in proj:
        if t in ("so", "po"):
            rows = tuple(sorted((k, tuple((c, x) for c, x in cells if c == "trang_thai")) for k, cells in rows))
        out.append((t, rows))
    return tuple(out)

PIPE = [("T0 as-is", []), ("T1 casefold+strip", [t_case]),
        ("T2 +soline surrogate key", [t_case, t_surrogate]),
        ("T3 +draft synonyms", [t_case, t_surrogate, t_status]),
        ("T4 +own events only", [t_case, t_surrogate, t_status, t_ownevt]),
        ("T5 +persona keeps status only", [t_case, t_surrogate, t_status, t_ownevt, t_persona])]

def summary(obs):
    by = defaultdict(list)
    for o in obs: by[o.item].append(o)
    w = [0, 0]; bt = [0, 0]; modal = 0; classes = 0
    modal_m = Counter(); n_m = Counter()
    for i, ob in by.items():
        c = Counter(o.klass for o in ob); classes += len(c)
        top = c.most_common(1)[0][0]; modal += c[top]
        for o in ob:
            n_m[o.model] += 1; modal_m[o.model] += o.klass == top
        for a, b in combinations(ob, 2):
            if a.model == b.model: w[0] += a.klass == b.klass; w[1] += 1
            else: bt[0] += a.klass == b.klass; bt[1] += 1
    ub = {m: (modal_m[m] + 6) / (n_m[m] + 8) for m in sorted(n_m)}
    return len(by), classes, modal / len(obs), w[0] / max(w[1], 1), bt[0] / max(bt[1], 1), ub

print("| projection | items | classes (sum) | modal share | within-model pair agreement | between-model pair agreement | θ fit() | θ ceiling (modal class, r=1) |")
print("|---|---:|---:|---:|---:|---:|---|---|")
for name, fns in PIPE:
    obs = []
    for v in views:
        step = v.item.split("|", 1)[1]
        p = v.projection
        for f in fns:
            p = f(p, step)
        k = hashlib.sha1(repr(p).encode()).hexdigest()[:16]
        obs.append(Observation(v.item, v.run_key, v.model, v.family, k))
    n_items, classes, modal, w, bt, ub = summary(obs)
    post = fit(obs, models=pop, k_max=6, p_single=0.9)
    th = ", ".join(f"{m.split(':')[0]} {post.theta[m]:.3f}" for m in sorted(post.theta))
    ubs = ", ".join(f"{m.split(':')[0]} {ub[m]:.3f}" for m in sorted(ub))
    print(f"| {name} | {n_items} | {classes} | {modal:.3f} | {w:.3f} | {bt:.3f} | {th} | {ubs} |")
