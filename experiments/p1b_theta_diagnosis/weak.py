# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import defaultdict
from common import *
from specdx.infer.scores import _weak, _reference
d = load()
ln, _ = lens(d["steps"])
bs = [b for _, b in d["smoke"]]
views, acts, violated, rasch, post = build(bs, ln, SMOKE_POP)
by = defaultdict(list)
for v in views:
    if v.acted: by[v.item].append(v)
for item in ["happy|P2:decide_quote", "discount8|P2:decide_quote", "discount8|DE-FN-01:draft_sales_invoice",
             "happy|DE-FN-01:draft_sales_invoice", "happy|DE-FN-03:draft_payment", "discount8|DE-FN-03:draft_payment"]:
    ip = post.items[item]; mine = by[item]
    w = _weak(ip, mine, 2)
    if w is None: print(item, "no weak"); continue
    k, score = w; ref = _reference(ip)
    proj = {v.klass: v.projection for v in mine}
    def show(p): return [(t, [(key, {c: x for c, x in cells if c in ('trang_thai','doi_soat','actor','sl_dat')}) for key, cells in rows]) for t, rows in p if rows]
    print(item, f"score {score:.3f}; P(MAP set) {ip.set_posterior[ip.map_set]:.3f}")
    print("   reference (MAP):", show(proj[ref]), "n =", sum(v.klass == ref for v in mine))
    print("   weak deviant   :", show(proj[k]), "n =", sum(v.klass == k for v in mine), "fields", sorted(ln.diff(proj[k], proj[ref])))
