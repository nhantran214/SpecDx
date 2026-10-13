# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from common import *
from specdx.infer.scores import run_weights
d = load()
ln, _ = lens(d["steps"])
for which, pop in (("smoke", SMOKE_POP), ("p1", P1_POP)):
    bs = [b for _, b in d[which]]
    views, acts, violated, rasch, post = build(bs, ln, pop)
    w = run_weights(views, post)
    for m in sorted(pop):
        ws = sorted(w[b.run_key] for b in bs if b.interpreter.model == m)
        k = len(ws)
        print(which, m, f"run weights min {ws[0]:.4f} median {ws[k//2]:.4f} max {ws[-1]:.4f}")
