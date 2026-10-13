# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from common import *
d = load()
ln, pack = lens(d["steps"])
for s in ln.steps:
    print(s.id, s.kind, s.actor, "slice=", s.slice, "drafting=", s.id in ln.drafting)
cm = ln.columns
for t in sorted(cm.keys):
    print(t, "pk", cm.keys[t], {c: str(k) for (tt, c), k in cm.classes.items() if tt == t})
