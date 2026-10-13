# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import Counter, defaultdict
from common import *
d = load()
ln, _ = lens(d["steps"])
which = sys.argv[2]
bs = [b for _, b in d[which]]
by = defaultdict(list)
for b in bs:
    for v in ln.views(b, frozenset()):
        if v.acted: by[v.item].append(v)
for item in sys.argv[3:]:
    c = Counter(v.klass for v in by[item])
    for k, n in c.most_common(3):
        proj = next(v.projection for v in by[item] if v.klass == k)
        print(item, k, n, "rows:", [(t, [(key, dict(cells)) for key, cells in rows]) for t, rows in proj])
