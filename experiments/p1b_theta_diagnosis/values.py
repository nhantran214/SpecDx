# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import Counter, defaultdict
from common import *
d = load()
ln, _ = lens(d["steps"])
which = sys.argv[2]
bs = [b for _, b in d[which]]
vals = defaultdict(Counter)
rowcount = defaultdict(Counter)
keys = defaultdict(Counter)
seen = set()
for b in bs:
    if b.status != "ok":
        continue
    # unique rows per run: use full end-state tables restricted to the tables of slices
    for v in ln.views(b, frozenset()):
        if not v.acted:
            continue
        step = v.item.split("|")[1]
        for table, rows in v.projection:
            rowcount[(step, table)][len(rows)] += 1
            for key, cells in rows:
                keys[table][key] += 1
                for c, x in cells:
                    vals[(table, c)][x] += 1
print("== rows per (step, table) in acted projections")
for k in sorted(rowcount):
    print(k, dict(sorted(rowcount[k].items())))
print("== row keys per table (counts over acted views)")
for t in sorted(keys):
    print(t, keys[t].most_common(40))
print("== values per locked/blank column")
for k in sorted(vals):
    print(k, vals[k].most_common(25))
