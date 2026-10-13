# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
from collections import Counter, defaultdict
from common import *
d = load()
ln, _ = lens(d["steps"])
which, item = sys.argv[2], sys.argv[3]
pop = SMOKE_POP if which == "smoke" else P1_POP
bs = {b.run_key: b for _, b in d[which]}
for b in bs.values():
    for v in ln.views(b, ln.violations(b)):
        if v.item == item and v.acted:
            print(f"--- {v.klass} {v.model} v{v.variant} s{b.interpreter.seed} viol={sorted(v.violated)}")
            for table, rows in v.projection:
                for key, cells in rows:
                    print(f"     {table} {key} {dict(cells)}")
