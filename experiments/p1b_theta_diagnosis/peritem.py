# ruff: noqa
import sys; sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[0], sys.argv[1], "smoke"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import normalise as N
from collections import Counter, defaultdict
items = ["happy|DE-SM-02:draft_quote", "discount8|DE-FN-04:draft_purchase_order", "discount8|P6:decide_po", "happy|DE-FN-01:draft_sales_invoice", "grn12|P6:decide_po"]
print("| item | " + " | ".join(n for n, _ in N.PIPE) + " |")
for it in items:
    cells = []
    for name, fns in N.PIPE:
        ks = []
        for v in N.views:
            if v.item != it: continue
            p = v.projection
            for f in fns: p = f(p, it.split("|", 1)[1])
            ks.append(repr(p))
        c = Counter(ks)
        cells.append(f"{len(c)} cl / modal {c.most_common(1)[0][1]}/{len(ks)}")
    print(f"| {it} | " + " | ".join(cells) + " |")
