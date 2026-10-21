"""Modal readings per model or per interpreter (model, Tell variant, seed) of each cell."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from specdx.infer.cells import CellData
from specdx.types import Behaviour


def mode(values: Sequence[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def model_votes(data: CellData) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for cid, rs in data.by_cell().items():
        by: dict[str, list[str]] = {}
        for r in rs:
            by.setdefault(r.model, []).append(r.value)
        if by:
            out[cid] = {m: mode(vs) for m, vs in sorted(by.items())}
    return out


def interpreter_votes(data: CellData, behaviours: Sequence[Behaviour]) -> dict[str, dict[str, str]]:
    who = {b.run_key: f"{b.interpreter.model}|v{b.interpreter.variant}|s{b.interpreter.seed}"
           for b in behaviours}
    out: dict[str, dict[str, str]] = {}
    for cid, rs in data.by_cell().items():
        by: dict[str, list[str]] = {}
        for r in rs:
            by.setdefault(who.get(r.run_key, r.model), []).append(r.value)
        if by:
            out[cid] = {w: mode(vs) for w, vs in sorted(by.items())}
    return out
