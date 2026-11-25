"""Permutation placebo for the ability-gradient mechanism (design §6.2; ruling P-C).

Within each task the act vectors of the runs without a tool error are shuffled across those
runs. That keeps every task's multiset of readings, and so every cell's (a tool-error run gives
no reading and keeps its own record: the minimal intervention), but cuts the link between a
reading and its reader's ability, so v4's findings, which come from that link, should mostly
vanish. v3's findings are driven by majorities and should not. The unit counted is the cell:
a changeable cell carrying at least one defect type counts once.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from specdx.infer.anchored import V3Params, fit_anchored
from specdx.infer.anchored_v4 import AnchoredFitV4, V4Params, fit_v4
from specdx.infer.typing import CellFinding, type_cells
from specdx.infer.typing_v4 import DEFECT_TYPES_V4, type_cells_v4
from specdx.subjects.sopbench.cells import S3CellData, cells_for
from specdx.subjects.sopbench.load import S3Run, S3Task

ACT_FIELDS = ("succeeded", "db_match", "constraints", "process")


@dataclass(frozen=True)
class PlaceboResult:
    """real_*: the changeable cells of the real runs that carry at least one defect type
    (hidden, ambiguity or weak), counted once each; placebo_*: the same count per seed;
    ratio_*: mean placebo count over the real count (nan when that is 0); changeable: the
    number of changeable cells."""

    domain: str
    real_v4: int
    real_v3: int
    placebo_v4: list[int]
    placebo_v3: list[int]
    ratio_v4: float
    ratio_v3: float
    changeable: int


def permute_within_tasks(runs: Sequence[S3Run], seed: int) -> list[S3Run]:
    """Shuffle each task's act vectors (succeeded, db_match, constraints, process) across
    its runs without a tool error (ruling P-C); a tool-error run is left as it is. Each run
    keeps its model, family, mode, prompt and tool_error, and the output keeps the input
    order. A task's runs are taken in run_key order and its generator is keyed by (seed,
    crc32(domain), task), so the result does not depend on the order of `runs` and differs
    between domains."""
    out = list(runs)
    by_task: dict[tuple[str, int], list[int]] = {}
    for i, r in enumerate(runs):
        if not r.tool_error:
            by_task.setdefault((r.domain, r.task), []).append(i)
    for (domain, task), idx in sorted(by_task.items()):
        idx.sort(key=lambda i: runs[i].run_key)
        key = [seed, zlib.crc32(domain.encode("utf-8")), task]
        perm = np.random.default_rng(key).permutation(len(idx))
        for pos, j in zip(idx, perm, strict=True):
            src = runs[idx[j]]
            out[pos] = replace(runs[pos], **{f: getattr(src, f) for f in ACT_FIELDS})
    return out


def changeable_cells(cd: S3CellData) -> frozenset[str]:
    """The cells whose readings are not all equal: only these can change under a
    permutation within tasks. Tool-error runs give no reading, so they are excluded."""
    values: dict[str, set[str]] = {}
    for r in cd.data.readings:
        values.setdefault(r.cell, set()).add(r.value)
    return frozenset(c for c, v in values.items() if len(v) > 1)


def _defect_cells(found: Iterable[CellFinding], changeable: frozenset[str]) -> int:
    """The changeable cells carrying at least one defect type, each counted once (P-C)."""
    return len({f.cell for f in found if f.type in DEFECT_TYPES_V4 and f.cell in changeable})


def _count_v4(cd: S3CellData, models: Mapping[str, str], params: V4Params,
              changeable: frozenset[str], fit: AnchoredFitV4 | None = None) -> int:
    data = cd.data
    if fit is None:
        fit = fit_v4(data.cells, data.readings, models=models, params=params)
    return _defect_cells(type_cells_v4(data.cells, fit, data.by_cell(), params), changeable)


def _count_v3(cd: S3CellData, models: Mapping[str, str], params: V3Params,
              changeable: frozenset[str]) -> int:
    data = cd.data
    fit = fit_anchored(data.cells, data.readings, models=models, params=params)
    return _defect_cells(type_cells(data.cells, fit, data.by_cell(), params), changeable)


def _ratio(counts: Sequence[int], real: int) -> float:
    return float(np.mean(counts)) / real if real and counts else math.nan


def _validate(domain: str, tasks: Sequence[S3Task], runs: Sequence[S3Run]) -> None:
    """Every task and run belongs to `domain`, and every run's task is among `tasks`."""
    for kind, items in (("task", tasks), ("run", runs)):
        other = sorted({x.domain for x in items} - {domain})
        if other:
            raise ValueError(f"placebo of {domain}: {kind}s of other domains: {other}")
    unknown = sorted({r.task for r in runs} - {t.index for t in tasks})
    if unknown:
        raise ValueError(f"placebo of {domain}: runs of unknown tasks {unknown[:10]}")


def placebo(
    domain: str, tasks: Sequence[S3Task], runs: Sequence[S3Run], models: Mapping[str, str],
    params_v4: V4Params, params_v3: V3Params, seeds: Sequence[int] = range(20),
    real_fit: AnchoredFitV4 | None = None,
) -> PlaceboResult:
    """Count the changeable cells carrying a hidden, ambiguity or weak finding, on the real
    data and on each placebo (everything refitted, α included), for v4 and v3 (ruling P-C).
    The changeable set comes from the real data; a permutation within tasks preserves it,
    and this is checked for every seed. The witnesses are not used (they speak to task
    labels, not to the gradient). real_fit: v4's fit of the real runs, when the caller has it
    (the s3-placebo stage does), so the real data is not fitted twice."""
    _validate(domain, tasks, runs)
    real = cells_for(tasks, runs)
    changeable = changeable_cells(real)
    real_v4 = _count_v4(real, models, params_v4, changeable, real_fit)
    real_v3 = _count_v3(real, models, params_v3, changeable)
    p4: list[int] = []
    p3: list[int] = []
    for seed in seeds:
        cd = cells_for(tasks, permute_within_tasks(runs, seed))
        if changeable_cells(cd) != changeable:
            raise AssertionError(f"placebo of {domain}, seed {seed}: the permutation changed "
                                 "the changeable cells")
        p4.append(_count_v4(cd, models, params_v4, changeable))
        p3.append(_count_v3(cd, models, params_v3, changeable))
    return PlaceboResult(domain, real_v4, real_v3, p4, p3, _ratio(p4, real_v4),
                         _ratio(p3, real_v3), len(changeable))
