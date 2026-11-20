"""Permutation placebo for the ability-gradient mechanism (design §6.2).

Within each task the raters' whole act vectors are shuffled across the raters. That keeps
every task's multiset of readings but cuts the link between a reading and its reader's
ability, so v4's findings, which come from that link, should mostly vanish. v3's findings
are driven by majorities and should not.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from specdx.infer.anchored import V3Params, fit_anchored
from specdx.infer.anchored_v4 import V4Params, fit_v4
from specdx.infer.typing import type_cells
from specdx.infer.typing_v4 import DEFECT_TYPES_V4, type_cells_v4
from specdx.subjects.sopbench.cells import S3CellData, cells_for
from specdx.subjects.sopbench.load import S3Run, S3Task

ACT_FIELDS = ("succeeded", "db_match", "constraints", "process", "tool_error")


@dataclass(frozen=True)
class PlaceboResult:
    """real_*: typed defects on changeable cells of the real runs; placebo_*: the same count
    per seed; ratio_*: mean placebo count over the real count (nan when that is 0);
    changeable: the number of changeable cells."""

    domain: str
    real_v4: int
    real_v3: int
    placebo_v4: list[int]
    placebo_v3: list[int]
    ratio_v4: float
    ratio_v3: float
    changeable: int


def permute_within_tasks(runs: Sequence[S3Run], seed: int) -> list[S3Run]:
    """Shuffle each task's act vectors (succeeded, db_match, constraints, process,
    tool_error) across its runs. Each run keeps its model, family, mode and prompt, and the
    output keeps the input order. Deterministic in (seed, task)."""
    out = list(runs)
    by_task: dict[tuple[str, int], list[int]] = {}
    for i, r in enumerate(runs):
        by_task.setdefault((r.domain, r.task), []).append(i)
    for (_, task), idx in sorted(by_task.items()):
        perm = np.random.default_rng([seed, task]).permutation(len(idx))
        for pos, j in zip(idx, perm, strict=True):
            src = runs[idx[j]]
            out[pos] = replace(runs[pos], **{f: getattr(src, f) for f in ACT_FIELDS})
    return out


def changeable_cells(cd: S3CellData) -> frozenset[str]:
    """The cells whose non-excluded readings are not all equal: only these can change under
    a permutation within tasks. Tool-error runs give no reading, so they are excluded."""
    values: dict[str, set[str]] = {}
    for r in cd.data.readings:
        values.setdefault(r.cell, set()).add(r.value)
    return frozenset(c for c, v in values.items() if len(v) > 1)


def _count_v4(cd: S3CellData, models: Mapping[str, str], params: V4Params,
              changeable: frozenset[str]) -> int:
    data = cd.data
    fit = fit_v4(data.cells, data.readings, models=models, params=params)
    return sum(1 for f in type_cells_v4(data.cells, fit, data.by_cell(), params)
               if f.type in DEFECT_TYPES_V4 and f.cell in changeable)


def _count_v3(cd: S3CellData, models: Mapping[str, str], params: V3Params,
              changeable: frozenset[str]) -> int:
    data = cd.data
    fit = fit_anchored(data.cells, data.readings, models=models, params=params)
    return sum(1 for f in type_cells(data.cells, fit, data.by_cell(), params)
               if f.type in DEFECT_TYPES_V4 and f.cell in changeable)


def _ratio(counts: Sequence[int], real: int) -> float:
    return float(np.mean(counts)) / real if real and counts else math.nan


def placebo(
    domain: str, tasks: Sequence[S3Task], runs: Sequence[S3Run], models: Mapping[str, str],
    params_v4: V4Params, params_v3: V3Params, seeds: Sequence[int] = range(20),
) -> PlaceboResult:
    """Count the hidden, ambiguity and weak findings on changeable cells of the real data
    and of each placebo (everything refitted, α included), for v4 and v3. The changeable
    set comes from the real data: a permutation within tasks preserves it. The witnesses
    are not used (they speak to task labels, not to the gradient)."""
    real = cells_for(tasks, runs)
    changeable = changeable_cells(real)
    real_v4 = _count_v4(real, models, params_v4, changeable)
    real_v3 = _count_v3(real, models, params_v3, changeable)
    p4: list[int] = []
    p3: list[int] = []
    for seed in seeds:
        cd = cells_for(tasks, permute_within_tasks(runs, seed))
        p4.append(_count_v4(cd, models, params_v4, changeable))
        p3.append(_count_v3(cd, models, params_v3, changeable))
    return PlaceboResult(domain, real_v4, real_v3, p4, p3, _ratio(p4, real_v4),
                         _ratio(p3, real_v3), len(changeable))
