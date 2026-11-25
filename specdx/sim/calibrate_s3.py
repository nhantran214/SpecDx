"""The S3 (SOPBench) calibrator and the seeded domain split (P4a design §6.1; ruling F-C1).

split_domains draws the 3 development and 4 held-out domains once; its output is committed
as experiments/p4-s3-split.json. calibrate_s3 turns the published runs of each development
domain into a SimConfig of its own, in the S3 layout (SimConfig.goals): that domain's raters,
its tasks per user goal, and its difficulty spreads, between groups (beta_sd) and within them
(beta_within). The raters' abilities are the logits of their accuracy against the anchors;
cells are binary and read four to a run (act_group 4). G0v4 fits each simulated domain
separately, as the real fits are per domain. It reads published runs only: no network, no
model, no database.

Accuracy is measured on the cells whose plurality reading is the anchor (as v3's calibrator
does): a cell where most raters deviate is set aside as possibly defective, which biases the
abilities and the cell accuracies upward. The bias is kept for parity with v3 and is the same
for every candidate G0v4 compares.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from specdx.config import REPO_ROOT
from specdx.sim.calibrate import CalibrationError, logit
from specdx.sim.generator import SimConfig
from specdx.subjects.sopbench import DOMAINS
from specdx.subjects.sopbench.cells import S3CellData
from specdx.subjects.sopbench.load import S3Run

SPLIT_SEED = 20261120
SPLIT_FILE = REPO_ROOT / "experiments" / "p4-s3-split.json"
DEFECT_RATE = 0.15  # v3's planted rate, unchanged
SPREAD_FLOOR = 0.25  # the smallest rater spread (logit sd) taken at face value
BETA_SD_RANGE = (0.25, 3.0)  # the group spread is clipped to this range, as in v3
BETA_WITHIN_MAX = 3.0


def split_domains(seed: int = SPLIT_SEED) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(dev, held_out): 3 development and 4 held-out domains, each in DOMAINS order."""
    order = np.random.default_rng(seed).permutation(len(DOMAINS))
    dev = {int(i) for i in order[:3]}
    return (tuple(d for i, d in enumerate(DOMAINS) if i in dev),
            tuple(d for i, d in enumerate(DOMAINS) if i not in dev))


def load_split(path: Path = SPLIT_FILE) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(raw["dev"]), tuple(raw["held_out"])


def calibrate_s3(cds: Sequence[S3CellData], runs: Sequence[S3Run],
                 dev: Sequence[str] | None = None) -> tuple[SimConfig, ...]:
    """One SimConfig per development domain (default: the committed split's), in that order.
    Cells and runs of other domains are ignored."""
    domains = tuple(load_split()[0] if dev is None else dev)
    return tuple(_calibrate_domain(d, cds, runs) for d in domains)


def _spreads(logits: dict[str, list[float]]) -> tuple[float, float]:
    """(between, within): the sd of the group means, and the pooled sd of the cells about
    their group's mean (0 when no group has two cells)."""
    means = [float(np.mean(v)) for v in logits.values()]
    between = float(np.std(means)) if means else 0.0
    dof = sum(len(v) - 1 for v in logits.values())
    ss = sum(float(((np.array(v) - np.mean(v)) ** 2).sum()) for v in logits.values())
    return between, math.sqrt(ss / dof) if dof > 0 else 0.0


def _calibrate_domain(domain: str, cds: Sequence[S3CellData],
                      runs: Sequence[S3Run]) -> SimConfig:
    prefix = f"s3:{domain}:"
    hits: dict[str, list[int]] = {}
    logits: dict[str, list[float]] = {}
    goals: Counter[str] = Counter()
    for cd in cds:
        by_cell = cd.data.by_cell()
        for cid, cell in cd.data.cells.items():
            if not cid.startswith(prefix):
                continue
            if cid.endswith(":outcome"):
                goals[cell.group] += 1
            rs = by_cell.get(cid, [])
            if cell.anchor is None or not rs:
                continue
            values = Counter(r.value for r in rs)
            if min(values, key=lambda v: (-values[v], v)) != cell.anchor:
                continue  # the anchor is not the plurality: no evidence of accuracy
            for r in rs:
                h = hits.setdefault(r.model, [0, 0])
                h[0] += r.value == cell.anchor
                h[1] += 1
            n_hit = sum(r.value == cell.anchor for r in rs)
            logits.setdefault(cell.group, []).append(logit((n_hit + 0.5) / (len(rs) + 1)))
    if not hits or not goals:
        raise CalibrationError(f"no anchored reading in the development domain {domain}")
    models = sorted(hits)
    alphas = tuple(logit((hits[m][0] + 0.5) / (hits[m][1] + 1)) for m in models)
    made: Counter[str] = Counter()
    clean: Counter[str] = Counter()
    for r in runs:
        if r.domain == domain:
            made[r.model] += 1
            clean[r.model] += not r.tool_error
    act = tuple(clean[m] / made[m] if made[m] else 1.0 for m in models)
    spread = float(np.std(alphas))
    # A near-homogeneous panel is widened to the floor; otherwise the logits stand as given.
    alpha_spread = SPREAD_FLOOR / spread if 0.0 < spread < SPREAD_FLOOR else 1.0
    between, within = _spreads(logits)
    sizes = tuple(goals[g] for g in sorted(goals))
    tasks = sum(sizes)
    return SimConfig(
        alphas=alphas, act=act, columns=4 * tasks, items=tasks, pooled_share=0.0,
        scenarios=1, variants=1, seeds=1, anchored_share=1.0, constraint_share=0.75,
        beta_sd=float(min(max(between, BETA_SD_RANGE[0]), BETA_SD_RANGE[1])),
        beta_within=float(min(within, BETA_WITHIN_MAX)),
        values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), defect_rate=DEFECT_RATE,
        act_group=4, alpha_spread=alpha_spread, goals=sizes)
