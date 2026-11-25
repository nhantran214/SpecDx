"""The S3 (SOPBench) calibrator and the seeded domain split (P4a design §6.1; ruling F-C1).

split_domains draws the 3 development and 4 held-out domains once; its output is committed
as experiments/p4-s3-split.json. calibrate_s3 turns the published runs of each development
domain into a SimConfig of its own, in the S3 layout (SimConfig.goals): that domain's raters,
its tasks per user goal, and its difficulty spreads, between groups (beta_sd) and within them
(beta_within). The raters' abilities are the logits of their accuracy against the anchors;
cells are binary and read four to a run (act_group 4). G0v4 fits each simulated domain
separately, as the real fits are per domain. It reads published runs only: no network, no
model, no database.

Difficulty spreads (ruling R-5). A cell's logit x_c = logit(p̂_c), p̂_c = (hits + 0.5)/(n + 1),
spreads by binomial sampling as well as by difficulty: its sampling variance is about
v_c = 1/(n·p̂_c·(1 − p̂_c)), with p̂_c clipped to [0.01, 0.99] there (the +0.5 smoothing
already keeps it off 0 and 1). By the method of moments the within-group
variance is the pooled variance of the cells about their group means minus the mean v_c, and
the between-group variance is the variance of the group means minus the mean over groups of
(σ_w² + v̄_g)/n_g. Both sds are floored at SPREAD_FLOOR (pure sampling noise calibrates to the
floor) and capped at 3. Only cells with readings count, and a goal whose tasks have no
reading (every run a tool error) is dropped (ruling R-1): the real fit never sees its cells.

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
P_CLIP = 0.01  # the sampling variance's p is kept in [0.01, 0.99]
SPREAD_MAX = 3.0  # both difficulty spreads are capped here, as v3 capped its dispersion


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


def _spreads(logits: dict[str, list[tuple[float, float]]]) -> tuple[float, float]:
    """(between, within) difficulty sds from each group's (logit, sampling variance) pairs,
    with the binomial sampling variance subtracted (module docstring), floored at
    SPREAD_FLOOR and capped at SPREAD_MAX."""
    groups = [np.array(v) for v in logits.values() if v]
    dof = sum(len(g) - 1 for g in groups)
    ss = sum(float(((g[:, 0] - g[:, 0].mean()) ** 2).sum()) for g in groups)
    noise = float(np.mean(np.concatenate([g[:, 1] for g in groups]))) if groups else 0.0
    within_var = max(ss / dof - noise, 0.0) if dof > 0 else 0.0
    means = np.array([g[:, 0].mean() for g in groups])
    of_means = float(np.mean([(within_var + g[:, 1].mean()) / len(g) for g in groups])) \
        if groups else 0.0
    # Unbiased (ddof=1) variance of the group means; one group carries no between spread.
    between_var = max(float(np.var(means, ddof=1)) - of_means, 0.0) if len(groups) > 1 else 0.0

    def clip(var: float) -> float:
        return float(min(max(math.sqrt(var), SPREAD_FLOOR), SPREAD_MAX))

    return clip(between_var), clip(within_var)


def _calibrate_domain(domain: str, cds: Sequence[S3CellData],
                      runs: Sequence[S3Run]) -> SimConfig:
    prefix = f"s3:{domain}:"
    hits: dict[str, list[int]] = {}
    logits: dict[str, list[tuple[float, float]]] = {}
    goals: Counter[str] = Counter()
    for cd in cds:
        by_cell = cd.data.by_cell()
        for cid, cell in cd.data.cells.items():
            if not cid.startswith(prefix):
                continue
            rs = by_cell.get(cid, [])
            if cell.anchor is None or not rs:
                continue  # no reading: the real fit never sees this cell (ruling R-1)
            if cid.endswith(":outcome"):
                goals[cell.group] += 1
            values = Counter(r.value for r in rs)
            if min(values, key=lambda v: (-values[v], v)) != cell.anchor:
                continue  # the anchor is not the plurality: no evidence of accuracy
            for r in rs:
                h = hits.setdefault(r.model, [0, 0])
                h[0] += r.value == cell.anchor
                h[1] += 1
            n_hit = sum(r.value == cell.anchor for r in rs)
            p = (n_hit + 0.5) / (len(rs) + 1)
            q = min(max(p, P_CLIP), 1.0 - P_CLIP)
            logits.setdefault(cell.group, []).append(
                (logit(p), 1.0 / (len(rs) * q * (1.0 - q))))
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
    sizes = tuple(goals[g] for g in sorted(goals) if goals[g] > 0)
    tasks = sum(sizes)
    return SimConfig(
        alphas=alphas, act=act, columns=4 * tasks, items=tasks, pooled_share=0.0,
        scenarios=1, variants=1, seeds=1, anchored_share=1.0, constraint_share=0.75,
        beta_sd=between, beta_within=within,
        values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), defect_rate=DEFECT_RATE,
        act_group=4, alpha_spread=alpha_spread, goals=sizes)
