"""The S3 (SOPBench) calibrator and the seeded domain split (P4a design §6.1).

split_domains draws the 3 development and 4 held-out domains once; its output is committed
as experiments/p4-s3-split.json. calibrate_s3 turns the published runs of the development
domains into a SimConfig: the raters' abilities are the logits of their accuracy against the
anchors, the dispersion of cell accuracy is beta_sd, the cells are binary and read four to a
run (act_group 4). It reads published runs only: no network, no model, no database. The
pure statistics are those of specdx.sim.calibrate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from specdx.config import REPO_ROOT
from specdx.sim.calibrate import CalibrationError, _logit
from specdx.sim.generator import SimConfig
from specdx.subjects.sopbench import DOMAINS
from specdx.subjects.sopbench.cells import S3CellData
from specdx.subjects.sopbench.load import S3Run

SPLIT_SEED = 20261120
SPLIT_FILE = REPO_ROOT / "experiments" / "p4-s3-split.json"
DEFECT_RATE = 0.15  # v3's planted rate, unchanged
SPREAD_FLOOR = 0.25  # the smallest rater spread (logit sd) taken at face value


def split_domains(seed: int = SPLIT_SEED) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(dev, held_out): 3 development and 4 held-out domains, each in DOMAINS order."""
    order = np.random.default_rng(seed).permutation(len(DOMAINS))
    dev = {int(i) for i in order[:3]}
    return (tuple(d for i, d in enumerate(DOMAINS) if i in dev),
            tuple(d for i, d in enumerate(DOMAINS) if i not in dev))


def load_split(path: Path = SPLIT_FILE) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = json.loads(path.read_text())
    return tuple(raw["dev"]), tuple(raw["held_out"])


def calibrate_s3(cds: Sequence[S3CellData], runs: Sequence[S3Run],
                 dev: Sequence[str] | None = None) -> SimConfig:
    """The SimConfig of the development domains (default: the committed split's). Cells and
    runs of other domains are ignored."""
    domains = set(load_split()[0] if dev is None else dev)
    keep = tuple(f"s3:{d}:" for d in sorted(domains))
    runs = [r for r in runs if r.domain in domains]
    hits: dict[str, list[int]] = {}
    cell_logits: list[float] = []
    for cd in cds:
        by_cell = cd.data.by_cell()
        for cid, cell in cd.data.cells.items():
            rs = by_cell.get(cid, [])
            if cell.anchor is None or not rs or not cid.startswith(keep):
                continue
            values = Counter(r.value for r in rs)
            if min(values, key=lambda v: (-values[v], v)) != cell.anchor:
                continue  # the anchor is not the plurality: no evidence of accuracy
            for r in rs:
                h = hits.setdefault(r.model, [0, 0])
                h[0] += r.value == cell.anchor
                h[1] += 1
            n_hit = sum(r.value == cell.anchor for r in rs)
            cell_logits.append(_logit((n_hit + 0.5) / (len(rs) + 1)))
    if not hits or not cell_logits:
        raise CalibrationError("no anchored reading in the development domains")
    models = sorted(hits)
    alphas = tuple(_logit((hits[m][0] + 0.5) / (hits[m][1] + 1)) for m in models)
    made: Counter[str] = Counter()
    clean: Counter[str] = Counter()
    for r in runs:
        made[r.model] += 1
        clean[r.model] += not r.tool_error
    act = tuple(clean[m] / made[m] if made[m] else 1.0 for m in models)
    spread = float(np.std(alphas))
    # A near-homogeneous panel is widened to the floor; otherwise the logits stand as given.
    alpha_spread = SPREAD_FLOOR / spread if 0.0 < spread < SPREAD_FLOOR else 1.0
    n_cells = sum(1 for cd in cds for cid in cd.data.cells if cid.startswith(keep))
    tasks = max(1, n_cells // 4)
    return SimConfig(
        alphas=alphas, act=act, columns=4 * tasks, items=tasks, pooled_share=0.0,
        scenarios=1, variants=1, seeds=1, anchored_share=1.0, constraint_share=0.75,
        beta_sd=float(min(max(np.std(cell_logits), 0.25), 3.0)),
        values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), defect_rate=DEFECT_RATE,
        act_group=4, alpha_spread=alpha_spread)

