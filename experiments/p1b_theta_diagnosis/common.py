# ruff: noqa
from __future__ import annotations

import os
import pickle
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
os.environ["SPECDX_WORK"] = str(SCRATCH / "work")

from tonomyx.canon.pack import CanonPack  # noqa: E402

from specdx.config import BASE_PACK  # noqa: E402
from specdx.db import golden_tables, split_background  # noqa: E402
from specdx.infer.act import drafting_steps, fit_rasch  # noqa: E402
from specdx.infer.items import SpecLens, acted_violations, observations  # noqa: E402
from specdx.infer.latent import fit  # noqa: E402
from specdx.spec.clauses import ClauseSet  # noqa: E402
from specdx.spec.columns import ColumnModel  # noqa: E402

SMOKE_POP = {"gpt-oss:20b": "openai", "qwen3:30b-a3b": "qwen"}
P1_POP = {"qwen3:14b": "qwen", "qwen2.5:7b-instruct-q4_K_M": "qwen", "llama3.1:8b": "llama"}


def load():
    with (SCRATCH / "data.pkl").open("rb") as h:
        return pickle.load(h)


def lens(steps):
    pack = CanonPack(BASE_PACK)
    clauses = ClauseSet.from_pack(pack, steps)
    background, _ = split_background(golden_tables(pack))
    return SpecLens(clauses, steps, ColumnModel.of(pack),
                    drafting=drafting_steps(pack, steps), initial=background), pack


def build(behaviours, ln, population):
    views, acts, violated = [], [], {}
    for b in behaviours:
        mine = ln.views(b, ln.violations(b))
        violated[b.run_key] = acted_violations(mine)
        views.extend(mine)
        acts.extend(ln.acts(b))
    rasch = fit_rasch(acts)
    post = fit(observations(views), models=population, k_max=6, p_single=0.9)
    return views, acts, violated, rasch, post
