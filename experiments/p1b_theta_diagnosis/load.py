# ruff: noqa
"""Read-only loader: smoke population (P1b) and P1 base population, with lenses.

No database, no Ollama, no writes outside the scratchpad. Pickles the views to the scratchpad.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
os.environ["SPECDX_WORK"] = str(SCRATCH / "work")  # nothing defaults to the real work dir
RUNS = Path("/home/ethan/AILab/SpecDx/work/runs.jsonl")

from tonomyx.canon.pack import CanonPack  # noqa: E402

from specdx.config import BASE_PACK, PDF_DIR  # noqa: E402
from specdx.db import golden_tables, split_background  # noqa: E402
from specdx.infer.act import drafting_steps  # noqa: E402
from specdx.infer.items import SpecLens  # noqa: E402
from specdx.interpreters.cache import run_key  # noqa: E402
from specdx.interpreters.population import committee  # noqa: E402
from specdx.interpreters.prompt import load_variants  # noqa: E402
from specdx.observe.run import context_digest  # noqa: E402
from specdx.orchestrate.work_orders import load_work_orders  # noqa: E402
from specdx.scenarios.space import boundary_domain, canon_scenarios, enumerate_scenarios  # noqa
from specdx.spec.clauses import ClauseSet  # noqa: E402
from specdx.spec.columns import ColumnModel  # noqa: E402
from specdx.spec.tell import TellCorpus, pdf_chunks  # noqa: E402
from specdx.types import Behaviour, InterpreterId  # noqa: E402
from specdx.variant import SpecVariant  # noqa: E402
from specdx.workbench import tell_payload  # noqa: E402
import dataclasses  # noqa: E402

P1_SETTINGS = {"temperature": 0.7, "max_steps": 16, "timeout": 180.0, "num_ctx": 16384,
               "search_k": 5, "search_reply_limit": 12000}


def read_lines():
    out = []
    with RUNS.open("rb") as h:
        for n, raw in enumerate(h, start=1):
            if raw.strip():
                out.append((n, Behaviour.from_json(raw.decode("utf-8"))))
    return out


def lens_for_base(steps):
    pack = CanonPack(BASE_PACK)
    clauses = ClauseSet.from_pack(pack, steps)
    background, _ = split_background(golden_tables(pack))
    return SpecLens(clauses, steps, ColumnModel.of(pack),
                    drafting=drafting_steps(pack, steps), initial=background), pack


def main():
    lines = read_lines()
    by_key = {b.run_key: (n, b) for n, b in lines}
    steps = tuple(load_work_orders())
    lens, pack = lens_for_base(steps)

    # Smoke population: committee members (seeds 1-3) on the four scenarios, base spec,
    # keyed under the current schema. Compute keys exactly as Runner.key does.
    pdf = pdf_chunks(PDF_DIR, SCRATCH / "work" / "tell")
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, pdf))
    steps_digest = hashlib.sha256(repr(steps).encode()).hexdigest()
    from specdx.interpreters.executor import interpreter_settings
    variants = load_variants()
    ctx_now = context_digest(tell_payload(variants), interpreter_settings())
    digests = {}
    for _, b in lines:
        digests.setdefault(b.interpreter.model, set()).add(b.model_digest)
    smoke_pop = {"openai": ("gpt-oss:20b",), "qwen": ("qwen3:30b-a3b",)}
    members = committee(smoke_pop, (0, 1, 2), (1, 2, 3))
    # scenarios: canon + boundary + T-boundary, by id
    from specdx.scenarios.tguided import extract_literals, tboundary_scenarios
    canon = canon_scenarios(pack)
    literals = extract_literals(base.corpus)
    cands = enumerate_scenarios(boundary_domain(pack, [x.bound for x in literals]))
    tb = tboundary_scenarios(canon, literals)
    by_id = {s.id: s for s in [*canon.values(), *cands, *tb]}
    smoke = []
    for sid in ["discount8", "grn12", "happy", "o15-r12-d4-g1"]:
        sc = by_id[sid]
        params = json.dumps(sc.params(), sort_keys=True)
        for i in members:
            ds = digests[i.model]
            assert len(ds) == 1, (i.model, ds)
            k = run_key(base.digest(), steps_digest, ctx_now, params, i.key, "", next(iter(ds)))
            if k in by_key:
                n, b = by_key[k]
                smoke.append((n, dataclasses.replace(b, scenario=sc, interpreter=i)))
    print("smoke population runs mapped:", len(smoke), file=sys.stderr)
    print("line ranges:", min(n for n, _ in smoke), max(n for n, _ in smoke), file=sys.stderr)

    # P1 base population, as experiments/p1b_replay.py maps it.
    ctx_p1 = context_digest(tell_payload(variants), P1_SETTINGS, schema=2)
    p1_pop = {"qwen": ("qwen3:14b", "qwen2.5:7b-instruct-q4_K_M"), "llama": ("llama3.1:8b",)}
    p1_members = committee(p1_pop, (0, 1, 2), (1, 2, 3))
    by_sig = {s.signature(): s for s in enumerate_scenarios(boundary_domain(pack))}
    for s in canon_scenarios(pack).values():
        by_sig[s.signature()] = s
    p1 = []
    for _, sc in sorted(by_sig.items()):
        params = json.dumps(sc.params(), sort_keys=True)
        for i in p1_members:
            ds = digests[i.model]
            assert len(ds) == 1, (i.model, ds)
            k = run_key(base.digest(), steps_digest, ctx_p1, params, i.key, "", next(iter(ds)))
            if k in by_key:
                n, b = by_key[k]
                p1.append((n, dataclasses.replace(b, scenario=sc, interpreter=i)))
    print("P1 base population runs mapped:", len(p1), file=sys.stderr)

    with (SCRATCH / "data.pkl").open("wb") as h:
        pickle.dump({"smoke": smoke, "p1": p1, "steps": steps}, h)


if __name__ == "__main__":
    main()
