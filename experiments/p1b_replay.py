"""P1b offline replay (spec v2 §6): the v2 inference on the P1 run cache, before any new run.

Read-only. It never calls Ollama, never executes a run, never opens the database and never
writes under SPECDX_WORK. It reads the run cache (runs.jsonl), the mutant packs (mutants/)
and the P1 gate's per-mutant lines (results/p1-gate-mutants.jsonl), and prints Markdown on
stdout (or writes it to --out). Only the base and the 20 development mutants are used: the
old population (qwen3:14b, qwen2.5:7b-instruct-q4_K_M, llama3.1:8b) and its 200
population runs per specification.

How cached runs are mapped to a specification: as experiments/p1_dev_diagnostics.py does,
by recomputing Runner.run's key, run_key(variant digest, steps digest, context digest,
scenario parameters, interpreter key, suffix "", model digest). The context digest is
pinned to what P1 used (RUN_SCHEMA_VERSION 2 and P1's executor settings), so later P1b
changes cannot break the mapping. A specification with other than 200 mapped runs stops
the script.

No refutation is attempted (no run may execute), so a singleton candidate types hidden
when Hidden(c) ≥ τ_H (spec §11 item 13) and a larger set is dropped. Observability, θ, κ
and S2 do not depend on refutation.

The GO criterion is pre-registered in the P1b plan (Task 7) and in
experiments/p1b-preregistration.md as a mechanism check: (a) on the base and on every
development mutant, llama3.1:8b, which never writes, has the lowest κ; (b) every model with
an acted behaviour keeps θ ≥ 0.5; (c) at least half of the P1 base findings become
Unobservable. A θ ordering is not judged here (one writing family; Proposition 2′).

Usage: SPECDX_WORK=/path/to/work .venv/bin/python experiments/p1b_replay.py [--out PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tonomyx.canon.pack import CanonPack

from specdx.config import REPO_ROOT, WORK_DIR
from specdx.eval.pilot import GateConfig
from specdx.infer.act import Act, Rasch, act_rate, fit_rasch
from specdx.infer.items import ItemView, SpecLens, acted_violations, items_reading, observations
from specdx.infer.latent import Posterior, fit
from specdx.infer.scores import Diagnosis, diagnose
from specdx.interpreters.cache import run_key
from specdx.interpreters.population import committee
from specdx.mutate.operators import A1_SITES, A2_SITES
from specdx.observe.run import context_digest
from specdx.scenarios.space import boundary_domain, canon_scenarios, enumerate_scenarios
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InterpreterId, RefuteResult, Scenario
from specdx.variant import SpecVariant
from specdx.workbench import Workbench, tell_payload

RUNS = WORK_DIR / "runs.jsonl"
GATE_LINES = REPO_ROOT / "results" / "p1-gate-mutants.jsonl"
GATE_CONFIG = REPO_ROOT / "experiments" / "p1-gate.json"
P1_SCHEMA = 2
P1_SETTINGS: dict[str, float | int] = {
    "temperature": 0.7, "max_steps": 16, "timeout": 180.0, "num_ctx": 16384, "search_k": 5,
    "search_reply_limit": 12000,
}
POPULATION_RUNS = 200
LLAMA = "llama3.1:8b"
THETA_FLOOR = 0.5  # (b): P1 collapsed qwen3:14b to θ 0.003
UNOBSERVABLE_SHARE = 0.5  # (c)


def _never(*_: Any, **__: Any) -> Any:
    raise RuntimeError("the replay never executes a run")


class NoRefutation:
    """Every candidate stays unattempted: no run may execute here."""

    def refute(
        self, targets: frozenset[str], scenario: Scenario, behaviours: Sequence[Behaviour]
    ) -> RefuteResult:
        return RefuteResult(False, None, 0, attempted=False)


@dataclass
class Replay:
    spec: str
    lens: SpecLens
    behaviours: list[Behaviour]
    views: list[ItemView]
    acts: list[Act]
    rasch: Rasch
    posterior: Posterior
    diagnosis: Diagnosis


def load_cache(path: Path = RUNS) -> dict[str, Behaviour]:
    cache: dict[str, Behaviour] = {}
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                b = Behaviour.from_json(raw.decode("utf-8"))
                cache[b.run_key] = b
    return cache


def model_digests(cache: dict[str, Behaviour]) -> dict[str, str]:
    seen: dict[str, set[str]] = {}
    for b in cache.values():
        seen.setdefault(b.interpreter.model, set()).add(b.model_digest)
    if any(len(v) != 1 for v in seen.values()):
        raise SystemExit(f"a model has several digests in the cache: {seen}")
    return {m: next(iter(v)) for m, v in seen.items()}


def variant_of(bench: Workbench, sid: str) -> SpecVariant:
    """The SpecVariant all_mutants built for this id, rebuilt read-only from its pack."""
    if sid == "base":
        return bench.base
    pack = WORK_DIR / "mutants" / sid / bench.base.pack.name
    corpus = TellCorpus.build(CanonPack(pack), bench.pdf)
    op, index = sid.split("-")
    if op == "A1":
        corpus = corpus.without(corpus.find(A1_SITES[int(index)]))
    elif op == "A2":
        old, new = A2_SITES[int(index)]
        for sentence in corpus.find(old):
            corpus = corpus.replace(sentence, new)
    return SpecVariant(sid, pack, corpus)


def population_runs(
    bench: Workbench, variant: SpecVariant, members: Sequence[InterpreterId],
    cache: dict[str, Behaviour], digests: dict[str, str], context: str,
) -> list[Behaviour]:
    pack = CanonPack(variant.pack)
    # One scenario per parameter vector; a canon name wins over a synthesised id.
    by_signature = {s.signature(): s for s in enumerate_scenarios(boundary_domain(pack))}
    for s in canon_scenarios(pack).values():
        by_signature[s.signature()] = s
    digest = variant.digest()
    steps_digest = bench.runner._steps_digest
    out: list[Behaviour] = []
    for _, scenario in sorted(by_signature.items()):
        params = json.dumps(scenario.params(), sort_keys=True)
        for interp in members:
            key = run_key(digest, steps_digest, context, params, interp.key, "",
                          digests[interp.model])
            b = cache.get(key)
            if b is not None:
                out.append(dataclasses.replace(b, scenario=scenario, interpreter=interp))
    if len(out) != POPULATION_RUNS:
        raise SystemExit(f"{variant.name}: {len(out)} population runs mapped, "
                         f"expected {POPULATION_RUNS}; the mapping is broken")
    return out


def replay(bench: Workbench, sid: str, config: GateConfig, cache: dict[str, Behaviour],
           digests: dict[str, str], context: str) -> Replay:
    cfg = config.specdx()
    variant = variant_of(bench, sid)
    members = committee(config.models_by_family, config.variants, config.seeds)
    behaviours = population_runs(bench, variant, members, cache, digests, context)
    lens = bench.loop_context(variant, cfg).lens
    assert isinstance(lens, SpecLens)
    views: list[ItemView] = []
    acts: list[Act] = []
    violated: dict[str, frozenset[str]] = {}
    for b in behaviours:
        mine = lens.views(b, lens.violations(b))
        violated[b.run_key] = acted_violations(mine)
        views.extend(mine)
        acts.extend(lens.acts(b))
    rasch = fit_rasch(acts)
    population = {m: f for f, ms in config.models_by_family.items() for m in ms}
    posterior = fit(observations(views), models=population, k_max=cfg.k_max,
                    p_single=cfg.p_single)
    diagnosis = diagnose(behaviours, violated, views, posterior, NoRefutation(), lens.diff,
                         observable=rasch.observable_items(), relevant=lens.relevant,
                         tau=cfg.tau, rng_seed=cfg.rng_seed)
    return Replay(sid, lens, behaviours, views, acts, rasch, posterior, diagnosis)


def go(r: Replay) -> tuple[bool, str]:
    """Criteria (a) and (b) on one specification: stage 1 puts the model that never writes
    lowest in κ, and no model with acted evidence collapses below THETA_FLOOR."""
    theta, kappa = r.posterior.theta, r.rasch.kappa
    first = LLAMA in kappa and all(kappa[LLAMA] < kappa[m] for m in kappa if m != LLAMA)
    acted = sorted({a.model for a in r.acts if a.acted})
    low = [m for m in acted if theta.get(m, 0.0) < THETA_FLOOR]
    second = bool(acted) and not low
    why = (f"κ: {', '.join(f'{m} {kappa[m]:.3f}' for m in sorted(kappa))}; "
           f"acting models below θ {THETA_FLOOR}: {', '.join(low) or 'none'}")
    return first and second, why


def p1_status(r: Replay, defect: dict[str, Any]) -> str:
    """'observable' or 'unobservable' for one P1 base finding under v2's observability."""
    observable = r.rasch.observable_items()
    if defect["type"] in ("ambiguity", "weak"):
        return "observable" if defect["note"] in observable else "unobservable"
    for sid in defect["scenarios"]:
        if all(any(i in observable for i in items_reading(c, sid, r.lens.relevant))
               for c in defect["location"]):
            return "observable"
    return "unobservable"


def unobservable_share(statuses: Sequence[str]) -> float:
    """Criterion (c): the share of P1 base findings v2 reports as Unobservable."""
    return sum(s == "unobservable" for s in statuses) / len(statuses) if statuses else 0.0


def verdict(replays: list[Replay], base_line: dict[str, Any]) -> bool:
    """GO: (a) and (b) on every specification, and (c) on the base."""
    statuses = [p1_status(replays[0], d) for d in base_line["defects"]["specdx"]]
    return (all(go(r)[0] for r in replays)
            and unobservable_share(statuses) >= UNOBSERVABLE_SHARE)


def _f(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def render(replays: list[Replay], base_line: dict[str, Any]) -> str:
    out: list[str] = ["<!-- generated by experiments/p1b_replay.py -->", "",
                      "# P1b offline replay: v2 inference on the P1 cache", ""]
    verdicts = {r.spec: go(r) for r in replays}
    passed = verdict(replays, base_line)
    share = unobservable_share([p1_status(replays[0], d)
                                for d in base_line["defects"]["specdx"]])
    out += [f"## Verdict: {'GO' if passed else 'NO-GO'}", "",
            "Pre-registered mechanism check (author's ruling, 2026-10-12). "
            f"(a) {LLAMA}, which never writes, has the lowest κ on every specification. "
            f"(b) Every model with an acted behaviour has θ ≥ {THETA_FLOOR}. "
            f"(c) At least {UNOBSERVABLE_SHARE:.0%} of the P1 base findings become "
            f"Unobservable: here {share:.1%}.", "",
            "| specification | (a) and (b) hold | detail |", "|---|---|---|"]
    out += [f"| {sid} | {'yes' if ok else 'no'} | {why} |" for sid, (ok, why) in
            verdicts.items()]
    base = replays[0]
    rates = act_rate(base.acts)
    out += ["", "## Base: stage 1 and stage 2 per model", "",
            "| model | act rate | κ | θ (v2) |", "|---|---:|---:|---:|"]
    for m in sorted(base.posterior.theta):
        out.append(f"| {m} | {rates.get(m, 0.0):.3f} | {_f(base.rasch.kappa.get(m))} | "
                   f"{base.posterior.theta[m]:.3f} |")
    statuses = Counter((d["type"], p1_status(base, d)) for d in base_line["defects"]["specdx"])
    total_unobs = sum(n for (_, s), n in statuses.items() if s == "unobservable")
    out += ["", "## P1 base findings under v2 observability", "",
            f"{total_unobs} of {len(base_line['defects']['specdx'])} P1 base findings become "
            "Unobservable.", "", "| P1 type | observable | unobservable |", "|---|---:|---:|"]
    for t in ("contradiction", "hidden", "ambiguity", "weak"):
        out.append(f"| {t} | {statuses[(t, 'observable')]} | {statuses[(t, 'unobservable')]} |")
    out += ["", "## v2 diagnosis without refutation, per specification", "",
            "| specification | observable items | typed (C/H/A/W) | unobservable | S2 tested | "
            "S2 significant |", "|---|---:|---|---:|---:|---:|"]
    for r in replays:
        kinds = Counter(d.type for d in r.diagnosis.defects)
        tested = [s for s in r.diagnosis.signals.values() if s.p is not None]
        out.append(
            f"| {r.spec} | {len(r.rasch.observable_items())} of {len(r.rasch.delta)} | "
            f"{kinds['contradiction']}/{kinds['hidden']}/{kinds['ambiguity']}/{kinds['weak']} | "
            f"{len(r.diagnosis.unobservable)} | {len(tested)} | "
            f"{sum(s.significant for s in tested)} |")
    out += ["", "## Base: per item", "",
            "| item | δ | O_s | acted | observable | S1 | S2 | p | BH |",
            "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for item in sorted(base.rasch.delta):
        sig = base.diagnosis.signals.get(item)
        out.append(
            f"| {item} | {base.rasch.delta[item]:.3f} | {base.rasch.observability[item]:.3f} | "
            f"{base.rasch.acted[item]} | {'yes' if base.rasch.observable(item) else 'no'} | "
            f"{_f(sig.s1 if sig else None)} | {_f(sig.s2 if sig else None)} | "
            f"{_f(sig.p if sig else None)} | {'yes' if sig and sig.significant else ''} |")
    out += ["", "## S2 p-values on the mutants (tested items)", "",
            "| specification | item | S2 | p | BH |", "|---|---|---:|---:|---|"]
    for r in replays[1:]:
        for item, s in sorted(r.diagnosis.signals.items()):
            if s.p is not None:
                out.append(f"| {r.spec} | {item} | {_f(s.s2)} | {_f(s.p)} | "
                           f"{'yes' if s.significant else ''} |")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the Markdown here instead of stdout")
    args = parser.parse_args(argv)
    cache = load_cache()
    digests = model_digests(cache)
    bench = Workbench.open(factory=_never, model_digest=digests.__getitem__)
    context = context_digest(tell_payload(bench.variants), P1_SETTINGS, schema=P1_SCHEMA)
    config = GateConfig.load(GATE_CONFIG)
    lines = [json.loads(x) for x in GATE_LINES.read_text(encoding="utf-8").splitlines()]
    ids = ["base"] + [ln["mutant"] for ln in lines if not ln.get("base")]
    replays = [replay(bench, sid, config, cache, digests, context) for sid in ids]
    text = render(replays, lines[0])
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if verdict(replays, lines[0]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
