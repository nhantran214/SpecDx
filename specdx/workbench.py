"""Wires the database, the run cache, the Tell corpus, the runner and the loop context."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path

import httpx
from psycopg.conninfo import conninfo_to_dict
from tonomyx.canon.pack import CanonPack
from tonomyx.de.executor import DeExecutor
from tonomyx.tools.catalog import build_registry

from specdx.active.budget import RunBudget
from specdx.active.ddmin import OutOfBudget, localize_ambiguity
from specdx.active.loop import LoopContext, SpecDxConfig
from specdx.active.refute import CombinedRefuter, DirectedProbe, SymbolicRepair
from specdx.config import BASE_PACK, OLLAMA_URL, OLLAMA_URLS, PDF_DIR, WORK_DIR
from specdx.db import PackDatabase, golden_tables, split_background
from specdx.infer.act import drafting_steps
from specdx.infer.items import SpecLens
from specdx.interpreters.cache import RunCache
from specdx.interpreters.executor import SpecInterpreterExecutor, interpreter_settings
from specdx.interpreters.prompt import TellVariants, load_variants
from specdx.observe.run import ExecutorFactory, Runner, RunRequest, context_digest
from specdx.orchestrate.work_orders import WorkStep, load_work_orders
from specdx.parallel import ParallelRunner, ensure_database, worker_url
from specdx.scenarios.space import (
    boundary_domain,
    canon_scenarios,
    discount_cap_pct,
    enumerate_scenarios,
)
from specdx.scenarios.tguided import extract_literals, tboundary_scenarios
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import Chunk, TellCorpus, pdf_chunks
from specdx.types import Defect, InterpreterId, Scenario
from specdx.variant import SpecVariant


def ollama_digests(
    base_urls: str | Sequence[str] = OLLAMA_URLS, transport: httpx.BaseTransport | None = None
) -> Callable[[str], str]:
    """Model name → Ollama digest, fetched from every host on first use, so the run cache
    never mixes two builds of one model. An unknown model, an unreachable host, or hosts
    that disagree on a digest are errors."""
    hosts = [base_urls] if isinstance(base_urls, str) else list(base_urls)
    listed: dict[str, dict[str, str]] = {}

    def lookup(model: str) -> str:
        if not listed:
            with httpx.Client(transport=transport, timeout=10.0) as client:
                for host in hosts:
                    response = client.get(f"{host}/api/tags")
                    response.raise_for_status()
                    listed[host] = {str(m["name"]): str(m["digest"])
                                    for m in response.json().get("models", [])}
        name = model if ":" in model else f"{model}:latest"
        found = {host: listed[host].get(name) for host in hosts}
        absent = [host for host, digest in found.items() if digest is None]
        if len(absent) == len(hosts):
            raise KeyError(f"Ollama does not list model {model!r} on {', '.join(absent)}")
        if absent:  # listed elsewhere: the hosts disagree, as with different digests
            raise ValueError(f"model {model!r} is missing on {', '.join(absent)} "
                             "but listed on the other hosts")
        if len(set(found.values())) > 1:
            raise ValueError(f"model {model!r} has different digests across hosts: {found}")
        return str(found[hosts[0]])

    return lookup


def tell_payload(variants: TellVariants) -> dict[str, object]:
    """The Tell variants as the context digest covers them: texts only, no provenance."""
    return {"standing": list(variants.standing),
            "statements": {k: list(v) for k, v in variants.statements.items()}}


class Workbench:
    def __init__(
        self,
        db: PackDatabase,
        cache: RunCache,
        steps: tuple[WorkStep, ...],
        variants: TellVariants,
        pdf: list[Chunk],
        base: SpecVariant,
        work_dir: Path,
    ) -> None:
        self.db, self.cache, self.steps, self.variants = db, cache, steps, variants
        self.pdf, self.base, self.work_dir = pdf, base, work_dir
        self.settings = interpreter_settings()
        self.runner: Runner | ParallelRunner

    @classmethod
    def open(
        cls,
        work_dir: Path = WORK_DIR,
        *,
        factory: ExecutorFactory | None = None,
        model_digest: Callable[[str], str] | None = None,
        workers: int = 1,
        hosts: Sequence[str] = OLLAMA_URLS,
    ) -> Workbench:
        """workers: with 2 or more, runs go to that many worker databases at once (spec v2
        §5.1), worker i on Ollama host i modulo the hosts; symbolic repair keeps the main
        database."""
        pdf = pdf_chunks(PDF_DIR, work_dir / "tell")
        base = SpecVariant("base", BASE_PACK, TellCorpus.build(CanonPack(BASE_PACK), pdf))
        variants = load_variants()
        bench = cls(PackDatabase(), RunCache(work_dir / "runs.jsonl"), load_work_orders(),
                    variants, pdf, base, work_dir)
        digests = model_digest or ollama_digests(hosts)
        context = context_digest(tell_payload(variants), bench.settings)

        def runner(db: PackDatabase, host: str) -> Runner:
            def on_host(i: InterpreterId, c: TellCorpus, s: str) -> DeExecutor:
                return bench.llm_factory(i, c, s, base_url=host)

            return Runner(db, bench.cache, factory or on_host, bench.steps,
                          model_digest=digests, context_digest=context)

        if workers <= 1:
            bench.runner = runner(bench.db, hosts[0])
            return bench
        pool = []
        for index in range(workers):
            url = worker_url(bench.db.url, index)
            ensure_database(bench.db.url, str(conninfo_to_dict(url)["dbname"]))
            pool.append(runner(PackDatabase(url), hosts[index % len(hosts)]))
        bench.runner = ParallelRunner(pool, bench.cache)
        return bench

    def llm_factory(
        self, interp: InterpreterId, corpus: TellCorpus, suffix: str, base_url: str = OLLAMA_URL
    ) -> DeExecutor:
        return SpecInterpreterExecutor(
            interp.model, build_registry(), corpus, self.variants,
            variant=interp.variant, seed=interp.seed, suffix=suffix, base_url=base_url,
            temperature=float(self.settings["temperature"]),
            max_steps=int(self.settings["max_steps"]), timeout=float(self.settings["timeout"]),
            max_nudges=int(self.settings["nudges"]),
        )

    def loop_context(self, variant: SpecVariant, config: SpecDxConfig) -> LoopContext:
        if max(config.variants) >= self.variants.count:
            raise ValueError(
                f"configured Tell variants {sorted(config.variants)} exceed the "
                f"{self.variants.count} available; run `specdx make-variants` first"
            )
        pack = CanonPack(variant.pack)
        clauses = ClauseSet.from_pack(pack, self.steps)
        background, _ = split_background(golden_tables(pack))
        lens = SpecLens(clauses, self.steps, ColumnModel.of(pack),
                        drafting=drafting_steps(pack, self.steps), initial=background)
        canon = canon_scenarios(pack)
        literals = extract_literals(variant.corpus)
        candidates = enumerate_scenarios(boundary_domain(pack, [x.bound for x in literals]))
        tboundary = tboundary_scenarios(canon, literals)
        by_id = {s.id: s for s in [*canon.values(), *candidates, *tboundary]}

        def refuter_factory(
            interps: Sequence[InterpreterId], budget: RunBudget
        ) -> CombinedRefuter:
            return CombinedRefuter(
                clauses, SymbolicRepair(self.db, variant, clauses),
                DirectedProbe(self.runner, variant, clauses, interps, budget=budget),
            )

        def localizer(v: SpecVariant, defect: Defect, interps: Sequence[InterpreterId],
                      budget: RunBudget) -> tuple[str, ...] | None:
            scenario_id, step_id = defect.note.split("|", 1)
            scenario: Scenario = by_id[scenario_id]
            step = next(s for s in self.steps if s.id == step_id)
            query = f"{self.variants.statements[step.intent][0]} {' '.join(sorted(step.tables))}"
            # Top 5 chunks, not §3.7's 20: bounds the ddmin cost in P1 (§11).
            chunks = v.corpus.search(query, k=5)
            pool = v.corpus.sentence_ids([c.id for c in chunks])
            fields = frozenset(defect.location)

            def split_persists(corpus: TellCorpus) -> bool:
                """Spec v2 §3.6: the re-run (one behaviour per competent model and variant,
                seed 1) still yields two classes that differ on the ambiguity's fields. The
                test's runs are charged, then dispatched as one batch."""
                probe = SpecVariant(f"{v.name}+ddmin", v.pack, corpus)
                for _ in interps:
                    if not budget.take("ddmin"):
                        raise OutOfBudget
                runs = self.runner.run_batch(
                    [RunRequest(probe, scenario, i, kind="ddmin") for i in interps])
                seen = [view.projection for b in runs for view in lens.views(b, lens.violations(b))
                        if view.item == defect.note and view.acted]
                return any(lens.diff(a, b) & fields for a, b in combinations(seen, 2))

            found = localize_ambiguity(v.corpus, pool, split_persists,
                                       max_tests=config.localize_tests)
            return None if found is None else tuple(found)

        return LoopContext(self.runner, lens, canon, candidates, discount_cap_pct(pack),
                           refuter_factory, localizer, tboundary=tboundary, literals=literals)
