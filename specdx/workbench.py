"""Wires the database, the run cache, the Tell corpus, the runner and the loop context."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from tonomyx.canon.pack import CanonPack
from tonomyx.de.executor import DeExecutor
from tonomyx.tools.catalog import build_registry

from specdx.active.budget import RunBudget
from specdx.active.ddmin import OutOfBudget, localize_ambiguity
from specdx.active.loop import LoopContext, SpecDxConfig
from specdx.active.refute import CombinedRefuter, DirectedProbe, SymbolicRepair
from specdx.config import BASE_PACK, OLLAMA_URL, PDF_DIR, WORK_DIR
from specdx.db import PackDatabase
from specdx.infer.items import SpecLens
from specdx.interpreters.cache import RunCache
from specdx.interpreters.executor import SpecInterpreterExecutor, interpreter_settings
from specdx.interpreters.prompt import TellVariants, load_variants
from specdx.observe.run import ExecutorFactory, Runner, context_digest
from specdx.orchestrate.work_orders import WorkStep, load_work_orders
from specdx.scenarios.space import (
    boundary_domain,
    canon_scenarios,
    discount_cap_pct,
    enumerate_scenarios,
)
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import Chunk, TellCorpus, pdf_chunks
from specdx.types import Defect, InterpreterId, Scenario
from specdx.variant import SpecVariant


def ollama_digests(
    base_url: str = OLLAMA_URL, transport: httpx.BaseTransport | None = None
) -> Callable[[str], str]:
    """Model name → Ollama digest, fetched on first use, so the run cache never mixes two
    builds of one model. An unknown model or an unreachable Ollama is an error."""
    digests: dict[str, str] = {}

    def lookup(model: str) -> str:
        if not digests:
            with httpx.Client(transport=transport, timeout=10.0) as client:
                response = client.get(f"{base_url}/api/tags")
                response.raise_for_status()
                for m in response.json().get("models", []):
                    digests[str(m["name"])] = str(m["digest"])
        name = model if ":" in model else f"{model}:latest"
        if name not in digests:
            raise KeyError(f"Ollama does not list model {model!r}")
        return digests[name]

    return lookup


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
        self.runner: Runner

    @classmethod
    def open(
        cls,
        work_dir: Path = WORK_DIR,
        *,
        factory: ExecutorFactory | None = None,
        model_digest: Callable[[str], str] | None = None,
    ) -> Workbench:
        pdf = pdf_chunks(PDF_DIR, work_dir / "tell")
        base = SpecVariant("base", BASE_PACK, TellCorpus.build(CanonPack(BASE_PACK), pdf))
        variants = load_variants()
        bench = cls(PackDatabase(), RunCache(work_dir / "runs.jsonl"), load_work_orders(),
                    variants, pdf, base, work_dir)
        tell = {"standing": list(variants.standing),
                "statements": {k: list(v) for k, v in variants.statements.items()}}
        bench.runner = Runner(bench.db, bench.cache, factory or bench.llm_factory, bench.steps,
                              model_digest=model_digest or ollama_digests(),
                              context_digest=context_digest(tell, bench.settings))
        return bench

    def llm_factory(self, interp: InterpreterId, corpus: TellCorpus, suffix: str) -> DeExecutor:
        return SpecInterpreterExecutor(
            interp.model, build_registry(), corpus, self.variants,
            variant=interp.variant, seed=interp.seed, suffix=suffix,
            temperature=float(self.settings["temperature"]),
            max_steps=int(self.settings["max_steps"]), timeout=float(self.settings["timeout"]),
        )

    def loop_context(self, variant: SpecVariant, config: SpecDxConfig) -> LoopContext:
        if max(config.variants) >= self.variants.count:
            raise ValueError(
                f"configured Tell variants {sorted(config.variants)} exceed the "
                f"{self.variants.count} available; run `specdx make-variants` first"
            )
        pack = CanonPack(variant.pack)
        clauses = ClauseSet.from_pack(pack, self.steps)
        lens = SpecLens(clauses, self.steps, ColumnModel.of(pack))
        canon = canon_scenarios(pack)
        candidates = enumerate_scenarios(boundary_domain(pack))
        by_id = {s.id: s for s in [*canon.values(), *candidates]}

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
            one_per_family: dict[str, InterpreterId] = {}
            for interp in interps:
                one_per_family.setdefault(interp.family, interp)

            def split_persists(corpus: TellCorpus) -> bool:
                probe = SpecVariant(f"{v.name}+ddmin", v.pack, corpus)
                classes = set()
                for interp in one_per_family.values():
                    if not budget.take("ddmin"):
                        raise OutOfBudget
                    b = self.runner.run(probe, scenario, interp, kind="ddmin")
                    for view in lens.views(b, lens.violations(b)):
                        if view.item == defect.note:
                            classes.add(view.klass)
                return len(classes) >= 2

            found = localize_ambiguity(v.corpus, pool, split_persists,
                                       max_tests=config.localize_tests)
            return None if found is None else tuple(found)

        return LoopContext(self.runner, lens, canon, candidates, discount_cap_pct(pack),
                           refuter_factory, localizer)
