"""The baselines' shared interface (baselines spec §2.2, §4.1).

A baseline reads one specification's SpecDx evidence (BaselineInput) and returns every
candidate it would report, scored and unthresholded (BaselineOutput). Thresholds are
applied at scoring time with `above`, so selection on the dev split never re-runs a
baseline, and the ranked list gives average precision and Detection@k.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from specdx.cost import Cost
from specdx.types import Defect, Scenario, State

if TYPE_CHECKING:
    from specdx.active.loop import LoopContext, Report, SpecDxConfig
    from specdx.baselines.llm import Gateway
    from specdx.infer.cells import CellData
    from specdx.orchestrate.work_orders import WorkStep
    from specdx.spec.clauses import ClauseSet
    from specdx.variant import SpecVariant

TELL_CHARS = 8000  # spec §9.6


@dataclass
class BaselineInput:
    variant: SpecVariant
    report: Report
    data: CellData
    violated: Mapping[str, frozenset[str]]
    clauses: ClauseSet
    ctx: LoopContext
    config: SpecDxConfig
    steps: Sequence[WorkStep]
    statements: Mapping[str, Sequence[str]]
    drafts: Mapping[str, frozenset[str]]
    golden: State | None = None
    gateway: Gateway | None = None
    # False when PostgreSQL refused the golden state and `golden` holds the canon's tables
    # without derived values (spec §9.8: the C2 mutants forbid a golden status by design).
    golden_materialised: bool = True


@dataclass
class BaselineOutput:
    defects: list[Defect]
    cost: Cost = Cost()
    failure: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Baseline(Protocol):
    name: str
    uses_llm: bool
    reports: tuple[str, ...]
    strategies: tuple[str | None, ...]

    def run(self, inp: BaselineInput, *, model: str | None = None,
            strategy: str | None = None) -> BaselineOutput: ...


REGISTRY: dict[str, Baseline] = {}


def register(b: Baseline) -> Baseline:
    REGISTRY[b.name] = b
    return b


BASELINE_MODULES: tuple[str, ...] = ("jest_rule", "truth_inference", "irt_audit",
                                     "semantic_entropy", "mus_c", "whowhen", "benchguard")


def load_all() -> dict[str, Baseline]:
    """Import every baseline module (they register on import). Kept out of the package
    __init__ so the gate's imports never need the optional [baselines] extra."""
    import importlib

    for name in BASELINE_MODULES:
        importlib.import_module(f"specdx.baselines.{name}")
    return REGISTRY


def ranked(defects: Sequence[Defect]) -> list[Defect]:
    return sorted(defects, key=lambda d: (-d.score, d.type, d.location))


def above(defects: Sequence[Defect], tau: float) -> list[Defect]:
    return [d for d in ranked(defects) if d.score >= tau]


def families(config: SpecDxConfig) -> dict[str, str]:
    return {m: f for f, ms in config.models_by_family.items() for m in ms}


def scenario_text(s: Scenario) -> str:
    return (f"Case CASE-0031: the customer orders {s.ordered} units; {s.received} units are "
            f"received in {s.grn_count} goods receipt(s); the requested discount is "
            f"{s.discount_pct}%.")


def tell_excerpt(inp: Any, step_ids: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """The T sentences the localiser's query retrieves for the given steps (all steps by
    default): per step, top 5 chunks for its statement plus its tables; in step order, then
    sentence order; deduplicated; capped at TELL_CHARS characters (spec §9.6)."""
    corpus = inp.variant.corpus
    wanted = set(step_ids) if step_ids is not None else None
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    used = 0
    for step in inp.steps:
        if (wanted is not None and step.id not in wanted) or step.intent not in inp.statements:
            continue
        query = f"{inp.statements[step.intent][0]} {' '.join(sorted(step.tables))}"
        chunks = corpus.search(query, k=5)
        for sid in corpus.sentence_ids([c.id for c in chunks]):
            if sid in seen:
                continue
            text = corpus.sentence(sid)
            if used + len(text) > TELL_CHARS:
                return out
            seen.add(sid)
            used += len(text)
            out.append((sid, text))
    return out
