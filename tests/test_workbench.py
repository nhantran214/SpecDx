import dataclasses

import pytest

from specdx.active.loop import SpecDxConfig
from specdx.interpreters.prompt import TellVariants, load_variants
from specdx.orchestrate.reference import reference_executor
from specdx.types import InterpreterId
from specdx.workbench import Workbench

pytestmark = pytest.mark.db
REF = InterpreterId("reference", "ref", 0, 1)


def test_the_workbench_wires_a_runnable_context(tmp_path) -> None:
    bench = Workbench.open(tmp_path, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    ctx = bench.loop_context(bench.base, SpecDxConfig({"ref": ("reference",)}))
    assert set(ctx.canon) == {"grn12", "happy", "discount8"} and ctx.cap_pct == 5
    assert len(ctx.candidates) > 50
    b = ctx.runner.run(bench.base, ctx.canon["grn12"], REF)
    assert ctx.lens.violations(b) == frozenset()


def _context(work_dir) -> str:
    bench = Workbench.open(work_dir, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    return bench.runner.context_digest


@pytest.mark.parametrize(("setting", "value"), [
    ("specdx.interpreters.executor.TEMPERATURE", 0.2),
    ("specdx.interpreters.executor.MAX_STEPS", 8),
    ("specdx.interpreters.executor.TIMEOUT", 60.0),
    ("specdx.interpreters.executor.NUM_CTX", 8192),
    ("specdx.interpreters.executor.SEARCH_K", 3),
    ("specdx.interpreters.executor.SEARCH_REPLY_LIMIT", 6000),
    ("specdx.interpreters.harness.MAX_NUDGES", 1),
    ("specdx.interpreters.harness.NUDGE", "Xong chưa?"),
    ("specdx.observe.run.RUN_SCHEMA_VERSION", 4),
])
def test_any_interpreter_visible_setting_changes_the_context_digest(
    tmp_path, monkeypatch, setting, value
) -> None:
    before = _context(tmp_path)
    assert _context(tmp_path) == before
    monkeypatch.setattr(setting, value)
    assert _context(tmp_path) != before


def test_provenance_stays_outside_the_context_digest(tmp_path, monkeypatch) -> None:
    plain = _context(tmp_path)
    marked = dataclasses.replace(load_variants(), provenance={"standing": ("hand-written",)})
    monkeypatch.setattr("specdx.workbench.load_variants", lambda: marked)
    assert _context(tmp_path) == plain


def test_interpreters_run_with_the_settings_in_the_digest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("specdx.interpreters.executor.TEMPERATURE", 0.2)
    monkeypatch.setattr("specdx.interpreters.executor.MAX_STEPS", 8)
    bench = Workbench.open(tmp_path, model_digest=lambda m: m)
    executor = bench.llm_factory(REF, bench.base.corpus, "")
    assert (executor.temperature, executor.max_steps) == (0.2, 8)
    assert bench.settings["temperature"] == 0.2 and bench.settings["max_steps"] == 8


def test_configured_tell_variants_must_exist(tmp_path) -> None:
    bench = Workbench.open(tmp_path, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    bench.variants = TellVariants(
        bench.variants.standing[:1], {k: v[:1] for k, v in bench.variants.statements.items()}
    )
    config = SpecDxConfig({"ref": ("reference",)}, variants=(0, 1))
    with pytest.raises(ValueError, match="make-variants"):
        bench.loop_context(bench.base, config)


def test_the_context_carries_the_t_literals_and_t_boundary_scenarios(tmp_path) -> None:
    bench = Workbench.open(tmp_path, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    ctx = bench.loop_context(bench.base, SpecDxConfig({"ref": ("reference",)}))
    assert {x.bound for x in ctx.literals} == {("discount_pct", 5), ("discount_pct", 8)}
    assert [s.id for s in ctx.tboundary] == [f"o15-r12-d{d}-g1" for d in (4, 5, 6, 7, 8, 9)]
    signatures = {s.signature() for s in ctx.candidates}
    assert all(s.signature() in signatures for s in ctx.tboundary)
