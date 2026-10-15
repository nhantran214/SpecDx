import dataclasses

import pytest

from specdx.active.loop import SpecDxConfig
from specdx.db import _hold_lock
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


def _split_oracle(tmp_path, monkeypatch, interps):
    """The workbench's ddmin oracle (split_persists) on a fake runner: it charges and
    dispatches one re-run per interpreter, as one batch, and asks the loop's test. Returns
    (the test, the budget, the batches dispatched, the runs the loop's test saw)."""
    from specdx.active.budget import RunBudget
    from specdx.types import Defect

    batches: list[list] = []
    seen: list[list] = []

    class Runner:
        executed = 0

        def run_batch(self, requests):
            batches.append([(r.interp.model, r.kind, r.scenario.id, r.variant.name)
                            for r in requests])
            return [r.interp.model for r in requests]

    captured = []
    monkeypatch.setattr("specdx.workbench.localize_ambiguity",
                        lambda corpus, pool, test, max_tests=None: captured.append(test))
    bench = Workbench.open(tmp_path, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    bench.runner = Runner()
    ctx = bench.loop_context(bench.base, SpecDxConfig({"ref": ("reference",)}))
    budget = RunBudget(4, 10)
    defect = Defect("ambiguity", ("po.trang_thai",), ("grn12",), (), 1.0,
                    "grn12|DE-FN-04:draft_purchase_order")

    def persists(runs):
        seen.append(list(runs))
        return len(set(runs)) >= 2

    assert ctx.localizer(bench.base, defect, interps, budget, persists) is None
    [test] = captured
    return test, budget, batches, seen


def test_the_ddmin_oracle_reruns_each_interpreter_and_asks_the_loops_test(
    tmp_path, monkeypatch
) -> None:
    interps = [InterpreterId("m1", "f", 0, 1), InterpreterId("m2", "f", 0, 1)]
    test, budget, batches, seen = _split_oracle(tmp_path, monkeypatch, interps)
    assert test(None) is True and budget.ddmin == 2
    assert batches == [[("m1", "ddmin", "grn12", "base+ddmin"),
                        ("m2", "ddmin", "grn12", "base+ddmin")]]
    assert seen == [["m1", "m2"]]


def test_an_empty_oracle_leaves_the_ambiguity_unlocalised_and_spends_nothing(
    tmp_path, monkeypatch
) -> None:
    test, budget, batches, seen = _split_oracle(tmp_path, monkeypatch, [])
    assert test(None) is False and budget.ddmin == 0 and batches == [] and seen == []


def test_opening_a_workbench_takes_the_main_database_lock(tmp_path, monkeypatch) -> None:
    taken = []
    real = _hold_lock
    monkeypatch.setattr("specdx.workbench._hold_lock",
                        lambda url: taken.append(url) or real(url), raising=False)
    bench = Workbench.open(tmp_path, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    assert taken == [bench.db.url]
