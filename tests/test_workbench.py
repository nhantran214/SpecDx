import pytest

from specdx.active.loop import SpecDxConfig
from specdx.interpreters.prompt import TellVariants
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


def test_configured_tell_variants_must_exist(tmp_path) -> None:
    bench = Workbench.open(tmp_path, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    bench.variants = TellVariants(
        bench.variants.standing[:1], {k: v[:1] for k, v in bench.variants.statements.items()}
    )
    config = SpecDxConfig({"ref": ("reference",)}, variants=(0, 1))
    with pytest.raises(ValueError, match="make-variants"):
        bench.loop_context(bench.base, config)
