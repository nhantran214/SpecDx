import pytest

from specdx.active.loop import SpecDxConfig
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
