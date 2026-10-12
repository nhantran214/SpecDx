"""The whole gate on the real database: workbench, loop, refutation, ddmin and scoring.

Interpreters are scripted at the executor boundary: the reference flow, and a deviant
that also flags the receipt when 12 units arrive (grn12), an event the golden state lacks.
"""

import json

import pytest
from tonomyx.contracts.envelope import ResultEnvelope, TaskEnvelope
from tonomyx.tools.impl.log_event import log_event
from tonomyx.tools.registry import ToolContext

from specdx.active.loop import run_specdx
from specdx.eval.pilot import GateConfig, gate_markdown, run_gate
from specdx.mutate.operators import all_mutants
from specdx.orchestrate.reference import reference_executor
from specdx.workbench import Workbench

pytestmark = pytest.mark.db


class Deviant:
    def __init__(self) -> None:
        self.inner = reference_executor()

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        result = self.inner.run(task, ctx)
        if task.intent == "record_goods_receipt" and str(task.inputs.get("received")) == "12":
            log_event(ctx, case_id=task.case_id, kind="FLAGGED", ref="GRN-0015", actor="AI",
                      basis="phiếu nhập lệch so với đơn")
        return result


CONFIG = GateConfig(
    models_by_family={"ref": ("reference",), "dev": ("deviant",)},
    variants=(0,), seeds=(1, 2), budget=12, dev_per_type=1, rng_seed=1, probe_budget=6,
    localize_budget=6,
)


def _bench(work_dir) -> Workbench:
    return Workbench.open(
        work_dir,
        factory=lambda i, c, s: Deviant() if i.model == "deviant" else reference_executor(),
        model_digest=lambda m: f"digest-{m}",
    )


def test_the_gate_runs_end_to_end_on_real_mutants(tmp_path) -> None:
    bench = _bench(tmp_path)
    mutants = [m for m in all_mutants(bench.base, bench.pdf, bench.work_dir)
               if m.id in ("C2-00", "W1-00")]
    out = tmp_path / "results" / "p1-gate-mutants.jsonl"
    result = run_gate(CONFIG, bench, mutants, mutants_out=out)

    assert result.failed == {} and result.headline in ("PASSED", "NOT PASSED")
    assert "## Verdict" in gate_markdown(result)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [x.get("mutant") for x in lines] == [None, "C2-00", "W1-00"]
    assert lines[0]["base"] is True and lines[0]["runs"]["population"] == 12
    for line in lines[1:]:
        assert line["failure"] is None and line["runs"]["population"] == 12
        assert line["runs"]["probe"] <= 6 and line["runs"]["ddmin"] <= 6
    # C2-00 forbids the approved PO the golden state requires: no run can reach it
    c2 = lines[1]
    assert ["contradiction", ["golden:po:PO-0015:trang_thai"]] in c2["matches"]["specdx"]["typed"]
    # the deviant's split is base noise: the differential removes it from both mutants
    assert not [d for x in lines[1:] for d in x["defects"]["specdx"] if d["type"] == "ambiguity"]
    noise = ["ambiguity", "DE-OP-02:record_goods_receipt", ["evt:FLAGGED:GRN-0015"]]
    assert lines[0]["ambiguity_keys"] == [noise]
    assert all(x["ambiguity_dropped"] == [noise] for x in lines[1:])


def test_a_split_that_no_tell_sentence_causes_stays_unlocalised(tmp_path) -> None:
    bench = _bench(tmp_path)
    config = CONFIG.specdx()
    report = run_specdx(bench.base, bench.loop_context(bench.base, config), config)
    [split] = [d for d in report.defects if d.type == "ambiguity"]
    assert split.note == "grn12|DE-OP-02:record_goods_receipt" and split.location == ()
    assert report.ddmin_runs == 2  # fails([]): one run per family, and the split persists
