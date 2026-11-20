"""v4 diagnosis of one SOPBench domain with execution witnesses (P4a Task 7). The witness is
replaced by a stub that counts calls; the real replay is tested in test_sopbench_witness."""

from specdx.infer.anchored_v4 import V4Params
from specdx.subjects.sopbench import diagnose_v4
from specdx.subjects.sopbench.cells import cell_id, cells_for
from specdx.subjects.sopbench.diagnose_v4 import S3DiagnosisV4, diagnose_domain_v4
from specdx.subjects.sopbench.load import S3Run, S3Task
from specdx.subjects.sopbench.witness import WitnessResult

DOMAIN = "library"
RATERS = {
    "gpt-4o": "openai", "gpt-4o-mini": "openai",
    "gemini-2.5-pro": "google", "gemini-2.5-flash": "google",
    "qwen2.5-7b-instruct": "qwen", "qwen2.5-72b-instruct": "qwen",
}
OUTCOME0 = cell_id(DOMAIN, 0, "outcome")


def _run(task: int, model: str, ok: bool) -> S3Run:
    return S3Run(DOMAIN, task, model, RATERS[model], "fc", ok, True, True, True, False,
                 f"prompt {task}")


def _setup(stub, monkeypatch):
    """Task 0 should succeed but every rater refuses (unanimous, so undetermined);
    tasks 1-3 are agreed."""
    tasks = [S3Task(DOMAIN, i, f"goal{i}", True, f"prompt {i}") for i in range(4)]
    runs = [_run(0, m, False) for m in RATERS] + [_run(i, m, True) for i in range(1, 4)
                                                  for m in RATERS]
    calls: list[int] = []

    def fake(domain, task):
        calls.append(task["i"])
        return stub

    monkeypatch.setattr(diagnose_v4, "witness", fake)
    raw = [{"i": i} for i in range(4)]
    dx = diagnose_domain_v4(DOMAIN, cells_for(tasks, runs), {r.model: r.family for r in runs},
                            raw, V4Params())
    return dx, calls


def test_unanimous_refusal_on_a_should_succeed_task_is_witnessed_shared_when_replay_succeeds(
        monkeypatch):
    dx, calls = _setup(WitnessResult("succeeds", ("a",)), monkeypatch)
    assert isinstance(dx, S3DiagnosisV4)
    assert [f.cell for f in dx.shared] == [OUTCOME0]
    assert dx.shared[0].values == ("witnessed",)
    assert dx.defects == []
    assert dx.witnessed == {"defect": 0, "shared": 1, "error": 0}
    assert dx.undetermined["outcome"] == 0
    assert calls == [0]


def test_replay_mismatch_becomes_a_self_inconsistent_hidden_defect(monkeypatch):
    dx, calls = _setup(WitnessResult("refused", ("a", "b")), monkeypatch)
    [d] = dx.defects
    assert d.type == "hidden" and d.location == (OUTCOME0,)
    assert d.scenarios == (f"{DOMAIN}:0",) and d.witnesses == ("strict replay: ('a', 'b')",)
    assert d.score == 1.0 and d.note == "witnessed: strict replay"
    assert d.subtype == "self-inconsistent"
    assert dx.shared == [] and dx.witnessed == {"defect": 1, "shared": 0, "error": 0}


def test_witness_error_keeps_the_cell_undetermined(monkeypatch):
    dx, calls = _setup(WitnessResult("error", (), "boom"), monkeypatch)
    assert dx.defects == [] and dx.shared == []
    assert dx.witnessed == {"defect": 0, "shared": 0, "error": 1}
    assert dx.undetermined["outcome"] == 1
    assert calls == [0]
