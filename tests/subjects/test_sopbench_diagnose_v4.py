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


def test_cell_outcomes_name_each_cells_fate(monkeypatch):
    dx, _ = _setup(WitnessResult("error", (), "boom"), monkeypatch)
    assert dx.cell_outcomes[OUTCOME0] == "witnessed_error"
    dx, _ = _setup(WitnessResult("succeeds", ("a",)), monkeypatch)
    assert dx.cell_outcomes[OUTCOME0] == "witnessed_shared"
    dx, _ = _setup(WitnessResult("refused", ("a",)), monkeypatch)
    assert dx.cell_outcomes[OUTCOME0] == "witnessed_defect"
    assert set(dx.cell_outcomes.values()) <= {"none", "defect", "shared", "undetermined",
                                              "witnessed_defect", "witnessed_shared",
                                              "witnessed_error"}


def test_use_witness_false_leaves_undetermined_outcome_cells_alone(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(diagnose_v4, "witness", lambda d, t: calls.append(1))
    tasks = [S3Task(DOMAIN, i, f"goal{i}", True, f"prompt {i}") for i in range(4)]
    runs = [_run(0, m, False) for m in RATERS] + [_run(i, m, True) for i in range(1, 4)
                                                  for m in RATERS]
    dx = diagnose_domain_v4(DOMAIN, cells_for(tasks, runs), {r.model: r.family for r in runs},
                            [{}] * 4, V4Params(), use_witness=False)
    assert calls == [] and dx.undetermined["outcome"] >= 1
    assert dx.witnessed == {"defect": 0, "shared": 0, "error": 0}
    assert dx.cell_outcomes[OUTCOME0] == "undetermined"


# --- Ruling W-1, F-I4 and the Task 7 Minors (final fix wave) ---

def test_an_undetermined_cell_that_agrees_with_its_anchor_is_not_witnessed(monkeypatch):
    # W-1: only an undetermined outcome cell whose plurality reading differs from the anchor
    # goes to the witness. Task 1 is read by two raters, both agreeing with the anchor: too
    # few readings to decide (theory note T5.5(c)), but no shared deviation to witness.
    tasks = [S3Task(DOMAIN, i, f"goal{i}", True, f"prompt {i}") for i in range(4)]
    runs = [_run(0, m, False) for m in RATERS]
    runs += [_run(1, m, True) for m in list(RATERS)[:2]]
    runs += [_run(i, m, True) for i in (2, 3) for m in RATERS]
    calls: list[int] = []

    def fake(domain, task):
        calls.append(task["i"])
        return WitnessResult("succeeds", ("a",))

    monkeypatch.setattr(diagnose_v4, "witness", fake)
    dx = diagnose_domain_v4(DOMAIN, cells_for(tasks, runs), {r.model: r.family for r in runs},
                            [{"i": i} for i in range(4)], V4Params())
    one = cell_id(DOMAIN, 1, "outcome")
    assert dx.cell_outcomes[one] == "undetermined"
    assert calls == [0]
    assert dx.witnessed == {"defect": 0, "shared": 1, "error": 0}


def test_undetermined_cells_carry_their_would_be_type(monkeypatch):
    # F-I4: every statistically undetermined cell is given its would-be type.
    from specdx.infer.typing_v4 import WOULD_BE

    dx, _ = _setup(WitnessResult("error", (), "boom"), monkeypatch)
    undetermined = {c for c, fate in dx.cell_outcomes.items()
                    if fate in ("undetermined", "witnessed_error")}
    assert undetermined and undetermined <= set(dx.would_be)
    assert set(dx.would_be.values()) <= set(WOULD_BE)


def test_raw_tasks_must_align_with_the_cells(monkeypatch):
    # Task 7 Minor 5: the raw tasks are indexed like the cells; a short or misordered list is
    # an error, never a silent witness of the wrong task.
    import pytest

    tasks = [S3Task(DOMAIN, i, f"goal{i}", True, f"prompt {i}") for i in range(4)]
    runs = [_run(i, m, True) for i in range(4) for m in RATERS]
    cd = cells_for(tasks, runs)
    models = {r.model: r.family for r in runs}
    with pytest.raises(ValueError, match="raw tasks"):
        diagnose_domain_v4(DOMAIN, cd, models, [{}] * 3, V4Params())
    raw = [{"user_goal": f"goal{i}"} for i in range(4)]
    raw[2]["user_goal"] = "other"
    with pytest.raises(ValueError, match="raw tasks"):
        diagnose_domain_v4(DOMAIN, cd, models, raw, V4Params())


def test_the_real_witness_on_bank():
    # Task 7 Minor 3: an integration test with SOPBench's real strict replay. Every rater
    # deviates on two bank tasks: one whose label contradicts the replay (cancel_credit_card)
    # and one whose label the replay confirms.
    from specdx.subjects.sopbench.load import load_tasks
    from specdx.subjects.sopbench.witness import load_raw_tasks, witness

    tasks = load_tasks("bank")
    raw = load_raw_tasks("bank")
    label = ["succeeds" if t.should_succeed else "refused" for t in tasks]
    replay = {}
    for t in tasks:
        replay[t.index] = witness("bank", raw[t.index]).outcome
        bad = [i for i, o in replay.items() if o not in ("error", label[i])]
        good = [i for i, o in replay.items() if o == label[i]]
        if bad and len(good) >= 4:
            break
    bad, good = bad[0], good[:4]
    models = dict(RATERS)

    def run(task, model, deviate):
        ok = tasks[task].should_succeed != deviate
        return S3Run("bank", task, model, models[model], "fc", ok, True, True, True, False,
                     f"prompt {task}")

    runs = [run(i, m, True) for i in (bad, good[0]) for m in models]
    runs += [run(i, m, False) for i in good[1:] for m in models]
    dx = diagnose_domain_v4("bank", cells_for(tasks, runs), models, raw, V4Params())
    assert dx.cell_outcomes[cell_id("bank", bad, "outcome")] == "witnessed_defect"
    assert dx.cell_outcomes[cell_id("bank", good[0], "outcome")] == "witnessed_shared"
    [hidden] = [d for d in dx.defects if d.note.startswith("witnessed")]
    assert hidden.subtype == "self-inconsistent"
    assert hidden.location == (cell_id("bank", bad, "outcome"),)
