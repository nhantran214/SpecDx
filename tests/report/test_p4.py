"""The P4a stage functions, on tiny synthetic trajectories (no network, no real data).

The development domain is university and the held-out domain is library, so the stages stay
small; `split` is injected. Nothing is written outside tmp_path.
"""

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from specdx.cli import main
from specdx.infer.anchored_v4 import V4Params
from specdx.report import p4
from specdx.report.p4 import (
    SMOKE_PARAMS,
    g0v4_stage,
    placebo_stage,
    placebo_verdict,
    unanimous_fates,
    v4_stage,
    witness_stage,
)
from specdx.report.stages import SMOKE_LINE, StageOutcome, already_done
from specdx.subjects.sopbench.cells import cells_for
from specdx.subjects.sopbench.diagnose import OFFLINE
from specdx.subjects.sopbench.diagnose_v4 import S3DiagnosisV4
from specdx.subjects.sopbench.load import S3Run, load_tasks
from specdx.subjects.sopbench.placebo import PlaceboResult
from specdx.subjects.sopbench.witness import WitnessResult
from specdx.types import Defect

SPLIT = (("university",), ("library",))
MODELS = (("gpt-4o", "openai", False), ("claude-3-5-sonnet-20241022", "anthropic", False),
          ("gpt-4o-mini", "openai", False), ("qwen2.5-7b-instruct", "qwen", True),
          ("llama3.1-8b-instruct", "meta", True))
GRID = {"cell_sd": (1.0,), "lambda_prior": ((1.0, 3.0),), "tau_u": (2.0,), "tau_h": (0.5,),
        "tau_a": (0.3,)}
NAME = "output/{d}/ast_{m}-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"


def _record(task, k: int) -> dict:
    """Model k (0 strongest): every 3rd task, the stronger two deviate on db_match; on the
    other tasks the weaker models slip on the constraints check more often."""
    if task.index % 3 == 0:
        db, con = k >= 2, True
    else:
        db, con = True, not ((task.index * 7 + k * 3) % 8 < max(0, k - 1))
    ev = {"action_successfully_called": task.should_succeed, "database_match": db,
          "constraint_not_violated": con, "dirgraph_satisfied": True,
          "no_tool_call_error": True}
    fields = {"user_prompt": task.user_prompt, "user_goal": task.user_goal,
              "action_should_succeed": task.should_succeed, "user_known": task.user_known,
              "constraint_parameters": task.constraint_parameters}
    return {"domain": task.domain, "setup": {}, "task": fields,
            "interactions": [{"prompt": f"P{task.index}", "interaction": [], "database": {}}],
            "evaluations": [ev]}


def _data(tmp_path: Path, domains=("university", "library")) -> tuple[Path, Path]:
    files = []
    for domain in domains:
        tasks = load_tasks(domain)
        for k, (model, family, open_w) in enumerate(MODELS):
            path = NAME.format(d=domain, m=model)
            target = tmp_path / "traj" / domain / Path(path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps([_record(t, k) for t in tasks]))
            files.append({"path": path, "domain": domain, "model": model, "canonical": model,
                          "family": family, "open_weights": open_w, "mode": "fc", "size": 1,
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"commit": "c", "files": files}))
    return manifest, tmp_path / "traj"


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> tuple[Path, Path]:
    return _data(tmp_path_factory.mktemp("p4data"))


@pytest.fixture(scope="module")
def params() -> V4Params:
    return V4Params()


# --- s3-g0v4 ---

def test_g0v4_stage_calibrates_runs_and_freezes_only_on_pass(data, tmp_path) -> None:
    manifest, traj = data
    frozen = tmp_path / "frozen.json"
    out = g0v4_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                     replicates=1, grid=GRID, params_file=frozen)
    assert out.exit_code in (0, 1) and out.verdict == ("PASS" if out.exit_code == 0 else "FAIL")
    for section in ("## Calibration", "## Selected parameters", "## v3 beside v4",
                    "## G0v4 in full", "not certified by the theory"):
        assert section in out.body
    assert "| hidden F1 |" in out.body and "act_group" in out.body
    assert frozen.exists() == (out.exit_code == 0)
    assert not (tmp_path / SMOKE_PARAMS).exists()
    assert (tmp_path / "s3-g0v4-search.json").exists()
    assert out.data["params"]["tau_u"] == 2.0


def test_smoke_g0v4_never_freezes_and_writes_smoke_params(data, tmp_path, monkeypatch) -> None:
    # Plumbing only: run_g0v4 is stubbed (the real search is covered by the test above).
    manifest, traj = data
    seen = {}

    def fake_run(config, out, *, replicates, grid, jobs):
        seen.update(replicates=replicates, grid=grid)
        result = {"chosen": {"params": dataclasses.asdict(V4Params(tau_u=2.0)),
                             "metrics": {"hidden_f1": 0.5, "ambiguity_f1": 0.5,
                                         "shared_fp": 0.0}},
                  "v3": {"hidden_f1": 0.5, "ambiguity_f1": 0.5, "shared_hidden": 0.1},
                  "criteria": {}, "failed": ["1"]}
        out.with_suffix(".json").write_text(json.dumps(result))
        return 1

    monkeypatch.setattr(p4, "run_g0v4", fake_run)
    monkeypatch.setattr(p4, "render", lambda result: "a\nb\nc\nrendered")
    monkeypatch.setattr(p4, "freeze_v4", lambda *a, **k: pytest.fail("smoke must not freeze"))
    frozen = tmp_path / "frozen.json"
    out = g0v4_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                     smoke=True, replicates=99, grid=None, params_file=frozen)
    assert out.exit_code == 1 and not frozen.exists()
    assert seen["replicates"] == 2 and seen["grid"] == dict(p4.SMOKE_GRID)
    assert json.loads((tmp_path / SMOKE_PARAMS).read_text())["params"]["tau_u"] == 2.0
    assert "SMOKE" in out.body and out.data["frozen"] is False


def test_g0v4_stage_without_data_exits_2_and_does_not_fetch(tmp_path) -> None:
    out = g0v4_stage(manifest=tmp_path / "none.json", trajectories=tmp_path / "t",
                     results=tmp_path, split=SPLIT)
    assert (out.exit_code, out.verdict) == (2, "FAIL") and "manifest" in out.body
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"files": []}))
    out = g0v4_stage(manifest=manifest, trajectories=tmp_path / "t", results=tmp_path,
                     split=SPLIT)
    assert out.exit_code == 2 and "run fetch-sopbench first" in out.body


# --- s3-witness ---

def test_witness_stage_counts_match_mismatch_and_error_per_domain(monkeypatch) -> None:
    # The stub reads a task's position from its raw record: 0 and 3 replay as "succeeds",
    # 1 errors, 2 is refused.
    monkeypatch.setattr(p4, "load_raw_tasks", lambda d, root: [{"i": i} for i in range(66)])
    replay = {0: "succeeds", 1: "error", 2: "refused", 3: "succeeds"}

    def stub(domain, task):
        outcome = replay.get(task["i"], "succeeds")
        return WitnessResult(outcome, ("c",), "boom" if outcome == "error" else "")

    tasks = load_tasks("library")[:4]
    out = witness_stage(domains=("library",), limit=4, witness_fn=stub)
    row = out.data["domains"]["library"]
    anchors = ["succeeds" if t.should_succeed else "refused" for t in tasks]
    expect_match = sum(a == replay[i] for i, a in enumerate(anchors))
    assert row["tasks"] == 4 and row["error"] == 1
    assert row["match"] == expect_match and row["match"] + row["mismatch"] == 3
    assert (out.exit_code, out.verdict) == (0, "DONE")
    assert "## Mismatches" in out.body and "## Errors" in out.body and "boom" in out.body
    assert "| library | 4 |" in out.body and "| total |" in out.body
    assert out.data["errors"] == [{"domain": "library", "task": 1, "error": "boom"}]
    assert len(out.data["mismatches"]) == row["mismatch"]


def test_witness_stage_runs_the_real_replay_on_a_few_tasks() -> None:
    out = witness_stage(domains=("library",), limit=3)
    row = out.data["domains"]["library"]
    assert row["tasks"] == 3 and row["match"] + row["mismatch"] + row["error"] == 3


def test_smoke_witness_takes_20_tasks_per_domain() -> None:
    seen: list[str] = []

    def stub(domain, task):
        seen.append(domain)
        return WitnessResult("succeeds", ())

    out = witness_stage(domains=("university", "library"), smoke=True, witness_fn=stub)
    assert seen == ["university"] * 20 + ["library"] * 20
    assert out.data["limit"] == 20


# --- s3-v4 ---

def test_v4_stage_reports_the_domain_table_v3_and_the_ablations(data, params, tmp_path) -> None:
    manifest, traj = data
    out = v4_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                   params=params)
    assert (out.exit_code, out.verdict) == (0, "DONE")
    for header in ("## v4 per domain", "## v3 on the same data", "## Ablations (design §6.4)"):
        assert header in out.body
    for column in ("raters", "undetermined outcome", "undetermined check", "witnessed defect",
                   "witnessed shared", "witnessed error", "cells accounted", "shared typed",
                   "shared witnessed", "θ min", "θ median", "converged"):
        assert column in out.body
    for label in ("full v4 / library", "no per-cell β / library",
                  "no composite weights / library", "no witnesses / library",
                  "plug-in instead of profile / library", "full v3 / library",
                  "no attractor / library", "vacuous on S3 (binary cells)"):
        assert label in out.body
    assert "not certified by the theory" in out.body
    [row] = out.data["v4"]
    # Task 7's counting: a witness error is already an undetermined outcome cell.
    assert row["cells accounted"] == (row["undetermined outcome"] + row["undetermined check"]
                                      + row["witnessed defect"] + row["witnessed shared"])
    assert row["witnessed error"] <= row["undetermined outcome"]
    assert len(out.data["ablations"]) == 7
    # "no witnesses" never witnesses; every undetermined outcome cell stays undetermined.
    nowit = next(r for r in out.data["ablations"] if r[0] == "no witnesses / library")
    assert nowit[8:11] == [0, 0, 0]


def test_v4_stage_needs_frozen_parameters_and_data(data, tmp_path) -> None:
    manifest, traj = data
    out = v4_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                   params_file=tmp_path / "absent.json")
    assert out.exit_code == 2 and "run s3-g0v4 first" in out.body
    out = v4_stage(manifest=tmp_path / "no.json", trajectories=traj, results=tmp_path,
                   split=SPLIT, params=V4Params())
    assert out.exit_code == 2


def _stub_diagnoses(monkeypatch, calls: list[dict]) -> None:
    """Replace the two diagnoses with cheap fakes; record each v4 call's keyword switches."""
    fit = SimpleNamespace(theta={"a": 0.5, "b": 0.7}, converged=True)
    contradiction = Defect("contradiction", ("c1", "c2"), ("s",), (), 0.9, OFFLINE)

    def fake_v4(domain, cd, models, raw, params, **kw):
        calls.append(kw)
        return S3DiagnosisV4(domain, ("a", "b"), fit, [contradiction], [],
                             {"outcome": 1, "check": 2}, {"defect": 0, "shared": 0, "error": 0})

    monkeypatch.setattr(p4, "diagnose_domain_v4", fake_v4)
    monkeypatch.setattr(p4, "diagnose_domain", lambda d, cd, m: SimpleNamespace(
        raters=("a", "b"), defects=[contradiction], fit=fit))


def test_smoke_v4_uses_smoke_params_when_nothing_is_frozen(data, tmp_path, monkeypatch) -> None:
    manifest, traj = data
    _stub_diagnoses(monkeypatch, [])
    (tmp_path / SMOKE_PARAMS).write_text(json.dumps({"params": {"tau_u": 3.0}}))
    out = v4_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                   smoke=True, params_file=tmp_path / "absent.json")
    assert out.exit_code == 0 and out.data["params"]["tau_u"] == 3.0
    assert "smoke (not frozen)" in out.body
    # Without either file, smoke also exits 2.
    (tmp_path / SMOKE_PARAMS).unlink()
    out = v4_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                   smoke=True, params_file=tmp_path / "absent.json")
    assert out.exit_code == 2


def test_v4_stage_ablation_switches_and_key_number_count_contradictions(
        data, params, tmp_path, monkeypatch) -> None:
    manifest, traj = data
    calls: list[dict] = []
    _stub_diagnoses(monkeypatch, calls)
    out = v4_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                   params=params)
    assert calls == [{}, {"cell_beta": False}, {"composite": False}, {"use_witness": False},
                     {"profile": False}]
    # One contradiction, three undetermined cells: the key number counts the contradiction.
    assert out.key_number == "1 typed defects, 3 undetermined over 1 domains"


# --- s3-placebo ---

def _result(domain, real, placebo_counts, real_v3=10, placebo_v3=(10,)) -> PlaceboResult:
    ratio = sum(placebo_counts) / len(placebo_counts) / real if real else math.nan
    return PlaceboResult(domain, real, real_v3, list(placebo_counts), list(placebo_v3), ratio,
                         1.0, 50)


def test_placebo_verdict_follows_ruling_p_n() -> None:
    ok = [_result("a", 10, [1, 2]), _result("b", 10, [0, 0])]
    assert placebo_verdict(ok) == ("PASS", ["a", "b"], [])
    # A domain over the bound fails.
    assert placebo_verdict([*ok, _result("c", 10, [5, 5])]) == ("FAIL", ["a", "b", "c"], ["c"])
    # Not evaluable (real v4 = 0) is excluded, not a pass and not a fail.
    assert placebo_verdict([*ok, _result("d", 0, [0])]) == ("PASS", ["a", "b"], [])
    # Fewer than 2 evaluable domains cannot pass, however clean.
    assert placebo_verdict([ok[0], _result("d", 0, [0])])[0] == "FAIL"
    assert placebo_verdict([_result("d", 0, [0])]) == ("FAIL", [], [])
    # The bound is inclusive: 0.2 passes.
    assert placebo_verdict([_result("a", 10, [2]), _result("b", 10, [2])])[0] == "PASS"


def test_placebo_stage_reports_domains_unanimous_cells_and_the_verdict(
        data, params, tmp_path) -> None:
    manifest, traj = data
    out = placebo_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                        seeds=[0, 1], params=params)
    assert out.verdict in ("PASS", "FAIL") and out.exit_code == (out.verdict != "PASS")
    for text in ("## Placebo per domain", "## Unanimous cells (reported apart)",
                 "changeable cells", "v4 ratio", "v3 ratio", "evaluable", "not certified"):
        assert text in out.body
    [r] = out.data["results"]
    # One held-out domain can never reach the two evaluable domains the verdict needs.
    assert out.verdict == "FAIL" and out.data["evaluable"] == (["library"] if r["real_v4"]
                                                                else [])
    if r["real_v4"] == 0:
        assert "not evaluable (real v4 = 0)" in out.body
    deviating = out.data["unanimous"]["library"]
    assert set(deviating) == {"agreeing", "deviating"}
    for header in ("witnessed_defect", "witnessed_shared", "witnessed_error", "undetermined"):
        assert header in out.body


def test_unanimous_fates_split_agreeing_and_deviating_cells() -> None:
    tasks = load_tasks("library")[:4]
    models = {"m0": "a", "m1": "b", "m2": "c"}

    def run(t, m, ok):
        return S3Run("library", t, m, models[m], "fc", ok, True, True, True, False, f"p{t}")

    # Every rater reads task t as "succeeds" except task 0, which all read as "refused":
    # every cell is unanimous.
    runs = [run(t, m, ok=(t != 0)) for t in range(4) for m in models]
    cd = cells_for(tasks, runs)
    cells = {r.cell for r in cd.data.readings}
    dx = S3DiagnosisV4("library", tuple(models), None, [], [], {}, {},
                       {c: "undetermined" for c in cells})
    fates = unanimous_fates(cd, dx)
    assert sum(sum(c.values()) for c in fates.values()) == len(cells) == 16
    deviating = sum(1 for t in tasks if t.should_succeed == (t.index == 0))
    assert sum(fates["deviating"].values()) == deviating
    assert set(fates["agreeing"]) | set(fates["deviating"]) == {"undetermined"}


def test_placebo_stage_without_params_or_data_exits_2(data, tmp_path) -> None:
    manifest, traj = data
    out = placebo_stage(manifest=manifest, trajectories=traj, results=tmp_path, split=SPLIT,
                        params_file=tmp_path / "none.json")
    assert out.exit_code == 2 and "run s3-g0v4 first" in out.body
    out = placebo_stage(manifest=manifest, trajectories=tmp_path / "missing", results=tmp_path,
                        split=SPLIT, params=V4Params())
    assert out.exit_code == 2 and "fetch-sopbench" in out.body


# --- the CLI and smoke mode ---

def _cli(tmp_path, *args, smoke=False, monkeypatch=None) -> int:
    base = ["p3", args[0], "--results", str(tmp_path / "res"), "--work", str(tmp_path / "work"),
            *args[1:]]
    if monkeypatch is not None:
        if smoke:
            monkeypatch.setenv("P4_SMOKE", "1")
        else:
            monkeypatch.delenv("P4_SMOKE", raising=False)
    return main(base)


def test_a_smoke_stage_writes_under_smoke_with_the_label_and_leaves_the_official_stage_alone(
        tmp_path, monkeypatch) -> None:
    code = _cli(tmp_path, "s3-witness", smoke=True, monkeypatch=monkeypatch)
    assert code == 0
    res = tmp_path / "res"
    report = res / "smoke" / "s3-witness.md"
    assert report.read_text().startswith(SMOKE_LINE + "\n")
    assert (res / "smoke" / "STATUS.md").read_text().startswith("# SMOKE — not results")
    assert not (res / "s3-witness.md").exists() and not (res / "STATUS.md").exists()
    assert not already_done("s3-witness", res)
    assert not (tmp_path / "work" / "p3-s3-witness.pid").exists()


def test_smoke_never_runs_a_non_p4a_stage(tmp_path, monkeypatch) -> None:
    assert _cli(tmp_path, "replication", smoke=True, monkeypatch=monkeypatch) == 2
    assert not (tmp_path / "res" / "smoke" / "replication.md").exists()


def test_the_official_cli_stage_without_data_exits_2_and_writes_a_report(
        tmp_path, monkeypatch) -> None:
    code = _cli(tmp_path, "s3-v4", monkeypatch=monkeypatch)
    assert code == 2
    text = (tmp_path / "res" / "s3-v4.md").read_text()
    assert not text.startswith(SMOKE_LINE) and "FAIL" in text
    assert not (tmp_path / "res" / "smoke").exists()


def test_the_stage_outcome_type_is_returned() -> None:
    assert isinstance(witness_stage(domains=("library",), limit=1), StageOutcome)


def test_a_bad_p4_smoke_value_exits_2_with_a_message(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("P4_SMOKE", "yes")
    assert _cli(tmp_path, "s3-witness") == 2
    assert "P4_SMOKE must be unset, empty, 0 or 1" in capsys.readouterr().err
    assert not (tmp_path / "res").exists()
    for fine in ("", "0"):
        monkeypatch.setenv("P4_SMOKE", fine)
        assert _cli(tmp_path, "status") == 0
    assert not (tmp_path / "res" / "smoke").exists()
