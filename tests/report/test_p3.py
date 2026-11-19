"""The P3 stage functions, run on fake trajectory files (no network, no recursion)."""

import hashlib
import json
from pathlib import Path

from specdx.report.p3 import (
    MISSING_EVALUATIONS,
    SHA_MISMATCH,
    fetch_stage,
    local_stage,
    rank_findings,
    replication_stage,
)
from specdx.report.stages import StageOutcome
from specdx.subjects.sopbench.diagnose import PROVISIONAL
from specdx.subjects.sopbench.load import load_tasks
from specdx.types import Defect

NAME = "output/library/ast_{m}-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
MODELS = {"gpt-4o-mini": ("openai", False), "qwen2.5-7b-instruct": ("qwen", True)}
N = 66  # library tasks


def _record(task, ok=True, db=True, err=False, evaluated=True, prompt=None):
    ev = {"action_successfully_called": ok, "database_match": db,
          "constraint_not_violated": True, "dirgraph_satisfied": True,
          "no_tool_call_error": not err}
    fields = {"user_prompt": task.user_prompt, "user_goal": task.user_goal,
              "action_should_succeed": task.should_succeed, "user_known": task.user_known,
              "constraint_parameters": task.constraint_parameters}
    return {"domain": "library", "setup": {}, "task": fields,
            "interactions": [{"prompt": prompt or f"P{task.index}", "interaction": [],
                              "database": {}}],
            "evaluations": [ev] if evaluated else []}


def _write(root: Path, model: str, records: list[dict], sha: str | None = None) -> dict:
    path = NAME.format(m=model)
    target = root / "traj" / "library" / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records))
    family, open_w = MODELS[model]
    return {"path": path, "domain": "library", "model": model, "canonical": model,
            "family": family, "open_weights": open_w, "mode": "fc", "size": 1,
            "sha256": sha or hashlib.sha256(target.read_bytes()).hexdigest()}


def _setup(tmp_path: Path, misaligned: bool = False, second: list[dict] | None = None,
           sha: str | None = None) -> tuple[Path, Path]:
    tasks = load_tasks("library")
    if second is None:
        second = [_record(t, db=t.index % 5 != 0) for t in tasks]
    files = [
        _write(tmp_path, "gpt-4o-mini", [_record(t) for t in tasks]),
        _write(tmp_path, "qwen2.5-7b-instruct", second[:-1] if misaligned else second, sha),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"commit": "c", "files": files}))
    return manifest, tmp_path / "traj"


def _rows(out) -> dict[str, dict]:
    return {r["composition"]: r for r in out.data["table1"] if r["domain"] == "library"}


def test_replication_stage_on_two_fake_files_writes_both_tables_and_the_provisional_line(
        tmp_path):
    manifest, traj = _setup(tmp_path)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert (out.exit_code, out.verdict) == (0, "DONE")
    assert PROVISIONAL in out.body
    assert "not refuted offline" in out.body
    assert "| raters | runs | excluded runs |" in out.body
    assert "Table 2" in out.body
    assert out.key_number.endswith("findings over 1 domains")
    assert set(out.data) >= {"domains", "defects", "theta", "table1", "loading", "variants"}
    rows = _rows(out)
    assert [(rows[c]["raters"], rows[c]["runs"], rows[c]["excluded runs"])
            for c in ("all", "open", "hosted")] == [(2, 2 * N, 0), (1, N, 0), (1, N, 0)]
    # The plain contradiction column needs a refutation, which never runs offline (M-7).
    assert all(r["contradiction"] == "n/a offline" for r in rows.values())
    unrefuted = sum(d["type"] == "contradiction" for d in out.data["defects"])
    assert rows["all"]["contradiction (unrefuted)"] == unrefuted
    assert rows["all"]["hidden"] == sum(d["type"] == "hidden" for d in out.data["defects"])
    assert "| library | all | 2 | 132 | 0 |" in out.body
    assert out.data["loading"]["library"] == {
        "files": 2, "loaded": 2, "excluded": 0, "records": 2 * N, "runs": 2 * N,
        "missing_evaluation": 0}


def test_replication_reports_saturation_and_distinct_tasks(tmp_path):
    manifest, traj = _setup(tmp_path)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert "≥0.999" in out.body and "distinct tasks" in out.body
    sat = out.data["saturation"]["library/all"]
    defects = out.data["defects"]
    assert sat["findings"] == len(defects)
    assert sat["at_0.999"] == sum(d["score"] >= 0.999 for d in defects)
    hidden_tasks = {d["location"][0].split(":")[2] for d in defects if d["type"] == "hidden"}
    assert sat["distinct_tasks"]["hidden"] == len(hidden_tasks)
    assert all("deviating" in d for d in defects)


def test_rank_findings_breaks_score_ties_by_deviating_readings():
    a = Defect("hidden", ("s3:x:1:outcome",), ("x:1",), (), 1.0)
    b = Defect("hidden", ("s3:x:2:outcome",), ("x:2",), (), 1.0)
    c = Defect("weak", ("s3:x:3:process",), ("x:3",), (), 0.7)
    dev = {a.key: 3, b.key: 9, c.key: 20}
    assert rank_findings([a, c, b], dev) == [b, a, c]


def test_replication_lists_a_misaligned_file_as_excluded(tmp_path):
    manifest, traj = _setup(tmp_path, misaligned=True)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert out.exit_code == 0
    assert NAME.format(m="qwen2.5-7b-instruct") in out.body
    assert "Excluded files" in out.body
    assert _rows(out)["all"]["raters"] == 1


def test_a_file_without_evaluations_is_excluded_and_counted(tmp_path):
    # Ruling I-2: no silent drops. gpt-5-mini's bank, dmv and university files are like this.
    unevaluated = [_record(t, evaluated=False) for t in load_tasks("library")]
    manifest, traj = _setup(tmp_path, second=unevaluated)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert out.exit_code == 0
    assert f"`{NAME.format(m='qwen2.5-7b-instruct')}`: {MISSING_EVALUATIONS}" in out.body
    assert out.data["excluded_files"] == [NAME.format(m="qwen2.5-7b-instruct")]
    assert out.data["loading"]["library"] == {
        "files": 2, "loaded": 1, "excluded": 1, "records": 2 * N, "runs": N,
        "missing_evaluation": N}
    assert f"| library | 2 | 1 | 1 | {2 * N} | {N} | {N} |" in out.body
    assert _rows(out)["all"]["raters"] == 1


def test_a_file_whose_sha256_differs_from_the_manifest_is_excluded(tmp_path):
    # Ruling I-4.
    manifest, traj = _setup(tmp_path, sha="0" * 64)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert out.exit_code == 0
    assert f"`{NAME.format(m='qwen2.5-7b-instruct')}`: {SHA_MISMATCH}" in out.body
    assert _rows(out)["all"]["raters"] == 1


def test_a_missing_trajectory_file_is_excluded(tmp_path):
    manifest, traj = _setup(tmp_path)
    (traj / "library" / Path(NAME.format(m="qwen2.5-7b-instruct")).name).unlink()
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert out.exit_code == 0
    assert out.data["excluded_files"] == [NAME.format(m="qwen2.5-7b-instruct")]


def test_every_file_excluded_fails_with_exit_2(tmp_path):
    manifest, traj = _setup(tmp_path)
    for f in (traj / "library").iterdir():
        f.unlink()
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert (out.exit_code, out.verdict) == (2, "FAIL")
    assert "library: no runs" in out.body


def test_a_domain_with_zero_readings_fails_with_exit_2(tmp_path):
    errors = [_record(t, err=True) for t in load_tasks("library")]
    manifest, traj = _setup(tmp_path, second=errors)
    tasks = load_tasks("library")
    path = traj / "library" / Path(NAME.format(m="gpt-4o-mini")).name
    path.write_text(json.dumps([_record(t, err=True) for t in tasks]))
    files = json.loads(manifest.read_text())
    files["files"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(files))
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert (out.exit_code, out.verdict) == (2, "FAIL")
    assert "library: no readings" in out.body


def test_prompt_variants_are_reported_per_domain_with_their_models(tmp_path):
    # Ruling I-3: qwen saw another render of tasks 0-9; on a 1-1 tie the string decides.
    tasks = load_tasks("library")
    second = [_record(t, prompt=f"Q{t.index}" if t.index < 10 else None) for t in tasks]
    manifest, traj = _setup(tmp_path, second=second)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert out.data["variants"]["library"] == {
        "runs": {"0": 2 * N - 10, "1": 10}, "models": {"qwen2.5-7b-instruct": 10}}
    assert "confounded with model generation" in out.body
    assert "- library: variant 0: 122 runs; other variants: 10 runs" in out.body


def test_replication_without_a_manifest_fails_with_exit_2(tmp_path):
    out = replication_stage(manifest=tmp_path / "none.json", trajectories=tmp_path)
    assert (out.exit_code, out.verdict) == (2, "FAIL")
    assert "manifest" in out.body


def test_replication_without_a_trajectory_directory_fails_with_exit_2(tmp_path):
    manifest, _ = _setup(tmp_path)
    out = replication_stage(manifest=manifest, trajectories=tmp_path / "missing")
    assert (out.exit_code, out.verdict) == (2, "FAIL")


def test_fetch_stage_turns_a_network_failure_into_a_fail_outcome(tmp_path):
    def tree():
        raise OSError("network down")

    out = fetch_stage(tmp_path / "d", tmp_path / "m.json", opener=lambda u: b"", tree=tree)
    assert (out.exit_code, out.verdict) == (2, "FAIL")
    assert "network down" in out.body


def test_fetch_stage_downloads_the_selected_files(tmp_path):
    listing = [{"path": NAME.format(m="gpt-4o-mini"), "size": 20_000}]
    out = fetch_stage(tmp_path / "d", tmp_path / "m.json", opener=lambda u: b"[1]",
                      tree=lambda: listing)
    assert (out.exit_code, out.verdict) == (0, "DONE")
    assert out.key_number == "1 files, 1 models, 1 families"
    assert "openai" in out.body


def test_local_stage_passes_and_fails_with_the_runner(tmp_path):
    calls = []

    def ok(cmd):
        calls.append(cmd)
        return 0, "fine"

    out = local_stage(runner=ok, manifest=tmp_path / "none.json", trajectories=tmp_path)
    assert (out.exit_code, out.verdict) == (0, "PASS")
    assert len(calls) == 2
    assert "replication smoke skipped: run fetch-sopbench first" in out.body
    bad = local_stage(runner=lambda c: (1, "boom"), manifest=tmp_path / "none.json",
                      trajectories=tmp_path)
    assert (bad.exit_code, bad.verdict) == (1, "FAIL")
    assert isinstance(bad, StageOutcome)
