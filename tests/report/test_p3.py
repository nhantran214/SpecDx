"""The P3 stage functions, run on fake trajectory files (no network, no recursion)."""

import json
from pathlib import Path

from specdx.report.p3 import fetch_stage, local_stage, replication_stage
from specdx.report.stages import StageOutcome
from specdx.subjects.sopbench.diagnose import PROVISIONAL
from specdx.subjects.sopbench.load import load_tasks

NAME = "output/library/ast_{m}-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
MODELS = {"gpt-4o-mini": ("openai", False), "qwen2.5-7b-instruct": ("qwen", True)}


def _record(task, ok=True, db=True):
    ev = {"action_successfully_called": ok, "database_match": db,
          "constraint_not_violated": True, "dirgraph_satisfied": True,
          "no_tool_call_error": True}
    return {"domain": "library", "setup": {}, "task": {"user_prompt": task.user_prompt},
            "interactions": [{"prompt": f"P{task.index}", "interaction": [], "database": {}}],
            "evaluations": [ev]}


def _write(root: Path, model: str, records: list[dict]) -> dict:
    path = NAME.format(m=model)
    target = root / "traj" / "library" / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records))
    family, open_w = MODELS[model]
    return {"path": path, "domain": "library", "model": model, "canonical": model,
            "family": family, "open_weights": open_w, "mode": "fc", "size": 1,
            "sha256": "x"}


def _setup(tmp_path: Path, misaligned: bool = False) -> tuple[Path, Path]:
    tasks = load_tasks("library")
    second = [_record(t, db=t.index % 5 != 0) for t in tasks]
    files = [
        _write(tmp_path, "gpt-4o-mini", [_record(t) for t in tasks]),
        _write(tmp_path, "qwen2.5-7b-instruct", second[:-1] if misaligned else second),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"commit": "c", "files": files}))
    return manifest, tmp_path / "traj"


def test_replication_stage_on_two_fake_files_writes_both_tables_and_the_provisional_line(
        tmp_path):
    manifest, traj = _setup(tmp_path)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert (out.exit_code, out.verdict) == (0, "DONE")
    assert PROVISIONAL in out.body
    assert "not refuted offline" in out.body
    assert "| raters | runs | excluded runs |" in out.body
    assert "contradiction (unrefuted)" in out.body
    assert "Table 2" in out.body
    assert "library" in out.body
    assert out.key_number.endswith("findings over 1 domains")
    assert set(out.data) >= {"domains", "defects", "theta"}
    print(out.body)


def test_replication_lists_a_misaligned_file_as_excluded(tmp_path):
    manifest, traj = _setup(tmp_path, misaligned=True)
    out = replication_stage(manifest=manifest, trajectories=traj)
    assert out.exit_code == 0
    assert NAME.format(m="qwen2.5-7b-instruct") in out.body
    assert "Excluded files" in out.body


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
