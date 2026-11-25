import json
import os

from specdx.report import stages
from specdx.report.stages import (
    STAGES,
    StageOutcome,
    already_done,
    run_stage,
    stale_commit,
    update_status,
)


def fake_git(args):
    return " M file.py\n" if args[0] == "status" else "abc1234\n"


def _run(tmp_path, name, fn, **kw):
    return run_stage(
        name, fn, command=f"specdx p3 {name}", results=tmp_path / "res",
        work=tmp_path / "work", git=fake_git, **kw,
    )


def test_run_stage_writes_header_rows_and_body(tmp_path):
    code = _run(
        tmp_path, "local",
        lambda: StageOutcome(0, "PASS", "42", "hello body", {"a": 1}),
        subject="s", split="dev",
    )
    assert code == 0
    text = (tmp_path / "res" / "local.md").read_text()
    for row in ("Started", "Ended", "Duration", "Commit", "Command", "Exit code",
                "Subject", "Split", "Verdict"):
        assert f"| {row} |" in text
    assert "PASS" in text and "abc1234 (dirty)" in text
    assert "## Results" in text and "hello body" in text
    assert json.loads((tmp_path / "res" / "local.data.json").read_text()) == {"a": 1}
    assert json.loads((tmp_path / "res" / "local.json").read_text())["exit_code"] == 0
    assert (tmp_path / "res" / "STATUS.md").exists()


def test_an_exception_becomes_exit_2_with_the_traceback_in_the_report(tmp_path):
    def boom():
        raise ValueError("kaboom-marker")

    assert _run(tmp_path, "local", boom) == 2
    text = (tmp_path / "res" / "local.md").read_text()
    assert "kaboom-marker" in text and "FAIL" in text and "Traceback" in text


def test_already_done_only_after_exit_0(tmp_path):
    res = tmp_path / "res"
    assert not already_done("local", res)
    _run(tmp_path, "local", lambda: StageOutcome(1, "FAIL", "", "x"))
    assert not already_done("local", res)
    _run(tmp_path, "local", lambda: StageOutcome(0, "PASS", "", "x"))
    assert already_done("local", res)


def test_status_lists_every_stage_in_order_with_pending_for_later_plans(tmp_path):
    _run(tmp_path, "local", lambda: StageOutcome(0, "PASS", "7", "x"))
    path = update_status(tmp_path / "res", tmp_path / "work")
    text = path.read_text()
    assert "| Stage | Plan | Status | Verdict | Key number | Ended | Duration | Report |" in text
    pos = [text.index(f"| {s.name} |") for s in STAGES]
    assert pos == sorted(pos)
    assert "pending (plan P3b)" in text and "pending (plan P3d)" in text
    assert "not run" in text and "done" in text


def test_status_shows_running_for_a_live_pid_file(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "p3-replication.pid").write_text(str(os.getpid()))
    (work / "p3-fetch-sopbench.pid").write_text("999999999")
    text = update_status(tmp_path / "res", work).read_text()
    lines = text.splitlines()
    rep = next(x for x in lines if x.startswith("| replication |"))
    fetch = next(x for x in lines if x.startswith("| fetch-sopbench |"))
    assert "running" in rep and "running" not in fetch


def test_stage_names_unique_and_default_git_runs():
    assert len({s.name for s in STAGES}) == len(STAGES) == 23
    sha, state = stages.git_state()
    assert state in ("clean", "dirty") and sha


def test_the_status_row_reads_n_a_offline(tmp_path):
    # M-7: `p3 status` writes no metadata of its own, so "not run" would always show.
    text = update_status(tmp_path / "res", tmp_path / "work").read_text()
    row = next(x for x in text.splitlines() if x.startswith("| status |"))
    assert row.startswith("| status | P3a | n/a offline |")


def test_stale_commit_names_a_done_stage_recorded_at_another_commit(tmp_path):
    # M-3: the skip-when-done rule ignores the commit, so a stale result must be flagged.
    res = tmp_path / "res"
    _run(tmp_path, "local", lambda: StageOutcome(0, "PASS", "", "x"))
    assert stale_commit("local", res, git=fake_git) is None
    meta = json.loads((res / "local.json").read_text())
    (res / "local.json").write_text(json.dumps(meta | {"commit": "fff0000"}))
    assert stale_commit("local", res, git=fake_git) == ("fff0000", "abc1234")
    (res / "local.json").write_text(json.dumps(meta | {"commit": "fff0000", "exit_code": 1}))
    assert stale_commit("local", res, git=fake_git) is None
    assert stale_commit("replication", res, git=fake_git) is None


def test_the_p4a_stages_follow_replication_in_order_and_are_implemented():
    names = [s.name for s in STAGES]
    i = names.index("replication")
    assert names[i + 1:i + 5] == ["s3-g0v4", "s3-witness", "s3-v4", "s3-placebo"]
    assert all(s.plan == "P4a" and s.implemented for s in STAGES[i + 1:i + 5])


def test_a_smoke_report_starts_with_the_smoke_line_and_a_smoke_dashboard(tmp_path):
    res = tmp_path / "res" / "smoke"
    code = run_stage("s3-witness", lambda: StageOutcome(0, "DONE", "1", "body"),
                     command="c", results=res, work=tmp_path / "work", git=fake_git,
                     smoke=True)
    assert code == 0
    assert (res / "s3-witness.md").read_text().startswith(stages.SMOKE_LINE + "\n")
    assert json.loads((res / "s3-witness.json").read_text())["smoke"] is True
    assert (res / "STATUS.md").read_text().startswith("# SMOKE — not results")
    assert not (tmp_path / "res" / "STATUS.md").exists()
    assert not already_done("s3-witness", tmp_path / "res")  # the official stage is not run
