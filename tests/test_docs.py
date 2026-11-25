import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from specdx.cli import main

ROOT = Path(__file__).parents[1]
RUNNING = ROOT / "docs" / "RUNNING.md"


def _subcommands(capsys) -> set[str]:
    with pytest.raises(SystemExit):
        main(["--help"])
    match = re.search(r"\{([a-z0-9,-]+)\}", capsys.readouterr().out)
    assert match is not None
    return set(match.group(1).split(","))


def test_every_command_the_guide_shows_exists(capsys) -> None:
    text = RUNNING.read_text(encoding="utf-8")
    shown = set(re.findall(r"specdx ([a-z][a-z-]+)", text))
    assert shown and shown <= _subcommands(capsys)


@pytest.mark.parametrize("needle", [
    "OLLAMA_NUM_PARALLEL", "SPECDX_OLLAMA_URL", "SPECDX_WORK", "SPECDX_DATABASE_URL",
    "kill -STOP", "kill -CONT", "never start a second", "cache-info", "specdx doctor",
    "specdx screen", "setsid nohup", "## Bắt đầu nhanh (tiếng Việt)", "Troubleshooting",
])
def test_the_guide_covers_what_the_spec_asks(needle) -> None:
    assert needle.lower() in RUNNING.read_text(encoding="utf-8").lower()


def test_the_reproduce_block_names_the_commits_and_the_commands() -> None:
    text = RUNNING.read_text(encoding="utf-8")
    section = text.split("## 12. Reproducing G0 and G1", 1)[1].split("## Bắt đầu", 1)[0]
    for needle in ("46beddf", "a97a67c", "e9aac9b", "specdx.sim.calibrate", "specdx.sim.g0 run",
                   "--params-out", "specdx simulate --sweep all", "experiments/p2_replay.py",
                   "db1e1823c66df102648c69eb6a8789b3"):
        assert needle in section, needle
    # the scratch directory lies outside work/, the run cache p2_replay refuses to write to
    assert "out=$(mktemp -d)" in section and "work/repro" not in section
    commands = section.split("```bash\n", 1)[1].split("```", 1)[0]
    assert "--out work/" not in commands and "--out \"work/" not in commands
    assert commands.count('"$out/') >= 8
    # the recorded run times (results/p2-g0.json seconds 3354; progress ledger: study 1195 s)
    assert "about 56 min (3354 s recorded)" in section
    assert "about 20 min (1195 s recorded)" in section
    g0_seconds = json.loads((RUNNING.parents[1] / "results" / "p2-g0.json")
                            .read_text(encoding="utf-8"))["seconds"]
    assert round(g0_seconds) == 3354


def _run_all(tmp_path, *args):
    env = {**os.environ, "P3_RESULTS": str(tmp_path / "results"), "P3_WORK": str(tmp_path / "work")}
    return subprocess.run(["bash", str(ROOT / "run_all.sh"), *args], capture_output=True,
                          text=True, env=env, cwd=ROOT)


def test_run_all_usage_lists_every_p3_stage(tmp_path) -> None:
    from specdx.report.stages import STAGES

    r = _run_all(tmp_path)
    assert r.returncode == 2
    for st in STAGES:
        assert st.name in r.stdout, st.name
    assert "all" in r.stdout and "status" in r.stdout


def test_a_later_plan_stage_is_pending_and_exits_0(tmp_path) -> None:
    r = _run_all(tmp_path, "s2-screen")
    assert r.returncode == 0, r.stderr
    assert "s2-screen: pending \u2014 implemented in plan P3c" in r.stdout
    assert (tmp_path / "results" / "STATUS.md").exists()


def test_run_all_no_longer_offers_case_0031_stages(tmp_path) -> None:
    usage = _run_all(tmp_path).stdout
    for gone in ("g3", "gate", "preflight", "freeze-baselines", "baselines-dev"):
        assert not re.search(rf"(?<![\w-]){gone}(?![\w-])", usage), gone


def test_run_all_status_leaves_no_pid_file(tmp_path) -> None:
    r = _run_all(tmp_path, "status")
    assert r.returncode == 0, r.stderr
    assert str(tmp_path / "results" / "STATUS.md") in r.stdout
    assert not list((tmp_path / "work").glob("*.pid"))


def test_p3_status_done_reflects_the_stage_metadata(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    args = ["p3", "status", "--results", str(results), "--work", str(tmp_path / "w"),
            "--done", "local"]
    assert main(args) == 1
    (results / "local.json").write_text(json.dumps({"exit_code": 0}))
    assert main(args) == 0


def test_p3_status_warns_when_a_done_stage_was_recorded_at_another_commit(
        tmp_path, capsys) -> None:
    # M-3: `run_all.sh all` skips a done stage; after new commits that result is stale.
    results = tmp_path / "results"
    results.mkdir()
    (results / "replication.json").write_text(json.dumps({"exit_code": 0, "commit": "0000000"}))
    args = ["p3", "status", "--results", str(results), "--work", str(tmp_path / "w")]
    assert main([*args, "--done", "replication"]) == 0
    err = capsys.readouterr().err
    assert "warning: replication was recorded at commit 0000000" in err
    assert main(args) == 0
    assert "warning: replication was recorded at commit 0000000" in capsys.readouterr().err


def test_the_old_run_all_steps_are_marked_historical() -> None:
    section = RUNNING.read_text(encoding="utf-8").split("## 13.", 1)[1].split("\n## ", 1)[0]
    banner = section.split("\n\n", 2)[1]
    assert banner.startswith("> **Historical.**") and "no longer exist" in banner


def test_the_readme_troubleshooting_row_matches_the_local_stage() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`local` fails with `near_ctx`" not in readme
    assert "`run_all.sh local` fails" in readme


def test_running_guide_warns_about_stale_results() -> None:
    text = RUNNING.read_text(encoding="utf-8")
    assert "recorded at another commit" in text


def test_running_guide_describes_p3() -> None:
    text = RUNNING.read_text(encoding="utf-8")
    assert "P3: SOPBench and \u03c4\u00b2-bench" in text
    assert "results/p3/STATUS.md" in text and "PYTHONHASHSEED=0" in text


def test_running_guide_describes_p4a_and_smoke_mode() -> None:
    text = RUNNING.read_text(encoding="utf-8")
    for stage in ("s3-g0v4", "s3-witness", "s3-v4", "s3-placebo"):
        assert f"| `{stage}` |" in text
    assert "./run_all.sh smoke" in text and "P4_SMOKE=1" in text
    assert "SMOKE RUN — system check only, not a result" in text
    assert "results/p3/smoke/" in text and "48 GB host" in text


def test_running_guide_documents_the_official_p4a_run() -> None:
    text = RUNNING.read_text(encoding="utf-8")
    section = text.split("### The official P4a run on the 48 GB host", 1)[1].split("\n## ", 1)[0]
    # J-1 and F-I6: back to run_all.sh, with the fetch step a fresh host needs first.
    assert "./run_all.sh fetch-sopbench" in section
    assert "P4_JOBS=N ./run_all.sh s3-g0v4" in section and "specdx p3 s3-g0v4" not in section
    assert "results: P4a — frozen v4 parameters for S3 (G0v4)" in section
    assert "./run_all.sh s3-witness s3-v4 s3-placebo" in section
    assert "results/p3/*.md" in section and "work/g0v4-cache" in section
    assert "never results" in section and "280 v4 fits" in section
    # Task 11 Minors: the cost figures and the FAIL sentence are pinned.
    for needle in ("about 36,000 s of CPU", "about 10 CPU-hours", "about 0.6 hours at `P4_JOBS=20`",
                   "estimated 479 s and took 466 s", "allow up to about 2 hours",
                   "A G0v4 FAIL (exit 1) freezes nothing, and the later stages then exit 2 for "
                   "want of parameters"):
        assert needle in section, needle
    assert "`local`, `fetch-sopbench` (the only stage that uses the network), `replication`" in text


def test_run_all_usage_documents_smoke_mode(tmp_path) -> None:
    usage = _run_all(tmp_path).stdout
    assert "./run_all.sh smoke" in usage and "P4_SMOKE" in usage
    assert "results/p3/smoke" in usage or "$P3_RESULTS/smoke/" in usage


def test_run_all_smoke_refuses_a_non_p4a_stage(tmp_path) -> None:
    env = {**os.environ, "P3_RESULTS": str(tmp_path / "results"),
           "P3_WORK": str(tmp_path / "work"), "P4_SMOKE": "1"}
    r = subprocess.run(["bash", str(ROOT / "run_all.sh"), "replication"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    assert r.returncode == 2 and "P4a stages only" in r.stderr
    assert not (tmp_path / "results" / "smoke" / "replication.md").exists()


def test_running_guide_and_usage_warn_that_all_runs_the_official_p4a_stages(tmp_path) -> None:
    text = RUNNING.read_text(encoding="utf-8")
    assert "`./run_all.sh all` includes the official P4a stages" in text
    assert "48 GB host only" in text
    assert "OFFICIAL P4a stages" in _run_all(tmp_path).stdout
    script = (ROOT / "run_all.sh").read_text(encoding="utf-8")
    assert "all includes the official P4a stages" in script  # the run-time warning


def test_run_all_smoke_on_a_pending_stage_exits_2_like_the_cli(tmp_path) -> None:
    env = {**os.environ, "P3_RESULTS": str(tmp_path / "results"),
           "P3_WORK": str(tmp_path / "work"), "P4_SMOKE": "1"}
    r = subprocess.run(["bash", str(ROOT / "run_all.sh"), "s2-screen"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    assert r.returncode == 2 and "P4a stages only" in r.stderr
    assert "pending" not in r.stdout


def test_run_all_rejects_a_bad_p4_smoke_value(tmp_path) -> None:
    env = {**os.environ, "P3_RESULTS": str(tmp_path / "results"),
           "P3_WORK": str(tmp_path / "work"), "P4_SMOKE": "yes"}
    r = subprocess.run(["bash", str(ROOT / "run_all.sh"), "status"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    assert r.returncode == 2 and "P4_SMOKE must be" in r.stderr


def test_smoke_exit_code_is_the_maximum_so_a_crash_is_never_hidden(tmp_path) -> None:
    def code(*codes: str) -> str:
        r = subprocess.run(
            ["bash", "-c", f'source "{ROOT / "run_all.sh"}"; max_code {" ".join(codes)}'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    assert code("1", "0", "2", "1") == "2"  # an expected FAIL (1) first, a crash (2) later
    assert code("2", "1") == "2"
    assert code("0", "0") == "0" and code("1", "0") == "1"


def test_run_all_smoke_prints_a_summary_and_returns_the_largest_code(tmp_path) -> None:
    env = {**os.environ, "P3_RESULTS": str(tmp_path / "results"),
           "P3_WORK": str(tmp_path / "work")}
    r = subprocess.run(["bash", str(ROOT / "run_all.sh"), "smoke"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    # No trajectories here: s3-witness passes (0); the other three stop with exit 2.
    assert r.returncode == 2, r.stderr
    summary = r.stdout.split("Smoke summary")[1]
    for stage, rc, verdict in (("s3-g0v4", "2", "FAIL"), ("s3-witness", "0", "DONE"),
                               ("s3-v4", "2", "FAIL"), ("s3-placebo", "2", "FAIL")):
        assert re.search(rf"{stage}\s+{rc}\s+{verdict}", summary), (stage, summary)
    assert not (tmp_path / "results" / "STATUS.md").exists()


# --- Rulings J-1, F-I6 and F-I7 (final fix wave): run_all.sh ---

def _sourced(tmp_path, script: str, **env) -> subprocess.CompletedProcess:
    """Source run_all.sh (which sets -euo pipefail, cds to the repository, validates
    P4_SMOKE and P4_JOBS and installs its traps, but runs no stage) and run `script`."""
    full = {**os.environ, "P3_RESULTS": str(tmp_path / "results"),
            "P3_WORK": str(tmp_path / "work"), **env}
    return subprocess.run(["bash", "-c", f'source "{ROOT / "run_all.sh"}"; {script}'],
                          capture_output=True, text=True, cwd=ROOT, env=full)


def test_run_all_runs_every_stage_it_is_given_in_order(tmp_path) -> None:
    r = _run_all(tmp_path, "s2-screen", "status", "s2-dev")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out.index("s2-screen: pending") < out.index("s2-dev: pending")


def test_run_all_stops_at_the_first_non_zero_exit(tmp_path) -> None:
    # s3-v4 has no parameters and no data here: exit 2, and s2-screen never runs.
    r = _run_all(tmp_path, "s3-v4", "s2-screen")
    assert r.returncode == 2
    assert "s2-screen: pending" not in r.stdout


def test_run_all_checks_every_stage_name_before_running_any(tmp_path) -> None:
    r = _run_all(tmp_path, "s2-screen", "no-such-stage")
    assert r.returncode == 2 and "unknown stage 'no-such-stage'" in r.stderr
    assert "s2-screen: pending" not in r.stdout


def test_p4_jobs_defaults_to_the_cores_and_must_be_a_positive_integer(tmp_path) -> None:
    r = _sourced(tmp_path, 'echo "$JOBS"', P4_JOBS="")
    assert r.returncode == 0 and r.stdout.strip() == str(os.cpu_count())
    assert _sourced(tmp_path, 'echo "$JOBS"', P4_JOBS="3").stdout.strip() == "3"
    for bad in ("0", "-2", "many", "2.5"):
        r = _sourced(tmp_path, "true", P4_JOBS=bad)
        assert r.returncode == 2 and "P4_JOBS must be" in r.stderr, bad


def test_a_parallel_stage_gets_its_jobs_and_one_blas_thread(tmp_path) -> None:
    # J-1 and F-I7: s3-g0v4 gets --jobs $P4_JOBS, with OMP, OpenBLAS and MKL at one thread
    # each, set before the stage's Python (and numpy) starts; other stages get neither.
    script = ('p3() { echo "ARGS $* T=${OMP_NUM_THREADS:-}/${OPENBLAS_NUM_THREADS:-}/'
              '${MKL_NUM_THREADS:-}"; }; rebuild_status() { :; }; '
              'run_stage s3-g0v4; run_stage s3-witness')
    r = _sourced(tmp_path, script, P4_JOBS="3", OMP_NUM_THREADS="",
                 OPENBLAS_NUM_THREADS="", MKL_NUM_THREADS="")
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("ARGS")]
    assert lines == ["ARGS s3-g0v4 --jobs 3 T=1/1/1", "ARGS s3-witness T=//"]
