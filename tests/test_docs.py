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
