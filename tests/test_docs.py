import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from specdx.cli import main

RUNNING = Path(__file__).parents[1] / "docs" / "RUNNING.md"


def _subcommands(capsys) -> set[str]:
    with pytest.raises(SystemExit):
        main(["--help"])
    match = re.search(r"\{([a-z,-]+)\}", capsys.readouterr().out)
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


def _section_11() -> str:
    text = RUNNING.read_text(encoding="utf-8")
    return text.split("## 11. P2b", 1)[1].split("## 12.", 1)[0]


def test_the_g3_check_is_the_v3_reading_check_not_v2s_prior_mode() -> None:
    section = _section_11()
    assert "NO READINGS" in section
    assert "at least one reading for each population model" in section
    # P1b check (4) tested |θ − 0.75| < 1e-9 and can never fire under v3 (θ = σ(1.0) = 0.731)
    assert "fabs" not in section and "NO ACTED BEHAVIOUR" not in section
    assert "replaces P1b's check (4)" in section


def test_the_resume_wording_says_to_rerun_the_same_command() -> None:
    section = _section_11()
    assert ("re-run the same command once the tunnel is back; "
            "finished runs come from the cache (§7)") in section
    assert "the gate resumes (§7)" not in section


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
def test_the_reading_check_flags_a_population_model_without_readings(tmp_path) -> None:
    block = next(b for b in re.findall(r"```bash\n(.*?)```", _section_11(), re.S)
                 if "NO READINGS" in b)
    start = block.index("jq -r --slurpfile g")
    end = block.index("results/p2-smoke-base.json", start) + len("results/p2-smoke-base.json")
    command = shlex.split(block[start:end])
    readings = {"gpt-oss:20b": 3, "qwen3:30b-a3b": 2}
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"cells": {"c1": {"readings": readings},
                                            "c2": {"readings": {"gpt-oss:20b": 1}}}}))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"models_by_family": {
        "openai": ["gpt-oss:20b"], "qwen": ["qwen3:30b-a3b", "qwen2.5:14b"]}}))
    command[command.index("experiments/p2-gate.json")] = str(gate)
    command[-1] = str(report)
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    assert out.splitlines() == ["gpt-oss:20b\t4 readings\tok", "qwen3:30b-a3b\t2 readings\tok",
                                "qwen2.5:14b\t0 readings\tNO READINGS"]


def test_the_reproduce_block_names_the_commits_and_the_commands() -> None:
    text = RUNNING.read_text(encoding="utf-8")
    section = text.split("## 12. Reproducing G0 and G1", 1)[1].split("## Bắt đầu", 1)[0]
    for needle in ("46beddf", "a97a67c", "e9aac9b", "specdx.sim.calibrate", "specdx.sim.g0 run",
                   "--params-out", "specdx simulate --sweep all", "experiments/p2_replay.py",
                   "db1e1823c66df102648c69eb6a8789b3"):
        assert needle in section, needle
