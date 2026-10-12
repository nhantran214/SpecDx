import re
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
