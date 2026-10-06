import pytest

from specdx.cli import main


def test_help_lists_every_subcommand(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("build-tell", "make-variants", "capability", "mutants", "run", "gate"):
        assert command in out
