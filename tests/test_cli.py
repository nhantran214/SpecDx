import logging

import pytest

from specdx.cli import main
from specdx.interpreters.prompt import TellVariants


def test_help_lists_every_subcommand(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("build-tell", "make-variants", "capability", "mutants", "run", "gate"):
        assert command in out


def _frozen() -> TellVariants:
    return TellVariants(("a", "b", "c"), {"x": ("1", "2", "3")})


def test_frozen_paraphrases_are_not_regenerated_without_force(monkeypatch, capsys) -> None:
    def no_model(*args, **kwargs):
        raise AssertionError("the model must not be called")

    monkeypatch.setattr("specdx.interpreters.prompt.load_variants", _frozen)
    monkeypatch.setattr("specdx.interpreters.paraphrase.make_variants", no_model)
    assert main(["make-variants"]) == 2
    assert "frozen" in capsys.readouterr().err


def test_the_cli_logs_specdx_progress_to_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setattr("specdx.interpreters.prompt.load_variants", _frozen)
    main(["make-variants"])
    capsys.readouterr()
    logging.getLogger("specdx.observe.run").info("run base grn12 ok")
    assert "run base grn12 ok" in capsys.readouterr().err
    assert logging.getLogger("specdx").getEffectiveLevel() == logging.INFO


def test_force_regenerates_and_saves_the_paraphrases(monkeypatch, capsys) -> None:
    saved: list[TellVariants] = []
    monkeypatch.setattr("specdx.interpreters.prompt.load_variants", _frozen)
    monkeypatch.setattr("specdx.interpreters.prompt.save_variants", saved.append)
    monkeypatch.setattr("specdx.interpreters.paraphrase.make_variants",
                        lambda variants, *, model, count: _frozen())
    assert main(["make-variants", "--force"]) == 0
    assert saved == [_frozen()] and "variants: 3" in capsys.readouterr().out
