import logging
from pathlib import Path

import pytest

from specdx.cli import main, mutants_path
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


def test_only_regenerates_named_entries_at_one_variant_without_force(monkeypatch, capsys) -> None:
    saved: list[TellVariants] = []
    asked = []

    def fake_regenerate(variants, keys, index, *, model, seed, original):
        asked.append((keys, index, model, seed, original))
        return TellVariants(variants.standing, {"x": ("1", "2-new", "3")})

    monkeypatch.setattr("specdx.interpreters.prompt.load_variants", _frozen)
    monkeypatch.setattr("specdx.interpreters.prompt.save_variants", saved.append)
    monkeypatch.setattr("specdx.interpreters.paraphrase.regenerate", fake_regenerate)
    assert main(["make-variants", "--only", "x,standing", "--variant", "1",
                 "--model", "qwen2.5:14b"]) == 0
    assert asked == [(["x", "standing"], 1, "qwen2.5:14b", None, False)]
    assert saved[0].statements["x"] == ("1", "2-new", "3")


def test_only_needs_a_variant_index(monkeypatch, capsys) -> None:
    monkeypatch.setattr("specdx.interpreters.prompt.load_variants", _frozen)
    assert main(["make-variants", "--only", "x"]) == 2
    assert "--variant" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--only", "x", "--variant", "3"],
                                  ["--only", "x", "--variant", "0"],
                                  ["--only", "nope", "--variant", "1"]])
def test_a_bad_entry_or_index_is_refused_before_any_model_call(monkeypatch, capsys, args) -> None:
    saved: list[TellVariants] = []
    monkeypatch.setattr("specdx.interpreters.prompt.load_variants", _frozen)
    monkeypatch.setattr("specdx.interpreters.prompt.save_variants", saved.append)
    assert main(["make-variants", *args]) == 2
    assert "error:" in capsys.readouterr().err and saved == []


def test_force_regenerates_and_saves_the_paraphrases(monkeypatch, capsys) -> None:
    saved: list[TellVariants] = []
    monkeypatch.setattr("specdx.interpreters.prompt.load_variants", _frozen)
    monkeypatch.setattr("specdx.interpreters.prompt.save_variants", saved.append)
    monkeypatch.setattr("specdx.interpreters.paraphrase.make_variants",
                        lambda variants, *, model, count: _frozen())
    assert main(["make-variants", "--force"]) == 0
    assert saved == [_frozen()] and "variants: 3" in capsys.readouterr().out


def test_per_mutant_results_are_named_after_the_gate_output() -> None:
    assert mutants_path(Path("results/p1b-gate.json")) == Path("results/p1b-gate-mutants.jsonl")
    assert mutants_path(Path("results/p1-gate.json")) == Path("results/p1-gate-mutants.jsonl")
