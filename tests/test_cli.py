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
    for command in ("build-tell", "make-variants", "capability", "screen", "mutants", "run",
                    "gate"):
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


def test_the_gate_refuses_an_empty_population_before_building_any_pack(
    monkeypatch, tmp_path
) -> None:
    config = tmp_path / "gate.json"
    config.write_text('{"models_by_family": {}, "variants": [0], "seeds": [1], "budget": 1, '
                      '"dev_per_type": 1, "rng_seed": 1}', encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("a pack was built before the population was checked")

    monkeypatch.setattr("specdx.workbench.Workbench.open", boom)
    monkeypatch.setattr("specdx.mutate.operators.all_mutants", boom)
    with pytest.raises(ValueError, match="empty"):
        main(["gate", "--config", str(config), "--out", str(tmp_path / "g.json")])


def test_screen_writes_both_files_and_exits_3_on_the_stop_rule(monkeypatch, tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from tests.eval.test_screen import BASE, CANON, _Lens, _Runner

    profile = tmp_path / "gpu.json"
    profile.write_text(json.dumps({"profile": "gpu", "workers": 1, "ollama_num_parallel": 1,
                                   "candidates": {"a": ["good"], "b": ["idle"]}}))
    bench = SimpleNamespace(base=BASE, runner=_Runner(),
                            loop_context=lambda v, c: SimpleNamespace(lens=_Lens(), canon=CANON))
    monkeypatch.setattr("specdx.workbench.Workbench.open", lambda workers: bench)
    out = tmp_path / "results" / "screen"
    assert main(["screen", "--profile", str(profile), "--out", str(out)]) == 3
    assert json.loads((tmp_path / "results" / "screen.json").read_text())["stop"] is True
    assert "STOP" in (tmp_path / "results" / "screen.md").read_text()
