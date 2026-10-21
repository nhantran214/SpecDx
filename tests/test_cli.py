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
    for command in ("build-tell", "make-variants", "capability", "screen", "doctor",
                    "cache-info", "mutants", "run", "gate", "baselines", "freeze-baselines",
                    "study", "report"):
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


def test_cache_info_prints_the_cached_runs(monkeypatch, tmp_path, capsys) -> None:
    from specdx.interpreters.cache import RunCache
    from specdx.types import Behaviour, InterpreterId, Scenario, State

    RunCache(tmp_path / "runs.jsonl").put(Behaviour(
        "k", InterpreterId("qwen3:14b", "qwen", 0, 1), Scenario("s", 15, 12, 0), State({}),
        model_digest="abc", spec="base"))
    monkeypatch.setattr("specdx.cli.WORK_DIR", tmp_path)
    assert main(["cache-info"]) == 0
    assert "| qwen3:14b | abc | ok | 1 |" in capsys.readouterr().out


def _run_config(tmp_path, population) -> Path:
    import json

    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"models_by_family": population, "variants": [0], "seeds": [1],
                                "budget": 1, "dev_per_type": 1, "rng_seed": 1}))
    return path


def test_run_refuses_an_empty_population_before_any_pack_is_built(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    opened = []
    monkeypatch.setattr("specdx.workbench.Workbench.open",
                        lambda workers: opened.append(workers) or SimpleNamespace())
    with pytest.raises(ValueError, match="empty"):
        main(["run", "--config", str(_run_config(tmp_path, {})),
              "--out", str(tmp_path / "r.json")])
    assert opened == []


@pytest.mark.parametrize(("variant", "localize"), [("base", False), ("H1-00", True)])
def test_run_localises_only_when_the_variant_is_not_the_base(
    monkeypatch, tmp_path, variant, localize
) -> None:
    from types import SimpleNamespace

    from specdx.active.loop import Report

    seen = []
    mutant = SimpleNamespace(id="H1-00", variant=SimpleNamespace(name="H1-00"))
    bench = SimpleNamespace(base=SimpleNamespace(name="base"), pdf=None, work_dir=tmp_path,
                            loop_context=lambda v, c: None)
    monkeypatch.setattr("specdx.workbench.Workbench.open", lambda workers: bench)
    monkeypatch.setattr("specdx.mutate.operators.all_mutants", lambda *a: [mutant])
    monkeypatch.setattr("specdx.active.loop.run_specdx",
                        lambda v, ctx, cfg: seen.append(cfg.localize) or Report(
                            v.name, [], 0, 0, [], {}))
    assert main(["run", "--config", str(_run_config(tmp_path, {"q": ["a"]})),
                 "--variant", variant, "--out", str(tmp_path / "r.json")]) == 0
    assert seen == [localize]


def test_study_refuses_without_a_passed_gate(tmp_path) -> None:
    from specdx.cli import study_guard

    gate = tmp_path / "g.json"
    gate.write_text('{"passed": false}')
    reason = study_guard(tmp_path, gate, tmp_path / "p.json")
    assert reason and "gate" in reason


def test_study_refuses_an_uncommitted_preregistration(tmp_path) -> None:
    import subprocess

    from specdx.cli import study_guard

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    gate = tmp_path / "g.json"
    gate.write_text('{"passed": true}')
    pre = tmp_path / "p.json"
    pre.write_text("{}")
    reason = study_guard(tmp_path, gate, pre)
    assert reason and "commit" in reason


def test_the_new_commands_parse() -> None:
    from specdx.cli import build_parser

    p = build_parser()
    a = p.parse_args(["baselines", "--split", "dev", "--config", "c.json", "--out", "o.json"])
    assert (a.command, a.split) == ("baselines", "dev")
    assert p.parse_args(["report", "--dev", "d.json", "--out", "r.md"]).command == "report"


def _git_repo(root: Path) -> None:
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                        *args], check=True, capture_output=True)

    git("init", "-q")
    (root / "g.json").write_text('{"passed": true}')
    (root / "p.json").write_text("{}")
    git("add", "g.json", "p.json")
    git("commit", "-q", "-m", "prereg")


def test_study_guard_admits_a_committed_preregistration_on_a_clean_tree(tmp_path) -> None:
    from specdx.cli import study_guard

    _git_repo(tmp_path)
    (tmp_path / "untracked.txt").write_text("not part of the tree's state")
    assert study_guard(tmp_path, tmp_path / "g.json", tmp_path / "p.json") is None


def test_study_guard_refuses_a_dirty_tree(tmp_path) -> None:
    from specdx.cli import study_guard

    _git_repo(tmp_path)
    (tmp_path / "p.json").write_text('{"edited": true}')
    reason = study_guard(tmp_path, tmp_path / "g.json", tmp_path / "p.json")
    assert reason and "uncommitted" in reason


def test_study_guard_refuses_a_missing_gate_and_an_outside_preregistration(tmp_path) -> None:
    from specdx.cli import study_guard

    reason = study_guard(tmp_path, tmp_path / "missing.json", tmp_path / "p.json")
    assert reason and "gate" in reason
    (tmp_path / "repo").mkdir()
    _git_repo(tmp_path / "repo")
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")
    reason = study_guard(tmp_path / "repo", tmp_path / "repo" / "g.json", outside)
    assert reason and "commit" in reason


def _no_workbench(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the workbench (and its database lock) must not be opened")

    monkeypatch.setattr("specdx.workbench.Workbench.open", refuse)


def _frozen_repo(root: Path, *, track_config: bool = True, gate: str = '{"passed": true}'):
    """A git repository holding the frozen study files, all committed (the configuration
    only when track_config). Returns (config, prereg, doc)."""
    import json
    import subprocess

    from specdx.eval.freeze import canonical_json, prereg

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                        *args], check=True, capture_output=True)

    git("init", "-q")
    (root / "experiments").mkdir()
    (root / "results").mkdir()
    config = root / "experiments" / "p2-gate.json"
    config.write_text(json.dumps({"models_by_family": {"a": ["m"]}, "variants": [0],
                                  "seeds": [1], "budget": 1, "dev_per_type": 1,
                                  "rng_seed": 1}))
    src = root / "dev.json"
    src.write_text(_dev_result().to_json(), encoding="utf-8")
    (root / "results" / "p2-gate.json").write_text(gate)
    doc = prereg(_dev_result(), src, gate_config=config,
                 gate_result=root / "results" / "p2-gate.json")
    pre = root / "experiments" / "p3-baselines-prereg.json"
    pre.write_text(canonical_json(doc), encoding="utf-8")
    files = ["dev.json", "experiments/p3-baselines-prereg.json", "results/p2-gate.json"]
    git("add", *files, *(["experiments/p2-gate.json"] if track_config else []))
    git("commit", "-q", "-m", "frozen")
    return config, pre, doc


def _study(tmp_path, config: Path, pre: Path) -> int:
    return main(["study", "--split", "test", "--config", str(config), "--baselines", str(pre),
                 "--out", str(tmp_path / "out" / "t.json")])


@pytest.mark.parametrize("gate", ['{"passed": false}', "{not json", "[true]", '{"passed": "yes"}'])
def test_study_exits_4_before_opening_the_workbench_without_a_passed_gate(
        monkeypatch, tmp_path, capsys, gate) -> None:
    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, _ = _frozen_repo(tmp_path, gate=gate)
    assert _study(tmp_path, config, pre) == 4
    assert "gate has not PASSED" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_study_exits_4_without_a_gate_file(monkeypatch, tmp_path, capsys) -> None:
    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, _ = _frozen_repo(tmp_path)
    (tmp_path / "results" / "p2-gate.json").unlink()
    assert _study(tmp_path, config, pre) == 4
    assert "gate has not PASSED" in capsys.readouterr().err


@pytest.mark.parametrize("which", ["config", "baselines"])
def test_study_exits_4_on_any_file_but_the_frozen_one(
        monkeypatch, tmp_path, capsys, which) -> None:
    import shutil

    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, _ = _frozen_repo(tmp_path)
    frozen = config if which == "config" else pre
    other = tmp_path / "experiments" / f"other-{frozen.name}"
    shutil.copy(frozen, other)  # the same contents under another name are still refused
    args = (other, pre) if which == "config" else (config, other)
    assert _study(tmp_path, *args) == 4
    err = capsys.readouterr().err
    assert f"frozen {frozen}" in err and str(other) in err


def test_study_exits_4_on_an_untracked_frozen_config(monkeypatch, tmp_path, capsys) -> None:
    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, _ = _frozen_repo(tmp_path, track_config=False)
    assert _study(tmp_path, config, pre) == 4
    assert "experiments/p2-gate.json is not committed" in capsys.readouterr().err


def test_study_exits_4_on_an_edited_frozen_config(monkeypatch, tmp_path, capsys) -> None:
    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, _ = _frozen_repo(tmp_path)
    config.write_text(config.read_text().replace('"budget": 1', '"budget": 2'))
    assert _study(tmp_path, config, pre) == 4
    assert "uncommitted changes" in capsys.readouterr().err


def _dev_result():
    from specdx.baselines.api import REGISTRY
    from specdx.eval.study import SplitResult
    from specdx.mutate.operators import GroundTruth
    from specdx.types import Defect

    truths = {"H1": GroundTruth("H1", "hidden", frozenset({"c1"}), "clause")}
    hit = [Defect("hidden", ("c1",), (), (), 0.7)]
    defects = {"specdx": {"base": [], "H1": hit}}
    for name in sorted(REGISTRY):
        defects[name] = {"base": [], "H1": hit}
    return SplitResult("dev", truths, {"H1"}, defects, {}, {}, {})


def _gate_files(monkeypatch, root: Path) -> tuple[Path, Path]:
    """The gate configuration and result freeze-baselines fingerprints, under root."""
    import specdx.cli

    monkeypatch.setattr(specdx.cli, "REPO_ROOT", root)
    (root / "experiments").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    config, gate = root / "experiments" / "p2-gate.json", root / "results" / "p2-gate.json"
    config.write_text('{"budget": 1}')
    gate.write_text('{"passed": true}')
    return config, gate


def test_freeze_baselines_writes_canonical_json_whose_hash_the_markdown_prints(
        monkeypatch, tmp_path) -> None:
    import hashlib

    _no_workbench(monkeypatch)
    config, gate = _gate_files(monkeypatch, tmp_path)
    src = tmp_path / "dev.json"
    src.write_text(_dev_result().to_json(), encoding="utf-8")
    cache = tmp_path / "baseline-llm.jsonl"
    cache.write_bytes(b'{"a": 1}\n')
    out = tmp_path / "exp" / "p3-baselines-prereg"
    assert main(["freeze-baselines", "--from", str(src), "--out", str(out),
                 "--cache", str(cache)]) == 0
    body = out.with_suffix(".json").read_bytes()
    assert hashlib.sha256(body).hexdigest() in out.with_suffix(".md").read_text()
    import json

    doc = json.loads(body)
    assert doc["baseline_cache"]["lines"] == 1 and set(doc["chosen"]) >= {"jest"}
    assert doc["gate_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert doc["gate_result_sha256"] == hashlib.sha256(gate.read_bytes()).hexdigest()


def test_freeze_baselines_refuses_without_the_gate_result(monkeypatch, tmp_path, capsys) -> None:
    _no_workbench(monkeypatch)
    _, gate = _gate_files(monkeypatch, tmp_path)
    gate.unlink()
    src = tmp_path / "dev.json"
    src.write_text(_dev_result().to_json(), encoding="utf-8")
    out = tmp_path / "prereg"
    assert main(["freeze-baselines", "--from", str(src), "--out", str(out)]) == 1
    assert "p2-gate.json is missing" in capsys.readouterr().err
    assert not out.with_suffix(".json").exists()


def test_freeze_baselines_refuses_failed_specifications_and_incomplete_choices(
        monkeypatch, tmp_path, capsys) -> None:
    _no_workbench(monkeypatch)
    _gate_files(monkeypatch, tmp_path)
    failed = _dev_result()
    failed.failures = {"jest": {"H1": "boom"}}
    src = tmp_path / "dev.json"
    src.write_text(failed.to_json(), encoding="utf-8")
    out = tmp_path / "prereg"
    assert main(["freeze-baselines", "--from", str(src), "--out", str(out)]) == 1
    assert "failed specifications" in capsys.readouterr().err
    incomplete = _dev_result()
    del incomplete.defects["jest"]
    src.write_text(incomplete.to_json(), encoding="utf-8")
    assert main(["freeze-baselines", "--from", str(src), "--out", str(out)]) == 1
    assert "jest" in capsys.readouterr().err
    assert not out.with_suffix(".json").exists() and not out.with_suffix(".md").exists()


def test_report_uses_tau_zero_on_dev_and_the_frozen_tau_on_test(
        monkeypatch, tmp_path, capsys) -> None:
    import json

    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    dev = tmp_path / "dev.json"
    dev.write_text(_dev_result().to_json(), encoding="utf-8")
    out = tmp_path / "report.md"
    assert main(["report", "--dev", str(dev), "--out", str(out)]) == 0
    text = out.read_text()
    assert "dev split" in text and "test split" not in text
    assert "| jest | 0 |" in text
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "p3-baselines-prereg.json").write_text(json.dumps(
        {"chosen": {"jest": {"arm": "jest", "model": None, "strategy": None, "tau": 0.5,
                             "dev_f1": 1.0}}}))
    test = _dev_result()
    test.split = "test"
    test_json = tmp_path / "test.json"
    test_json.write_text(test.to_json(), encoding="utf-8")
    assert main(["report", "--dev", str(dev), "--test", str(test_json), "--out", str(out)]) == 0
    tail = out.read_text().split("test split", 1)[1]
    assert "| jest | 0.5 |" in tail and "| specdx | 0 |" in tail
    captured = capsys.readouterr()
    assert captured.out.startswith("# Comparison study: dev split")
    assert "embeds no τ" in captured.err and "embeds no τ" in out.read_text()
    # The τ the study embedded wins over the pre-registration's.
    test.taus = {"jest": 0.25}
    test_json.write_text(test.to_json(), encoding="utf-8")
    assert main(["report", "--dev", str(dev), "--test", str(test_json), "--out", str(out)]) == 0
    tail = out.read_text().split("test split", 1)[1]
    assert "| jest | 0.25 |" in tail and "embeds no τ" not in out.read_text()


def _wired(monkeypatch, tmp_path):
    """A fake workbench and run_split recording what the baselines and study commands pass."""
    from types import SimpleNamespace

    from specdx.eval.study import SplitResult

    def behaviour(model, digest):
        return SimpleNamespace(interpreter=SimpleNamespace(model=model), model_digest=digest)

    cache = [behaviour("a:14b", "old"), behaviour("a:14b", "new"), behaviour("b:8b", "")]
    bench = SimpleNamespace(base="base", pdf=None, work_dir=tmp_path, cache=cache)
    mutants = [SimpleNamespace(id=i) for i in ("D1", "D2", "T1", "T2")]
    seen = {}

    def fake_split(bench_, config, ms, arms, gateway, out_lines, split):
        seen.update(mutants=[m.id for m in ms], arms=[a.label for a in arms], split=split,
                    pinned=gateway.pinned, cache=gateway.cache_path, lines=out_lines)
        return SplitResult(split, {}, set(), {"specdx": {}}, {}, {}, {})

    monkeypatch.setattr("specdx.workbench.Workbench.open", lambda workers: bench)
    monkeypatch.setattr("specdx.mutate.operators.all_mutants", lambda *a: mutants)
    monkeypatch.setattr("specdx.mutate.operators.dev_split", lambda ms, n, seed: ms[:2])
    monkeypatch.setattr("specdx.eval.study.run_split", fake_split)
    return seen


def test_baselines_runs_the_dev_arms_on_the_dev_split_with_the_cache_digests(
        monkeypatch, tmp_path) -> None:
    seen = _wired(monkeypatch, tmp_path)
    out = tmp_path / "results" / "dev.json"
    config = _run_config(tmp_path, {"a": ["a:14b"], "b": ["b:8b"]})
    assert main(["baselines", "--split", "dev", "--config", str(config), "--out", str(out),
                 "--models", "a:14b"]) == 0
    assert seen["mutants"] == ["D1", "D2"] and seen["split"] == "dev"
    assert seen["pinned"] == {"a:14b": "new"}  # the last digest per model; none for empty
    assert seen["cache"] == tmp_path / "baseline-llm.jsonl"
    assert seen["lines"] == out.with_suffix(".progress.jsonl")
    assert seen["arms"] and not any("b:8b" in a for a in seen["arms"])
    assert any("@a:14b" in a for a in seen["arms"]) and "jest" in seen["arms"]
    assert out.exists()
    assert main(["baselines", "--split", "base", "--config", str(config),
                 "--out", str(out)]) == 0
    assert seen["mutants"] == [] and seen["split"] == "base"


def test_study_runs_the_frozen_arms_on_the_mutants_outside_the_dev_split(
        monkeypatch, tmp_path) -> None:
    import specdx.cli
    from specdx.eval.freeze import study_arms

    seen = _wired(monkeypatch, tmp_path)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, doc = _frozen_repo(tmp_path)
    assert _study(tmp_path, config, pre) == 0
    assert seen["mutants"] == ["T1", "T2"] and seen["split"] == "test"
    assert seen["arms"] == [a.label for a, _ in study_arms(doc)]
    import hashlib
    import json

    out = json.loads((tmp_path / "out" / "t.json").read_text())
    assert out["prereg_sha256"] == hashlib.sha256(pre.read_bytes()).hexdigest()
    assert out["taus"] == {a.label: tau for a, tau in study_arms(doc)}


@pytest.mark.parametrize("which", ["experiments/p2-gate.json", "results/p2-gate.json"])
def test_study_exits_4_when_a_bound_gate_file_changed_since_the_freeze(
        monkeypatch, tmp_path, capsys, which) -> None:
    import subprocess

    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, _ = _frozen_repo(tmp_path)
    path = tmp_path / which
    path.write_text(path.read_text().replace("}", ', "edited": 1}', 1))
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-am", "edit"], check=True)  # committed: the tree is clean
    assert _study(tmp_path, config, pre) == 4
    assert f"{path} changed since the freeze" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_study_exits_4_on_a_preregistration_without_the_gate_hashes(
        monkeypatch, tmp_path, capsys) -> None:
    import json
    import subprocess

    import specdx.cli

    _no_workbench(monkeypatch)
    monkeypatch.setattr(specdx.cli, "REPO_ROOT", tmp_path)
    config, pre, doc = _frozen_repo(tmp_path)
    del doc["gate_result_sha256"]
    pre.write_text(json.dumps(doc))
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-am", "old"], check=True)
    assert _study(tmp_path, config, pre) == 4
    assert "records no gate_result_sha256" in capsys.readouterr().err


def test_freeze_baselines_refuses_to_replace_a_frozen_preregistration_without_force(
        monkeypatch, tmp_path, capsys) -> None:
    _no_workbench(monkeypatch)
    _gate_files(monkeypatch, tmp_path)
    src = tmp_path / "dev.json"
    src.write_text(_dev_result().to_json(), encoding="utf-8")
    out = tmp_path / "prereg"
    out.with_suffix(".json").write_text("frozen")
    assert main(["freeze-baselines", "--from", str(src), "--out", str(out)]) == 1
    assert "--force" in capsys.readouterr().err
    assert out.with_suffix(".json").read_text() == "frozen"
    assert not out.with_suffix(".md").exists()
    assert main(["freeze-baselines", "--from", str(src), "--out", str(out), "--force"]) == 0
    assert out.with_suffix(".json").read_text() != "frozen" and out.with_suffix(".md").exists()
