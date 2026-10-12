"""The offline replay's pieces that need no run cache: the pinned P1 context digest and
the pre-registered GO criterion."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from specdx.infer.act import Act, Rasch
from specdx.infer.latent import Posterior
from specdx.infer.scores import Diagnosis
from specdx.observe.run import RUN_SCHEMA_VERSION, Runner, context_digest
from specdx.types import Behaviour, InterpreterId, Scenario, State

SCRIPT = Path(__file__).parents[1] / "experiments" / "p1b_replay.py"


@pytest.fixture(scope="module")
def replay():
    spec = importlib.util.spec_from_file_location("p1b_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module while it loads
    spec.loader.exec_module(module)
    return module


def test_an_older_schema_version_gives_the_context_digest_runs_were_cached_under() -> None:
    tell, settings = {"standing": ["a"]}, {"temperature": 0.7}
    assert context_digest(tell, settings) == context_digest(tell, settings,
                                                            schema=RUN_SCHEMA_VERSION)
    assert context_digest(tell, settings, schema=2) != context_digest(tell, settings, schema=99)


def _result(replay, theta, kappa, acted=()):
    rasch = Rasch(kappa, {}, {}, {})
    acts = [Act("grn12|step", f"{m}|1", m, True) for m in acted]
    return replay.Replay("base", None, [], [], acts, rasch, Posterior(theta, {}, {}),
                         Diagnosis([]))


def test_go_is_the_mechanism_check_on_kappa_theta_floor_and_unobservable_share(replay) -> None:
    q3, q25, ll = "qwen3:14b", "qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b"
    kappa, writers = {q3: 0.5, q25: 0.0, ll: -1.0}, (q3, q25)
    # the θ ordering is not judged: llama above qwen3, as on the P1 population, still passes
    assert replay.go(_result(replay, {q3: 0.74, q25: 0.6, ll: 0.96}, kappa, writers))[0]
    # (a) the model that never writes must have the lowest κ, strictly
    assert not replay.go(_result(replay, {q3: 0.74, q25: 0.6, ll: 0.96},
                                 {q3: 0.5, q25: -2.0, ll: -1.0}, writers))[0]
    assert not replay.go(_result(replay, {q3: 0.74, q25: 0.6, ll: 0.96},
                                 {q3: 0.5, q25: -1.0, ll: -1.0}, writers))[0]
    # (b) an acting model collapsing below 0.5 fails; a model that never acted is not judged
    assert not replay.go(_result(replay, {q3: 0.74, q25: 0.4, ll: 0.96}, kappa, writers))[0]
    assert replay.go(_result(replay, {q3: 0.74, q25: 0.6, ll: 0.3}, kappa, writers))[0]
    assert not replay.go(_result(replay, {q3: 0.74, q25: 0.6, ll: 0.96}, kappa, ()))[0]
    # (a) cannot pass vacuously: a κ without the other models of the population fails
    assert not replay.go(_result(replay, {q3: 0.74, q25: 0.6, ll: 0.96}, {ll: -1.0}, writers))[0]
    assert not replay.go(_result(replay, {ll: 0.96}, {ll: -1.0}, (ll,)))[0]
    # (c) the share of P1 base findings that become Unobservable
    assert replay.THETA_FLOOR == 0.5 and replay.UNOBSERVABLE_SHARE == 0.5
    assert replay.unobservable_share(["unobservable"] * 61 + ["observable"] * 30) > 0.5
    assert replay.unobservable_share(["unobservable", "observable"]) == 0.5
    assert replay.unobservable_share([]) == 0.0


def test_the_replay_never_refutes_executes_or_writes_a_run(replay, tmp_path, capsys) -> None:
    scenario, interp = Scenario("s", 15, 12, 0), InterpreterId("m", "f", 0, 0)
    result = replay.NoRefutation().refute(frozenset({"c"}), scenario, [])
    assert not result.refuted and not result.attempted
    # the run cache is read-only, and its error is not an Exception the runner could catch
    assert not issubclass(replay.ReplayWriteError, Exception)
    path = tmp_path / "runs.jsonl"
    cache = replay.ReadOnlyRunCache(path)
    with pytest.raises(replay.ReplayWriteError):
        cache.put(Behaviour("k", interp, scenario, State.empty()))
    assert not path.exists() and len(cache) == 0
    # through Runner.run: neither an executed run nor the error run it would cache gets out
    db = SimpleNamespace(use=lambda pack: None)
    variant = SimpleNamespace(digest=lambda: "d", pack=None, corpus=None, name="v")
    runner = Runner(db, cache, replay._never, (), model_digest=lambda model: "digest")
    with pytest.raises(replay.ReplayWriteError):
        runner.run(variant, scenario, interp)

    def failing(*_):
        raise ValueError("a factory failure Runner.run would cache as an error run")

    runner.factory = failing
    with pytest.raises(replay.ReplayWriteError):
        runner.run(variant, scenario, interp)
    assert not path.exists()
    # a stop that is no verdict exits 2, never NO-GO's 1
    with pytest.raises(SystemExit) as stop:
        replay._stop("the mapping is broken")
    assert stop.value.code == 2 and "the mapping is broken" in capsys.readouterr().err
