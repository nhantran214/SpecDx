"""The offline replay's pieces that need no run cache: the pinned P1 context digest and
the pre-registered GO criterion."""

import importlib.util
import sys
from pathlib import Path

import pytest

from specdx.infer.act import Act, Rasch
from specdx.infer.latent import Posterior
from specdx.infer.scores import Diagnosis
from specdx.observe.run import RUN_SCHEMA_VERSION, context_digest
from specdx.types import Scenario

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
    # (c) the share of P1 base findings that become Unobservable
    assert replay.THETA_FLOOR == 0.5 and replay.UNOBSERVABLE_SHARE == 0.5
    assert replay.unobservable_share(["unobservable"] * 61 + ["observable"] * 30) > 0.5
    assert replay.unobservable_share(["unobservable", "observable"]) == 0.5
    assert replay.unobservable_share([]) == 0.0


def test_the_replay_never_attempts_a_refutation(replay) -> None:
    result = replay.NoRefutation().refute(frozenset({"c"}), Scenario("s", 15, 12, 0), [])
    assert not result.refuted and not result.attempted
