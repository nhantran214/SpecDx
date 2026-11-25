"""G0v4's grid, scoring of undetermined cells, selection rule, criteria and freeze."""

import dataclasses
import json
from pathlib import Path

import pytest

from specdx.infer.anchored_v4 import V4Params
from specdx.infer.typing import CellFinding
from specdx.sim import g0v4
from specdx.sim.generator import SimConfig, generate
from specdx.sim.metrics import TYPES

TINY = SimConfig(alphas=(2.5, 1.0, 0.5), act=(1.0, 0.9, 0.8), columns=10, items=3,
                 scenarios=2, seeds=2, seed=5)
SMALL_GRID = {"cell_sd": (1.0,), "lambda_prior": ((1.0, 3.0),), "tau_u": (1.0, 3.0),
              "tau_h": (0.5,), "tau_a": (0.3,)}


def test_the_grid_is_the_pre_registered_one() -> None:
    assert g0v4.CELL_SD == (0.5, 1.0, 2.0) and g0v4.LAMBDA_PRIOR == ((1.0, 3.0), (1.0, 1.0))
    assert g0v4.TAU_U == (1.0, 2.0, 3.0) and g0v4.TAU_H == (0.5, 0.7, 0.9)
    assert g0v4.TAU_A == (0.3, 0.5, 0.7)
    assert (g0v4.TAU_W, g0v4.TAU_LAMBDA) == (0.9, 0.5)
    assert len(g0v4.prior_settings()) == 6 and len(g0v4.thresholds()) == 27
    assert g0v4.ARMS == ("base", "shared", "homogeneous", "saturated")
    assert all((p.tau_w, p.tau_lambda) == (0.9, 0.5) for p in g0v4.prior_settings())


def test_g0v4_scores_undetermined_separately() -> None:
    spec = generate(dataclasses.replace(TINY, columns=30))
    truth_hidden = sorted(c for c, t in spec.truth.items() if t.hidden)
    assert len(truth_hidden) >= 2
    a, b = truth_hidden[:2]
    found = [CellFinding(a, "hidden", 1.0),                    # a true prediction
             CellFinding(b, g0v4.UNDETERMINED, 0.1)]           # b is undetermined
    scored = g0v4.score_findings(spec, found)
    assert scored.undetermined == {b}
    c = scored.counts["hidden"]
    assert c.tp == 1 and c.fp == 0
    assert c.fn == len(truth_hidden) - 2          # b is not a miss
    # An undetermined cell is not a false positive either, even if a type is also reported.
    clean = next(x for x, t in spec.truth.items() if not (t.hidden or t.ambiguity or t.weak))
    scored = g0v4.score_findings(spec, [CellFinding(clean, g0v4.UNDETERMINED, 0.0),
                                        CellFinding(clean, "hidden", 1.0)])
    assert scored.counts["hidden"].fp == 0
    assert scored.counts["hidden"].fn == len(truth_hidden)
    assert set(scored.counts) == set(TYPES)


def test_selection_prefers_smaller_tau_u_on_ties() -> None:
    # Ruling F-I2a: undetermined must not be free, so ties go to the smaller tau_u.
    def cand(tau_u, objective, passes=True):
        return g0v4.Candidate(V4Params(tau_u=tau_u), objective, {}, passes=passes)

    low, high, mid = cand(1.0, 0.5), cand(3.0, 0.5), cand(2.0, 0.5)
    assert g0v4.choose([high, low, mid]) is low
    same = cand(1.0, 0.5)
    assert g0v4.choose([low, same]) is low                  # then grid order
    # Ties are within 1e-9, wherever a rounding boundary falls.
    assert g0v4.choose([cand(3.0, 0.5000000004), cand(1.0, 0.4999999996)]).params.tau_u == 1.0
    assert g0v4.choose([cand(3.0, 0.4), cand(1.0, 0.6)]).params.tau_u == 1.0
    # Candidates that fail a criterion lose to any that passes.
    assert g0v4.choose([cand(3.0, 0.9, passes=False), cand(1.0, 0.1)]).params.tau_u == 1.0
    # With no pass, the best by the same rule is reported.
    assert g0v4.choose([cand(1.0, 0.2, False), cand(2.0, 0.3, False)]).params.tau_u == 2.0


def _metrics(**over):
    m = {"shared_fp": 0.0, "shared_hidden": 0, "shared_determined": 100, "planted":
         {"homogeneous": 50, "saturated": 50}, "hidden_f1": 0.7, "ambiguity_f1": 0.5,
         "undetermined": 0.2, "planted_undetermined": {"homogeneous": 0.9, "saturated": 0.85}}
    return m | over


V3 = {"hidden_f1": 0.7, "ambiguity_f1": 0.5, "shared_hidden": 0.6}


def test_criteria_verdict() -> None:
    ok = g0v4.judge(_metrics(), V3)
    assert ok == {"1": True, "2": {"hidden": True, "ambiguity": True},
                  "3": {"homogeneous": True, "saturated": True}}
    assert g0v4.verdict(ok) == g0v4.PASS
    # Boundaries: 0.1, v3 - 0.05 and 0.8 are inclusive.
    edge = g0v4.judge(_metrics(shared_hidden=0, hidden_f1=0.65 + 1e-12, ambiguity_f1=0.45 + 1e-12,
                               planted_undetermined={"homogeneous": 0.8, "saturated": 0.8}), V3)
    assert g0v4.verdict(edge) == g0v4.PASS
    for bad in (_metrics(shared_hidden=5), _metrics(hidden_f1=0.6), _metrics(ambiguity_f1=0.4),
                _metrics(planted_undetermined={"homogeneous": 0.79, "saturated": 0.9}),
                _metrics(planted_undetermined={"homogeneous": 0.9, "saturated": 0.5})):
        assert g0v4.verdict(g0v4.judge(bad, V3)) == g0v4.FAIL
    worse = g0v4.judge(_metrics(shared_hidden=50, hidden_f1=0.0), V3)
    assert g0v4.failed(worse) == ["1", "2.hidden"]


def _official(tmp_path, **over) -> Path:
    """A search JSON that may be frozen (ruling F-I5), with fields overridden."""
    layout = dataclasses.asdict(TINY) | {"seed": 0}
    result = {"verdict": "PASS", "smoke": False, "replicates": g0v4.REPLICATES,
              "grid": g0v4.full_grid(), "layouts": [layout],
              "seeds": {"selection": list(range(10)), "held_out": list(range(10, 20))},
              "chosen": {"params": dataclasses.asdict(V4Params(cell_sd=2.0, tau_u=3.0))}}
    result |= over
    path = tmp_path / "search.json"
    path.write_text(json.dumps(result))
    return path


def _sha() -> str:
    return "abc123"


def test_freeze_writes_the_params_with_provenance(tmp_path) -> None:
    import hashlib

    path = tmp_path / "frozen" / "p4-s3-v4-params.json"
    search = _official(tmp_path)
    ok, message = g0v4.freeze_result(search, path, head=_sha)
    assert ok, message
    frozen = json.loads(path.read_text())
    assert frozen["params"] == dataclasses.asdict(V4Params(cell_sd=2.0, tau_u=3.0))
    assert frozen["provenance"]["commit"] == "abc123"
    assert frozen["provenance"]["search_sha256"] == hashlib.sha256(
        search.read_bytes()).hexdigest()


@pytest.mark.parametrize("over, why", [
    ({"verdict": "FAIL"}, "not PASS"),
    ({"smoke": True}, "smoke"),
    ({"smoke": None}, "smoke"),
    ({"replicates": 2}, "replicates"),
    ({"grid": g0v4._grid_json({"cell_sd": (1.0,)})}, "grid"),
    ({"seeds": {"selection": list(range(1, 11)), "held_out": list(range(11, 21))}}, "seeds"),
])
def test_freeze_refuses_what_was_not_pre_registered(tmp_path, over, why) -> None:
    path = tmp_path / "frozen.json"
    ok, message = g0v4.freeze_result(_official(tmp_path, **over), path, head=_sha)
    assert not ok and why in message and not path.exists()


def test_freeze_never_overwrites_other_parameters(tmp_path) -> None:
    path = tmp_path / "frozen.json"
    assert g0v4.freeze_result(_official(tmp_path), path, head=_sha)[0]
    before = path.read_text()
    again = g0v4.freeze_result(_official(tmp_path), path, head=lambda: "other")
    assert again[0] and path.read_text() == before          # same parameters: left alone
    other = {"params": dataclasses.asdict(V4Params(tau_u=1.0))}
    ok, message = g0v4.freeze_result(_official(tmp_path, chosen=other), path, head=_sha)
    assert not ok and "overwrite" in message and path.read_text() == before


def test_run_g0v4_writes_outputs_on_a_tiny_config(tmp_path) -> None:
    out = tmp_path / "g0v4"
    code = g0v4.run_g0v4(TINY, out, replicates=2, grid=SMALL_GRID)
    assert code in (0, 1)
    result = json.loads(out.with_suffix(".json").read_text())
    assert result["verdict"] == ("PASS" if code == 0 else "FAIL")
    assert result["replicates"] == 2 and len(result["candidates"]) == 2
    # Ruling F-I5: the run records what freezing checks.
    assert result["smoke"] is False and result["grid"] == g0v4._grid_json(SMALL_GRID)
    assert result["seeds"] == {"selection": [5, 6], "held_out": [7, 8]}
    assert "Every candidate" in out.with_suffix(".md").read_text()
    assert set(result["criteria"]) == {"1", "2", "3"}
    assert "held_out" in result and "v3" in result
    text = out.with_suffix(".md").read_text()
    assert "G0v4" in text and "undetermined" in text.lower()
    assert not (tmp_path / "experiments").exists()      # run never freezes


def test_run_g0v4_returns_2_on_an_error(tmp_path) -> None:
    assert g0v4.run_g0v4(TINY, tmp_path / "y", replicates=0, grid=SMALL_GRID) == 2


def test_wilson_bound_and_criterion_1_boundary() -> None:
    assert abs(g0v4.wilson_upper(0, 100) - 0.0370) < 5e-4
    assert abs(g0v4.wilson_upper(5, 100) - 0.1117) < 5e-4
    assert g0v4.judge(_metrics(shared_hidden=0, shared_determined=100), V3)["1"]
    assert g0v4.judge(_metrics(shared_hidden=4, shared_determined=100), V3)["1"]
    assert not g0v4.judge(_metrics(shared_hidden=5, shared_determined=100), V3)["1"]
    # A small n cannot pass with a point estimate of 0.05: the bound is what counts.
    assert not g0v4.judge(_metrics(shared_hidden=1, shared_determined=20), V3)["1"]


def test_not_evaluable_criteria_fail() -> None:
    none = g0v4.judge(_metrics(shared_determined=0, shared_hidden=0), V3)
    assert none["1"] is False and "1" in g0v4.failed(none)
    empty = g0v4.judge(_metrics(planted={"homogeneous": 0, "saturated": 50},
                                planted_undetermined={"homogeneous": 1.0, "saturated": 1.0}), V3)
    assert empty["3"] == {"homogeneous": False, "saturated": True}
    assert g0v4.verdict(empty) == g0v4.FAIL
    assert g0v4.failed(empty) == ["3.homogeneous"]


def test_fit_ignores_the_typing_thresholds() -> None:
    from specdx.infer.anchored_v4 import fit_v4
    spec = generate(dataclasses.replace(TINY, arm="shared"))
    prior = g0v4.prior_settings()[0]
    ref = fit_v4(spec.cells, spec.readings, models=spec.models, params=prior)
    for name, value in (("tau_u", 3.0), ("tau_h", 0.9), ("tau_a", 0.7), ("tau_w", 0.5),
                        ("tau_lambda", 0.9)):
        other = fit_v4(spec.cells, spec.readings, models=spec.models,
                       params=dataclasses.replace(prior, **{name: value}))
        assert dataclasses.replace(other, params=ref.params) == ref, name


def _stable(out):
    d = json.loads(out.with_suffix(".json").read_text())
    for key in ("seconds", "estimate", "jobs"):
        d.pop(key)
    return d


def test_jobs_do_not_change_the_output_and_nothing_forks_a_threaded_parent(
        tmp_path, monkeypatch) -> None:
    # Ruling F-I7: the pool is a forkserver one. This process never calls os.fork (whose
    # DeprecationWarning about a multi-threaded parent is an error here), and the workers get
    # one BLAS thread each.
    import os
    import warnings

    assert g0v4.run_g0v4(TINY, tmp_path / "a", replicates=2, grid=SMALL_GRID, jobs=1) in (0, 1)

    def no_fork():
        raise AssertionError("os.fork called in the parent")

    monkeypatch.setattr(os, "fork", no_fork)
    for name in g0v4.BLAS_THREADS:
        monkeypatch.setenv(name, "4")  # restored after the test
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        code = g0v4.run_g0v4(TINY, tmp_path / "b", replicates=2, grid=SMALL_GRID, jobs=2)
    assert all(os.environ.get(name) == "1" for name in g0v4.BLAS_THREADS)
    assert code in (0, 1)
    assert _stable(tmp_path / "a") == _stable(tmp_path / "b")
    assert "Cost estimate" in (tmp_path / "a.md").read_text()


def test_the_cache_resumes_a_run_with_the_same_output(tmp_path, monkeypatch) -> None:
    # Ruling F-I8: every fit's summary is cached, so a rerun fits nothing and writes the same.
    cache = tmp_path / "cache"
    assert g0v4.run_g0v4(TINY, tmp_path / "a", replicates=2, grid=SMALL_GRID,
                         cache=cache) in (0, 1)
    files = sorted(cache.glob("*.json"))
    # 4 arms x 2 replicates (selection) + 4 x 2 (held out) v4 fits, 2 x 2 x 2 v3 fits.
    assert len(files) == 8 + 8 + 8
    monkeypatch.setattr(g0v4, "run_task", lambda task: pytest.fail("a cached fit ran again"))
    assert g0v4.run_g0v4(TINY, tmp_path / "b", replicates=2, grid=SMALL_GRID,
                         cache=cache) in (0, 1)
    assert _stable(tmp_path / "a") == _stable(tmp_path / "b")
    assert json.loads((tmp_path / "b.json").read_text())["estimate"]["cached"] == 16


def test_the_cache_key_follows_the_configuration_and_the_code() -> None:
    prior = g0v4.prior_settings()[0]
    task = g0v4.FitTask("v4", 0, TINY, prior, ((1.0, 0.5, 0.3),))
    assert g0v4.task_key(task) == g0v4.task_key(dataclasses.replace(task))
    for other in (dataclasses.replace(task, config=dataclasses.replace(TINY, seed=6)),
                  dataclasses.replace(task, prior=dataclasses.replace(prior, cell_sd=2.0)),
                  dataclasses.replace(task, thresholds=((2.0, 0.5, 0.3),)),
                  dataclasses.replace(task, kind="v3", prior=None)):
        assert g0v4.task_key(other) != g0v4.task_key(task)


def test_replicates_rotate_over_the_layouts() -> None:
    # Ruling F-C1: replicate r uses layout r mod 3, and each simulated domain is one fit.
    layouts = [dataclasses.replace(TINY, items=n) for n in (2, 3, 4)]
    tasks = g0v4.v4_tasks(layouts, g0v4.prior_settings()[0], 5, 0, [(1.0, 0.5, 0.3)])
    base = [t for t in tasks if t.config.arm == "base"]
    assert [t.layout for t in base] == [0, 1, 2, 0, 1]
    assert [t.config.items for t in base] == [2, 3, 4, 2, 3]
    assert [t.config.seed for t in base] == [5, 6, 7, 8, 9]
    held = g0v4.v4_tasks(layouts, g0v4.prior_settings()[0], 5, 5, [(1.0, 0.5, 0.3)])
    assert [t.config.seed for t in held if t.config.arm == "base"] == [10, 11, 12, 13, 14]
    assert len(tasks) == len(g0v4.ARMS) * 5


def test_the_estimate_simulates_the_pool_and_its_barriers() -> None:
    # Ruling F-I8: fewer tasks than jobs leave workers idle; each phase ends at a barrier.
    assert g0v4._makespan([10.0, 10.0, 10.0], 4) == 10.0
    assert g0v4._makespan([10.0] * 5, 4) == 20.0
    assert g0v4._makespan([5.0, 1.0, 1.0, 1.0], 2) == 5.0
    t4 = g0v4.FitTask("v4", 0, TINY, g0v4.prior_settings()[0])
    t3 = g0v4.FitTask("v3", 0, TINY)
    est = g0v4.estimate_cost([(t4, {"seconds": 8.0}), (t3, {"seconds": 2.0})], 9.0,
                             [t4] * 3 + [t3] * 2, [t4] * 4, jobs=2, cached=0)
    # Two workers: 8 | 8, then 8 on the first free (16), the v3 fits at 10 and 12.
    assert est["selection_s"] == 16.0 and est["held_out_s"] == 16.0
    assert est["total_s"] == 9.0 + 16.0 + 16.0
    assert (est["v4_fits"], est["v3_fits"]) == (8, 3)


def test_the_objective_counts_undetermined_planted_cells_as_misses() -> None:
    # Ruling F-I2a: abstaining on a planted cell lowers the objective.
    spec = generate(dataclasses.replace(TINY, columns=30))
    hidden = sorted(c for c, t in spec.truth.items() if t.hidden)
    b = hidden[1]
    found = [CellFinding(c, "hidden", 1.0) for c in hidden if c != b]
    found.append(CellFinding(b, g0v4.UNDETERMINED, 0.1))
    scored = g0v4.score_findings(spec, found)
    assert scored.counts["hidden"].fn == 0                     # the criteria: not a miss
    assert scored.coverage["hidden"].fn == 1                   # the objective: a miss
    abstain = g0v4.score_findings(spec, [CellFinding(c, g0v4.UNDETERMINED, 0.1)
                                         for c in hidden])
    assert abstain.coverage["hidden"].f1 == 0.0
    tallies = {arm: g0v4.ArmTally() for arm in g0v4.ARMS}
    for t in tallies.values():
        t.add(g0v4.summarise(spec, scored))
    full = g0v4.objective_of(tallies)
    quiet = {arm: g0v4.ArmTally() for arm in g0v4.ARMS}
    for t in quiet.values():
        t.add(g0v4.summarise(spec, abstain))
    assert g0v4.objective_of(quiet) < full


def test_ambiguity_is_non_binding_when_v3_cannot_find_any() -> None:
    # Ruling F-I2b: with v3's ambiguity F1 below 0.05 the bar is met by construction.
    weak_v3 = V3 | {"ambiguity_f1": 0.0}
    criteria = g0v4.judge(_metrics(ambiguity_f1=0.0), weak_v3)
    assert criteria["2"]["ambiguity"] is None
    assert g0v4.failed(criteria) == [] and g0v4.non_binding(criteria) == ["2.ambiguity"]
    assert g0v4.verdict(criteria) == g0v4.PASS
    lines = "\n".join(g0v4._criteria_lines(_metrics(ambiguity_f1=0.0) | {
        "shared_fp_upper": 0.03}, weak_v3, criteria))
    assert "non-binding (v3 F1 < 0.05)" in lines and "neither passes nor fails" in lines
    binding = g0v4.judge(_metrics(), V3)
    assert binding["2"]["ambiguity"] is True and g0v4.non_binding(binding) == []


def test_cli_freeze_applies_the_same_checks(tmp_path) -> None:
    target = tmp_path / "frozen.json"
    bad = _official(tmp_path, replicates=2)
    assert g0v4.main(["freeze", str(bad), "--path", str(target)]) == 1
    assert not target.exists()
    good = _official(tmp_path)
    assert g0v4.main(["freeze", str(good), "--path", str(target)]) == 0
    assert json.loads(target.read_text())["params"]["tau_u"] == 3.0


def test_held_out_failure_is_warned_about(tmp_path) -> None:
    out = tmp_path / "w"
    g0v4.run_g0v4(TINY, out, replicates=2, grid=SMALL_GRID)
    result = json.loads(out.with_suffix(".json").read_text())
    text = g0v4.render(result)
    assert ("WARNING: the chosen candidate fails on the held-out" in text) == bool(
        g0v4.failed(result["held_out"]["criteria"]))
    result["held_out"]["criteria"]["1"] = False
    assert "WARNING: the chosen candidate fails on the held-out" in g0v4.render(result)
