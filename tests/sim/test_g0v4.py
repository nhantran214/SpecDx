"""G0v4's grid, scoring of undetermined cells, selection rule, criteria and freeze."""

import dataclasses
import json

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


def test_selection_prefers_larger_tau_u_on_ties() -> None:
    def cand(tau_u, objective, passes=True):
        return g0v4.Candidate(V4Params(tau_u=tau_u), objective, {}, passes=passes)

    low, high, mid = cand(1.0, 0.5), cand(3.0, 0.5), cand(2.0, 0.5)
    assert g0v4.choose([low, high, mid]) is high
    same = cand(3.0, 0.5)
    assert g0v4.choose([high, same]) is high                # then grid order
    assert g0v4.choose([cand(3.0, 0.4), cand(1.0, 0.6)]).params.tau_u == 1.0
    # Candidates that fail a criterion lose to any that passes.
    assert g0v4.choose([cand(3.0, 0.9, passes=False), cand(1.0, 0.1)]).params.tau_u == 1.0
    # With no pass, the best by the same rule is reported.
    assert g0v4.choose([cand(1.0, 0.2, False), cand(2.0, 0.3, False)]).params.tau_u == 2.0


def _metrics(**over):
    m = {"shared_fp": 0.05, "shared_determined": 20, "hidden_f1": 0.7, "ambiguity_f1": 0.5,
         "undetermined": 0.2, "planted_undetermined": {"homogeneous": 0.9, "saturated": 0.85}}
    return m | over


V3 = {"hidden_f1": 0.7, "ambiguity_f1": 0.5, "shared_hidden": 0.6}


def test_criteria_verdict() -> None:
    ok = g0v4.judge(_metrics(), V3)
    assert ok == {"1": True, "2": {"hidden": True, "ambiguity": True},
                  "3": {"homogeneous": True, "saturated": True}}
    assert g0v4.verdict(ok) == g0v4.PASS
    # Boundaries: 0.1, v3 - 0.05 and 0.8 are inclusive.
    edge = g0v4.judge(_metrics(shared_fp=0.1, hidden_f1=0.65 + 1e-12, ambiguity_f1=0.45 + 1e-12,
                               planted_undetermined={"homogeneous": 0.8, "saturated": 0.8}), V3)
    assert g0v4.verdict(edge) == g0v4.PASS
    for bad in (_metrics(shared_fp=0.11), _metrics(hidden_f1=0.6), _metrics(ambiguity_f1=0.4),
                _metrics(planted_undetermined={"homogeneous": 0.79, "saturated": 0.9}),
                _metrics(planted_undetermined={"homogeneous": 0.9, "saturated": 0.5})):
        assert g0v4.verdict(g0v4.judge(bad, V3)) == g0v4.FAIL
    assert g0v4.failed(g0v4.judge(_metrics(shared_fp=0.5, hidden_f1=0.0), V3)) == ["1", "2.hidden"]


def test_freeze_writes_the_params(tmp_path) -> None:
    path = tmp_path / "frozen" / "p4-s3-v4-params.json"
    params = V4Params(cell_sd=2.0, tau_u=3.0)
    g0v4.freeze_v4(params, path)
    assert json.loads(path.read_text())["params"] == dataclasses.asdict(params)


def test_run_g0v4_writes_outputs_on_a_tiny_config(tmp_path) -> None:
    out = tmp_path / "g0v4"
    code = g0v4.run_g0v4(TINY, out, replicates=2, grid=SMALL_GRID)
    assert code in (0, 1)
    result = json.loads(out.with_suffix(".json").read_text())
    assert result["verdict"] == ("PASS" if code == 0 else "FAIL")
    assert result["replicates"] == 2 and len(result["candidates"]) == 2
    assert set(result["criteria"]) == {"1", "2", "3"}
    assert "held_out" in result and "v3" in result
    text = out.with_suffix(".md").read_text()
    assert "G0v4" in text and "undetermined" in text.lower()
    assert not (tmp_path / "experiments").exists()      # run never freezes


def test_run_g0v4_returns_2_on_an_error(tmp_path) -> None:
    assert g0v4.run_g0v4(TINY, tmp_path / "y", replicates=0, grid=SMALL_GRID) == 2
