"""The simulation's metrics, baselines, sweeps and the `specdx simulate` command (§8.1)."""

import dataclasses
import json
import math

import pytest

from specdx.cli import main
from specdx.infer.anchored import V3Params
from specdx.infer.cells import Cell, Reading
from specdx.sim.baselines import anchor_only, majority_vote, stage2_v2
from specdx.sim.generator import SimConfig, SimSpec, Truth
from specdx.sim.metrics import Counts, calibration, theta_error, typed_counts
from specdx.sim.study import (
    METHODS,
    SWEEPS,
    config_dict,
    gamma_min,
    load_config,
    min_replicates,
    regime_config,
    replicate,
    versus,
)

P = V3Params(pi=0.1, s_mis=0.5, s_amb=0.25, s_empty=0.25, beta_sd=1.0,
             mu_alpha=1.0, sigma_alpha=1.5, tau_h=0.5, tau_a=0.5, tau_w=0.8)
TINY = SimConfig(alphas=(2.0, 0.5), act=(1.0, 1.0), columns=12, items=4, scenarios=2,
                 seeds=1, seed=3)
NONE, HIDDEN = Truth("none", ("d",), False, False, False), Truth("mis", ("a",), True, False, False)
WEAK, AMB = Truth("none", ("x",), False, False, True), Truth("amb", ("x", "y"), False, True, False)


def _spec(rows: dict[str, tuple[str | None, Truth, dict[str, list[str]]]]) -> SimSpec:
    """cell → (anchor, truth, model → values). A reading is clean when it is unanchored or
    reads the anchor, as if each cell were alone at its item."""
    cells, readings, truth = {}, [], {}
    for cid, (anchor, t, by_model) in rows.items():
        cells[cid] = Cell(cid, "value", anchor, "clause" if anchor else None, cid, cid, ("i0",))
        truth[cid] = t
        for m, values in by_model.items():
            readings += [Reading(cid, f"{m}|{i}", m, m, "s0", v, anchor in (None, v))
                         for i, v in enumerate(values)]
    return SimSpec(TINY, cells, readings, truth, {"m0": "m0", "m1": "m1"}, {"m0": 2.0, "m1": 0.5})


def test_each_type_is_scored_over_cells_with_exact_f1() -> None:
    truth = {"a": HIDDEN, "b": NONE, "c": WEAK}
    counts = typed_counts([("a", "hidden"), ("b", "hidden"), ("b", "weak")], truth)
    assert counts["hidden"].as_dict() == {"tp": 1, "fp": 1, "fn": 0, "f1": pytest.approx(2 / 3)}
    assert (counts["weak"].tp, counts["weak"].fp, counts["weak"].fn) == (0, 1, 1)
    assert counts["ambiguity"].f1 == 0.0 and Counts(3, 5, 4).f1 == pytest.approx(0.4)


def test_accuracy_error_and_calibration() -> None:
    assert theta_error({"m": 0.9, "n": 0.4}, {"m": 0.8, "n": 0.5}) == pytest.approx(0.1)
    ece, brier = calibration([(0.0, False)] * 5 + [(1.0, True)] * 5)
    assert ece == 0.0 and brier == 0.0
    ece, brier = calibration([(0.8, False), (0.8, True)])
    assert ece == pytest.approx(0.3) and brier == pytest.approx((0.64 + 0.04) / 2)


def test_majority_vote_and_anchor_only() -> None:
    spec = _spec({
        "h": ("d", HIDDEN, {"m0": ["a", "a", "a"], "m1": ["a", "e", "a"]}),
        "n": ("d", NONE, {"m0": ["d", "d", "d"], "m1": ["d", "e", "d"]}),
        "w": (None, WEAK, {"m0": ["x", "x", "y"], "m1": ["x", "x", "x"]}),
        "z": (None, AMB, {"m0": ["x", "x", "y"], "m1": ["y", "y", "x"]}),
    })
    assert majority_vote(spec) == {("h", "hidden"), ("w", "weak"), ("z", "ambiguity")}
    assert anchor_only(spec) == {("h", "hidden")}


def test_v2_stage_2_on_joint_keys_flags_a_cell_every_model_reads_off_its_anchor() -> None:
    reads = {m: ["d"] * 6 for m in ("m0", "m1")}
    spec = _spec({"h": ("d", HIDDEN, {m: ["a"] * 6 for m in ("m0", "m1")}),
                  "n": ("d", NONE, reads)})
    assert ("h", "hidden") in stage2_v2(spec) and ("n", "hidden") not in stage2_v2(spec)


def test_a_replicate_scores_every_method_and_is_deterministic() -> None:
    result, fit, spec = replicate(TINY, P)
    again, _, _ = replicate(TINY, P)
    assert set(result.counts) == set(METHODS)
    assert result.summary() == again.summary()
    assert result.theta_errors[0] == pytest.approx(theta_error(fit.theta, spec.theta))
    summary = result.summary()
    assert {"theta_error", "ece", "brier", "converged", "n", "delta", "t4a", "accuracy",
            "chance", "t4c"} <= set(summary)
    assert 0.5 < summary["accuracy"] < 1 and 0 <= summary["t4c"] <= 1


def test_t2_prime_s_chance_line_and_t4c_s_distance() -> None:
    # At p = |A|/K a set's prediction is flat, like ∅'s: γ_min is 0 on the chance line.
    assert gamma_min(4, 0.25) == pytest.approx(0.0) and gamma_min(4, 0.5) == pytest.approx(0.0)
    assert gamma_min(4, 0.9) > 0.1
    assert (min_replicates("regime"), min_replicates("arms"), min_replicates("n")) == (5, 10, 10)


def test_each_sweep_varies_its_own_factor() -> None:
    base = SimConfig(alphas=(2.0, 1.0, 0.5, 0.0), act=(0.9, 0.8, 0.6, 0.4))
    assert [len(c.alphas) for _, c in SWEEPS["models"](base)] == [2, 3, 4]
    spreads = [c.alphas for _, c in SWEEPS["delta"](base)]
    assert spreads[0] == (0.875,) * 4 and spreads[2] == base.alphas
    arms = SWEEPS["arms"](base)
    assert [label for label, _ in arms] == ["base", "a3", "a5", "within"]
    assert [c.arm for _, c in arms] == ["base", "a3", "a5", "base"]
    assert arms[3][1].beta_within == 0.5
    assert [c.seeds for _, c in SWEEPS["n"](base)] == [1, 2, 3, 5, 8]
    assert {c.beta_sd for _, c in SWEEPS["tau_true"](base)} == {0.25, 0.5, 1.0, 2.0}
    assert len(SWEEPS) == 8 and len(SWEEPS["regime"](base)) == 30


def test_the_regime_map_shifts_every_alpha_and_judges_against_the_best_baseline() -> None:
    base = SimConfig(alphas=(3.0, 2.0, 1.0, 0.0), act=(0.9, 0.3, 0.8, 0.6))
    config = regime_config(base, 4, 0.9, 0.2)
    accuracy = [1 / (1 + math.exp(-a)) for a in config.alphas]
    assert accuracy == pytest.approx([0.995, 0.9 + 0.1 / 3, 0.9 - 0.1 / 3, 0.8])  # 1.0 clips
    assert config.act == base.act and regime_config(base, 2, 0.7, 0.0).act == (0.9, 0.3)
    f1 = {"v3": {"hidden": 0.8, "ambiguity": 0.5, "weak": 0.3},
          "majority": {"hidden": 0.7, "ambiguity": 0.54, "weak": 0.4},
          "anchor_only": {"hidden": 0.78, "ambiguity": 0.0, "weak": 0.0},
          "v2_stage2": {"hidden": 0.5, "ambiguity": 0.0, "weak": 0.1}}
    assert versus(f1) == {"hidden": "T", "ambiguity": "T", "weak": "L"}
    f1["v3"]["ambiguity"] = 0.6
    assert versus(f1)["ambiguity"] == "W"


def test_the_simulate_command_writes_json_and_markdown(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(config_dict(TINY)), encoding="utf-8")
    params = tmp_path / "params.json"
    params.write_text(json.dumps(dataclasses.asdict(P)), encoding="utf-8")
    out = tmp_path / "sim"
    assert main(["simulate", "--sweep", "arms", "--replicates", "1", "--config", str(config),
                 "--params", str(params), "--out", str(out)]) == 0
    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert [p["point"] for p in written["sweeps"]["arms"]] == ["base", "a3", "a5", "within"]
    assert set(written["sweeps"]["arms"][0]["versus"]) == {"hidden", "ambiguity", "weak"}
    assert written["params"] == dataclasses.asdict(P) and load_config(config) == TINY
    assert written["replicates"] == {"arms": 1}
    report = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Sweep: arms" in report and "Exploratory: below §8.1's minimum of 10" in report
    assert main(["simulate", "--sweep", "nope", "--out", str(out)]) == 2
