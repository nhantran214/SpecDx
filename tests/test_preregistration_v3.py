"""Every value of the v3.1 pre-registration (spec v3.1 §9, experiments/p2-preregistration.md)
is the value the code uses, including those G0 chose. Changing one after a P2b result is
seen breaks this test."""

import dataclasses
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

from specdx.active.loop import SpecDxConfig
from specdx.eval import matching, observability, screen
from specdx.eval.pilot import GateConfig, load_population
from specdx.infer import act, anchored, cells, hitting
from specdx.infer.anchored import V3Params, fit_anchored
from specdx.infer.cells import Cell, Reading
from specdx.interpreters import executor, harness
from specdx.scenarios import tguided
from specdx.sim import calibrate, g0, study

ROOT = Path(__file__).parents[1]
DOC = ROOT / "experiments" / "p2-preregistration.md"
_CFG = SpecDxConfig.__dataclass_fields__


@pytest.mark.parametrize(("value", "registered"), [
    (executor.TEMPERATURE, 0.7), (harness.MAX_NUDGES, 2),
    (act.OMEGA, 0.3), (act.M_MIN, 3), (act.RASCH_L2, 1.0),
    (tguided.MAX_TBOUNDARY, 12), (observability.SCORABLE_K, 3),
    (matching.MIN_SCORABLE, 3), (matching.ABSOLUTE_BAR, 0.4),
    (inspect.signature(matching.gate_verdict).parameters["required"].default, 3),
    (inspect.signature(hitting.soft_hitting_sets).parameters["max_size"].default, 3),
    (_CFG["k_max"].default, 6), (_CFG["tau"].default, 0.05),
    (anchored.K_MAX, 6), (len(anchored.SETS), 22), (max(len(s) for s in anchored.SETS), 2),
    (anchored.EM_TOL, 1e-6), (anchored.EM_MAX_ITER, 200), (anchored.STARTS, (0.0, 1.0, 2.0, 3.0)),
    (screen.ACT_RATE_MIN, 0.3), (screen.VALID_RATE_MIN, 0.8), (screen.PER_FAMILY, 2),
    (screen.MAX_MODELS, 4), (screen.MIN_FAMILIES, 2), (screen.MIN_MODELS, 4),
    (anchored.KAPPA, 10.0),
    ({f.name for f in dataclasses.fields(V3Params)},
     {"pi", "s_mis", "s_amb", "s_empty", "beta_sd", "mu_alpha", "sigma_alpha", "tau_h", "tau_a",
      "tau_w"}),
    (cells.CONSTRAINT_KINDS,
     frozenset({"invariant", "expectation", "blank", "reconcile", "guard_pay", "step"})),
    (g0.PI, (0.05, 0.1, 0.2)),
    (g0.SHARES, ((0.6, 0.2, 0.2), (0.4, 0.3, 0.3), (0.2, 0.4, 0.4))),
    (g0.BETA_SD, (1.0, 2.0)), (g0.ALPHA_PRIOR, ((0.0, 1.5), (1.0, 1.0))),
    (g0.TAU_H, (0.5, 0.7, 0.9)), (g0.TAU_A, (0.3, 0.5, 0.7)), (g0.TAU_W, (0.6, 0.8, 0.9)),
    (g0.ARMS, ("base", "a3", "a5")), (g0.REPLICATES, 10),
    (g0.THETA_ERROR_MAX, 0.05), (g0.F1_MIN, {"hidden": 0.6, "ambiguity": 0.4, "weak": 0.4}),
    (g0.AMBIGUITY_WIN, 0.05), (study.MARGIN, 0.05),
    (study.REGIME_ACCURACY, (0.6, 0.7, 0.8, 0.9, 0.95)), (study.REGIME_DELTA, (0.0, 0.1, 0.2)),
    (study.REGIME_MODELS, (2, 4)),
    ((study.min_replicates("arms"), study.min_replicates("regime")), (10, 5)),
    (calibrate.BUDGET, 200), (calibrate.DEFECT_RATE, 0.15),
])
def test_the_code_uses_the_v3_pre_registered_values(value, registered) -> None:
    assert value == registered


def test_errors_pull_to_a_flat_pool_and_difficulty_is_grouped() -> None:
    cells_ = {c: Cell(c, "value", "d", f"golden:{c}", "t.x", g, ("s",))
              for c, g in (("c1", "t.x"), ("c2", "t.x"), ("c3", "t.y"))}
    readings = [Reading(c, f"{m}|{c}|{i}", m, m, "s", "d" if i % 4 else "e")
                for c in cells_ for m in ("m1", "m2") for i in range(8)]
    fit = fit_anchored(cells_, readings, models={"m1": "m1", "m2": "m2"}, params=V3Params())
    assert fit.cells["c1"].beta == fit.cells["c2"].beta  # one β per group (v3.1 §3.2)
    assert fit.cells["c1"].beta != fit.cells["c3"].beta
    flat = [Reading("c1", f"m1|{i}", "m1", "m1", "s", "d") for i in range(3)]
    alone = fit_anchored({"c1": cells_["c1"]}, flat, models={"m1": "m1"}, params=V3Params())
    # ε_c = (counts + κ·u_c)/(N + κ) with u_c flat over V_c and OTHER: the only error counts are
    # the anchor readings' share under ∅, 3·P(A_c = ∅)
    post = alone.cells["c1"]
    e = 3 * post.sets[frozenset()]
    kappa = anchored.KAPPA
    assert post.error["d"] == pytest.approx((kappa / 2 + e) / (kappa + e), abs=1e-4)
    assert post.error[anchored.OTHER] == pytest.approx((kappa / 2) / (kappa + e), abs=1e-4)


def test_the_pre_registration_states_the_values_g0_chose() -> None:
    # tests/test_g0_frozen.py (Task 13) pins V3Params to G0's choice; this pins the document.
    assert g0.params_table(V3Params()) in DOC.read_text(encoding="utf-8")


def test_the_g1_criteria_are_the_pre_registered_ones() -> None:
    spec = importlib.util.spec_from_file_location("p2_replay", ROOT / "experiments" /
                                                  "p2_replay.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.THETA_FLOOR == 0.5
    assert module.NO_COLLAPSE == ("gpt-oss:20b", "qwen3:30b-a3b")
    assert module.R_SO == "golden:so:SO-0031:trang_thai"


def test_the_committed_g1_record_passed_without_collapse() -> None:
    # Spec v3.1 §11's regression on the P1b smoke data is G1 (Tasks 12 and 14): its committed
    # record must stay a PASS with both named models at θ_m ≥ 0.5.
    text = (ROOT / "results" / "p2-g1.md").read_text(encoding="utf-8")
    assert text.splitlines()[2] == "# G1: PASS"
    theta = re.search(r"\(θ: gpt-oss:20b ([0-9.]+) \(\d+ readings\), "
                      r"qwen3:30b-a3b ([0-9.]+) \(\d+ readings\)\)", text)
    assert theta is not None and min(float(theta[1]), float(theta[2])) >= 0.5
    assert re.search(r"reported as a hidden requirement: (reported|not observable)\.", text)


def test_the_gate_v3_configuration_is_the_p1_split_and_budgets(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(ROOT)  # the screen path is relative to the repository root
    path = ROOT / "experiments" / "p2-gate.json"
    config = GateConfig.load(path)
    p1 = GateConfig.load(ROOT / "experiments" / "p1-gate.json")
    assert (config.dev_per_type, config.rng_seed) == (p1.dev_per_type, p1.rng_seed) == (
        5, 20261006)
    assert (config.budget, config.probe_budget, config.localize_budget) == (200, 100, 80)
    assert (config.variants, config.seeds) == ((0, 1, 2), (1, 2, 3))
    assert config.screen == Path("results/p2-screen.json")
    # The gate runs on the frozen V3Params, never on parameters a JSON carries.
    carried = tmp_path / "p2-gate.json"
    carried.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")),
                                   "params": {**dataclasses.asdict(V3Params()), "pi": 0.9}}),
                       encoding="utf-8")
    assert GateConfig.load(carried).params == V3Params()
    if config.models_by_family:  # frozen at G2: it must be the re-admitted population
        assert config.models_by_family == load_population(ROOT / "results" / "p2-screen.json")
    else:
        with pytest.raises(ValueError, match="empty"):
            config.check_population()


def test_the_pre_registration_states_each_value() -> None:
    text = DOC.read_text(encoding="utf-8")
    for needle in ("| Temperature | 0.7 |", "0–2 / 1–3", "200 / 100 / 80", "k = 3", "K_max = 6",
                   "τ = 0.05 / 3", "`dev_per_type` 5, `rng_seed` 20261006", "ω = 0.3, m_min = 3",
                   "λ = 1.0", "at most 12", "OLLAMA_NUM_PARALLEL", r"\|A_c\| ≤ 2",
                   "u_c flat over V_c, κ = 10, fixed", "an event cell's kind",
                   "a constraint cell's clause", "common α of 0, 1, 2 and 3",
                   "tolerance 1e-6", "at most 200 iterations", "no rank-correlation rule",
                   "co-occur", "≥ 0.3", "at least 4 acting models", "at least 10 replicates",
                   "5 per point of the regime map", "τ ∈ {1, 2}", r"mean \|θ̂ − θ\| ≤ 0.05",
                   "≥ 0.6 for hidden requirement",
                   "≥ 0.4 for ambiguity; the weak check's ≥ 0.4 is reported for information only",
                   "population's own M and Δ", "by more than 0.05",
                   "θ_m ≥ 0.5 for gpt-oss:20b and qwen3:30b-a3b", "R-SO",
                   "author ruling, 2026-10-14", "at least 2 families", "budget 60",
                   "highest θ_m", "π ∈ {0.05, 0.1, 0.2}", "τ_W ∈ {0.6, 0.8, 0.9}",
                   "Ten replicates", "equally weighted", "smaller π", "qwen3:14b (0.407)",
                   "no parameter is re-tuned", "test split stays untouched", "Both passed",
                   # §3: every planning resolution and ruling of the plan (rulings F10, F17,
                   # F18, F24)
                   "1. **Lifted vocabulary.**", "2. **Booleans.**",
                   "3. **Scope of tables no step drafts.**", "4. **Readings.**",
                   "5. **Difficulty groups.**", "6. **The M-step.**",
                   "7. **Refutation of a pooled candidate.**",
                   "8. **Singleton hitting sets defer to the cell path.**",
                   "9. **Defect locations.**", "10. **Provisional parameters.**",
                   "11. **Typing without an accuracy contrast, and co-occurrence.**",
                   "12. **The ddmin oracle.**", "13. **G2 on the cached screen.**",
                   "14. **Simulation design.**", "15. **Calibration.**",
                   "16. **G0 (iii)'s \"population's own M and Δ\"**",
                   "17. **θ_m and w_b.**", "18. **§5.2's V(b).**",
                   "never scored", "most probable defect form", "0.51–0.995",
                   "single-form", "population's readings", "shares of columns",
                   "raises instead of falling back", "every clause of C",
                   # §4 and §6: how G0 is judged, and what was seen before it (F22, F23)
                   "judged on the selection replicates", "for information only",
                   "results/p2-spikes/g0-preview-full.md", "F1", "F12",
                   "No grid, threshold or seed was changed after either preview",
                   "results/p2-sim.md",
                   # §6, §7, §8 and §9: what the author decided and what execution ruled
                   "G0's weak-check bar is information only", "option B", "§13 item 15",
                   "results/p2-spikes/amended-preview/", "0.24–0.34", "about ±0.08",
                   "weak 0.338", "0.427 out of sample",
                   "Every G0-altering change made after the first full preview",
                   "ambiguity and hidden co-occur", "calibration over the population's members",
                   "per-column cell-mix shares", "at least half its cells",
                   "the exact EM for the truncated error model",
                   "reports no weak check", "share v3.1's cell-level post-processing",
                   "19. **Numerals**", "int, bigint and numeric columns", "ids and other text",
                   "20. **The ddmin localiser**", "P(\\|A\\| = 2) ≥ τ_A",
                   "21. **G1 (i)**", "at least one reading for each named model",
                   "22. **A reading is clean**", "some in-scope acted item is clean",
                   "23. **The anchored prior is renormalised**", "has no candidate set",
                   "24. **Singleton hitting sets defer to the cell path**",
                   "a cell posterior below τ_H is not refuted",
                   "25. **Calibration's top error, |V| and reads**",
                   "every screened model's readings",
                   "**G0 (`results/p2-g0.md`): PASS**", "**G1 (`results/p2-g1.md`): PASS**",
                   "60 of 1080 search fits", "2 of 10 A3 fits and 1 of 10 A5 fits",
                   "about 8e-5", "0.989 for gpt-oss:20b and 0.948 for qwen3:30b-a3b",
                   "results/p2-vocab-diagnosis.md", "43 have nothing to do with vocabulary",
                   "Only 4 have an out-of-vocabulary MAP value", "T itself writes \"draft\"",
                   "SO-0031, INV-0031 and PAY-0031", "C declares no status domain",
                   "**The R-SO caveat.**", "readers follow T's \"draft\" while C has \"Chờ duyệt\"",
                   "It does not pass because the readers inferred",
                   "**The gate risk.**",
                   "Status-column mutants may be masked by the base differential"):
        assert needle in text, needle
