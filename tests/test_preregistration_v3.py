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
from specdx.sim import generator as generate_module

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
    assert re.search(r"reported as a hidden requirement: reported\.", text)


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
                   "results/p2-spikes/g0-preview-full.md", "- F1: on an anchored cell an ambiguity",
                   "- F12 (a), (c): the calibration's plurality",
                   "No grid, threshold or seed was changed after either preview",
                   "results/p2-sim.md",
                   # §6, §7, §8 and §9: what the author decided and what execution ruled
                   "G0's weak-check bar is information only", "option B", "§13 item 15",
                   "results/p2-spikes/amended-preview/", "0.24–0.34",
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
                   "0.989 for gpt-oss:20b and 0.948 for qwen3:30b-a3b",
                   "results/p2-vocab-diagnosis.md",
                   "Only 4 findings have an out-of-vocabulary MAP value",
                   "T itself writes \"draft\"",
                   "SO-0031, INV-0031 and PAY-0031", "C declares no status domain",
                   "**The R-SO caveat.**", "readers follow T's \"draft\" while C has \"Chờ duyệt\"",
                   "It does not pass because the readers inferred",
                   "**The gate risk.**",
                   "Mutants may be masked by the base differential",
                   # review round 1: figures, M-step, timeline, diagnosis, caveat
                   "§13 items 8–15", "option B of A/B/C/D",
                   "ε_c(k) = (counts_k + rejected_k + κ·u_c(k)) / (total + κ)",
                   "(0.1, 0.2, 0.5, 1, 2, 5)", "exact EM step of the truncated error model",
                   "26. **Hidden suppresses weak.**",
                   "after the amended preview's weak failure was known",
                   "From the amended preview's weak failure on, every path is outcome-informed",
                   "No preview ran on the final code", "made on 2026-10-15, before G0 ran",
                   "Under G0 (ii) as registered before 2026-10-15, G0 fails",
                   "weak 0.338 < 0.4", "P2 would have stopped",
                   "0.084 (amended preview) and 0.088 (G0)", "both upward",
                   "kept within 0.25–3",
                   # final review: M4 (the ε prior), M6 (seen seeds), I1 (G3), I2 (disclosure)
                   "the posterior mean under Dirichlet(κ·u_c) given all draws",
                   "equivalently the mode under Dirichlet(1 + κ·u_c)",
                   "κ pseudo-counts towards u_c", "would apply κ − K",
                   'the "next ten" whose amended-preview result (weak 0.427) the author had '
                   "already seen", "out of sample for the selection, but not unseen",
                   "and at least one reading for each population model",
                   "**G3 amendment (2026-10-19).**",
                   "after this file was frozen and before any G2 or G3 run",
                   "mirroring ruling 21 for G1 (i)", "σ(1.0) = 0.731",
                   "passes θ ≥ 0.5 vacuously", "v2's Beta(7, 3) mode",
                   "`docs/RUNNING.md` §11", "`docs/RUNNING.md` §12",
                   "## 10. What the calibrated simulation exercises",
                   "`forms` = (0.4, 0.3, 0.3)", "`unanchored` = (0.5, 0.2, 0.3)",
                   "`kappa_true` = 30", "flat Dirichlet with concentration 0.3",
                   "the defect rate 0.15", "anchored share is 0.0326 of the value columns",
                   "seeds 0–9 of the calibrated configuration",
                   "32.5 on constraint cells", "2.4 on value cells",
                   "about 93% of the hidden truths are binary constraint cells",
                   "58.3 on unanchored value cells and 8.1 on anchored cells",
                   "mainly about hidden requirements on constraint cells",
                   "37 of G1's 53 real hidden findings",
                   "the selected prior matches the planted one",
                   "rounds this to ±0.08",
                   "8.06e-5", "`results/p2-g0.json`", "top 8 settings lie within 5.5e-4",
                   "typed without refutation 0/53/19/4",
                   "43 lie outside the status columns", "of the 10 status findings",
                   "0/54/16/2", "one new hidden finding",
                   "4 of the 19 reported ambiguity locations and 2 of the 4 weak-check locations",
                   "absolute 0.4 bar", "18 `apr` cells", "8 absent golden events",
                   "16 violated constraint clauses",
                   "No acted run wrote APR-0102", "`evt:APPROVED:APR-0102`",
                   "G1 (ii) passes nominally", "SpecDx flags a T↔C status conflict"):
        assert needle in text, needle


_BASELINES = {"majority": "majority", "anchor_only": "anchor-only", "v2_stage2": "v2 stage 2"}


def test_section_8_g0_figures_are_those_of_the_committed_record() -> None:
    g0_json = json.loads((ROOT / "results" / "p2-g0.json").read_text(encoding="utf-8"))
    text = DOC.read_text(encoding="utf-8")
    calibrated = g0_json["criteria"]["f1"]["calibrated"]
    v3 = calibrated["v3"]
    best = {}
    for kind in ("hidden", "ambiguity", "weak"):
        name = max(_BASELINES, key=lambda b, k=kind: calibrated[b][k])
        best[kind] = (calibrated[name][kind], _BASELINES[name])
    assert (f"hidden {v3['hidden']:.3f}, ambiguity {v3['ambiguity']:.3f}, "
            f"weak {v3['weak']:.3f} (information only)") in text
    assert ("against the best baseline's "
            + ", ".join(f"{best[k][0]:.3f} ({best[k][1]})" for k in ("hidden", "ambiguity"))
            + f" and {best['weak'][0]:.3f} ({best['weak'][1]})") in text
    assert (f"{v3['hidden'] - best['hidden'][0]:.3f} on hidden and "
            f"{v3['weak'] - best['weak'][0]:.3f} on weak") in text
    win = v3["ambiguity"] - best["ambiguity"][0]
    assert f"the ambiguity win over the best baseline is {win:.3f}" in text
    gaps = {(arm, kind): g0_json["criteria"]["f1"][arm]["v3"][kind] - max(
        g0_json["criteria"]["f1"][arm][b][kind] for b in _BASELINES)
        for arm in g0_json["criteria"]["f1"] for kind in ("hidden", "ambiguity", "weak")}
    worst = min(gaps, key=gaps.__getitem__)
    assert worst == ("a3", "weak")
    assert f"−{-gaps[worst]:.3f} against 0.05 allowed" in text
    assert f"objective {g0_json['chosen']['objective']:.4f}" in text
    assert f"mean |θ̂ − θ| {g0_json['criteria']['theta_error']:.4f}" in text
    out = g0_json["out_of_sample"]["f1"]["calibrated"]["v3"]["weak"]
    assert f"weak {out:.3f}" in text
    assert f"{out - v3['weak']:.3f} (G0)" in text  # §6: the selection-to-out-of-sample gap
    nc = g0_json["nonconverged"]
    assert f"{nc['search']} of {nc['search_fits']} search fits" in text
    assert (f"{nc['chosen']['a3']} of 10 A3 fits and {nc['chosen']['a5']} of 10 A5 fits") in text
    objectives = sorted((c["objective"] for c in g0_json["candidates"]), reverse=True)
    assert f"{(objectives[0] - objectives[1]) * 1e5:.2f}e-5" in text
    assert f"within {(objectives[0] - objectives[7]) * 1e4:.1f}e-4" in text


def test_section_8_g1_figures_are_those_of_the_committed_record() -> None:
    record = (ROOT / "results" / "p2-g1.md").read_text(encoding="utf-8")
    text = DOC.read_text(encoding="utf-8")
    theta = re.search(r"\(θ: gpt-oss:20b ([0-9.]+) \(\d+ readings\), "
                      r"qwen3:30b-a3b ([0-9.]+) \(\d+ readings\)\)", record)
    counts = re.search(r"(\d+) runs; (\d+) cells .*?; (\d+) readings", record)
    typed = re.search(r"Typed without refutation \(C/H/A/W\): (\d+/\d+/\d+/\d+)", record)
    assert theta and counts and typed
    assert f"θ_m is {theta[1]} for gpt-oss:20b and {theta[2]} for qwen3:30b-a3b" in text
    assert f"{counts[1]} runs, {counts[2]} cells, {counts[3]} readings" in text
    assert f"typed without refutation {typed[1]}" in text


def test_section_10_generator_constants_and_truth_mix_are_the_codes() -> None:
    text = DOC.read_text(encoding="utf-8")
    calibration = json.loads((ROOT / "results" / "p2-calibration.json").read_text(encoding="utf-8"))
    base = g0.load_config(ROOT / "results" / "p2-calibration.json")
    assert (base.forms, base.unanchored, base.kappa_true, base.defect_rate) == (
        (0.4, 0.3, 0.3), (0.5, 0.2, 0.3), 30.0, 0.15)
    assert base.forms == (V3Params().s_mis, V3Params().s_amb, V3Params().s_empty)
    assert "dirichlet(np.full(len(vocab), 0.3))" in inspect.getsource(generate_module)
    assert f"{calibration['anchored_share']:.4f} of the value columns" in text
    assert f"{calibration['constraint_share'] * 100:.1f}% constraint columns" in text
    assert f"{calibration['pooled_share'] * 100:.1f}% pooled" in text
    assert text.count(f"{calibration['columns']} columns") >= 1
    mix: dict[tuple[str, str, bool], int] = {}
    for seed in range(10):
        spec = generate_module.generate(dataclasses.replace(base, seed=base.seed + seed))
        for cid, cell in spec.cells.items():
            truth = spec.truth[cid]
            for name, flag in (("hidden", truth.hidden), ("ambiguity", truth.ambiguity)):
                if flag:
                    key = (name, cell.kind, cell.anchor is not None)
                    mix[key] = mix.get(key, 0) + 1
    per = {k: v / 10 for k, v in mix.items()}
    assert f"{per['hidden', 'constraint', True]:.1f} on constraint cells" in text
    assert f"{per['hidden', 'value', True]:.1f} on value cells" in text
    assert f"{per['ambiguity', 'value', False]:.1f} on unanchored value cells" in text
    anchored_amb = per["ambiguity", "constraint", True] + per["ambiguity", "value", True]
    assert f"{anchored_amb:.1f} on anchored cells" in text
    share = per["hidden", "constraint", True] / (
        per["hidden", "constraint", True] + per["hidden", "value", True])
    assert 0.92 < share < 0.94


def test_the_anchored_wording_matches_the_corrected_rationale() -> None:
    source = inspect.getsource(anchored)
    assert "posterior mean under Dirichlet(κ·u_c)" in source
    assert "equivalently the mode under Dirichlet(1 + κ·u_c)" in source
    assert "κ pseudo-counts towards u_c" in source
    assert "the literal mode of Dirichlet" not in source
    assert "the accuracy at β = 0" in source


def test_the_calibration_and_clip_values_the_document_states_are_in_the_code() -> None:
    assert calibrate.CONCENTRATIONS == (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    assert "min(max(p, 0.51), 0.995)" in inspect.getsource(study)
    assert "min(max(np.std(cell_logits), 0.25), 3.0)" in inspect.getsource(calibrate)
