"""G0's grid, selection rule, criteria, outputs and freeze (spec v3 §8.2, §9)."""

import dataclasses
import itertools
import json
from types import SimpleNamespace

import pytest

from specdx.infer.anchored import AnchoredFit, CellPosterior, V3Params, fit_anchored
from specdx.infer.cells import Cell, Reading
from specdx.sim import g0
from specdx.sim.generator import SimConfig, Truth, generate
from specdx.sim.study import config_dict, v3_counts

TINY = SimConfig(alphas=(2.5, 1.0), act=(1.0, 0.8), columns=14, items=4, scenarios=2,
                 seeds=2, seed=5)
P = V3Params(pi=0.1, s_mis=0.4, s_amb=0.3, s_empty=0.3, beta_sd=1.0,
             mu_alpha=0.0, sigma_alpha=1.5, tau_h=0.5, tau_a=0.3, tau_w=0.6)


def test_the_grid_and_the_criteria_are_the_pre_registered_ones() -> None:
    assert len(g0.prior_settings()) == 36 and len(g0.thresholds()) == 27
    assert g0.BETA_SD == (1.0, 2.0) and "kappa" not in {f.name for f in dataclasses.fields(P)}
    assert (g0.REPLICATES, g0.ARMS) == (10, ("base", "a3", "a5"))
    assert g0.THETA_ERROR_MAX == 0.05 and g0.AMBIGUITY_WIN == 0.05
    assert g0.F1_MIN == {"hidden": 0.6, "ambiguity": 0.4, "weak": 0.4}
    assert all(abs(p.s_mis + p.s_amb + p.s_empty - 1) < 1e-12 for p in g0.prior_settings())


def test_ties_go_to_the_more_conservative_setting_then_grid_order() -> None:
    def candidate(pi, s_mis, objective):
        return g0.Candidate(dataclasses.replace(P, pi=pi, s_mis=s_mis), objective, {})

    loose, firm, firmer = candidate(0.2, 0.4, 0.5), candidate(0.1, 0.4, 0.5), candidate(
        0.1, 0.6, 0.5)
    assert g0.choose([loose, firm, firmer]) is firmer  # higher P(|A| = 1)
    same = candidate(0.1, 0.6, 0.5)
    assert g0.choose([firmer, same]) is firmer  # then grid order
    assert g0.choose([firmer, candidate(0.2, 0.2, 0.6)]).objective == 0.6
    assert g0.conservative(firm.params) < g0.conservative(loose.params)


def test_thresholds_filter_the_zero_threshold_findings_as_the_typing_core_types() -> None:
    spec = generate(TINY)
    fit = fit_anchored(spec.cells, spec.readings, models=spec.models, params=P)
    found = g0.potential(spec, fit)
    # (0.9, 0.3, 0.6): τ_A below τ_H, where an ambiguity and its hidden requirement co-occur
    for tau_h, tau_a, tau_w in [(0.5, 0.3, 0.6), (0.9, 0.7, 0.9), (0.9, 0.3, 0.6),
                                (0.0, 0.0, 0.0)]:
        params = dataclasses.replace(P, tau_h=tau_h, tau_a=tau_a, tau_w=tau_w)
        expected = {t: c.as_dict() for t, c in v3_counts(spec, fit, params).items()}
        got = {t: c.as_dict() for t, c in g0.counts_at(spec, found, tau_h, tau_a, tau_w).items()}
        assert got == expected


def test_the_threshold_filter_keeps_an_ambiguity_and_its_hidden_requirement_together() -> None:
    # Ruling F1 in counts_at: the hand-built cells of Task 7's co-occurrence test, scored at
    # every threshold setting around them, agree with the typing core.
    f = frozenset
    cells = {"c": Cell("c", "value", "d", "golden:t:K:c", "t.c", "t.c", ("i0",)),
             "e": Cell("e", "value", "d", "golden:t:K:e", "t.e", "t.e", ("i0",)),
             "u": Cell("u", "value", None, None, "t.u", "t.u", ("i0",)),
             "w": Cell("w", "value", "d", "golden:t:K:w", "t.w", "t.w", ("i0",))}
    sets = {"c": {f({"d"}): 0.45, f({"d", "a"}): 0.4, f({"a"}): 0.15},
            "e": {f({"d"}): 0.05, f({"d", "a"}): 0.34, f({"a"}): 0.31, f(): 0.3},
            "u": {f({"x"}): 0.55, f({"x", "y"}): 0.45},
            # a hidden cell whose MAP set is another single value read cleanly by a deviant
            # behaviour: weak at τ_W ≤ 0.7, suppressed wherever it also reports hidden
            "w": {f({"x"}): 0.7, f({"d"}): 0.3}}
    posts = {c: CellPosterior(c, cells[c].anchor, tuple(sorted({v for x in s for v in x})), s,
                              0.0, {}) for c, s in sets.items()}
    fit = AnchoredFit({"m": 1.0}, {}, {}, posts, {}, True, 1, P)
    readings = [Reading("c", "m|0", "m", "m", "s", "a"), Reading("e", "m|1", "m", "m", "s", "a"),
                Reading("u", "m|2", "m", "m", "s", "y", True),
                Reading("w", "m|3", "m", "m", "s", "d", True)]
    truth = {"c": Truth("amb", ("a", "d"), True, True, False),
             "e": Truth("amb", ("a", "d"), True, True, False),
             "u": Truth("none", ("x",), False, False, True),
             "w": Truth("mis", ("x",), True, False, False)}
    spec = SimpleNamespace(cells=cells, readings=readings, truth=truth)
    found = g0.potential(spec, fit)
    for tau_h, tau_a, tau_w in itertools.product((0.5, 0.9), (0.3, 0.35, 0.5), (0.5, 0.9)):
        params = dataclasses.replace(P, tau_h=tau_h, tau_a=tau_a, tau_w=tau_w)
        expected = {t: c.as_dict() for t, c in v3_counts(spec, fit, params).items()}
        got = {t: c.as_dict() for t, c in g0.counts_at(spec, found, tau_h, tau_a, tau_w).items()}
        assert got == expected, (tau_h, tau_a, tau_w)
    at = g0.counts_at(spec, found, 0.9, 0.35, 0.5)
    assert at["hidden"].as_dict()["tp"] == 2 and at["ambiguity"].as_dict()["tp"] == 2
    # ruling (resolve_calls): w reports hidden, hence no weak check, at τ_H = 0.5; at
    # τ_H = 0.9 it reports weak instead
    fp = lambda c: c["weak"].as_dict()["fp"]  # noqa: E731
    assert fp(g0.counts_at(spec, found, 0.9, 0.35, 0.5)) == fp(
        g0.counts_at(spec, found, 0.5, 0.35, 0.5)) + 1


def test_the_verdict_needs_every_criterion_and_has_no_fallback() -> None:
    good = {"i": True, "ii": {"hidden": True, "ambiguity": True}, "ii_weak_information": True,
            "iii": {"calibrated": {"hidden": True, "ambiguity": True, "weak": True},
                    "a3": {"hidden": True, "ambiguity": True, "weak": True}},
            "ambiguity_win": True}
    assert g0.verdict(good) == g0.PASS
    assert g0.verdict({**good, "ii": {**good["ii"], "ambiguity": False}}) == g0.FAIL
    assert g0.verdict({**good, "ii": {**good["ii"], "hidden": False}}) == g0.FAIL
    # spec §13 item 15: the weak bar is information only, so failing it alone is a PASS
    assert g0.verdict({**good, "ii_weak_information": False}) == g0.PASS
    assert g0.verdict({**good, "i": False}) == g0.FAIL
    assert g0.verdict({**good, "ambiguity_win": False}) == g0.FAIL
    lost = {**good["iii"], "a3": {"hidden": True, "ambiguity": False, "weak": True}}
    assert g0.verdict({**good, "iii": lost}) == g0.FAIL
    # (iii) still compares weak
    weak_lost = {**good["iii"], "a3": {"hidden": True, "ambiguity": True, "weak": False}}
    assert g0.verdict({**good, "iii": weak_lost}) == g0.FAIL


def test_criterion_iii_is_judged_at_four_points_with_a_margin() -> None:
    base = SimConfig(alphas=(3.6, 3.0, 2.7, 2.2), act=(0.9, 0.35, 0.8, 0.58))
    assert g0.population_point(base) == (4, 0.95, 0.1)  # accuracies 0.97–0.90
    names = list(g0.evaluations(base))
    assert names == ["calibrated", "a3", "a5", "own M=4 acc=0.95 Δ=0.1"]
    f1 = {"v3": {"hidden": 0.74, "ambiguity": 0.8, "weak": 0.3},
          "majority": {"hidden": 0.7, "ambiguity": 0.6, "weak": 0.36},
          "anchor_only": {"hidden": 0.78, "ambiguity": 0.0, "weak": 0.0},
          "v2_stage2": {"hidden": 0.5, "ambiguity": 0.0, "weak": 0.1}}
    assert g0.holds(f1) == {"hidden": True, "ambiguity": True, "weak": False}


def test_a_g0_run_writes_its_report_and_the_chosen_parameters(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"config": config_dict(TINY)}), encoding="utf-8")
    grid = {"pi": (0.1,), "shares": ((0.4, 0.3, 0.3),), "beta_sd": (1.0,),
            "alpha_prior": ((0.0, 1.5),), "tau_h": (0.5,), "tau_a": (0.3, 0.5), "tau_w": (0.6,)}
    out, params = tmp_path / "results" / "p2-g0", tmp_path / "params.json"
    verdict = g0.run(calibration, out, replicates=1, grid=grid, params_out=params)
    assert verdict in (g0.PASS, g0.FAIL)
    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["verdict"] == verdict and len(written["candidates"]) == 2
    assert set(written["criteria"]) == {"theta_error", "f1", "i", "ii", "ii_weak_information",
                                        "iii", "ambiguity_win", "nonconverged"}
    assert set(written["criteria"]["ii"]) == {"hidden", "ambiguity"}
    assert len(written["criteria"]["iii"]) == 4
    # §3.4: non-convergence is counted in the search and at every criteria point
    nc = written["nonconverged"]
    assert nc["search_fits"] == 3 and set(nc["chosen"]) == set(g0.ARMS)
    assert 0 <= nc["search"] <= 3 and set(written["criteria"]["nonconverged"]) == set(
        written["criteria"]["iii"])
    assert all(set(x["nonconverged"]) == set(g0.ARMS) for x in written["candidates"])
    # ruling F22: an out-of-sample report on the next seeds, for information only
    assert set(written["out_of_sample"]) == set(written["criteria"])
    chosen = json.loads(params.read_text(encoding="utf-8"))["params"]
    assert chosen == written["chosen"]["params"] and chosen["tau_a"] in (0.3, 0.5)
    report = out.with_suffix(".md").read_text(encoding="utf-8")
    assert f"# G0: {verdict}" in report and "Non-converged fits (§3.4)" in report
    assert "## Out of sample (information only, not a criterion)" in report
    assert "information only, spec v3.1 §13 item 15" in report


def test_freeze_rewrites_only_the_marked_block(tmp_path) -> None:
    source = (g0.ANCHORED).read_text(encoding="utf-8")
    target = tmp_path / "anchored.py"
    target.write_text(source, encoding="utf-8")
    chosen = tmp_path / "params.json"
    chosen.write_text(json.dumps({"params": dataclasses.asdict(P)}), encoding="utf-8")
    g0.freeze(chosen, target)
    text = target.read_text(encoding="utf-8")
    assert "PARAMS_FROZEN = True" in text and text != source
    before, after = source.split("# --- G0 frozen parameters: begin ---")[0], source.split(
        "# --- G0 frozen parameters: end ---")[1]
    assert text.startswith(before) and text.endswith(after)
    block = g0._BLOCK.search(text).group(0)
    namespace: dict = {"dataclass": dataclasses.dataclass}
    exec(compile(block, "block", "exec"), namespace)  # noqa: S102 - our own generated code
    assert dataclasses.asdict(namespace["V3Params"]()) == dataclasses.asdict(P)
    with pytest.raises(ValueError, match="no G0 frozen parameters block"):
        g0.freeze(chosen, chosen)


def test_the_pre_registration_table_names_every_parameter() -> None:
    table = g0.params_table(P)
    assert table.splitlines()[:2] == ["| parameter | field | value |", "|---|---|---:|"]
    assert len(table.splitlines()) == 2 + len(dataclasses.fields(V3Params))
    assert "| π | `pi` | 0.1 |" in table and "| τ_W | `tau_w` | 0.6 |" in table
