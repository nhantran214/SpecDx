import json

from specdx.eval.study import Arm, SplitResult, equal_cost, rank_metrics, score
from specdx.mutate.operators import GroundTruth
from specdx.types import Defect


def _t(type_, loc, kind, mid="M"):
    return GroundTruth(mid, type_, frozenset({loc}), kind)


def _d(type_, loc, s):
    return Defect(type_, (loc,), (), (), s)


def _result():
    truths = {"H1": _t("hidden", "c1", "clause", "H1"), "H2": _t("hidden", "c2", "clause", "H2")}
    defects = {
        "specdx": {"base": [], "H1": [_d("hidden", "c1", 0.9)], "H2": [_d("hidden", "zz", 0.8)]},
        "jest": {"base": [], "H1": [_d("hidden", "x", 0.9), _d("hidden", "c1", 0.5)],
                 "H2": []},
    }
    cps = {"H1": [{"at": 25, "runs": 25, "cost": {"requests": 40},
                   "defects": [{"type": "hidden", "location": ["c1"], "scenarios": [],
                                "witnesses": [], "score": 0.7, "note": "", "subtype": ""}]}],
           "H2": []}
    costs = {"specdx": {"H1": {"requests": 400}, "H2": {"requests": 400}},
             "jest": {"H1": {"requests": 0}, "H2": {"requests": 0}},
             "llm@m": {"H1": {"requests": 50}, "H2": {"requests": 10}}}
    defects["llm@m"] = {"base": [], "H1": [], "H2": []}
    return SplitResult("dev", truths, {"H1", "H2"}, defects, costs, {}, cps)


def test_typed_f1_uses_the_existing_matching() -> None:
    s = score(_result(), "specdx")
    assert (s["hidden"].tp, s["hidden"].fn, s["hidden"].fp) == (1, 1, 1)


def test_threshold_filters_before_scoring() -> None:
    assert score(_result(), "jest", tau=0.6)["hidden"].tp == 0
    assert score(_result(), "jest", tau=0.0)["hidden"].tp == 1


def test_ap_is_reciprocal_rank_and_detection_at_k() -> None:
    m = rank_metrics(_result(), "jest")["hidden"]
    assert m["ap"] == (1 / 2 + 0) / 2 and m["detect@1"] == 0 and m["detect@3"] == 0.5


def test_equal_cost_picks_the_checkpoint_within_the_arms_requests() -> None:
    eq = equal_cost(_result(), "llm@m")
    assert [d.location for d in eq["H1"]] == [("c1",)]   # checkpoint 40 ≤ 50 requests
    assert eq["H2"] == []                                  # no checkpoint within 10


def test_round_trip_is_byte_stable() -> None:
    r = _result()
    text = r.to_json()
    assert SplitResult.from_json(text).to_json() == text
    assert json.loads(text)["split"] == "dev" and "taus" not in json.loads(text)
    r.prereg_sha256, r.taus = "ab" * 32, {"jest": 0.5, "b5_mus": 0.0}
    text = r.to_json()
    back = SplitResult.from_json(text)
    assert back.to_json() == text and (back.prereg_sha256, back.taus) == (r.prereg_sha256, r.taus)


def test_an_arm_with_no_findings_scores_all_fn() -> None:
    s = score(_result(), "llm@m")
    assert (s["hidden"].tp, s["hidden"].fn) == (0, 2)


def test_arm_labels() -> None:
    assert Arm("whowhen", "m:7b", "step_by_step").label == "whowhen@m:7b/step_by_step"
    assert Arm("jest").label == "jest"


def test_run_split_is_deterministic(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    import specdx.eval.study as st
    from specdx.baselines.api import BaselineOutput
    from specdx.cost import Cost

    rep = SimpleNamespace(defects=[_d("hidden", "c1", 0.9)], behaviours=[], runs=0,
                          probe_runs=0, ddmin_runs=0, checkpoints=[])
    monkeypatch.setattr(st, "run_specdx", lambda *a, **k: rep)
    monkeypatch.setattr(st, "make_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "touch_counts", lambda *a, **k: {"H1": 3})
    monkeypatch.setattr(st, "scorable_ids", lambda t: {"H1"})
    fake = SimpleNamespace(uses_llm=False, strategies=(None,),
                           run=lambda inp, **k: BaselineOutput([_d("hidden", "c1", 0.4)], Cost()))
    monkeypatch.setattr(st, "REGISTRY", {"jest": fake})
    cfg = SimpleNamespace(specdx=lambda: st.SpecDxConfig({"f": ("m",)}))
    bench = SimpleNamespace(base="B", loop_context=lambda v, c: SimpleNamespace(lens=None))
    mut = SimpleNamespace(id="H1", variant="V", truth=_t("hidden", "c1", "clause", "H1"))
    a = st.run_split(bench, cfg, [mut], [st.Arm("jest")], None, tmp_path / "a.jsonl", "dev")
    b = st.run_split(bench, cfg, [mut], [st.Arm("jest")], None, tmp_path / "b.jsonl", "dev")
    assert a.to_json() == b.to_json()


def _behaviour(requests):
    from types import SimpleNamespace

    return SimpleNamespace(status="ok", harness={"s1": {"requests": requests}})


def _fake_split(monkeypatch, runs, specdx=None, make_input=None, requests=3, extra_runs=0):
    """run_split on base B + mutants H1 (variant V1) and H2 (V2), SpecDx faked with one run
    of `requests` requests per spec (plus extra_runs probe runs), and one fake arm per entry
    of runs (name -> run).
    specdx / make_input, if given, replace SpecDx on the mutants / the input builder; the
    fake input is the variant name."""
    from types import SimpleNamespace

    import specdx.eval.study as st

    def report(defects):
        return SimpleNamespace(defects=defects, behaviours=[_behaviour(requests)], runs=1,
                               probe_runs=extra_runs, ddmin_runs=0, checkpoints=[])

    def fake_specdx(variant, ctx, cfg, base=None):
        if variant == "B":
            return report([])
        if specdx is not None:
            return specdx(variant)
        return report([_d("hidden", "c" + variant[1:], 0.9)])

    monkeypatch.setattr(st, "run_specdx", fake_specdx)
    monkeypatch.setattr(st, "make_input", make_input or (lambda bench, v, *a, **k: v))
    monkeypatch.setattr(st, "touch_counts", lambda *a, **k: {"H1": 3, "H2": 3})
    monkeypatch.setattr(st, "scorable_ids", lambda t: {"H1", "H2"})
    monkeypatch.setattr(st, "REGISTRY", {
        n: SimpleNamespace(uses_llm=False, strategies=(None,), run=r) for n, r in runs.items()})
    cfg = SimpleNamespace(specdx=lambda: st.SpecDxConfig({"f": ("m",)}))
    bench = SimpleNamespace(base="B", loop_context=lambda v, c: SimpleNamespace(lens=None))
    muts = [SimpleNamespace(id=f"H{i}", variant=f"V{i}",
                            truth=_t("hidden", f"c{i}", "clause", f"H{i}")) for i in (1, 2)]
    arms = [st.Arm(n) for n in runs]
    return lambda path: st.run_split(bench, cfg, muts, arms, None, path, "dev")


def _empty(inp, **k):
    from specdx.baselines.api import BaselineOutput

    return BaselineOutput([])


def test_an_infrastructure_error_in_an_arm_stops_the_split(monkeypatch, tmp_path) -> None:
    import pytest

    from specdx.baselines.llm import DigestMismatch
    from specdx.types import InfrastructureError

    for exc in (InfrastructureError("ollama down"), DigestMismatch("m:7b is not pulled")):
        def run(inp, exc=exc, **k):
            raise exc

        def build(*a, exc=exc, **k):
            raise exc

        with pytest.raises(type(exc)):
            _fake_split(monkeypatch, {"x": run})(tmp_path / "out.jsonl")
        with pytest.raises(type(exc)):
            _fake_split(monkeypatch, {"x": _empty}, make_input=build)(tmp_path / "out.jsonl")


def test_another_arm_error_fails_that_spec_only(monkeypatch, tmp_path) -> None:
    from specdx.baselines.api import BaselineOutput

    def run(inp, **k):
        if inp == "V1":
            raise ValueError("bad reply")
        return BaselineOutput([])

    out = tmp_path / "out.jsonl"
    r = _fake_split(monkeypatch, {"x": run})(out)
    assert r.failures == {"x": {"H1": "ValueError: bad reply"}}
    assert r.defects["x"] == {"base": [], "H1": [], "H2": []}
    assert score(r, "specdx")["hidden"].tp == 2
    assert (score(r, "x")["hidden"].fn, score(r, "x")["hidden"].tp) == (1, 0)
    assert [json.loads(line)["spec"] for line in out.read_text().splitlines()] == [
        "base", "H1", "H2"]


def test_an_arm_that_fails_on_the_base_fails_everywhere(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def run(inp, **k):
        calls.append(inp)
        if inp == "B":
            raise ValueError("no base")
        return _empty(inp)

    r = _fake_split(monkeypatch, {"x": run, "y": _empty})(tmp_path / "out.jsonl")
    assert calls == ["B"]
    assert r.failures == {"x": {"base": "ValueError: no base",
                                "H1": "base failed: ValueError: no base",
                                "H2": "base failed: ValueError: no base"}}
    s = score(r, "x")["hidden"]
    assert (s.tp, s.fn, s.fp) == (0, 0, 0)
    assert score(r, "y")["hidden"].fn == 2


def test_a_mutant_specdx_cannot_run_fails_every_arm_on_it(monkeypatch, tmp_path) -> None:
    calls: list[object] = []

    def run(inp, **k):
        calls.append(inp)
        return _empty(inp)

    def broken(variant):
        raise KeyError("no such step")

    r = _fake_split(monkeypatch, {"x": run}, specdx=broken)(tmp_path / "out.jsonl")
    assert calls == ["B"]
    assert set(r.failures) == {"specdx", "x"} and set(r.failures["x"]) == {"H1", "H2"}
    assert r.failures["x"]["H1"].startswith("no SpecDx report: KeyError")
    assert (score(r, "specdx")["hidden"].fn, score(r, "x")["hidden"].fn) == (0, 0)
    assert SplitResult.from_json(r.to_json()).to_json() == r.to_json()


def test_an_input_that_cannot_be_built_fails_every_arm_on_that_spec(monkeypatch,
                                                                    tmp_path) -> None:
    def build(bench, variant, *a, **k):
        if variant == "V2":
            raise RuntimeError("the database refuses it")
        return variant

    out = tmp_path / "out.jsonl"
    r = _fake_split(monkeypatch, {"x": _empty, "y": _empty}, make_input=build)(out)
    reason = "no baseline input: RuntimeError: the database refuses it"
    assert r.failures == {"x": {"H2": reason}, "y": {"H2": reason}}
    assert "H2" in r.defects["specdx"] and len(out.read_text().splitlines()) == 3


def test_arms_that_read_specdx_runs_are_charged_those_runs_only(monkeypatch, tmp_path) -> None:
    from specdx.baselines.api import BaselineOutput
    from specdx.cost import Cost

    def own(inp, **k):
        return BaselineOutput([], Cost(requests=2))

    r = _fake_split(monkeypatch, {"jest": own, "benchguard": own}, requests=3, extra_runs=2)(
        tmp_path / "out.jsonl")
    # SpecDx: its one run (3 requests) plus two probe runs at the mean, a bound.
    assert (r.costs["specdx"]["H1"]["requests"], r.costs["specdx"]["H1"]["exact"]) == (9, False)
    # jest reads the population run (3, exact), not the probes, plus its own 2.
    assert (r.costs["jest"]["H1"]["requests"], r.costs["jest"]["H1"]["exact"]) == (5, True)
    assert r.costs["benchguard"]["H1"]["requests"] == 2


def test_equal_cost_reaches_specdx_final_report(monkeypatch, tmp_path) -> None:
    from specdx.baselines.api import BaselineOutput
    from specdx.cost import Cost

    def rich(inp, **k):
        return BaselineOutput([], Cost(requests=3))

    def poor(inp, **k):
        return BaselineOutput([], Cost(requests=2))

    r = _fake_split(monkeypatch, {"benchguard": rich, "b5_mus": poor}, requests=3)(
        tmp_path / "out.jsonl")
    assert r.checkpoints["H1"][-1]["at"] == "end"
    eq = equal_cost(r, "benchguard")
    assert eq == {"H1": r.defects["specdx"]["H1"], "H2": r.defects["specdx"]["H2"]}
    assert eq["H1"] and equal_cost(r, "b5_mus") == {"H1": [], "H2": []}


def test_equal_cost_skips_the_base_and_failed_specs() -> None:
    r = _result()
    r.checkpoints["base"] = r.checkpoints["H1"]
    assert set(equal_cost(r, "llm@m")) == {"H1", "H2"}
    r.failures = {"specdx": {"H1": "KeyError"}, "llm@m": {"H2": "ValueError"}}
    assert equal_cost(r, "llm@m") == {}


def test_rank_metrics_count_the_truths_of_each_type() -> None:
    m = rank_metrics(_result(), "jest")
    assert m["hidden"]["n"] == 2 and m["contradiction"]["n"] == 0


def test_dev_arms_cross_llm_baselines_with_every_model_and_strategy() -> None:
    from specdx.eval.study import dev_arms

    arms = dev_arms({"b": ("b:7b",), "a": ("a:3b", "a:8b")})
    labels = [a.label for a in arms]
    assert len(labels) == len(set(labels))
    assert "jest" in labels and not any(lbl.startswith("jest@") for lbl in labels)
    assert [lbl for lbl in labels if lbl.startswith("benchguard")] == [
        "benchguard@a:3b", "benchguard@a:8b", "benchguard@b:7b"]
    assert sum(lbl.startswith("whowhen@") for lbl in labels) == 3 * 3


def test_make_input_loads_the_variants_pack_before_its_golden_state() -> None:
    from types import SimpleNamespace

    from specdx.config import BASE_PACK
    from specdx.eval.study import make_input
    from specdx.orchestrate.work_orders import load_work_orders

    calls: list[object] = []
    db = SimpleNamespace(use=lambda p: calls.append(("use", p)),
                         golden_state=lambda: calls.append("golden") or "G")
    bench = SimpleNamespace(db=db, steps=load_work_orders(),
                            variants=SimpleNamespace(statements={"s": ("x",)}))
    run = SimpleNamespace(run_key="k1")
    ctx = SimpleNamespace(cells=SimpleNamespace(extract=lambda bs: ("data", len(bs))),
                          lens=SimpleNamespace(violations=lambda b: frozenset({b.run_key})))
    variant = SimpleNamespace(pack=BASE_PACK)
    inp = make_input(bench, variant, SimpleNamespace(behaviours=[run]), ctx, "cfg", "gw")
    assert calls == [("use", BASE_PACK), "golden"] and inp.golden == "G"
    assert inp.violated == {"k1": frozenset({"k1"})} and inp.data == ("data", 1)
    assert len(inp.clauses) > 0 and inp.gateway == "gw" and inp.config == "cfg"


def _refusing_input(variant):
    from types import SimpleNamespace

    from tonomyx.canon.pack import CanonPack

    from specdx.eval.study import make_input
    from specdx.orchestrate.work_orders import load_work_orders
    from specdx.scenarios.space import canon_scenarios

    def refuse():
        raise RuntimeError("the database refuses the canon's own golden state")

    db = SimpleNamespace(use=lambda p: None, golden_state=refuse)
    bench = SimpleNamespace(db=db, steps=load_work_orders(),
                            variants=SimpleNamespace(statements={}))
    ctx = SimpleNamespace(cells=SimpleNamespace(extract=lambda bs: None),
                          lens=SimpleNamespace(violations=lambda b: frozenset()),
                          canon=canon_scenarios(CanonPack(variant.pack)))
    return make_input(bench, variant, SimpleNamespace(behaviours=[]), ctx, "cfg", None)


def test_a_refused_golden_state_falls_back_to_the_unmaterialised_tables() -> None:
    from types import SimpleNamespace

    from tonomyx.canon.pack import CanonPack

    from specdx.config import BASE_PACK
    from specdx.db import golden_tables
    from specdx.types import State

    inp = _refusing_input(SimpleNamespace(pack=BASE_PACK))
    assert inp.golden == State(golden_tables(CanonPack(BASE_PACK)))
    assert inp.golden.derived == {} and inp.golden_materialised is False


def test_b5_and_b1_run_on_the_unmaterialised_golden_state(tmp_path, tell_corpus) -> None:
    import dataclasses

    from specdx.baselines.api import REGISTRY
    from specdx.baselines.llm import Reply
    from specdx.config import BASE_PACK
    from specdx.cost import Cost
    from specdx.variant import SpecVariant

    inp = _refusing_input(SpecVariant("base", BASE_PACK, tell_corpus))
    b5 = REGISTRY["b5_mus"].run(inp)
    assert b5.failure is None and b5.extra["golden"] == "unmaterialised"

    class G:
        cache_path = tmp_path / "c.jsonl"

        def chat(self, model, messages, **kw):
            return Reply("not json", Cost(1), "ok")

    b1 = REGISTRY["benchguard"].run(dataclasses.replace(inp, gateway=G()), model="m")
    assert b1.failure is None and b1.extra["golden"] == "unmaterialised"
    solution = tmp_path / "baselines" / "benchguard" / "base" / "m" / inp.clauses.anchor.id
    written = json.loads((solution / "solution" / "golden_state.json").read_text())
    assert written == inp.golden.tables


def test_the_report_renders_both_tables() -> None:
    from specdx.eval.study_report import study_markdown

    md = study_markdown(_result(), {"specdx": 0.0, "jest": 0.0, "llm@m": 0.0})
    assert "Table A" in md and "Table B" in md and "| llm@m |" in md


def _table_a(md: str, arm: str) -> dict[str, str]:
    lines = md.split("## Table A")[1].splitlines()
    head = next(x for x in lines if x.startswith("| arm |"))
    row = next(x for x in lines if x.startswith(f"| {arm} |"))
    cut = lambda x: [c.strip() for c in x.strip("|").split("|")]  # noqa: E731
    return dict(zip(cut(head), cut(row), strict=True))


def test_table_a_means_skip_types_without_truths_and_marks_incomplete() -> None:
    from specdx.eval.study_report import study_markdown

    r = _result()
    r.failures = {"jest": {"H2": "boom"}}
    md = study_markdown(r, {"specdx": 0.0, "jest": 0.0})
    specdx = _table_a(md, "specdx")
    # Only "hidden" has truths, so the mean is its own value: SpecDx hits H1 and misses H2.
    assert specdx["AP (mean)"] == "0.50" and specdx["Detection@1"] == "0.50"
    assert specdx["failed specs"] == "0"
    assert _table_a(md, "jest")["failed specs"] == "1 (INCOMPLETE)"


def test_the_dev_report_shows_table_a_in_sample_then_at_tau_zero() -> None:
    from specdx.eval.study_report import study_markdown

    md = study_markdown(_result(), {"specdx": 0.0, "jest": 0.0, "llm@m": 0.0},
                        in_sample={"specdx": 0.0, "jest": 0.6})
    first, second = md.split("## Table A: τ selected on dev (in-sample)")[1].split(
        "## Table A: τ = 0 (all candidates)")
    assert "| jest | 0.6 |" in first and "| llm@m |" not in first
    assert "| jest | 0 |" in second.split("## Paired")[0] and "| llm@m | 0 |" in second
    # jest's only hit scores 0.5, below τ 0.6: hidden F1 0 in-sample, 0.5 at τ 0.
    assert _table_a(md, "jest")["hidden F1"] == "0.00"
    assert "native protocol" not in md


def test_table_a_means_are_dashes_when_no_type_has_truths() -> None:
    from specdx.eval.study_report import study_markdown

    r = _result()
    r.truths, r.scorable = {}, set()
    row = _table_a(study_markdown(r, {"specdx": 0.0}), "specdx")
    assert row["AP (mean)"] == "–" and row["Detection@1"] == "–"


def test_table_b_caption_states_its_conventions_and_rows_cover_llm_arms() -> None:
    from specdx.eval.study_report import study_markdown

    md = study_markdown(_result(), {"specdx": 0.0, "jest": 0.0, "llm@m": 0.0})
    table_b = md.split("## Table B")[1]
    assert "unlocalised (spec §9.2)" in table_b
    assert "localised final report" in table_b
    assert "charged the population runs they read" in table_b and "not SpecDx's" in table_b
    assert "| llm@m |" in table_b and "| jest |" not in table_b


def test_paired_counts_discordant_mutants_and_the_ci_is_seeded() -> None:
    from specdx.eval.stats import bootstrap_ci, paired

    r = _result()
    assert paired(r, "specdx", "jest", 0.0, 0.0) == {
        "b": 0, "c": 0, "n": 2, "risk_difference": 0.0, "p": 1.0}
    # jest's only hit (score 0.5) falls below tau 0.6, so SpecDx alone detects H1.
    p = paired(r, "specdx", "jest", 0.0, 0.6)
    assert (p["b"], p["c"], p["n"], p["risk_difference"]) == (1, 0, 2, 0.5)
    assert bootstrap_ci(r, "specdx", 0.0) == bootstrap_ci(r, "specdx", 0.0)
    assert bootstrap_ci(_result(), "specdx", 0.0) == (0.0, 1.0)


def test_markdown_tables_have_matching_header_and_separator_widths() -> None:
    from specdx.eval.study_report import study_markdown

    md = study_markdown(_result(), {"specdx": 0.0, "jest": 0.0, "llm@m": 0.0})
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            assert lines[i - 1].count("|") == line.count("|") == lines[i + 1].count("|")


def _many(n_specdx: int, n_arm: int, n: int = 10) -> SplitResult:
    """n hidden mutants; SpecDx hits the first n_specdx of them, the arm "x" the first n_arm."""
    ids = [f"H{i}" for i in range(n)]
    truths = {m: _t("hidden", f"c{m}", "clause", m) for m in ids}
    hit = lambda m, k: [_d("hidden", f"c{m}", 0.9)] if int(m[1:]) < k else []  # noqa: E731
    defects = {"specdx": {"base": [], **{m: hit(m, n_specdx) for m in ids}},
               "x": {"base": [], **{m: hit(m, n_arm) for m in ids}}}
    costs = {"specdx": {m: {"requests": 5} for m in ids}, "x": {m: {"requests": 1} for m in ids}}
    return SplitResult("dev", truths, set(ids), defects, costs, {}, {})


def test_paired_ci_excludes_zero_when_specdx_detects_all_and_the_arm_none() -> None:
    from specdx.eval.stats import paired_bootstrap_ci

    lo, hi = paired_bootstrap_ci(_many(10, 0), "specdx", "x", 0.0, 0.0)
    assert lo == hi == 1.0
    lo, hi = paired_bootstrap_ci(_many(5, 5), "specdx", "x", 0.0, 0.0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = paired_bootstrap_ci(_many(7, 2), "specdx", "x", 0.0, 0.0)
    assert 0.0 < lo < 0.5 < hi < 1.0
    r = _many(10, 0)
    r.failures = {"x": {m: "boom" for m in r.truths}}
    assert paired_bootstrap_ci(r, "specdx", "x", 0.0, 0.0) == (0.0, 0.0)


def test_paired_table_prints_the_ci_and_the_holm_adjusted_p() -> None:
    from specdx.eval.study_report import study_markdown

    md = study_markdown(_many(10, 0), {"specdx": 0.0, "x": 0.0})
    row = next(x for x in md.splitlines() if x.startswith("| x | 10 |"))
    assert row == "| x | 10 | 0 | 10 | +1.00 | [+1.00, +1.00] | 0.002 |"
    md = study_markdown(_many(5, 5), {"specdx": 0.0, "x": 0.0})
    assert "| x | 0 | 0 | 10 | +0.00 | [+0.00, +0.00] | 1.000 |" in md


def test_table_b_row_carries_the_scores_the_mutant_count_and_the_incomplete_mark() -> None:
    from specdx.eval.study_report import study_markdown

    arms = {"specdx": 0.0, "llm@m": 0.0}
    md = study_markdown(_result(), arms)
    assert "| arm | arm requests | n mutants |" in md
    assert "| llm@m | 60 | 2 | 0.00 | 0.67 | 0.00 | 0.00 |" in md
    r = _result()
    r.failures = {"llm@m": {"H2": "boom"}}
    md = study_markdown(r, arms)
    assert "| llm@m (INCOMPLETE) | 60 | 1 | 0.00 | 1.00 | 0.00 | 0.00 |" in md
