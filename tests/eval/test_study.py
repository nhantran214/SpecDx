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
    assert json.loads(text)["split"] == "dev"


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

    rep = SimpleNamespace(defects=[_d("hidden", "c1", 0.9)], behaviours=[], probe_runs=0,
                          ddmin_runs=0, checkpoints=[])
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


def _fake_split(monkeypatch, run, specdx=None):
    """run_split on base + one mutant H1, with SpecDx faked and one fake arm `run`."""
    from types import SimpleNamespace

    import specdx.eval.study as st

    rep = SimpleNamespace(defects=[_d("hidden", "c1", 0.9)], behaviours=[], probe_runs=0,
                          ddmin_runs=0, checkpoints=[])
    base_report = SimpleNamespace(defects=[], behaviours=[], probe_runs=0, ddmin_runs=0,
                                  checkpoints=[])

    def fake_specdx(variant, ctx, cfg, base=None):
        if variant == "B":
            return base_report
        return rep if specdx is None else specdx()

    monkeypatch.setattr(st, "run_specdx", fake_specdx)
    monkeypatch.setattr(st, "make_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "touch_counts", lambda *a, **k: {"H1": 3})
    monkeypatch.setattr(st, "scorable_ids", lambda t: {"H1"})
    monkeypatch.setattr(st, "REGISTRY", {"x": SimpleNamespace(uses_llm=False, strategies=(None,),
                                                               run=run)})
    cfg = SimpleNamespace(specdx=lambda: st.SpecDxConfig({"f": ("m",)}))
    bench = SimpleNamespace(base="B", loop_context=lambda v, c: SimpleNamespace(lens=None))
    mut = SimpleNamespace(id="H1", variant="V", truth=_t("hidden", "c1", "clause", "H1"))
    return lambda path: st.run_split(bench, cfg, [mut], [st.Arm("x")], None, path, "dev")


def test_an_infrastructure_error_in_an_arm_stops_the_split(monkeypatch, tmp_path) -> None:
    import pytest

    from specdx.baselines.llm import DigestMismatch
    from specdx.types import InfrastructureError

    for exc in (InfrastructureError("ollama down"), DigestMismatch("m:7b is not pulled")):
        def run(inp, exc=exc, **k):
            raise exc

        with pytest.raises(type(exc)):
            _fake_split(monkeypatch, run)(tmp_path / "out.jsonl")


def test_another_arm_error_fails_that_spec_only(monkeypatch, tmp_path) -> None:
    def run(inp, **k):
        raise ValueError("bad reply")

    out = tmp_path / "out.jsonl"
    r = _fake_split(monkeypatch, run)(out)
    assert r.failures == {"x": {"base": "ValueError: bad reply", "H1": "ValueError: bad reply"}}
    assert r.defects["x"] == {"base": [], "H1": []}
    assert score(r, "specdx")["hidden"].tp == 1
    assert [json.loads(line)["spec"] for line in out.read_text().splitlines()] == ["base", "H1"]


def test_a_mutant_specdx_cannot_run_fails_every_arm_on_it(monkeypatch, tmp_path) -> None:
    from specdx.baselines.api import BaselineOutput

    calls: list[object] = []

    def run(inp, **k):
        calls.append(inp)
        return BaselineOutput([])

    def broken():
        raise KeyError("no such step")

    r = _fake_split(monkeypatch, run, specdx=broken)(tmp_path / "out.jsonl")
    assert len(calls) == 1  # the base only
    assert set(r.failures) == {"specdx", "x"} and set(r.failures["x"]) == {"H1"}
    assert r.failures["x"]["H1"].startswith("no SpecDx report: KeyError")
    assert (score(r, "specdx")["hidden"].fn, score(r, "x")["hidden"].fn) == (0, 0)
    assert SplitResult.from_json(r.to_json()).to_json() == r.to_json()


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
