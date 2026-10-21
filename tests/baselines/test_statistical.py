from types import SimpleNamespace

import specdx.baselines.jest_rule  # noqa: F401 - registers "jest"
import specdx.baselines.truth_inference  # noqa: F401 - registers "glad", "dawid_skene"
from specdx.baselines.api import REGISTRY
from specdx.baselines.votes import model_votes
from specdx.infer.cells import Cell, CellData, Reading
from specdx.types import Behaviour, InterpreterId, Scenario, State

S = Scenario("happy", 15, 15, 0)


def _data(spec: dict[str, tuple[str, dict[str, list[str]]]]):
    """cell -> (anchor, model -> readings); one run per reading."""
    cells, readings, runs = {}, [], []
    for cid, (anchor, by_model) in spec.items():
        cells[cid] = Cell(cid, "value", anchor, f"golden:{cid}", f"t.{cid}", cid, ("s1",))
        for m, values in by_model.items():
            for i, v in enumerate(values):
                key = f"{cid}-{m}-{i}"
                readings.append(Reading(cid, key, m, "fam-" + m[0], "happy", v))
                runs.append(Behaviour(key, InterpreterId(m, "fam-" + m[0], 0, i + 1), S,
                                      State.empty()))
    return CellData(cells, readings), runs


def _inp(spec):
    data, runs = _data(spec)
    return SimpleNamespace(data=data, report=SimpleNamespace(behaviours=runs, theta={}))


MISKEY = {
    "c1": ("d", {"strong1": ["x", "x", "x"], "strong2": ["x", "x", "x"], "weak": ["d", "y", "x"]}),
    "c2": ("d", {"strong1": ["d", "d", "d"], "strong2": ["d", "d", "d"], "weak": ["d", "y", "d"]}),
}


def test_model_votes_take_each_models_mode() -> None:
    data, _ = _data(MISKEY)
    assert model_votes(data)["c1"] == {"strong1": "x", "strong2": "x", "weak": "d"}


def test_jest_scores_the_share_of_deviating_models() -> None:
    out = REGISTRY["jest"].run(_inp(MISKEY))
    scores = {d.location: d.score for d in out.defects}
    assert scores == {("golden:c1",): 2 / 3}


def test_truth_inference_flags_a_miskeyed_cell_only() -> None:
    for name in ("glad", "dawid_skene"):
        out = REGISTRY[name].run(_inp(MISKEY))
        assert [d.location for d in out.defects if d.score > 0.5] == [("golden:c1",)], name


def test_nothing_to_judge_gives_an_empty_output() -> None:
    for name in ("jest", "glad", "dawid_skene"):
        assert REGISTRY[name].run(_inp({})).defects == []


def _fixed_posterior(monkeypatch, rows: dict[str, dict[str, float]]) -> None:
    import pandas as pd

    from specdx.baselines.truth_inference import TruthInference

    monkeypatch.setattr(TruthInference, "_proba", lambda self, df: pd.DataFrame(rows).T)


def test_truth_inference_reports_only_when_the_aggregate_differs_from_the_anchor(
        monkeypatch) -> None:
    spec = {c: ("d", {"m1": ["d"], "m2": ["x"], "m3": ["y"]}) for c in ("plural", "other")}
    _fixed_posterior(monkeypatch, {
        "plural": {"d": 0.4, "x": 0.35, "y": 0.25},  # plurality below 0.5 but it is the anchor
        "other": {"d": 0.3, "x": 0.45, "y": 0.25},  # the argmax is not the anchor
    })
    out = REGISTRY["dawid_skene"].run(_inp(spec))
    assert [(d.location, d.score) for d in out.defects] == [(("golden:other",), 0.7)]


def test_truth_inference_breaks_a_posterior_tie_by_label(monkeypatch) -> None:
    spec = {"c": ("x", {"m1": ["d"], "m2": ["x"]})}
    _fixed_posterior(monkeypatch, {"c": {"d": 0.5, "x": 0.5}})
    out = REGISTRY["glad"].run(_inp(spec))  # 'd' < 'x', so the aggregate is d, not the anchor x
    assert [d.location for d in out.defects] == [("golden:c",)]


def test_a_single_label_is_not_modelled(recwarn) -> None:
    spec = {"c1": ("d", {"m1": ["x"], "m2": ["x"]}), "c2": ("x", {"m1": ["x"], "m2": ["x"]})}
    for name in ("glad", "dawid_skene"):
        out = REGISTRY[name].run(_inp(spec))
        assert [(d.location, d.score) for d in out.defects] == [(("golden:c1",), 1.0)], name
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
