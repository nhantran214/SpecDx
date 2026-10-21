import math
from types import SimpleNamespace

import pytest

from specdx.active.loop import SpecDxConfig
from specdx.baselines.semantic_entropy import REGISTRY, cell_score, entropy
from specdx.infer.cells import Cell, CellData, Reading


def test_entropy() -> None:
    assert entropy(["a", "a"]) == 0.0
    assert math.isclose(entropy(["a", "b"]), math.log(2))


def test_a_split_cell_scores_above_a_consensus_cell() -> None:
    split = [Reading("c", f"r{i}", "m", "f", "s", v) for i, v in enumerate("abab")]
    same = [Reading("c", f"r{i}", "m", "f", "s", "a") for i in range(4)]
    assert cell_score(split) > cell_score(same) == 0.0


def _inp(localizer, budget=80):
    cells = {"c1": Cell("c1", "value", "d", "golden:c1", "t.col1", "t.col1", ("S1",)),
             "c2": Cell("c2", "value", "d", "golden:c2", "t.col2", "t.col2", ("S1",))}
    readings = ([Reading("c1", f"a{i}", "m1", "f", "happy", v) for i, v in enumerate("abab")]
                + [Reading("c2", f"b{i}", "m1", "f", "sad", v) for i, v in enumerate("aaab")])
    ctx = SimpleNamespace(localizer=localizer, cells=object())
    cfg = SpecDxConfig({"f": ("m1",)}, localize_budget=budget)
    return SimpleNamespace(data=CellData(cells, readings), ctx=ctx, config=cfg,
                           variant="V", report=SimpleNamespace(theta={"m1": 0.9}))


def test_candidates_are_localised_highest_score_first() -> None:
    order = []

    def localizer(variant, defect, interps, budget, persists):
        order.append(defect.note)
        return (f"KB:{defect.note}",)

    out = REGISTRY["semantic_entropy"].run(_inp(localizer))
    assert order == ["happy|S1", "sad|S1"]
    assert [d.location for d in out.defects] == [("KB:happy|S1",), ("KB:sad|S1",)]
    assert out.defects[0].score > out.defects[1].score


def test_defects_localised_to_the_same_sentences_merge() -> None:
    out = REGISTRY["semantic_entropy"].run(_inp(lambda *a: ("KB:KB-0003:02",)))
    assert len(out.defects) == 1
    d = out.defects[0]
    assert d.location == ("KB:KB-0003:02",)
    assert math.isclose(d.score, math.log(2))
    assert d.scenarios == ("happy", "sad")
    assert d.witnesses == ("a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3")


def test_an_unlocalised_candidate_keeps_an_empty_location() -> None:
    out = REGISTRY["semantic_entropy"].run(_inp(lambda *a: None))
    assert len(out.defects) == 2
    assert all(d.location == () for d in out.defects)


def test_once_the_ddmin_budget_is_spent_the_localizer_is_not_called() -> None:
    calls = []

    def localizer(variant, defect, interps, budget, persists):
        calls.append(defect.note)
        budget.take("ddmin")
        return ("KB:KB-0003:02",)

    out = REGISTRY["semantic_entropy"].run(_inp(localizer, budget=1))
    assert calls == ["happy|S1"]
    assert [d.location for d in out.defects] == [("KB:KB-0003:02",), ()]


def test_localisation_needs_the_cell_lens() -> None:
    inp = _inp(lambda *a: None)
    inp.ctx.cells = None
    with pytest.raises(ValueError, match="cell lens"):
        REGISTRY["semantic_entropy"].run(inp)
