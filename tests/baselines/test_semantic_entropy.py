import math
from types import SimpleNamespace

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


def _inp(localizer):
    cells = {"c1": Cell("c1", "value", "d", "golden:c1", "t.col1", "t.col1", ("S1",)),
             "c2": Cell("c2", "value", "d", "golden:c2", "t.col2", "t.col2", ("S1",))}
    readings = ([Reading("c1", f"a{i}", "m1", "f", "happy", v) for i, v in enumerate("abab")]
                + [Reading("c2", f"b{i}", "m1", "f", "happy", v) for i, v in enumerate("aaab")])
    ctx = SimpleNamespace(localizer=localizer, cells=None)
    cfg = SpecDxConfig({"f": ("m1",)}, localize_budget=80)
    return SimpleNamespace(data=CellData(cells, readings), ctx=ctx, config=cfg,
                           variant="V", report=SimpleNamespace(theta={"m1": 0.9}))


def test_candidates_are_localised_highest_score_first() -> None:
    order = []

    def localizer(variant, defect, interps, budget, persists):
        order.append(defect.note)
        return ("KB:KB-0003:02",)

    out = REGISTRY["semantic_entropy"].run(_inp(localizer))
    assert [d.location for d in out.defects] == [("KB:KB-0003:02",), ("KB:KB-0003:02",)]
    assert out.defects[0].score > out.defects[1].score
    assert order == ["happy|S1", "happy|S1"]


def test_an_unlocalised_candidate_keeps_an_empty_location() -> None:
    out = REGISTRY["semantic_entropy"].run(_inp(lambda *a: None))
    assert all(d.location == () for d in out.defects)
