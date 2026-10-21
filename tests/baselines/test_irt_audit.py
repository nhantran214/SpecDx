from types import SimpleNamespace

import numpy as np

import specdx.baselines.irt_audit  # noqa: F401 - registers "irt_audit"
from specdx.baselines.api import REGISTRY
from specdx.baselines.irt_audit import delta_loglik
from specdx.infer.cells import Cell, CellData, Reading
from specdx.types import Behaviour, InterpreterId, Scenario, State

S = Scenario("happy", 15, 15, 0)
THETA = np.linspace(-2, 2, 12)
ONES = np.ones(12)


def _logistic(a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-a * (THETA - b)))


def test_a_normal_item_prefers_the_unit_ceiling() -> None:
    rng = np.random.default_rng(0)
    n = np.full(12, 9)
    for b in (-1.0, 0.0, 1.0):
        y = rng.binomial(n, _logistic(1.5, b))
        assert delta_loglik(y, n, THETA) < 0


def test_an_inverted_item_has_the_largest_ceiling_gain() -> None:
    rng = np.random.default_rng(0)
    n = np.full(12, 9)
    normal = [rng.binomial(n, _logistic(1.5, b)) for b in (-1.0, 0.0, 1.0)]
    inverted = rng.binomial(n, 1 - _logistic(1.5, 0.0))  # the strongest fail it most
    flat = rng.binomial(n, np.full(12, 0.1))  # nobody gets it
    scores = [delta_loglik(y, n, THETA) for y in [*normal, flat, inverted]]
    assert int(np.argmax(scores)) == 4
    assert scores[4] > 0 and all(s < 0 for s in scores[:3])
    assert scores[3] > max(scores[:3])


def test_the_score_is_the_average_log_likelihood_per_response() -> None:
    y = (THETA < 0).astype(float)  # exactly inverted
    one = delta_loglik(y, ONES, THETA)
    assert one > 0.5
    assert abs(delta_loglik(np.tile(y, 3), np.tile(ONES, 3), np.tile(THETA, 3)) - one) < 1e-9
    assert abs(delta_loglik(9 * y, np.full(12, 9.0), THETA) - one) < 1e-9


def test_a_fixed_chance_floor_is_respected() -> None:
    y = (THETA < 0).astype(float)
    assert delta_loglik(y, ONES, THETA, c=0.25) < delta_loglik(y, ONES, THETA)


def _inp(accuracy_by_cell: dict[str, list[bool]], n_models: int = 12):
    """Interpreter k (model m{k}, one seed) reads each cell as the anchor 'd' when
    accuracy_by_cell[cell][k], else 'x'."""
    cells, readings, runs = {}, [], []
    for cid, right in accuracy_by_cell.items():
        cells[cid] = Cell(cid, "value", "d", f"golden:{cid}", f"t.{cid}", cid, ("s1",))
        for k in range(n_models):
            key = f"{cid}-{k}"
            readings.append(Reading(cid, key, f"m{k:02d}", "fam", "happy",
                                    "d" if right[k] else "x"))
            runs.append(Behaviour(key, InterpreterId(f"m{k:02d}", "fam", 0, 1), S, State.empty()))
    return SimpleNamespace(data=CellData(cells, readings),
                           report=SimpleNamespace(behaviours=runs, theta={}))


def test_the_registered_baseline_flags_the_inverted_cell_only() -> None:
    cells = {f"ok{t}": [k >= t for k in range(12)] for t in (2, 3, 5, 6, 8, 9)}
    cells["miskey"] = [k < 4 for k in range(12)]  # only the weakest interpreters match it
    out = REGISTRY["irt_audit"].run(_inp(cells))
    assert REGISTRY["irt_audit"].reports == ("hidden",)
    assert [d.location for d in out.defects] == [("golden:miskey",)]
    d = out.defects[0]
    assert d.type == "hidden" and d.score > 0
    assert len(d.witnesses) == 8  # the eight interpreters whose reading deviates from d


def test_a_cell_nobody_matches_is_flagged() -> None:
    cells = {f"ok{t}": [k >= t for k in range(12)] for t in (2, 4, 6, 8, 10)}
    cells["nobody"] = [False] * 12
    out = REGISTRY["irt_audit"].run(_inp(cells))
    assert [d.location for d in out.defects] == [("golden:nobody",)]


def test_nothing_to_judge_gives_an_empty_output() -> None:
    assert REGISTRY["irt_audit"].run(_inp({})).defects == []
    one_cell = {"c": [k >= 4 for k in range(12)]}
    assert REGISTRY["irt_audit"].run(_inp(one_cell)).defects == []
    assert REGISTRY["irt_audit"].run(_inp({"a": [True] * 12, "b": [True] * 12}, 1)).defects == []


def test_interpreters_of_equal_ability_give_no_signal() -> None:
    same = {"a": [k % 2 == 0 for k in range(12)], "b": [k % 2 == 1 for k in range(12)]}
    assert REGISTRY["irt_audit"].run(_inp(same)).defects == []
