import math

from specdx.eval.stats import holm, mcnemar_exact


def test_mcnemar_exact_matches_the_binomial() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(5, 0), 2 * 0.5 ** 5)
    assert math.isclose(mcnemar_exact(8, 2), 2 * sum(math.comb(10, i) for i in range(3)) / 2 ** 10)


def test_holm_step_down() -> None:
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06)
    assert math.isclose(adj["b"], 0.06)  # monotone: never below the previous adjusted value
