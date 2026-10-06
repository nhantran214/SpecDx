import random

import numpy as np

from specdx.baselines.b2_single import b2_report
from specdx.baselines.b3_irt import b3_report, fit_2pl
from specdx.types import Behaviour, InterpreterId, Scenario, State

S = Scenario("grn12", 15, 12, 0)


def _b(key, model):
    return Behaviour(key, InterpreterId(model, "f", 0, 1), S, State({}))


def test_b2_flags_whatever_the_best_interpreter_violates() -> None:
    bs = [_b("a", "m1"), _b("b", "m1"), _b("c", "m2")]
    violated = {"a": frozenset({"x"}), "b": frozenset({"x", "y"}), "c": frozenset({"z"})}
    report = b2_report(bs, violated, {"m1": 0.9, "m2": 0.3}, {"grn12"})
    assert [(d.type, d.location, d.score) for d in report] == [
        ("hidden", ("x",), 1.0), ("hidden", ("y",), 0.5)
    ]


def test_fit_2pl_gives_negative_discrimination_to_a_miskeyed_item() -> None:
    rng = random.Random(0)
    rows = []
    for i in range(30):
        strong = i < 15
        row = [1.0 if (strong or rng.random() < 0.2) else 0.0 for _ in range(5)]
        row.append(0.0 if strong else 1.0)  # the miskeyed item: the strong get it "wrong"
        rows.append(row)
    _, a, _ = fit_2pl(np.array(rows))
    assert a[5] < 0 < min(a[:5])


def test_b3_reports_negatively_discriminating_clauses() -> None:
    rng = random.Random(1)
    bs, violated = [], {}
    for i in range(30):
        strong = i < 15
        bad = {f"c{j}" for j in range(5) if not (strong or rng.random() < 0.2)}
        if strong:
            bad.add("miskeyed")
        key = f"r{i}"
        bs.append(_b(key, "m"))
        violated[key] = frozenset(bad)
    report = b3_report(bs, violated)
    assert report and report[0].location == ("miskeyed",)
