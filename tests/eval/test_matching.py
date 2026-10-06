from specdx.eval.matching import Prf, evaluate_reports, gate_verdict, location_matches
from specdx.mutate.operators import GroundTruth
from specdx.types import Defect


def _d(kind, *loc):
    return Defect(kind, tuple(loc), ("s",), (), 0.9)


def test_location_rules_per_kind() -> None:
    clause = GroundTruth("m", "hidden", frozenset({"c1"}), "clause")
    assert location_matches(_d("hidden", "c1"), clause)
    conflict = GroundTruth("m", "contradiction", frozenset({"a", "b"}), "conflict")
    assert location_matches(_d("contradiction", "a"), conflict)
    assert not location_matches(_d("contradiction", "a", "x", "y"), conflict)
    chunk = GroundTruth("m", "ambiguity", frozenset({"KB:KB-0003"}), "chunk")
    assert location_matches(_d("ambiguity", "KB:KB-0003:02"), chunk)
    assert not location_matches(_d("ambiguity", "soline.chiet_khau"), chunk)
    field = GroundTruth("m", "weak", frozenset({"pay.so_tien"}), "field")
    assert location_matches(_d("weak", "pay.so_tien", "pay.trang_thai"), field)


def test_base_findings_are_removed_and_errors_counted() -> None:
    truths = {"m1": GroundTruth("m1", "hidden", frozenset({"c1"}), "clause"),
              "m2": GroundTruth("m2", "weak", frozenset({"pay.so_tien"}), "field")}
    reports = {"m1": [_d("hidden", "c1"), _d("hidden", "noise")],
               "m2": [_d("hidden", "pay.so_tien")]}
    base = [_d("hidden", "noise")]
    typed = evaluate_reports(reports, truths, base)
    assert (typed["hidden"].tp, typed["hidden"].fp, typed["weak"].fn) == (1, 1, 1)
    untyped = evaluate_reports(reports, truths, base, typed=False)
    assert untyped["weak"].tp == 1


def test_f1_and_the_gate_verdict() -> None:
    assert Prf(tp=2, fp=2, fn=0).f1 == 2 * 0.5 * 1.0 / 1.5
    good, poor = Prf(tp=3, fp=1, fn=1), Prf(tp=1, fp=3, fn=3)
    types = ("contradiction", "hidden", "ambiguity", "weak")
    scores = {"specdx": dict.fromkeys(types, good), "b2": dict.fromkeys(types, poor),
              "b3": {t: (good if t == "hidden" else poor) for t in types}}
    passed, wins = gate_verdict(scores)
    assert passed and wins == {"contradiction": True, "hidden": False, "ambiguity": True,
                               "weak": True}
