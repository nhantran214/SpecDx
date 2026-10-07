import pytest

from specdx.eval.matching import (
    ABSOLUTE_BAR,
    BASELINE_BLIND,
    Prf,
    evaluate_reports,
    gate_rule,
    gate_verdict,
    literal_wins,
    location_matches,
    match_findings,
    reaches_bar,
)
from specdx.infer.scores import UNATTEMPTED
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


def test_an_unlocalised_ambiguity_matches_no_truth() -> None:
    unlocalised = Defect("ambiguity", (), ("s",), (), 0.9)
    chunk = GroundTruth("m", "ambiguity", frozenset({"KB:KB-0003"}), "chunk")
    sentence = GroundTruth("m", "ambiguity", frozenset({"KB:KB-0003:01"}), "sentence")
    assert not location_matches(unlocalised, chunk)
    assert not location_matches(unlocalised, sentence)
    scores = evaluate_reports({"m": [unlocalised]}, {"m": sentence}, [])
    assert (scores["ambiguity"].tp, scores["ambiguity"].fn) == (0, 1)


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


def test_an_unattempted_hidden_finding_at_a_base_location_is_base_noise() -> None:
    truth = GroundTruth("m", "hidden", frozenset({"c1"}), "clause")
    base = [Defect("contradiction", ("x",), ("s",), (), 1.0)]
    unattempted = Defect("hidden", ("x",), ("s",), (), 1.0, UNATTEMPTED)
    refuted = Defect("hidden", ("x",), ("s",), (), 1.0, "refuted by w")
    fresh = Defect("hidden", ("y",), ("s",), (), 1.0, UNATTEMPTED)
    new, _ = match_findings([unattempted, refuted, fresh], truth, base)
    assert new == [refuted, fresh]
    # Only a hidden finding's type can be a default; any other type is evidence
    odd = Defect("contradiction", ("z",), ("s",), (), 1.0, UNATTEMPTED)
    held = [Defect("hidden", ("z",), ("s",), (), 1.0)]
    assert match_findings([odd], truth, held)[0] == [odd]


def test_an_exact_f1_tie_with_a_baseline_is_no_win() -> None:
    types = ("contradiction", "hidden", "ambiguity", "weak")
    mine, theirs = Prf(tp=1, fp=0, fn=4), Prf(tp=2, fp=5, fn=3)  # both exactly 1/3
    assert mine.f1 > theirs.f1  # the float route would call it a win
    typed = {"specdx": dict.fromkeys(types, mine), "b2": dict.fromkeys(types, Prf(fn=5)),
             "b3": dict.fromkeys(types, Prf(fn=5))}
    untyped = {"specdx": typed["specdx"], "b2": dict.fromkeys(types, theirs),
               "b3": dict.fromkeys(types, theirs)}
    _, wins = gate_verdict(typed, untyped)
    assert not wins["contradiction"] and not wins["hidden"]


def test_f1_and_the_gate_verdict() -> None:
    assert Prf(tp=2, fp=2, fn=0).f1 == 2 * 0.5 * 1.0 / 1.5
    good, poor = Prf(tp=3, fp=1, fn=1), Prf(tp=1, fp=3, fn=3)
    types = ("contradiction", "hidden", "ambiguity", "weak")
    typed = {"specdx": dict.fromkeys(types, good), "b2": dict.fromkeys(types, Prf(fn=4)),
             "b3": dict.fromkeys(types, Prf(fn=4))}
    untyped = {"specdx": dict.fromkeys(types, good), "b2": dict.fromkeys(types, poor),
               "b3": {t: (good if t == "hidden" else poor) for t in types}}
    passed, wins = gate_verdict(typed, untyped)
    assert passed and wins == {"contradiction": True, "hidden": False, "ambiguity": True,
                               "weak": True}


def test_beating_the_baselines_typed_f1_alone_does_not_win_contradiction_or_hidden() -> None:
    types = ("contradiction", "hidden", "ambiguity", "weak")
    mine, zero, located = Prf(tp=2, fp=2, fn=2), Prf(fn=4), Prf(tp=4, fp=1, fn=0)
    typed = {"specdx": dict.fromkeys(types, mine), "b2": dict.fromkeys(types, zero),
             "b3": dict.fromkeys(types, zero)}
    untyped = {"specdx": dict.fromkeys(types, mine), "b2": dict.fromkeys(types, located),
               "b3": dict.fromkeys(types, zero)}
    assert all(literal_wins(typed).values())  # typed against typed: vacuous wins
    passed, wins = gate_verdict(typed, untyped)
    assert not passed and not wins["contradiction"] and not wins["hidden"]


def test_ambiguity_and_weak_must_reach_the_absolute_bar() -> None:
    assert sorted(BASELINE_BLIND) == ["ambiguity", "weak"] and ABSOLUTE_BAR == 0.4
    one_hit = Prf(tp=1, fp=0, fn=4)  # one TP of 5 truths, no FP
    two_hits = Prf(tp=2, fp=3, fn=3)  # precision 0.4, recall 0.4
    assert one_hit.f1 == pytest.approx(1 / 3) and two_hits.f1 == pytest.approx(0.4)
    good, zero = Prf(tp=3, fp=1, fn=1), Prf(fn=5)
    types = ("contradiction", "hidden", "ambiguity", "weak")
    typed = {"specdx": {"contradiction": good, "hidden": good, "ambiguity": one_hit,
                        "weak": two_hits},
             "b2": dict.fromkeys(types, zero), "b3": dict.fromkeys(types, zero)}
    untyped = {"specdx": typed["specdx"], "b2": dict.fromkeys(types, zero),
               "b3": dict.fromkeys(types, zero)}
    passed, wins = gate_verdict(typed, untyped)
    # 0.333 beats the baselines' 0 but is no win; 0.4 is
    assert wins == {"contradiction": True, "hidden": True, "ambiguity": False, "weak": True}
    assert passed
    assert "0.40" in gate_rule("weak") and "untyped" in gate_rule("hidden")


def test_an_exact_tie_with_the_bar_reaches_it() -> None:
    tie = Prf(tp=3, fp=5, fn=4)  # F1 = 6/15 = 0.4 exactly; the float route gives 0.3999…
    assert tie.f1 < ABSOLUTE_BAR and reaches_bar(tie)
    assert not reaches_bar(Prf(tp=1, fp=0, fn=4)) and not reaches_bar(Prf())


def test_untyped_false_positives_belong_to_the_mutant_type() -> None:
    truths = {"w": GroundTruth("w", "weak", frozenset({"pay.so_tien"}), "field")}
    reports = {"w": [_d("hidden", "pay.so_tien"), _d("hidden", "noise1"), _d("hidden", "noise2")]}
    untyped = evaluate_reports(reports, truths, [], typed=False)
    assert (untyped["weak"].tp, untyped["weak"].fp) == (1, 2)
    assert untyped["hidden"].fp == 0
    typed = evaluate_reports(reports, truths, [])
    assert typed["weak"].fn == 1 and typed["hidden"].fp == 3
