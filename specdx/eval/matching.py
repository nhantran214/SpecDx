"""Scoring reports against ground truth (spec §4.6–§4.8)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from specdx.infer.scores import UNATTEMPTED
from specdx.mutate.operators import GroundTruth
from specdx.spec.tell import TellCorpus
from specdx.types import DEFECT_TYPES, Defect


@dataclass
class Prf:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": self.precision,
                "recall": self.recall, "f1": self.f1}


def location_matches(defect: Defect, truth: GroundTruth) -> bool:
    found = set(defect.location)
    kind = truth.location_kind
    if kind in ("clause", "field", "sentence"):
        return bool(found & truth.location)
    if kind == "conflict":
        union = found | truth.location
        return bool(union) and len(found & truth.location) / len(union) >= 0.5
    if kind == "chunk":
        return any(TellCorpus.chunk_of(s) in truth.location for s in found)
    raise ValueError(f"unknown location kind {kind!r}")


def match_findings(
    defects: Sequence[Defect],
    truth: GroundTruth,
    base: Sequence[Defect],
    *,
    typed: bool = True,
    k: int | None = None,
) -> tuple[list[Defect], list[Defect]]:
    """The findings new against the base report (top k if given), and those that match.
    A hidden finding whose refutation was never attempted carries no type evidence, so it
    is base noise when the base report holds its location under any type (§11)."""
    base_keys = {d.key for d in base}
    base_locations = {d.location for d in base}
    new = [d for d in defects if d.key not in base_keys
           and not (d.note == UNATTEMPTED and d.location in base_locations)]
    if k is not None:
        new = new[:k]
    hits = [d for d in new if (not typed or d.type == truth.type) and location_matches(d, truth)]
    return new, hits


def evaluate_reports(
    reports: Mapping[str, Sequence[Defect]],
    truths: Mapping[str, GroundTruth],
    base: Sequence[Defect],
    *,
    typed: bool = True,
    k: int | None = None,
) -> dict[str, Prf]:
    """One TP or FN per mutant under its true type; every unmatched new finding is an FP."""
    scores = {t: Prf() for t in DEFECT_TYPES}
    for mutant_id, defects in reports.items():
        truth = truths[mutant_id]
        new, hits = match_findings(defects, truth, base, typed=typed, k=k)
        if hits:
            scores[truth.type].tp += 1
        else:
            scores[truth.type].fn += 1
        for d in new:
            if d not in hits:
                scores[d.type if typed else truth.type].fp += 1
    return scores


# Types whose truths the baselines cannot locate: B2 and B3 report clause ids, which never
# match a Tell-sentence (ambiguity) or locked-field (weak) truth. Measured over every
# mutant: 0/7 A and 0/13 W truths, against 9/9 C and 7/7 H. Their untyped F1 there is 0 by
# construction, so beating it would prove nothing (spec §11 item 12).
BASELINE_BLIND: frozenset[str] = frozenset({"ambiguity", "weak"})
# SpecDx's typed F1 must reach this on a baseline-blind type. One TP and no FP on 5 truths
# gives 1/3, so the bar needs at least two true positives.
ABSOLUTE_BAR = 0.4


def _wins(
    mine: Mapping[str, Prf], theirs: Sequence[Mapping[str, Prf]]
) -> dict[str, bool]:
    return {t: mine[t].f1 > max(other[t].f1 for other in theirs) for t in DEFECT_TYPES}


def reaches_bar(score: Prf) -> bool:
    """F1 >= ABSOLUTE_BAR in exact arithmetic: F1 = 2tp / (2tp + fp + fn), and the float
    route can land a hair under the bar on an exact tie (for example tp 3, fp 5, fn 4)."""
    denominator = 2 * score.tp + score.fp + score.fn
    return bool(denominator) and Fraction(2 * score.tp, denominator) >= Fraction(
        str(ABSOLUTE_BAR))


def gate_rule(defect_type: str) -> str:
    """How the verdict judges this type, as the report states it."""
    if defect_type in BASELINE_BLIND:
        return f"≥ {ABSOLUTE_BAR:.2f} (baselines cannot locate it)"
    return "> baselines' untyped F1"


def gate_verdict(
    typed: Mapping[str, Mapping[str, Prf]],
    untyped: Mapping[str, Mapping[str, Prf]],
    *,
    method: str = "specdx",
    baselines: Sequence[str] = ("b2", "b3"),
    required: int = 3,
) -> tuple[bool, dict[str, bool]]:
    """Spec §4.8 as clarified in §11 item 12. Contradiction and hidden: SpecDx's typed F1
    must exceed every baseline's untyped F1 (the baselines report hidden only, so their
    typed F1 is no bar). Ambiguity and weak: the baselines cannot locate those truths at
    all, so SpecDx's typed F1 must reach ABSOLUTE_BAR. The gate needs `required` wins."""
    compared = _wins(typed[method], [untyped[b] for b in baselines])
    wins = {t: reaches_bar(typed[method][t]) if t in BASELINE_BLIND else compared[t]
            for t in DEFECT_TYPES}
    return sum(wins.values()) >= required, wins


def literal_wins(
    typed: Mapping[str, Mapping[str, Prf]],
    *,
    method: str = "specdx",
    baselines: Sequence[str] = ("b2", "b3"),
) -> dict[str, bool]:
    """§4.8 read literally, typed against typed: reported for transparency, never the verdict."""
    return _wins(typed[method], [typed[b] for b in baselines])
