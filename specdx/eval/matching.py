"""Scoring reports against ground truth (spec §4.6–§4.8)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
    """The findings new against the base report (top k if given), and those that match."""
    base_keys = {d.key for d in base}
    new = [d for d in defects if d.key not in base_keys]
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


def _wins(
    mine: Mapping[str, Prf], theirs: Sequence[Mapping[str, Prf]]
) -> dict[str, bool]:
    return {t: mine[t].f1 > max(other[t].f1 for other in theirs) for t in DEFECT_TYPES}


def gate_verdict(
    typed: Mapping[str, Mapping[str, Prf]],
    untyped: Mapping[str, Mapping[str, Prf]],
    *,
    method: str = "specdx",
    baselines: Sequence[str] = ("b2", "b3"),
    required: int = 3,
) -> tuple[bool, dict[str, bool]]:
    """Spec §4.8 as clarified in §11: a type is a win when SpecDx's typed F1 exceeds every
    baseline's untyped F1 for it. The baselines cannot assign a type by construction (they
    report hidden only), so typed against typed would be won by any one true positive."""
    wins = _wins(typed[method], [untyped[b] for b in baselines])
    return sum(wins.values()) >= required, wins


def literal_wins(
    typed: Mapping[str, Mapping[str, Prf]],
    *,
    method: str = "specdx",
    baselines: Sequence[str] = ("b2", "b3"),
) -> dict[str, bool]:
    """§4.8 read literally, typed against typed: reported for transparency, never the verdict."""
    return _wins(typed[method], [typed[b] for b in baselines])
