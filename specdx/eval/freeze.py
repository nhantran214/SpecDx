"""Selection on the dev split and the baseline pre-registration (baselines spec §3.3, §7).

Per baseline: the arm (model, strategy) and threshold τ maximising mean typed F1 over the
types it reports, on the primary (scorable) table. τ ranges over 0 and the deciles of the
arm's dev scores, except for the untuned baselines (spec §2.9), whose only τ is 0.0.
Ties: smaller model (the NNb in the tag), then higher τ, then label."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from specdx.baselines.api import REGISTRY, TELL_CHARS
from specdx.baselines.whowhen import K_RUNS
from specdx.eval.stats import RESAMPLES, SEED
from specdx.eval.study import CHECKPOINTS, Arm, SplitResult, score

# No threshold beyond the aggregate itself (spec §2.9; B5 has none), and the gate's own
# secondary arms at the gate's rule (spec §9.8): τ is 0.0 for these.
UNTUNED: frozenset[str] = frozenset({"glad", "dawid_skene", "b5_mus", "b2_naive", "b3_2pl"})

FIXED: dict[str, Any] = {
    "k_failing_runs": K_RUNS, "tell_chars": TELL_CHARS, "checkpoints": list(CHECKPOINTS),
    "bootstrap": {"resamples": RESAMPLES, "seed": SEED}, "grid": "deciles+0",
    "untuned": sorted(UNTUNED),
    "benchguard_commit": "7d6a52e5627a1a734ecd895e0c73778ab54aca63",
    "whowhen_commit": "b2bae5c5b06d681d04ea5e9b63b7a30525c04925",
}


def model_size(tag: str | None) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)b", tag or "")
    return float(m.group(1)) if m else float("inf")


def thresholds(result: SplitResult, arm: str) -> list[float]:
    """The grid: 0 and the deciles of the arm's dev scores; just 0 for an untuned baseline."""
    if arm.partition("@")[0] in UNTUNED:
        return [0.0]
    scores = [d.score for ds in result.defects.get(arm, {}).values() for d in ds]
    if not scores:
        return [0.0]
    deciles = np.quantile(np.array(scores), np.linspace(0, 0.9, 10))
    return sorted({0.0, *(round(float(x), 12) for x in deciles)})


def _parse(label: str) -> tuple[str, str | None, str | None]:
    name, _, rest = label.partition("@")
    model, _, strategy = rest.partition("/")
    return name, model or None, strategy or None


def choose(result: SplitResult) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, b in sorted(REGISTRY.items()):
        cands = []
        for label in sorted(result.defects):
            n, model, strategy = _parse(label)
            if n != name:
                continue
            for tau in thresholds(result, label):
                s = score(result, label, tau)
                f1 = sum(s[t].f1 for t in b.reports) / len(b.reports)
                cands.append((-f1, model_size(model), -tau, label, model, strategy, tau, f1))
        if cands:
            *_, label, model, strategy, tau, f1 = min(cands)
            out[name] = {"arm": label, "model": model, "strategy": strategy, "tau": tau,
                         "dev_f1": f1}
    return out


def canonical_json(doc: dict[str, Any]) -> str:
    """The one serialization the pre-registration is hashed and written in."""
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The files the test split is bound to (spec §9.8): `study` refuses when either changed.
BOUND: tuple[str, ...] = ("gate_config_sha256", "gate_result_sha256")


def prereg(result: SplitResult, source: Path, cache: Path | None = None,
           gate_config: Path | None = None, gate_result: Path | None = None) -> dict[str, Any]:
    """gate_config / gate_result: experiments/p2-gate.json and results/p2-gate.json, whose
    sha256 the pre-registration records so that the test split runs on exactly them."""
    chosen = choose(result)
    if missing := sorted(set(REGISTRY) - set(chosen)):
        raise ValueError("no dev arm for baseline(s) " + ", ".join(missing)
                         + ": an incomplete pre-registration is never written")
    doc: dict[str, Any] = {"schema": 1, "source": str(source),
                           "source_sha256": sha256_file(source),
                           "chosen": chosen, "fixed": FIXED}
    for key, path in zip(BOUND, (gate_config, gate_result), strict=True):
        if path is not None:
            doc[key] = sha256_file(path)
    if cache is not None and cache.exists():
        data = cache.read_bytes()
        lines = data.splitlines(keepends=True)
        doc["baseline_cache"] = {"lines": len(lines),
                                 "md5": hashlib.md5(b"".join(lines)).hexdigest()}
    return doc


def bound_mismatch(doc: dict[str, Any], gate_config: Path, gate_result: Path) -> str | None:
    """Why the current gate configuration or result is not the one the freeze recorded."""
    for key, path in zip(BOUND, (gate_config, gate_result), strict=True):
        if key not in doc:
            return f"the pre-registration records no {key}: re-run freeze-baselines"
        if not path.exists():
            return f"{path} is missing; the freeze recorded its sha256"
        if sha256_file(path) != doc[key]:
            return f"{path} changed since the freeze (sha256 differs from the recorded {key})"
    return None


def prereg_markdown(doc: dict[str, Any]) -> str:
    body = canonical_json(doc)
    lines = ["# Baseline pre-registration (baselines spec §3.3)", "",
             f"Chosen on the dev split from `{doc['source']}` (sha256 `{doc['source_sha256']}`).",
             f"JSON sha256: `{hashlib.sha256(body.encode()).hexdigest()}`.", "",
             "| baseline | arm | τ | dev mean typed F1 |", "|---|---|---:|---:|"]
    for name, c in doc["chosen"].items():
        lines.append(f"| {name} | {c['arm']} | {c['tau']:.6g} | {c['dev_f1']:.3f} |")
    lines += ["", "Fixed before selection: " + json.dumps(doc["fixed"], sort_keys=True), ""]
    return "\n".join(lines)


def study_arms(doc: dict[str, Any]) -> list[tuple[Arm, float]]:
    return [(Arm(n, c["model"], c["strategy"]), float(c["tau"]))
            for n, c in sorted(doc["chosen"].items())]
