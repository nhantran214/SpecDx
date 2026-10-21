"""Markdown for the comparison study: Table A (native protocols), Table B (equal cost)."""

from __future__ import annotations

from collections.abc import Mapping

from specdx.eval.matching import evaluate_reports
from specdx.eval.stats import bootstrap_ci, holm, paired, paired_bootstrap_ci
from specdx.eval.study import equal_cost, rank_metrics, score
from specdx.types import DEFECT_TYPES


def _cost(result, arm: str) -> str:
    c = result.costs.get(arm, {})
    req = sum(int(v.get("requests", 0)) for v in c.values())
    exact = all(v.get("exact", True) for v in c.values())
    return f"{req}" + ("" if exact else " (bound)")


def _mean_over_truths(rows: Mapping[str, Mapping[str, float]], key: str) -> str:
    """Mean of key over the defect types that have truths in the split; "–" if none."""
    vals = [rows[t][key] for t in DEFECT_TYPES if rows[t]["n"] > 0]
    return f"{sum(vals) / len(vals):.2f}" if vals else "–"


def study_markdown(result, arms: Mapping[str, float]) -> str:
    lines = [f"# Comparison study: {result.split} split", "",
             "## Table A: each method at its native protocol (primary table, typed)", "",
             "AP (mean) and Detection@1 average over the defect types that have truths in "
             "this split.", "",
             "| arm | τ | " + " | ".join(f"{t} F1" for t in DEFECT_TYPES)
             + " | AP (mean) | Detection@1 | requests | detection 95% CI | failed specs |",
             "|---|---:|" + "---:|" * (len(DEFECT_TYPES) + 5)]
    for arm, tau in arms.items():
        s, r = score(result, arm, tau), rank_metrics(result, arm, tau)
        lo, hi = bootstrap_ci(result, arm, tau)
        failed = len(result.failures.get(arm, {}))
        lines.append(f"| {arm} | {tau:.3g} | " + " | ".join(f"{s[t].f1:.2f}" for t in DEFECT_TYPES)
                     + f" | {_mean_over_truths(r, 'ap')} | {_mean_over_truths(r, 'detect@1')}"
                     f" | {_cost(result, arm)} | [{lo:.2f}, {hi:.2f}] | "
                     + (f"{failed} (INCOMPLETE)" if failed else "0") + " |")
    rivals = {a: t for a, t in arms.items() if a != "specdx"}
    tests = {a: paired(result, "specdx", a, 0.0, t) for a, t in rivals.items()}
    adj = holm({a: v["p"] for a, v in tests.items()})
    lines += ["", "## Paired comparison with SpecDx (exact McNemar, Holm-adjusted)", "",
              "| arm | SpecDx only | arm only | n | risk difference | "
              "risk difference 95% CI | p (Holm) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for a, v in tests.items():
        lo, hi = paired_bootstrap_ci(result, "specdx", a, 0.0, rivals[a])
        lines.append(f"| {a} | {v['b']} | {v['c']} | {v['n']} | {v['risk_difference']:+.2f} | "
                     f"[{lo:+.2f}, {hi:+.2f}] | {adj[a]:.3f} |")
    lines += ["", "## Table B: SpecDx at each LLM arm's request budget", "",
              "Ambiguity at intermediate checkpoints is unlocalised (spec §9.2). The \"end\" "
              "checkpoint is SpecDx's localised final report. Arms that read SpecDx's runs are "
              "charged the population runs they read plus their own calls, not SpecDx's "
              "probes or ddmin (spec §9.8).", "",
              "| arm | arm requests | n mutants | "
              + " | ".join(f"SpecDx {t} F1" for t in DEFECT_TYPES) + " |",
              "|---|---:|---:|" + "---:|" * len(DEFECT_TYPES)]
    for a in rivals:
        if "@" not in a:
            continue
        eq = equal_cost(result, a)
        reports = {k: v for k, v in eq.items() if k != "base" and k in result.truths
                   and k in result.scorable}
        s = evaluate_reports(reports, result.truths, result.defects["specdx"].get("base", []))
        incomplete = " (INCOMPLETE)" if result.failures.get(a) else ""
        lines.append(f"| {a}{incomplete} | {_cost(result, a)} | {len(reports)} | "
                     + " | ".join(f"{s[t].f1:.2f}" for t in DEFECT_TYPES) + " |")
    return "\n".join(lines) + "\n"
