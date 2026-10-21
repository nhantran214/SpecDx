"""Markdown for the comparison study: Table A (native protocols), Table B (equal cost)."""

from __future__ import annotations

from collections.abc import Mapping

from specdx.eval.matching import evaluate_reports
from specdx.eval.stats import bootstrap_ci, holm, paired, paired_bootstrap_ci
from specdx.eval.study import equal_cost, rank_metrics, score
from specdx.types import DEFECT_TYPES

KS: tuple[int, ...] = (1, 3, 5)


def _spent(result, arm: str) -> list[dict]:
    """The arm's cost per specification, leaving out the specifications it failed on."""
    failed = result.failures.get(arm, {})
    return [v for k, v in result.costs.get(arm, {}).items() if k not in failed]


def _cost(result, arm: str) -> str:
    c = _spent(result, arm)
    req = sum(int(v.get("requests", 0)) for v in c)
    exact = all(v.get("exact", True) for v in c)
    return f"{req}" + ("" if exact else " (bound)")


def _usage(result, arm: str) -> tuple[str, str]:
    """Prompt + eval tokens and duration in seconds; "(bound)" when some runs predate the
    counters (those runs record neither, so both are then lower bounds)."""
    c = _spent(result, arm)
    mark = "" if all(v.get("exact", True) for v in c) else " (bound)"
    prompt = sum(int(v.get("prompt_tokens", 0)) for v in c)
    evals = sum(int(v.get("eval_tokens", 0)) for v in c)
    secs = sum(int(v.get("duration_ms", 0)) for v in c) / 1000
    return f"{prompt} + {evals}{mark}", f"{secs:.0f}{mark}"


def _mean_over_truths(rows: Mapping[str, Mapping[str, float]], key: str) -> str:
    """Mean of key over the defect types that have truths in the split; "–" if none."""
    vals = [rows[t][key] for t in DEFECT_TYPES if rows[t]["n"] > 0]
    return f"{sum(vals) / len(vals):.2f}" if vals else "–"


def _table_a(result, arms: Mapping[str, float], heading: str) -> list[str]:
    head = ([f"{t} F1" for t in DEFECT_TYPES] + [f"{t} F1 untyped" for t in DEFECT_TYPES]
            + [f"{t} AP" for t in DEFECT_TYPES] + ["AP (mean)"]
            + [f"Detection@{k}" for k in KS]
            + ["requests", "tokens (prompt + eval)", "duration (s)", "detection 95% CI",
               "failed specs"])
    lines = [f"## Table A: {heading}", "",
             "F1 is at the arm's τ. AP and Detection@k are threshold-free (τ = 0); their means "
             "average over the defect types that have truths in this split. Costs leave out "
             "the specifications an arm failed on; \"(bound)\": some runs predate the "
             "counters, so requests are an upper bound and tokens and duration a lower one.",
             "", "| arm | τ | " + " | ".join(head) + " |",
             "|---|---:|" + "---:|" * len(head)]
    for arm, tau in arms.items():
        s, u = score(result, arm, tau), score(result, arm, tau, typed=False)
        r = rank_metrics(result, arm, 0.0, ks=KS)
        lo, hi = bootstrap_ci(result, arm, tau)
        failed = len(result.failures.get(arm, {}))
        tokens, secs = _usage(result, arm)
        cells = ([f"{s[t].f1:.2f}" for t in DEFECT_TYPES]
                 + [f"{u[t].f1:.2f}" for t in DEFECT_TYPES]
                 + [f"{r[t]['ap']:.2f}" if r[t]["n"] else "–" for t in DEFECT_TYPES]
                 + [_mean_over_truths(r, "ap")]
                 + [_mean_over_truths(r, f"detect@{k}") for k in KS]
                 + [_cost(result, arm), tokens, secs, f"[{lo:.2f}, {hi:.2f}]",
                    f"{failed} (INCOMPLETE)" if failed else "0"])
        lines.append(f"| {arm} | {tau:.3g} | " + " | ".join(cells) + " |")
    return lines


def study_markdown(result, arms: Mapping[str, float],
                   in_sample: Mapping[str, float] | None = None) -> str:
    """arms: every arm and its τ; the paired comparison and Table B use them. in_sample (the
    dev report): the arms and τ chosen on this same split, shown as a first Table A, with
    `arms` (every candidate at τ = 0) as the second (spec §9.8)."""
    lines = [f"# Comparison study: {result.split} split", ""]
    if in_sample is None:
        lines += _table_a(result, arms, "each method at its native protocol (primary table, "
                                        "typed)")
    else:
        lines += _table_a(result, in_sample, "τ selected on dev (in-sample)")
        lines += ["", "τ and arm chosen on this same split: optimistic, not a held-out result.",
                  ""]
        lines += _table_a(result, arms, "τ = 0 (all candidates)")
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
