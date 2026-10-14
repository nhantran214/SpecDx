# P2 design spikes: from v3 to v3.1 (simulation only)

On 2026-10-14, before any P2 implementation, three simulation runs tested the v3 design:
- the P2a plan's pre-check;
- spike 1;
- spike 2.

All three used only the simulator, calibrated to summary statistics of the cached base runs: the P1b smoke run and the P1b screen.
- No mutant was looked at, nothing had been pre-registered yet, and no LLM was called.
- The run cache (`work/runs.jsonl`, md5 `db1e1823…`) was unchanged before and after every run.

The code was throwaway: it lived in a scratch copy of the P2a plan's pre-check. The per-run reports are in `results/p2-spikes/`. These results led to spec v3.1 (`docs/superpowers/specs/2026-10-14-specdx-v3-design.md`, §13 items 8–14).

## Calibration

The simulator was calibrated to cell-level statistics of the cached base runs.

| Statistic | Value |
|---|---|
| Accuracy per model | gpt-oss:20b 0.974; mistral-small3.2:24b 0.955; qwen3:30b-a3b 0.940; qwen2.5:14b 0.897 |
| Act rate per model | 0.914, 0.346, 0.802 and 0.580 (same order) |
| Readings per model and cell | about 6 |
| Share of the error mass held by the most frequent wrong value | 0.84 |

At cell level, the screened models agree far more than they did on v2's joint class keys, where agreement was about 27–30%.

## Pre-check: v3 as specified

This is the reduced G0 (4 prior settings × 27 threshold settings, 3 replicates). Report: `p2-spikes/precheck-reduced-g0.md`.

| Method | Hidden | Ambiguity | Weak |
|---|---:|---:|---:|
| v3 | 0.72 | 0.006 | 0.32 |
| Majority vote | 0.77 | 0.68 | 0.32 |
| Anchor-only | 0.77 | 0 | 0 |
| v2 stage 2 | 0.49 | 0 | 0.06 |

The verdict was FAIL: ambiguity was near 0, and v3 lost to the simple baselines.

## Spike 1: per-column difficulty (β_col)

| Run | v3 (hidden / ambiguity / weak) | Majority | Anchor-only |
|---|---|---|---|
| Per-cell β, arm (i) | 0.76 / 0.01 / 0.41 | 0.78 / 0.65 / 0.37 | 0.82 / 0 / 0 |
| β_col, arm (i) | 0.81 / 0.01 / 0.40 | 0.78 / 0.65 / 0.37 | 0.82 / 0 / 0 |
| β_col, arm (ii), τ_within 1.0 | 0.69 / 0.01 / 0.35 | 0.77 / 0.64 / 0.35 | 0.76 / 0 / 0 |

Findings:
- **Silence absorbs ambiguity.** The real competitor of a pair is ∅ (silence) with a free per-cell error distribution, which fits any split: at true pair cells, ∅ won in about 80 of 83 cases. Grouping difficulty by column does not change that.
- **The Spearman robustness rule passed at about 0% of true pairs**, with about 6 readings per model.
- **Three replicates are too noisy.** Changing only the RNG stream moved weak F1 from 0.32 to 0.41.
- **Difficulty spread.** Within a value column it is at sampling-noise level. Within a constraint clause kind it is about 1.6.

## Spike 2: the v3.1 candidate

The candidate made four changes:
- a flat error pool with κ ∈ {10, 50, fixed};
- no Spearman rule;
- β grouped per value column and per clause;
- 10 replicates.

### Calibrated point

| κ | v3.1 (hidden / ambiguity / weak) | Majority | Anchor-only | v2 stage 2 | Mean θ error |
|---|---|---|---|---|---|
| 10 | **0.83 / 0.81 / 0.45** | 0.73 / 0.69 / 0.40 | 0.78 / 0 / 0 | 0.45 / 0 / 0.11 | 0.016 |
| 50 | 0.81 / 0.79 / 0.44 | same | same | same | 0.018 |
| fixed | 0.79 / 0.78 / 0.43 | same | same | same | 0.019 |

- **Criteria.** v3.1 passes every reduced-G0 criterion of v3 except the absolute weak-check bar of 0.6, which no method reaches.
- **Misspecification arms.** v3.1 also wins under both:
  - A3 (model-specific errors): hidden 0.87, ambiguity 0.63;
  - A5 (model-dependent silence): hidden 0.88, ambiguity 0.60.

### Regime map

κ = 10, 5 replicates per point. Each entry is win (W), tie (T) or loss (L) against the best baseline, with a margin of 0.05, for hidden / ambiguity / weak. "T/L" means it varies across the accuracy levels of that row.

| M | Accuracy | Δ = 0 | Δ = 0.1 | Δ = 0.2 |
|---|---|---|---|---|
| 4 | 0.95 | TTT | TWW | TWW |
| 4 | 0.9 | TTT | TTW | WWW |
| 4 | 0.8 | TLW | LLT | TLW |
| 4 | 0.6–0.7 | TLW | TLW | T L/T W |
| 2 | 0.9–0.95 | TLT | T/L L T | T T/L W |
| 2 | 0.6–0.8 | TLT | T L T/W | TLW |

- **Where v3.1 wins.** Only at high accuracy with four models. The calibrated population is in that regime.
- **Two models.** With two models, ambiguity loses to majority vote at 13 of 15 points.
- **Hidden requirements** mostly tie with anchor-only.

## Decisions taken (author, 2026-10-14)

Spec v3.1 adopts the spike-2 candidate with κ = 10. It also revises G0:
- weak check ≥ 0.4, the gate's own bar;
- a regime-based criterion (iii), which requires a win on ambiguity at the calibrated point;
- at least 10 replicates;
- τ ∈ {1, 2} in the grid.

It requires at least 4 acting models (G2), and judges G1 (ii) on the P1b smoke run only.
