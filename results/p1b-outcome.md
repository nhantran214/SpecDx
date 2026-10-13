# P1b outcome: stopped before gate v2

**Decision (author, 2026-10-14).** P1b stops before gate v2. The gate was not run: no development mutant was run, no gate result exists, and the test split was never touched. The decision was taken on base-specification evidence only (the screen and the smoke run). It is not a tuning step, and no pre-registered value was changed.

The pre-registration (`experiments/p1b-preregistration.md`) and the frozen gate configuration (`experiments/p1b-gate.json`, with the screened population) are left unchanged, so a later run can use them as they are.

## What ran

| Stage | Date (KST) | Result |
|---|---|---|
| Spec v2 implementation (plan Tasks 1–17) | 2026-10-12 | All tasks reviewed, plus a whole-branch review and its fix wave; 447 tests pass. |
| Offline replay on the P1 cache (Task 7) | 2026-10-12 | GO on the mechanism check (`results/p1b-replay.md`). |
| Author decisions before the screen | 2026-10-12 | Pilot escalation counts acted views only; `--workers 1`, because `OLLAMA_NUM_PARALLEL` is unset on this host. |
| Capability screen (Task 18 Step 3) | 2026-10-12, 15:01–19:07 | Population: gpt-oss:20b (openai) and qwen3:30b-a3b (qwen) (`results/p1b-screen.md`). |
| Smoke run on the base specification (Step 5) | 2026-10-12 19:11 to 2026-10-13 18:33 | Every pre-registered check passes (see below). |
| θ diagnosis (read-only) | 2026-10-14 | Stage 2 does not identify θ on this population (`results/p1b-theta-diagnosis.md`). |

### Capability screen

Admission requires act rate ≥ 0.6 and valid-call rate ≥ 0.8, from 9 runs per model on `happy`.

| Model | Family | Act rate | Valid-call rate | Admitted | Mean s/run |
|---|---|---:|---:|---|---:|
| gpt-oss:20b | openai | 0.914 | 0.988 | yes | 140 |
| qwen3:30b-a3b | qwen | 0.802 | 0.952 | yes | 944 |
| qwen2.5:14b | qwen | 0.580 | 0.943 | no | 157 |
| qwen3:14b | qwen | 0.407 | 1.000 | no | 52 |
| mistral-small3.2:24b | mistral | 0.346 | 0.914 | no | 213 |
| llama3.1:8b | meta | 0.148 | 0.828 | no | 46 |
| granite3.3:8b | ibm | 0.025 | 1.000 | no | 97 |

The hardware was one NVIDIA RTX 5070 Ti (16 GB) running Ollama 0.30.4. qwen3:30b-a3b does not fit in 16 GB and runs about 30% on the CPU.

### Smoke run (base, population budget 60)

- **Runs:** 142 in all, 60 population and 100 probe. 18 of the population runs were cache hits from the screen. There were 0 error runs.
- **Pre-registered checks:**
  - ReadTimeout claims: 0.
  - Both models acted: 449 acted behaviours over 36 items.
  - Duplicate-evt runs: 0. Calls parsed from text: 11.
  - θ ordering: OK. The highest-κ model, gpt-oss:20b (κ 2.146), has θ 0.0967, above qwen3:30b-a3b's 0.0781.
- **Report** (`results/p1b-smoke-base.json`):
  - all 36 items observable, 0 Unobservable;
  - defects: 9 contradiction, 1 hidden, 4 weak;
  - S2 tested at 0 items.

## Why the gate was not run

**θ is below the competence threshold.** Stage 2 gives θ = 0.097 for gpt-oss:20b and 0.078 for qwen3:30b-a3b. Both are far below the pre-registered competence threshold θ ≥ 0.5.

**The diagnosis.** It is in `results/p1b-theta-diagnosis.md`, and its scripts are in `experiments/p1b_theta_diagnosis/`. It finds:

- **No code bug.** The code computes the pre-registered formulas. EM reaches the same unique optimum from every start, and the profile likelihood confirms it.
- **The models genuinely disagree.**
  - Only 26.8% of same-item run pairs agree on the class key between the two models, and 29.6% within one model.
  - Runs on the same paraphrase agree barely more often than runs on different paraphrases, so the variation is mostly seed noise.
  - The disagreement is substantive: drafted quantities (`po.sl_dat`), whether APPROVED is logged, and invented PO ids.
- **The class key adds some noise.** Sources are invented surrogate ids, case and language variants of status values, placeholders, and rows inherited through shared slices. Removing all of it lifts θ only to 0.381 / 0.100.
- **Stage 2 does not identify the level of θ.** The pre-registered model generates p = θ·δ_A + (1 − θ)·ε, where ε is per item, shared by all models, and allowed on admissible classes. Along a one-parameter family, θ and ε fit equally well.
  - The level of θ is set by the Beta prior, the smoothing of ε and a feasibility bound, and θ ends up near the share of a model's runs in the plurality joint class.
  - The re-estimated ε falls back to P1's plug-in, so planning resolution 4 makes no difference on this data.
- **P1's high θ measured agreement on writing nothing.** P1's θ (0.61–0.96) came from items where nothing was written (DE-OP-01, P2, P6), with the priors acting on a flat likelihood.
- **The θ-ordering check carries no information.** A bootstrap gives P(θ_gpt > θ_qwen) ≈ 0.5.

**Gate prognosis.** The gate needs 3 of 4 wins.

| Type | Expected effect |
|---|---|
| Ambiguity | Cannot be found. S1 ≤ 0.12, and S2 and the ddmin oracle admit no interpreter. |
| Weak | Follows plurality noise. All 4 base weak findings are noise. |
| Hidden | Depressed. |
| Contradiction | Unaffected. |

A NOT PASSED was the expected outcome.

**Cost.** With this population, the gate is about 6,300–7,800 runs at a mean of about 540 s per run on this machine, roughly 40–50 days. A larger GPU would cut this to days, but it would not change the prognosis.

## What P1b established

1. **Stage 1 works.** The act indicator and the Rasch fit separate models that act from models that do not (replay; smoke κ).
2. **Screened models act everywhere.** Capable, screened models draft at every interpreter step. All 36 base items are observable, against only the 24 non-drafting items under P1's population.
3. **The joint class key fragments.** Capable models' readings of a step fragment under a class key over a whole multi-row slice. Agreement is about 27–30%, and seed noise dominates paraphrase effects.
4. **Stage 2 needs redesign.** Under the pre-registered stage-2 model, θ is not identified once models write. Everything gated on θ ≥ 0.5 is then inert: S2, the ddmin oracle, and the competent-pair ambiguity.

## Open questions for the next design (not decisions)

- **Identifiability of θ.** For example, restrict ε to classes outside A_s, put a shared prior on ε, or use anchor items with a known reading.
- **Comparison granularity.** Compare readings per field or per row, instead of one joint key over a multi-row slice.
- **≡_S normalisation.** Handle surrogate ids that T does not fix, status synonyms (`draft` ≡ `Nháp`), and rows inherited from other steps.
- **Competence that does not depend on an absolute θ threshold.**
- **Population size.** More raters per family; with two raters, Proposition 2′'s bound is vacuous.

## Run cache

The P1 cache (`/home/ethan/AILab/TonomyX-paper/work/runs.jsonl`) now holds 6,505 runs:
- 6,300 P1 runs;
- 63 screen runs (schema 3);
- 142 smoke runs (schema 3).

Runs were only appended; nothing was rewritten. The 36 P1 mutant packs are unchanged.
