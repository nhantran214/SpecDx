# Baselines for SpecDx

These baselines were selected on 2026-10-21 from the verified survey in [`../related_work/`](../related_work/README.md), which has 587 papers in 9 themes. Each folder holds a `NOTES.md` covering:
- the paper(s), with keys in `related_work/references.bib`;
- the code, with the exact commit and licence;
- what the method does and which SpecDx defect types it can address;
- the input it needs, the concrete adaptation to S1 (TonomyX) and S2 (τ²-bench), model requirements, effort and caveats.

The `code*/` folders are shallow clones of third-party repositories. They are not part of SpecDx; see **Licences** below before reusing anything.

## Selection criteria

1. **Close to SpecDx.** The method either attacks the same problem (automatic detection of specification, benchmark or ground-truth defects, or separating agent errors from task defects) or replaces one SpecDx component on the same input.
2. **Released code preferred.** 14 of the 15 baselines have a repository. The JEST rule is reimplemented in a few lines because its code is specific to ECMAScript.
3. **Reputable venue preferred.** Most are at CORE A* or A venues: ICML, FSE, ASE, ICSE, ACL, JAIR, Nature, Constraints, SAT. Four preprints are kept because they are direct competitors or the only released code for a component: BenchGuard (reported accepted at COLM 2026), BenchJack and the Land & Bikel IRT audit (reported accepted at EMNLP 2026), all from 2026, and IntellAgent from 2025.
4. **Runnable under SpecDx's constraints** (local models, no human labels), either directly or with a documented port.

## The 15 baselines

| # | Baseline | Venue | Code / licence | SpecDx slot | Defect types |
|---|---|---|---|---|---|
| **A. Direct competitors (whole method)** | | | | | |
| 01 | [ClarifyGPT](01_clarifygpt/NOTES.md): ambiguity when sampled executions disagree | FSE 2024 (A*) | yes / none | Ambiguity; single-family ablation; B4 | Ambiguity |
| 02 | [SpecFix](02_specfix/NOTES.md): interpretation distribution anchored to examples | ASE 2025 (A*) | yes / none | Ambiguity + hidden requirement | Ambiguity, Hidden (partial) |
| 03 | [AgentSuite / COBA](03_agentsuite_coba/NOTES.md): component-based benchmark audit; finds τ²-bench issues | ICML 2026 (A*) | yes / none | B1 + B5 (published) | Hidden, Weak, Ambiguity |
| 04 | [BenchGuard](04_benchguard/NOTES.md): LLM cross-verification of instruction, grader and solution | COLM 2026 (arXiv 2604.24955) | yes / Apache-2.0 | B1 (published) | all four |
| 05 | [Who&When](05_who_and_when/NOTES.md): LLM-judge failure attribution | ICML 2025 (A*) | yes / MIT | Agent error vs specification defect | Separation, Hidden |
| 07 | [BenchJack](07_benchjack/NOTES.md): agents synthesise exploits of benchmark checkers | arXiv 2605.12673 | yes / Apache-2.0 | Weak check with witnesses | Weak |
| **B. Statistical aggregation on the reading matrix** | | | | | |
| 06 | [IRT audit](06_irt_audit/NOTES.md): py-irt 2PL/4PL and the Land & Bikel forced-ceiling indicator | ACL 2021 (A*), IJOC 2023, EMNLP 2026 (arXiv 2605.30504) | py-irt / MIT | **B3** (revise, see below) | Hidden / mis-key |
| 08 | [Crowd-Kit](08_crowdkit/NOTES.md): MV, Dawid–Skene, GLAD, MACE, KOS, M-MSR | JOSS 2024; DS 1979, GLAD NeurIPS 2009, MACE NAACL 2013 | yes / Apache-2.0 | Simulation baselines; "no set truth, no anchor" ablation | Hidden / mis-key |
| 09 | [cleanlab](09_cleanlab/NOTES.md): Confident Learning and CROWDLAB (specification as one annotator) | JAIR 2021 (A) | yes / Apache-2.0 | Mis-key detector | Hidden / mis-key |
| 10 | [Semantic entropy](10_semantic_entropy/NOTES.md) | Nature 2024; ICLR 2023 (A*) | yes / BSD-3-Clear | **B4** (published) | Ambiguity |
| 15 | [JEST rule](15_jest_spec_vote/NOTES.md): N+1-version testing, where a majority deviation means a specification bug | ICSE 2021 (A*) | reimplement | Named form of majority vote / anchor-only | Hidden; separation |
| **C. Component baselines and ablations** | | | | | |
| 11 | [PySAT + MARCO](11_pysat_marco/NOTES.md): MUS/MCS enumeration on C alone | SAT 2018 (A), Constraints 2016 (A) | yes / MIT, custom | **B5**; hard vs soft hitting sets | Contradiction |
| 12 | [ProbDD](12_probdd/NOTES.md): probabilistic delta debugging | ESEC/FSE 2021 (A*) | yes / GPL-3.0 | Replaces ddmin (ambiguity localisation) | Localisation in T |
| 13 | [Active testing](13_active_testing/NOTES.md): acquisition for efficient evaluation | ICML 2021 (A*) | yes / MIT | Replaces Thompson-EIG; compare with A4 | Efficiency |
| 14 | [IntellAgent](14_intellagent/NOTES.md): policy-graph scenario generation | arXiv 2501.11067 | yes / Apache-2.0 | Replaces scenario synthesis (S2) | Scenario source |

## Recommended core set for the paper

The evaluation has five rival detectors (B1–B5) and the simulation baselines of v3 §8.1. A compact, defensible comparison is:

| Slot | Use | Reason |
|---|---|---|
| **B1** static LLM review | **BenchGuard** (04), with **COBA** (03) on S2 | Published, automatic auditors of agent benchmarks. COBA is peer-reviewed (ICML 2026) and was evaluated on τ²-bench, SpecDx's own subject S2. |
| **B2** best single interpreter | Keep, and add **Who&When** (05) | Gives B2 a published per-trace attribution judge, not only "flag every violation". |
| **B3** IRT | **Land & Bikel forced-ceiling indicator** on py-irt (06), with plain 2PL as a secondary arm | The published IRT auditor reports that "negative discrimination" (the current B3) is the weaker variant. Keeping only that variant would invite a reviewer objection. |
| **B4** disagreement | **Semantic entropy** (10) | A citable estimator instead of an ad hoc entropy. |
| **B5** check-only | **PySAT/MARCO** (11) | Standard MUS/MCS tooling. |
| Simulation baselines | **Crowd-Kit** DS/GLAD/MACE (08), **cleanlab** CL/CROWDLAB (09), **JEST rule** (15) | Shows what set-valued truth, anchors and the restricted error model add over classic truth inference. |
| Direct-competitor arms (ambiguity) | **ClarifyGPT** (01) and **SpecFix** (02) | The two peer-reviewed execution-disagreement ambiguity detectors. Both use one model and no rater model. |
| Component ablations | ProbDD (12), active testing (13), IntellAgent (14), BenchJack (07, weak checks) | Each replaces one SpecDx component at an equal run budget. |

## Methods without usable code (reimplement or cite)

These are in `related_work/catalog.md` with full notes:
- **VERIMED** (Hall et al., arXiv 2026: neurosymbolic auditing of NL requirements; LLM formalisation disagreement plus SMT). The closest "statistics plus logic" competitor on the T side. No code.
- **FAMAS** (PACMSE 2026: spectrum-based failure attribution over repeated LLM-agent runs) and **Barinel** (ASE 2009: Bayesian SFL with hitting sets). Both run on SpecDx's run × clause matrix; Zoltar is archived, so reimplement.
- **Ambiguity Detection and Elimination in Executable Process Modeling** (arXiv 2604.10884): dispersion of simulated KPIs over repeated BPMN generations. The closest work in the BPM community. No code.
- **Policy Loopholes in Agent Evaluation** (arXiv 2609.14400) and the **τ²-Bench-Verified** repairs (SABER, arXiv 2512.07850; github.com/amazon-agi/tau2-bench-verified, MIT). These are sources of real labelled defects, not runnable baselines.
- **Auto Benchmark Audit** (arXiv 2605.26079; github.com/IsThatYou/auto-bench-audit, no licence). An alternative to BenchGuard as B1. Not cloned (296 MB, mostly HTML reports).

## Practical notes

- **Local models.** BenchGuard, BenchJack and COBA default to frontier models through Claude Code, Codex or OpenAI APIs. SpecDx's constraint is local models only (v1 §1.2), so run them through their OpenAI-compatible or litellm clients pointed at Ollama (`http://localhost:11434/v1`), and state the quality gap in the threats to validity. ClarifyGPT, SpecFix and Who&When also use OpenAI clients and can be redirected the same way. IntellAgent has an Ollama provider.
- **Offline statistics.** The statistical baselines (06, 08, 09, 10, 15) need no LLM calls. They run on the cached readings (`work/runs.jsonl` and the cell extraction of `specdx/infer/cells.py`), so they add no GPU cost.
- **Licences.**
  - ClarifyGPT, SpecFix and AgentSuite have **no licence file**: use them for local research only, and do not redistribute or vendor their code.
  - ProbDD is **GPL-3.0**: reimplement it rather than copy code into SpecDx.
  - The others are MIT, Apache-2.0, BSD or custom permissive.
- **Size.** About 3.2 GB in total, of which AgentSuite is 2.3 GB (it vendors τ-bench and τ²-bench plus pipeline data). These clones should not be committed. Add `baseline/*/code*/` to `.gitignore` if the folder is put under version control.
- **Reproducibility.** Every clone is pinned to the commit recorded in its `NOTES.md`. Re-create one with `git clone <repo> && git checkout <commit>`.
