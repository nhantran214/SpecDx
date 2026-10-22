# SpecDx: related work survey

This is a verified literature base for the SpecDx paper (target venue: ACM TOSEM). It was built on 2026-10-20/21. **612 papers** around the method of `docs/superpowers/specs/2026-10-14-specdx-v3-design.md` (anchored cell model, set-valued truth, MUS/MCS with refutation, ddmin over T, active scenario synthesis, mutation-based evaluation).

## Files

| Path | What it is |
|---|---|
| [`catalog.md`](catalog.md) | The catalogue, grouped by theme, then relevance and year. Each entry gives venue, tier, status, link, local PDF, code, summary and its relation to SpecDx |
| [`catalog.csv`](catalog.csv) | The same as one row per paper (filter by `relevance`, `role`, `specdx_components`, `baseline_candidate`, `tier`) |
| [`references.bib`](references.bib) | BibTeX for every paper; keys match the catalogue (e.g. `mu2024clarifygpt`) |
| `pdfs/<theme>/<key>.pdf` | 273 open-access PDFs (arXiv, ACL Anthology, PMLR, NeurIPS, USENIX, author copies). Paywalled papers have links only |
| `raw/<theme>.jsonl` | The agents' verified records (source of truth; edit here, then rebuild) |
| `raw/<theme>-searchlog.md` | Queries, snowball anchors, close-call exclusions and unverified items per theme |
| `raw/BRIEF.md` | The search protocol given to each theme agent |
| `tools/lit.py` | Lookup and verification (CrossRef/doi.org, arXiv, GitHub, OpenAlex as last resort) |
| `tools/build.py` | Merges, deduplicates, assigns keys, looks up venue tiers on the CORE portal, downloads PDFs: `python3 tools/build.py [--tiers] [--pdfs]` |

## Protocol

- **Scope.** Nine themes, each searched by one agent following [`raw/BRIEF.md`](raw/BRIEF.md):

  | Theme | Screened (approx.) | Included (raw) | Primary theme after dedup |
  |---|---:|---:|---:|
  | t1 Truth inference, label aggregation, latent-class models | 600 | 93 | 87 |
  | t2 IRT, benchmark auditing, label errors, benchmark repair | 290 | 53 | 47 |
  | t3 Requirements quality (NLP4RE/LLM4RE), protocol specs | 815 hits / 190 candidates | 87 | 84 |
  | t4 Specification and oracle defects, specification mutation | 190 | 62 | 60 |
  | t5 Model-based diagnosis, MUS/MCS, SFL, delta debugging | 600 | 99 | 92 |
  | t6 LLM disagreement and ambiguity, differential and N-version testing | 230 | 60 | 49 |
  | t7 Active testing, Bayesian experimental design, bandits, scenario generation | 160 | 64 | 60 |
  | t8 Business-process agents and policies, conformance, DMN/Declare verification | 560 | 85 | 81 |
  | t9 Agent failure attribution, evaluation reliability, specification gaming | 560 | 59 | 52 |
  | **Total** | **≈ 4,000 hits** | **662** | **612 unique** |

- **Sources.** CrossRef; arXiv; WebSearch; official proceedings indexes (NeurIPS, PMLR, ICLR, ACL Anthology, USENIX); backward snowballing through CrossRef reference lists; forward snowballing through web search. DBLP blocks scripted access and Semantic Scholar was rate-limited, so neither was used.
- **Verification (no fabricated entries).** Every record was matched by the tool to one of:
  - a CrossRef DOI record (BibTeX from doi.org);
  - an arXiv record;
  - an official proceedings page (OpenAlex was used only as a last resort).

  Unverifiable leads were dropped and listed in the search logs. `read_level` records whether the summary comes from the abstract, the full text or metadata only. About 40 records are metadata only, mostly old papers or publisher-elided abstracts.
- **Venue quality.** Tiers come from the CORE portal: CORE2023 for conferences, CORE2020 for journals. Tracks (NIER, SEIP, D&B, Industry) are labelled with their parent's rank, and journal-format proceedings (PACMSE, PVLDB, Proc. AAAI) take the rank of their conference. Statistics and general-science journals (Annals of Statistics, Biometrika, Psychometrika, JRSS, Nature, PNAS, Science Advances) are outside CORE and are shown as "not in CORE journal list", not as unranked.

  | Status or tier | Papers |
  |---|---:|
  | Peer-reviewed | 511 |
  | Preprints (arXiv; most from 2025–2026, kept only when directly relevant) | 82 |
  | Books, chapters, other | 19 |
  | CORE A* | 310 |
  | CORE A* or A | 381 |

- **Relevance.** 202 high, 302 medium, 108 low; 23 papers are marked `direct competitor`.

## What the survey says about SpecDx's novelty

**The claim that survives.** No paper found combines:
1. statistics over a *population* of LLM executors (a rater model with accuracy and difficulty, set-valued truth, the specification as an anchored fallible rater);
2. *logic* (MUS/MCS over sampled correction sets, refutation with witness states);

to produce a **typed** (contradiction, hidden requirement, ambiguity, weak check), **localised** (cell, clause, sentence of T) diagnosis of an executable specification, fully automatically.

**The claim that no longer holds alone.** "Automatic detection of benchmark or specification defects" is crowded in 2026. Each half of SpecDx has close precedents:

| Closest work | What it shares with SpecDx | What SpecDx adds |
|---|---|---|
| **JEST** `park2021jest` (ICSE 2021) | The specification as one more version: a majority deviation means a specification bug | Stochastic interpreters of uneven competence (accuracy/difficulty model), set-valued truth, typing, MUS witnesses |
| **ClarifyGPT** `mu2024clarifygpt` (FSE 2024), **SpecFix** `jia2025automated` (ASE 2025) | Ambiguity from disagreement among sampled executions; SpecFix anchors to examples | Multi-family population with a rater model; C can be wrong (SpecFix never blames the examples); four types; localisation in T |
| **VERIMED** `hall2026neurosymbolic` (arXiv 2026) | LLM-sample disagreement plus SMT, with counterexamples | Samples executions rather than formalisations; anchored C; separates interpreter error from specification defect |
| **Land & Bikel** `land2026auditing` (EMNLP 2026) | A population of models plus IRT finds mis-keyed and ambiguous items | Set-valued truth, executable checks, contradiction logic, witnesses. **This is the published form of B3** |
| **COBA/AgentSuite** `suh2026agentsuite` (ICML 2026), **BenchGuard** `tu2026who`, **ABA** `wang2026automated`, **ELT-Bench-Verified** `zanoli2026elt` | Automatic auditing of agent benchmarks, including τ²-bench | They read the artifacts with one auditor and do not execute them with a population; no statistical separation from agent error; no MUS certificates |
| **AutoTriage** `hu2026verifying` (ICLR-W 2026), **FAMAS** `ge2026spectrum` (FSE 2026), **Terminal-Bench fake hardness** `lip2026makes`, **transcript scanners** `mohl2026automated` | Agent error vs task defect; statistics over repeated runs (FAMAS) | Clause-level typed localisation; FAMAS trusts the task and checker |
| **BenchJack** `wang2026androids` | Automatic weak-check witnesses (exploits) | Honest-executor statistics; the other three types |
| **Ambiguity detection in executable process modelling** `matei2026ambiguity`, **MANTRA** `anand2026mantra`, **generalized conformance checking** `solti2016log` (BPM 2016) | Repeated generation or simulation to expose ambiguity; SMT-validated checks; trusting neither log nor model | Multi-family population, anchors, four types, witnesses on an existing specification |
| **Stress-testing model specs** `zhang2025stress`, **Experts Rise Where LLMs Disagree** `he2026experts`, **Observational equivalence** `nakamura2026observational`, **Policy Loopholes** `cao2026policy`, **Nahum et al.** `nahum2025llms` (EMNLP 2025) | Cross-model disagreement localises ambiguity or label errors in a specification, codebook or benchmark | Execution against executable checks; automatic typing; Policy Loopholes is a manual audit of the same τ²-bench domains (useful as labelled real defects) |

**Recommendation for the paper.** Frame the contribution as the *combination*: a rater model for LLM executors with the specification as an anchored rater, plus MUS/MCS and refutation, giving typed, witnessed diagnoses. Do not frame it as "automatic defect detection". Cite JEST, ClarifyGPT/SpecFix, VERIMED, Land & Bikel, COBA and BenchGuard explicitly in the introduction.

## Findings that bear on the v3 design

1. **Revise B3.** Land & Bikel report that the plain 2PL "negative/low discrimination" indicator, which is the current B3, is weaker than their forced-ceiling likelihood ratio from a constrained 4PL. Use theirs as B3 and keep 2PL as a secondary arm (see `../baseline/06_irt_audit/NOTES.md`).
2. **Assumption (A3) and family independence.** "Correlated errors in LLMs" `kim2025correlated` (ICML 2025) and "Great models think alike" `goel2025great` (ICML 2025, with the CAPA metric) show that errors are correlated across models. The classical counterparts are Eckhardt & Lee (TSE 1985) and Knight & Leveson (TSE 1986). Measure within-family versus cross-family error correlation on SpecDx's runs with CAPA. Eckhardt–Lee's difficulty-driven coincident failure is the classical analogue of the β_g term.
3. **Repeated runs of one model are correlated.** The beta-binomial latent-class result for LLM raters (arXiv 2510.23874) bears on SpecDx's 3 variants × 3 seeds per model. It supports pooling within a model rather than treating its runs as independent raters.
4. **Semantic collapse supports the anchor.** "Underspecification does not imply incoherence" (arXiv 2607.01953) shows that models often agree on one wrong reading, so disagreement-only detectors (B4, ClarifyGPT) miss it. This is a direct argument for the anchored model. Separately, Deng & Tang report AUC 0.47 for raw ensemble disagreement on contested IFRS rules (NeurIPS 2026 workshop; OpenReview could not be verified, so not catalogued).
5. **Set-valued truth has precedents to cite.**
   - Latent Truth Model `zhao2012bayesian` (PVLDB 2012);
   - multi-truth discovery (CIKM 2015; PVLDB 2018);
   - "truth may not exist" (KDD 2015), which corresponds to A_c = ∅;
   - Jeong & Chung, top-two answers (ICML 2023), which corresponds to A_c = {x, y};
   - Batchelder & Romney's cultural consensus theory (Psychometrika 1988) and Bachrach et al. (ICML 2012), which jointly infer the answer key, item difficulty and rater ability.
6. **Scenario selection.** SpecDx's goal is identification, not regret, so top-two Thompson sampling (Russo, Operations Research 2020) or information-directed sampling (Operations Research 2018) are principled replacements for plain Thompson sampling and are cheap ablations. "Trust or Escalate" (ICLR 2025) is a direct precedent for the pilot-then-committee escalation.
7. **Noisy-observation diagnosis.** Prior art for "statistics + hitting sets":
   - Barinel `abreu2009spectrum` (ASE 2009) and Abreu & van Gemund (AIJ 2010);
   - Cazes & Kalech, model-based diagnosis with uncertain observations (AAAI 2020);
   - Flesch et al. (IJCAI 2007).

   These assume known error rates or trust the specification; SpecDx estimates interpreter accuracy.
8. **Real-defect sources for evaluation.**
   - τ²-Bench-Verified: SABER `cuadron2025saber` (arXiv 2512.07850); its `FIXES.md` is the repair diff.
   - Policy Loopholes (τ²-bench, manual).
   - ABC (NeurIPS 2025), which documents τ-bench's weak checks.
   - "Benchmarking the Benchmarks" (arXiv 2607.02577).
   - ImpossibleBench (ICLR 2026), whose tasks pit the specification against the tests.

## Limitations of this survey

- **Coverage limits.**
  - About 15 papers per theme were verified but left out under the agents' caps; they are listed in the search logs.
  - Recall on 2026 arXiv work is bounded by the date of the search (2026-10-21) and by arXiv rate limits during the run.
  - Some verified leads were listed in the search logs only. Examples: Molina & Gorla (TOSEM 2025), Heitmeyer et al.'s SCR consistency checking (TOSEM 1996), SWE-Bench Pro Verified.
- **Verification gaps.**
  - OpenReview served a bot challenge, so ICLR and TMLR papers were confirmed on iclr.cc and proceedings pages instead.
  - A few code links (NEZHA, JEST, CodeT, fluid-benchmarking, IntellAgent in t7) were confirmed through README files after the GitHub API rate limit, so their `code_info` may be empty.
- **Tier labels.** These are CORE ranks as of the 2023/2020 lists. They are a guide, not a quality judgment of individual papers.
- **PDFs.** 26 open-access links failed (mostly publisher links that serve HTML or need a session). Paywalled papers are not downloaded.

## Rebuilding

```bash
cd related_work
python3 tools/build.py --tiers --pdfs   # after editing raw/*.jsonl; cached lookups make reruns fast
```
