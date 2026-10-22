# Search log: t5_diagnosis_mus_dd

Theme: consistency/model-based diagnosis; MUS/MCS enumeration and hitting-set duality; diagnosis of inconsistent constraint, knowledge and configuration bases; spectrum-based (and Bayesian) fault localisation; delta debugging and input reduction. Special focus: diagnosis with noisy or uncertain observations.

Dates: 2026-10-20 to 2026-10-21. Tool: `related_work/tools/lit.py` (CrossRef, arXiv, GitHub, OpenAlex once), WebSearch and WebFetch for discovery, abstracts and venue pages. arXiv's export API was heavily rate-limited (HTTP 429) because several agents shared it, so most arXiv look-ups were replaced by CrossRef, by DOI, or by reading `arxiv.org/abs/<id>` pages.

## Queries

### WebSearch (discovery and abstracts), about 10 hits screened each

| # | Query (abridged) | Useful finds |
|---|---|---|
| 1 | model-based diagnosis uncertain / noisy observations consistency-based | Cazes & Kalech AAAI 2020 |
| 2 | intermittent faults, spectrum/MBD, Barinel, noisy test outcomes | Barinel ASE 2009, AIJ 2010 |
| 3 | minimal hitting set, noisy/soft/probabilistic conflict sets | Flesch et al. IJCAI 2007 |
| 4 | Cazes Kalech AAAI 2020 (venue check) | AAAI 34(3) |
| 5 | ProbDD follow-ups 2023–2025 | CDD (ICSE 2025), WDD (ICSE 2025) |
| 6 | LLM-aided test-input / program reduction | LPR (ISSTA 2024), ReduceFix (arXiv 2025), GReduce |
| 7 | delta debugging, nondeterministic or flaky oracle | Delta debugging for CPS with flaky executions (arXiv 2026) |
| 8 | delta debugging applied to LLM prompts | DDOR (arXiv 2026), Delta Debugging for LLM-integrated Systems (ICSE-SEIP 2026) |
| 9 | interactive ontology debugging, faulty oracle (Rodler, Shchekotykhin) | J. Web Semantics 2012 |
| 10 | Alloy fault localisation (AlloyFL, FLACK) | FLACK (ICSE 2021) |
| 11 | SFL for LLM multi-agent systems / failure attribution | FAMAS (PACMSE 2026), Who&When (ICML 2025) |
| 12 | SFL for model transformations, spreadsheets, decision tables | Troya et al. TOSEM 2018; Hofer et al. ASE journal |
| 13 | flaky tests and SFL | Vancsics VST 2020; Qin SANER 2021; SFFL AST 2023 |
| 14 | MUS for requirements inconsistency | (generic MUS results) |
| 15 | unsat cores for UML/OCL | Shaikh & Wiil, SPE |
| 16 | LLM + SMT for requirements inconsistency | SAT-LLM (ASE NIER 2024); string-requirements satisfiability (RE 2025) |
| 17 | Staccato approximate MHS | SARA 2009; Gainer-Dewar & Vera-Licona |
| 18 | de Kleer intermittent faults, IJCAI 2009 | IJCAI 2009 p. 733 |
| 19 | test-case reduction survey | Kreutzer et al. ICSME 2021 |
| 20 | BugAssist | PLDI 2011 |
| 21–23 | Barinel code; PyMBD; ProbDD code | Zoltar (archived); ProbDD replication repo |
| 24 | QuickXplain AAAI 2004 | aaai.org page and PDF |
| 25 | Felfernig "Plausible repairs", IJCAI 2009 | IJCAI 2009 p. 791 |
| 26 | Delta Debugging for LLM-integrated Systems (authors) | ICSE-SEIP 2026 |
| 27 | MBD with MaxSAT, IJCAI 2015 | IJCAI 2015 p. 1966 |
| 28 | MCS IJCAI 2013 page | ijcai.org/Abstract/13/098 |
| 29 | SARA 2009 page (aaai.org) | also: Jannach et al. AAAI 2015 |
| 30 | LLM repeated runs locating defects in task specs, MUS/MaxSAT (threat search) | BenchGuard, Task Verification Bench, ABA; all outside this theme (see below) |
| 31 | MBD + LLM, knowledge-base debugging 2024–2025 (threat search) | nothing in theme |
| 32–60 | abstract look-ups, one per paper (Reiter, GDE, CAMUS, MARCO, Bacchus, PySAT, Torlak, Kalyanpur, Abreu ×3, DDU, Zeller TSE, HDD, Perses, SmartFL, LDP, MSMP, Shlyakhter, Greiner, de Kleer 1990/1992, SMUS, MaxHS, Critical observations, Tarantula, Hofer, Lamperti, DynamicHS, metamorphic slice, Coarse HDD, Cazes IJIS, Kreutzer, Bailey & Stuckey, Marques-Silva ISMVL, Who&When PMLR) | abstracts |

### CrossRef (`lit.py search`), 4–10 hits screened each

- model-based diagnosis uncertain observations (10)
- consistency-based diagnosis noisy observations probabilistic (10)
- diagnosis sensor faults model-based unreliable observations (10)
- spectrum-based fault localization noisy test oracle incorrect test labels (10)
- minimal hitting sets diagnosis probabilistic conflicts (10)
- crowdsourcing ontology verification debugging (8)
- spectrum-based fault localization SQL queries database applications (8)
- diagnosis inconsistent requirements specification minimal conflict sets (8)
- delta debugging natural language requirements specification (8)
- hitting set noisy observations diagnosis probabilistic MaxSAT soft (8)
- 11 title look-ups for exact records: Abreu JSS, Abreu TAICPART, ProbDD, Plausible repairs, C-Reduce PLDI, DD for LLM systems, GReduce, Kreutzer, FRACTAL, Coarse HDD, Rodler AAAI 2022 (4–5 each)

### arXiv (`lit.py arxiv-search`)

- "delta debugging LLM prompt", "minimal unsatisfiable subset large language model" and "spectrum-based fault localization LLM agents" all timed out (HTTP 429).
- "model-based diagnosis large language models": 10 hits, all off-topic (medical diagnosis).
- arXiv abstract pages were read directly for 2604.14525, 1707.01972 and 1902.06002.

### Verification (`lit.py verify` / CrossRef DOI / proceedings pages)

- 102 candidate records were verified or attempted through the scripted pipeline.
- 9 more were verified through official proceedings pages: IJCAI ×6, AAAI 2004, PMLR ICML 2025, plus OpenAlex for SARA 2009.

## Anchor papers used for snowballing

- Backward, with `lit.py refs`:
  - Zhang et al., "Toward a Better Understanding of Probabilistic Delta Debugging" (ICSE 2025). This surfaced Vulcan (PACMPL 2023), T-Rec (TOSEM 2025), "Avoiding the familiar" (QRS 2018), Picire and Picireny. Picire's paper (ICSOFT 2016) was included.
  - FAMAS (PACMSE 2026). This surfaced Who&When, AgenTracer and others; Who&When was included.
  - Cazes & Kalech (AAAI 2020) and Ignatiev et al. (IJCAI 2019) returned no CrossRef reference lists.
- Forward and co-citation, through WebSearch: Reiter 1987 (to Greiner 1989, de Kleer 1992, Grastien's "A More General Theory of Diagnosis", Gainer-Dewar 2017), Liffiton & Sakallah 2008 (to MARCO, Bailey & Stuckey 2005, ReMUS, MUST), Barinel ASE 2009 (to AIJ 2010, Staccato, Zoltar), ddmin TSE 2002 (to HDD, Perses, ProbDD, CDD, WDD, PMA), and Felfernig AIJ 2004 (to FastDiag, Plausible repairs).

## Close calls (excluded)

| Paper | Reason for excluding |
|---|---|
| Hunter & Konieczny, "Measuring inconsistency through minimal inconsistent sets" (KR 2008) | Not found on CrossRef. Inconsistency measures are tangential to SpecDx. |
| Mu et al., "A general framework for measuring inconsistency through MIS" (KAIS 2010) | Tangential, as above. |
| Mortensen, "Crowdsourcing Ontology Verification" (ISWC 2013, doctoral consortium) | Superseded by the included JAMIA 2015 paper. |
| Zeller, "Isolating cause-effect chains" (FSE 2002) | Verified, but about program states, not input reduction. |
| Ignatiev et al., arXiv 1707.01972 (MBD of multiple observations with implicit hitting sets) | Pre-2024 preprint, superseded by IJCAI 2019. |
| Salla et al., "Quantifying Cross-Query Contradictions in Multi-Query LLM Reasoning" (arXiv 2604.14525, ICLR 2026 workshop) | Repairs LLM answer consistency with a solver, not specification diagnosis; workshop paper; low relevance. |
| Rodler, "Don't Treat the Symptom, Find the Cause!" (arXiv 2306.12850) | 2023 preprint (outside the 2024–2026 preprint window). |
| Vulcan (PACMPL 2023); T-Rec (TOSEM 2025); "Avoiding the familiar" (QRS 2018); "Modernizing HDD" (A-TEST 2016); "Extending HDD with hoisting" (AST 2021) | Program-reduction engineering; we already include enough reducer prior art. |
| Program Reduction: A Comprehensive Survey (Sun group, 2026) | Preprint page only; no verifiable record found. |
| LLMAO / AutoFL and other LLM-based fault localisation; SATQuest; CoreForge | LLMs as code fault localisers or SAT solvers; not related to specification diagnosis. |
| "Tackling the oracle problem in SBFL" (book chapter, 2020) | Covered by the included IST 2013 paper. |
| Inconsistency measurement in LTLf (2024); TreeMerge and LinearMerge MHS; "Compression with wildcards" | Peripheral hitting-set papers. |
| Felfernig, "Utility-based repair of inconsistent requirements" (2009) | Same line of work as the included IJCAI 2009 paper. |
| Out of theme, but they **may threaten SpecDx** and should be checked by the benchmark-auditing theme: BenchGuard (arXiv 2602.12665?), "Can LLMs Detect Benchmark Defects? A Meta-Benchmark from Benchmark Updates" (ICML 2026), Auto Benchmark Audit (arXiv 2607.14818?) | Seen only in WebSearch results; **not verified by me**. They use LLM auditing of task specifications (contradictions, underspecification), with no population statistics or MUS/refutation. |

## Unverified (dropped)

- Hofer et al., "Spectrum-based fault localization in spreadsheets" (2013): no CrossRef match.
- Shchekotykhin et al., "Direct computation of diagnoses for ontology debugging" (arXiv 1209.0997): no peer-reviewed record matched.
- Grastien et al., "A More General Theory of Diagnosis from First Principles" (arXiv 2309.16180): no CrossRef match; a pre-2024 preprint.
- Hunter & Konieczny (KR 2008): see the close calls above.

## Notes on records

- **Corrected years and venues:**
  - Stern et al.: CrossRef gives 2021 (an OJS redeposit), recorded as AAAI-12 (2012).
  - Liffiton & Sakallah: CrossRef gives the 2007 online date, recorded as JAR 2008.
  - MARCO: recorded as Constraints 2016.
  - BugAssist and C-Reduce: the PLDI proceedings DOIs were used, not the SIGPLAN Notices DOIs.
- **Records summarised from metadata only (6):** de Kleer 1990, Ignatiev et al. CP 2015, Bailey & Stuckey PADL 2005, Marques-Silva ISMVL 2010, Hodován et al. ICSME 2017, Xie et al. IST 2013. Each summary says so.
- **Abstracts from WebSearch:** for Elsevier, Springer and IEEE records whose abstracts the APIs elide, the abstracts were read through WebSearch results quoting them.

## Totals

- **Screened:** about 600 search hits (about 60 WebSearch queries, about 21 CrossRef queries, 4 arXiv queries, 2 reference lists), plus 113 candidate records sent to verification.
- **Included:** 99 records in `t5_diagnosis_mus_dd.jsonl`, each verified by CrossRef, arXiv, an official proceedings page, or (once) OpenAlex.
  - Relevance: 25 high, 57 medium, 17 low.
  - Status: 3 preprints (2025–2026).
  - Baseline candidates: 5.
