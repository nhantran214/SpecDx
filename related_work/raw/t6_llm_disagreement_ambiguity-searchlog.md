# Search log: t6_llm_disagreement_ambiguity

Theme: disagreement, inconsistency and uncertainty across LLM samples or models used as a signal; detection and resolution of ambiguity and underspecification by LLMs; differential testing and N-version programming (including correlated failures) as ways to expose specification ambiguity.

Dates: 2026-10-20 to 2026-10-21. The run was interrupted twice by usage limits. Verification records are kept in the scratchpad (`t6/v/*.json`).

## Verification route
- `lit.py verify` was used for title, arXiv or DOI lookup. A CrossRef-only wrapper around `lit.py` functions (`crossref_search`, `doi_bibtex`) was used when the arXiv API was congested.
- ICLR, TMLR and ICML papers were confirmed on official pages: iclr.cc virtual poster pages and proceedings.mlr.press. OpenReview returned a challenge page (HTTP 403) and could not be used.
- Abstracts came from four sources:
  - the arXiv API;
  - CrossRef `abstract` fields;
  - ACL Anthology pages;
  - Semantic Scholar's `/paper/DOI:` endpoint.
- Abstracts of old IEEE TSE and ACM papers came from WebSearch summaries of official or author pages. The ACM DL and IEEE Xplore pages returned 403 or empty responses.
- Code links were accepted only when the repository's README names the paper, or when the abstract or arXiv comment links it. The GitHub API hit its rate limit late in the run, so the NEZHA, JEST and CodeT repositories were confirmed through their raw README only, and their `code_info` is null.

## Queries (source: hits screened)
- verify (lead list), 7 sampling-uncertainty leads: self-consistency, semantic uncertainty ICLR'23, semantic entropy Nature'24, SelfCheckGPT, Generating with Confidence, verbalised confidence, multi-agent debate. 7 screened.
- verify (lead list), 7 agent and ambiguity leads: arXiv 2602.11619, 2510.07686, AmbigQA, AmbiEnt, CLAMBER, CLAM, ClarifyGPT. 7 screened.
- verify (lead list), 6 code-ambiguity leads: TiCoder TSE'24, TiCoder arXiv'22, HumanEvalComm, ImpossibleBench, "LLMs should ask clarifying questions" (MAPS'23), plus one mis-hit. 6 screened.
- verify (lead list), 7 differential-testing leads: McKeeman, Csmith, Frankencerts, JVM classfuzz, NEZHA, RFC-directed, Differential Prompting. 7 screened; PLDI DOIs re-resolved for Csmith and classfuzz.
- WebSearch "NEZHA … delta-diversity", "classfuzz", "RFC-directed …": about 28 results. This led to JEST (ICSE'21) and MLCerts "Hallucinating Certificates" (ICSE'26).
- CrossRef, N-version / coincident failures / monoculture / cross-model consistency (17 titles): Avizienis'85, Eckhardt-Lee'85, Knight-Leveson'86, Littlewood-Miller'89, Littlewood-Popov-Strigini'01, Brilliant et al.'90, Avizienis-Chen'77, algorithmic monoculture, outcome homogenization, silicon crowd, Look Before You Leap, SAC3, LM vs LM, MBR-EXEC, Sage, HDiff, B4 (ASE'24). 17 screened.
- WebSearch "ClarifyGPT github …": 9 results, which surfaced ClarifyCodeBench and "Do Code LMs Follow Tests?".
- WebSearch "Correlated Errors in LLMs ICML 2025": 9 results.
- WebSearch "SpecFix / requirement ambiguity 2025": 9 results.
- CrossRef, 13 titles: SpecFix, CodeT, Ambig-SWE, APA, IN3, Noisy ToolBench, DiffSpec, Mokav, KLE, LM-Polygraph (x2), Great Models Think Alike, McKeeman. 13 screened.
- WebSearch on official pages: Great Models Think Alike, CodeT, Ambig-SWE, ImpossibleBench, self-consistency, semantic uncertainty, verbalised confidence, Generating with Confidence. About 60 results.
- arXiv search "N-version programming large language models code diversity": 8 results (Galápagos kept).
- arXiv search "uncertainty code generation LLM symbolic execution semantic clustering correctness": 0 results.
- arXiv search "underspecification LLM prompts": 8 results (4 kept).
- arXiv search "inter-model disagreement specification ambiguity detection": 0 results.
- OpenAlex title "Differential Testing for Software": 3 results (McKeeman kept).
- CrossRef, 3 discovery queries on LLM-based requirements ambiguity detection and differential testing for specification ambiguity: 30 results. All excluded: requirements-engineering ambiguity is left to theme t3, and the rest are off-topic or from low-tier venues.

## Anchor papers used for snowballing
- ClarifyGPT (FSE'24): led to MBR-EXEC, CodeT, HumanEvalComm, SpecFix, ClarifyCodeBench and semantic collapse.
- Frankencerts and NEZHA: led to JEST, RFC-directed testing, HDiff, DiffSpec and MLCerts.
- Knight & Leveson'86 and Eckhardt & Lee'85: led to Littlewood & Miller, Littlewood-Popov-Strigini, Brilliant et al., Bommasani'22, Kim et al. ICML'25, Goel et al. ICML'25 and Galápagos.
- Semantic entropy: led to KLE, LM-Polygraph, SAC3 and "Generating with Confidence".

## Close-call exclusions
- CLAM: Selective Clarification (arXiv 2212.07769, 2022). A pre-2024 preprint with no venue in the arXiv metadata.
- Interactive Code Generation via Test-Driven User-Intent Formalization (arXiv 2208.05950). Superseded by the TSE'24 TiCoder paper, which is included.
- Large Language Models Should Ask Clarifying Questions… (MAPS'23 workshop, arXiv only). Superseded by HumanEvalComm in TOSEM'25.
- Identifying and Mitigating API Misuse in LLMs (TSE'26). Off-topic: it was a mis-hit.
- Algorithmic Monoculture and Social Welfare (Kleinberg & Raghavan). CrossRef matched only a 2023 book-chapter reprint without authors, and the PNAS original was not verified. Bommasani et al. covers the topic instead.
- Beyond Single Models: Enhancing LLM Detection of Ambiguity in Requests Through Debate (SICE FES 2025). Off-target venue and SICE domain.
- RE-venue LLM ambiguity detection papers (REW'26 RADAR, CASCON'25 multi-label ambiguity). Deferred to the requirements-NLP theme (t3).
- Detecting Hallucinations… Token Probability Approach (arXiv 2405.19648). A wrong-ID hit while looking up KLE; not relevant.
- Agnostics, AutoCodeBench, CompilerDream and the other N-version arXiv search hits. Not about diversity or correlated failures.

## Unverified
- Avizienis & Chen (1977), "On the implementation of N-version programming for software fault tolerance during execution" (COMPSAC). There was no CrossRef match (best similarity 0.56), so it was dropped.
- The ClarifyGPT official code repository was not found, so `code_url` is null.

## Notes on venue status
- Ambig-SWE (arXiv 2502.13069): the arXiv comment says "Accepted at ICLR 2026". No official page was retrieved, so it is recorded as a preprint.
- "When Agents Disagree With Themselves" (arXiv 2602.11619) is an ICML 2026 workshop paper and is recorded as a preprint.
- "Generating with Confidence" is TMLR 05/2024. It was confirmed via its iclr.cc 2025 poster page, with OpenReview id DWkJCSxKU5.
- McKeeman (1998) appeared in Digital Technical Journal 10(1). It was verified via OpenAlex only, has no DOI, and has `read_level` metadata only.

## Totals
- Screened: about 230 candidates, counting search hits examined plus 75 explicit verifications.
- Included: 60 records.
  - Status: 49 peer-reviewed, 10 preprints (2024–2026), 1 other (DTJ 1998).
  - Relevance: 24 high, 29 medium, 7 low.
  - 13 are baseline candidates.
