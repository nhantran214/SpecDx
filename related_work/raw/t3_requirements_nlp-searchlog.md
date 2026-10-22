# Search log: t3_requirements_nlp (requirements/specification text-quality defects, NLP4RE/LLM4RE, protocol specs, doc-code inconsistency)

Agent run: 2026-10-20 to 2026-10-21. Tools: `lit.py` (CrossRef search/verify, arXiv verify, refs, github), Semantic Scholar abstracts by DOI/title (scratch helper), WebSearch/WebFetch for discovery, arXiv abstract pages, USENIX/NeurIPS/ACL/ICLR proceedings pages.

## Queries (source; hits screened)

### CrossRef (`lit.py search`)
| Query | Hits |
|---|---|
| From contract drafting to software specification: linguistic sources of ambiguity Berry Kamsties | 5 |
| Identifying nocuous ambiguities in natural language requirements Chantree | 5 |
| Automated handling of anaphoric ambiguity in requirements TAPHSIR | 5 |
| terminological ambiguity user stories Dalpiaz | 6 |
| Rapid quality assurance with Requirements Smells | 3 |
| requirements defects NLP patterns railway domain Ferrari | 6 |
| Automated checking of conformance to requirements templates NLP Arora | 6 |
| Quality User Story framework and tool AQUSA | 6 |
| Automated smell detection and recommendation in natural language requirements | 6 |
| Improving requirements completeness automated assistance through LLMs Luitel (2019+) | 8 |
| AI-based question answering assistance for analyzing NL requirements Ezzini (2019+) | 8 |
| Using domain-specific corpora for improved handling of ambiguity (2019+) | 8 |
| requirements ambiguity detection large language model (2019+) | 8 |
| requirements inconsistency detection large language models (2019+) | 8 |
| conflict detection natural language requirements (2019+) | 8 |
| FRET generation of formal requirements; Hanfor scalable analysis; SpeAR v2.0; ASSERT; nl2spec; FRET realizability | 6 x 6 |
| NLP4RE mapping study; LLM4RE SLR; LLM guidelines for RE; "Requirements are all you need"; NLP4RE tools | 5 x 8 |
| Sage; RFCNLP; Bookworm game; CellularLint; Hermes; 3GPP inconsistency LLM | 6 x 5 |
| ALICE; Gärtner contradiction; SAT-LLM; Gervasi & Zowghi; Hunter & Nuseibeh; van Lamsweerde; van der Aa text-model; PolicyLint | 8 x 4 |
| Requirements ambiguity detection and explanation with LLMs industrial | 4 |
| Manolios/ASSERT complete and conflict-free; FRET realizability; LLM formalization FRET; LLM consistency checking requirements (2017+) | 4 x 5 |
| RFCScope (2024+) | 3 |
| @tComment; iComment; deep JIT inconsistency; API directive defects; DocTer; Jdoctor; LLM code-comment inconsistency; code-comment inconsistency empirical | 8 x 4 |
| ClarifyGPT; HumanEvalComm; TiCoder; requirements quality theory; requirements quality mapping; Arora LLM4RE chapter | 6 x 4 |
| SpecFix; semantic collapse; Orchid; When Prompts Go Wrong (2025+) | 4 x 3 |
| Validating formal specifications with LLM-generated test cases (2025+) | 4 |
| ARM (Wilson et al.); QuARS; Lubos LLM QA ISO 29148; smells in prompts; conflict detection sentence embeddings; Vietnamese requirements NLP | 6 x 4 |
| Malik transfer learning conflicts; Helmeczi prompt-based conflicts; Guo finer semantic conflicts (2021+) | 3 x 3 |
| requirements conflicts natural language inference | 5 |
| de Marneffe contradictions; Ambig-SWE; Gärtner contradiction taxonomy | 3 + 3 + 2 |

CrossRef total: about 380 hits screened.

### arXiv (`lit.py arxiv-search`)
- requirements ambiguity LLM detection (12); requirements inconsistency LLM conflict detection (6); requirements contradiction detection (12); requirements smells large language models (12); requirements completeness LLM (12); requirements quality LLM evaluation ISO 29148 (0); Validating formal specifications LLM generated test cases Alloy (2); software requirements conflict detection (10). About 66 hits in total.
- One query (Ezzini QA paper) timed out under arXiv contention, so the abstract was read on arxiv.org/abs/2302.04793 instead.

### Web (WebSearch, used for discovery and abstracts)
About 25 queries, roughly 9 results each (about 220 results screened). The topics were:
- LLM requirements ambiguity detection 2024-2025;
- requirements inconsistency LLM + SMT;
- the LLM4RE SLR;
- CellularLint/Hermes;
- LLM protocol-spec ambiguity (RFCScope, RFC-parser validation);
- the PROSPER/SPEC5G/FSM-extraction family;
- abstracts of the Chantree, Ferrari, Ferrari & Esuli, van der Aa, van Lamsweerde, Zhou and iComment papers;
- code for TAPHSIR, ALICE/SAT-LLM, Sage, Hermes/CellularLint, SpecFix/ClarifyGPT and RFCScope;
- abstracts for FRET, ASSERT and SpeAR;
- execution-based ambiguity detection with multiple LLM interpretations;
- underspecification in code generation;
- LLM-agent requirements validation;
- the Cunha & Macedo FM 2026 paper;
- Ambig-SWE at OpenReview/ICLR.

### Backward snowballing (`lit.py refs`)
Anchors:
- ALICE (10.1007/s10515-024-00452-x), with 55 references screened. This added Heitmeyer et al., TOSEM 1996, and de Marneffe et al., ACL 2008. It also surfaced the Gärtner 2022 taxonomy, which was excluded (see below).
- SpecFix (10.1109/ase63991.2025.00038), with 44 references screened. This added Ambig-SWE and confirmed that SpecFix builds on ClarifyGPT.
- ClarifyGPT (10.1145/3660810), with 50 references screened. Its references give the repository github.com/ClarifyGPT/ClarifyGPT.

These anchors were also used for informal forward discovery, through Web "cited-by"-style queries: Chantree et al. RE 2006 (nocuous ambiguity), Sage (SIGCOMM 2021), nl2spec (CAV 2023), Fantechi et al. RE 2023 and Zhao et al. CSUR 2021.

## Close calls (excluded), one line each
- Yatkın & Ovatman 2024, SAT-solver contradiction detection in DO-178C high-level requirements (CSIT/AIRCC): the venue quality is doubtful.
- Malik et al. 2023, transfer learning for conflict and duplicate detection (SSRN only), and Malik et al. 2023, rule-based conflict detection (Research Square): not peer-reviewed.
- Guo, Zhang, Lian 2021, conflict detection based on finer semantic analysis (arXiv 2103.02255): an old preprint with no published version found.
- Wang et al. 2026, dual-encoder conflict detection (PLOS One): peripheral, supervised pair classifier.
- Kisso & Fotrousi REW 2024, conversational AI for conflict detection: workshop, peripheral.
- NLI consistency checking on semisynthetic automotive data (ICMI 2026): the venue is outside the scope.
- Gärtner, Fay, Göhlich 2022, contradiction taxonomy (Applied Sciences): already covered by ALICE (ASE journal 2024).
- RADAR, risk-aware ambiguity detection for regulatory requirements (REW 2026): no abstract available.
- SCLAT (COMPSAC 2026) and AirReq (REW 2025), on user-story and smell quality: peripheral.
- Fernandes et al. 2026, systematic mapping of LLMs for NL requirements (ICEIS), and Khan et al. 2025, LLM4RE SLR (Research Square): a weaker venue and a non-peer-reviewed preprint; Zadenoori et al. 2025 covers the same ground.
- Porter, DeFranco, Laplante 2025, "Requirements specification automated quality analysis: past, present, and future" (IEEE Computer): a magazine overview.
- NLP4RE Handbook chapters (Ferrari & Ginde overview; Frattini et al. tools; Berry "hairy tasks"; Sabetzadeh & Arora guidelines): only the Berry/Kamsties/Ribeiro defect chapter and the Vogelsang/Fischbach LLM guideline were kept.
- Kamsties, "Understanding ambiguity in RE", and Berry & Kamsties, "The syntactically dangerous all and plural" (IEEE Software 2005): subsumed by the Berry/Kamsties/Krieger handbook and the 2004 chapter.
- Yang, De Roeck, Gervasi, "Speculative requirements" (RE 2012): concerns uncertainty cues, which is peripheral.
- The MAANA tool (ICSE companion 2021) and the DRONE tool (ICSE companion 2019): tool versions of included papers.
- DocChecker (EACL demo 2024) and the Rust code-comment inconsistency paper (ASE 2024): covered by C4RLLaMA (ICSE 2025).
- Uncovering gaps between RFC updates and TCP/IP implementations (arXiv 2510.24408), RFCLLM (arXiv 2609.13389), and PROSPER, SpecGPT, FlowFSM and the 3GPP LLM-ensemble FSM extraction: these concern implementation conformance or FSM extraction, already represented by RFCNLP, Hermes, SpecFSM and PSMBench.
- BaseSpec (NDSS 2021): spec-versus-baseband conformance, which is peripheral.
- LLM debate for ambiguity in requests (arXiv 2507.12370), ambiguity in surgical-robot instructions (arXiv 2507.11525) and text-to-structured-data ambiguity (arXiv 2505.11679): not specifications; possibly relevant to another theme.
- "Uncovering systematic failures of LLMs in verifying code against NL specifications" (ASE 2025 NIER): belongs to the LLM-as-judge theme.
- LLM2Alloy (ICST 2026), GUISpector (ICSE 2026 demo) and ClarifySTL (arXiv 2026): peripheral.
- Learning from failures: NL to LTL (QRS 2024): covered by Req2LTL and nl2spec.
- Semantic uncertainty (Kuhn et al.) and input-clarification ensembling (Hou et al., ICML 2024): uncertainty theory, left to the theme on LLM uncertainty and disagreement.

## Unverified / incomplete
- Helmeczi, Cevik, Yıldırım 2022, "A prompt-based few-shot learning approach to software conflict detection" (arXiv 2211.02709; the comment says CASCON 2022): the CrossRef title search found no record, so it was dropped.
- Code not found or not confirmed:
  - TAPHSIR: Zenodo only (recorded as the Zenodo DOI URL);
  - ALICE, SAT-LLM, VERIMED, the pragmatic-ambiguity RAG paper, RFCScope, CellularLint, Sage (the paper says it is on GitHub, but no URL was found), Cunha & Macedo, and Lubos et al.: no repository verified.
- read_level "metadata only" (abstract elided by the publisher or not found): the Berry & Kamsties 2004 chapter, Chantree et al. 2006, van Lamsweerde et al. 1998, SpeAR v2.0, ASSERT (RE 2017), iComment, ARM, QuARS, the Berry/Kamsties/Ribeiro chapter and de Marneffe et al. 2008.
- The QuARS year was missing in CrossRef. It was set to 2001 from the verified BibTeX (series SEW-01).

## Totals
- Hits screened: about 380 (CrossRef) + about 66 (arXiv) + about 220 (Web) + 149 (references) ≈ 815, with duplicates. Roughly 190 distinct candidates were examined at title or abstract level.
- Included: **87** records in `raw/t3_requirements_nlp.jsonl`:
  - by status: 72 peer-reviewed, 9 book or book chapter, 5 arXiv preprints, 1 self-published handbook;
  - 13 have a verified code URL;
  - 13 are flagged `baseline_candidate`.
