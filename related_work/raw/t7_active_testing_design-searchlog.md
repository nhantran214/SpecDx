# Search log: t7_active_testing_design

Theme: active and adaptive testing, Bayesian experimental design, bandits and Thompson sampling, information-directed sampling, budget allocation for noisy raters, sequential testing, and search-based or boundary-oriented scenario generation. This theme maps to SpecDx component 6 (active scenario synthesis: Thompson-EIG over scenario cells, boundary values from clause literals, and pilot-then-committee escalation).

Run dates: 2026-10-20 to 2026-10-21. The run was interrupted once by an API usage limit and resumed. The coordinator then stopped broad searching at 61 records, and the verified items already in hand were flushed.

## Tools and sources

- `lit.py verify` (CrossRef plus arXiv). The arXiv export API often returned HTTP 429. A CrossRef-only wrapper that calls `lit.py`'s own functions (`crossref_search`, `crossref_doi`, `doi_bibtex`) was used instead.
- For arXiv abstracts, `https://arxiv.org/abs/<id>` pages were read when the export API was rate-limited.
- Abstracts of DOI records came from the CrossRef `abstract` field, or else from the OpenAlex `abstract_inverted_index` looked up by DOI. One lookup per paper.
- DOI-less proceedings were checked on official pages, reading the `citation_*` meta tags and the abstract:
  - proceedings.mlr.press: COLT 2012, ICML 2013/2017/2021/2024/2025;
  - proceedings.neurips.cc: NeurIPS 2011/2019;
  - proceedings.iclr.cc: ICLR 2025;
  - aclanthology.org: NAACL 2024, with its .bib;
  - the OpenReview API: COLM 2025;
  - jmlr.org (Kaufmann et al., verified but not included; see below).
- WebSearch was used for discovery and to locate venue pages.
- `lit.py github` checked code. The GitHub API became rate-limited (shared IP, 60 requests per hour) partway through. After that, the existence of each repository was confirmed by HTTP 200 on github.com, with `code_info` left null.

## Queries and lookups (source, hits screened)

| # | Query / lookup | Source | Screened |
|---|---|---|---|
| 1 | Seed leads, verified one by one: Thompson 1933; TS tutorial; Agrawal & Goyal; Chapelle & Li; IDS; Chaloner & Verdinelli; Rainforth et al.; VBOED; BALD | lit.py verify | 9 |
| 2 | Settles survey or book; active testing; ASE; BatchBALD; Gal 2017; Lindley 1956 | lit.py verify | 7 |
| 3 | Optimistic KG (ICML 2013 and JMLR version); Karger–Oh–Shah; Sheng et al.; Khetan & Oh; Frazier KG | verify / CrossRef | 6 |
| 4 | Wald SPRT; Pocock; O'Brien–Fleming; Chernoff 1959; Jennison & Turnbull; Naghshvar & Javidi; Howard et al.; Ramdas et al. | CrossRef | 8 |
| 5 | "adaptive testing large language models cognitive ability" | arXiv search | 8 |
| 6 | Zhuang et al.; Fluid Benchmarking; tinyBenchmarks; Active Evaluation Acquisition; Truong et al.; Perlitz et al. | verify / arXiv / WebSearch | 6 |
| 7 | CAT foundations: Chang & Ying; Weiss & Kingsbury; van der Linden & Glas; CAT survey (TPAMI) | CrossRef | 4 |
| 8 | Scenario generation: Ben Abdessalem ASE 2018 and ICSE 2018; AsFault; DeepJanus; DeepHyperion; Nie & Leung; Kuhn et al.; AETG; QuickCheck; Deshmukh et al.; Mullins et al. | CrossRef | 11 |
| 9 | Boundary values: BVE ICSTW 2020; Feldt & Dobslaw SSBSE 2019; Dobslaw & Feldt SBFT 2024; "boundary value analysis testing"; "automated black-box boundary value detection"; "boundary value analysis specification-based testing domain testing"; White & Cohen | CrossRef | 6 + 6 + 8 + 6 |
| 10 | Operational DNN testing: Li et al. FSE 2019; DeepEST ICSE 2021; PACE TOSEM 2020 | CrossRef | 3 |
| 11 | "policy graph scenario generation evaluating conversational agents"; "automatic test generation LLM agents policy compliance scenarios"; "synthesize test scenarios from business policies tool-calling agents"; "adaptive testing LLM agents information gain scenario selection" | arXiv search (mostly timeouts or 429) | 1 |
| 12 | "IntellAgent ... policy graph"; "automatic generation of test tasks for tool-calling agents from domain policies (tau-bench style)" | WebSearch | 9 + 9 |
| 13 | QBC; DeepXplore; Entropic; ART (JSS 2010); ADS testing survey (TOSEM); AdaTest; LM red teaming | CrossRef | 7 |
| 14 | Russo top-two TS; Kaufmann et al. BAI; MAP-Elites; O'Kelly et al. rare-event; Spieker et al. RL test prioritisation; FrugalGPT | CrossRef / JMLR page | 6 |
| 15 | Trust or Escalate; Gotovos et al. level-set estimation; Active Statistical Inference | WebSearch + proceedings pages | 3 |

## Anchor papers used for snowballing

- Active testing (Kossen et al., ICML 2021) led to ASE (NeurIPS 2022), the operational DNN testing line (FSE 2019, ICSE 2021, TOSEM 2020) and active statistical inference (ICML 2024).
- Fluid Benchmarking (COLM 2025) and Zhuang et al. (ICML 2025) led to tinyBenchmarks, Truong et al., the TPAMI CAT survey, Chang & Ying and Weiss & Kingsbury.
- DeepJanus (FSE 2020) led to DeepHyperion and to the boundary-value exploration line (Dobslaw, Feldt et al.), which in turn led back to White & Cohen's domain testing.
- IntellAgent and tau2-Bench (WebSearch) led to TASTE / tau^c-Bench (2026).
- From the Mullins et al. web search: ScenePilot (ICML 2026, boundary-driven critical scenario generation) and a 2025 multi-output GP boundary-scenario paper for multi-UAV systems were seen but not verified, because the coordinator stopped broad searching.

## Included: 64 records (22 high, 28 medium, 14 low relevance)

| Subtopic | Count | Papers |
|---|---|---|
| Bandits / TS / IDS / best-arm identification | 7 | Thompson 1933; TS tutorial; Agrawal & Goyal; Chapelle & Li; IDS; Russo top-two TS; Frazier KG |
| BOED / information gain | 4 | Lindley; Chaloner & Verdinelli; Rainforth et al.; VBOED |
| Active learning, BALD, QBC | 5 | BALD; BatchBALD; Gal 2017; Settles book; QBC |
| Active testing and label-efficient evaluation | 7 | Kossen 2021; ASE; active statistical inference; Li et al. FSE 2019; DeepEST; PACE; AEA (preprint) |
| CAT and adaptive LLM evaluation | 8 | Zhuang ICML 2025; Fluid; tinyBenchmarks; Truong; Perlitz; Chang & Ying; Weiss & Kingsbury; CAT survey |
| Crowdsourcing budget allocation | 3 | opt-KG; Karger–Oh–Shah; Sheng et al. |
| Sequential testing, escalation, anytime-validity | 9 | Wald; Chernoff; Naghshvar & Javidi; Pocock; O'Brien–Fleming; Howard et al.; Ramdas et al.; Trust or Escalate; Weiss (adaptive mastery testing) |
| Scenario/test generation, search-based, boundary, combinatorial, PBT | 21 | see JSONL |

Preprints included:
- BALD (2011): a seminal method the caller named; an exception to the recency rule.
- Active Evaluation Acquisition (2024).
- IntellAgent (2025).
- TASTE (2026).

## Excluded close calls

- **Settles, "Active Learning Literature Survey" (2009 TR)**: not in CrossRef; arXiv returned 429. Replaced by the verified Settles 2012 book.
- **Chen, Lin & Zhou JMLR version of opt-KG**: CrossRef matched only an SSRN copy. Kept the ICML 2013 version.
- **Jennison & Turnbull book**: CrossRef matched only a book review. Pocock and O'Brien–Fleming cover the topic.
- **van der Linden & Glas, *Elements of Adaptive Testing***: CrossRef matched only a chapter. CAT is covered by the TPAMI survey and the primary papers.
- **Feldt & Dobslaw SSBSE 2019**: verified (10.1007/978-3-030-27455-9_11), but no abstract was available. The 2023 PeerJ CS journal version and BVE 2020 are included instead.
- **Adaptive Random Testing (Chen et al., JSS 2010)**: verified (10.1016/j.jss.2009.02.022), but no abstract was available and its relevance is lower than ART-free diversity selection. Dropped for budget.
- **Kaufmann, Cappé & Garivier, JMLR 2016, best-arm identification**: verified on jmlr.org. Dropped as redundant with Russo 2020 after the coordinator stopped the search.
- **Spieker et al., ISSTA 2017 (RL test prioritisation in CI)**: verified (10.1145/3092703.3092709). Dropped: it targets regression prioritisation, not information-driven scenario choice.
- **Gotovos et al., IJCAI 2013 (active level-set estimation)**: highly relevant to boundary-seeking active sampling, but the IJCAI page returned no metadata to the parser and there was no time to verify it another way. Listed as unverified below; recommended follow-up.
- **ElectriQ (2025)** and other arXiv hits from query 5 (cognitive-simulation papers): off-topic.

## Unverified (dropped)

- MAP-Elites, "Illuminating search spaces by mapping elites" (2015): arXiv-only, and not matched in CrossRef.
- O'Kelly et al., "Scalable End-to-End Autonomous Vehicle Testing via Rare-event Simulation" (NeurIPS 2018): not matched in CrossRef; the proceedings page was not checked.
- FrugalGPT (Chen, Zaharia & Zou; TMLR): not matched in CrossRef; the OpenReview page was not checked. Trust or Escalate covers the cascade idea.
- Khetan & Oh, "Achieving budget-optimality with adaptive schemes in crowdsourcing" (NeurIPS 2016): not matched in CrossRef.
- Gotovos et al. (IJCAI 2013): see above.
- ScenePilot (ICML 2026) and the multi-UAV GP boundary-scenario preprint (2025): seen in search results, not verified.
- Not reached: Rainbow Teaming (NeurIPS 2024) and ToolEmu (ICLR 2024), which were planned for the LLM-agent scenario subtopic.

## Totals

- Screened: about 160 candidates (titles and abstracts across the queries above).
- Verified: 74.
- Included: 64.
