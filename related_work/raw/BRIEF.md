# Brief for theme agents: SpecDx related-work survey (2026-10-20)

You are collecting **verified** related work for a paper targeting **ACM TOSEM**. Read this whole brief before searching.

## 1. What SpecDx is (the method the related work must surround)

**Problem.** An *executable business specification* is S = ⟨T, C, Π⟩:
- T (Tell): natural-language requirement documents, policies, a knowledge base, tool descriptions;
- C (Check): executable clauses (database invariants, expected outcomes, a golden end state, guardrail triggers, SQL policy functions);
- Π: a space of business scenarios.

Systems and agent benchmarks (e.g. τ-bench / τ²-bench) treat S as ground truth, but S has defects. τ²-Bench-Verified had to repair many tasks.

**Idea.** A *population of fallible LLM agents* (local models, several families) executes the specification, and each run is one sample of how a reasonable executor reads T. SpecDx separates *specification defects* (every competent interpreter deviates at the same place) from *agent errors* (scattered, interpreter-specific). It is fully automatic: no human labels.

**Defect taxonomy (defined by satisfaction).**
- Contradiction: no behaviour satisfies C.
- Hidden requirement / mis-key: C is stricter than T.
- Ambiguity: T admits two or more readings.
- Weak check: C is looser than T, so a wrong behaviour passes every clause.

**Method components (current design v3).**
1. **Anchored cell model.** The unit is a *cell*: a (table, row, column) value or one clause, read at a scenario step. The specification's declared value d_c is an **anchor**, so the specification acts as a fallible rater. Each cell has a latent **entailment set** A_c ∈ {{x}, {x,y}, ∅}, which makes the truth set-valued and not single-label.
   - Readings are faithful with accuracy σ(α_m − β_g), which is Rasch/IRT-like.
   - Otherwise readings come from an error distribution restricted to values outside A_c, with a Dirichlet prior around a flat pool.
   - It is fitted by MAP/EM.
   - This is a Dawid–Skene/GLAD-style truth-inference model with set-valued truth, anchors and item difficulty.
2. **Identifiability theory.** Accuracy-gradient signatures, i.e. how reading mass changes with model accuracy; identification from two models with distinct accuracy, or from one model with a known error pool; anchor-based contrasts with a median breakdown point of ½; Hoeffding finite-sample bounds. Errors are assumed independent across model families, which relates to correlated LLM errors and N-version programming.
3. **Contradictions via MUS/MCS duality.** Each behaviour's violated-clause set is a sampled *correction set*. Soft minimal hitting sets give candidate MUSes (Reiter; Liffiton & Sakallah).
4. **Sound refutation.** A candidate is refuted by a witness state, found by symbolic local search on PostgreSQL or by a directed LLM probe.
5. **Ambiguity localisation in T.** ddmin (delta debugging) over the sentences of T, with an LLM-population oracle.
6. **Active scenario synthesis.** Thompson sampling to maximise information gain, boundary values taken from the clauses, and sequential escalation (a pilot first, then the full committee).
7. **Evaluation.** Specification *mutation operators* with kill witnesses (seeded defects); real defects from TonomyX history and the τ²-Bench-Verified diff; a pre-registered simulation study.

**Baselines already planned (find their published counterparts):**
- B1: static LLM review of the requirements (map-reduce plus self-consistency);
- B2: best single interpreter, with every violation flagged;
- B3: 2PL IRT on the interpreter × clause matrix, flagging negative discrimination (as in "Auditing LLM benchmarks with IRT");
- B4: inter-family disagreement entropy;
- B5: symbolic consistency analysis without interpreters;
- in simulation: majority vote, anchor-only, and the older joint-class latent model.

**Positioning claims to test.** No prior work combines (statistics over a population of LLM executors) + (logic, i.e. MUS/MCS and refutation) to give a *typed*, *localised* diagnosis of *specification* defects, automatically and with witnesses. Look actively for work that **threatens** this claim, and mark it `relevance: high`, `role: direct competitor`.

## 2. Your job

Search your theme (given in your task prompt) **broadly**: widen the funnel, screen many candidates, keep every paper that is genuinely relevant. Cover the following:
- seminal and foundational work;
- surveys;
- the last 5 years, especially 2023–2026, including 2026 arXiv;
- the papers closest to SpecDx.

**Prioritise reputable venues.**
- SE: ICSE, FSE/ESEC-FSE, ASE, ISSTA, ICST, RE, TOSEM, TSE, EMSE, IST, JSS, REJ.
- AI/ML/NLP: NeurIPS, ICML, ICLR, AAAI, IJCAI, KDD, ACL, EMNLP, NAACL, JMLR, TACL, TMLR, JAIR, AIJ.
- DB/data: VLDB, SIGMOD, ICDE.
- Formal methods and logic: CAV, TACAS, FM, SAT, CP, KR, JAR.
- Security: S&P, USENIX Security, CCS, NDSS.
- BPM/IS: BPM, CAiSE, Information Systems, DKE, SoSyM.
- Statistics: JASA, JRSS, Annals of Statistics, Biometrika, Psychometrika.

**Preprints.** Include arXiv preprints only when they are recent (2024–2026) and clearly relevant, and label them `preprint`. Exclude predatory venues and blog posts. A dataset or tool may be included if it is a paper, or an official release that a paper uses.

## 3. Verification (non-negotiable)

**Never record a paper you have not verified with the tool.** Do not write metadata from memory.

```bash
T=/home/ethan/AILab/TonomyX-paper/related_work/tools/lit.py
python3 $T search "query words" --rows 10 [--from-year 2021]   # CrossRef discovery
python3 $T arxiv-search "query words" --rows 10                # arXiv discovery
python3 $T verify --title "Exact Title" [--year 2021] --no-abstract   # canonical record + bibtex
python3 $T verify --arxiv 2506.07982
python3 $T verify --doi 10.1145/3510003.3510157
python3 $T refs 10.1109/TSE.2014.2372785                       # backward snowballing
python3 $T github owner/repo                                   # code check
python3 $T openalex-title "Exact Title"     # LAST resort only: small shared daily budget
python3 $T s2-search "query"                # often HTTP 429; do not rely on it
```

**Which source verifies a paper.**
- `verify` returns `matched: crossref`: peer-reviewed metadata from CrossRef, with BibTeX from doi.org.
- `matched: arxiv`: a preprint.
  - If the arXiv `journal_ref` or `comment` names a venue, run `verify --title` again to find the published DOI.
  - If none is found, open the official proceedings page to confirm.
- Proceedings without DOIs need the official page, fetched with WebFetch or curl:
  - NeurIPS: proceedings.neurips.cc or papers.nips.cc;
  - ICML and AISTATS: proceedings.mlr.press;
  - ICLR: openreview.net;
  - ACL venues: aclanthology.org;
  - USENIX: usenix.org.
  - Record `verified_by: "proceedings page <url>"`, and build BibTeX from that page; ACL Anthology and PMLR offer `.bib` files.
- `matched: null`: try exact title variants and `openalex-title`, then the venue page. If it still cannot be verified, **drop it** and list it under "unverified" in your search log.

**Abstracts.** Read the abstract before writing a summary: `verify` without `--no-abstract` returns the arXiv abstract, or use WebFetch on the landing page. Summaries paraphrase what you actually read. Set `read_level` honestly.

**Code.** If you find a code link in the paper, the abstract, the project page or a GitHub search, check it with `lit.py github` and record the URL. Never guess a repository.

**Web search.** WebSearch and WebFetch can be loaded with ToolSearch (`select:WebSearch,WebFetch`). Use them for discovery, snowballing ("cited by", Google Scholar-style queries) and venue pages.

## 4. Output (write ONLY these two files)

1. `/home/ethan/AILab/TonomyX-paper/related_work/raw/<theme_id>.jsonl`
   - One JSON object per line, one line per included paper.
   - Append as you go, so that partial work survives.
2. `/home/ethan/AILab/TonomyX-paper/related_work/raw/<theme_id>-searchlog.md`
   - The queries you ran, with their source and the number of hits screened.
   - The anchor papers used for snowballing.
   - The excluded papers that were close calls, with a one-line reason each.
   - Unverified items.
   - Totals: screened, included.

Do **not**:
- download PDFs or clone repositories;
- edit any other file in the repository;
- touch `work/`, `vendor/` or the database.

**JSONL record schema** (use exactly these keys; use null when unknown):

```json
{
 "title": "as in the verified record",
 "authors": ["First Last", "..."],
 "year": 2022,
 "venue": "full venue name",
 "venue_short": "ICSE",
 "venue_kind": "journal | conference | workshop | preprint | book | thesis | software | dataset | report",
 "status": "peer-reviewed | preprint | book | other",
 "doi": "10.xxxx/... or null",
 "arxiv_id": "2506.07982 or null",
 "url": "canonical URL (doi.org, official proceedings page, or arXiv abs)",
 "pdf_url": "open-access PDF URL (arXiv pdf, official OA proceedings PDF) or null",
 "code_url": "verified repository URL or null",
 "code_info": {"stars": 0, "license": "MIT", "pushed_at": "...", "language": "Python"},
 "citations": 123,
 "citations_source": "crossref | openalex | s2 | null",
 "verified_by": "crossref | arxiv | proceedings page <url> | openalex",
 "theme": "<theme_id>",
 "subtopics": ["short tags"],
 "relevance": "high | medium | low",
 "role": "direct competitor | component prior art | foundation | evaluation/benchmark | survey | context",
 "specdx_components": ["anchored cell model", "set-valued truth", "identifiability", "item difficulty/IRT", "family effects/correlated errors", "MUS/MCS", "hitting sets", "refutation", "ddmin", "active synthesis", "mutation evaluation", "defect taxonomy", "real-defect benchmark", "baseline B1", "baseline B2", "baseline B3", "baseline B4", "baseline B5"],
 "summary": "1-2 sentences paraphrasing the abstract",
 "relation_to_specdx": "1-2 sentences: what it shares with SpecDx and how SpecDx differs",
 "baseline_candidate": false,
 "baseline_notes": "if true: what it would detect in our setting, inputs it needs, how to adapt, code status",
 "bibtex": "verified BibTeX string",
 "read_level": "abstract | full text | metadata only"
}
```

`specdx_components` lists only the components that apply.

`baseline_candidate: true` is for methods that could be **run against SpecDx on the same input**, meaning one of:
- a specification: NL requirements plus executable checks;
- a matrix of LLM readings or behaviours;
- a clause set.

Strongly prefer methods with released, runnable code that works offline. Explain the adaptation concretely.

## 5. Finish

Your final message must contain the following:
- the path of your JSONL and search log;
- included and screened counts;
- the 5–10 most important papers;
- every paper that threatens SpecDx's novelty claim, and why;
- your top 3–5 baseline candidates, with their code status.
