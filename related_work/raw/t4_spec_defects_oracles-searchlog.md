# Search log: t4_spec_defects_oracles

Theme: defects in formal/executable specifications and test oracles (finding, localising, repairing, evaluating); specification mutation; oracle quality with mutants.

Tools: `lit.py` (CrossRef `verify`/`search`, arXiv `verify --arxiv`), OpenAlex singleton DOI lookups (zero-cost endpoint) and Semantic Scholar DOI lookups for abstracts, WebSearch for discovery, GitHub search API + `lit.py github` for code.

## Queries run

### Lead verification (CrossRef `verify --title`, batches)
| Batch | Content | Items | Verified |
|---|---|---|---|
| b1 | mutation testing / spec mutation / equivalent mutants | 12 | 12 |
| b2 | oracle quality (Barr, Staats, checked coverage, Jahangirova, GAssert, Daikon, pseudo-tested, SpecFuzzer, EvoSpex) | 21 | 20 (+1 via DOI: Pezze & Zhang) |
| b3 | vacuity, coverage for temporal logic, Alloy faults (unsat cores, ARepair, AlloyFL, FLACK, BeAFix, ICEBAR, ATR, MuAlloy, AUnit) | 23 | 23 |
| b4 | Alloy4Fun, nl2postcond, Lahiri FMCAD, SpecGen, AutoSpec, EuroSys'17, Clover, TiCoder, nl2spec, EvalPlus, APR overfitting, SCR/RSML consistency, SQL mutation, test-code bugs | 20 | 18 (TiCoder arXiv title -> TSE version; FMCAD via arXiv) |
| b5 | arXiv IDs (FMCAD'24, MutDafny, LLM Alloy repair, ABC, BenchJack, SWE-Bench Pro Verified, SpecMind, Spec-Harness, TOGLL, UTBoost, PatchDiff ...) | 14 | 7 at first pass; 7 hit arXiv HTTP 429 (retried later) |
| b6 | Molina&Gorla TOSEM, SchemaAnalyst family, IronSpec, Alloy4Fun studies, BDD quality, neural oracles, misc | 20 | 15 |

### Discovery (WebSearch)
- "LLM repair Alloy declarative specifications empirical evaluation 2024 2025" (9 hits screened)
- "evaluating LLM-generated formal specifications soundness completeness mutants tests postconditions 2025" (9)
- "BDD Gherkin scenarios quality assessment smells empirical study" (9)
- "specification bugs formally verified systems empirical study incorrect specifications Dafny Coq Verus" (9)
- "weak test cases insufficient tests SWE-bench patches incorrectly pass evaluation UTBoost PatchDiff" (9)
- "Alloy4Fun dataset specification mistakes students analysis" (10)
- "LLM test oracle generation false positives false negatives assertions expected vs actual behaviour study" (10)
- "mutation testing formal specifications verification-aware language assess specification strength mutants 2024 2025 arXiv" (9)
- "agentic benchmark outcome validity checklist task validity flawed tests reward hacking benchmark grader" (9)
- "IronSpec increasing the reliability of formal specifications OSDI 2024" (10)
- "mutation testing database integrity constraints schema SchemaAnalyst mutation analysis relational schema" (10)
- "\"test oracle automation in the era of LLMs\" TOSEM" (9)

### Code discovery (GitHub search API)
Searched repository names for GAssert, OASIs, ARepair, AlloyFL, FLACK, BeAFix, ICEBAR, MuAlloy, AUnit, SchemaAnalyst, Daikon, SpecFuzzer, nl2postcond, ATR, MutDafny, IronSpec, Clover, EvalPlus, Alloy4Fun, nl2spec, SpecGen, AutoSpec, OraclePolish, BenchJack, Spec-Harness, UTBoost.

(continued below)

### Additional verification rounds
- b7: arXiv retry after HTTP 429 (ABC, NL2Contract, FormalBench, TOGLL, Right-or-Wrong arXiv, UTBoost, PatchDiff, Clover): all 8 resolved.
- b8: published versions by DOI/title: TOGLL (ICSE'25), PatchDiff (ICSE'26), MutDafny (ICSE'26), ABC (NeurIPS'25), Clover (SAIV'24). FMCAD'24 DOI is DataCite (not CrossRef): confirmed via doi.org BibTeX. UTBoost confirmed on ACL Anthology (.bib). IronSpec confirmed on the USENIX OSDI'24 page.
- Abstracts: OpenAlex singleton DOI endpoint, Semantic Scholar by DOI, and the HUJI CRIS pages / WebSearch for Springer-only papers (Kupferman et al., Chockler et al., Torlak et al., Beer et al.).
- Code checked with `lit.py github`: codespecs/daikon, valerio-terragni/gassert, kaiyuanw/ARepair, kaiyuanw/MuAlloy, guolong-zheng/atmprep (ATR; URL from the paper abstract), MutDafny/mutdafny, GLaDOS-Michigan/IronSpec, schemaanalyst/schemaanalyst, CUHK-Shenzhen-SE/UTBoost, benchjack/benchjack, Mondego/vACT, realChrisHahn2/nl2spec.

## Anchor papers for snowballing
Barr et al. TSE 2015 (oracle survey); Jahangirova et al. ISSTA 2016; Kupferman & Vardi STTT 2003; Torlak et al. FM 2008; Endres et al. FSE 2024 (nl2postcond); Gutierrez Brida et al. ICSE 2021 (BeAFix) and its successors (ICEBAR, ATR, LLM repair); Zhu et al. ABC (NeurIPS 2025).

## Excluded close calls (verified, not recorded)
- Pezze & Zhang, "Automated Test Oracles" (Adv. Computers 2014): redundant with Barr et al.; no abstract found.
- Kupferman & Vardi CHARME 1999 and Beer et al. FMSD 2001, Chockler et al. FMSD 2006 / STTT 2006: journal or conference duplicates of included versions.
- Armoni et al. CAV 2003, Gurfinkel & Chechik TACAS 2004 (vacuity refinements): marginal; the context budget was spent on higher-relevance items.
- Budd & Gopal 1985, Budd & Angluin 1982: verified (CrossRef) but no abstract was readable; Black et al. ASE 2000 and Ammann et al. cover spec mutation. Should be cited as the historical origin of specification mutation.
- Molina, Gorla & d'Amorim, "Test Oracle Automation in the Era of LLMs" (TOSEM 2025): relevant (oracle false positives/negatives with LLMs), but omitted to stay within the cap. A good addition if there is room.
- TOGA (ICSE'22), Hossain et al. (FSE'23), Liu et al. (ISSTA'23): neural oracle generation, close to TOGLL; omitted.
- SchemaAnalyst ICSME'16, Kapfhammer ICST'13, McMinn AST'16 (virtual mutation): covered by the TOSEM 2015 record.
- Fraser et al. STVR testing-with-model-checkers survey: included. Newcombe CACM 2015, Zave CCR 2012, Yang PLDI 2011: context only, not specification-defect methods.
- Clover (SAIV 2024), SpecGen (ICSE'25), AutoSpec (CAV'24), nl2spec (CAV'23), TiCoder (TSE'24), FormalBench (ACL'25), SpecMind (ACL'26), NL2Contract (preprint), EvalPlus (NeurIPS'23), SWE-Bench Pro Verified (preprint), APR overfitting (Smith FSE'15, Qi ISSTA'15), test-code bugs (Vahabzadeh ICSME'15), false test alarms (Herzig ICSE'15), Heimdahl & Leveson TSE'96, Heitmeyer TOSEM'96 (SCR consistency checking), BDD quality papers (Oliveira REW'17, Binamungu SANER'18 / VST'18 / XP'20): all verified, with abstracts read for most. They were not recorded under the coordinator's cap of about 12 additions. Heitmeyer TOSEM'96 and nl2spec are the strongest of these for a later pass. The requirements-consistency items probably overlap theme t3.

## Unverified
- FLACK and AlloyFL repositories: GitHub search found candidates (AlloyFL/AlloyFLCode, kaiyuanw/AlloyFLCore), but none is linked from a paper, so none was recorded.
- OASIs and OraclePolish code: no repository found.

## Totals
- Candidates screened: about 190 (leads, plus about 110 web/GitHub search hits).
- Verified: about 125.
- Included: 62 (JSONL).
