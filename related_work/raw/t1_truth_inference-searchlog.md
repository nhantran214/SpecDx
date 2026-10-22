# Search log: t1_truth_inference (truth inference and label aggregation from fallible raters)

Status: COMPLETE (2026-10-21).

## Sources and queries (round 1)
- CrossRef via `lit.py verify`/`search` (fast mode, CrossRef only, for DOI'd classics): Dawid-Skene, CBCC, Passonneau & Carpenter, Paun et al., Sheng et al., Snow et al., TruthFinder (KDD + TKDE), Allman-Matias-Rhodes, Kruskal, Hui-Walter, Qu-Tan-Kutner, Albert-Dodd, Batchelder-Romney, LTM, Dong et al. (source dependence), Li et al. (deep web), Li et al. (truth discovery survey), Zheng et al. (truth inference), CRH, multi-truth (Wang CIKM'15, CIKM'16), Lin & Chen (domain-aware), Zhi et al. (truth existence), Pochampally et al. (correlations), Yin & Tan (semi-supervised), Karger-Oh-Shah (Oper. Res.), Crowd-Kit (JOSS).
- NeurIPS proceedings index pages (papers.nips.cc, 2009-2017), grepped for crowd/unlabeled/weak supervision titles: about 25 hits screened, 9 included.
- PMLR volume index pages (AISTATS, ICML and UAI, v22-v286, 2012-2025), grepped with two regexes (crowd, weak supervision, label model, annotator, unlabeled, LLM judge, truth discovery; then correlated, similarity, identifiable, ambiguity, IRT, partial label): about 330 titles screened, 22 opened, 20 included.
- JMLR pages (Raykar 2010; Zhang et al. 2016), ACL Anthology (MACE), ICML 2012 official paper list (Bachrach et al.).
- WebSearch: "Dawid-Skene aggregation of multiple LLM annotators" (9 hits), "Replacing Judges with Juries" (9), "latent class model LLM judges without ground truth" (9), "LLM annotators disagreement reveals codebook ambiguity" (9), "Bayesian Calibration of Win Rate Estimation" (9), "crowdsourcing truth inference LLM workers hybrid" (9), "TruthFinder abstract" (9), "Domain-aware multi-truth abstract" (9), "Truth inference ... code" (9).
- Semantic Scholar graph API (abstract/TLDR lookup only, for DOI records whose CrossRef entry has no abstract); PubMed E-utilities for Hui-Walter and Qu et al. abstracts.
- arXiv export API: heavily rate-limited (HTTP 429) on 2026-10-20 because it is shared with other theme agents. Discovery was moved to WebSearch, and arXiv was used only for single-id verification.

## Round 2 and round 3 queries
- CrossRef fast verify, round 2 (26 titles): CrowdTruth/Aroyo-Welty, Uma et al. JAIR, Pavlick-Kwiatkowski, Daniel et al. CSUR, Snorkel PVLDB, MeTaL AAAI, Confident Learning JAIR, Klie et al. CL, Knight-Leveson, Eckhardt-Lee, Littlewood-Miller, Prelec et al. Nature, Reitsma et al., Dumitrache et al. TiiS, Parisi et al. PNAS, Anandkumar et al. (the match was a DTIC report, so it was replaced by the JMLR page), Dalvi et al., Shah et al. TIT, Goodman 1974, Cour et al. (the match was an encyclopedia entry, so it was replaced by the JMLR page), Schoenegger et al., Li ICASSP'24, Bommasani et al., Ipeirotis et al., Dong et al. KBT, Li et al. WWW'19.
- CrossRef fast verify, round 3 (11 titles): Gao et al. EMNLP'24, Li EMNLP'24, Plank EMNLP'22, Nie et al. EMNLP'20, Dhurandhar et al. Findings ACL'24, GEM'23 ensemble disagreement (not in CrossRef; found on ACL Anthology), Kulmizev et al. ACL'26, Smith et al. ACM/IMS JDS'24, Toups et al. NeurIPS'23, Jiang & de Marneffe TACL'22, Liu et al. EMNLP'23. Abstracts were read on the ACL Anthology pages.
- NeurIPS index pages 2013-2025, grepped with a wider regex (about 140 titles screened); opened Liu-Ihler-Steyvers 2013, Ibrahim et al. 2019, Nguyen et al. 2024, Maia Polo et al. 2024 and Toups et al. 2023.
- WebSearch: "ensemble of LLMs detect mislabeled or ambiguous benchmark items" (9), "aggregating multiple LLM judges Dawid-Skene OR IRT" (9), "Testing LLM Ensemble Disagreement ... Accounting Standards" (9), "Eckhardt Lee coincident errors abstract" (10), "Knight Leveson abstract" (10).
- arXiv abs pages (arxiv.org/abs, read for abstracts) and `lit.py verify --arxiv` for 12 recent preprints (all matched=arxiv).

## Anchor papers used for snowballing
Dawid & Skene 1979; Zheng et al. PVLDB 2017; Li et al. SIGKDD Explor. 2016; Zhao et al. PVLDB 2012 (LTM); Ratner et al. NeurIPS 2016; Kim et al. ICML 2025 (correlated errors).

## Close-call exclusions (rounds 2 and 3)
- Deng & Tang, "Testing LLM Ensemble Disagreement as an Ex-Ante Signal of Underspecification in Accounting Standards" (NeurIPS 2026 workshop "Can We Trust AI Evaluation?", openreview lqdnN5lFmr). Highly relevant negative evidence: raw ensemble disagreement gives AUC 0.47 for contested IFRS requirements. It is a workshop paper, and its only record is the NeurIPS virtual page and OpenReview, which lit.py cannot verify. Not recorded; the coordinator may add it after an OpenReview check.
- "A Judge-Aware Ranking Framework" (arXiv 2601.21817) and BT-sigma "Who can we trust? LLM-as-a-jury" (arXiv 2602.16610, ICML 2026): both verified on arXiv. They are judge-reliability Bradley-Terry extensions for pairwise ranking, judged lower relevance than the included Ising and beta-binomial papers, and left out under the coordinator's cap of about 10 additions.
- "Benchmark Illusion" (arXiv 2602.11898) and "Variance-Aware LLM Annotation" (arXiv 2601.02370): both verified. They are context on LLM disagreement and variance, with medium-low relevance, and were left out under the cap.
- Dalvi et al. WWW'13 and Goodman 1974 Biometrika: verified, but no abstract could be retrieved. Their content is covered by Karger et al. and Allman et al.
- Plank EMNLP'22, Nie et al. EMNLP'20, Jiang & de Marneffe TACL'22, Liu et al. EMNLP'23 (AmbiEnt), Kulmizev et al. ACL'26, Smith et al. JDS'24, Toups et al. NeurIPS'23, Dhurandhar et al. Findings'24, Li EMNLP'24, GEM'23 disagreement scores, Ibrahim et al. NeurIPS'19, Nguyen et al. NeurIPS'24, Maia Polo et al. NeurIPS'24: all verified and relevant, but excluded under the coordinator's cap. The disagreement and ambiguity side is represented by Aroyo-Welty, Uma et al., Pavlick-Kwiatkowski and Dumitrache et al. Records for these are in the scratchpad if needed.
- AMA prompting (ICLR 2023), WRENCH (NeurIPS'21 D&B), PiCO (ICLR'25), Oleson et al. "Programmatic Gold" (HCOMP'11), Pasternack & Roth COLING'10: not verified within the session budget.

## Close-call exclusions (round 1)
- Gao & Zhou 2013, "Minimax optimal convergence rates for estimating ground truth from crowdsourced labels" (arXiv 1310.5764): old preprint with no published version found. Superseded by Gao, Lu & Zhou ICML 2016 (included).
- Li & Yu 2014, "Error rate bounds and iterative weighted majority voting for crowdsourcing" (arXiv 1411.4086): old preprint with no published version found in CrossRef.
- Zhang et al. NeurIPS 2014 "Spectral methods meet EM": kept only the JMLR 2016 journal version.
- Kuang et al. AISTATS 2022 "Firebolt": weak-supervision variant (class balance, class-specific LF accuracy) without dependency, set or anchor aspects. Low relevance.
- Ben Shabat et al. UAI 2022 "Empirical Bayes approach to truth discovery": real-valued truth discovery. Low relevance.
- Fang, WWW'17 Companion "Truth discovery from conflicting multi-valued objects": a PhD-symposium paper, covered by Wang et al. CIKM'15/'16.
- Yin et al. KDD 2007 TruthFinder: kept the TKDE 2008 journal version (the KDD DOI is noted in its summary).

## Unverified (not recorded)
- Deng & Tang NeurIPS 2026 workshop (see above); AMA, WRENCH, PiCO, Programmatic Gold.
- Pasternack & Roth, COLING 2010 "Knowing what to believe (when you already know something)": the CrossRef lookup failed. Pending an ACL Anthology check.
- EBCC code (github.com/yuan-li/truth-inference-at-scale): the repository exists, but its link to the ICML 2019 paper is not confirmed (no README fetched). Not recorded as code_url.

## Totals
- Screened: about 600 titles (about 330 PMLR, about 165 NeurIPS index titles, about 100 web and CrossRef results), plus about 130 records opened or verified.
- Verified with lit.py or an official proceedings page: 118.
- Included: 93 (all in t1_truth_inference.jsonl, no duplicate titles).
- Preprints included: 9 (2024-2026 arXiv).
