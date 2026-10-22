# Search log: t2_irt_benchmark_audit (IRT/psychometrics for AI evaluation, label-error detection, benchmark repair)

Agent run 2026-10-20/21. Verification uses `tools/lit.py verify` (CrossRef/arXiv), plus official proceedings pages (PMLR, proceedings.iclr.cc, papers.nips.cc, datasets-benchmarks-proceedings.neurips.cc, aclanthology.org). Abstracts come from arXiv (lit.py), ACL Anthology, OpenAlex/Semantic Scholar (by DOI), or the PMLR bibliography. Code was checked with `lit.py github`.

## Queries (source: hits screened)
- lit.py verify on 24 seed titles from the brief (CrossRef+arXiv): 24 screened, 21 included; beta3-IRT and the first Gema title lookup were unmatched, and Gema was then verified by DOI.
- lit.py verify --arxiv 2605.30504, 2603.29399, 2609.14400, 2406.04127: 4/4 verified.
- WebSearch "τ²-Bench-Verified repaired tasks paper": 9 screened; led to amazon-agi/tau2-bench-verified, whose README cites SABER (arXiv 2512.07850).
- WebSearch "OSWorld-Verified paper arXiv" and "OSWorld-Verified benchmark fixes tasks": 18 screened; OSWorld-Verified is a blog/release only, so the OSWorld NeurIPS paper is included instead.
- WebSearch "item response theory negative discrimination mislabeled benchmark items LLM 2025": 10 screened; found ATLAS and PSN-IRT.
- WebSearch "Are LLMs Better than Reported? Detecting Label Errors": 9 screened; found Nahum et al. (EMNLP 2025).
- WebSearch "Lost in Benchmarks item response theory PSN-IRT": 9 screened; found AAAI 2026 and Laplace-PSN-IRT (excluded).
- WebSearch "WebArena Verified benchmark audit task errors 2025 paper": 10 screened; found mis-score CUA (2607.28367). WebArena Verified itself remains unverified.
- WebSearch "agent benchmark audit broken tasks ... Verified 2026 arXiv": 9 screened; found ABA (2605.26079) and BenchGuard (2604.24955).
- WebSearch "Are Solved Issues in SWE-bench Really Solved Correctly": 8 screened; found the ICSE 2026 paper and SWE-ABS (2603.00520).
- WebSearch for proceedings pages of tinyBenchmarks, metabench, τ-bench, SWE-bench, AUM, MMLU-Redux and platinum benchmarks: about 50 screened.
- PMLR v306 (ICML 2026) bibliography grep for "item response | psychometric | label error | benchmark audit": 5 hits, 4 included (ATLAS, AgentSuite/COBA, IRT LLM-as-judge, Pólya-Gamma EM Rasch). "Item Response Scaling Laws" was excluded as off-topic.
- CrossRef search (lit.py search), 12 queries: miskeyed items (x2), IRT identification in Psychometrika, unidentifiability of 3PL, anchor items/common metric, nominal response model, Stocking–Lord, Kolen–Brennan, Bock–Aitkin EM, Bayes modal estimation, answer-key errors, distractor analysis. About 80 screened, 7 included.
- lit.py verify batch 3 (construct validity, Bowman & Dahl, BetterBench, Chong, UTBoost, Dickinson & Meurers, OSWorld, CROWDLAB): 8/8 verified and included.
- arXiv-search "WebArena Verified reliable evaluation web agents": 3 screened, 0 relevant.
- arXiv-search runs on general IRT/label-error queries were aborted because sibling agents had congested the export API; WebSearch was used instead.

## Anchor papers used for snowballing
- Rodriguez et al. 2021 (ACL): led to py-irt, Lalor 2016/2018/2019 and Vania 2021.
- Land & Bikel 2026 (arXiv 2605.30504). The method section was read: a forced-ceiling 4PL likelihood-ratio indicator. Its references and data pointer (evaleval/every_eval_ever) led to ATLAS and PSN-IRT.
- Zhu et al. 2025 (ABC). Its τ-bench and SWE-bench issues led to SABER/τ²-Bench-Verified, SWE-Bench+, PatchDiff, UTBoost and SWE-ABS.
- Northcutt et al. 2021 (confident learning, pervasive label errors) and Klie et al. 2023 (AED survey) led to AUM, cartography, Chong 2022, Dickinson & Meurers 2003 and CROWDLAB.

## Close-call exclusions
- Laplace-PSN-IRT (arXiv 2607.25257): an uncertainty add-on to PSN-IRT; marginal.
- Item Response Scaling Laws (ICML 2026): scaling-law estimation, not item auditing.
- SWE-Bench Pro Verified (arXiv 2609.08149) and TwinCheck (2609.26911): seen in search but not screened in depth because of the record budget. SWE-Bench Pro Verified is a candidate for a follow-up.
- Auditing Web Agent Evaluation on WebArena-Lite (NeurIPS 2026 virtual page): not verified within budget.
- "Detecting Faulty Within-Item Category Functioning With the Nominal Response Model" (Handbook of IRT Modeling 2014, DOI 10.4324/9781315736013-29): relevant to miskey detection, but CrossRef/OpenAlex give no authors or abstract.
- Detecting Compromised Items Using Information From Secure Items (JEBS 2020): item security, not keys.
- β³-IRT (Chen et al., AISTATS 2019): title lookup failed and the PMLR page was not checked within budget.
- OpenAI "SWE-bench Verified" and XLANG "OSWorld-Verified" announcements: blog posts (excluded by the brief); mentioned inside the SWE-bench and OSWorld records.
- ChaosNLI, Plank 2022 (human label variation), Liao et al. 2021, Wallach et al. 2025, AI Agents That Matter, HAL, ImpossibleBench, HOW2BENCH, Fluid Benchmarking, Leveraging LLM-Respondents for Item Evaluation: relevant leads left for budget reasons. Several of them probably belong to other themes (ambiguity, truth inference).

## Unverified
- WebArena Verified: Reliable Evaluation for Web Agents (NeurIPS 2025 virtual page 124576). Not found on arXiv or CrossRef, so dropped.
- β³-IRT: A New Item Response Model and its Applications (AISTATS 2019). lit.py matched nothing, and the proceedings page was not checked.
- "Using IRT to identify miskeyed items" style psychometric papers: no specific verifiable title found by CrossRef search.

## Totals
- Screened: about 290 (titles and abstracts across web, CrossRef, arXiv and PMLR).
- Included: 53 (raw/t2_irt_benchmark_audit.jsonl).
- Baseline candidates flagged: 10 (Rodriguez 2021, py-irt, Land & Bikel 2026, confident learning, Klie/nessie, Nahum 2025, ABA, BenchGuard, COBA, CROWDLAB).
