# Search log: t9_agent_failure_attribution (2026-10-20/21)

## Tooling notes
- `lit.py verify` worked for the first ~10 records. Afterwards export.arxiv.org returned HTTP 429 for the shared host, so arXiv records were verified from the canonical abs page `https://arxiv.org/abs/<id>` (citation_* meta tags, comments, journal-ref) with a small fallback script. These records carry `verified_by: "arxiv (arxiv abs page ... export.arxiv.org API returned HTTP 429)"`.
- DOIs were verified with lit.py's own `crossref_doi` + `doi_bibtex` functions (skipping the arXiv lookup that stalled on 429).
- ICLR 2025/2026 records: proceedings.iclr.cc listing + official `-Bibtex.bib`. ICML 2025/2026: proceedings.mlr.press v267/v306 pages. TMLR: jmlr.org/tmlr bib. ACL/EACL: aclanthology.org .bib (+ CrossRef DOI). Workshop papers: iclr.cc / neurips.cc virtual pages (OpenReview blocked by a challenge page).
- GitHub API was rate-limited for ~1 h; repos were first confirmed by HTTP 200 on github.com and from the paper's own HTML/abs links, then `lit.py github` filled `code_info` after reset. No repository was guessed: every code_url appears in the paper, abstract, or proceedings page.

## Queries (source -> hits screened)
- WebSearch: "automated failure attribution LLM multi-agent Who&When follow-up" (9); "failure attribution counterfactual replay AgenTracer ECHO A2P GraphTracer" (10); "survey failure attribution LLM MAS 2026" (9); "ECHO hierarchical error attribution" (10); "FAMAS / A2P / RAFFLES" (27); "Abduct Act Predict" (9); "Where LLM Agents Fail AgentDebug" (9); "empirical failures SE agents SWE-bench taxonomy" (9); "agent benchmark broken tasks automatic detection audit" (9); "tau2-bench verified repaired tasks" (9); "is it the model or the benchmark / infrastructure errors" (9); "Verifying the Verifiers AutoTriage" (10); "web agent failure taxonomy" (10); "tool-use failure taxonomy" (10); "reward hacking coding agents EvilGenie" (9); "specification gaming reasoning models / monitoring misbehavior" (10); "survey evaluation of LLM agents / agent-as-a-judge survey" (10); "Illusion of Progress WebJudge" (9); "Towards a Science of AI Agent Reliability" (9); "agent benchmark noise infrastructure reproducibility" (10); "SWE-bench Verified flawed tests" (9); "verifier blind spots / when the verifier is wrong" (19); "Who Verifies the Agents NeurIPS 2026 workshop" (27); "detect broken benchmark tasks cross-model failure consensus" (9); "AI Agents That Matter TMLR" (10); "AgentRewardBench COLM" (9); "tau-bench ICLR proceedings" (10); "Agent-as-a-Judge PMLR" (9).
- CrossRef (`lit.py search`, from 2024): "failure attribution multi-agent systems" (12); "agent trajectories failure taxonomy LLM" (12); "LLM agent benchmark task defects" (12); "coding agent failures empirical study issue resolution" (12); UTBoost / Solved Issues / SWE-Bench+ / TAR trajectories / Agent-as-a-Judge / AI Agents That Matter (30); 12 title checks for published versions of preprints (24).
- Curated list github.com/yzhao062/awesome-auditable-ai (45 entries screened; used for discovery only, all included items re-verified).
- Proceedings listings grepped: ICLR 2025 + 2026 (titles on failure/attribution/judge/reward hacking/benchmark validity, ~120 screened), PMLR v306 ICML 2026 (~100 screened), ACL 2026 event page (~30 screened).
- arxiv.org title search page for arXiv ids of UTBoost, Solved Issues, SWE-Bench+, TAR, What's in a Benchmark.

## Anchor papers for snowballing
Who&When (ICML 2025), MAST (NeurIPS 2025), TraceElephant related work (ACL 2026), LIFE survey (arXiv 2605.14892), ABC (NeurIPS 2025), BenchGuard / ABA (2026), awesome-auditable-ai list.

## Close calls excluded (one line each)
- GraphTracer (arXiv 2510.10581): withdrawn by the authors for a methodological error.
- What's in a Benchmark? SWE-Bench in APR (ICSE-SEIP 2026, 10.1145/3786583.3786904): leaderboard demographics, not noise or defects.
- Intention-behavior consistency-based failure attribution (ESWA 2026, 10.1016/j.eswa.2026.133044): verified on CrossRef but abstract not obtainable; another agent-step attribution method.
- EMFA (2610.11600), DCFA (2609.04749), Detect Before You Attribute (2608.29646), FALAT (2606.00765), causal-inference attribution (2509.08682), Adaptive Influence Graphs (2608.24361), Who Broke the System?/AgentLocate (2607.07989, COLM 2026), OAT "Tracing Agentic Failure from the Flow of Success" (2607.12747, NeurIPS 2026): further Who&When-style agent-step attribution methods; representative ones already included.
- Span-level deep-research error localisation (2606.02060), SearchAuditor, SAFARI, GRADE, CUADebug: domain-specific trace debugging, agent-side only.
- TelemetrySuffBench (2608.07899), CatchBench (2608.22808), When Is an Agent Evaluation Over? (2608.14940): diagnosis/observability benchmarks, tangential.
- Beyond pass@1 reliability framework (2603.29231), behavioural consistency as uncertainty (2602.11619, ICML-W), ReliabilityBench, Long-Horizon Task Mirage (2604.11978): reliability metrics; Towards a Science of AI Agent Reliability included instead.
- ToolFailBench (2607.04686), Beyond the Leaderboard synthesis (2607.05775), ToolScan, EconWebArena, BrowserArena, WebSuite: agent-side failure taxonomies for tool/web agents; lower priority under the size budget.
- Agent-as-a-Judge survey (2601.05111), Evaluation and Benchmarking of LLM Agents survey (2507.21504): surveys; Yehudai et al. (ACL Findings 2026) kept.
- An Illusion of Progress? (COLM 2025, 2504.01382), Autonomous Evaluation and Refinement of Digital Agents (2404.06474): LLM-judge evaluators for web agents; AgentRewardBench kept as the judge-reliability representative.
- Demonstrating specification gaming in reasoning models (2502.13295), Monitoring Reasoning Models for Misbehavior (2503.11926), Reward Hacking Benchmark with tool use (2605.02964, ICML 2026 per comment), Recontextualization (ICML 2026): reward-hacking context; ImpossibleBench, EvilGenie, SpecBench, TRACE kept.
- Passing Without Repairing (SRE verifier blind spots) and Who Verifies the Agents? case series (NeurIPS 2026 workshop): single-author workshop papers; the AppWorld/WorkArena verifier paper kept as representative.
- Noisy but Valid (ICLR 2026), Diagnosing LLM-as-a-Judge via IRT (ICML 2026), Dependence-aware label aggregation via Ising models (ICML 2026): judge statistics, belong to the truth-inference / IRT themes.
- RCAgent and cloud-incident RCA work: root-cause analysis of IT incidents, not of agent task failures.
- OpenAI "Why we no longer evaluate SWE-bench Verified", Anthropic "infrastructure noise", Epoch AI review, IBM/MAST blog, Atla tau-bench post: blog posts, excluded by the brief.

## Unverified
- None recorded. tau2-Bench-Verified (amazon-agi/tau2-bench-verified) is a dataset release without a located paper; it is mentioned in the tau2-bench record but not included as a paper.

## Totals
- Screened: about 560 candidate hits (web, CrossRef, proceedings listings, curated list), about 110 read at abstract level.
- Included: 59 (peer-reviewed 30, including 2 workshop papers; preprints 29).
