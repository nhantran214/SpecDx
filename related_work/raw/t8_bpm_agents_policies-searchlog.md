# Search log: t8_bpm_agents_policies

Theme: business-process and policy-executing LLM agents and their benchmarks; policy enforcement for agents; agentic BPM and LLMs for process modelling; conformance checking and model repair; text-versus-model inconsistency; DMN/Declare/compliance verification.

Dates: 2026-10-20 to 2026-10-21 (interrupted once by a usage limit and resumed from scratch files).

## Queries

### CrossRef (`lit.py search` / title verification via lit.py functions)
| Query | Hits screened |
|---|---|
| ~60 exact-title verifications of seeded leads (conformance, repair, text-vs-model, DMN, Declare, compliance, agent benchmarks' ACL versions) | ~60 x 8 candidates |
| Friedrich Mendling Puhlmann process model generation natural language text | 5 |
| Process Mining Data Science in Action van der Aalst | 6 |
| Kourani Berti Schuster van der Aalst process modeling large language models | 6 |
| ProMoAI process modeling generative AI | 6 |
| Evaluating large language models on business process modeling benchmark self-improvement | 6 |
| PM-LLM-Benchmark process mining large language models | 6 |
| large language models business process compliance checking (x2) | 11 |
| Solver-Aided Verification / Enforcing Temporal Constraints / Towards Enforcing Company Policy Adherence (venue lookup) | 9 |
| COMPASS process mining prompt optimization LLM agents | 5 |
| Interactive Resolution of Inconsistencies in Declarative Process Models | 5 |
| large language models conformance checking explain deviations | 5 |
| LLM detecting inconsistencies process model textual description | 5 |

### arXiv (`lit.py verify --arxiv`, `arxiv-search`)
About 50 arXiv IDs verified (several retried after HTTP 429). arxiv-search: PolicyGuard (1 hit), Scenario Studio (0), COMPASS (0), tau3-bench (0).

### Web search (discovery, venue confirmation, abstracts)
| Query | Results screened |
|---|---|
| LLM compliance checking business process models regulations 2024-2025 | 10 |
| ToolGuard "Towards Enforcing Company Policy Adherence in Agentic Workflows" | 9 |
| policy adherence benchmark customer service agents 2025 | 9 |
| tau2-bench verified airline retail fixed errors audit | 9 |
| SOPBench standard operating procedures | 9 |
| LLM agents policy compliance SMT solver 2025 | 9 |
| LLM inconsistency process model vs text 2024-2025 | 9 |
| process mining LLM agent execution logs conformance | 9 |
| Corea Grant Thimm inconsistency measurement follow-ups | 10 |
| LLM DMN decision tables generation/verification | 9 |
| policy-driven test scenario generation (IntellAgent-style) | 10 |
| LLM agents executing BPMN / agentic process automation | 10 |
| Agentic Benchmark Checklist (ABC) | 9 |
| runtime monitoring LLM agents LTL / Pro2Guard | 9 |
| LLM conformance checking Declare 2024-2025 | 10 |
| successor to tau2-bench 2026 | 9 |
| Progent / policy-as-code for agents | 18 |
| WorkBench COLM 2024 | 9 |
| DMN verification tool (Corea, Blatt, Delfmann) | 9 |
| Gianola & Zerva LLM+symbolic public administration | 9 |
| contradiction/ambiguity detection in agent policy documents 2026 | 9 |
| venue checks: tau-bench ICLR 2025, WorkArena++ NeurIPS 2024, TheAgentCompany, ST-WebAgentBench ICLR 2026, GuardAgent/ShieldAgent ICML 2025, Agent-C, AgentSpec code | ~50 |
| abstract lookups for 20 BPM classics (research portals) | ~90 |

Official pages fetched: proceedings.iclr.cc (tau-bench, ST-WebAgentBench), proceedings.mlr.press (WorkArena, GuardAgent, ShieldAgent), proceedings.neurips.cc (WorkArena++, ABC), aclanthology.org (NeMo Guardrails, ABCD, COLING 2025 compliance, COLING 2018 NLP-BPM), eprints.gla.ac.uk (WorkBench at COLM 2024), research.tue.nl (van der Aa BPM 2015 and IS 2017), neurips.cc and icml.cc virtual pages (PolicyGuard, Scenario Studio), homes.cs.washington.edu (Solver-aided, confirmed preprint). Missing CrossRef abstracts were taken from a Semantic Scholar batch query by DOI (15 found) or from research-portal pages.

## Anchor papers used for snowballing
- van der Aa, Leopold, Reijers, IS 2017 (text-vs-model) led to the BPM 2015 predecessor, behavioural ambiguity (BPM 2016, IS 2018), declarative extraction (CAiSE 2019) and Sànchez-Ferreres CAiSE 2017.
- Fahland & van der Aalst, IS 2015 (model repair) led to impact-driven repair (TOSEM 2016) and generalized conformance (BPM 2016).
- Di Ciccio et al., IS 2017, and Corea et al., BPM 2022, led to the IS 2024 paraconsistent follow-up, the ASP framework and MINERful.
- Calvanese et al., BPM 2016 (DMN) led to the IS 2018 journal version, Semantic DMN (TPLP) and DMN disambiguation (BIS 2018).
- τ-bench / τ²-bench led to ToolGuard, Solver-aided verification, Agent-C, PolicyGuard, τ-break, ABC, the validity audit and IntellAgent.
- AgentSpec (ICSE 2026) led to ProbGuard (ASE 2026), whose abstract cites AgentSpec.

## Close-call exclusions
- Coverage-Aware Test Generation for Conversational AI Agents (Scenario Studio; ICML 2026 workshop poster): an IntellAgent-style successor, but only a workshop virtual page exists, with no paper record or arXiv.
- Progent: Securing AI Agents with Privilege Control (arXiv 2504.11703): a policy language for tool privileges, but aimed at prompt-injection defence (excluded per the brief).
- LegalWiz/ContraGen (arXiv 2510.03418): contradiction-injection benchmark for legal RAG, outside this theme.
- Enhancing Legal Compliance and Regulation Analysis with LLMs (arXiv 2404.17522): requirements-engineering theme.
- Metamon (LLM4Code 2025), documentation-vs-behaviour inconsistency: belongs to the spec-defects theme.
- Orchestrating LLMs to support medical process conformance checking (Neuroscience Informatics 2026): weak venue, narrow domain.
- Can conformance checking monitor process compliance? A taxonomy of regulations (Process Science 2026): tangential.
- Interactive Resolution of Inconsistencies in Declarative Process Models (Nagel & Delfmann 2026, LNISO): low-tier venue; the Di Ciccio and Corea lines cover it.
- Sadiq et al. BPM 2007; Governatori et al. EDOC 2006; Elgammal et al. SoSyM; Maggi et al. BPM 2011 (colored automata); Batoulis & Weske soundness (2017); Awad et al. BPMN-Q (BPM 2008): verified, but these are compliance and Declare foundations that the included surveys cover. Abstracts were not all accessible.
- Fahland & van der Aalst BPM 2012 (superseded by IS 2015); Replaying history (WIREs 2012) and Mannhardt et al. Computing 2016 (redundant with the included conformance papers); Conversational Process Modelling (BPM Forum 2023); COLING 2018 NLP-for-BPM position paper.
- τ²-Bench-Verified (github.com/amazon-agi/tau2-bench-verified; MIT; 55 stars): a dataset release with no paper. It is relevant as a real-defect source, but it is not a paper, so it is not recorded. Note it for the evaluation section.
- τ³-bench: only leaderboard and blog mentions, no paper.

## Unverified items (dropped)
- Gianola & Zerva, LLM + RAG + symbolic reasoning position paper for public-administration processes: not found in CrossRef or arXiv.
- Corea, Blatt, Delfmann, "A Tool for Decision Logic Verification in DMN Decision Tables" (BPM 2019 demo, CEUR): no DOI and no proceedings page fetched.
- Laurson & Maggi, "A Tool for the Analysis of DMN Decision Tables" (BPM 2016 demo, CEUR): same.
- Corea & Delfmann, "Detecting Inconsistencies in Business Rule Bases" (2017): CrossRef nearest match was a different paper.
- Kuhlmann, Corea, Grant, "An ASP-Based Framework for Solving Problems Related to Declarative Process Specifications": no exact match. The nearest was "Non-Automata Based Conformance Checking of Declarative Process Specifications Based on ASP" (BPM Workshops 2023), which was not included.
- Process Mining Manifesto (BPM Workshops 2011): no CrossRef match.
- Kourani et al., "Process Modeling with Large Language Models" (BPMDS 2024): no CrossRef match. Its SoSyM 2025 extension and ProMoAI (IJCAI 2024) are included instead.

## Notes on verification
- 85 records. Peer-reviewed items are verified by CrossRef DOI or an official proceedings page. Preprints are verified by the arXiv API.
- Corrections found during verification:
  - The Agentic BPM manifesto (arXiv 2603.18916) is published in Information Systems 2026 (doi 10.1016/j.is.2026.102738).
  - ProbGuard (arXiv 2508.00500, formerly Pro2Guard) is in ASE 2026.
  - AgentSpec is in ICSE 2026.
  - ST-WebAgentBench is in ICLR 2026.
  - Agent-C's arXiv title is "Enforcing Temporal Constraints for LLM Agents".
  - CrossRef's booktitle for Friedrich et al. (CAiSE 2011) is wrong, so the BibTeX was rebuilt by hand.
- `read_level` is "metadata only" for 4 records whose abstracts could not be read (van der Aalst book, Friedrich 2011, Vidgof et al. 2023, Grohs et al. 2024).
- Several GitHub checks failed because the shared API budget ran out (HTTP 403). The repositories listed in records were checked after the reset. The NeMo Guardrails repository is not recorded because the paper page gives no link.

## Totals
- Screened: about 560 candidates (search hits plus snowballed references and title lookups).
- Included: 85 (23 high, 40 medium, 22 low relevance); 8 are flagged as baseline candidates.
