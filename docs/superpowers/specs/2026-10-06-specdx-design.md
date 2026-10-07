# SpecDx — Design Spec

**Date** 2026-10-06 · **Status** draft, awaiting review · **Paper direction** D2 ("the specification is the bug") · **Artefact base** TonomyX (`/home/ethan/AILab/TonomyX`), vendored read-only

Working title: *When the Specification Is the Bug: Probabilistic Diagnosis of Executable Business Specifications with Populations of LLM Interpreters*

---

## 0. Summary

Executable business specifications (a locked "canon" of schema, data, invariants, guardrails and expected outcomes, plus the natural-language requirement documents it was derived from) are trusted as ground truth by the systems and benchmarks built on them. They contain defects. Running TonomyX against its own canon exposed several, for example `SO-0031` staying `Chờ duyệt` after its approval, and a reconciliation function that refuses an input the schema allows (decision R12). τ²-bench needed a whole "Verified" release to repair its tasks.

SpecDx turns a population of fallible LLM agents into an automatic instrument for finding such defects. Each agent run is a sample of how a reasonable executor reads the specification. A probabilistic model separates *specification defects* (every competent interpreter deviates at the same clause) from *agent errors* (interpreter-specific, scattered). Logical tests derived from the satisfaction relation type each defect as a contradiction, a hidden requirement, an ambiguity, or a weak check. An active loop then synthesises the scenarios most likely to expose the remaining defects. Evaluation is fully automatic: seeded defects from specification mutation operators, real defects recorded in TonomyX's history, and real defects repaired by τ²-Bench-Verified.

---

## 1. Intent and constraints

### 1.1 What was asked and what was decided

| Item | Source |
|---|---|
| A Q1 journal paper based on the TonomyX code | User |
| The idea must differ from the FAA paper, as much as possible and creatively | User |
| Direction D2: agents judge the specification, rather than the specification judging the agents | User choice among H1, H2, D1, D2, D3, D5 |
| Fully automatic on the local machine. No human experts or annotators | User |
| The paper's weight is a new method and new algorithms with clear novelty. Compute volume is not the concern | User |
| Core approach A: probabilistic diagnosis (statistics and logic combined), with B (statistics only) and C (symbolic only) as baselines | User choice |
| Two subjects: TonomyX canon (S1) and τ²-bench airline + retail (S2) | Approved in design Part 3 |
| Target venue ACM TOSEM; fall-backs IST, JSS, Requirements Engineering | Approved in design Part 4 |

### 1.2 Constraints

1. **Nothing in TonomyX or its canon is modified.** Both are vendored byte-identical and frozen. Mutated specifications are copies outside the canon directory, selected through `TONOMYX_CANON_PACK`.
2. **No dependency on FAA.** Nothing is imported from `/home/ethan/AILab/FAA` or `aebench`.
3. **Local models only.** Ollama on one RTX 5070 Ti (16 GB). No hosted APIs.
4. **No human judgement in any metric.** Every label comes from a mutation operator, from TonomyX's recorded history, or from τ²-Bench-Verified.
5. Code, comments and documents in English. Vietnamese appears only in canon data and in the S1 Tell corpus.

### 1.3 Separation from FAA

| | FAA | SpecDx |
|---|---|---|
| Stance towards the canon | Trusted as ground truth | Under diagnosis |
| Object evaluated | Enforcement architecture (layer ablation) | The specification itself |
| Stimulus | Attacks, prompt injection, compositional violations | Ordinary execution and synthesised scenarios |
| Community and venues | IS/BPM: *Information Systems*, DSS, BISE | SE/RE: TOSEM, IST, JSS, REJ |
| Shared material | TonomyX as a cited artefact | Same, nothing else (no shared runs, data, RQs or code) |

The cover letter states the relationship and cites FAA.

### 1.4 Success criteria (pre-registered hypotheses)

- **H1 (effectiveness).** On seeded defects, SpecDx's typed detection F1 exceeds the best baseline for each defect type, on S1 and on S2. Tested with paired tests per mutant and Holm correction.
- **H2 (real defects).** SpecDx rediscovers the in-scope real defects of S1 (§4.4). On S2, its recall over τ²-Bench-Verified repairs exceeds that of static LLM review (B1).
- **H3 (components).** Removing any component (A1–A7, §4.5) lowers typed F1 or raises runs-to-detection.
- **H4 (efficiency).** Information-directed synthesis with escalation reaches a given detection level with fewer interpreter runs than random scenario selection. Measured as the area under the detection-versus-runs curve.
- **H5 (population).** At a fixed run budget, adding model families improves detection more than adding seeds (the empirical face of Proposition 2).

### 1.5 Non-goals

- Hosted frontier models, human studies, or expert labelling.
- Repairing the specifications. SpecDx reports defects with witnesses; it does not rewrite the canon.
- Scoring documentation-only inconsistencies that no behaviour can expose (§4.4). They are reported qualitatively.
- Any change to TonomyX enforcement, and any adversarial or attack scenario.

---

## 2. Problem formulation

### 2.1 Executable business specification

An executable business specification is **S = ⟨T, C, Π⟩**.

- **T (Tell)** is everything an executor is told:
  - For S1: the three requirement documents in `TonomyX/demo/` (BA 97 pp., demo description 46 pp., SAD 18 pp.), the KB articles `kb/KB-0001..0003.md`, the principles `nguyen_tac` in `canon.meta.json`, the tool catalogue, the schema, `de_scope`, and the refusal messages tools return at run time.
  - For S2: `policy.md` and the tool descriptions.
  - **TonomyX's hand-written playbooks are excluded from T.** They are a human interpretation of the canon. Including them would test the playbooks, not the specification.
- **C = {c₁ … cₘ} (Check)** is every clause that can be evaluated on a behaviour:
  - For S1: the 42 invariants in `invariants.json`, the expectations in `scenarios.json`, the golden state `seed_operational.sql`, the guardrail triggers and SQL policy functions, the event vocabulary, the scope table, and the blank-column list `cot_de_trong_khong_canon`.
  - For S2: each task's evaluation criteria (§4.1).
- **Π** is the scenario space. For S1, π is a parameter vector (ordered quantity, received quantity, requested discount, GRNs per PO). Facts that only the environment writes and no interpreter ever touches (defective units, late days, number of order lines) are fixed at their canon values, because varying them cannot change any interpreter's behaviour (§11). The canon fixes three points: `happy`, `discount8`, `grn12`. For S2, Π is the given task set.

### 2.2 Behaviours and spec-induced equivalence

A **behaviour** b of one run is the triple (committed effects, refused tool calls, final claim). Committed effects are the database rows written and the event chain.

**Equivalence ≡_S** compares behaviours only on what the specification fixes. The specification defines the relation, not the researcher. For S1 each schema column is classified by a rule derived from `schema.json` and `canon.meta.json`:

| Class | Rule | Treatment in ≡_S |
|---|---|---|
| Locked | Primary and foreign keys, status columns (`trang_thai`, `doi_soat`, …), integer quantity columns, approval rows, and event `(kind, ref, actor)` | Must be equal |
| Blank | Columns listed in `cot_de_trong_khong_canon` | Must both be null |
| Ignored | Serial ids, timestamps, free text (`apr.noi_dung`, `evt.basis`) | Not compared |

A unit test pins this classification for every column of the 26 tables.

### 2.3 Interpreters

An **interpreter** I maps (T, π) to a behaviour b at random. The ideal interpreter I\* produces exactly **Adm_T(π)**, the set of behaviours T permits. Adm_T is not observable, and the method estimates it (§3.2).

### 2.4 Defect taxonomy

Each type is defined purely by satisfaction on a scenario π:

| Type | Definition | Location reported | Known real instance |
|---|---|---|---|
| **Contradiction** | No b satisfies C(π) | A minimal unsatisfiable subset M ⊆ C | R12: the schema allows several GRNs per PO; `fn_reconcile` refuses that input |
| **Hidden requirement / mis-key** (C stricter than T) | ∃ b ∈ Adm_T(π): b ⊨ C∖{c} and b ⊭ c | Clause c | `SO-0031` stays `Chờ duyệt` after `APR-0102` is approved |
| **Ambiguity** (T too loose) | Adm_T(π)/≡_S has two or more classes | The locked fields that separate the classes, mapped to sentences of T | Seeded only on S1; "Evaluation ambiguity" repairs on S2 |
| **Weak check** (C looser than T) | ∃ b ∉ Adm_T(π): b ⊨ C(π) | The rule of T that no clause guards | Seeded |

The types can co-occur. When T is ambiguous and C picks one reading, both Ambiguity and Hidden requirement are reported.

**Proposition 1 (soundness under ideal interpreters).** If every observed behaviour is drawn from Adm_T(π), then:
- a Hidden-requirement flag at c implies that C is stricter than T at c;
- two or more observed classes imply Ambiguity;
- both tests become complete as the samples cover Adm_T(π).

Two of the types are outside the scope of Proposition 1:
- **Weak check** can only be witnessed by a behaviour outside Adm_T, so an ideal interpreter cannot expose it. Its detection relies on fallible interpreters and is sound given a correct classification of non-admissible classes (§3.2).
- **Contradiction** concerns C alone. Sampling cannot prove unsatisfiability. Candidates are reported as unrefuted conflicts with a confidence score, and refutation is sound (Proposition 3).

### 2.5 Clause lifting

Most S1 invariants state facts of CASE-0031 as literals (`po-15: po.sl_dat = 15`). **Lifting** rewrites a clause into scenario-parametric form by anti-unifying its literals with the parameters of the canon scenarios. For example, `sl_dat = 15` becomes `sl_dat = π.ordered`, and `sl_nhan = 12` becomes `sl_nhan = π.received`.

- A literal is lifted only if it matches the corresponding parameter on all three canon scenarios.
- A clause with a literal that does not lift stays **case-anchored** and is checked only on the canon scenario it came from.
- Lifting is done once, stored as data, and pinned by a test over all 42 invariants.

The lifted check set is written C'.

---

## 3. The SpecDx algorithm

### 3.1 Data types

| Type | Fields |
|---|---|
| `Clause` | `id`, `source` (invariant / expectation / golden / guardrail / scope / vocabulary / blank / τ²-criterion), `lifted` expression, `case_anchored`, `evaluate(behaviour) → bool \| NA` |
| `Scenario` | `id`, parameter vector, path of its working specification copy |
| `Interpreter` | `model`, `family`, Tell `variant`, `seed` |
| `Behaviour` | run key, interpreter, scenario, step, effects, refused calls, claim, status (`ok` / `error`), V(b) ⊆ C', class key [b] |
| `Posterior` | θᵢ, family effects u_f, per-step distribution over A_s, typed defect scores |
| `Defect` | type, location, witness scenario(s), witness behaviour keys, posterior score |

### 3.2 Component 1: set-valued latent-class model

Classic Dawid–Skene assumes one true label per item. Here an ambiguous T admits several correct outcomes, so the latent truth for each scenario step s is a **set** A_s of admissible classes.

- **Competence:** logit θᵢ = μ + u_f(i) + vᵢ, with Gaussian priors on u_f and vᵢ. The family effect u_f absorbs errors that models of the same family share.
- **Generation:** P([b] = k | A_s, θᵢ) = θᵢ · w_{s,k} · 𝟙[k ∈ A_s] + (1 − θᵢ) · ε_{s,k}. Here w is the preference weight over admissible classes and ε is the error distribution over all classes.
- **Cross-scenario borrowing:** θᵢ is shared across all scenarios and steps. A specification defect makes every competent interpreter deviate at the same clause; an agent error is specific to the interpreter. This separation is the core of the method.
- **Inference:** EM with MAP estimates of μ, u_f and vᵢ. In the E-step, A_s is enumerated exactly over the K_max most frequent classes of each step, and the remaining classes are pooled into "rare". Posterior uncertainty is estimated by bootstrap over interpreters.
- **Outputs:** P(k ∈ A_s) for each class, and an admissibility weight for each behaviour.

### 3.3 Component 2: contradictions through MUS/MCS duality

If b ⊨ C'∖V(b), then V(b) is a **correction set**. By the hitting-set duality (Liffiton & Sakallah 2008), every MUS is a minimal hitting set of the minimal correction sets. SpecDx therefore treats every LLM behaviour as a free sample of a correction set:

1. Collect {V(b)}, each weighted by the behaviour's admissibility weight times θᵢ.
2. Enumerate *soft* minimal hitting sets H up to size 3. H must hit at least a fraction (1 − τ) of the weighted mass. PySAT `Hitman` is used, or brute force below 100 clauses.
3. Classify each H:
   - |H| = 1: passed to refutation (§3.6) on π. If a state in π satisfies the clause, it is a **Hidden requirement**; if none is found, it is a **Contradiction** of size 1 (§11, item 4).
   - |H| ≥ 2, and every clause of H is satisfied individually by some behaviour but never jointly: a **Contradiction** candidate, passed to refutation (§3.6). If refuted, it is dropped.

This needs no full SAT encoding of the specification.

### 3.4 Component 3: typed scores and localisation

| Score | Definition | Localisation |
|---|---|---|
| Hidden(c) | P(∃ k ∈ A_s : V(k) = {c}) | Clause c |
| Ambig(s) | P(\|A_s\| ≥ 2) | Locked fields that differ between the classes of A_s, then T sentences via Component 6 |
| Weak(s) | P(∃ k ∉ A_s : V(k) = ∅) | Locked fields where k departs from A_s, mapped to the T rule that governs them, which no clause of C' guards |
| Contra(M) | Weighted support of hitting set M. Set to zero when a refutation succeeds; each failed refutation attempt is recorded with the defect | Clause set M |

### 3.5 Component 4: information-directed scenario synthesis

- **Scenario space (S1).** The domain of each parameter comes from schema types, plus **boundary values** taken from the literals of lifted clauses:
  - discount cap 5% gives 4, 5 and 6%;
  - ordered quantity 15 gives received quantities 14, 15 and 16;
  - GRNs per PO ∈ {1, 2}.

  A scenario enters a run as task inputs and environment events. It needs no copy of the specification; only mutants do.
- **Acquisition.** Choose π to maximise the expected information gain about the defect hypotheses of §3.4. Exact EIG is intractable, so the method uses **Thompson sampling** with a Normal–Gamma posterior per scenario *cell*. A cell is the pair (set of active lifted clauses, boundary flags). The reward is the realised KL divergence between the posteriors after and before the run.
- **Sequential escalation.** Each new π first runs a **pilot**: one interpreter per family (the family's highest-θ model, variant 0, seed 1; before any θ exists, the family's largest model). It is escalated to the full committee (all models × 3 variants × 3 seeds) only if the pilot yields at least one of:
  - a correction set not seen before;
  - two or more classes among the pilot at some step.

  Class novelty across scenarios is not a trigger: a class key embeds the scenario's quantities, so every new scenario would look novel.
- **Scope.** Synthesis applies to S1. On S2 the task set is fixed.

### 3.6 Component 5: refutation of contradiction candidates

For each candidate M, the method searches for a state that satisfies all of M along two routes:

1. **Directed probe.** T is augmented with the goal "reach a state satisfying every clause of M", written from the clause descriptions. The three highest-θ interpreters each make 3 attempts.
2. **Symbolic repair.** Starting from the behaviours that violate the fewest clauses of M, a local search over locked fields runs for at most 200 clause evaluations. Each step uses the real invariant evaluator and the guardrail functions on PostgreSQL inside a savepoint, then rolls back.

**Proposition 3 (sound refutation).** If either route returns a witness w and every clause of M evaluates true on w, then M is satisfiable and is not a MUS. Confirmation is only probabilistic. The report gives the number of failed refutation attempts.

### 3.7 Component 6: ddmin localisation of ambiguity in T

When Ambig(s) is high, the method runs delta debugging (ddmin) over the sentences of T relevant to step s: the sentences of the top 20 retrieved chunks. Each ddmin probe removes a sentence subset and re-runs the pilot committee. The test passes while A_s still splits. The output is a minimal sentence set that causes the ambiguity, at a cost of O(log |T_s|) pilot rounds in the typical case. This is the result a specification author can act on. On S2, ddmin runs over the sentences of `policy.md`.

### 3.8 Main loop

```
SpecDx(S = ⟨T, C, Π⟩, population 𝓘, budget B)
  C' ← Lift(C, Π)                                   # §2.5
  queue ← canon scenarios (S1) | task set (S2)
  while B > 0 and max EIG ≥ ε:
      π ← queue.pop() or ThompsonEIG(Π)             # §3.5
      run pilot on π; if escalation trigger: run full committee
      update Posterior (EM)                          # §3.2
      candidates ← SoftMHS({V(b)})                   # §3.3
      for each contradiction candidate M: Refute(M)  # §3.6
      for each step s with high Ambig(s): ddmin(T_s) # §3.7
  return ranked typed Defects with witnesses         # §3.4
```

The loop is anytime: stopping at any point yields a report. The headline efficiency figure is runs-to-detection.

### 3.9 Theory

- **Proposition 1:** §2.4.
- **Proposition 2 (identifiability).** Assume errors are conditionally independent across families given (A_s, θ), and that mean family competence is at least ½ + γ on defect-free steps. Then the estimate of A_s is consistent, and the probability of misclassifying a step is at most exp(−2·n_f·γ²), where n_f is the number of **families**, not the number of runs. The proof reduces to a family-level weighted vote and Hoeffding's inequality, following the Dawid–Skene consistency literature. Test H5 checks the consequence.
- **Proposition 3:** §3.6.

The paper gives full proofs for 1 and 3 and a proof sketch with precise assumptions for 2.

### 3.10 Default hyper-parameters

All defaults are tuned only on the **development split** of the mutants (20%) and frozen before the test split (80%) is run.

| Parameter | Default |
|---|---|
| Temperature / seeds | 0.7 / {1, 2, 3} |
| Tell variants | 3. Variant 0 is the original. Variants 1–2 paraphrase only the standing instruction and the task statement; retrieved corpus text stays verbatim, so seeded ambiguity is not washed out. Variants are generated once by `qwen3:14b`, frozen, and published |
| Retrieval | BM25 over 300-word chunks, k = 5 per query |
| Prior P(\|A_s\| = 1) | 0.9 |
| K_max enumerated classes per step | 6 |
| Soft hitting tolerance τ / maximum MHS size | 0.05 / 3 |
| Refutation | 3 interpreters × 3 attempts; at most 200 clause evaluations for repair |
| Stopping threshold ε on EIG | 0.01 nats |
| Ambig(s) threshold that triggers ddmin | 0.5 |

---

## 4. Evaluation design

### 4.1 Subjects

| | **S1 — TonomyX CASE-0031** | **S2 — τ²-bench airline + retail** |
|---|---|---|
| Language | Vietnamese | English |
| T | BA + demo description + SAD + KB + `nguyen_tac`, served through a `search_spec` retrieval tool | `policy.md` + tool descriptions |
| C | Canon clauses (§2.1), executed on PostgreSQL through TonomyX's nine gated tools | Each expected action (name and arguments) is a clause. Each field of the expected final database that the expected actions touch is a clause. Each `communicate_info` string is a clause. NL assertions are excluded because they would need an LLM judge |
| Π | Synthesised (§3.5) | The given tasks |
| ≡_S | §2.2 | Database fields touched by any expected action, plus the multiset of actions |
| Environment | SpecDx orchestrator (§6.3). Human gate decisions are simulated deterministically from the authority rules | τ² harness; user simulator fixed to `qwen3:14b` at T = 0 |
| Real defects | §4.4 | Tasks changed between the original τ²-bench and τ²-Bench-Verified (MIT), mapped to types in §4.4 |
| Human reference | — | Findings of the manual audit in *Policy Loopholes in Agent Evaluation* (arXiv:2609.14400), compared, not re-run |

### 4.2 Interpreter population

Checked on 2026-10-06: among the installed models, only the qwen family (`qwen3:14b`, `qwen2.5` 3b/7b/14b) and `llama3.1:8b` support native tool calling. `llama3:8b`, `exaone3.5` and `solar` do not.

- **Requirement:** at least 4 families with native tool calling, and at least 2 models per family where VRAM allows.
- **Candidates to pull locally:** `mistral-nemo:12b` (Mistral), `granite3.3:8b` (IBM), `phi4-mini` (Microsoft), `command-r7b` (Cohere).
- **Selection:** the final pool is fixed by a capability pilot, a neutral tool-call smoke task run on S1 and S2.
- Families without native tool calling are excluded; no text-protocol adapter is built.
- Vietnamese ability varies by model. That variation is absorbed by θ, which is the point of the model.
- Model digests are recorded with every run.

### 4.3 Seeded defects: specification mutation operators

Operators touch only artefacts the running system reads, so each mutation is executable. `policy.rego` is excluded because its rules are mirrored in frozen Python code, so a mutation there would never take effect. Every operator emits a **kill witness**: a behaviour proving the defect is real and can be activated. This removes the equivalent-mutant problem.

| Type | S1 operators | S2 operators |
|---|---|---|
| Contradiction | Add an invariant that conflicts with an existing row-level `eq` invariant (same row and column, different value); add a guardrail trigger, `BEFORE INSERT OR UPDATE`, that forbids a status the golden state requires | Make two expected actions mutually exclusive under the database (cancel and modify the same reservation) |
| Hidden / mis-key | Change a locked status value in the golden state without support in T; add an invariant demanding an event that T never motivates (`PAID` for `PAY-0031`) | Change an expected argument to a value the policy does not imply |
| Ambiguity | Delete a disambiguating sentence of T ("mức tối đa 5%"); add a KB article with an overlapping effective date and a different cap; replace a number in a sentence of T with a vague phrase ("mức hợp lý") | Delete or vague-ify a policy sentence that decides the expected action |
| Weak check | Remove a column from the blank list *and* delete the invariant that guards it (`pay.so_tien`); delete a golden event an interpreter is told to log (`FLAGGED ITEM-0012`) | Delete the database-field clause guarding a policy rule |

Three operators considered earlier were dropped (§11): revoking a tool in `de_scope` shows up as a Hidden requirement rather than a Contradiction, because symbolic repair writes state directly; dropping the `Cần kiểm` condition from `trg_guard_pay` cannot be activated, because no scenario pays; relaxing an `eq` invariant to `in` is an equivalent mutant, because the golden-state clause still pins the value.

- **Sampling:** application sites are enumerated per operator, giving about 150–250 mutants for S1 and 80–150 for S2. They are split 20/80 into development and test sets, stratified by type.
- **Differential scoring:** a mutant is scored only on flags present in the mutant's report and absent from the base specification's report.

### 4.4 Real defects

| Id | Subject | Defect | Expected type | Scored? |
|---|---|---|---|---|
| R-SO | S1 | `SO-0031` remains `Chờ duyệt` after `APR-0102` is approved, while `PO-0015` moves to `Đã duyệt` | Hidden requirement / mis-key | Yes |
| R-12 | S1 | The schema allows several GRNs per PO; `fn_reconcile` raises `CardinalityViolation` on that input. Visible only outside the three canon scenarios | Contradiction | Yes (direct test of Component 4) |
| R-DE | S1 | `24_de.csv` declares 15 digital employees; `de_scope.csv` grants capability to 9 | Hidden requirement, where T assigns work to an unscoped employee; otherwise out of behavioural scope | Yes, where T assigns such work |
| R-DOC | S1 | The canon README says 33 invariants while `invariants.json` has 42; the README's load order differs from `canon.meta.json` | Documentation-only | No; qualitative check of B1 only |
| R-τ² | S2 | Every task changed by τ²-Bench-Verified. *Policy compliance* and *Database accuracy* repairs map to Hidden requirement; *Logical consistency* maps to Contradiction; *Evaluation ambiguity* maps to Ambiguity | As mapped | Yes |

### 4.5 Baselines and ablations

| Id | Baseline | Represents |
|---|---|---|
| B1 | Static LLM review of T + C: map-reduce over chunks, self-consistency over 5 samples, each model in the pool | LLM-based requirements review |
| B2 | Best single interpreter (highest pilot θ) × 3 seeds; every violated clause flagged as Hidden | Naïve differential testing against a reference |
| B3 | 2PL IRT on the interpreter × clause matrix; clauses with negative discrimination flagged | *Auditing LLM Benchmarks with IRT* (arXiv:2605.30504) |
| B4 | Steps whose inter-family disagreement entropy exceeds a threshold flagged as ambiguous | *Stress-Testing Model Specs* (arXiv:2510.07686); behavioural consistency (arXiv:2602.11619) |
| B5 | Symbolic repair (§3.6) over a scenario grid starting from the golden state, with no interpreters | Check-only consistency analysis (approach C) |

| Id | Ablation |
|---|---|
| A1 | Single-truth Dawid–Skene in place of set-valued A_s |
| A2 | No family random effect |
| A3 | Single-clause tests only, without MUS duality |
| A4 | Random scenario selection in place of Thompson EIG |
| A5 | Always run the full committee, with no escalation |
| A6 | No refutation |
| A7 | Lexical-overlap localisation in place of ddmin |

### 4.6 Research questions and metrics

| RQ | Question | Metrics |
|---|---|---|
| RQ1 | How accurately does SpecDx detect, localise and type seeded defects, compared with B1–B5? | Typed precision, recall, F1 and average precision per type; Detection@k (k = 1, 3, 5) |
| RQ2 | Does it rediscover real defects, and how does it compare with a manual audit? | Recall on R-SO, R-12, R-DE and R-τ²; overlap with the Policy Loopholes findings |
| RQ3 | What does each component contribute? | A1–A7 against full SpecDx |
| RQ4 | How efficient is active synthesis? | Runs-to-detection; area under the detection-versus-runs curve |
| RQ5 | How should the population be composed? | Detection at a fixed budget across numbers of families (1–4), numbers of seeds, and model size |

**Matching rules** (a typed detection needs a location match and a type match):
- Hidden requirement: the reported clause equals the ground-truth clause.
- Weak check: the reported locked fields include the column constrained by the deleted or relaxed clause. Each Weak-check operator records that column.
- Contradiction: the Jaccard index between the reported M and the ground-truth conflict set is at least 0.5.
- Ambiguity: the reported sentence set intersects the mutated sentence or sentences. For a deletion operator the deleted sentence no longer exists, so a report matches when any reported sentence lies in the chunk that held it.

### 4.7 Statistics

- 95% bootstrap confidence intervals over mutants.
- Paired comparisons per mutant: exact McNemar on typed detection, with Holm correction across baselines.
- Effect sizes as paired risk differences.
- Every run is cached, so tables are regenerated from raw JSONL.

### 4.8 Pilot go/no-go gate

Before the full study, a pilot runs on S1 with 20 development mutants (5 per type) and 2 families. If SpecDx's typed F1 does not exceed both B2 and B3 on at least 3 of the 4 types, the full runs stop and the model in §3.2–3.4 is revisited.

---

## 5. The paper

**Venue:** ACM TOSEM first; IST, JSS and Requirements Engineering as fall-backs.

**Contributions**

1. **C1.** A formal model of executable business specifications: ⟨T, C, Π⟩, spec-induced equivalence, a four-type defect taxonomy defined by satisfaction, clause lifting, and Proposition 1.
2. **C2.** SpecDx inference: a set-valued latent-class model with family effects, and MUS/MCS duality with LLM behaviours as sampled correction sets (Proposition 2).
3. **C3.** Active witness synthesis: Thompson EIG with sequential escalation, sound refutation (Proposition 3), and ddmin localisation in T.
4. **C4.** An evaluation instrument: specification mutation operators with kill witnesses, mutant benchmarks for two subjects, and real-defect sets from TonomyX history and the τ²-Bench-Verified diff.
5. **C5.** An empirical study of RQ1–RQ5 and an open artefact.

**Positioning**

| Prior work | What it does | What SpecDx adds |
|---|---|---|
| LLM requirements-ambiguity detection | Static reading of text | Execution, with C as arbiter |
| Ambiguity via executable BPMN simulation (arXiv:2604.10884) | One deterministic simulator | A population of interpreters and typed localisation |
| Benchmark label errors: IRT (arXiv:2605.30504), ELT-Bench-Verified (arXiv:2603.29399), Policy Loopholes (arXiv:2609.14400) | Statistics only, or manual work | Statistics and logic, automatic, with witnesses |
| LLM disagreement as a signal (arXiv:2510.07686, arXiv:2602.11619) | An uncertainty signal | Typed diagnosis down to a clause or a sentence |
| Model-based diagnosis and MUS/MCS (Reiter 1987; Liffiton & Sakallah 2008; Ignatiev et al., IJCAI 2019) | Perfect observations | Noisy observers with latent competence |
| Truth inference (Dawid & Skene 1979) | One true label | Set-valued truth constrained by logic |

**Outline**
1. Introduction
2. Motivating example: R-SO and R-12 on CASE-0031
3. Formal model
4. SpecDx
5. Theory
6. Evaluation design
7. Results for RQ1–RQ5
8. Discussion: implications for specification authors and benchmark builders
9. Threats to validity
10. Related work
11. Conclusion and artefact availability

**Threats to validity**
- **Construct.** Mutation operators may not represent real defects. Mitigated by the real defects of S1 and S2, and by deriving operators from τ²-Bench-Verified's repair categories.
- **Internal.**
  - Correlated LLM errors: mitigated by family effects, and RQ5 tests the assumption of Proposition 2.
  - Non-determinism: mitigated by seeds, fixed temperatures, recorded model digests and run caching.
  - LLM-generated Tell variants: frozen and published.
  - Hyper-parameter overfitting: tuning only on the development split.
- **External.** Two specifications, two languages, two domains, and local models of at most about 14B parameters.
- **Conclusion.** Paired tests, bootstrap confidence intervals and Holm correction.

---

## 6. Implementation

### 6.1 Rules

1. `vendor/tonomyx/` is a byte-identical `git archive` of the latest TonomyX `master` commit at vendoring time. The commit hash and a `diff`-based verification recipe are recorded in `vendor/VENDORED.md`. It is never edited.
2. `vendor/tau2/` holds the original τ²-bench at the commit from which τ²-Bench-Verified was forked, and τ²-Bench-Verified itself, both pinned and recorded in `vendor/VENDORED.md`.
3. SpecDx has its own PostgreSQL on **port 55434**. TonomyX uses 55432 and FAA 55433.
4. No imports from FAA or `aebench`.
5. Each SpecDx unit has one purpose and talks to the others through the types of §3.1.

### 6.2 Repository layout

```
TonomyX-paper/
├── vendor/            tonomyx/ · tau2/ · VENDORED.md        (frozen)
├── specdx/
│   ├── spec/          Tell corpus (PDF→chunks, KB, nguyen_tac) · Clause extraction · lifting · ≡_S
│   ├── scenarios/     scenario space Π · boundary values · variant writer into working copies
│   ├── orchestrate/   parametric CASE-0031 flow on TonomyX's public runtime (§6.3)
│   ├── interpreters/  population · Tell variants · search_spec (BM25) · SpecInterpreterExecutor · run cache
│   ├── observe/       one run in its own transaction → Behaviour · V(b) · class key
│   ├── infer/         set-valued latent-class EM · family effects · soft MHS (PySAT) · typed scores
│   ├── active/        Thompson EIG · escalation · Refute · ddmin
│   ├── mutate/        operators with kill witnesses → work/mutants/<id>/
│   ├── subjects/tau2/ τ² adapter · real-defect extraction from the Verified diff
│   ├── baselines/     B1–B5
│   ├── eval/          RQ runners · metrics · statistics · table and figure generation
│   └── cli.py
├── experiments/       one JSON configuration per RQ
├── results/           raw JSONL (cached runs) · derived reports
├── paper/             LaTeX (acmart, TOSEM); tables and figures generated from results/
├── tests/
└── docs/superpowers/specs/
```

### 6.3 Orchestrator and executor

- **Orchestrator.** The vendored `run_case` accepts only `discount_pct` and `grn_received`; it hard-codes defective units and late days. `specdx/orchestrate/` therefore provides its own parametric CASE-0031 flow built on TonomyX's public API (`build_runtime(..., executor=...)`, the tool registry, `decide_approval`, `update_status`, `log_event`). A test requires it to reproduce the vendored flow's outcome and the golden state on `happy`, `discount8` and `grn12` before any extension is used. Human gate decisions are simulated deterministically from the authority rules of the specification being run.
- **Executor.** `SpecInterpreterExecutor` implements TonomyX's `DeExecutor` protocol. Its system prompt is built from the Tell variant, not from the vendored playbooks, and it adds the read-only `search_spec` tool over the T corpus. Every state-changing call goes through the vendored gated tools, so enforcement is unchanged, and the result envelope is built from tool effects, as `OllamaExecutor` does.

### 6.4 Testing

**Algorithms on planted ground truth**, with no LLM involved:
- EM recovers planted A_s and θ;
- soft MHS returns the planted MUS on toy clause sets;
- ddmin finds the planted sentence;
- Thompson EIG prefers a planted informative cell;
- each mutation operator's kill witness violates (or passes) exactly as designed;
- lifting reproduces the expected lifted form of all 42 invariants;
- ≡_S classifies every column of the 26 tables as pinned.

**Integration:**
- A scripted model through `httpx.MockTransport` on real PostgreSQL, the same pattern TonomyX uses.
- The orchestrator's golden reproduction on the three canon scenarios.
- The τ² adapter scores the original tasks as τ² itself does.

Real models are used only in experiments.

### 6.5 Error handling and reproducibility

- A timeout, crash or malformed tool call is recorded as a behaviour with status `error`. It is counted, never dropped, and never assigned to a class.
- Each run executes in its own transaction and rolls back. Mutants and scenario variants live in `work/`, never in the canon.
- The run cache is keyed by a hash of (specification digest, scenario parameters, interpreter, variant, seed, step, model digest). Re-running resumes and reproduces.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Interpreters without playbooks are too weak on Vietnamese S1 (§7.6 of the TonomyX report: without a playbook the model "reads the order three times and reports done") | T includes the full requirement documents through retrieval; θ down-weights weak interpreters; the pilot gate (§4.8) catches failure early |
| Fewer than 4 tool-capable families fit in 16 GB | Candidate list in §4.2; Proposition 2 and RQ5 are then reported with the families available |
| Shared biases make all families wrong together | Family effects; RQ5; stated as the main assumption of Proposition 2 |
| τ² adapter cost, and noise from the user simulator | User simulator fixed at T = 0; S2 runs without synthesis; adapter validated by reproducing τ²'s own scores on the original tasks |
| A competitor publishes automatic policy auditing first | Positioning rests on the combination of statistics and logic, the typed taxonomy, refutation, and active synthesis, which no single prior work has |

---

## 8. Decisions log

| # | Decision | Alternatives rejected | Why |
|---|---|---|---|
| 1 | Direction D2 | H1/D1 (claim–effect integrity: crowded in 2026), H2 (cross-tier conformance: overlaps FAA's pillars), D3 (process mining: same venues as FAA), D5 (meta-science: would expose FAA's withdrawn results) | Most different from FAA; grounded in real TonomyX findings |
| 2 | Approach A | B (statistics only: close to arXiv:2605.30504), C (symbolic only: blind to T) | The combination is the novelty |
| 3 | Playbooks excluded from T | Including them | They are a human reading of the canon |
| 4 | Full requirement documents in T through retrieval | KB only | KB-only T is too thin for competent behaviour |
| 5 | S2 = τ²-bench + Verified | S1 only | An independent source of real defects with no human labelling |
| 6 | Own orchestrator on TonomyX's public API | Patching the vendored flow | Vendored code is frozen |
| 7 | `policy.rego` excluded from mutation | Mutating it | Its rules are mirrored in frozen code, so the mutation would not take effect |

---

## 9. Delivery phases

The work is too large for one implementation plan. Each phase gets its own plan, and each later phase starts only after the previous one is done.

| Phase | Scope | Exit condition |
|---|---|---|
| P1: core on S1 | Vendoring; `spec/`, `scenarios/`, `orchestrate/`, `interpreters/`, `observe/`, `infer/`, `active/`, `mutate/` for S1; baselines B2 and B3 and the matching rules, which the gate needs; the capability pilot (§4.2) | All §6.4 tests pass, and the pilot go/no-go gate (§4.8) passes |
| P2: S2 and comparison | `subjects/tau2/`, baselines B1, B4 and B5, the rest of `eval/`; S2 mutants and real defects | τ² adapter reproduces τ²'s own scores; B1–B5 run on the development split |
| P3: full study | Test-split runs for RQ1–RQ5, statistics, tables and figures | All tables regenerate from `results/raw` |
| P4: manuscript | `paper/` in acmart for TOSEM | Full draft |

---

## 11. Planning-time clarifications (2026-10-06)

Reading the vendored code while planning P1 settled the following points. They refine the design and do not change its claims.

1. **Lifting, made precise.** The anchor is `grn12`, because the 42 invariants and the golden state describe `grn12`'s end state. The anchor's parameters are ordered 15, received 12, discount 0, GRNs per PO 1.
   - Only values in integer-typed schema columns are lifted: `po.sl_dat` and the `ITEM-0012` line of `soline.sl` lift to `ordered`, and `grn.sl_nhan` lifts to `received`.
   - Count assertions and text are never lifted.
   - A lifted clause applies only when the scenario's GRNs per PO equals the anchor's; that parameter changes which rows exist, so it is *structural*.
   - Across the three canon scenarios, a numeric expectation that equals the same parameter in all three is lifted (`pay_units` lifts to `received`). A categorical expectation becomes a membership clause over the values seen (`three_way` ∈ {Khớp, Cần kiểm}; `apr_0102` ∈ {normal, over_cap}) that applies to every scenario, alongside its exact per-scenario clause.
   - Everything else stays anchored.
2. **Background rows.** Every run starts from master data plus the operational rows that do not belong to CASE-0031: the TH0 branch (`cand`, `att`, `esg`, and the approval with a blank `case_id`). Without them, every run would violate the TH0 invariants, which the process never touches.
3. **Human gate steps are interpreted.** At each gate, the scenario fixes the decision (approve). A persona interpreter (P2, P6), working with the human tools, records the decision and carries out its consequences as it reads T. Without this, R-SO could never surface, because whether `SO-0031` advances is exactly a consequence of the P2 decision.
4. **Singleton typing.** A singleton hitting set {c} is a Hidden requirement if refutation finds a state in π that satisfies c, and a Contradiction (a MUS of size 1) if it does not. This is how R-12 types as a Contradiction: `fn_reconcile` errors on two receipts, so no state satisfies `vinv.doi_soat = fn_reconcile(po)`.
5. **Competence is pooled per model** across seeds and Tell variants: θ is indexed by model, the family effect is indexed by family, and seeds and variants are sampling noise.
6. **Mutation operators revised** as in §4.3. With them, S1 has about 40 mutants in P1. Growing the site lists to the 150–250 target belongs to P3.
7. **Clause families for every scenario.** These apply to every scenario in addition to the lifted and anchored clauses:
   - blank columns (from `cot_de_trong_khong_canon`);
   - step completion (each step ends `done` or `needs_gate`);
   - reconciliation consistency (`vinv.doi_soat` equals `fn_reconcile(po)`, and an error counts as a violation);
   - the payment guard;
   - the event vocabulary.
8. **Bootstrap confidence intervals over interpreters** (§3.2) are deferred to P3; P1 reports posterior probabilities.

The final review of P1 (2026-10-07) added the following clarifications.

9. **Ambiguity localisation in P1.** The localiser retrieves the top 5 chunks for a step, not the top 20 of §3.7. This bounds the ddmin cost. Ambiguities sourced beyond rank 5 go unlocalised, which is conservative against SpecDx.
   - ddmin first tests the empty set. If the split persists with none of the candidate sentences, the ambiguity is *unlocalised*. It is also unlocalised when the ddmin tests or the localisation budget (item 13) run out before the result is 1-minimal.
   - An unlocalised ambiguity keeps its type with an empty location, so it matches no ground truth.
   - Before localisation, an ambiguity is identified by its step and the locked fields its admissible classes differ on. The differential against the base report compares ambiguities on that identity, because localisation gives different sentence ids from one run to the next. Only the mutant's remaining ambiguities are localised. Defects that then share type and location are merged.
10. **Errors and classes.**
    - Step-level errors enter the step's class key as interpreter behaviour. These are an interpreter's malformed or failed steps inside an otherwise completed run.
    - Run-level errors never get a class (§6.5).
    - Infrastructure errors never become behaviours. An Ollama outage is a transport failure or an HTTP 5xx that persists through five attempts with pauses of 2, 4, 8 and 16 s. A read timeout is still the interpreter's behaviour, and so is the HTTP 500 with which Ollama reports a tool call the model wrote malformed. An Ollama or database outage, or a database that another process holds, stops the run without caching it, and it stops the gate.
11. **B2 uses more data than §4.5 states.** B2 uses every Tell variant × seed of the best model, more than §4.5's "× 3 seeds". This gives the baseline more data, never less.
12. **The gate verdict (§4.8).**
    - For each type, SpecDx's typed F1 is compared with each baseline's *untyped* F1 for that type. Untyped scoring charges a baseline's unmatched findings to the mutant's true type.
    - A type is a win if SpecDx's typed F1 > max(B2 untyped F1, B3 untyped F1). The gate passes on at least 3 wins of 4.
    - Reason: B2 and B3 cannot assign types by construction, since they report only Hidden requirements. Their typed F1 for Contradiction, Ambiguity and Weak check is therefore 0, and a typed-against-typed comparison would be won by any single SpecDx true positive.
    - The literal typed-against-typed comparison is still reported, for transparency.
    - If any mutant could not run, the headline reads INCOMPLETE and no verdict is given. Infrastructure failures stop the gate instead (item 10).
13. **Probe and localisation budgets.** Two caps apply per specification, each to the runs it requests. A run counts whether it is cached or not, so a resumed study spends the budgets exactly as the original did. Separate caps mean refutation can never leave localisation without runs.
    - `probe_budget` (default 100) caps directed-probe runs, which serve refutation. Refutation candidates are processed in descending score order. Each pair of scenario and target set is refuted once per specification.
    - A singleton whose refutation was never attempted because the probe budget ran out is typed Hidden, because a Contradiction requires a failed refutation attempt. For the same reason, an unattempted candidate of two or more clauses is not reported.
    - `localize_budget` (default 80) caps ddmin runs. Ambiguities are localised in descending score order until it runs out. The ambiguity being localised when it runs out, and every one after it, stays unlocalised (item 9).
    - With the population budget of 200 runs, a specification costs at most 380 runs.

---

## 12. Key references

- Calvanese et al. *Agentic BPM: A Research Manifesto.* Information Systems, 2026. arXiv:2603.18916 (context only)
- *Policy Loopholes in Agent Evaluation: When Policy Ambiguity Masquerades as Agent Error.* arXiv:2609.14400
- τ²-Bench-Verified. https://github.com/amazon-agi/tau2-bench-verified
- *Auditing LLM Benchmarks with Item Response Theory.* arXiv:2605.30504
- *Stress-Testing Model Specs Reveals Character Differences among Language Models.* arXiv:2510.07686
- *When Agents Disagree With Themselves: Behavioral Consistency as an Uncertainty Signal for LLM Agents.* arXiv:2602.11619
- *Ambiguity Detection and Elimination in Automated Executable Process Modeling.* arXiv:2604.10884
- *ELT-Bench-Verified.* arXiv:2603.29399
- Reiter (1987) *A Theory of Diagnosis from First Principles*; Liffiton & Sakallah (2008) *Algorithms for Computing Minimal Unsatisfiable Subsets of Constraints*; Ignatiev et al. (2019) *Model-Based Diagnosis with Multiple Observations*, IJCAI
- Dawid & Skene (1979) *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*
- Zeller & Hildebrandt (2002) *Simplifying and Isolating Failure-Inducing Input* (ddmin)
- TonomyX technical report v1.1 (`TonomyX/technical-report.md`), §5, §7.6, §9 (R8, R12)
