# P1 development-split diagnostics: the evidence for revisiting §3.2–§3.4

**Scope.** These diagnostics cover the base specification and the 20 development mutants of the P1 gate (`p1-gate-diagnosis.md`), and nothing else. The analysis is read-only: no method was changed, nothing was tuned, and no run was executed. It reads the run cache, the gate's per-mutant lines and the mutant packs. The code at HEAD (`e64dc1e`) is the code that ran the gate (`cda573c`), since only docs and results changed in between.

**Reproduce.** Run `.venv/bin/python experiments/p1_dev_diagnostics.py` (about 20 s). It prints every table below, plus the per-item tables for each W mutant, the list of model-specific findings, the base contradictions and the per-model H1 value tables. The output is byte-identical across `PYTHONHASHSEED` values.

Each section separates **Facts** (measured) from **Interpretation**.

## Key findings

1. **Competence is inverted.** On the base, θ is 0.003 for `qwen3:14b`, 0.005 for `qwen2.5:7b` and 0.547 for `llama3.1:8b`. Yet llama wrote no row and no event in any of its 64 base runs and has the most violations per run (23.7, against 16.2 for qwen3). The MAP admissible set holds llama's modal class in 87 of 88 items, and llama's runs carry 100.0% of the §3.3 hitting-set weight.
2. **The population barely acts on the case.** 99% of the violations behind the base's golden-state findings are rows that were never written. 61 of the 90 clause findings are violated by every run of every model.
3. **The ambiguity edits reach the interpreters, but do not change what they do.**
   - **Conversations:** the claims diverge from their base twin at `draft_quote` in 18–29 runs per A mutant, against 3.0 in the control.
   - **Behaviour classes:** they change in only 6–14 of 200 runs (control 2–5). No item moves by more than 2 of 27 runs.
4. **Ambiguity is blocked on three counts.**
   - **Below threshold:** the highest P(|A|≥2) is 0.451 (θ held fixed, it would cross only with P(|A|=1) ≤ 0.881).
   - **No locked field differs:** the items nearest the threshold split llama's runs between `done` and `error`, with no locked field differing. diagnose() would drop them anyway.
   - **No prior helps:** even refitting at P(|A|=1) = 0.25 yields **0** new reportable ambiguities in any A mutant.
5. **The weak truths cannot be observed.**
   - **W2:** each truth event is absent in 200 of 200 runs, on the base and on the mutant alike.
   - **W1-04:** the truth field exists in only 5 of 200 runs, and deviates in 4 of them, at most 1 per item.
   - **Detector:** its first condition, a clean MAP set, holds in 8 of 88 items.
6. **The H1 truths are masked because no run on the anchor scenario ever writes the mutated value** (0 of 27 for each truth). The base already reports the same hidden finding, and the differential removes it.

---

## 0. Data, mapping and validation

**How runs map to specifications (the robust way).** Behaviours carry no variant field, so the mapping recomputes run keys exactly as `Runner.run` builds them:

`run_key(variant.digest(), steps digest, context digest, json(scenario params), interp.key, suffix, model digest)`

Each input comes from a fixed place:

- **Steps and context digests:** taken from a `Workbench` opened with a factory that refuses to run.
- **Model digest:** the one each cached behaviour records (exactly one per model).
- **Suffix:** "" for population runs.
- **Mutant variants:** rebuilt from `work/mutants/<id>` with the same corpus edits `all_mutants` applies. `all_mutants` itself is not called, because it rewrites the packs.

Facts:

- **Runs:** the cache holds 6,300 behaviours, all with status `ok`. Of these, 4,200 = 21 × 200 population runs are recovered by key, and no run key is shared between specifications. The remaining 2,100 are directed probes (100 per specification), and every gate line's run counts match.
- **Witnesses:** every SpecDx witness run key in the gate lines (71,198 references) is among the recovered population runs of its own specification.
- **Scenarios:** every specification ran the same 8 scenarios with the same counts.

| scenario | ordered | received | discount % | GRNs | runs per spec |
|---|---:|---:|---:|---:|---:|
| discount8 | 15 | 15 | 8 | 1 | 27 |
| grn12 | 15 | 12 | 0 | 1 | 27 |
| happy | 15 | 15 | 0 | 1 | 27 |
| o14-r15-d4-g2 | 14 | 15 | 4 | 2 | 27 |
| o14-r15-d8-g1 | 14 | 15 | 8 | 1 | 27 |
| o15-r11-d8-g1 | 15 | 11 | 8 | 1 | 27 |
| o15-r12-d8-g2 | 15 | 12 | 8 | 2 | 27 |
| o16-r14-d4-g1 | 16 | 14 | 4 | 1 | 11 |

- **Validation:** `latent.fit` with the gate config (K_max 6, P(|A|=1) 0.9, 50 EM iterations) on the 200 base runs reproduces the gate's posterior:
  - all 75 base Hidden(c) scores that are computable from views match to 1e-9 (largest error 4.4e-16);
  - the base weak finding's score matches (0.987060, recomputed 0.987060);
  - the largest P(|A|≥2) is 0.451 < 0.5, so there is no ambiguity, as recorded.
- **Code fact:** the bandit picks uniformly among *untried* cells while any remain (`CellBandit.choose`). With 3 canon cells tried and 15 left, all 5 synthesised scenarios were random picks, identical for every specification.

---

## Q1. Are the ambiguity (A) mutants observable?

### Facts

**The edits** (base corpus against mutant corpus, by sentence id):

| mutant | sentence id | before | after |
|---|---|---|---|
| A1-01 | `KB:KB-0003:04` | Yêu cầu vượt trần (vd 8%) phải gắn cờ ngoại lệ và dừng ở cổng P2 (APR-0102); | hệ thống không tự cấp mức ngoài chính sách. |
| A1-01 | `KB:KB-0003:05` | hệ thống không tự cấp mức ngoài chính sách. | (removed) |
| A1-02 | `PRINCIPLES:03` | GRN lệch / lô lỗi / giao trễ: P4 xử lý ngoại lệ — KHÔNG mở APR chuẩn. | Nhánh HR/offer là TH0: APR-TH-01, case_id trống; KHÔNG có CASE-0032. |
| A1-02 | `PRINCIPLES:04` | Nhánh HR/offer là TH0: … | (removed) |
| A2-01 | `KB:KB-0003:03` | \*\*mức tối đa 5%\*\*. | \*\*mức tối đa hợp lý\*\*. |
| A2-02 | `PRINCIPLES:02` | CASE-0031 có ĐÚNG 3 cổng: APR-0102/P2, APR-0155/P6 (maker-checker), APR-0210/P5 (đối soát lệch). | CASE-0031 có một số cổng phê duyệt phù hợp. |
| A3-00 | `KB:KB-0009:00–02` | (absent) | Chính sách giá & chiết khấu (cập nhật). / Mức chiết khấu tối đa là 7%; / yêu cầu đến 7% được áp dụng trực tiếp, không cần dừng ở cổng duyệt. |

A1 deletes the sentence in the pack file, so the following sentence ids shift by one. The A1 truth is chunk-level, so this does not affect scoring. C is identical to the base's in every A mutant (same clause ids, payloads and scopes), so class keys are directly comparable.

**Retrieval.** Behaviours do not record tool calls, queries or search results. A Behaviour stores the end state, the step statuses, the refused calls as (step, tool name) only, and each step's final claim. BM25 retrieval (k = 5, reply limit 12,000 characters, as in `executor.py`) was therefore rerun offline on each mutant's corpus. It used the 27 work-order statements (9 interpreted steps × 3 Tell variants) and the 9 localiser queries (`workbench.py`).

| mutant | edited chunk | in top 5 (statements) | in reply payload | in top 5 (localiser) | best rank of 210 (step) |
|---|---|---:|---:|---:|---|
| A1-01 | `KB:KB-0003` | 0/27 | 0/27 | 0/9 | 36 (draft_quote) |
| A1-02 | `PRINCIPLES` | 0/27 | 0/27 | 0/9 | 37 (draft_quote) |
| A2-01 | `KB:KB-0003` | 0/27 | 0/27 | 0/9 | 19 (draft_quote) |
| A2-02 | `PRINCIPLES` | 0/27 | 0/27 | 0/9 | 38 (draft_quote) |
| A3-00 | `KB:KB-0009` | 0/27 | 0/27 | 0/9 | 7 (draft_quote) |

The edit itself changes the BM25 top 5 for 1 of 135 statement queries, through IDF and length shifts.

**Paired comparison with the base.**

- **Design:** every mutant run has a base twin with the same scenario, model, Tell variant and seed, so the system prompt is identical.
- **Control:** the 12 dev mutants that edit only C (C1-00/01/02, H1-02/03/04, H2-00/01, W2-01/03/04/05). No interpreter can see their edits, so they measure Ollama's own drift.
- **Classes:** all classes are computed under the base lens.

| specification | identical end state (of 200) | any step claim differs | any step class differs |
|---|---:|---:|---:|
| A1-01 | 194 | 26 | 6 |
| A1-02 | 193 | 56 | 7 |
| A2-01 | 186 | 31 | 14 |
| A2-02 | 188 | 56 | 12 |
| A3-00 | 189 | 42 | 11 |
| control (12 specs) | 195–198 (mean 196.8) | 9–15 (mean 12.5) | 2–5 (mean 3.2) |

First step whose claim differs from the twin's (runs). The control column is the mean over the 12 control specifications.

| step | A1-01 | A1-02 | A2-01 | A2-02 | A3-00 | control |
|---|---:|---:|---:|---:|---:|---:|
| DE-SM-02:draft_quote | 18 | 21 | 23 | 18 | 29 | 3.0 |
| P2:decide_quote | 0 | 0 | 0 | 16 | 6 | 0.1 |
| P6:decide_po | 6 | 26 | 7 | 20 | 6 | 8.0 |
| DE-OP-02:record_goods_receipt | 2 | 9 | 1 | 2 | 1 | 1.4 |
| the other 5 interpreted steps | 0 | 0 | 0 | 0 | 0 | 0.0 |

Runs with any claim difference, by model:

- **qwen2.5:** diverges most, in 21–36 runs against a control of 8.1.
- **llama:** above its control of 1.0 in A1-02 and A2-02 (8 each), and most in A3-00 (16).
- **qwen3:** clearly above its control only in A1-02 and A2-02 (26 and 12 against 3.4).

| model | A1-01 | A1-02 | A2-01 | A2-02 | A3-00 | control |
|---|---:|---:|---:|---:|---:|---:|
| qwen3:14b | 4 | 26 | 3 | 12 | 4 | 3.4 |
| qwen2.5:7b | 21 | 22 | 27 | 36 | 22 | 8.1 |
| llama3.1:8b | 1 | 8 | 1 | 8 | 16 | 1.0 |

Claims containing a marker of the edited text (runs of 200, base → mutant):

| marker | runs |
|---|---|
| A1-01 "vượt trần" | 0 → 0 |
| A1-01 "gắn cờ" | 4 → 0 |
| A1-02 "APR chuẩn" | 0 → 0 |
| A1-02 "xử lý ngoại lệ" | 5 → 4 |
| A2-01 "5%" | 12 → 8 |
| A2-01 "hợp lý" | 7 → 10 |
| A2-02 "3 cổng" | 1 → 0 |
| A2-02 "một số cổng" | 0 → 3 |
| A3-00 "7%" | 0 → 2 |
| A3-00 "KB-0009" | 0 → 0 |

**Behaviour classes.** Runs whose class differs from the twin's:

| step | A1-01 | A1-02 | A2-01 | A2-02 | A3-00 | control |
|---|---:|---:|---:|---:|---:|---:|
| draft_quote | 4 | 4 | 9 | 3 | 4 | mean 1.0, max 1 |
| P2:decide_quote | 4 | 4 | 9 | 9 | 3 | mean 1.0, max 1 |
| P6:decide_po | 2 | 3 | 5 | 3 | 5 | mean 2.2, max 4 |

The other steps differ in at most 2 runs.

What differs in those pairs, and the discount of the scenario where each differing pair occurs:

| mutant | differing locked fields (top) | items touched | pairs by discount |
|---|---|---:|---|
| A1-01 | so.\* ×8 (SO-0031 row present vs absent), evt:APPROVED:APR-0155 ×2 | 10 | d0 5, d4 2, d8 3 |
| A1-02 | so.\* ×8, evt:APPROVED:APR-0155 ×3 | 11 | d0 6, d4 2, d8 3 |
| A2-01 | so.\* ×18, evt:APPROVED:APR-0155 ×5 | 15 | d0 14, d4 3, d8 6 |
| A2-02 | evt:APPROVED:APR-0102 ×6, so.\* ×6, evt:APPROVED:APR-0155 ×3 | 12 | d0 3, d4 6, d8 6 |
| A3-00 | status only ×5, evt:APPROVED:APR-0155 ×5, so.\* ×4 | 11 | d0 2, d4 9, d8 4 |

**Class distributions per item** (mutant against base):

- **A mutants:** items with total-variation distance > 0 are 10, 11, 15, 11 and 11 of 88. The control has 57 of 1,056, or 4.75 per specification.
- **Size of the shift:** the maximum TV is 0.037 (1 of 27 runs) everywhere except A3-00, where it is 0.074 (2 of 27).

**Scenario coverage.** The runs covered discounts 0, 4 and 8%. A3-00's article (cap 7%, no gate up to 7%) disagrees with KB-0003 (cap 5%) only at 6–7%. The boundary domain holds 6%, but no 6% scenario was run on any specification.

### Interpretation

- **Could search_spec have retrieved the edits?** No query is on record, and the work-order statements never put the edited chunk in the top 5 (best ranks 7–38 of 210). The interpreters' own queries evidently did reach changed text, though:
  - **Excess divergence:** first divergences at `draft_quote` exceed the control by 15–26 runs per mutant, and the system prompt is identical between twins.
  - **Targeted steps:** the two PRINCIPLES edits, both about approvals (A1-02: no standard APR for GRN exceptions; A2-02: the number of gates), diverge most at the approval steps (A1-02 at P6: 26 runs; A2-02 at P2: 16 and P6: 20), against control means of 8.0 at P6 and 0.1 at P2.
  - **Markers move:** "một số cổng" and "7%" appear only in the mutants, and "gắn cờ" disappears in A1-01.
- **Is the edit observable in behaviour?** Barely, and not as a split:
  - **Size:** the excess class changes over the control are 3–11 runs of 200, and no item moves by more than 2 of 27 runs.
  - **What changes:** mostly whether the SO-0031 row gets written at all, which is not the field each edit concerns. The exception is A2-02's APPROVED APR-0102 events.
  - **Where:** the changes are not concentrated where the edit matters. For A2-01, 14 of 23 differing pairs are at discount 0, where a cap is irrelevant.
  - **Conclusion:** the A edits act as a perturbation of the sampling path, not as a second reading adopted by part of the population. A3-00 additionally had no scenario that could separate its two readings.

---

## Q2. Are the weak (W) mutants observable?

### Facts

§3.4's Weak(s) needs four conditions at one item:

1. a MAP set none of whose classes violates a step-relevant clause;
2. a clean class outside it with at least weak_min_runs (2) runs;
3. P(that class ∉ A) ≥ 0.5;
4. to match the truth, the fields separating that class from the top admissible class must include the truth field.

Condition 1 holds in **8 of 88 items** on the base and on every W mutant. The MAP set holds a class whose step status is `error` in **77 of 88**.

| mutant | truth field | removed from C | owning step | runs with the field observable (q3 / q2.5 / llama) | runs deviating from golden (q3 / q2.5 / llama) | max deviating runs in one item | items meeting cond. 1 | items where the detector reports the field |
|---|---|---|---|---|---|---:|---:|---:|
| W1-04 | vinv.so_tien (golden blank) | `blank:vinv.so_tien`, `inv:vinv-sotien-blank` | DE-FN-02:reconcile | 5/72, 0/64, 0/64 (vinv row written) | 4, 0, 0 (value '0'); base 0 | 1 | 0/8 | 0 |
| W2-01 | evt:FLAGGED:ITEM-0012 (actor AI) | `event:FLAGGED:ITEM-0012:AI` | DE-OP-01:check_availability | all runs | 72, 64, 64 (absent); base identical | 27 | 0/8 | 0 |
| W2-03 | evt:RECEIVED:GRN-0015 (AI+) | `event:RECEIVED:GRN-0015:AI+` | DE-OP-02:record_goods_receipt | all runs | 72, 64, 64 (absent); base identical | 27 | 0/8 | 0 |
| W2-04 | evt:RECONCILED:VINV-0015 (AI+) | `event:RECONCILED:VINV-0015:AI+` | DE-FN-02:reconcile | all runs | 72, 64, 64 (absent); base identical | 27 | 0/8 | 0 |
| W2-05 | evt:DRAFTED:PAY-0031 (AI) | `event:DRAFTED:PAY-0031:AI` | DE-FN-03:draft_payment | all runs | 72, 64, 64 (absent); base identical | 27 | 1/8 | 0 |

- **W1-04:** its 4 deviating runs are qwen3's, one in each of four items, and none is clean. On the base, the vendored executor denies any non-null write to a canon-blank column (`_invented_numbers`), so the deviation becomes possible only in the mutant.
- **W2:** the removed event clauses are anchored to grn12. On the other seven scenarios the event was already unguarded on the base.
- **The events in the base end states** are, as model, event and runs:
  - qwen2.5: APPROVED APR-0102 (actor "Trưởng Kinh doanh (P2)") ×8, INSPECTED APR-0102 ×7, APPROVED APR-0155 (actor "P6") ×2;
  - qwen3: FLAGGED "so:SO-0031" ×3 and FLAGGED "vinv" ×2;
  - environment recorders, in qwen3 runs where the GRN and PO rows exist: INSPECTED NCR-0007 ×15 and SHIPPED SHIP-0031 ×16.

  None of the four truth events appears in any run.

### Interpretation

- **W2:** these mutants are unobservable with this population. A weak check is witnessed by a behaviour outside Adm_T that C accepts, which needs an admissible class showing the field beside a deviant one. Here every run, on the base and on the mutant, omits the event. So no two classes differ on the truth field, and condition 4 can never hold.
- **W1-04:** observable in principle, because the guard's removal changes behaviour (4 runs write so_tien = 0). But the field exists only where a vinv row is written (5 of 200 runs), at most one deviant per item. That is below weak_min_runs, and none of those runs is clean.
- **Condition 1 fails almost everywhere for a reason outside the W operators.** The MAP "admissible" class is usually llama's failing class (Q3), so a clean admissible class is rarely available as the reference.

---

## Q3. Why does ambiguity never fire?

### Facts

| spec | θ qwen3:14b | θ qwen2.5:7b | θ llama3.1:8b | family effect u_f |
|---|---:|---:|---:|---|
| base | 0.003 | 0.005 | 0.547 | llama +1.64, qwen −1.65 |
| A1-01 / A1-02 / A2-01 / A2-02 | 0.003 | 0.005 | 0.547–0.548 | llama +1.65, qwen −1.65 |
| A3-00 | 0.003 | 0.005 | 0.540 | llama +1.64, qwen −1.64 |

P(|A_s| ≥ 2) over the 88 items (8 scenarios × 11 steps):

| spec | max | median | p90 | items ≥ 0.1 | items ≥ 0.5 | median classes per item | max classes | items with ≥ 2 classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.451 | 0.017 | 0.163 | 21 | 0 | 3 | 6 | 87 |
| A1-01 | 0.455 | 0.017 | 0.166 | 23 | 0 | 3 | 6 | 87 |
| A1-02 | 0.455 | 0.017 | 0.166 | 23 | 0 | 3 | 6 | 87 |
| A2-01 | 0.453 | 0.017 | 0.148 | 22 | 0 | 3 | 6 | 87 |
| A2-02 | 0.452 | 0.017 | 0.166 | 21 | 0 | 3 | 6 | 87 |
| A3-00 | 0.437 | 0.024 | 0.158 | 21 | 0 | 3 | 6 | 87 |

Who is "competent", and do the competent models split? (Base.)

| model | θ | mean violated clauses per run | runs writing any DE row | runs with any event | step statuses `error` | items whose MAP set holds the model's modal class | items where its runs fall in ≥ 2 classes (second class ≥ 2 runs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3:14b | 0.003 | 16.2 | 26% | 22% | 20% | 34/88 | 65/88 (41) |
| qwen2.5:7b | 0.005 | 20.0 | 25% | 23% | 40% | 47/88 | 56/88 (48) |
| llama3.1:8b | 0.547 | 23.7 | 0% | 0% | 72% | 87/88 | 63/88 (56) |

EM trajectory of θ on the base (qwen3 / qwen2.5 / llama). All three start at θ₀ = 0.7.

| iterations | 1 | 3 | 5 | 8 | 12 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| θ | .62/.67/.69 | .49/.60/.66 | .38/.54/.63 | .16/.36/.59 | .01/.13/.57 | .004/.03/.55 | .003/.005/.55 |

What separates the two most frequent classes, over the 87 base items with ≥ 2 classes:

- **only the step status differs** (done / error / needs_gate): 67 items;
- **locked fields differ:** 20 items.

The items closest to firing on the base. Class labels are per item, and the counts are runs per model:

| item | qwen3 | qwen2.5 | llama | MAP set | P(\|A\|≥2) | log BF | P(\|A\|=1) needed | locked fields between top 2 |
|---|---|---|---|---|---:|---:|---:|---|
| 7 scenarios × `DE-OP-01:check_availability` | k0×9 | k0×9 | k0×6 k1×3 | {k0,k1} | 0.451 | 2.00 | 0.881 | none (k1 = status `error`) |
| o14-r15-d4-g2 \| draft_quote | k0×9 | k0×7 k1 k2 | k0×5 k1×4 | {k1} | 0.171 | 0.62 | 0.650 | none |
| o14-r15-d8-g1 \| P2:decide_quote | k0×5 k1×4 | k0×5 k1×2 k2 k3 | k0×5 k1×4 | {k0} | 0.165 | 0.58 | 0.640 | none |

**Counterfactual, for understanding only.** With θ held fixed, an item crosses 0.5 exactly when P(|A|=1) ≤ BF/(1+BF). "Reportable" adds diagnose's second condition, that the top classes differ on a locked field. "New" counts the reportable (step, fields) identities the base would not also report, since the differential removes the rest.

| spec | largest P(\|A\|=1) at which any item crosses | threshold that would fire (max P) | at 0.75: crossing / reportable / new | at 0.5 | at 0.25 |
|---|---:|---:|---|---|---|
| base | 0.881 | 0.451 | 7 / 0 / 0 | 21 / 1 / 0 | 34 / 5 / 0 |
| A1-01 | 0.882 | 0.455 | 8 / 0 / 0 | 23 / 1 / 0 | 33 / 5 / 0 |
| A1-02 | 0.882 | 0.455 | 8 / 0 / 0 | 23 / 1 / 0 | 33 / 5 / 0 |
| A2-01 | 0.882 | 0.453 | 8 / 0 / 0 | 22 / 1 / 0 | 33 / 5 / 0 |
| A2-02 | 0.881 | 0.452 | 7 / 0 / 0 | 21 / 1 / 0 | 33 / 5 / 0 |
| A3-00 | 0.875 | 0.437 | 8 / 0 / 0 | 21 / 1 / 0 | 35 / 5 / 0 |

A full EM refit at the counterfactual prior (θ re-estimated) gives:

| P(\|A\|=1) | θ (qwen3 / qwen2.5 / llama) | max P(\|A\|≥2) | items ≥ 0.5 | reportable | new in any A mutant |
|---|---|---:|---:|---:|---:|
| 0.5 | .003 / .007 / .643 | 0.950 | 25 | 1 | 0 |
| 0.25 | .005 / .012 / .714 | 0.993 | 43 | 5 | 0 |

The other A mutants have 25–27 and 43–44 items ≥ 0.5 at these two priors.

The A edits move P(|A|≥2) by more than 0.01 in 2–4 items, or 11 for A3-00. The largest move in each mutant is `o14-r15-d4-g2|draft_quote`, from 0.171 to 0.287–0.300 (A1-01, A1-02, A2-01, A3-00), and `o14-r15-d4-g2|P2:decide_quote`, from 0.137 to 0.149 (A2-02).

### Interpretation

Ambiguity does not fire, for four measured reasons. In order of how binding each is:

1. **Competence is wrong.**
   - **The fixed point:** EM converges to a point where the only "competent" model is the one that never writes anything (θ 0.547), and the model with the fewest violations is treated as noise (θ 0.003).
   - **How it gets there:** no θ rises between any two consecutive EM iterations from 1 to 50 (the script checks each one), and both qwen values fall steadily. qwen3's occasional successful runs (for example, the 2 runs per scenario whose GRN exists, so the inspection succeeds) disagree with the failing majority. That costs it θ, which hands the MAP set to llama's classes, which costs qwen more.
   - **What P(|A|≥2) measures:** in effect, llama's own within-model inconsistency.
2. **The classes encode failure modes, not readings.** In 67 of 87 multi-class items the top two classes differ only in step status. At the 0.451 items, llama's runs split 6/3 between `done` and `error` with identical (empty) projections. Even if those items crossed 0.5, diagnose() would find no locked field and report nothing.
3. **The prior is not the binding constraint.** At the gate's prior, the nearest item needs P(|A|=1) ≤ 0.881 against the configured 0.9, or a threshold ≤ 0.451. But lowering the prior to 0.25, even with a full refit, yields 5 reportable identities on the base and on every A mutant, and **0 new** ones. Those 5 would be removed by the differential.
4. **There is no split to detect** (Q1). The A edits shift at most 2 of 27 runs per item.

---

## Q4. Where does the base noise come from?

### Facts

**The base report.** It holds 91 findings: 78 hidden, 12 contradiction and 1 weak. Every location is a single clause, except the weak one, which is located on a field.

**Competence and weight.** θ is in Q3. Of the §3.3 hitting-set weight (responsibility × θ), llama's runs carry 100.0%, and qwen3's and qwen2.5's carry 0.0% each.

**Failure per step** (base, 200 runs). Each cell is error / any step-relevant violation / the step's slice holds a row at the end.

| step | qwen3:14b | qwen2.5:7b | llama3.1:8b | relevant clauses |
|---|---|---|---|---:|
| DE-SM-02:draft_quote | 0% / 75% / 8% | 17% / 75% / 8% | 58% / 88% / 0% | 31 |
| DE-OP-01:check_availability | 0% / 12% / 0% | 0% / 14% / 0% | 34% / 44% / 0% | 2 |
| P2:decide_quote | 44% / 51% / 8% | 23% / 34% / 28% | 45% / 53% / 0% | 13 |
| DE-FN-04:draft_purchase_order | 0% / 75% / 22% | 22% / 78% / 11% | 77% / 94% / 0% | 20 |
| P6:decide_po | 22% / 81% / 22% | 53% / 88% / 11% | 78% / 94% / 0% | 20 |
| DE-OP-02:record_goods_receipt | 0% / 67% / 22% | 20% / 77% / 0% | 72% / 91% / 0% | 9 |
| ENV:inspection | 79% / 79% / 21% | 100% / 100% / 0% | 100% / 100% / 0% | 7 |
| ENV:shipment | 78% / 78% / 22% | 100% / 100% / 0% | 100% / 100% / 0% | 5 |
| DE-FN-01:draft_sales_invoice | 0% / 12% / 7% | 45% / 53% / 11% | 66% / 70% / 0% | 8 |
| DE-FN-02:reconcile | 0% / 99% / 7% | 34% / 100% / 0% | 78% / 100% / 0% | 15 |
| DE-FN-03:draft_payment | 0% / 12% / 6% | 22% / 30% / 0% | 88% / 89% / 0% | 15 |

**Why the steps write so little.** These counts cover the 9 interpreted steps of each run. The claim columns overlap, and the refusal reasons are not recorded.

| model | step claims | claim is a tool call written as text | "finished with no tool call that succeeded" | refused calls (top tools) |
|---|---:|---:|---:|---|
| qwen3:14b | 648 | 4 (1%) | 48 (7%) | 729 (read_records 415, write_draft 154, decide_approval 128) |
| qwen2.5:7b | 576 | 77 (13%) | 151 (26%) | 1,247 (read_records 418, write_draft 351, decide_approval 197, log_event 141) |
| llama3.1:8b | 576 | 222 (39%) | 381 (66%) | 238 (write_draft 150, decide_approval 35) |

**Failure per clause** (base, rate over the runs where the clause applies):

| clause kind | clauses | violated by every run | 50–99% | 1–49% | never |
|---|---:|---:|---:|---:|---:|
| golden | 54 | 36 | 16 | 0 | 2 |
| golden_event | 10 | 8 | 2 | 0 | 0 |
| invariant | 42 | 10 | 2 | 0 | 30 |
| expectation | 12 | 5 | 2 | 2 | 3 |
| step | 11 | 0 | 3 | 8 | 0 |
| blank, guard_pay, vocabulary, reconcile | 12 | 0 | 0 | 1 | 11 |
| all | 141 | 59 | 25 | 11 | 46 |

**Shared failure or model-specific?** For each finding and each model, this is the share of runs that violate the location, over the finding's scenarios where the clause applies.

| category | contradiction | hidden |
|---|---:|---:|
| every run of every model fails | 12 | 49 |
| every model fails in ≥ 50% of its runs | 0 | 24 |
| model-specific (some model < 50%) | 0 | 5 |

- **Model-specific findings:** all 5 are step-completion clauses on the partial scenario o16-r14-d4-g1, where qwen2.5 and llama have 1 run each.
- **Missing rows:** of the violations behind golden-state findings, 1,701 of 1,719 (99%) are a missing keyed row, not a wrong value.
- **Hidden(c) scores:** of the 78 hidden findings, 68 score 0, 1 scores in (0, 0.5), 5 in [0.5, 1) and 4 score 1. diagnose() reports every singleton hitting set whatever its Hidden(c) score.
- **Refutation notes:** 45 were refuted by symbolic repair, 33 were never attempted (probe budget), and 12 are unrefuted, typed contradiction.
- **The 12 contradictions:** every run violates all of them.
  - **9 golden columns** (APR-0102 ×6, APR-0155 ×2, SOLINE-0031-2.sl): no run wrote their keyed row (27 of 27 missing).
  - **3 grn12 expectations** (three_way ×2, open_units): no run wrote a vinv row (27 of 27).
  - **Why repair cannot refute them:** symbolic repair has no move for any of them. `propose` returns no candidate for a golden clause whose keyed row is absent, none for `three_way` without a vinv row, and none for `open_units` at all.

**Concentration:**

- **By scenario:** 72 of the 90 clause findings are on grn12 only. grn12 is the anchor, with 135 applicable clauses, against 25–34 on every other scenario.
- **By clause kind:** golden 53, invariant 12, golden_event 10, expectation 8, step 7.
- **By table:** apr 24, evt 12, soline 11, vinv 10.
- **By owning step:** draft_quote 23, P6 15, draft_purchase_order 14, reconcile 12, P2 12, draft_payment 10, others ≤ 7. The findings spread over all 11 steps.

**The masked H1 truths.**

- **Matches:** in each of the three cases, the mutant report holds the truth location as a hidden finding with the same note as the base's (score 0), and the differential removes it, so there are no typed and no untyped matches.
- **What the runs produce:** on grn12, where these anchored clauses apply, no run of any model produces either the base golden value or the mutated one.

| mutant | truth clause | golden: base → mutant | base finding | runs on grn12 producing either value | what the 200 base runs hold |
|---|---|---|---|---:|---|
| H1-02 | golden:inv:INV-0031:trang_thai | Nháp → Chờ duyệt | hidden, refuted by symbolic repair | 0/27 | no row 188; "CHUA_DUYET" 5 (qwen3); "draft" 7 (qwen2.5) |
| H1-03 | golden:pay:PAY-0031:trang_thai | Chờ duyệt → Nháp | hidden, not attempted | 0/27 | no row 196; "Chờ duyệt" 3 and "Chờ xử lý" 1 (qwen3, none on grn12) |
| H1-04 | golden:apr:APR-0210:muc_tu_hanh | L3 → L2 | hidden, not attempted | 0/27 | no row 200 |

### Interpretation

- **Where the noise comes from.** Not from a few steps or clauses, but from one cause: the interpreters seldom commit anything.
  - **qwen3** finishes its DE steps with status `done`, but writes in ≤ 22% of them, and many of its calls are refused.
  - **llama** writes nothing and often emits tool calls as text.
  - **The consequence:** among the clauses that apply somewhere, 52 of 54 golden and all 10 golden_event clauses are violated by at least half of the runs, and 36 and 8 of them by every run. This is shared incompetence, not a specification signal.
- **How the method amplifies it.** The latent model gives all the weight to the model that fails most uniformly, and the soft hitting sets need ≥ 95% weighted coverage. So every clause that llama violates becomes a candidate. Hidden(c) scores of 0 do not suppress these candidates, and rows that nobody writes become contradictions, because neither refutation route can create a row.
- **Why the H1 truths are masked.** The interpreters cannot produce the golden values.
  - **pay and apr:** the PAY-0031 and APR-0210 rows are written on grn12 by none of the 27 runs.
  - **inv:** the INV-0031 row is written on grn12 in 2 of 27 runs, with invented statuses ("CHUA_DUYET", "draft").
  - **The consequence:** a golden-value mutation of a row that is unwritten, or written with an off-vocabulary value, is indistinguishable from the base, which already flags the same clause.

---

## Implications for §3.2–§3.4

- **§3.2, competence.** θ as specified rewards reproducible failure. The model that never wrote a row (llama, 0 of 64 runs, 23.7 violations per run) gets θ = 0.547. The model with the fewest violations (qwen3, 16.2) gets 0.003, and EM drifts there monotonically from θ₀ = 0.7. Competence needs an anchor that does not let a failing model define A_s.
- **§3.2, class definition.** Classes mostly encode step status and failure, not readings of T. In 67 of 87 multi-class items the top two classes differ only in `done`/`error`, and the items nearest to ambiguity (0.451) differ in no locked field. Non-acting or failed steps need to be separated from admissible outcomes before A_s is inferred.
- **§3.2, the prior and the threshold.** These are not the bottleneck. Down to P(|A|=1) = 0.25 with a full refit, no A mutant yields a single reportable ambiguity that is not also the base's. The A edits shift at most 2 of 27 runs per item and are not aimed at the edited field. Revising the prior alone cannot recover them.
- **§3.4, Weak(s).** The detector presupposes a clean admissible class and a deviant class that both show the field. The MAP set is clean in 8 of 88 items, the W2 truth events are absent in 200 of 200 runs on the base and on the mutant, and W1-04's field exists in 5 of 200 runs. Weak(s) needs an observability precondition; otherwise these truths count as misses of the method rather than as unobservable.
- **§3.3–§3.4, Hidden and Contradiction.** Hitting sets are weighted 100% by one model, and 61 of 90 base clause findings fail in every run of every model. 99% of the golden violations are missing rows, and 68 of 78 hidden findings are reported with Hidden(c) = 0. The typing needs to distinguish "no interpreter produced the row" from "C is stricter than T". At present the first becomes Hidden, or becomes Contradiction when no refutation route can create the row.
- **The differential and masking.** The 3 H1 truths sit on rows that no grn12 run writes (0 of 27 for each). The base report holds the same key, so the §4.3 differential removes them for every method. Clause-level truths are only scorable where the population demonstrably reaches the clause's row. That is a property of the population and the item definition (§3.2), not of the score threshold.
