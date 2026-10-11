# P1 gate: per-mutant diagnosis (development split)

**Verdict: NOT PASSED. SpecDx wins none of the 4 types** (`p1-gate.md`).

The gate ran from 2026-10-07 18:08 to 2026-10-11 03:13 KST on commit cda573c, with a pause from 10-08 13:39 to 10-09 08:02. It executed 4,226 LLM runs, and none failed for infrastructure reasons. Every specification used its full population budget (200 runs) and probe budget (100 runs). No ddmin run happened anywhere.

As spec §4.8 requires, this diagnosis uses only the 20 development mutants and changes nothing.

## Per mutant

"New" counts SpecDx findings that are not in the base report after the differential. hitT and hitU are SpecDx's typed and untyped matches; b2U is B2's untyped matches.

| mutant | type | truth | new C/H/A/W | hitT | hitU | b2U |
|---|---|---|---|---:|---:|---:|
| A1-01 | ambiguity | KB:KB-0003 | 0/0/0/1 | 0 | 0 | 0 |
| A1-02 | ambiguity | PRINCIPLES | 0/0/0/1 | 0 | 0 | 0 |
| A2-01 | ambiguity | KB:KB-0003:03 | 0/0/0/0 | 0 | 0 | 0 |
| A2-02 | ambiguity | PRINCIPLES:02 | 0/0/0/0 | 0 | 0 | 0 |
| A3-00 | ambiguity | KB:KB-0009 sentences | 0/0/0/0 | 0 | 0 | 0 |
| C1-00 | contradiction | inv:mut-c1-po-15, inv:po-15 | 0/1/0/0 | 0 | 1 | 1 |
| C1-01 | contradiction | inv:grn-12, inv:mut-c1-grn-12 | 1/1/0/0 | 1 | 2 | 1 |
| C1-02 | contradiction | inv:mut-c1-vinv-cankiem, … | 1/0/0/0 | 1 | 1 | 1 |
| C2-00 | contradiction | golden:po:PO-0015:trang_thai | 1/0/0/0 | 1 | 1 | 0 |
| C2-01 | contradiction | golden:apr:APR-0102:trang_thai | 0/0/0/0 | 0 | 0 | 0 |
| H1-02 | hidden | golden:inv:INV-0031:trang_thai | 0/0/0/0 | 0 | 0 | 0 |
| H1-03 | hidden | golden:pay:PAY-0031:trang_thai | 0/0/0/0 | 0 | 0 | 0 |
| H1-04 | hidden | golden:apr:APR-0210:muc_tu_hanh | 0/0/0/0 | 0 | 0 | 0 |
| H2-00 | hidden | inv:mut-h2-PAID-PAY-0031 | 0/1/0/0 | 1 | 1 | 1 |
| H2-01 | hidden | inv:mut-h2-REJECTED-APR-0155 | 0/1/0/0 | 1 | 1 | 1 |
| W1-04 | weak | vinv.so_tien | 0/0/0/0 | 0 | 0 | 0 |
| W2-01 | weak | evt:FLAGGED:ITEM-0012 | 0/0/0/0 | 0 | 0 | 0 |
| W2-03 | weak | evt:RECEIVED:GRN-0015 | 0/0/0/0 | 0 | 0 | 0 |
| W2-04 | weak | evt:RECONCILED:VINV-0015 | 0/0/0/0 | 0 | 0 | 0 |
| W2-05 | weak | evt:DRAFTED:PAY-0031 | 0/0/0/0 | 0 | 0 | 0 |

The base specification's report holds 91 findings: 12 contradiction, 78 hidden, 0 ambiguity and 1 weak.

## What failed, by type

- **Ambiguity (F1 0): the detector never fired.** On no specification, base or mutant, did any item reach P(|A_s| ≥ 2) ≥ 0.5. So no ambiguity finding existed and ddmin never ran. Spec §3.2 is the first suspect: with the prior P(|A| = 1) = 0.9 and noisy 7–14B interpreters, a genuine split reads as interpreter error.
  - A second suspect is the operators. The A truths edit KB and PRINCIPLES text that the interpreters may never retrieve through `search_spec`.
  - Not measured: whether any A-mutant run retrieved the edited chunk.
- **Weak (F1 0; 2 FP, both on A1 mutants at `happy|P2:decide_quote`): the detector never fired on a W mutant.** W1 and W2 remove a check. For a weak finding, the population must split into a clean deviant class beside an admissible class on the unchecked field. The interpreters behaved uniformly on those fields, so nothing split.
- **Contradiction:**
  - **Scores:** SpecDx typed F1 is 0.75 (3/5, no false positives) and untyped F1 is 0.89 (4/5). B2 untyped F1 is 0.75 (3/5).
  - **Result:** SpecDx typed exactly ties B2 untyped, and the rule needs strictly greater, so SpecDx does not win.
  - **C1-00:** found but typed Hidden; a refutation succeeded.
  - **C2-01:** masked. The base report already holds its truth location as a contradiction.
- **Hidden:**
  - **Typed F1:** 0.44. The 2 H2 truths are found. The 2 false positives are the Hidden-typed findings on C1-00 and C1-01.
  - **Untyped F1:** SpecDx and B2 tie at 0.57.
  - **Masking:** all 3 H1 truths are masked for every method. The base report already holds them as hidden, because the interpreters already miss those golden values on the unmutated specification, and the §4.3 key differential then removes them. Hidden recall is therefore capped at 2/5 for every method.

## What this means for §3.2–§3.4 (spec §4.8: revisit before any full run)

1. **Base noise is the dominant problem.** The base specification yields 91 findings, the interpreters' shared failures. They mask H1 truths and leave little signal for the typed detectors.
   - Candidates: a stronger interpreter population, or a §3.2 model that separates "every interpreter fails" from "the specification requires something no reading supports".
2. **The ambiguity and weak detectors (§3.2 set-valued truth, §3.4 scores) did not fire at all** at this population size and noise level. They need either a different prior or threshold, or evidence that the A and W operators produce observable splits. Check the second first, by measuring retrieval of the edited chunks and the field-level spread on W mutants.
3. **Contradiction typing works** (3/5 typed, no false positives) but adds nothing over B2's detection here. B2 cannot type at all, which the literal typed-vs-typed table shows.

Raw data: `p1-gate-mutants.jsonl` (one line per mutant, plus the base) and `p1-gate.json`. Any rule can be rescored from these without new runs.
