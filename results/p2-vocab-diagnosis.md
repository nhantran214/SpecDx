# SpecDx P2a — vocabulary diagnosis of the base hidden findings (read-only)

Date: 2026-10-19. HEAD e9aac9b, frozen `V3Params` (pi 0.2, s_mis 0.4, s_amb 0.3, s_empty 0.3,
beta_sd 1.0, mu_alpha 1.0, sigma_alpha 1.0, tau_h 0.5, tau_a 0.3, tau_w 0.9).
Data: P1b smoke run, base specification only (60 runs; gpt-oss:20b and qwen3:30b-a3b). No mutant was
built, read or run, no LLM was called and no database was opened.

Cache fingerprints, before and after every run:
- `md5sum work/runs.jsonl` = `db1e1823c66df102648c69eb6a8789b3` (expected value);
- `ls -la --time-style=full-iso work | md5sum` = `5f3760e7fce8cbac7e53a22ad24f35f8` (expected value).

The worktree is unchanged (`git status` clean). This file is the only file written in it.

Scratch scripts are in `(scratch path)`
(`run_base.py`, `analyse.py`, `extra.py`, `tquote.py`; raw output in `analysis.out` and `extra.out`).
They import `experiments/p2_replay.py` and call its `replay()`, which reproduces G1 exactly:
- θ: gpt-oss 0.989, qwen3 0.948;
- C/H/A/W: 0/53/19/4.

Definitions:
- **Canon vocabulary** of a `table.column`: the distinct values of that column across every golden row
  of the operational tables (`specdx.db.golden_tables`), normalised by §2.5.
- `absent`: the row is not in the end state. It is not counted as a value.
- `∅str`: an empty-string reading.

## 0. Verdict

**The hypothesis is mostly false for hidden findings.**
- Only **4 of 53** hidden findings (7.5%) have a MAP value outside the canon vocabulary.
- Only **1** of the 4 (`inv.trang_thai`: `draft` for `Nháp`) is a pure English translation.
- Status columns account for **10 of 53** hidden findings. Of these, 3 are `apr` rows that were never
  written, which has nothing to do with vocabulary.
- **43 of 53** have nothing to do with status vocabulary:
  - 18 `apr` value cells whose row is absent;
  - 8 golden events that are absent;
  - 16 violated constraint clauses;
  - `po.sl_dat`, `soline.du_hang`, `vinv.case_id` and `vinv.po`.

**The proposed rule removes no hidden finding.** The rule is "readings outside the column's domain are
errors, never in A_c".
- On the status cells almost no reading equals d_c.
- Pushing `draft` into OTHER therefore moves the MAP from `{draft}` to ∅. P(A≠{d}) stays at 1.000.
- Only the subtype changes, from mis-key to unstated.
- θ rises by about 0.004 per model.

**`draft` is not an interpreter artefact. T states it.**
- T uses `draft` as a status label for exactly these rows, for example "SO-0031 draft", "INV-0031 ở
  trạng thái draft" and "PAY-0031 draft".
- T's data dictionary lists status enums that contain `draft` (BA:076).
- C (the canon CSVs) says `Chờ duyệt`/`Nháp`. So the status findings are a real T↔C inconsistency in the
  specification, not a measurement error.

## 1. Every cell typed hidden (P1b smoke, base, frozen parameters)

- Each hidden cell finding is reported as its own defect: the 53 cell findings give 53 reported hidden
  defects.
- "MAP vs vocab" is `n/a` for constraint and event cells and for an ∅ MAP. A lifted placeholder is also
  `n/a`.
- Readings are the top four values per model. gpt is gpt-oss:20b and qwen is qwen3:30b-a3b.

| # | cell | kind | group | d_c | MAP | P(A≠{d}) | subtype | readings per model | MAP vs canon vocab |
|---:|---|---|---|---|---|---:|---|---|---|
| 1 | `expect:apr_0102:discount8` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×9; qwen: `violated`×6 | n/a |
| 2 | `expect:open_units:discount8` | constraint | same | `satisfied` | {`satisfied`, `violated`} | 1.000 | ambiguous-pick | gpt: `violated`×8, `satisfied`×1; qwen: `satisfied`×6, `violated`×3 | n/a |
| 3 | `expect:open_units:grn12` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×9; qwen: `violated`×7, `satisfied`×2 | n/a |
| 4 | `expect:open_units:happy` | constraint | same | `satisfied` | {`satisfied`, `violated`} | 1.000 | ambiguous-pick | gpt: `violated`×9; qwen: `satisfied`×5, `violated`×4 | n/a |
| 5 | `expect:three_way:discount8` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×8; qwen: `violated`×3 | n/a |
| 6 | `expect:three_way:grn12` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×8; qwen: `violated`×3, `satisfied`×1 | n/a |
| 7 | `expect:three_way:happy` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×9; qwen: `violated`×4 | n/a |
| 8 | `expect:three_way:in` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×28, `satisfied`×1; qwen: `violated`×7, `satisfied`×4 | n/a |
| 9 | `golden:po:PO-0015:sl_dat` | value | po.sl_dat | `{ordered}` | {} | 1.000 | unstated | gpt: `1`×11, `35`×11, `{discount_pct}`×4, `{ordered}`×2; qwen: `absent`×10, `{ordered}`×7, `1`×3, `0`×2 | n/a (MAP ∅; numeric) |
| 10 | `grn12\|apr:APR-0102:case_id` | value | apr.case_id | `case-0031` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 11 | `grn12\|apr:APR-0102:hoso_ref` | value | apr.hoso_ref | `so-0031` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 12 | `grn12\|apr:APR-0102:loai` | value | apr.loai | `báo giá cần duyệt (p2)` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 13 | `grn12\|apr:APR-0102:muc_tu_hanh` | value | apr.muc_tu_hanh | `l3` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 14 | `grn12\|apr:APR-0102:nguoi_tao` | value | apr.nguoi_tao | `de-sm-02` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 15 | `grn12\|apr:APR-0102:trang_thai` | value | apr.trang_thai | `đã duyệt` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 16 | `grn12\|apr:APR-0155:case_id` | value | apr.case_id | `case-0031` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 17 | `grn12\|apr:APR-0155:hoso_ref` | value | apr.hoso_ref | `po-0015` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 18 | `grn12\|apr:APR-0155:loai` | value | apr.loai | `đơn mua cần duyệt (p6, maker-checker)` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 19 | `grn12\|apr:APR-0155:muc_tu_hanh` | value | apr.muc_tu_hanh | `l3` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 20 | `grn12\|apr:APR-0155:nguoi_tao` | value | apr.nguoi_tao | `de-fn-04` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 21 | `grn12\|apr:APR-0155:trang_thai` | value | apr.trang_thai | `đã duyệt` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | absent |
| 22 | `grn12\|apr:APR-0210:case_id` | value | apr.case_id | `case-0031` | {`absent`} | 1.000 | mis-key | gpt: `absent`×5; qwen: `absent`×1 | absent |
| 23 | `grn12\|apr:APR-0210:hoso_ref` | value | apr.hoso_ref | `pay-0031` | {`absent`} | 1.000 | mis-key | gpt: `absent`×5; qwen: `absent`×1 | absent |
| 24 | `grn12\|apr:APR-0210:loai` | value | apr.loai | `đề nghị chi cần duyệt (p5)` | {`absent`} | 1.000 | mis-key | gpt: `absent`×5; qwen: `absent`×1 | absent |
| 25 | `grn12\|apr:APR-0210:muc_tu_hanh` | value | apr.muc_tu_hanh | `l3` | {`absent`} | 1.000 | mis-key | gpt: `absent`×5; qwen: `absent`×1 | absent |
| 26 | `grn12\|apr:APR-0210:nguoi_tao` | value | apr.nguoi_tao | `de-fn-03` | {`absent`} | 1.000 | mis-key | gpt: `absent`×5; qwen: `absent`×1 | absent |
| 27 | `grn12\|apr:APR-0210:trang_thai` | value | apr.trang_thai | `chờ duyệt` | {`absent`} | 1.000 | mis-key | gpt: `absent`×5; qwen: `absent`×1 | absent |
| 28 | `grn12\|evt:APPROVED:APR-0102:Người` | event | evt:APPROVED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×8; qwen: `absent`×9 | n/a |
| 29 | `grn12\|evt:APPROVED:APR-0155:Người` | event | evt:APPROVED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×8; qwen: `absent`×9 | n/a |
| 30 | `grn12\|evt:DRAFTED:PAY-0031:AI` | event | evt:DRAFTED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×5; qwen: `absent`×1 | n/a |
| 31 | `grn12\|evt:DRAFTED:PO-0015:AI+` | event | evt:DRAFTED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×8; qwen: `absent`×9 | n/a |
| 32 | `grn12\|evt:DRAFTED:SO-0031:AI` | event | evt:DRAFTED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×7 | n/a |
| 33 | `grn12\|evt:FLAGGED:ITEM-0012:AI` | event | evt:FLAGGED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×9; qwen: `absent`×9 | n/a |
| 34 | `grn12\|evt:RECEIVED:GRN-0015:AI+` | event | evt:RECEIVED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×7; qwen: `absent`×7 | n/a |
| 35 | `grn12\|evt:RECONCILED:VINV-0015:AI+` | event | evt:RECONCILED | `present` | {`absent`} | 1.000 | mis-key | gpt: `absent`×8; qwen: `absent`×4 | n/a |
| 36 | `grn12\|inv:INV-0031:doi_soat` | value | inv.doi_soat | `chưa` | {} | 1.000 | unstated | gpt: `∅str`×4, `1`×3, `chưa`×1, `draft`×1; qwen: `∅str`×6, `n/a`×1, `chưa`×1 | n/a (MAP ∅; 15/17 readings OUT) |
| 37 | `grn12\|inv:INV-0031:trang_thai` | value | inv.trang_thai | `nháp` | {`draft`} | 1.000 | mis-key | gpt: `draft`×7, `nháp`×1, `∅str`×1; qwen: `draft`×8 | **OUT** |
| 38 | `grn12\|pay:PAY-0031:trang_thai` | value | pay.trang_thai | `chờ duyệt` | {`draft`} | 0.997 | ambiguous-pick | gpt: `draft`×4, `chờ duyệt`×1; qwen: `chờ duyệt`×1 | **OUT** |
| 39 | `grn12\|po:PO-0015:trang_thai` | value | po.trang_thai | `đã duyệt` | {} | 1.000 | unstated | gpt: `draft`×5, `nháp`×3; qwen: `draft`×4, `absent`×3, `chờ duyệt`×2 | n/a (MAP ∅; 14/17 readings OUT) |
| 40 | `grn12\|so:SO-0031:trang_thai` | value | so.trang_thai | `chờ duyệt` | {`draft`} | 1.000 | mis-key | gpt: `draft`×7, `nháp`×2; qwen: `draft`×7 | **OUT** |
| 41 | `grn12\|soline:SOLINE-0031-1:du_hang` | value | soline.du_hang | `true` | {`false`} | 1.000 | mis-key | gpt: `false`×6, `true`×3; qwen: `false`×7 | in |
| 42 | `grn12\|vinv:VINV-0015:case_id` | value | vinv.case_id | `case-0031` | {`absent`, `case-0031`} | 0.938 | ambiguous-pick | gpt: `case-0031`×6, `absent`×2; qwen: `absent`×3, `∅str`×1 | absent / in |
| 43 | `grn12\|vinv:VINV-0015:doi_soat` | value | vinv.doi_soat | `cần kiểm` | {} | 1.000 | unstated | gpt: `∅str`×4, `absent`×2, `1`×1, `-12`×1; qwen: `absent`×3, `cần kiểm`×1 | n/a (MAP ∅; 6/12 OUT) |
| 44 | `grn12\|vinv:VINV-0015:po` | value | vinv.po | `po-0015` | {`absent`, `po-0015`} | 1.000 | ambiguous-pick | gpt: `po-0015`×6, `absent`×2; qwen: `absent`×3, `po-0015`×1 | absent / in |
| 45 | `grn12\|vinv:VINV-0015:trang_thai` | value | vinv.trang_thai | `chờ duyệt` | {`absent`, `cần kiểm`} | 1.000 | ambiguous-pick | gpt: `cần kiểm`×6, `absent`×2; qwen: `absent`×3, `cần kiểm`×1 | absent / **OUT** |
| 46 | `inv:apr0102-approved` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×9; qwen: `violated`×9 | n/a |
| 47 | `inv:apr0155-approved` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×9; qwen: `violated`×9 | n/a |
| 48 | `inv:apr0210-pending` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×5; qwen: `violated`×1 | n/a |
| 49 | `inv:item0005-du` | constraint | same | `satisfied` | {`satisfied`, `violated`} | 1.000 | ambiguous-pick | gpt: `violated`×6, `satisfied`×3; qwen: `violated`×7 | n/a |
| 50 | `inv:item0012-thieu` | constraint | same | `satisfied` | {`satisfied`, `violated`} | 1.000 | ambiguous-pick | gpt: `satisfied`×6, `violated`×3; qwen: `satisfied`×4, `violated`×3 | n/a |
| 51 | `inv:po-15` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×30, `satisfied`×2; qwen: `violated`×21, `satisfied`×7 | n/a |
| 52 | `inv:vinv-cankiem` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×8; qwen: `violated`×3, `satisfied`×1 | n/a |
| 53 | `reconcile:consistency` | constraint | same | `satisfied` | {`violated`} | 1.000 | mis-key | gpt: `violated`×18, `satisfied`×2; qwen: `violated`×4, `satisfied`×2 | n/a |

Three of the constraint clauses test status values on a row that is absent here:
- `inv:apr0102-approved` requires `apr.trang_thai = Đã duyệt`.
- `inv:apr0155-approved` requires `apr.trang_thai = Đã duyệt`.
- `inv:apr0210-pending` requires `apr.trang_thai = Chờ duyệt`.

Every reading of those `apr` rows is `absent`, so these clauses fail because the row is missing, not
because of its wording. `inv:vinv-cankiem` requires `vinv.doi_soat = Cần kiểm`. There the readings are
mostly blank or absent, not English words.

## 2. Aggregates

### 2.1 Hidden findings by kind and cause (53 in total)

| cause | n |
|---|---:|
| value cell, MAP {`absent`} (row not written; all 18 are `apr` rows APR-0102/0155/0210) | 18 |
| constraint clause violated | 16 |
| golden event absent | 8 |
| value cell, MAP holds an out-of-vocabulary value | **4** |
| value cell, MAP ∅ (unstated) | 4 |
| value cell, MAP pair {absent, in-vocab value} | 2 |
| value cell, MAP an in-vocabulary other value (`soline.du_hang`: false for true) | 1 |

### 2.2 Hidden findings per table.column, clause or event

- `apr.case_id`, `apr.hoso_ref`, `apr.loai`, `apr.muc_tu_hanh`, `apr.nguoi_tao` and `apr.trang_thai`:
  3 each, 18 in all.
- One each:
  - `inv.doi_soat`, `inv.trang_thai`, `pay.trang_thai`, `po.trang_thai`, `so.trang_thai`;
  - `vinv.case_id`, `vinv.doi_soat`, `vinv.po`, `vinv.trang_thai`;
  - `po.sl_dat`, `soline.du_hang`.
- Eight event cells: `evt:APPROVED` ×2, `evt:DRAFTED` ×3, `evt:FLAGGED`, `evt:RECEIVED`, `evt:RECONCILED`.
- Sixteen constraint cells:
  - `expect:three_way` ×4;
  - `expect:open_units` ×3;
  - `expect:apr_0102:discount8`;
  - `inv:*` ×7;
  - `reconcile:consistency`.

### 2.3 Out-of-vocabulary MAP values

- Out of all 53 hidden findings: **4 (7.5%)**.
- Out of the 29 value cells: 4 of 29.
- Out of the 11 value cells whose MAP is neither {absent} nor ∅ and that are not apr rows: 4 of 11.

The four cells:

| cell | d_c | MAP | is the MAP value an English translation of d_c? |
|---|---|---|---|
| `inv.trang_thai` | `nháp` | `draft` | **yes**: T itself glosses "draft (nháp)" (BA:012:06, BA:027:00) |
| `so.trang_thai` | `chờ duyệt` | `draft` | no: T's lifecycle has nháp (draft) → chờ duyệt (pending) as different states. T does say "SO-0031 draft" (DEMO:018:18) |
| `pay.trang_thai` | `chờ duyệt` | `draft` | no: same reason. T says "PAY-0031 draft" (DEMO:023:10) |
| `vinv.trang_thai` | `chờ duyệt` | {absent, `cần kiểm`} | no: Vietnamese, and the value belongs to `vinv.doi_soat`. This is column confusion |

Three more status hidden cells have an ∅ MAP but readings that are mostly out of vocabulary:
- `po.trang_thai`: `draft`/`nháp`/`chờ duyệt` for `đã duyệt`; 14 of 17 readings out of vocabulary.
- `inv.doi_soat`: mostly blank or `1`; 15 of 17.
- `vinv.doi_soat`: blank or numeric; 6 of 12.

Counting these too gives 7 of 53 hidden cells whose evidence is dominated by out-of-vocabulary readings.
Of these, only `inv.trang_thai` and part of `po.trang_thai` (`draft` vs `nháp`) involve translation.

### 2.4 Mappings observed

These are counts of out-of-vocabulary readings at anchored value cells over the whole smoke fit, not
only the hidden cells:

| written | where d_c is | readings | kind of mismatch |
|---|---|---:|---|
| `draft` | `chờ duyệt` (so, pay) | 18 | English word, different lifecycle state (T: nháp (draft) → chờ duyệt (pending)) |
| `draft` | `nháp` (inv) | 15 | **English translation** |
| `draft` | `đã duyệt` (po) | 9 | English word, different state |
| `nháp` | `đã duyệt` (po) | 3 | Vietnamese, different state |
| `nháp` | `chờ duyệt` (so) | 2 | Vietnamese, different state |
| `chờ duyệt` | `đã duyệt` (po) | 2 | Vietnamese, different state |
| `cần kiểm` | `chờ duyệt` (vinv.trang_thai) | 7 | doi_soat value written into trang_thai |
| `∅str` / `1` / `draft` / `n/a` | `chưa` (inv.doi_soat) | 10 / 3 / 1 / 1 | blank or boolean-like, not translation |
| `∅str` / `1` / `-12` | `cần kiểm` (vinv.doi_soat) | 4 / 1 / 1 | blank or a numeric difference |
| `1`, `35`, `0`, … | `{ordered}` (po.sl_dat) | 32 | numeric, not vocabulary |

The full list of English words observed anywhere in the status columns, unanchored cells included:
- `draft`: in every trang_thai column except apr and vinv;
- `pending`: qwen ×2, `inv.doi_soat`;
- `reconciled`: gpt ×3 `vinv.trang_thai`, ×1 `vinv.doi_soat`;
- `flagged`: qwen ×2, `vinv.trang_thai`;
- `discrepancy`: qwen ×2, `vinv.doi_soat`.

`approved` was never observed. The only translation pair in the strict sense is `draft`↔`nháp`.

### 2.5 Ambiguity and weak findings

**Ambiguity.** There are 45 cell-level findings, which merge into 19 reported ambiguities.
- 8 of the 45 cells contain an out-of-vocabulary value:
  - `po.trang_thai` ×2: {draft, nháp} and {chờ duyệt, draft}, against po's vocabulary {đã duyệt};
  - `pay.trang_thai` ×1: {chờ duyệt, draft};
  - `vinv.trang_thai` ×3: {absent, cần kiểm};
  - `vinv.doi_soat` ×2: {∅str, absent}.
- These are 4 of the 19 reported locations: `po.trang_thai`, `pay.trang_thai`, `vinv.trang_thai`,
  `vinv.doi_soat`.
- The remaining 37 cells are:
  - 24 event present/absent cells;
  - 4 constraint cells;
  - 9 value cells: vinv.case_id/po, soline.du_hang/sl, pay.ma_vinv.

**Weak.** There are 6 cell-level findings, which merge into 4 reported weak checks.
- 3 of the 6 contain an out-of-vocabulary value:
  - `inv.trang_thai` (happy): draft, ∅str and nháp;
  - `pay.trang_thai` ×2 (happy, discount8): draft and chờ duyệt.
- These are 2 of the 4 reported locations: `inv.trang_thai` and `pay.trang_thai`.
- The other 3 cells are event cells.

**Ambiguity and weak checks are where vocabulary matters proportionally more than for hidden
findings.**

## 3. Status columns (`*.trang_thai`, `*.doi_soat`)

| | cells | anchored | hidden | ambiguity (cell) | weak (cell) |
|---|---:|---:|---:|---:|---:|
| `trang_thai` (apr, inv, pay, po, so, vinv) | 38 | 8 | 8 | 6 | 3 |
| `doi_soat` (inv, vinv) | 9 | 2 | 2 | 2 | 0 |
| total | 47 | 10 | **10** | 8 | 3 |

**Every anchored status cell is typed hidden.**
- 3 of them (`apr.trang_thai`) are absent rows.
- The other 7 are wording or status disagreements.

What the models write, over all readings of all status cells, with the canon vocabulary per column:

| column | canon vocab | gpt-oss:20b | qwen3:30b-a3b |
|---|---|---|---|
| `apr.trang_thai` | {chờ duyệt, đã duyệt} | `absent`×83 | `absent`×62 |
| `inv.trang_thai` | {nháp} | `draft`×22, `nháp`×6, `∅str`×3 | `draft`×27 |
| `pay.trang_thai` | {chờ duyệt} | `draft`×17, `chờ duyệt`×1, `absent`×1 | `chờ duyệt`×5, `draft`×1 |
| `po.trang_thai` | {đã duyệt} | `absent`×37, `draft`×18, `nháp`×9, `∅str`×1, `chờ duyệt`×1 | `absent`×37, `draft`×15, `chờ duyệt`×9, `nháp`×4 |
| `so.trang_thai` | {chờ duyệt} | `draft`×24, `nháp`×3, `∅str`×1 | `draft`×21 |
| `vinv.trang_thai` | {chờ duyệt} | `cần kiểm`×17, `absent`×17, `reconciled`×3 | `absent`×8, `cần kiểm`×3, `flagged`×2, `∅str`×1 |
| `inv.doi_soat` | {chưa} | `∅str`×15, `1`×6, `chưa`×5, `draft`×4, `false`×1 | `∅str`×15, `0`×7, `pending`×2, `n/a`×1, `chưa`×1, `false`×1 |
| `vinv.doi_soat` | {cần kiểm} | `absent`×17, `∅str`×15, `cần kiểm`×2, `1`×1, `-12`×1, `reconciled`×1 | `absent`×8, `cần kiểm`×4, `discrepancy`×2 |

`draft` is the majority non-absent status of both models in so, inv and po, and of gpt-oss in pay.

## 4. Does the specification declare a value domain for these columns?

**In C, the canon pack `vendor/tonomyx/data/canon_pack_CASE-0031_v2.0/canon_pack_CASE-0031/`: no.**
- `targets/postgres/ddl.sql` declares every `trang_thai`/`doi_soat` as plain `TEXT`, with no CHECK and
  no enum type.
- `schema.json` gives only name, type, pk, nullable and fk. There are no enum keys.
- `targets/postgres/guardrails.sql` mentions the literals `'Đã chi'`, `'Đã duyệt'` and `'Cần kiểm'`
  inside the `fn_guard_pay` trigger, and `'Khớp'`/`'Cần kiểm'` in `fn_reconcile`. These are point
  tests, not a domain.
- The only enforced vocabulary is `evt.kind`: `data/event_kinds.json` and the `evt_kind` table with its
  trigger in `guardrails.sql`.
- `invariants.json` asserts point values: `apr0102-approved`, `apr0155-approved` and `apr0210-pending`
  for `apr.trang_thai`, `esg01-thieu`, the `pending-2` count, and `no-paid` (`ne 'Đã chi'`).
- `targets/tonomyx/tool-catalog.json` (stack-dependent per `canon.meta.json`) says "legal transitions"
  and `APR 'Chờ duyệt'` without listing a domain.

**Outside the pack, in vendored app code (not the specification):**
- `vendor/tonomyx/tonomyx/tools/impl/update_status.py` `STATUS_FLOW`: `Chờ duyệt → {Đã duyệt, Từ chối}`,
  and `Đã chi` for pay.
- `write_draft.py` `FINAL_STATUSES`.
- `policy/rules.py`: APPROVED, REJECTED, PENDING.

Neither of these contains `Nháp`.

**In T, the Tell corpus that the interpreters search: yes, informally, and it contains `draft`.**
- BA:076:05 (PAY): "status enum ✔ draft / chờ duyệt / đã duyệt / đã chi".
- BA:076:02: "status enum draft / đã duyệt / đã ghi nhận".
- BA:076:08 (PO): "status … (draft/duyệt/đã đặt/đã nhận)".
- BA:076:12 (APR): "status enum ✔ pending / approved / rejected".
- BA:027:00: "Vòng đời trạng thái: nháp (draft) → chờ duyệt (pending) → đã duyệt (approved) / từ chối
  (rejected) → hoàn tất (done)".
- DEMO:015:25: "Nhãn trạng thái: draft, chờ duyệt, đã duyệt, đã gửi, khớp, ngoại lệ, thiếu dữ liệu, bị
  chặn".
- Per row:
  - DEMO:018:18 "SO-0031 draft gắn CASE-0031";
  - DEMO:021:03 / BA:120:16 "SO-0031 và QUOTE-0031 ở trạng thái draft/chờ duyệt";
  - DEMO:022:36 "INV-0031 ở trạng thái draft";
  - DEMO:023:10 "PAY-0031 draft".

**Consequence.** A rule "readings outside the column's declared domain are errors" cannot be built from
C's declarations, because C declares none.
- From the golden rows alone, the domain of a single-row column is just {d_c}. The rule then reduces to
  "A_c ⊆ {d_c}". That is circular, and it would blind the method to mis-key defects on those columns,
  including any mutant that edits a golden status.
- The union of the canon's status values per column name is computable automatically:
  - `trang_thai`: chờ duyệt, nháp, thiếu dữ liệu, trễ, đang làm, đang xét, đang xử lý, đã duyệt, đủ;
  - `doi_soat`: chưa, cần kiểm.
- From T, a domain could be extracted only by parsing prose tables. That domain would **include**
  `draft`, so the rule would not reject the dominant reading at all.

## 5. What would change: counterfactual refits (scratch only, worktree untouched)

**Method.**
- Readings at value cells of the named columns that fall outside the domain are relabelled so that
  `candidate_values` skips them. They land in the OTHER bucket, so they can never be in A_c.
- `absent` and lifted placeholders are always kept.
- The fit is `fit_anchored` with the frozen parameters, followed by `diagnose_v3` (NoRefutation), as in
  G1.

| variant | readings moved to OTHER | hidden gone | hidden new | reported C/H/A/W | θ gpt-oss | θ qwen3 |
|---|---:|---:|---:|---|---|---|
| base (G1) | 0 | — | — | 0/53/19/4 | 0.989 | 0.948 |
| R1: status cols, golden vocab, blank kept | 223 | **0** | 1 | 0/54/16/2 | 0.993 | 0.952 |
| R1s: as R1, blank also out | 274 | 0 | 1 | 0/54/15/2 | 0.993 | 0.953 |
| R2: every text value column, golden vocab | 224 | 0 | 1 | 0/54/16/2 | 0.993 | 0.952 |
| R2s: as R2, blank also out | 281 | 0 | 1 | 0/54/15/2 | 0.993 | 0.953 |
| R3: status cols, union of the canon's status values for the column name | — | 0 | 1 | 0/54/16/2 | 0.993 | 0.952 |
| T1: semantic map draft→nháp, pending→chờ duyệt, approved→đã duyệt (not §2.5-compliant) | — | **1** (`inv.trang_thai`) | 1 | 0/53/19/5 | 0.992 | 0.948 |
| U: status-column cells dropped entirely (upper bound) | — | **10** | 1 | 0/44/15/2 | 0.994 | 0.956 |

### 5.1 Why the domain rule removes nothing

Take `so.trang_thai`: 16 readings, 0 equal to `chờ duyệt`.
- Once `draft` cannot be in A_c, the best explanation is A_c = ∅: every reading is an error from the
  flat ε.
- A_c = {d_c} would need about 16 errors from models with θ ≈ 0.95–0.99.
- So the MAP changes, as follows:

| cell | MAP before → after (R1) | subtype after |
|---|---|---|
| `inv.trang_thai` | {draft} → ∅ | unstated |
| `so.trang_thai` | {draft} → ∅ | unstated |
| `pay.trang_thai` | {draft} → ∅ | unstated |
| `vinv.trang_thai` | {absent, cần kiểm} → ∅ | unstated |

- P(A≠{d}) stays at 1.000 for every one of them.
- A hidden finding can disappear only if readings move onto d_c. That requires a semantic mapping, and
  only `draft`→`nháp` is a true translation: T1 removes 1 finding.
- The 3 `apr.trang_thai` cells are absent rows, so no vocabulary rule touches them.

### 5.2 The new hidden finding

`grn12|soline:SOLINE-0031-2:du_hang` appears in every variant.
- Base: d_c `false`, MAP {false}, P(A≠{d}) = 0.162 (amb 0.159).
- Readings: gpt false×6, true×3; qwen false×4, absent×3.
- The re-weighting of α/β pushes it over τ_H = 0.5. It is a side-effect, not a vocabulary finding.

### 5.3 Other effects

- Ambiguity and weak findings react more than hidden findings. Under R1:
  - the 6 status-column ambiguity cells drop out, and `soline.du_hang` (grn12) and `vinv.case_id`
    (o15) come in;
  - the 3 status-column weak cells drop out.
- θ is insensitive: +0.003 to +0.005 per model under every variant, and +0.005/+0.008 even when the
  status cells are dropped entirely. The 223–281 relabelled readings are about 5–6% of 4572 readings.
  They are concentrated on 6–9 cells, whose groups take most of the misfit through β.

## 6. Reading for the pre-registration

1. A vocabulary rule would not fix "many" hidden findings. At most it would fix 1 (with a semantic map)
   or 10 (by removing the status columns entirely).
   - The bulk of the 53 is structural: 18 apr rows never written, 8 absent golden events and 16
     violated constraints.
   - These come from the `apr`/event scope and from the expectations, not from wording.
   - That is where a pre-registration caveat is needed, if any.
2. The status findings are defensible as **real T↔C inconsistencies**, not artefacts.
   - T tells the interpreter "SO-0031 draft", "INV-0031 … draft" and "PAY-0031 draft".
   - T declares status enums that contain `draft`.
   - C stores `Chờ duyệt`/`Nháp`.
   - `po.trang_thai = Đã duyệt` is the post-approval state of a cell that the drafting step DE-FN-04
     alone is scoped to read (§2.4), so its anchor timing deserves a separate look.
3. A domain rule automatable "from the spec alone" has no declared source in C.
   - Golden-row vocabularies are circular for single-row tables.
   - T's informal enums would admit `draft`.
   - If one is wanted anyway, the union of the canon's own values per column name (R3) is the only
     automatic candidate. It changes subtypes (mis-key → unstated) and ambiguity/weak counts, not the
     hidden count. It also changes θ by under 0.005.
