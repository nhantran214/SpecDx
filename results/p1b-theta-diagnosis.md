# Diagnosis: why stage 2 gives θ ≈ 0.08 in the P1b smoke run (base)

Date: 2026-10-14. Worktree HEAD c22db31. Read-only analysis: no code or data changed, no database, no Ollama. `work/runs.jsonl` md5 was checked before and after (unchanged, 6505 lines).

## Verdict: (d), a mix that is mostly (a)

- **(a) Genuine (decisive).** The two models disagree with each other and with themselves on substantive fields of what they draft. Examples are `po.sl_dat` ∈ {35, 15, 1, 0, 10, 5, 20}, whether the APPROVED event is logged, `Chờ duyệt` against `Nháp`, and wrong PO ids. Most of this disagreement is seed-level: within one model, pairs of runs on the same Tell variant agree barely more often than pairs on different variants.
- **(b) Class-key artefact (real, but not decisive).**
  - What enters the class key:
    - surrogate ids that the models invent and T does not fix (`soline.ma_soline`, 29 distinct values in 84 rows);
    - case and language variants of status strings (`draft`/`Draft`/`DRAFT`/`Nháp`);
    - placeholder values in columns that T leaves open at that step (`inv.doi_soat`);
    - rows that other steps wrote into a shared slice.
  - Where it enters:
    - `specdx/spec/columns.py:53–76` locks every non-serial, non-timestamp, non-free-text column at its exact string;
    - `specdx/infer/items.py:139–140` hashes the end-state projection.
  - Removing all of this lifts θ only to 0.38 / 0.10 (ceiling 0.61 / 0.51). θ stays below 0.5.
- **(c) No code bug.**
  - The code computes exactly the pre-registered formulas.
  - EM reaches the unique global MAP: the result is the same from any start, and the profile likelihood confirms it.
  - There is, however, a structural property of the pre-registered stage-2 model (§3.5). It makes θ ≈ "the share of a model's acted runs that fall in the plurality joint class". Resolution 4's stated rationale does not hold on this data.

**Gate prognosis.** Both θ are below 0.5, so S2 and the ddmin oracle are inert. Ambiguity cannot be reported (S1 ≤ 0.119 everywhere). Weak findings follow plurality noise, and Hidden is reachable only at uninformative items. Contradiction is unaffected. The Task 18 θ-ordering pre-check passes formally (0.097 > 0.078), but a bootstrap gives P(θ_gpt > θ_qwen) = 0.48: the ordering is a coin flip.

---

## 0. Method and reproduction

- **Smoke population (60 runs).** It was rebuilt exactly as `Runner.key` keys runs, without the database. Keys are `run_key(base digest, steps digest, current context digest, params, interp.key, "", model digest)` for the 4 report scenarios × 18 committee members. The 60 runs come from lines 6319–6505:
  - 42 smoke population runs logged in `work/p1b-smoke.log`;
  - 18 `happy` runs reused from the capability screen (cache hits). This is why `executed` is 142 = 42 + 100 probes.
- **P1 base population (200 runs).** It was mapped the same way as `experiments/p1b_replay.py:158–181` (P1 context digest, schema 2).
- **Exact reproduction.**
  - Smoke: θ = 0.0967 / 0.0781 and κ = 2.1464 / 1.3210, with 36 items, all observable. This matches `results/p1b-smoke-base.json`.
  - P1: θ = 0.9608 / 0.6095 / 0.7430. This matches `results/p1b-replay.md`.
- **Scripts.** All are in `experiments/p1b_theta_diagnosis/` (archived; see its README):
  - `load.py` (mapping) and `common.py`;
  - `classes.py`, `dump.py`, `values.py`, `fields.py` (data);
  - `trace.py`, `profile.py`, `epsplug.py`, `sens.py`, `cap.py`, `summ.py` (EM);
  - `normalise.py`, `peritem.py`, `weak.py`, `weights.py` (counterfactuals).

## 1. The data at each item

### Smoke run: 449 acted views, 36 items

| | smoke (P1b) | P1 base (replay) |
|---|---:|---:|
| acted views / items with any | 449 / 36 | 473 / 63 |
| distinct classes, summed over items | 247 | 126 |
| modal-class share (all acted views) | **0.425** | 0.865 |
| within-model pairwise agreement (same item) | **0.296** (443/1495) | 0.816 |
| between-model pairwise agreement | **0.268** (429/1599) | 0.849 |
| unanimous items | 4 (all DE-OP-01, empty projection) | 34 |

Per model, counting items where the model has at least 3 acted runs:

| model | items | mean acted | mean distinct classes | items where its own modal class holds > 50% of its runs | items where every run is its own class |
|---|---:|---:|---:|---:|---:|
| gpt-oss:20b | 36 | 7.03 | 4.31 | 12 | 3 |
| qwen3:30b-a3b | 25 | 7.48 | 4.80 | 10 | **11** |
| *P1* llama3.1:8b | 14 | 5.50 | 1.00 | 14 | 0 |
| *P1* qwen2.5:7b | 21 | 6.67 | 2.10 | 21 | 0 |
| *P1* qwen3:14b | 24 | 7.00 | 1.79 | 24 | 0 |

Between the two smoke models, 25 items have at least 3 acted runs from each, and the models share a modal class at only 11 of them.

The drafting items are where readings live, and they are the most fragmented. In the table below, a cell such as "9/7/2" reads as n / classes / modal count.

| item | n | classes | modal | gpt-oss (n/classes/modal) | qwen3 (n/classes/modal) |
|---|---:|---:|---:|---|---|
| discount8\|DE-FN-04:draft_purchase_order | 17 | 14 | 2 | 8/7/2 | 9/9/1 |
| grn12\|DE-FN-04:draft_purchase_order | 17 | 15 | 2 | 8/7/2 | 9/9/1 |
| grn12\|P6:decide_po | 17 | 16 | 2 | 8/8/1 | 9/9/1 |
| grn12\|DE-SM-02:draft_quote | 16 | 14 | 3 | 9/8/2 | 7/7/1 |
| happy\|DE-SM-02:draft_quote | 14 | 12 | 3 | 7/6/2 | 7/7/1 |
| discount8\|DE-FN-02:reconcile | 11 | 9 | 2 | 8/6/2 | 3/3/1 |
| happy\|DE-FN-01:draft_sales_invoice | 18 | 10 | 6 | 9/7/3 | 9/4/4 |
| happy\|DE-OP-02:record_goods_receipt | 15 | 2 | 12 | 9/2/6 | 6/1/6 |
| any\|DE-OP-01:check_availability | 18 | 1 | 18 | 9/1/9 | 9/1/9 |

`DE-OP-01`'s slice (`evt=ITEM-0005`, `evt=ITEM-0012`) is never written in any of the 60 runs. Its single class `d443fc6e5489920b` is the empty projection, so it carries no reading.

**Disagreement is seed noise as much as paraphrase.** Within a model, pairs on the same Tell variant agree at 0.309 (gpt-oss) and 0.333 (qwen3). Pairs on different variants agree at 0.278 and 0.301.

## 2. Does the class key carry run-specific noise?

### Excluded correctly (checked)

These sources of noise do not enter the class key:

- serial ids (`evt.id`);
- timestamps (`evt.at`, `apr.created_at`);
- the free text `apr.noi_dung` and `evt.basis` (`columns.py:20–22, 58`);
- row order (`columns.py:88` sorts rows, and `:70` sorts cells);
- duplicate rows: no slice holds duplicate events in these runs;
- whitespace and Unicode-punctuation variants: none were found;
- the hash itself, `sha1(repr(projection))` (`items.py:140`), which is deterministic. The same projection gives the same key across models; for example, `b45182a660cecb06` is shared by gpt-oss and qwen3.

### Included, and unrelated or only weakly related to reading T

Each cell is compared as an exact string (`columns.py:73`):

1. **Surrogate line ids, `soline.ma_soline`.** There are 29 distinct values in 84 rows, for example `SOLINE-001`, `SOL-001`, `SO-0031-ITEM-0005`, `SOLINE-0031a`, `SO-0031-LINE1`, `SO-0031-L01`, `DRAFT-SOLINE-1` and `SL-0001`.
   - The canon uses `SOLINE-0031-1/-2`. T's PDFs show only `SOLINE-0031a/b` and `SOLINE-xxxx`, so T does not fix this id.
   - It is a primary key, so it is LOCKED by v1 §2.2. It also becomes the row key, so `diff` reports every soline column as different.
   - **Example (`happy|DE-SM-02:draft_quote`).** Classes `b45182a660cecb06` (SOLINE-001/002), `11bae7c9fecf2ad6` (SO-0031-ITEM-0005/0012), `bd761bcac75323c6` (SO-0031-001/002), `4081a95fdd373cc8` (SO-0031-LINE1/2), `7b663e8b1408034b` (SO-0031-L1/L2) and `193e21ca91d7b2a8` (SOL-001/002) are all the same reading: `so.trang_thai=draft`, ITEM-0005 × 20 and ITEM-0012 × 15, `du_hang=false`. Only the invented line ids differ.
   - This item has 12 classes with a modal count of 3/14. Normalising case and this key leaves 3 classes with a modal count of 10/14.
2. **Case and language variants of status values.**
   - `po.trang_thai`: draft 34, Nháp 26, Draft 26, Chờ duyệt 20, DRAFT 2.
   - `so.trang_thai`: draft 68, Draft 22, Nháp 4, nháp 2.
   - `inv.trang_thai`: draft 44, Nháp 6, DRAFT 5.
   - **Example (`discount8|DE-FN-04:draft_purchase_order`).** Four gpt-oss runs all write `PO-0015, sl_dat=35`, and they fall into four classes: `ba35000bfb9c40ad` ('Nháp'), `10f5fefdb8a1973e` ('DRAFT'), `53ac3caf5b661187` ('draft') and `3d27dabc43ec1db5` ('Draft').
   - The clean reading is split too. qwen3's two violation-free runs (`sl_dat=15`) are `2694783a8d9609a2` ('Draft') and `b63175fcd3b1a98e` ('Nháp').
   - The golden checks do not flag `Draft` (V(b) = ∅), but the class key separates the two.
3. **Placeholder values in columns T leaves open at that step.**
   - `inv.doi_soat` at the invoice draft takes the values '' 30, 0 7, 1 6, Chưa 5, draft 4, false 2, pending 2, chưa 1 and N/A 1.
   - `evt.actor` takes the values system, SYSTEM, auto-reconciler, user-p6 and DE-OP.
   - `inv.doi_soat` alone separates 186 pairs of classes that are otherwise equal.
4. **Inheritance through shared slices.** By design (§3.1; `items.py:139` projects the end state):
   - `P6:decide_po`'s slice includes the whole `po` table that `DE-FN-04` drafts. `apr=APR-0155` never exists in any run. As a result, P6's classes mirror DE-FN-04's: at discount8 and happy, both have 14 classes with top counts [2, 2, 2, 1, 1, 1]. P6 adds only "APPROVED:APR-0155 logged or not".
   - `DE-FN-04`'s class also includes `FLAGGED:PO-0015`, which the later step `DE-FN-02` writes. Run `e83a84c566dd1c14` (qwen3 v1 s1) carries it, together with an invented `ma_po='DE-FN-04'`.

### Field accounting over all 2222 disagreeing same-item pairs

- **As-is.** The separating fields are:
  - `po.sl_dat` 33.5%;
  - `po.trang_thai` 32.9%;
  - `evt:APPROVED` 21.2%;
  - `so.trang_thai` 19.5%;
  - `inv.doi_soat` 14.7%;
  - the po row key 13.3%;
  - the soline row key 12.6%.
- **After normalisation T4 (see the table below).** 1937 pairs still disagree. The separating fields are:
  - `po.sl_dat` 38.5%;
  - `evt:APPROVED` 22.6%;
  - `po.trang_thai` 22.3% (`chờ duyệt` against `nháp`);
  - `inv.doi_soat` 16.8%;
  - the po row key 15.3% (`DE-FN-04`, `PO-0016`, `PO-0015-01`);
  - `soline.du_hang` 9.9%.

  These are substantive.

### Counterfactual: stage 2 refit with the same `fit()` after cumulative normalisations

"Ceiling" is (modal-class count + 6) / (n + 8), that is, θ if every modal-class run were fully faithful.

| projection | classes | modal share | within-model agreement | between-model agreement | θ gpt-oss / qwen3 | ceiling |
|---|---:|---:|---:|---:|---|---|
| T0 as-is | 247 | 0.425 | 0.296 | 0.268 | **0.097 / 0.078** | 0.471 / 0.392 |
| T1 casefold + strip | 220 | 0.454 | 0.331 | 0.295 | 0.120 / 0.074 | 0.502 / 0.417 |
| T2 + soline keyed by (so, ma_sp) | 200 | 0.486 | 0.363 | 0.322 | 0.139 / 0.071 | 0.536 / 0.441 |
| T3 + draft ≡ nháp | 176 | 0.519 | 0.395 | 0.346 | 0.306 / 0.061 | 0.571 / 0.471 |
| T4 + only the step's own events | 171 | 0.523 | 0.401 | 0.349 | 0.326 / 0.061 | 0.582 / 0.466 |
| T5 + personas keep only so/po status | 157 | 0.555 | 0.421 | 0.377 | **0.381 / 0.100** | 0.605 / 0.505 |

On the P1 data, T0–T3 change nothing (θ 0.961 / 0.610 / 0.743), and T4 gives 0.953 / 0.753 / 0.700.

## 3. Why EM lands at θ ≈ 0.08

### Trace

The trace runs `latent._item_posterior` and `latent._error_update` exactly as `fit` calls them (`latent.py:182–203`). S is the expected faithful count and n the acted count; values are gpt-oss / qwen3.

| iteration | θ in | S_m | n_m | θ out |
|---:|---|---|---|---|
| 0 | 0.700 / 0.700 | 124.35 / 91.32 | 253 / 196 | 0.4994 / 0.4771 |
| 1 | | 92.35 / 63.99 | | 0.3768 / 0.3431 |
| 2 | | 71.48 / 47.11 | | 0.2969 / 0.2603 |
| 5 | | 40.93 / 26.24 | | 0.1798 / 0.1581 |
| 10 | | 22.26 / 11.64 | | 0.1083 / 0.0865 |
| 40 (converged at 41 < 50) | | 19.25 / 9.93 | | 0.0967 / 0.0781 |

- **M-step.** θ = (S + 6) / (n + 8) (`latent.py:195–199`): (19.25 + 6) / 261 = 0.0967 and (9.93 + 6) / 204 = 0.0781. This is the pre-registered formula.
- **Start points.** θ0 ∈ {0.95, 0.9, 0.7, 0.5, 0.3, 0.1} all reach 0.0967 / 0.0781. The fixed point is unique.
- **MAP classes.** All 36 MAP sets are singletons. P(MAP) is 0.18–0.55, except DE-OP-01 at 1.0. Only 190 of the 449 acted views lie in their item's MAP class.
- **ε absorbs the mass.** Total responsibility is 29.2 of 449 (6.5% faithful; 93.5% attributed to ε).
  - At the unanimous item DE-OP-01 (18/18), ε on its class is 0.945 and responsibility is **0.092**.
  - The same one-class evidence in P1 (24/24) gets responsibility **0.771** (ε 0.867).
  - The faithful/ε split at a unanimous item simply follows θ: r ≈ θ at every θ on the profile below.
- **Re-estimation falls back to the P1 plug-in.** At the fixed point, the re-estimated ε is within total variation 0.009 (max 0.014) of P1's plug-in, the smoothed empirical distribution. Freezing ε at that plug-in gives θ 0.093 / 0.078, the same answer.
  - Resolution 4 re-estimates ε so that "agreement counts as competence". On this data it changes nothing: the fixed point is the P1 configuration.
  - On P1, by contrast, the fitted ε moved far from the plug-in (TV 0.19), and freezing ε would have given 0.918 / 0.330 / 0.515.

### Profile of the log posterior over a common θ

Here ε is fitted by the code's own EM with θ held fixed. The Dirichlet term is the +1-per-cell smoothing at `latent.py:103`.

| common θ | log-likelihood | Dirichlet(ε) | Beta(7, 3) | log posterior |
|---:|---:|---:|---:|---:|
| 0.05 | −577.5 | −354.1 | −36.2 | −967.7 |
| 0.08 | −581.2 | −353.8 | −30.6 | **−965.7** |
| 0.10 | −584.2 | −353.5 | −28.1 | −965.8 |
| 0.30 | −622.8 | −348.4 | −15.9 | −987.0 |
| 0.50 | −669.1 | −347.9 | −11.1 | −1028.1 |
| 0.75 | −753.5 | −351.9 | −9.0 | −1114.4 |

- Moving from θ 0.08 to 0.5 costs 88 nats of likelihood and gains only 25 nats from the two priors. The fit() MAP scores −965.6.
- For P1, the log-likelihood is nearly flat: −233 at 0.05, −248 at 0.3 and −246 at 0.75. There θ was set by the priors: the Dirichlet term gains 44.5 nats and the Beta term 40.7 between 0.05 and 0.75.

### Mechanism (spec-level, not code)

**Level of θ.** The generation formula (v1 §3.2, implemented at `latent.py:129–131`) gives, per model and item:

p_{m,s} = θ_m·δ_{A_s} + (1−θ_m)·ε_s

Here ε_s is item-specific, shared by all models, and allowed on admissible classes. Every (θ, ε) along a one-parameter family fits the data equally well, so the likelihood does not identify the *level* of θ. Three things pin it:
- (i) the Beta prior;
- (ii) the +1 smoothing of ε;
- (iii) the feasibility bound θ_m ≲ p̂_m(A_s) at every item.

**Why the bound binds here.** An item where a model never repeats itself caps its θ near 1/n_{m,s}:
- gpt-oss's share of runs in the MAP class has a minimum of 0.11 and a median of 0.375;
- qwen3's has a minimum of 0.00 and a median of 0.22, and is ≤ 0.2 at 12 of its 25 items.

**Result.** θ ≈ the share of a model's acted runs in the plurality joint class. The class is a joint over every locked cell of a multi-row slice, so independent variation in several fields multiplies the fragmentation.

### Two raters

- With two models, the only rater-specific signal at an item is p₁ − p₂ = (θ₁ − θ₂)(δ_A − ε). Since θ₁ ≈ θ₂, this is close to 0, and A_s is just the plurality class.
- Proposition 2′'s bound exp(−2·n_f·γ²) with n_f = 2 is at least e^{−1} ≈ 0.37 even at γ = ½: it is vacuous.
- The family effect is ±0.117 by construction.
- A bootstrap over runs within (model, scenario), R = 200, gives P(θ_gpt > θ_qwen) = 0.48, with 90% intervals of [0.04, 0.45] for gpt-oss and [0.04, 0.51] for qwen3. These intervals are biased upward, because duplicated runs agree with themselves.
- More raters would make the plurality less noisy, but would not by themselves identify the θ level under item-level shared ε.

### Restricting the items shows the same picture

| items | θ gpt-oss / qwen3 |
|---|---|
| drafting steps only | 0.109 / 0.122 |
| non-drafting steps only | 0.226 / 0.164 |
| DE-OP-01 only (empty, unanimous) | 0.878 / 0.865 |
| DE-OP-01, DE-OP-02 and DE-FN-03 | 0.652 / 0.762 |

### Code compared with the spec

- `fit` matches v2 §3.3 and resolutions 3–5:
  - Beta(7, 3) MAP;
  - acted views only;
  - ε re-estimated from 1 − r;
  - K_max = 6 with a RARE pool that is never admissible;
  - P(|A| = 1) = 0.9.
- w is uniform within A (unspecified in the spec), which is irrelevant here because every MAP set is a singleton.
- The family effect is reported only (resolution 5).
- No departure was found.

## 4. Contrast with P1

- **P1's θ measured agreement on "nothing written".** The acted, observable P1 data are DE-OP-01, P2 and P6 (8 scenarios × 3 = 24 items). Personas and DE-OP-01 act by completing (resolution 2). Their modal classes are empty projections:
  - `happy|P2` `902520aabffa805c` at 14/17 (apr, evt and so all empty);
  - `happy|P6` `ca705219731073e8` at 10/13;
  - DE-OP-01 `d443fc6e5489920b` at 24/24.
- The P1 drafting items have only 1–3 acted views each and are unobservable.
- The P1 θ of 0.96 / 0.61 / 0.74 therefore measured agreement on writing nothing, and was set by the priors on a flat likelihood.
- In the smoke run, the screened models actually draft at every step: 36 of 36 items are observable. Stage 2 now sees real readings, and each reading is encoded as a fine-grained joint key that the models seldom reproduce, even under the same paraphrase.

## 5. What this means for the gate

**The Task 18 pre-check passes, but carries no information.** The rule (`experiments/p1b-preregistration.md:96`; `docs/RUNNING.md` §6) needs the highest-κ model (gpt-oss, κ 2.146) to have θ 0.097 above the lowest other θ (0.078). It does, but the bootstrap says the ordering is noise.

With θ < 0.5 for both models, the expected effect on the four types is:

- **Ambiguity: cannot be found.**
  - S1 = P(|A| ≥ 2) is ≤ 0.119 at every item, close to the 0.1 prior (`scores.py:282`).
  - S2 drops every model (`sensitivity.py:118`).
  - The ddmin oracle gets no interpreters (`loop.py:197–198`).
  - Expected A-type recall is 0. Even with S2 enabled, same-variant agreement ≈ different-variant agreement, so paraphrase sensitivity would be swamped by seed noise.
- **Weak: driven by plurality noise.**
  - P(MAP) is 0.23–0.55, so `outside(k)` (`scores.py:159`) is high for any recurrent minority class. Any clean minority class with at least 2 runs scores ≥ 0.5.
  - The 4 base weak findings are all of this kind:
    - `Draft` against `draft` plus an event (happy|P2);
    - logging APPROVED against not logging it (discount8|P2);
    - `inv.doi_soat` '0' against '' (DE-FN-01);
    - `pay.trang_thai` 'Chờ duyệt' against 'draft' (DE-FN-03).
  - Expect false positives on mutants, and true positives only by chance.
- **Hidden: rarely reaches τ_H.** τ_H = 0.5 (`scores.py:260`) is reachable only where the set posterior is concentrated, which is effectively only DE-OP-01, whose projection is empty. Singletons refuted elsewhere are dropped, so H recall is depressed.
- **Contradiction: unaffected.** It does not depend on θ: refutation only orders interpreters by θ. Hitting-set run weights shrink to 0.018–0.055 and become nearly uniform, so they no longer separate competent runs (P1: up to 0.96).

## 6. Smallest fix for the (b) part, and what it would change

**The fix.** In `ColumnModel` (`specdx/spec/columns.py:53–76`):
- casefold and strip LOCKED cell values;
- classify `soline.ma_soline` as IGNORED and key soline rows by (so, ma_sp). This is a surrogate key that T does not fix.

**Pre-registration impact.**
- It changes ≡_S: the v1 §2.2 rule "primary and foreign keys … must be equal", resolution 3, and the pinned column-classification test.
- No numeric value in v2 §3.10 changes.
- It needs an author ruling.
- Status synonyms (draft ≡ Nháp) and keeping only the step's own events are judgement calls that would change ≡_S semantics further.

**Effect.** θ goes from 0.097 / 0.078 to 0.139 / 0.071 (T2). With every normalisation tried, it reaches 0.381 / 0.100 (T5). **None of these reaches 0.5, so none changes the gate prognosis.**

**The stage-2 model issue (§3).** It would need a spec change, for example ε supported only outside A_s, or a stronger shared prior on ε. That changes the v1 §3.2 generation model and resolution 4. Even the ideal "ε never on A" ceiling is 0.471 / 0.392 as-is, and 0.605 / 0.505 under T5.
