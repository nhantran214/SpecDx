"""Mutation operators for an executable business specification (spec §4.3, §11 item 6).

Each operator copies the base pack, applies one edit, and states its ground truth. State
operators also give a kill witness: a state that proves the defect is real.
"""

from __future__ import annotations

import copy
import random
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tonomyx.canon.pack import CanonPack

from specdx.db import golden_tables, is_background, pack_digest
from specdx.mutate.pack_copy import (
    append_text,
    copy_pack,
    edit_csv_rows,
    edit_json,
    regenerate_sql,
    replace_text,
)
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import Chunk, TellCorpus
from specdx.types import State, Target
from specdx.variant import SpecVariant

OPERATOR_TYPES: dict[str, str] = {
    "C1": "contradiction", "C2": "contradiction", "H1": "hidden", "H2": "hidden",
    "A1": "ambiguity", "A2": "ambiguity", "A3": "ambiguity", "W1": "weak", "W2": "weak",
}
C1_SITES = ("po-15", "grn-12", "vinv-cankiem", "apr0102-approved", "apr0155-approved",
            "apr0210-pending")
C2_SITES = (
    ("po", "ma_po", "PO-0015", "Đã duyệt", ("golden:po:PO-0015:trang_thai",)),
    ("apr", "ma_apr", "APR-0102", "Đã duyệt",
     ("golden:apr:APR-0102:trang_thai", "inv:apr0102-approved")),
    ("apr", "ma_apr", "APR-0155", "Đã duyệt",
     ("golden:apr:APR-0155:trang_thai", "inv:apr0155-approved")),
)
H1_SITES = (
    ("vinv", "ma_vinv", "VINV-0015", "trang_thai", "Nháp"),
    ("inv", "ma_inv", "INV-0031", "doi_soat", "Khớp"),
    ("inv", "ma_inv", "INV-0031", "trang_thai", "Chờ duyệt"),
    ("pay", "ma_pay", "PAY-0031", "trang_thai", "Nháp"),
    ("apr", "ma_apr", "APR-0210", "muc_tu_hanh", "L2"),
)
H2_SITES = (("PAID", "PAY-0031"), ("REJECTED", "APR-0155"))
A1_SITES = (
    "**mức tối đa 5%**.",
    "Yêu cầu vượt trần (vd 8%) phải gắn cờ ngoại lệ và dừng ở cổng P2 (APR-0102);",
    "GRN lệch / lô lỗi / giao trễ: P4 xử lý ngoại lệ — KHÔNG mở APR chuẩn.",
)
A2_SITES = (
    ("từ 100 sản phẩm giảm 5%;", "từ 100 sản phẩm giảm ở mức hợp lý;"),
    ("**mức tối đa 5%**.", "**mức tối đa hợp lý**."),
    ("CASE-0031 có ĐÚNG 3 cổng: APR-0102/P2, APR-0155/P6 (maker-checker), "
     "APR-0210/P5 (đối soát lệch).", "CASE-0031 có một số cổng phê duyệt phù hợp."),
)
# The steps each edited rule governs (spec v2 §4.1), one tuple per site above. The
# discount rules govern quoting and its approval; the GRN-exception rule governs the
# receipt, the reconciliation and the payment that could open an approval; the gate
# count governs every step that opens or decides one of the three gates.
QUOTE_STEPS = ("DE-SM-02:draft_quote", "P2:decide_quote")
A1_STEPS: tuple[tuple[str, ...], ...] = (
    QUOTE_STEPS,
    QUOTE_STEPS,
    ("DE-OP-02:record_goods_receipt", "DE-FN-02:reconcile", "DE-FN-03:draft_payment"),
)
A2_STEPS: tuple[tuple[str, ...], ...] = (
    ("DE-SM-02:draft_quote",),
    QUOTE_STEPS,
    (*QUOTE_STEPS, "DE-FN-04:draft_purchase_order", "P6:decide_po", "DE-FN-03:draft_payment"),
)
A3_STEPS: tuple[str, ...] = QUOTE_STEPS
A3_ARTICLE = (
    "---\nkb_id: KB-0009\nversion: 1\neffective_date: 2026-09-01\ndomain: sales\n"
    "title: Chính sách giá & chiết khấu (cập nhật)\n---\n"
    "Mức chiết khấu tối đa là 7%; yêu cầu đến 7% được áp dụng trực tiếp, "
    "không cần dừng ở cổng duyệt.\n"
)
W1_SITES = (
    ("soline", "chiet_khau", "chietkhau-0"), ("soline", "don_gia", "soline-dongia-blank"),
    ("po", "don_gia", "po-dongia-blank"), ("inv", "so_tien", "inv-sotien-blank"),
    ("vinv", "so_tien", "vinv-sotien-blank"), ("pay", "so_tien", "pay-sotien-blank"),
    ("apr", "gia_tri", "apr-giatri-blank"),
)
W2_SITES = (("DRAFTED", "SO-0031"), ("FLAGGED", "ITEM-0012"), ("DRAFTED", "PO-0015"),
            ("RECEIVED", "GRN-0015"), ("RECONCILED", "VINV-0015"), ("DRAFTED", "PAY-0031"))
_TRIGGER = """
-- specdx mutant C2: forbid a status the golden state requires
CREATE OR REPLACE FUNCTION fn_specdx_c2() RETURNS trigger AS $$
BEGIN
  IF NEW.{key} = '{value}' AND NEW.trang_thai = '{status}' THEN
    RAISE EXCEPTION 'specdx mutant: % may not be %', NEW.{key}, NEW.trang_thai;
  END IF;
  RETURN NEW;
END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_specdx_c2 ON {table};
CREATE TRIGGER trg_specdx_c2 BEFORE INSERT OR UPDATE ON {table}
  FOR EACH ROW EXECUTE FUNCTION fn_specdx_c2();
"""


@dataclass(frozen=True)
class GroundTruth:
    """observed_at: what the truth needs the population to reach (spec v2 §4.1). It is
    benchmark metadata for scorability, never an input to any method."""

    mutant_id: str
    type: str
    location: frozenset[str]
    location_kind: str
    observed_at: tuple[Target, ...] = ()


@dataclass(frozen=True)
class Mutant:
    id: str
    operator: str
    site: str
    variant: SpecVariant
    truth: GroundTruth


def _kb_files_with(pack: Path, text: str) -> list[str]:
    return [f"kb/{p.name}" for p in sorted((pack / "kb").glob("*.md"))
            if text in p.read_text(encoding="utf-8")]


def _edit_principles(pack: Path, old: str, new: str | None) -> None:
    def fn(meta: dict) -> dict:
        items = [p for p in meta["nguyen_tac"] if p != old]
        if new is not None and len(items) < len(meta["nguyen_tac"]):
            items = [new if p == old else p for p in meta["nguyen_tac"]]
        return {**meta, "nguyen_tac": items}
    edit_json(pack, "canon.meta.json", fn)


def _corpus(pack: Path, pdf: Sequence[Chunk]) -> TellCorpus:
    return TellCorpus.build(CanonPack(pack), pdf)


def _steps(ids: Sequence[str]) -> tuple[Target, ...]:
    return tuple(Target("step", s) for s in ids)


def _rows(pack: CanonPack, table: str) -> tuple[Target, ...]:
    """The golden case rows of a table (background rows belong to no case)."""
    keys = ColumnModel.of(pack).keys[table]
    return tuple(Target("row", table, ":".join(r[k] for k in keys))
                 for r in golden_tables(pack)[table] if not is_background(table, r))


def _settle(staged: Path, final: Path) -> Path:
    """A mutant pack is written once. A later build must reproduce it byte for byte, and is
    then left alone, so runs cached against it stay valid; a differing build is an error,
    never an overwrite."""
    if final.exists():
        if pack_digest(final) != pack_digest(staged):
            raise ValueError(f"{final} differs from the pack this code builds; it was made by "
                             "other code or edited by hand. Move it aside to rebuild it.")
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged, final)
    return final


def all_mutants(base: SpecVariant, pdf: Sequence[Chunk], work_dir: Path) -> list[Mutant]:
    """Every mutant, its pack under work_dir/mutants/<id>. Packs are built in a staging
    directory and settled there once (_settle)."""
    with tempfile.TemporaryDirectory() as staging:
        return _build(base, pdf, work_dir, Path(staging))


def _build(base: SpecVariant, pdf: Sequence[Chunk], work_dir: Path, staging: Path) -> list[Mutant]:
    out: list[Mutant] = []
    base_pack = CanonPack(base.pack)

    def fresh(mid: str) -> Path:
        return copy_pack(base.pack, staging / mid / base.pack.name)

    def _mutant(mid: str, op: str, site: str, pack: Path, corpus: TellCorpus,
                location: frozenset[str], kind: str, observed_at: tuple[Target, ...]) -> Mutant:
        final = _settle(pack, work_dir / "mutants" / mid / base.pack.name)
        return Mutant(mid, op, site, SpecVariant(mid, final, corpus),
                      GroundTruth(mid, OPERATOR_TYPES[op], location, kind, observed_at))

    for i, site in enumerate(C1_SITES):
        mid, pack = f"C1-{i:02d}", fresh(f"C1-{i:02d}")
        inv = next(x for x in CanonPack(pack).invariants if x["id"] == site)
        value = str(inv["assert"]["eq"])
        alt = str(int(value) + 3) if value.isdigit() else f"{value} (khác)"
        added = {"id": f"mut-c1-{site}", "desc": f"{inv['desc']} (đột biến)",
                 "table": inv["table"], "where": copy.deepcopy(inv["where"]),
                 "assert": {"col": inv["assert"]["col"], "eq": alt}}
        edit_json(pack, "invariants.json",
                  lambda d, a=added: {**d, "invariants": [*d["invariants"], a]})
        # Both invariants read the row the site's invariant selects.
        row = Target("row", inv["table"], ":".join(str(v) for v in inv["where"].values()))
        out.append(_mutant(mid, "C1", site, pack, _corpus(pack, pdf),
                           frozenset({f"inv:{site}", f"inv:mut-c1-{site}"}), "conflict", (row,)))

    for i, (table, key, value, status, location) in enumerate(C2_SITES):
        mid, pack = f"C2-{i:02d}", fresh(f"C2-{i:02d}")
        append_text(pack, "targets/postgres/guardrails.sql",
                    _TRIGGER.format(key=key, value=value, status=status, table=table))
        out.append(_mutant(mid, "C2", value, pack, _corpus(pack, pdf),
                           frozenset(location), "conflict", (Target("row", table, value),)))

    for i, (table, key, ident, column, value) in enumerate(H1_SITES):
        mid, pack = f"H1-{i:02d}", fresh(f"H1-{i:02d}")

        def change(rows, key=key, ident=ident, column=column, value=value):
            for r in rows:
                if r[key] == ident:
                    r[column] = value
            return rows

        edit_csv_rows(pack, table, change)
        regenerate_sql(pack)
        out.append(_mutant(mid, "H1", f"{table}.{column}", pack, _corpus(pack, pdf),
                           frozenset({f"golden:{table}:{ident}:{column}"}), "clause",
                           (Target("row", table, ident),)))

    for i, (kind, ref) in enumerate(H2_SITES):
        mid, pack = f"H2-{i:02d}", fresh(f"H2-{i:02d}")
        added = {"id": f"mut-h2-{kind}-{ref}", "desc": f"Có sự kiện {kind} cho {ref}",
                 "table": "evt", "where": {"kind": kind, "ref": ref}, "assert": {"count": 1}}
        edit_json(pack, "invariants.json",
                  lambda d, a=added: {**d, "invariants": [*d["invariants"], a]})
        # The demanded event is one T never motivates, so no admissible run writes it: the
        # truth is reached where events about its reference are written (any kind).
        out.append(_mutant(mid, "H2", f"{kind}:{ref}", pack, _corpus(pack, pdf),
                           frozenset({f"inv:mut-h2-{kind}-{ref}"}), "clause",
                           (Target("event", "", ref),)))

    for i, text in enumerate(A1_SITES):
        mid, pack = f"A1-{i:02d}", fresh(f"A1-{i:02d}")
        if not base.corpus.find(text):
            raise ValueError(f"A1 site matches no Tell sentence: {text!r}")
        for rel in _kb_files_with(pack, text):
            replace_text(pack, rel, text, "")
        _edit_principles(pack, text, None)
        corpus = _corpus(pack, pdf)
        corpus = corpus.without(corpus.find(text))
        chunks = frozenset(TellCorpus.chunk_of(s) for s in base.corpus.find(text))
        if not chunks:
            raise ValueError(f"{mid} has an empty ground truth")
        out.append(_mutant(mid, "A1", text, pack, corpus, chunks, "chunk", _steps(A1_STEPS[i])))

    for i, (old, new) in enumerate(A2_SITES):
        mid, pack = f"A2-{i:02d}", fresh(f"A2-{i:02d}")
        matched = False
        for rel in _kb_files_with(pack, old):
            if replace_text(pack, rel, old, new):
                matched = True
        old_canon_meta = (pack / "canon.meta.json").read_text(encoding="utf-8")
        _edit_principles(pack, old, new)
        new_canon_meta = (pack / "canon.meta.json").read_text(encoding="utf-8")
        if old_canon_meta != new_canon_meta:
            matched = True
        corpus = _corpus(pack, pdf)
        if corpus.find(old):
            matched = True
        for sid in corpus.find(old):
            corpus = corpus.replace(sid, new)
        truth = frozenset(corpus.find(new))
        if not matched or not truth:
            raise ValueError(f"A2 site matches nothing: {old!r}")
        out.append(_mutant(mid, "A2", old, pack, corpus, truth, "sentence", _steps(A2_STEPS[i])))

    mid, pack = "A3-00", fresh("A3-00")
    (pack / "kb" / "KB-0009.md").write_text(A3_ARTICLE, encoding="utf-8")
    corpus = _corpus(pack, pdf)
    truth = frozenset(corpus.sentence_ids(["KB:KB-0009"]))
    if not truth:
        raise ValueError(f"{mid} has an empty ground truth")
    out.append(_mutant(mid, "A3", "KB-0009", pack, corpus, truth, "sentence", _steps(A3_STEPS)))

    for i, (table, column, inv_id) in enumerate(W1_SITES):
        mid, pack = f"W1-{i:02d}", fresh(f"W1-{i:02d}")
        edit_json(pack, "invariants.json", lambda d, x=inv_id: {
            **d, "invariants": [v for v in d["invariants"] if v["id"] != x]})

        def unblank(meta, table=table, column=column):
            blank = dict(meta["cot_de_trong_khong_canon"])
            blank[table] = [c for c in blank[table] if c.split()[0] != column]
            return {**meta, "cot_de_trong_khong_canon": blank}

        edit_json(pack, "canon.meta.json", unblank)
        out.append(_mutant(mid, "W1", f"{table}.{column}", pack, _corpus(pack, pdf),
                           frozenset({f"{table}.{column}"}), "field", _rows(base_pack, table)))

    for i, (kind, ref) in enumerate(W2_SITES):
        mid, pack = f"W2-{i:02d}", fresh(f"W2-{i:02d}")
        edit_csv_rows(pack, "evt", lambda rows, k=kind, r=ref: [
            x for x in rows if not (x["kind"] == k and x["ref"] == r)])
        regenerate_sql(pack)
        out.append(_mutant(mid, "W2", f"{kind}:{ref}", pack, _corpus(pack, pdf),
                           frozenset({f"evt:{kind}:{ref}"}), "field",
                           (Target("event", kind, ref),)))
    return out


def dev_split(mutants: Sequence[Mutant], per_type: int, seed: int) -> list[Mutant]:
    by_type: dict[str, list[Mutant]] = defaultdict(list)
    for m in sorted(mutants, key=lambda m: m.id):
        by_type[m.truth.type].append(m)
    rng = random.Random(seed)
    chosen: list[Mutant] = []
    for kind in sorted(by_type):
        pool = by_type[kind]
        rng.shuffle(pool)
        chosen += pool[:per_type]
    return sorted(chosen, key=lambda m: m.id)


def witness_state(mutant: Mutant, golden: State) -> State | None:
    """A state proving a state-level defect: the canon's own end state, edited if needed."""
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    if mutant.operator in ("C1", "H1", "H2"):
        return State(tables, dict(golden.derived), dict(golden.steps))
    if mutant.operator == "W1":
        table, column = mutant.site.split(".")
        for r in tables[table]:
            r[column] = "1"
        return State(tables, dict(golden.derived), dict(golden.steps))
    if mutant.operator == "W2":
        kind, ref = mutant.site.split(":")
        tables["evt"] = [e for e in tables["evt"] if not (e["kind"] == kind and e["ref"] == ref)]
        return State(tables, dict(golden.derived), dict(golden.steps))
    return None
