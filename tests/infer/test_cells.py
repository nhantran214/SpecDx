"""Cells, part 1 (spec v3 §2.5–§2.6): normalisation and row matching on the real base pack."""

import unicodedata

import pytest

from specdx.infer.cells import (
    executor_generated,
    match_columns,
    normalise,
    occurs,
    table_rows,
)
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import Chunk, Sentence


@pytest.mark.parametrize(("raw", "column_type", "expected"), [
    (unicodedata.normalize("NFD", "Nháp"), "text", "nháp"),
    ("  Chờ \t  duyệt\n", "text", "chờ duyệt"),
    ("Draft", "text", "draft"), ("DRAFT", "text", "draft"),
    ("35", "int", "35"), ("35.0", "int", "35"), ("35,0", "numeric", "35"),
    ("0.50", "numeric", "0.5"), ("350", "int", "350"), ("-0", "int", "0"), ("+7", "int", "7"),
    ("2026-09-01", "date", "2026-09-01"),
    ("t", "bool", "true"), ("TRUE", "bool", "true"), ("f", "bool", "false"),
    ("t", "text", "t"),
])
def test_normalisation_is_syntactic(raw: str, column_type: str, expected: str) -> None:
    assert normalise(raw, column_type) == expected


def test_normalisation_applies_no_semantic_mapping() -> None:
    assert normalise("draft") != normalise("Nháp")
    assert normalise("Đã duyệt") != normalise("Da duyet")


def test_a_key_occurs_only_on_token_boundaries() -> None:
    texts = [normalise("Mã dòng SOL-10 và PO-0015-01."), normalise("Đơn PO-0015.")]
    assert not occurs("SOL-1", texts) and occurs("SOL-10", texts)
    assert occurs("PO-0015", texts[1:]) and not occurs("PO-0015", texts[:1])
    assert occurs("po-0015", texts)
    assert not occurs("", texts)


def test_the_executor_generated_keys_of_the_base_pack(pack, tell_corpus) -> None:
    columns = ColumnModel.of(pack)
    generated = executor_generated(pack, columns, tell_corpus)
    assert generated == {"soline", "insp", "ncr", "ship"}
    assert match_columns("soline", pack, columns, generated) == ("so", "ma_sp")
    assert match_columns("po", pack, columns, generated) == ("ma_po",)


def test_the_rule_reads_t_only(pack, tell_corpus) -> None:
    columns = ColumnModel.of(pack)
    told = tell_corpus.with_chunk(
        Chunk("EXTRA:000", "EXTRA", (Sentence("EXTRA:000:00", "Dòng SOLINE-0031-1 của đơn."),)))
    assert "soline" not in executor_generated(pack, columns, told)


def test_soline_rows_match_on_order_and_item_not_on_the_surrogate_id(pack, tell_corpus) -> None:
    rows = table_rows(pack, ColumnModel.of(pack), tell_corpus, ["soline", "po"])
    soline, po = rows["soline"], rows["po"]
    assert soline.name({"ma_soline": "SOLINE-001", "so": "SO-0031", "ma_sp": "ITEM-0005"}) == (
        "SOLINE-0031-1", True)
    assert soline.name({"ma_soline": "SO-0031-L2", "so": "so-0031", "ma_sp": "ITEM-0012 "}) == (
        "SOLINE-0031-2", True)
    assert soline.name({"ma_soline": "SOLINE-0031-1", "so": "SO-0031", "ma_sp": "ITEM-0099"}) == (
        "so-0031/item-0099", False)
    assert po.name({"ma_po": "PO-0015"}) == ("PO-0015", True)
    assert po.name({"ma_po": "PO-0015-01"}) == ("po-0015-01", False)
