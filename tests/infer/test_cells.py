"""Cells, part 1 (spec v3 §2.5–§2.6): normalisation and row matching on the real base pack."""

import unicodedata

import pytest
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.db import golden_tables, split_background
from specdx.infer.act import drafting_steps
from specdx.infer.cells import (
    CellData,
    CellLens,
    executor_generated,
    lift_reading,
    match_columns,
    normalise,
    occurs,
    table_rows,
)
from specdx.infer.items import SpecLens
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import Chunk, Sentence
from specdx.types import Behaviour, InterpreterId, Scenario, State


@pytest.mark.parametrize(("raw", "column_type", "expected"), [
    (unicodedata.normalize("NFD", "Nháp"), "text", "nháp"),
    ("  Chờ \t  duyệt\n", "text", "chờ duyệt"),
    ("Draft", "text", "draft"), ("DRAFT", "text", "draft"),
    ("35", "int", "35"), ("35.0", "int", "35"), ("35,0", "numeric", "35"),
    ("0.50", "numeric", "0.5"), ("350", "int", "350"), ("-0", "int", "0"), ("+7", "int", "7"),
    ("2026-09-01", "date", "2026-09-01"),
    ("t", "bool", "true"), ("TRUE", "bool", "true"), ("f", "bool", "false"),
    ("t", "text", "t"),
    # Numerals in text columns keep their form (zero-padding preserved).
    ("0015", "text", "0015"), ("007", "text", "007"),
    # Decimal-comma convention: 1,000 is 1 (not 1000); 1.000 would be 1000.
    ("1,000", "numeric", "1"),
    # bigint columns are numeric too.
    ("35.0", "bigint", "35"),
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


# Part 2 (spec v3 §2.2–§2.4, §2.7): cells, anchors, scope, pooling, readings.

GRN12 = Scenario("grn12", 15, 12, 0)
HAPPY = Scenario("happy", 15, 15, 0)
BIG = Scenario("o16-r16-d0-g1", 16, 16, 0)


@pytest.fixture(scope="module")
def cell_lens(pack, tell_corpus) -> CellLens:
    steps = load_work_orders()
    background, _ = split_background(golden_tables(pack))
    lens = SpecLens(ClauseSet.from_pack(pack, steps), steps, ColumnModel.of(pack),
                    drafting=drafting_steps(pack, steps), initial=background)
    return CellLens(lens, pack, tell_corpus)


def _run(key: str, model: str, scenario: Scenario, edit=None, idle=()) -> Behaviour:
    """A behaviour whose end state is the canon's golden state (every case row written, so
    it acts at every interpreter step but those in idle), after edit(tables)."""
    tables = {t: [dict(r) for r in rows] for t, rows in golden_tables(CanonPack(BASE_PACK)).items()}
    if edit is not None:
        edit(tables)
    steps = {s.id: ("error" if s.id in idle else "done") for s in load_work_orders()}
    return Behaviour(key, InterpreterId(model, model[0], 0, 1), scenario,
                     State(tables, {"fn_reconcile:PO-0015": "Cần kiểm"}, steps))


def _set(table: str, column: str, value: str, where=None):
    def edit(tables):
        for row in tables[table]:
            if where is None or all(row[k] == v for k, v in where.items()):
                row[column] = value
    return edit


def _values(data: CellData, cell: str) -> dict[str, str]:
    return {r.run_key: r.value for r in data.by_cell()[cell]}


def test_a_golden_value_is_anchored_where_its_clause_applies_and_unanchored_elsewhere(
    cell_lens,
) -> None:
    data = cell_lens.extract([
        _run("a", "m1", GRN12), _run("b", "m2", GRN12, _set("so", "trang_thai", "Draft")),
        _run("c", "m1", HAPPY)])
    anchored = data.cells["grn12|so:SO-0031:trang_thai"]
    assert (anchored.kind, anchored.anchor, anchored.clause, anchored.location) == (
        "value", "chờ duyệt", "golden:so:SO-0031:trang_thai", "so.trang_thai")
    assert anchored.group == "so.trang_thai"  # difficulty per column, over rows and scenarios
    assert anchored.steps == ("DE-SM-02:draft_quote",)
    assert _values(data, anchored.id) == {"a": "chờ duyệt", "b": "draft"}
    elsewhere = data.cells["happy|so:SO-0031:trang_thai"]
    assert elsewhere.anchor is None and elsewhere.clause is None


def test_a_lifted_golden_cell_is_one_cell_across_scenarios_in_the_lifted_vocabulary(
    cell_lens,
) -> None:
    data = cell_lens.extract([
        _run("a", "m1", GRN12),                                        # 15 = ordered
        _run("b", "m1", HAPPY),                                        # 15 = ordered = received
        _run("c", "m1", BIG, _set("po", "sl_dat", "16")),              # 16 = ordered
        _run("d", "m2", GRN12, _set("po", "sl_dat", "12")),            # 12 = received
        _run("e", "m2", BIG, _set("po", "sl_dat", "15")),              # no parameter
        _run("f", "m2", HAPPY, lambda t: t.update(po=[]))])            # no row
    cell = data.cells["golden:po:PO-0015:sl_dat"]
    assert cell.anchor == "{ordered}" and cell.clause == cell.id
    assert _values(data, cell.id) == {"a": "{ordered}", "b": "{ordered}", "c": "{ordered}",
                                      "d": "{received}", "e": "15", "f": "absent"}
    assert not [c for c in data.cells if c.endswith("|po:PO-0015:sl_dat")]


def test_lift_reading_keeps_text_and_prefers_the_anchor_on_a_tie() -> None:
    assert lift_reading("15", "int", HAPPY, "{ordered}") == "{ordered}"
    assert lift_reading("15", "int", HAPPY, "{received}") == "{received}"
    assert lift_reading("12", "int", GRN12, "{ordered}") == "{received}"
    assert lift_reading("7", "int", GRN12, "{ordered}") == "7"
    assert lift_reading("absent", "int", GRN12, "{ordered}") == "absent"
    assert lift_reading("draft", "text", GRN12, "{ordered}") == "draft"


def test_cell_scope_excludes_rows_inherited_through_a_shared_slice(cell_lens) -> None:
    # Only P6:decide_po acted: its slice holds the whole po table, but P6 drafts nothing,
    # so it reads no po value cell; po-15 is relevant to it, so it reads that clause.
    drafting = [s.id for s in load_work_orders() if s.kind in ("de", "persona")
                and s.id != "P6:decide_po"]
    data = cell_lens.extract([_run("a", "m1", GRN12, idle=drafting)])
    assert {r.cell for r in data.readings if ":PO-0015:" in r.cell} == set()
    assert "inv:po-15" in {r.cell for r in data.readings}
    assert data.cells["grn12|apr:APR-0155:trang_thai"].steps == (
        "DE-FN-04:draft_purchase_order", "P6:decide_po")


def test_soline_cells_follow_the_order_and_item_not_the_surrogate_id(cell_lens) -> None:
    def invented(tables):
        for i, row in enumerate(tables["soline"]):
            row["ma_soline"] = f"SOL-{i + 1:03d}"
        tables["soline"][1]["sl"] = "14"

    data = cell_lens.extract([_run("a", "m1", GRN12, invented)])
    assert _values(data, "grn12|soline:SOLINE-0031-1:sl") == {"a": "20"}
    assert _values(data, "golden:soline:SOLINE-0031-2:sl") == {"a": "14"}
    # key and matching columns are row identity, not readings (§2.6, resolution 3)
    assert not [c for c in data.cells
                if "soline:" in c and c.rsplit(":", 1)[1] in ("ma_soline", "so", "ma_sp")]
    assert [c for c in data.cells if c.startswith("grn12|soline:")]


def test_an_unmatched_row_gives_unanchored_cells(cell_lens) -> None:
    def extra(tables):
        tables["po"].append({**tables["po"][0], "ma_po": "PO-0016", "sl_dat": "5"})

    data = cell_lens.extract([_run("a", "m1", GRN12, extra), _run("b", "m2", GRN12)])
    cell = data.cells["grn12|po:po-0016:sl_dat"]
    assert cell.anchor is None and cell.location == "po.sl_dat"
    assert _values(data, cell.id) == {"a": "5", "b": "absent"}


def test_events_read_present_or_absent_and_a_golden_event_is_anchored(cell_lens) -> None:
    def logged(tables):
        tables["evt"].append({**tables["evt"][0], "actor": "DE-SM-02"})

    def dropped(tables):
        tables["evt"] = [e for e in tables["evt"] if e["kind"] != "DRAFTED"]

    data = cell_lens.extract([_run("a", "m1", GRN12, logged), _run("b", "m2", GRN12, dropped)])
    golden = data.cells["grn12|evt:DRAFTED:SO-0031:AI"]
    assert (golden.kind, golden.anchor, golden.clause, golden.location, golden.group) == (
        "event", "present", "event:DRAFTED:SO-0031:AI", "evt:DRAFTED:SO-0031", "evt:DRAFTED")
    assert _values(data, golden.id) == {"a": "present", "b": "absent"}
    observed = data.cells["grn12|evt:DRAFTED:SO-0031:DE-SM-02"]
    assert observed.anchor is None and _values(data, observed.id) == {"a": "present",
                                                                       "b": "absent"}


def test_constraint_cells_are_pooled_and_read_once_per_behaviour(cell_lens) -> None:
    data = cell_lens.extract([_run("a", "m1", GRN12), _run("b", "m1", HAPPY),
                              _run("c", "m2", HAPPY, _set("po", "sl_dat", "9"))])
    cell = data.cells["inv:po-15"]
    assert (cell.kind, cell.anchor, cell.location, cell.group) == (
        "constraint", "satisfied", "inv:po-15", "inv:po-15")  # difficulty per clause
    assert _values(data, "inv:po-15") == {"a": "satisfied", "b": "satisfied", "c": "violated"}
    assert all(c.kind != "constraint" or c.id == c.clause for c in data.cells.values())
    assert not [c for c in data.cells if c.startswith("vocab:")]


def test_only_acted_behaviours_read_and_clean_marks_a_violation_free_item(cell_lens) -> None:
    every = [s.id for s in load_work_orders()]
    data = cell_lens.extract([
        _run("idle", "m1", GRN12, idle=every),
        _run("ok", "m1", GRN12),
        _run("bad", "m2", GRN12, _set("so", "trang_thai", "Draft"))])
    assert {r.run_key for r in data.readings} == {"ok", "bad"}
    clean = {r.run_key: r.clean for r in data.by_cell()["grn12|so:SO-0031:trang_thai"]}
    assert clean == {"ok": True, "bad": False}


def test_extraction_does_not_depend_on_the_order_of_behaviours(cell_lens) -> None:
    runs = [_run("a", "m1", GRN12), _run("b", "m2", HAPPY, _set("so", "trang_thai", "x")),
            _run("c", "m2", GRN12, _set("po", "sl_dat", "12"))]
    assert cell_lens.extract(runs) == cell_lens.extract(runs[::-1])
