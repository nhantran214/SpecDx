from specdx.spec.columns import FILLED, ColumnClass, ColumnModel

BLANK = {
    ("soline", "don_gia"), ("soline", "chiet_khau"), ("po", "don_gia"), ("inv", "so_tien"),
    ("vinv", "so_tien"), ("pay", "so_tien"), ("apr", "gia_tri"), ("cand", "muc_offer"),
    ("cand", "khung_max"),
}
IGNORED = {("apr", "noi_dung"), ("apr", "created_at"), ("evt", "basis"), ("evt", "at"),
           ("evt", "id")}
CASE_TABLES = ("cases", "so", "soline", "inv", "po", "grn", "insp", "ncr", "ship", "vinv",
               "pay", "cand", "apr", "evt")


def test_case_table_columns_are_classified_as_pinned(pack) -> None:
    model = ColumnModel.of(pack)
    for (table, column), cls in model.classes.items():
        if table not in CASE_TABLES:
            continue
        if (table, column) in BLANK:
            assert cls is ColumnClass.BLANK, (table, column)
        elif (table, column) in IGNORED:
            assert cls is ColumnClass.IGNORED, (table, column)
        else:
            assert cls is ColumnClass.LOCKED, (table, column)


def _tables(**changes):
    base = {
        "so": [{"ma_so": "SO-0031", "ma_kh": "CUST-0007", "case_id": "CASE-0031",
                "trang_thai": "Chờ duyệt"}],
        "pay": [{"ma_pay": "PAY-0031", "ma_vinv": "VINV-0015", "so_tien": "",
                 "trang_thai": "Chờ duyệt"}],
        "evt": [{"id": "1", "case_id": "CASE-0031", "kind": "FLAGGED", "ref": "ITEM-0012",
                 "actor": "AI", "basis": "a", "at": "t1"}],
    }
    for table, row in changes.items():
        base[table] = [row]
    return base


def test_free_text_and_timestamps_do_not_separate_behaviours(pack) -> None:
    model = ColumnModel.of(pack)
    other = _tables(evt={"id": "9", "case_id": "CASE-0031", "kind": "FLAGGED",
                         "ref": "ITEM-0012", "actor": "AI", "basis": "zzz", "at": "t9"})
    entries = ["so", "pay", "evt=ITEM-0012"]
    assert model.project(_tables(), entries) == model.project(other, entries)


def test_a_locked_value_separates_behaviours_and_is_named(pack) -> None:
    model = ColumnModel.of(pack)
    other = _tables(so={"ma_so": "SO-0031", "ma_kh": "CUST-0007", "case_id": "CASE-0031",
                        "trang_thai": "Đã duyệt"})
    a = model.project(_tables(), ["so"])
    b = model.project(other, ["so"])
    assert a != b
    assert model.diff(a, b) == frozenset({"so.trang_thai"})


def test_a_filled_blank_column_is_one_value_whatever_it_holds(pack) -> None:
    model = ColumnModel.of(pack)
    one = _tables(pay={"ma_pay": "PAY-0031", "ma_vinv": "VINV-0015", "so_tien": "0",
                       "trang_thai": "Chờ duyệt"})
    two = _tables(pay={"ma_pay": "PAY-0031", "ma_vinv": "VINV-0015", "so_tien": "999",
                       "trang_thai": "Chờ duyệt"})
    assert model.project(one, ["pay"]) == model.project(two, ["pay"])
    assert ("so_tien", FILLED) in model.project(one, ["pay"])[0][1][0][1]
    assert model.diff(model.project(_tables(), ["pay"]),
                      model.project(one, ["pay"])) == frozenset({"pay.so_tien"})


def test_a_missing_event_is_named_by_kind_and_ref(pack) -> None:
    model = ColumnModel.of(pack)
    without = _tables()
    without["evt"] = []
    a = model.project(_tables(), ["evt=ITEM-0012"])
    b = model.project(without, ["evt=ITEM-0012"])
    assert model.diff(a, b) == frozenset({"evt:FLAGGED:ITEM-0012"})


def test_a_keyed_entry_keeps_only_that_row(pack) -> None:
    model = ColumnModel.of(pack)
    tables = {"apr": [
        {"ma_apr": "APR-0102", "loai": "q", "trang_thai": "Đã duyệt"},
        {"ma_apr": "APR-0155", "loai": "p", "trang_thai": "Chờ duyệt"},
    ]}
    projection = model.project(tables, ["apr=APR-0102"])
    assert [key for key, _ in projection[0][1]] == ["APR-0102"]
