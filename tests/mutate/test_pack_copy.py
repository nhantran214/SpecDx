from specdx.config import BASE_PACK
from specdx.mutate.pack_copy import copy_pack, csv_path, edit_csv_rows, regenerate_sql


def test_an_identity_csv_edit_leaves_every_file_byte_identical(tmp_path) -> None:
    pack = copy_pack(BASE_PACK, tmp_path / "pack")
    for path in sorted((pack / "data").glob("*.csv")):
        before = path.read_bytes()
        table = path.stem.split("_", 1)[1] if path.stem[0].isdigit() else path.stem
        edit_csv_rows(pack, table, lambda rows: rows)
        assert path.read_bytes() == before, path.name


def test_a_csv_edit_reaches_the_regenerated_seed(tmp_path) -> None:
    pack = copy_pack(BASE_PACK, tmp_path / "pack")

    def change(rows):
        for r in rows:
            if r["ma_vinv"] == "VINV-0015":
                r["trang_thai"] = "Nháp"
        return rows

    edit_csv_rows(pack, "vinv", change)
    regenerate_sql(pack)
    seed = (pack / "targets" / "postgres" / "seed_operational.sql").read_text(encoding="utf-8")
    assert "'Nháp'" in seed and csv_path(pack, "vinv").name == "13_vinv.csv"
