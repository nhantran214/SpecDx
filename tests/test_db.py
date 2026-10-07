import shutil
import time

import psycopg
import pytest

from specdx import db as db_module
from specdx.config import BASE_PACK
from specdx.db import PackDatabase, pack_digest, split_background
from specdx.observe.state import reconcile_value
from specdx.types import State

pytestmark = pytest.mark.db


def test_a_run_starts_from_master_data_and_background_rows(db) -> None:
    with db.isolated() as conn:
        assert conn.execute("SELECT count(*) AS n FROM so").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM cust").fetchone()["n"] > 0
        assert conn.execute("SELECT count(*) AS n FROM cand").fetchone()["n"] == 3
        aprs = conn.execute("SELECT ma_apr FROM apr").fetchall()
        assert [r["ma_apr"] for r in aprs] == ["APR-TH-01"]


def test_whatever_a_run_writes_is_rolled_back(db) -> None:
    with db.isolated() as conn:
        conn.execute("INSERT INTO cases (case_id, tieu_de, trang_thai) VALUES ('C-X', 't', 's')")
    with psycopg.connect(db.url) as other:
        assert other.execute("SELECT count(*) FROM cases").fetchone()[0] == 0


def test_the_golden_state_is_the_canon_end_state(golden) -> None:
    po = golden.tables["po"][0]
    assert (po["ma_po"], po["sl_dat"], po["trang_thai"]) == ("PO-0015", "15", "Đã duyệt")
    assert golden.tables["grn"][0]["sl_nhan"] == "12"
    assert golden.derived["fn_reconcile:PO-0015"] == "Cần kiểm"
    assert {r["ma_apr"] for r in golden.tables["apr"]} == {
        "APR-0102", "APR-0155", "APR-0210", "APR-TH-01"
    }


def test_two_receipts_for_one_order_make_fn_reconcile_error(db, golden) -> None:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["grn"].append({"ma_grn": "GRN-0016", "po": "PO-0015", "sl_nhan": "3"})
    real = db.materialize(State(tables))
    assert real is not None
    assert real.derived["fn_reconcile:PO-0015"] == "ERROR"


def test_a_state_the_database_refuses_does_not_materialise(db, golden) -> None:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["evt"].append({"case_id": "CASE-0031", "kind": "BOGUS", "ref": "SO-0031",
                          "actor": "AI", "basis": "x", "at": "2026-09-01 02:00:00+00"})
    assert db.materialize(State(tables)) is None


def test_background_rows_are_the_th0_branch(golden) -> None:
    background, case = split_background(golden.tables)
    assert {r["ma_apr"] for r in background["apr"]} == {"APR-TH-01"}
    assert "cand" not in case and len(background["cand"]) == 3


def test_the_digest_changes_with_any_file(tmp_path) -> None:
    copy = tmp_path / "pack"
    shutil.copytree(BASE_PACK, copy)
    before = pack_digest(copy)
    (copy / "kb" / "KB-0001.md").write_text("changed", encoding="utf-8")
    assert pack_digest(copy) != before


def test_use_recovers_from_failed_reload(db, tmp_path) -> None:
    """After failed reload, use(BASE_PACK) reloads, not skipping due to cached state."""
    broken = tmp_path / "pack"
    shutil.copytree(BASE_PACK, broken)
    (broken / "targets" / "postgres" / "guardrails.sql").write_text(
        "THIS IS NOT SQL;", encoding="utf-8"
    )
    with pytest.raises(psycopg.Error):
        db.use(broken)
    db.use(BASE_PACK)
    with db.isolated() as conn:
        assert conn.execute("SELECT count(*) AS n FROM cust").fetchone()["n"] > 0


def test_two_pack_databases_in_one_process_share_the_lock_and_stay_honest(db, tmp_path) -> None:
    other = tmp_path / "pack"
    shutil.copytree(BASE_PACK, other)
    (other / "kb" / "KB-0001.md").write_text("changed", encoding="utf-8")
    first, second = PackDatabase(db.url), PackDatabase(db.url)
    first.use(BASE_PACK)
    second.use(other)  # the database now holds the other specification
    first.use(BASE_PACK)  # so this must reload, though first last loaded the base
    with psycopg.connect(db.url) as conn:
        marker = conn.execute("SELECT digest FROM specdx_marker").fetchone()[0]
    assert marker == pack_digest(BASE_PACK)
    with first.isolated() as conn:
        assert conn.execute("SELECT count(*) AS n FROM cust").fetchone()["n"] > 0


def test_a_second_process_holding_the_lock_is_refused_fast(db) -> None:
    held = db_module._LOCKS.pop(db.url)  # act as if this process had never taken the lock
    held.close()
    with psycopg.connect(db.url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (db_module.LOCK_KEY,))
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="another process"):
            PackDatabase(db.url).use(BASE_PACK)
        assert time.monotonic() - started < 5
    db.use(BASE_PACK)  # the other session has ended: the lock is free again
    assert db.url in db_module._LOCKS


@pytest.mark.parametrize("port", [55432, 55433])
def test_the_ports_of_other_projects_are_refused(port) -> None:
    with pytest.raises(ValueError, match=str(port)):
        PackDatabase(f"postgresql://tonomyx:tonomyx@localhost:{port}/tonomyx")


def test_reconcile_value_raises_on_closed_connection(db) -> None:
    """reconcile_value raises on connection errors, not returning ERROR."""
    conn = psycopg.connect(db.url)
    conn.close()
    with pytest.raises(psycopg.Error):
        reconcile_value(conn, "PO-0015")
