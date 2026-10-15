"""G1's replay script (experiments/p2_replay.py): its criteria, its read-only guards and its
exit statuses, without the run cache."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from specdx.infer import anchored
from specdx.infer.act import Rasch
from specdx.infer.anchored import AnchoredFit, StartFit, V3Params
from specdx.infer.cells import Cell, CellData, Reading
from specdx.infer.typing import CellDiagnosis
from specdx.types import Defect, Scenario

SCRIPT = Path(__file__).parents[1] / "experiments" / "p2_replay.py"


@pytest.fixture(scope="module")
def g1():
    spec = importlib.util.spec_from_file_location("p2_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replay(g1, theta, readings=(), defects=(), starts=None, unread=()):
    """Every model in theta has a reading at an unrelated cell unless named in unread."""
    cell = Cell(g1.R_SO_CELL, "value", "chờ duyệt", g1.R_SO, "so.trang_thai", "so.trang_thai",
                ("DE-SM-02:draft_quote",))
    other = [Reading("grn12|po:PO-0015:sl_dat", f"o-{m}", m, "f", "grn12", "15")
             for m in theta if m not in unread]
    data = CellData({cell.id: cell} if readings else {}, other + list(readings))
    fit = AnchoredFit({m: 0.0 for m in theta}, theta, {}, {}, {}, True, 1, V3Params(),
                      starts=starts or {})
    return g1.Replay("smoke", [], data, Rasch({}, {}, {}, {}), fit,
                     CellDiagnosis(list(defects)), {})


def test_g1_needs_no_collapse_and_r_so_where_it_was_read(g1) -> None:
    good = {"gpt-oss:20b": 0.97, "qwen3:30b-a3b": 0.91}
    read = [Reading(g1.R_SO_CELL, "k1", "gpt-oss:20b", "openai", "grn12", "draft")]
    found = [Defect("hidden", (g1.R_SO,), ("grn12",), ("k1",), 1.0, "x")]
    assert g1.r_so_status(_replay(g1, good, read, found)) == "reported"
    assert g1.verdict(_replay(g1, good, read, found))
    assert g1.r_so_status(_replay(g1, good, read)) == "missed"
    assert not g1.verdict(_replay(g1, good, read))
    assert g1.r_so_status(_replay(g1, good)) == "not observable"
    assert g1.verdict(_replay(g1, good))  # not observable does not fail (ii)
    collapsed = {"gpt-oss:20b": 0.97, "qwen3:30b-a3b": 0.08}
    assert not g1.theta_holds(_replay(g1, collapsed))[0]
    assert not g1.verdict(_replay(g1, collapsed, read, found))
    assert not g1.theta_holds(_replay(g1, {"gpt-oss:20b": 0.97}))[0]  # a missing model fails
    assert g1.THETA_FLOOR == 0.5 and g1.NO_COLLAPSE == ("gpt-oss:20b", "qwen3:30b-a3b")


def test_the_replay_never_refutes(g1) -> None:
    result = g1.NoRefutation().refute(frozenset({"c"}), Scenario("s", 15, 12, 0), [])
    assert not result.refuted and not result.attempted


def test_rows_matching_no_golden_row_are_counted_per_column(g1) -> None:
    base = SimpleNamespace(cells=SimpleNamespace(rows={
        "po": SimpleNamespace(golden={("po-0015",): "PO-0015"})}))
    cells = {c.id: c for c in [
        Cell("grn12|po:PO-0015:sl_dat", "value", "15", "x", "po.sl_dat", "po.sl_dat", ("s",)),
        Cell("grn12|po:po-0016:sl_dat", "value", None, None, "po.sl_dat", "po.sl_dat", ("s",)),
        Cell("happy|po:de-fn-04:sl_dat", "value", None, None, "po.sl_dat", "po.sl_dat", ("s",)),
        Cell("golden:po:PO-0015:sl_dat", "value", "{ordered}", "x", "po.sl_dat", "po.sl_dat",
             ("s",))]}
    assert g1.unmatched_rows(base, CellData(cells, [])) == {"po.sl_dat": 2}


def test_g1_refuses_provisional_parameters_and_a_broken_mapping(g1, monkeypatch,
                                                                 tmp_path) -> None:
    out = tmp_path / "g1.md"
    monkeypatch.setattr(anchored, "PARAMS_FROZEN", False)
    assert g1.main(["--out", str(out)]) == 2 and not out.exists()
    monkeypatch.setattr(anchored, "PARAMS_FROZEN", True)
    monkeypatch.setattr(g1, "load_cache", lambda path: {})
    assert g1.main(["--out", str(out)]) == 2 and not out.exists()


def test_g1_needs_a_reading_for_each_named_model(g1) -> None:
    good = {"gpt-oss:20b": 0.97, "qwen3:30b-a3b": 0.91}
    holds, why = g1.theta_holds(_replay(g1, good, unread=("qwen3:30b-a3b",)))
    assert not holds and "no readings for qwen3:30b-a3b" in why
    assert not g1.verdict(_replay(g1, good, unread=("qwen3:30b-a3b",)))
    assert g1.theta_holds(_replay(g1, good))[0]


def test_any_crash_exits_2_not_1(g1, monkeypatch, tmp_path) -> None:
    out = tmp_path / "g1.md"
    monkeypatch.setattr(anchored, "PARAMS_FROZEN", True)

    def boom(path):
        raise OSError("no cache")

    monkeypatch.setattr(g1, "load_cache", boom)
    assert g1.main(["--out", str(out)]) == 2 and not out.exists()


def test_out_under_the_run_cache_is_refused(g1, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(anchored, "PARAMS_FROZEN", True)
    monkeypatch.setattr(g1, "WORK_DIR", tmp_path / "work")
    called = []
    monkeypatch.setattr(g1, "load_cache", lambda path: called.append(path) or {})
    assert g1.main(["--out", str(tmp_path / "work" / "sub" / "g1.md")]) == 2
    assert not called and not (tmp_path / "work").exists()


def test_render_reports_each_start_and_marks_provisional_runs(g1, monkeypatch) -> None:
    good = {"gpt-oss:20b": 0.97, "qwen3:30b-a3b": 0.91}
    starts = {"t3": StartFit(-100.5, True, 12), "alpha=2.0": StartFit(-300.0, False, 500)}
    r = _replay(g1, good, starts=starts)
    monkeypatch.setattr(anchored, "PARAMS_FROZEN", True)
    text = g1.render([r, r], V3Params())
    assert text.splitlines()[2] == "# G1: PASS"
    assert "t3 -100.5 True 12, alpha=2.0 -300.0 False 500" in text
    monkeypatch.setattr(anchored, "PARAMS_FROZEN", False)
    assert g1.render([r, r], V3Params()).splitlines()[2] == "# G1: PROVISIONAL (not G1)"


def test_a_crash_after_the_fit_also_exits_2(g1, monkeypatch, tmp_path, capsys) -> None:
    out = tmp_path / "g1.md"
    monkeypatch.setattr(anchored, "PARAMS_FROZEN", True)
    monkeypatch.setattr(g1, "load_cache", lambda path: {})
    monkeypatch.setattr(g1, "model_digests", lambda cache: {})
    monkeypatch.setattr(g1, "open_base", lambda path: object())
    monkeypatch.setattr(g1, "smoke_runs", lambda *a: [])
    monkeypatch.setattr(g1, "p1_runs", lambda *a: [])
    monkeypatch.setattr(g1, "replay", lambda *a: g1.Replay(
        "x", [], CellData({}, []), Rasch({}, {}, {}, {}),
        AnchoredFit({}, {}, {}, {}, {}, True, 1, V3Params()), CellDiagnosis([]), {}))

    def boom(replays, params):
        raise KeyError("cell")

    monkeypatch.setattr(g1, "render", boom)
    assert g1.main(["--out", str(out)]) == 2 and not out.exists()
    assert "error: KeyError" in capsys.readouterr().err
