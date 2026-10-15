"""Calibration from base runs (spec v3 §8.1), on the real base pack and without the run
cache: synthetic behaviours stand in for cached runs."""

import json
import math

import pytest
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.db import golden_tables
from specdx.infer.cells import CellData
from specdx.orchestrate.work_orders import load_work_orders
from specdx.sim.calibrate import (
    Calibration,
    CalibrationError,
    _matching_concentration,
    _top_error,
    base_runs,
    calibrate,
    column_of,
    open_base,
)
from specdx.sim.study import load_config
from specdx.types import Behaviour, InterpreterId, State

SCREENED = {"gpt-oss:20b": ("openai", 0.9135802469135802),
            "qwen3:30b-a3b": ("qwen", 0.8024691358024691),
            "qwen2.5:14b": ("qwen", 0.5802469135802469),
            "mistral-small3.2:24b": ("mistral", 0.345679012345679)}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return open_base(tmp_path_factory.mktemp("pdf"))


def test_the_configuration_uses_the_planned_budget_and_orders_models_by_accuracy(
    tmp_path,
) -> None:
    c = Calibration(population={"q": ("a", "b"), "o": ("c",), "m": ("d",)},
                    act={"a": 0.8, "b": 0.6, "c": 0.9, "d": 0.4},
                    accuracy={"a": 0.9, "b": 0.95, "c": 0.97, "d": 0.8}, dispersion=1.0,
                    top_error=0.8, error_conc=0.2, reads=6.0, pooled_share=0.2,
                    anchored_share=0.3, constraint_share=0.1,
                    values=(0.5, 0.3, 0.1, 0.05, 0.05, 0.0), columns=100, items=9, runs=105)
    config = c.config()
    assert config.scenarios == 200 // (4 * 9) == 5
    assert config.alphas == pytest.approx(
        tuple(math.log(p / (1 - p)) for p in (0.97, 0.95, 0.9, 0.8)))
    assert config.act == (0.9, 0.6, 0.8, 0.4) and config.defect_rate == 0.15
    path = tmp_path / "calibration.json"
    path.write_text(c.to_json(), encoding="utf-8")
    assert load_config(path) == config
    assert json.loads(path.read_text())["models"] == ["c", "b", "a", "d"]


def test_the_error_concentration_is_the_nearest_match_of_the_top_error_share() -> None:
    assert _matching_concentration(0.95) == 0.1
    assert _matching_concentration(0.4) >= 1.0
    # A concentrated error (a high top share) means a small concentration: the match falls
    # as the share rises (checked over 0.30–1.00 in steps of 0.01 at the plan's pre-check).
    assert (_matching_concentration(0.95) <= _matching_concentration(0.7)
            <= _matching_concentration(0.4))


def test_a_statistic_without_data_raises_instead_of_falling_back() -> None:
    with pytest.raises(CalibrationError, match="error concentration"):
        _top_error(CellData({}, []))
    assert column_of("grn12|po:PO-0015:sl_dat") == "po:PO-0015:sl_dat"
    assert column_of("golden:po:PO-0015:sl_dat") == "golden:po:PO-0015:sl_dat"


def test_calibration_reads_base_runs_through_the_cell_lens(base) -> None:
    tables = golden_tables(CanonPack(BASE_PACK))
    steps = {s.id: "done" for s in load_work_orders()}
    def state(m: str, v: int) -> State:
        copy = {t: [dict(r) for r in rows] for t, rows in tables.items()}
        if m == "mistral-small3.2:24b" and v < 2:  # two different errors at one cell
            copy["po"][0]["trang_thai"] = f"lỗi {v}"
        return State(copy, {"fn_reconcile:PO-0015": "Cần kiểm"}, steps)

    runs = [Behaviour(f"{m}|v{v}|s{s}", InterpreterId(m, f, v, s), base.scenarios["happy"],
                      state(m, v) if s == 1 else state(m, 2))
            for m, (f, _) in SCREENED.items() for v in (0, 1, 2) for s in (1, 2, 3)]
    c = calibrate(base, runs)
    assert c.population == {"openai": ("gpt-oss:20b",),
                            "qwen": ("qwen3:30b-a3b", "qwen2.5:14b"),
                            "mistral": ("mistral-small3.2:24b",)}
    assert c.act == {m: a for m, (_, a) in sorted(SCREENED.items())}
    assert all(a > 0.9 for a in c.accuracy.values()) and c.runs == 36
    assert c.dispersion == 0.25 and c.items == 9 and c.columns > 0
    assert sum(c.values) == pytest.approx(1.0) and c.top_error == 0.5
    # one happy scenario: 9 readings per model of a per-scenario cell, and the simulated
    # specification's readings per cell reported beside them (ruling F12)
    assert c.reads == 9.0 and c.simulated_reads > 0


def test_the_smoke_runs_must_all_map(base) -> None:
    with pytest.raises(CalibrationError, match="60 runs"):
        base_runs(base, {})
