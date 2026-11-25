"""The S3 calibrator and the seeded domain split (P4a design §6.1)."""

import math

import numpy as np
import pytest

from specdx.sim.calibrate import CalibrationError
from specdx.sim.calibrate_s3 import (
    SPLIT_FILE,
    SPLIT_SEED,
    SPREAD_FLOOR,
    calibrate_s3,
    load_split,
    split_domains,
)
from specdx.sim.generator import generate
from specdx.subjects.sopbench import DOMAINS
from specdx.subjects.sopbench.cells import cells_for
from specdx.subjects.sopbench.load import S3Run, S3Task


def test_split_is_seeded_and_disjoint() -> None:
    dev, held = split_domains()
    assert (len(dev), len(held)) == (3, 4)
    assert not set(dev) & set(held)
    assert set(dev) | set(held) == set(DOMAINS)
    assert split_domains(SPLIT_SEED) == (dev, held)
    assert split_domains(1) != (dev, held) or split_domains(2) != (dev, held)


def test_committed_split_is_the_seeded_draw() -> None:
    assert load_split(SPLIT_FILE) == split_domains()


def _run(domain: str, task: int, model: str, ok: bool, err: bool = False) -> S3Run:
    return S3Run(domain, task, model, "fam", "fc", ok, ok, ok, ok, err, "P")


def test_calibrate_s3_on_a_tiny_fixture() -> None:
    tasks = [S3Task("library", i, f"goal{i % 2}", True, "P") for i in range(5)]
    other = [S3Task("bank", i, f"goal{i}", True, "P") for i in range(5)]
    runs = [_run("library", t.index, m, ok=(t.index < n))
            for t in tasks for m, n in (("a", 5), ("b", 5), ("c", 5), ("mid", 4), ("weak", 3))]
    runs += [_run("library", 0, "weak", True, err=True)]  # a tool-error run
    runs += [_run("bank", t.index, "a", False) for t in other]  # not a dev domain
    cds = [cells_for(tasks, [r for r in runs if r.domain == "library"]),
           cells_for(other, [r for r in runs if r.domain == "bank"])]
    [config] = calibrate_s3(cds, runs, dev=("library",))
    # The raters, alphabetical: a, b, c, mid, weak. Accuracy against the anchors is
    # (hits + 0.5) / (n + 1) over the cells whose plurality is the anchor.
    assert len(config.alphas) == len(config.act) == 5
    a, b, c, mid, weak = config.alphas
    assert a == b == c > mid > weak
    assert math.isclose(a, math.log((20.5 / 21) / (0.5 / 21)), rel_tol=1e-9)
    assert math.isclose(mid, math.log(16.5 / 4.5), rel_tol=1e-9)
    assert config.act[4] < 1.0 and set(config.act[:4]) == {1.0}
    assert config.values == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert config.defect_rate == 0.15
    assert config.act_group == 4
    assert 0.25 <= config.beta_sd <= 3.0
    assert config.alpha_spread == 1.0  # the spread of the logits is above the floor
    assert (config.columns, config.items, config.pooled_share) == (20, 5, 0.0)
    assert config.goals == (3, 2)  # goal0: tasks 0, 2, 4; goal1: tasks 1, 3
    spec = generate(config)
    assert spec.cells and spec.readings


def _domain(domain: str, goals: dict[str, int], raters: dict[str, float],
            seed: int) -> tuple[list[S3Task], list[S3Run]]:
    """Tasks by goal; rater m reads each part correctly when a seeded draw is below its
    accuracy."""
    rng = np.random.default_rng(seed)
    tasks = [S3Task(domain, i, g, True, "P")
             for i, g in enumerate(g for g, n in goals.items() for _ in range(n))]
    runs = []
    for t in tasks:
        for m, acc in raters.items():
            ok = [bool(rng.random() < acc) for _ in range(4)]
            runs.append(S3Run(domain, t.index, m, "fam", "fc", ok[0], ok[1], ok[2], ok[3],
                              False, "P"))
    return tasks, runs


def test_calibrate_s3_gives_one_layout_per_development_domain() -> None:
    # Ruling F-C1: each development domain is its own layout, with its goals, its raters and
    # its own difficulty spreads.
    lib = _domain("library", {"g_a": 6, "g_b": 4}, {"x": 0.9, "y": 0.8, "z": 0.7}, 1)
    uni = _domain("university", {"g_c": 5}, {"x": 0.95, "w": 0.6}, 2)
    cds = [cells_for(*lib), cells_for(*uni)]
    configs = calibrate_s3(cds, lib[1] + uni[1], dev=("library", "university"))
    assert [c.goals for c in configs] == [(6, 4), (5,)]
    assert [len(c.alphas) for c in configs] == [3, 2]
    assert [c.items for c in configs] == [10, 5] and [c.columns for c in configs] == [40, 20]
    assert all(0.0 <= c.beta_within <= 3.0 and 0.25 <= c.beta_sd <= 3.0 for c in configs)
    assert all(c.seed == 0 for c in configs)


def test_calibrate_s3_splits_difficulty_between_and_within_groups() -> None:
    # Every outcome, constraints and process cell reads alike (75 of 100); within db_match the
    # tasks differ (90 or 60 of 100), far beyond binomial noise, so beta_within clears the
    # floor once the sampling variance is subtracted (ruling R-5).
    tasks = [S3Task("library", i, "g", True, "P") for i in range(40)]
    raters = [f"m{k:03d}" for k in range(100)]
    runs = []
    for t in tasks:
        for k, m in enumerate(raters):
            db = k < 90 if t.index % 2 else k < 60
            runs.append(S3Run("library", t.index, m, "fam", "fc", k < 75, db, k < 75, k < 75,
                              False, "P"))
    [config] = calibrate_s3([cells_for(tasks, runs)], runs, dev=("library",))
    assert config.beta_within > SPREAD_FLOOR
    assert config.goals == (40,)


def test_calibrate_s3_has_no_fallback_without_development_data() -> None:
    tasks = [S3Task("bank", 0, "g", True, "P")]
    runs = [_run("bank", 0, "m", True)]
    with pytest.raises(CalibrationError):
        calibrate_s3([cells_for(tasks, runs)], runs, dev=("library",))


def test_a_near_homogeneous_panel_is_widened_to_the_spread_floor() -> None:
    models = ("a", "b", "c", "d", "e")
    tasks = [S3Task("library", i, f"goal{i}", True, "P") for i in range(101)]
    # Rater k fails its own block of 20 tasks (21 for the last): about 0.8 accuracy each.
    runs = [_run("library", t.index, m, ok=not (20 * k <= t.index < 20 * k + 20 + (k == 4)))
            for t in tasks for k, m in enumerate(models)]
    [config] = calibrate_s3([cells_for(tasks, runs)], runs, dev=("library",))
    spread = float(np.std(config.alphas))
    assert 0.0 < spread < SPREAD_FLOOR
    assert math.isclose(config.alpha_spread * spread, SPREAD_FLOOR, rel_tol=1e-9)


def test_a_task_whose_runs_are_all_tool_errors_adds_no_goal() -> None:
    # Ruling R-1: such a task has no reading, so the real fit never sees its cells; its goal
    # is not counted, and a goal left with no task is dropped.
    tasks = [S3Task("library", i, "g_a" if i < 4 else ("g_b" if i < 6 else "g_c"), True, "P")
             for i in range(7)]
    runs = [_run("library", t.index, m, ok=(k > 0 or t.index % 2 == 0), err=(t.index >= 5))
            for t in tasks for k, m in enumerate(("x", "y", "z"))]
    [config] = calibrate_s3([cells_for(tasks, runs)], runs, dev=("library",))
    assert config.goals == (4, 1)  # g_a: 4 tasks; g_b: task 4 only; g_c: none read
    assert (config.items, config.columns) == (5, 20)


def test_pure_binomial_noise_calibrates_to_the_floor() -> None:
    # Ruling R-5: every cell has the same true accuracy, so the cell logits spread only by
    # binomial sampling; once that variance is subtracted, both spreads sit at the floor.
    rng = np.random.default_rng(7)
    goals = [f"g{k}" for k in range(8)]
    tasks = [S3Task("library", i, goals[i % 8], True, "P") for i in range(400)]
    raters = [f"m{k:02d}" for k in range(25)]
    runs = []
    for t in tasks:
        for m in raters:
            ok = [bool(rng.random() < 0.8) for _ in range(4)]
            runs.append(S3Run("library", t.index, m, "fam", "fc", ok[0], ok[1], ok[2], ok[3],
                              False, "P"))
    [config] = calibrate_s3([cells_for(tasks, runs)], runs, dev=("library",))
    assert config.beta_within == SPREAD_FLOOR
    assert config.beta_sd == SPREAD_FLOOR
