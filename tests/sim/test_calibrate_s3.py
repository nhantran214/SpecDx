"""The S3 calibrator and the seeded domain split (P4a design §6.1)."""

import math

import pytest

from specdx.sim.calibrate import CalibrationError
from specdx.sim.calibrate_s3 import SPLIT_FILE, SPLIT_SEED, calibrate_s3, load_split, split_domains
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
    tasks = [S3Task("library", i, f"goal{i}", True, "P") for i in range(5)]
    other = [S3Task("bank", i, f"goal{i}", True, "P") for i in range(5)]
    runs = [_run("library", t.index, m, ok=(t.index < n))
            for t in tasks for m, n in (("a", 5), ("b", 5), ("c", 5), ("mid", 4), ("weak", 3))]
    runs += [_run("library", 0, "weak", True, err=True)]  # a tool-error run
    runs += [_run("bank", t.index, "a", False) for t in other]  # not a dev domain
    cds = [cells_for(tasks, [r for r in runs if r.domain == "library"]),
           cells_for(other, [r for r in runs if r.domain == "bank"])]
    config = calibrate_s3(cds, runs, dev=("library",))
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
    spec = generate(config)
    assert spec.cells and spec.readings


def test_calibrate_s3_has_no_fallback_without_development_data() -> None:
    tasks = [S3Task("bank", 0, "g", True, "P")]
    runs = [_run("bank", 0, "m", True)]
    with pytest.raises(CalibrationError):
        calibrate_s3([cells_for(tasks, runs)], runs, dev=("library",))
