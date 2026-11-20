"""The simulation generator (spec v3 §8.1)."""

import math
from collections import Counter
from dataclasses import replace

import numpy as np

from specdx.sim.baselines import by_cell
from specdx.sim.generator import SimConfig, generate

BIG = SimConfig(alphas=(2.0, 0.0), act=(1.0, 1.0), columns=200, defect_rate=0.3, seed=1)


def test_generation_is_deterministic_in_its_seed() -> None:
    assert generate(SimConfig(seed=4)).readings == generate(SimConfig(seed=4)).readings
    assert generate(SimConfig(seed=4)).readings != generate(SimConfig(seed=5)).readings


def test_each_behaviour_reads_a_cell_at_most_once_and_pooled_cells_span_every_scenario() -> None:
    spec = generate(SimConfig(act=(1.0, 1.0), seed=2))
    per = Counter((r.cell, r.run_key) for r in spec.readings)
    assert max(per.values()) == 1
    reads = Counter((r.cell, r.model) for r in spec.readings)
    for cell in spec.cells:
        expected = 9 * (4 if cell.endswith("|pooled") else 1)
        assert all(reads[(cell, m)] == expected for m in spec.models)


def test_act_rates_thin_the_readings() -> None:
    spec = generate(SimConfig(act=(1.0, 0.3), columns=100, seed=3))
    count = Counter(r.model for r in spec.readings)
    assert 0.2 < count["m1"] / count["m0"] < 0.4


def test_planted_truths_follow_the_definitions() -> None:
    spec = generate(BIG)
    for cid, t in spec.truth.items():
        cell = spec.cells[cid]
        assert t.hidden == (cell.anchor is not None and t.admitted != (cell.anchor,))
        assert t.ambiguity == (len(t.admitted) == 2)
        assert t.weak == (cell.anchor is None and len(t.admitted) == 1)
    forms = Counter(t.form for cid, t in spec.truth.items() if spec.cells[cid].anchor)
    assert forms["mis"] and forms["amb"] and forms["empty"] and forms["none"]
    columns = {}
    for cid, t in spec.truth.items():
        columns.setdefault(spec.cells[cid].group, set()).add(t)
    assert all(len(ts) == 1 for ts in columns.values())  # a form belongs to a column


def test_readings_follow_the_accuracy_model_at_non_defective_cells() -> None:
    spec = generate(BIG)
    for m, a in spec.alphas.items():
        hits, expected = [], []
        for cid, t in spec.truth.items():
            if t.form == "none" and spec.cells[cid].anchor is not None:
                mine = [r.value for r in spec.readings if r.cell == cid and r.model == m]
                hits += [v == spec.cells[cid].anchor for v in mine]
                expected += [1 / (1 + math.exp(-(a - spec.betas[cid])))] * len(mine)
        assert abs(np.mean(hits) - np.mean(expected)) < 0.03, m


def test_a_behaviour_is_clean_when_every_anchored_cell_of_its_item_reads_its_anchor() -> None:
    spec = generate(SimConfig(seed=6))
    by_item: dict[tuple[str, str], list] = {}
    for r in spec.readings:
        by_item.setdefault((r.run_key, spec.cells[r.cell].steps[0]), []).append(r)
    for rs in by_item.values():
        clean = all(r.value == spec.cells[r.cell].anchor for r in rs
                    if spec.cells[r.cell].anchor is not None)
        assert all(r.clean == clean for r in rs)


def test_the_a3_arm_gives_each_model_its_own_errors() -> None:
    base = generate(SimConfig(alphas=(0.0, 0.0), columns=60, seed=8))
    a3 = generate(SimConfig(alphas=(0.0, 0.0), columns=60, seed=8, arm="a3"))

    def disagreement(spec) -> float:
        out = []
        for cid, t in spec.truth.items():
            if t.form == "none" and spec.cells[cid].kind == "value":
                tops = []
                for m in spec.models:
                    errs = Counter(r.value for r in spec.readings
                                   if r.cell == cid and r.model == m and r.value not in t.admitted)
                    tops.append(errs.most_common(1)[0][0] if errs else None)
                out.append(tops[0] != tops[1])
        return float(np.mean(out))

    assert disagreement(a3) > disagreement(base)


def test_the_a5_arm_lets_the_strongest_model_guess_where_t_is_silent() -> None:
    config = SimConfig(alphas=(3.0, -1.0), act=(1.0, 1.0), columns=200, defect_rate=0.5,
                       forms=(0.0, 0.0, 1.0), seed=9)

    def top_share(spec, model: str) -> float:
        shares = []
        for cid, t in spec.truth.items():
            if t.form == "empty":
                mine = Counter(r.value for r in spec.readings if r.cell == cid and r.model == model)
                shares.append(mine.most_common(1)[0][1] / sum(mine.values()))
        return float(np.mean(shares))

    base, a5 = generate(config), generate(SimConfig(**{**config.__dict__, "arm": "a5"}))
    assert top_share(a5, "m0") > top_share(base, "m0") + 0.2
    assert abs(top_share(a5, "m1") - top_share(base, "m1")) < 0.1


def test_difficulty_is_one_per_column_unless_the_within_group_arm_spreads_it() -> None:
    def spread(beta_within: float) -> float:
        spec = generate(SimConfig(columns=60, beta_sd=1.0, beta_within=beta_within, seed=12))
        by_column: dict[str, list[float]] = {}
        for cid, beta in spec.betas.items():
            by_column.setdefault(spec.cells[cid].group, []).append(beta)
        return float(np.mean([np.std(b) for b in by_column.values() if len(b) > 1]))

    assert spread(0.0) == 0.0
    assert 0.3 < spread(0.5) < 0.7  # the within-group arm of spec v3.1 §8.1


def test_shared_arm_plants_hard_cells_with_one_attractor() -> None:
    config = SimConfig(
        alphas=(3.0, 2.0), act=(1.0, 1.0), columns=30, pooled_share=0.0, constraint_share=0.0,
        anchored_share=1.0, defect_rate=0.0, beta_sd=0.0, arm="shared", shared_share=1.0,
        scenarios=2, variants=4, seeds=6, seed=7)
    spec = generate(config)
    assert all(t.shared for t in spec.truth.values())
    assert all(abs(b - 2.0) < 1e-12 for b in spec.betas.values())
    shares = []
    for cid, rs in by_cell(spec).items():
        errs = Counter(r.value for r in rs if r.value != spec.cells[cid].anchor)
        if sum(errs.values()) >= 8:
            shares.append(max(errs.values()) / sum(errs.values()))
    assert shares
    assert min(shares) > 0.4  # λ ≥ 0.5 pulls the errors to one value, plus noise
    base = generate(replace(config, arm="base"))
    assert not any(t.shared for t in base.truth.values())
    none = generate(replace(config, shared_share=0.0))
    assert not any(t.shared for t in none.truth.values())


def test_homogeneous_arm_equalises_alpha() -> None:
    spec = generate(SimConfig(alphas=(3.0, -1.0), arm="homogeneous", seed=1))
    assert spec.alphas == {"m0": 1.0, "m1": 1.0}


def test_alpha_spread_scales_about_the_mean() -> None:
    spec = generate(SimConfig(alphas=(3.0, -1.0), alpha_spread=0.5, seed=1))
    assert spec.alphas == {"m0": 2.0, "m1": 0.0}


def test_saturated_arm_makes_planted_cells_unanimous() -> None:
    spec = generate(SimConfig(alphas=(2.0, -1.0, 0.0), act=(1.0, 1.0, 1.0), columns=80,
                              defect_rate=0.5, arm="saturated", constraint_share=0.0,
                              anchored_share=1.0, seed=3))
    planted = [c for c, t in spec.truth.items() if t.hidden]
    assert planted
    values: dict[str, set[str]] = {c: set() for c in planted}
    for r in spec.readings:
        if r.cell in values:
            values[r.cell].add(r.value)
    assert all(len(v) == 1 for v in values.values())
    assert all(v != {spec.cells[c].anchor} for c, v in values.items())


def test_act_group_shares_one_draw() -> None:
    def faithful_counts(group: int) -> set[int]:
        spec = generate(SimConfig(
            alphas=(0.0,), act=(1.0,), columns=4, items=1, pooled_share=0.0, constraint_share=0.0,
            anchored_share=1.0, defect_rate=0.0, beta_sd=0.0, scenarios=2, variants=10, seeds=10,
            act_group=group, seed=2))
        faithful: dict[str, int] = {}
        for r in spec.readings:
            ok = r.value == spec.cells[r.cell].anchor
            faithful[r.run_key] = faithful.get(r.run_key, 0) + ok
        return set(faithful.values())

    assert faithful_counts(4) <= {0, 4}
    assert len(faithful_counts(1)) > 2
