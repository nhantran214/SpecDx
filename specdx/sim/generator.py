"""Synthetic specifications for the simulation study (spec v3.1 §8.1).

A synthetic specification is a set of columns of cells, value and constraint, anchored and
unanchored. A column is read either per scenario (one cell per scenario, as most golden
values are) or as one cell pooled over every scenario (as lifted and cross-cutting clauses
are), and all its cells share one planted form, because a defect is a property of a rule,
not of a scenario, and one difficulty β_g, as difficulty is grouped by column (v3.1 §3.2).
Columns are spread over items; a behaviour acts at an item with its model's act rate and
then reads every cell of the item in its scenario. Readings follow the §3.2 model with
known parameters, except in the two misspecification arms and the within-group arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from specdx.infer.cells import SATISFIED, VIOLATED, Cell, Reading

ARMS: tuple[str, ...] = ("base", "a3", "a5", "shared", "homogeneous", "saturated")


@dataclass(frozen=True)
class SimConfig:
    """alphas, act: per model, the true ability α_m and the act rate. values: P(|V| = k)
    for k = 2, 3, … on a value column (the truth and its error values). forms: the shares
    of mis, amb and empty among anchored defects. unanchored: the shares of a determined
    value (a weak-check truth), a pair and silence on unanchored columns. error_conc: the
    Dirichlet concentration of a column's error tendency (small: concentrated errors);
    kappa_true: a cell's ε_c ~ Dirichlet(kappa_true · tendency). arm: base, a3 (each model
    errs by its own distribution) or a5 (at silent cells each model guesses one value with
    its accuracy). beta_sd: τ_true, the spread of the group difficulties β_g; beta_within:
    each cell's difficulty spreads around its group's by this (0: difficulty truly per
    group; 0.5: the within-group arm of §8.1)."""

    alphas: tuple[float, ...] = (2.0, 1.0)
    act: tuple[float, ...] = (0.9, 0.9)
    columns: int = 40
    pooled_share: float = 0.3
    scenarios: int = 4
    variants: int = 3
    seeds: int = 3
    items: int = 10
    beta_sd: float = 0.5
    beta_within: float = 0.0
    anchored_share: float = 0.7
    constraint_share: float = 0.3
    values: tuple[float, ...] = (0.0, 0.3, 0.3, 0.2, 0.1, 0.1)
    defect_rate: float = 0.15
    forms: tuple[float, float, float] = (0.4, 0.3, 0.3)
    unanchored: tuple[float, float, float] = (0.5, 0.2, 0.3)
    error_conc: float = 1.0
    kappa_true: float = 30.0
    arm: str = "base"
    seed: int = 0
    # v4 arms. shared_share (arm "shared" only): the share of non-defect anchored columns that
    # are hard, with one attractor. act_group: cells read by one run in groups of this size
    # share one faithful-or-error draw. alpha_spread: the abilities spread about their mean
    # by this factor (1.0: as given).
    shared_share: float = 0.3
    act_group: int = 1
    alpha_spread: float = 1.0


@dataclass(frozen=True)
class Truth:
    """A cell's planted truth: its form (none, mis, amb, empty) and admitted set, and the
    defect types it is a truth of (spec v1 §2.4 as §8.1 scores them)."""

    form: str
    admitted: tuple[str, ...]
    hidden: bool
    ambiguity: bool
    weak: bool
    shared: bool = False


@dataclass
class SimSpec:
    config: SimConfig
    cells: dict[str, Cell]
    readings: list[Reading]
    truth: dict[str, Truth]
    models: dict[str, str]
    alphas: dict[str, float]
    betas: dict[str, float] = field(default_factory=dict)

    @property
    def theta(self) -> dict[str, float]:
        return {m: 1.0 / (1.0 + math.exp(-a)) for m, a in self.alphas.items()}


def _vocabulary(rng: np.random.Generator, config: SimConfig, constraint: bool,
                anchored: bool) -> tuple[str, ...]:
    if constraint:
        return (SATISFIED, VIOLATED)
    sizes = np.array(config.values, dtype=float)
    k = int(rng.choice(len(sizes), p=sizes / sizes.sum())) + 2
    first = ["d"] if anchored else []
    return tuple(first + [f"v{i}" for i in range(k - len(first))])


def _plant(rng: np.random.Generator, config: SimConfig, vocab: tuple[str, ...],
           anchored: bool) -> tuple[str, tuple[str, ...]]:
    """A column's form and admitted set. Anchored: {d} unless a defect is planted; mis is
    {a}, amb is {d, a} or {a, a'} (ambiguous-pick, both kinds), empty is ∅. Unanchored: one
    value, a pair, or ∅."""
    d, a, a2 = vocab[0], vocab[1], vocab[2] if len(vocab) > 2 else vocab[1]
    if anchored:
        if rng.random() >= config.defect_rate:
            return "none", (d,)
        form = ("mis", "amb", "empty")[int(rng.choice(3, p=np.array(config.forms)))]
        if form == "mis":
            return form, (a,)
        if form == "amb":
            return form, (d, a) if rng.random() < 0.5 or a2 == a else (a, a2)
        return form, ()
    form = ("none", "amb", "empty")[int(rng.choice(3, p=np.array(config.unanchored)))]
    if form == "none":
        return form, (d,)
    return (form, (d, a)) if form == "amb" else (form, ())


def generate(config: SimConfig) -> SimSpec:
    """One synthetic specification, deterministic in config (seed included)."""
    rng = np.random.default_rng(config.seed)
    models = {f"m{i}": f"f{i}" for i in range(len(config.alphas))}
    alphas = {f"m{i}": float(a) for i, a in enumerate(config.alphas)}
    if config.arm == "homogeneous":
        mean = sum(alphas.values()) / len(alphas)
        alphas = dict.fromkeys(alphas, mean)
    elif config.alpha_spread != 1.0:
        mean = sum(alphas.values()) / len(alphas)
        alphas = {m: mean + config.alpha_spread * (a - mean) for m, a in alphas.items()}
    strongest = max(alphas, key=lambda m: alphas[m])
    cells: dict[str, Cell] = {}
    truth: dict[str, Truth] = {}
    betas: dict[str, float] = {}
    errors: dict[str, dict[str, np.ndarray]] = {}
    guess: dict[str, str] = {}
    vocabs: dict[str, tuple[str, ...]] = {}
    fixed: dict[str, str] = {}  # saturated arm: the one value every model reads
    layout: dict[tuple[int, int], list[str]] = {}  # (item, scenario) → cells read there
    for col in range(config.columns):
        constraint = rng.random() < config.constraint_share
        anchored = constraint or rng.random() < config.anchored_share
        pooled = rng.random() < config.pooled_share
        vocab = _vocabulary(rng, config, constraint, anchored)
        form, admitted = _plant(rng, config, vocab, anchored)
        tendency = rng.dirichlet(np.full(len(vocab), config.error_conc))
        group_beta = float(rng.normal(0.0, config.beta_sd))
        hard = None
        if (config.arm in ("shared", "saturated") and anchored and form == "none"
                and rng.random() < config.shared_share):
            attractor = vocab[1 + int(rng.integers(len(vocab) - 1))]
            # Saturated: λ = 1, and every model reads the attractor.
            hard = (attractor, 1.0 if config.arm == "saturated" else float(rng.uniform(0.5, 1.0)))
        item = col % config.items
        group = f"col{col:03d}"
        for s in ([-1] if pooled else range(config.scenarios)):
            cid = f"{group}|{'pooled' if pooled else f's{s}'}"
            cells[cid] = Cell(cid, "constraint" if constraint else "value",
                              vocab[0] if anchored else None,
                              f"clause:{cid}" if anchored else None, group, group, (f"i{item}",))
            truth[cid] = Truth(form, admitted, anchored and admitted != (vocab[0],),
                               len(admitted) == 2, not anchored and len(admitted) == 1,
                               hard is not None)
            betas[cid] = group_beta + config.beta_within * float(rng.normal())
            if hard is not None:
                betas[cid] += 2.0
            # Floored: numpy's Dirichlet returns NaN for vanishing parameters.
            base = rng.dirichlet(np.maximum(config.kappa_true * tendency, 0.01))
            errors[cid] = {m: (rng.dirichlet(np.full(len(vocab), 0.3)) if config.arm == "a3"
                               else base) for m in models}
            if hard is not None:
                attractor, lam = hard
                pull = np.array([1.0 if x == attractor else 0.0 for x in vocab])
                errors[cid] = dict.fromkeys(
                    models, (1.0 - lam) * np.full(len(vocab), 1.0 / len(vocab)) + lam * pull)
            if config.arm == "saturated" and hard is not None:
                fixed[cid] = hard[0]
            elif config.arm == "saturated" and truth[cid].hidden:
                fixed[cid] = admitted[-1] if admitted else vocab[1]
            guess[cid] = vocab[int(rng.integers(len(vocab)))]
            vocabs[cid] = vocab
            for scenario in (range(config.scenarios) if pooled else [s]):
                layout.setdefault((item, scenario), []).append(cid)
    readings: list[Reading] = []
    for m, a in alphas.items():
        act = config.act[int(m[1:])]
        for scenario in range(config.scenarios):
            for v in range(config.variants):
                for k in range(config.seeds):
                    key = f"{m}|s{scenario}|v{v}|k{k}"
                    for item in range(config.items):
                        if rng.random() >= act:
                            continue
                        mine: list[tuple[str, str]] = []
                        cids = layout.get((item, scenario), [])
                        draw = 0.0
                        for n, cid in enumerate(cids):
                            if config.act_group > 1 and n % config.act_group == 0:
                                draw = float(rng.random())  # one draw for the whole group
                            if cid in fixed:
                                mine.append((cid, fixed[cid]))
                                continue
                            p = 1.0 / (1.0 + math.exp(-(a - betas[cid])))
                            mine.append((cid, _read(rng, truth[cid].admitted, vocabs[cid],
                                                    errors[cid][m], p, guess[cid],
                                                    config.arm == "a5", m == strongest,
                                                    draw if config.act_group > 1 else None)))
                        # With act groups a chunk is its own run, so (run_key, scenario) is
                        # the group of k cells that share one draw (S3: one run per task).
                        size = config.act_group if config.act_group > 1 else max(len(mine), 1)
                        for start in range(0, len(mine), size):
                            part = mine[start:start + size]
                            rkey = (f"{key}|i{item}|c{start // size}"
                                    if config.act_group > 1 else key)
                            clean = all(value == cells[cid].anchor for cid, value in part
                                        if cells[cid].anchor is not None)
                            readings += [Reading(cid, rkey, m, models[m], f"s{scenario}", value,
                                                 clean, v) for cid, value in part]
    readings.sort(key=lambda r: (r.cell, r.run_key))
    return SimSpec(config, cells, readings, truth, models, alphas, betas)


def _read(rng: np.random.Generator, admitted: tuple[str, ...], vocab: tuple[str, ...],
          eps: np.ndarray, p: float, guess: str, a5: bool, strongest: bool,
          u: float | None = None) -> str:
    """One reading under §3.2. In the A5 arm, at a silent cell the strongest model reads
    one guessed value with its accuracy (domain knowledge where T says nothing). u: the
    group's shared faithful-or-error draw (act groups), else a fresh one."""
    if admitted and (rng.random() if u is None else u) < p:
        return admitted[int(rng.integers(len(admitted)))]
    if not admitted and a5 and strongest and (rng.random() if u is None else u) < p:
        return guess
    outside = np.array([0.0 if x in admitted else e for x, e in zip(vocab, eps, strict=True)])
    if outside.sum() <= 0:
        # A pair that is the whole vocabulary (a constraint cell): no value is an error.
        return admitted[int(rng.integers(len(admitted)))]
    return vocab[int(rng.choice(len(vocab), p=outside / outside.sum()))]
