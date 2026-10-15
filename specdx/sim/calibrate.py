"""Calibration of the simulation from cached base runs (spec v3 §8.1), read-only.

It reads the run cache (the P1b smoke run's population runs and the capability screen's
runs on happy), maps runs to the base specification the way Runner.key keys them, and
computes summary statistics of base runs only, never of a mutant: per-model accuracy and
act rates, the dispersion of cell accuracy, error concentration, the cell mix and the
number of values per column (resolution 15). Readings per cell follow from the planned
budget; the base runs' readings per cell are reported beside the simulated ones. It never
opens the database, never calls a model, and writes nothing under the run cache's
directory: the Tell PDFs are extracted into a temporary directory. A statistic that cannot
be computed raises CalibrationError: there is no silent fallback (ruling F13).

The run mapping (smoke_runs, mapped, P1_SETTINGS, NoRefutation) is shared with G1's replay
(experiments/p2_replay.py). experiments/p1b_replay.py keeps its own copies: it is an
archived script whose results are pinned (ruling F7).

Usage: SPECDX_WORK=/path/to/work python -m specdx.sim.calibrate --out PATH
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK, PDF_DIR, REPO_ROOT, WORK_DIR
from specdx.db import golden_tables, split_background
from specdx.eval.screen import ScreenResult, rescreen
from specdx.infer.act import drafting_steps
from specdx.infer.cells import Cell, CellData, CellLens
from specdx.infer.items import SpecLens
from specdx.interpreters.cache import run_key
from specdx.interpreters.executor import interpreter_settings
from specdx.interpreters.population import committee
from specdx.interpreters.prompt import load_variants
from specdx.observe.run import context_digest
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import boundary_domain, canon_scenarios, enumerate_scenarios
from specdx.scenarios.tguided import extract_literals, tboundary_scenarios
from specdx.sim.baselines import by_cell as by_cell_of
from specdx.sim.generator import SimConfig, generate
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import TellCorpus, pdf_chunks
from specdx.types import Behaviour, InterpreterId, RefuteResult, Scenario
from specdx.variant import SpecVariant
from specdx.workbench import tell_payload

SCREEN = REPO_ROOT / "results" / "p1b-screen.json"
SMOKE = REPO_ROOT / "results" / "p1b-smoke-base.json"
SMOKE_GATE = REPO_ROOT / "experiments" / "p1b-gate.json"
SMOKE_RUNS = 60  # the P1b smoke run: 4 scenarios x its population x 3 variants x 3 seeds
# P1's context digest (RUN_SCHEMA_VERSION 2 and P1's executor settings), for P1's base runs.
P1_SCHEMA = 2
P1_SETTINGS: dict[str, float | int] = {
    "temperature": 0.7, "max_steps": 16, "timeout": 180.0, "num_ctx": 16384, "search_k": 5,
    "search_reply_limit": 12000,
}
# The planned budget (spec v3 §8.2, G0): population 200 runs per specification, which
# escalates floor(200 / (models x 3 variants x 3 seeds)) scenarios to the full committee.
BUDGET = 200
DEFECT_RATE = 0.15  # planted, not calibrated: the base's defect rate is unknown
CONCENTRATIONS: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


class CalibrationError(RuntimeError):
    """The cache does not map as expected, or a statistic has no data: no statistic is
    computed from a broken mapping, and none falls back to a default."""


class NoRefutation:
    """No run may execute in a replay: every candidate stays unattempted."""

    def refute(self, targets: frozenset[str], scenario: Scenario,
               behaviours: Sequence[Behaviour]) -> RefuteResult:
        return RefuteResult(False, None, 0, attempted=False)


@dataclass(frozen=True)
class Calibration:
    """population: the §7 population, the screen with act rate ≥ 0.3. accuracy: per model,
    the share of its readings equal to d_c at anchored cells whose plurality among the
    population's readings is d_c. dispersion: the standard deviation over those cells of
    their logit accuracy, from the population's readings. top_error: the mean share of a
    cell's most common error among its errors. reads: the median readings per model of a
    per-scenario cell in the base runs, reported beside simulated_reads, the same statistic
    on a specification generated from config() (readings per cell follow from the planned
    budget). The cell mix and values are shares of columns, as the generator draws them: a
    column is a pooled cell, or a per-scenario cell's row and column across scenarios."""

    population: dict[str, tuple[str, ...]]
    act: dict[str, float]
    accuracy: dict[str, float]
    dispersion: float
    top_error: float
    error_conc: float
    reads: float
    pooled_share: float
    anchored_share: float
    constraint_share: float
    values: tuple[float, ...]
    columns: int
    items: int
    runs: int
    simulated_reads: float = 0.0

    def config(self, seed: int = 0) -> SimConfig:
        models = [m for ms in self.population.values() for m in ms]
        models.sort(key=lambda m: -self.accuracy[m])
        scenarios = max(1, BUDGET // (len(models) * 9))
        return SimConfig(
            alphas=tuple(_logit(self.accuracy[m]) for m in models),
            act=tuple(self.act[m] for m in models), columns=self.columns,
            pooled_share=self.pooled_share, scenarios=scenarios, items=self.items,
            beta_sd=self.dispersion, anchored_share=self.anchored_share,
            constraint_share=self.constraint_share, values=self.values,
            defect_rate=DEFECT_RATE, error_conc=self.error_conc, seed=seed)

    def to_json(self) -> str:
        raw = dataclasses.asdict(self)
        raw["population"] = {f: list(ms) for f, ms in self.population.items()}
        raw["values"] = list(self.values)
        config = self.config()
        raw["config"] = {k: list(v) if isinstance(v, tuple) else v
                         for k, v in dataclasses.asdict(config).items()}
        raw["models"] = sorted((m for ms in self.population.values() for m in ms),
                               key=lambda m: -self.accuracy[m])
        return json.dumps(raw, ensure_ascii=False, indent=2)


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def load_cache(path: Path) -> dict[str, Behaviour]:
    out: dict[str, Behaviour] = {}
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                b = Behaviour.from_json(raw.decode("utf-8"))
                out[b.run_key] = b
    return out


def model_digests(cache: Mapping[str, Behaviour]) -> dict[str, str]:
    seen: dict[str, set[str]] = {}
    for b in cache.values():
        seen.setdefault(b.interpreter.model, set()).add(b.model_digest)
    if any(len(v) != 1 for v in seen.values()):
        raise CalibrationError(f"a model has several digests in the cache: {seen}")
    return {m: next(iter(v)) for m, v in seen.items()}


@dataclass
class Base:
    """The base specification as the smoke run saw it, built without a database."""

    pack: CanonPack
    variant: SpecVariant
    lens: SpecLens
    cells: CellLens
    scenarios: dict[str, Scenario]
    steps_digest: str
    context: str


def open_base(pdf_cache: Path) -> Base:
    pack = CanonPack(BASE_PACK)
    steps = load_work_orders()
    corpus = TellCorpus.build(pack, pdf_chunks(PDF_DIR, pdf_cache))
    variant = SpecVariant("base", BASE_PACK, corpus)
    clauses = ClauseSet.from_pack(pack, steps)
    background, _ = split_background(golden_tables(pack))
    lens = SpecLens(clauses, steps, ColumnModel.of(pack), drafting=drafting_steps(pack, steps),
                    initial=background)
    canon = canon_scenarios(pack)
    literals = extract_literals(corpus)
    every = [*canon.values(), *enumerate_scenarios(boundary_domain(pack, [x.bound for x in
                                                                          literals])),
             *tboundary_scenarios(canon, literals)]
    return Base(pack, variant, lens, CellLens(lens, pack, corpus),
                {s.id: s for s in every},
                hashlib.sha256(repr(steps).encode()).hexdigest(),
                context_digest(tell_payload(load_variants()), interpreter_settings()))


def mapped(base: Base, cache: Mapping[str, Behaviour], digests: Mapping[str, str],
           scenario: Scenario, interp: InterpreterId) -> Behaviour | None:
    """The cached base run of this interpreter on this scenario, under the current run key
    (schema 3), with the caller's scenario and interpreter as Runner.run returns it."""
    if interp.model not in digests:
        return None
    key = run_key(base.variant.digest(), base.steps_digest, base.context,
                  json.dumps(scenario.params(), sort_keys=True), interp.key, "",
                  digests[interp.model])
    b = cache.get(key)
    return None if b is None else dataclasses.replace(b, scenario=scenario, interpreter=interp)


def screen_results(path: Path = SCREEN) -> list[ScreenResult]:
    return rescreen(path).results


def gate_members(path: Path) -> dict[str, tuple[str, ...]]:
    """A gate configuration's population, family → models."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {f: tuple(ms) for f, ms in raw["models_by_family"].items()}


def smoke_runs(base: Base, cache: Mapping[str, Behaviour],
               digests: Mapping[str, str]) -> list[Behaviour]:
    """The P1b smoke run: its population (experiments/p1b-gate.json) on the smoke report's
    scenarios, Tell variants 0–2 and seeds 1–3, in scenario then committee order."""
    scenarios = json.loads(SMOKE.read_text(encoding="utf-8"))["scenarios"]
    members = committee(gate_members(SMOKE_GATE), (0, 1, 2), (1, 2, 3))
    out = [b for sid in scenarios for i in members
           if (b := mapped(base, cache, digests, base.scenarios[sid], i)) is not None]
    if len(out) != SMOKE_RUNS:
        raise CalibrationError(f"smoke: {len(out)} runs mapped, expected {SMOKE_RUNS} runs")
    return out


def base_runs(base: Base, cache: Mapping[str, Behaviour]) -> list[Behaviour]:
    """The smoke population's runs on the smoke scenarios, and every screened model's runs
    on happy (Tell variants 0–2, seeds 1–3). 18 runs are both."""
    digests = model_digests(cache)
    out = {b.run_key: b for b in smoke_runs(base, cache, digests)}
    for r in screen_results():
        for v in (0, 1, 2):
            for s in (1, 2, 3):
                interp = InterpreterId(r.model, r.family, v, s)
                if (b := mapped(base, cache, digests, base.scenarios["happy"], interp)):
                    out[b.run_key] = b
    return [out[k] for k in sorted(out)]


def _top_error(data: CellData) -> float:
    shares = []
    for rs in data.by_cell().values():
        values = Counter(r.value for r in rs)
        mode = min(values, key=lambda v: (-values[v], v))
        errors = Counter({v: n for v, n in values.items() if v != mode})
        if sum(errors.values()) >= 2:
            shares.append(max(errors.values()) / sum(errors.values()))
    if not shares:
        raise CalibrationError("no cell has two errors: the error concentration has no data")
    return float(np.mean(shares))


def column_of(cell_id: str) -> str:
    """A cell's column, as the generator draws one: a pooled cell is its own column; a
    per-scenario cell's column is its row and column across scenarios."""
    return cell_id.split("|", 1)[1] if "|" in cell_id else cell_id


def _columns(data: CellData) -> dict[str, list[Cell]]:
    out: dict[str, list[Cell]] = {}
    for cid in sorted(data.cells):
        out.setdefault(column_of(cid), []).append(data.cells[cid])
    return out


def _median_reads(by_cell: Mapping[str, Sequence[Any]], per_scenario: Any) -> float:
    """The median readings per model of a per-scenario cell."""
    reads = [n for cid, rs in by_cell.items() if per_scenario(cid)
             for n in Counter(r.model for r in rs).values()]
    if not reads:
        raise CalibrationError("no per-scenario cell was read")
    return float(np.median(reads))


def _matching_concentration(top: float, seed: int = 0) -> float:
    """The Dirichlet concentration whose error tendencies give the observed top-error share,
    by simulation over three to six error values, the nearest of CONCENTRATIONS."""
    rng = np.random.default_rng(seed)

    def share(conc: float) -> float:
        out = []
        for _ in range(2000):
            k = int(rng.integers(3, 7))
            out.append(float(rng.dirichlet(np.full(k, conc)).max()))
        return float(np.mean(out))

    return min(CONCENTRATIONS, key=lambda c: (abs(share(c) - top), c))


def calibrate(base: Base, runs: Sequence[Behaviour]) -> Calibration:
    """Resolution 15's statistics. The plurality, the accuracy and the dispersion read the
    population's readings only; the cell mix and |V| are shares of columns (ruling F12)."""
    screened = rescreen(SCREEN)
    results, population = screened.results, screened.population
    members = {m for ms in population.values() for m in ms}
    data = base.cells.extract(runs)
    by_cell = data.by_cell()
    accuracy_hits: dict[str, list[int]] = {m: [0, 0] for m in members}
    cell_logits = []
    for cid, cell in data.cells.items():
        rs = [r for r in by_cell[cid] if r.model in members]
        if cell.anchor is None or not rs:
            continue
        values = Counter(r.value for r in rs)
        if min(values, key=lambda v: (-values[v], v)) != cell.anchor:
            continue
        for r in rs:
            accuracy_hits[r.model][0] += r.value == cell.anchor
            accuracy_hits[r.model][1] += 1
        hits = sum(r.value == cell.anchor for r in rs)
        cell_logits.append(_logit((hits + 0.5) / (len(rs) + 1)))
    unread = sorted(m for m, (_, n) in accuracy_hits.items() if n == 0)
    if unread:
        raise CalibrationError(f"no anchored reading to measure accuracy for {unread}")
    if not cell_logits:
        raise CalibrationError("no anchored cell to measure the dispersion of accuracy")
    accuracy = {m: (h + 0.5) / (n + 1) for m, (h, n) in accuracy_hits.items()}
    rates = {r.model: r.act_rate for r in results}
    missing = sorted(members - set(rates))
    if missing:
        raise CalibrationError(f"the screen has no act rate for {missing}")
    columns = _columns(data)
    value_columns = [cs for cs in columns.values() if cs[0].kind != "constraint"]
    sizes = Counter(
        sorted(min(len({r.value for r in by_cell[c.id]}) + 1, 7) for c in cs)[(len(cs) - 1) // 2]
        for cs in value_columns)
    top = _top_error(data)
    calibration = Calibration(
        population=population,
        act={m: rates[m] for m in sorted(members)},
        accuracy=dict(sorted(accuracy.items())),
        dispersion=float(min(max(np.std(cell_logits), 0.25), 3.0)),
        top_error=top, error_conc=_matching_concentration(top),
        reads=_median_reads(by_cell, lambda cid: "|" in cid),
        pooled_share=sum("|" not in cs[0].id for cs in columns.values()) / len(columns),
        anchored_share=sum(2 * sum(c.anchor is not None for c in cs) >= len(cs)
                           for cs in value_columns) / len(value_columns),
        constraint_share=(len(columns) - len(value_columns)) / len(columns),
        values=tuple(sizes[k] / len(value_columns) for k in range(2, 8)),
        columns=len(columns), items=len(base.lens.steps), runs=len(runs))
    simulated = generate(calibration.config())
    return dataclasses.replace(calibration, simulated_reads=_median_reads(
        by_cell_of(simulated), lambda cid: not cid.endswith("|pooled")))


def summary(c: Calibration) -> dict[str, Any]:
    return {"population": c.population, "accuracy": c.accuracy, "act": c.act,
            "dispersion": c.dispersion, "top_error": c.top_error, "error_conc": c.error_conc,
            "reads": c.reads, "simulated_reads": c.simulated_reads, "columns": c.columns,
            "runs": c.runs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the calibration JSON")
    args = parser.parse_args(argv)
    try:
        cache = load_cache(WORK_DIR / "runs.jsonl")
        with tempfile.TemporaryDirectory() as pdf:
            base = open_base(Path(pdf))
            calibration = calibrate(base, base_runs(base, cache))
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(calibration.to_json(), encoding="utf-8")
    print(json.dumps(summary(calibration), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
