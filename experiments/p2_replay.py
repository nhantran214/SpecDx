"""G1 (spec v3.1 §8.2): the v3.1 fit, with the parameters frozen at G0, on cached base runs.

Read-only by construction: it reads the run cache (runs.jsonl) and never executes a run,
never opens the database (no Workbench) and never calls Ollama. The Tell PDFs are
extracted into a temporary directory, so nothing under SPECDX_WORK is written; the only
file written is --out. Runs are mapped to the base specification by recomputing
Runner.key's run key, with the mapping specdx/sim/calibrate.py shares with the calibration
(smoke_runs, mapped, P1_SETTINGS, NoRefutation; ruling F7):

- the P1b smoke run: the screened population (experiments/p1b-gate.json) on the smoke
  report's scenarios (results/p1b-smoke-base.json), under the current context digest
  (RUN_SCHEMA_VERSION 3); 60 runs;
- the P1 base runs: the P1 population (experiments/p1-gate.json) on P1's scenarios, under
  P1's context digest (RUN_SCHEMA_VERSION 2 and P1's executor settings); 200 runs.

No refutation is attempted (no run may execute), so a hidden candidate is typed hidden as
an unattempted refutation would type it (v1 §11 item 13).

Criteria (spec v3.1 §8.2): (i) θ_m ≥ 0.5 for gpt-oss:20b and qwen3:30b-a3b on the smoke
run; (ii) R-SO (golden:so:SO-0031:trang_thai, whose cell is grn12|so:SO-0031:trang_thai) is
reported as a hidden requirement, judged on the P1b smoke run only (author ruling,
2026-10-14: P1's two acted readings at that cell equal the anchor), and "not observable"
when no acted behaviour read its cell. The P1 base runs are fitted and reported as a
no-collapse check without a criterion.

Exit status: 0 PASS, 1 FAIL, 2 error (a broken mapping, or provisional parameters).
Usage: SPECDX_WORK=/path/to/work .venv/bin/python experiments/p2_replay.py --out PATH
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from specdx.config import REPO_ROOT, WORK_DIR
from specdx.infer import anchored
from specdx.infer.act import Act, Rasch, fit_rasch
from specdx.infer.anchored import AnchoredFit, V3Params, fit_anchored
from specdx.infer.cells import CellData
from specdx.infer.items import acted_violations
from specdx.infer.typing import CellDiagnosis, diagnose_v3
from specdx.interpreters.cache import run_key
from specdx.interpreters.population import committee
from specdx.interpreters.prompt import load_variants
from specdx.observe.run import context_digest
from specdx.scenarios.space import boundary_domain, canon_scenarios, enumerate_scenarios
from specdx.sim.calibrate import (
    P1_SCHEMA,
    P1_SETTINGS,
    SMOKE_GATE,
    Base,
    CalibrationError,
    NoRefutation,
    gate_members,
    load_cache,
    model_digests,
    open_base,
    smoke_runs,
)
from specdx.types import Behaviour
from specdx.workbench import tell_payload

P1_GATE = REPO_ROOT / "experiments" / "p1-gate.json"
P1_RUNS = 200
THETA_FLOOR = 0.5  # (i)
NO_COLLAPSE: tuple[str, ...] = ("gpt-oss:20b", "qwen3:30b-a3b")
R_SO = "golden:so:SO-0031:trang_thai"  # (ii): SO-0031 stays Chờ duyệt after APR-0102
R_SO_CELL = "grn12|so:SO-0031:trang_thai"


class ReplayError(Exception):
    """The cache does not map as expected, or the parameters are not G0's: no verdict."""


@dataclass
class Replay:
    """unmatched: per table.column, the cells of rows that match no golden row (§12's
    inspection of the row-matching rule)."""

    name: str
    runs: list[Behaviour]
    data: CellData
    rasch: Rasch
    fit: AnchoredFit
    diagnosis: CellDiagnosis
    unmatched: dict[str, int]


def unmatched_rows(base: Base, data: CellData) -> dict[str, int]:
    """Value cells of rows that match no golden row, counted per table.column."""
    out: Counter[str] = Counter()
    for cid, cell in data.cells.items():
        if cell.kind != "value" or "|" not in cid:
            continue
        table, rest = cid.split("|", 1)[1].split(":", 1)
        name = rest.rsplit(":", 1)[0]
        if name not in set(base.cells.rows[table].golden.values()):
            out[cell.location] += 1
    return dict(sorted(out.items()))


def p1_runs(base: Base, cache: Mapping[str, Behaviour],
            digests: Mapping[str, str]) -> list[Behaviour]:
    """P1's population on P1's scenarios: one per parameter vector of the boundary domain
    without T literals, a canon name winning over a synthesised id."""
    by_signature = {s.signature(): s for s in enumerate_scenarios(boundary_domain(base.pack))}
    for s in canon_scenarios(base.pack).values():
        by_signature[s.signature()] = s
    context = context_digest(tell_payload(load_variants()), P1_SETTINGS, schema=P1_SCHEMA)
    out = []
    for _, scenario in sorted(by_signature.items()):
        params = json.dumps(scenario.params(), sort_keys=True)
        for interp in committee(gate_members(P1_GATE), (0, 1, 2), (1, 2, 3)):
            if interp.model not in digests:
                continue
            key = run_key(base.variant.digest(), base.steps_digest, context, params,
                          interp.key, "", digests[interp.model])
            if (b := cache.get(key)) is not None:
                out.append(dataclasses.replace(b, scenario=scenario, interpreter=interp))
    if len(out) != P1_RUNS:
        raise ReplayError(f"P1 base: {len(out)} runs mapped, expected {P1_RUNS}")
    return out


def replay(name: str, base: Base, runs: list[Behaviour],
           population: Mapping[str, tuple[str, ...]], params: V3Params) -> Replay:
    lens = base.lens
    acts: list[Act] = []
    violated: dict[str, frozenset[str]] = {}
    for b in runs:
        violated[b.run_key] = acted_violations(lens.views(b, lens.violations(b)))
        acts.extend(lens.acts(b))
    rasch = fit_rasch(acts)
    data = base.cells.extract(runs)
    models = {m: f for f, ms in population.items() for m in ms}
    fit = fit_anchored(data.cells, data.readings, models=models, params=params)
    diagnosis = diagnose_v3(runs, violated, data, fit, NoRefutation(),
                            observable=rasch.observable_items(), relevant=lens.relevant)
    return Replay(name, runs, data, rasch, fit, diagnosis, unmatched_rows(base, data))


def theta_holds(r: Replay) -> tuple[bool, str]:
    """(i): no collapse. Every named model has θ_m ≥ THETA_FLOOR."""
    low = [m for m in NO_COLLAPSE if r.fit.theta.get(m, 0.0) < THETA_FLOOR]
    shown = ", ".join(f"{m} {r.fit.theta.get(m, float('nan')):.3f}" for m in NO_COLLAPSE)
    return not low, f"θ: {shown}"


def r_so_status(r: Replay) -> str:
    """(ii): reported, missed, or not observable (no acted behaviour read the cell)."""
    if not any(x.cell == R_SO_CELL for x in r.data.readings):
        return "not observable"
    reported = any(d.type == "hidden" and d.location == (R_SO,) for d in r.diagnosis.defects)
    return "reported" if reported else "missed"


def verdict(smoke: Replay) -> bool:
    return theta_holds(smoke)[0] and r_so_status(smoke) != "missed"


def _cell_line(r: Replay) -> str:
    post = r.fit.cells.get(R_SO_CELL)
    if post is None:
        return "no reading"
    readings = Counter((x.model, x.value) for x in r.data.readings if x.cell == R_SO_CELL)
    forms = ", ".join(f"{k} {v:.3f}" for k, v in post.forms.items())
    return (f"anchor {post.anchor!r}; readings {dict(sorted(readings.items()))}; "
            f"P(A ≠ {{d}}) = {post.p_defect:.3f}; forms {forms}; MAP {sorted(post.map_set)}")


def render(replays: list[Replay], params: V3Params) -> str:
    smoke = replays[0]
    passed = verdict(smoke)
    ok_i, why_i = theta_holds(smoke)
    out = ["<!-- generated by experiments/p2_replay.py -->", "",
           f"# G1: {'PASS' if passed else 'FAIL'}", "",
           f"Parameters (G0, frozen: {anchored.PARAMS_FROZEN}): {dataclasses.asdict(params)}.",
           "", "## Criteria (spec v3.1 §8.2, on the P1b smoke run)", "",
           f"- (i) θ_m ≥ {THETA_FLOOR} for {' and '.join(NO_COLLAPSE)}: "
           f"{'yes' if ok_i else 'no'} ({why_i}).",
           f"- (ii) R-SO (`{R_SO}`) reported as a hidden requirement: {r_so_status(smoke)}.",
           ""]
    for r in replays:
        kinds = Counter(c.kind for c in r.data.cells.values())
        anchored_cells = sum(c.anchor is not None for c in r.data.cells.values())
        typed = Counter(d.type for d in r.diagnosis.defects)
        out += [f"## {r.name}", "",
                f"{len(r.runs)} runs; {len(r.data.cells)} cells ({kinds['value']} value, "
                f"{kinds['event']} event, {kinds['constraint']} constraint; {anchored_cells} "
                f"anchored); {len(r.data.readings)} readings; "
                f"{len(r.rasch.observable_items())} of {len(r.rasch.delta)} items observable.",
                "", f"EM: converged {r.fit.converged} after {r.fit.iterations} iterations "
                f"from start {r.fit.start}; log posterior per start: "
                + ", ".join(f"{k} {v.score:.1f}" for k, v in r.fit.starts.items()) + ".", "",
                "| model | α | θ | readings |", "|---|---:|---:|---:|"]
        reads = Counter(x.model for x in r.data.readings)
        out += [f"| {m} | {r.fit.alpha[m]:.3f} | {r.fit.theta[m]:.3f} | {reads[m]} |"
                for m in sorted(r.fit.alpha)]
        out += ["", "Family effect: " + ", ".join(
            f"{f} {v:+.3f}" for f, v in sorted(r.fit.family_effect.items())) + ".", "",
            f"Typed without refutation (C/H/A/W): {typed['contradiction']}/{typed['hidden']}/"
            f"{typed['ambiguity']}/{typed['weak']}; unobservable {len(r.diagnosis.unobservable)}.",
            "", f"R-SO cell `{R_SO_CELL}`: {_cell_line(r)}.", "",
            "Rows matching no golden row (spec v3 §12: inspect the row-matching rule):", ""]
        out += [f"- {loc}: {n} cells" for loc, n in r.unmatched.items()] or ["- none"]
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provisional", action="store_true",
                        help="run with V3Params' provisional defaults (a plan pre-check, "
                             "never G1)")
    args = parser.parse_args(argv)
    try:
        if not anchored.PARAMS_FROZEN and not args.provisional:
            raise ReplayError("V3Params is not frozen: run G0 and freeze its parameters first")
        params = V3Params()
        cache = load_cache(WORK_DIR / "runs.jsonl")
        digests = model_digests(cache)
        with tempfile.TemporaryDirectory() as pdf:
            base = open_base(Path(pdf))
            replays = [
                replay("P1b smoke run (base)", base, smoke_runs(base, cache, digests),
                       gate_members(SMOKE_GATE), params),
                replay("P1 base runs", base, p1_runs(base, cache, digests),
                       gate_members(P1_GATE), params)]
    except (ReplayError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(replays, params), encoding="utf-8")
    return 0 if verdict(replays[0]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
