"""The P4a stages: G0v4 on the development domains, the execution witnesses, the v4
replication with its ablations, and the permutation placebo (design §6).

Each ``*_stage`` returns a StageOutcome; ``run_stage`` writes the report. Nothing here fetches
data: a stage without its trajectories exits 2 and says which stage to run first. Smoke mode
(``smoke=True``) shrinks every stage to a system check: it never freezes parameters, and its
reports are labelled by ``run_stage``. Official runs happen on the 48 GB host.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from specdx.infer.anchored import V3Params
from specdx.infer.anchored_v4 import V4Params
from specdx.report.p3 import _kind, _load_domain, _table
from specdx.report.stages import StageOutcome
from specdx.sim.calibrate import CalibrationError
from specdx.sim.calibrate_s3 import calibrate_s3, load_split
from specdx.sim.g0v4 import (
    CELL_SD,
    LAMBDA_PRIOR,
    PARAMS_FILE,
    REPLICATES,
    TAU_A,
    TAU_H,
    freeze_v4,
    params_dict,
    render,
    run_g0v4,
)
from specdx.subjects.sopbench import DOMAINS, SOPBENCH_DIR
from specdx.subjects.sopbench.cells import cells_for, prompt_variants
from specdx.subjects.sopbench.diagnose import diagnose_domain
from specdx.subjects.sopbench.diagnose_v4 import WITNESSED, S3DiagnosisV4, diagnose_domain_v4
from specdx.subjects.sopbench.load import load_tasks
from specdx.subjects.sopbench.placebo import PlaceboResult, changeable_cells, placebo
from specdx.subjects.sopbench.witness import load_raw_tasks, witness

SMOKE_PARAMS = "smoke-v4-params.json"
SMOKE_REPLICATES = 2
SMOKE_TASKS = 20
SMOKE_SEEDS = 2
SMOKE_GRID: dict[str, Sequence] = {
    "cell_sd": CELL_SD[:1], "lambda_prior": LAMBDA_PRIOR[:1], "tau_u": (2.0,),
    "tau_h": TAU_H[:1], "tau_a": TAU_A[:1]}
PLACEBO_SEEDS = 20
PLACEBO_MAX_RATIO = 0.2
PLACEBO_MIN_EVALUABLE = 2
TAU_U_NOTE = ("tau_u is selected by the G0v4 simulation; it is not certified by the theory "
              "(the T6 bound does not give it).")
REFUTATION_NOTE = ("Typed hidden findings were not refuted offline; witnessed ones were "
                   "refuted by SOPBench's strict replay.")


def _fail(message: str) -> StageOutcome:
    return StageOutcome(2, "FAIL", "", message)


def _load(
    manifest: Path, trajectories: Path, domains: Sequence[str], tasks_root: Path,
) -> tuple[dict[str, tuple[list, list]], StageOutcome | None]:
    """domain -> (tasks, runs) for each of `domains`, or an exit-2 outcome that says what is
    missing. A domain whose files are all excluded is an error, with the reasons."""
    manifest, trajectories = Path(manifest), Path(trajectories)
    try:
        listed = json.loads(manifest.read_text())["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {}, _fail(f"manifest {manifest} is missing or unreadable: {exc}")
    if not trajectories.is_dir():
        return {}, _fail(f"trajectory directory {trajectories} is missing; run fetch-sopbench "
                         "first (nothing is fetched implicitly)")
    absent = sorted(set(domains) - {f["domain"] for f in listed})
    if absent:
        return {}, _fail(f"domains not in the manifest: {', '.join(absent)}")
    out: dict[str, tuple[list, list]] = {}
    problems: list[str] = []
    for domain in domains:
        tasks = load_tasks(domain, tasks_root)
        excluded: list[tuple[str, str]] = []
        runs, _ = _load_domain(domain, listed, trajectories, tasks, excluded)
        if not runs:
            why = "; ".join(f"{p}: {r}" for p, r in excluded) or "no files"
            problems.append(f"{domain}: no runs ({why})")
            continue
        out[domain] = (tasks, runs)
    if problems:
        listing = "\n".join(f"- {p}" for p in problems)
        return {}, _fail("trajectory data is incomplete:\n\n" + listing)
    return out, None


def _cells(domain: str, tasks: list, runs: list):
    return cells_for(tasks, runs, prompt_variants(runs))


def _models(runs: Sequence) -> dict[str, str]:
    return {r.model: r.family for r in runs}


def _fmt(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.3f}"


def _theta(fit) -> tuple[str, str]:
    thetas = sorted(fit.theta.values())
    if not thetas:
        return "-", "-"
    return f"{thetas[0]:.3f}", f"{statistics.median(thetas):.3f}"


def _resolve_params(params_file: Path, results: Path, smoke: bool,
                    ) -> tuple[V4Params, str] | None:
    """The frozen parameters; in smoke mode, failing that, the smoke parameters that the smoke
    s3-g0v4 wrote. None when there are neither."""
    candidates = [(Path(params_file), "frozen")]
    if smoke:
        candidates.append((Path(results) / SMOKE_PARAMS, "smoke (not frozen)"))
    for path, label in candidates:
        if path.is_file():
            raw = json.loads(path.read_text())
            return V4Params(**raw.get("params", raw)), f"{label}: `{path}`"
    return None


def _no_params(params_file: Path, smoke: bool) -> StageOutcome:
    where = f"{params_file}" + (f" or {SMOKE_PARAMS} in the smoke results" if smoke else "")
    return _fail(f"no v4 parameters ({where}): run s3-g0v4 first")


# --- s3-g0v4 ---

def g0v4_stage(
    *, manifest: Path, trajectories: Path, results: Path, split: tuple | None = None,
    tasks_root: Path = SOPBENCH_DIR, replicates: int = REPLICATES,
    grid: Mapping[str, Sequence] | None = None, jobs: int = 1, smoke: bool = False,
    params_file: Path = PARAMS_FILE,
) -> StageOutcome:
    """Calibrate on the development domains, run G0v4 and, on PASS in an official run, freeze
    the chosen parameters. Smoke: 2 replicates, the first value of each grid axis (and
    tau_u = 2), and the parameters go to ``smoke-v4-params.json`` in `results`; never frozen."""
    dev = tuple((split or load_split())[0])
    if smoke:
        replicates, grid = SMOKE_REPLICATES, dict(SMOKE_GRID)
    data, error = _load(manifest, trajectories, dev, tasks_root)
    if error is not None:
        return error
    cds = [_cells(d, *data[d]) for d in dev]
    runs = [r for d in dev for r in data[d][1]]
    try:
        config = calibrate_s3(cds, runs, dev)
    except CalibrationError as exc:
        return _fail(f"calibration failed: {exc}")
    out = Path(results) / "s3-g0v4-search"
    code = run_g0v4(config, out, replicates=replicates, grid=dict(grid) if grid else None,
                    jobs=jobs)
    if code == 2:
        return _fail("G0v4 raised an error; the traceback is in the stage's stderr")
    result = json.loads(out.with_suffix(".json").read_text())
    chosen = V4Params(**result["chosen"]["params"])
    verdict = "PASS" if code == 0 else "FAIL"
    if smoke:
        (Path(results) / SMOKE_PARAMS).write_text(
            json.dumps({"params": params_dict(chosen)}, indent=2) + "\n")
        frozen = f"SMOKE: parameters written to {SMOKE_PARAMS}; nothing frozen."
    elif code == 0:
        freeze_v4(chosen, Path(params_file))
        frozen = f"Frozen to `{params_file}`."
    else:
        frozen = "Not frozen: the verdict is not PASS."
    metrics, v3 = result["chosen"]["metrics"], result["v3"]
    body = ["## Calibration (development domains)", "",
            f"Development domains: {', '.join(dev)}; held out: "
            f"{', '.join((split or load_split())[1])}.", ""]
    body += _table(("domain", "tasks", "runs"),
                   [[d, len(data[d][0]), len(data[d][1])] for d in dev])
    cal = dataclasses.asdict(config)
    body += ["", "| field | value |", "|---|---|"]
    body += [f"| {k} | {_fmt_cal(v)} |" for k, v in cal.items()]
    body += ["", "## Selected parameters", "", "| parameter | value |", "|---|---|"]
    body += [f"| {k} | {v} |" for k, v in params_dict(chosen).items()]
    body += ["", TAU_U_NOTE, "", frozen, "", "## v3 beside v4 (selection replicates)", ""]
    body += _table(("quantity", "v3", "v4"), [
        ["hidden F1", _fmt(v3["hidden_f1"]), _fmt(metrics["hidden_f1"])],
        ["ambiguity F1", _fmt(v3["ambiguity_f1"]), _fmt(metrics["ambiguity_f1"])],
        ["hidden false-positive rate on shared cells", _fmt(v3["shared_hidden"]),
         _fmt(metrics["shared_fp"])]])
    shown = render(result).split("\n", 3)[3]  # without the generated-by line and the title
    body += ["", "## G0v4 in full", "", shown.replace("\n## ", "\n### ")]
    data_out = {"verdict": verdict, "dev": list(dev), "config": cal,
                "params": params_dict(chosen), "criteria": result["criteria"],
                "failed": result["failed"], "metrics": metrics, "v3": v3,
                "frozen": bool(code == 0 and not smoke), "smoke": smoke}
    return StageOutcome(code, verdict, f"G0v4 {verdict}, tau_u={chosen.tau_u}",
                        "\n".join(body), data_out)


def _fmt_cal(v: object) -> str:
    if isinstance(v, tuple):
        return ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v)
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# --- s3-witness ---

def witness_stage(
    *, tasks_root: Path = SOPBENCH_DIR, domains: Sequence[str] = DOMAINS,
    limit: int | None = None, smoke: bool = False,
    witness_fn: Callable[[str, dict], object] = witness,
) -> StageOutcome:
    """Witness every task of every domain (smoke: the first 20 per domain): the strict replay
    matches the task label (match), contradicts it (mismatch) or fails (error)."""
    if smoke:
        limit = SMOKE_TASKS
    rows: list[list] = []
    mismatches: list[dict] = []
    errors: list[dict] = []
    for domain in domains:
        try:
            tasks = load_tasks(domain, tasks_root)
            raw = load_raw_tasks(domain, tasks_root)
        except (OSError, ValueError, KeyError) as exc:
            return _fail(f"cannot read the {domain} tasks under {tasks_root}: {exc}")
        n = Counter()
        for task, record in list(zip(tasks, raw, strict=True))[:limit]:
            anchor = "succeeds" if task.should_succeed else "refused"
            result = witness_fn(domain, record)
            if result.outcome == "error":
                n["error"] += 1
                errors.append({"domain": domain, "task": task.index, "error": result.error})
            elif result.outcome == anchor:
                n["match"] += 1
            else:
                n["mismatch"] += 1
                mismatches.append({"domain": domain, "task": task.index, "anchor": anchor,
                                   "replay": result.outcome, "calls": list(result.calls),
                                   "goal": task.user_goal})
        rows.append([domain, sum(n.values()), n["match"], n["mismatch"], n["error"]])
    total = [sum(r[i] for r in rows) for i in range(1, 5)]
    body = ["Each task is replayed on SOPBench's strict system. match: the replay's outcome "
            "equals the task label; mismatch: it differs, so the label contradicts the "
            "task's own constraints; error: the replay failed.", "",
            *_table(("domain", "tasks", "match", "mismatch", "error"), [*rows, ["total", *total]]),
            "", "## Mismatches", ""]
    body += _table(("domain", "task", "label", "replay", "user goal"),
                   [[m["domain"], m["task"], m["anchor"], m["replay"], m["goal"]]
                    for m in mismatches]) if mismatches else ["None."]
    body += ["", "## Errors", ""]
    body += _table(("domain", "task", "error"),
                   [[e["domain"], e["task"], e["error"]] for e in errors]) if errors \
        else ["None."]
    data = {"domains": {r[0]: dict(zip(("tasks", "match", "mismatch", "error"), r[1:],
                                       strict=True)) for r in rows},
            "mismatches": mismatches, "errors": errors, "limit": limit}
    key = f"{total[1]} match, {total[2]} mismatch, {total[3]} error"
    return StageOutcome(0, "DONE", key, "\n".join(body), data)


# --- s3-v4 ---

V4_COLUMNS = ("domain", "raters", "hidden", "ambiguity", "weak", "contradiction (unrefuted)",
              "undetermined outcome", "undetermined check", "witnessed defect",
              "witnessed shared", "witnessed error", "cells accounted", "shared typed",
              "shared witnessed", "θ min", "θ median", "converged")


def _typed(dx: S3DiagnosisV4) -> Counter:
    """The typed defects by kind; witnessed hidden findings are counted as witnessed."""
    return Counter(_kind(d) for d in dx.defects if not d.note.startswith("witnessed"))


def _v4_row(label: str, dx: S3DiagnosisV4) -> dict:
    """One row of counts. cells accounted = undetermined (outcome + check) + witnessed defect
    + witnessed shared: a witness error is already an undetermined outcome cell, and a
    witnessed shared cell is not also a typed shared one."""
    typed, u, w = _typed(dx), dx.undetermined, dx.witnessed
    lo, med = _theta(dx.fit)
    witnessed_shared = sum(f.values == (WITNESSED,) for f in dx.shared)
    return {
        "domain": label, "raters": len(dx.raters), "hidden": typed["hidden"],
        "ambiguity": typed["ambiguity"], "weak": typed["weak"],
        "contradiction (unrefuted)": typed["contradiction (unrefuted)"],
        "undetermined outcome": u["outcome"], "undetermined check": u["check"],
        "witnessed defect": w["defect"], "witnessed shared": w["shared"],
        "witnessed error": w["error"],
        "cells accounted": u["outcome"] + u["check"] + w["defect"] + w["shared"],
        "shared typed": len(dx.shared) - witnessed_shared,
        "shared witnessed": witnessed_shared,
        "θ min": lo, "θ median": med, "converged": "yes" if dx.fit.converged else "no"}


def _row_cells(row: dict) -> list:
    return [row[c] for c in V4_COLUMNS]


V4_NOTE = ("undetermined outcome counts every outcome cell the data cannot decide, including "
           "those whose witness errored (witnessed error); cells accounted = undetermined "
           "outcome + undetermined check + witnessed defect + witnessed shared, so a witness "
           "error is not counted twice. Shared errors: typed ones come from the fit "
           "(MAP set {d}, lambda at least tau_lambda); witnessed ones are cells whose replay "
           "agreed with the label.")


def _variants(params: V4Params) -> list[tuple[str, V4Params, dict]]:
    return [
        ("no per-cell β", params, {"cell_beta": False}),
        ("no composite weights", params, {"composite": False}),
        ("no witnesses", params, {"use_witness": False}),
        ("plug-in instead of profile", params, {"profile": False}),
    ]


def v4_stage(
    *, manifest: Path, trajectories: Path, results: Path, split: tuple | None = None,
    tasks_root: Path = SOPBENCH_DIR, smoke: bool = False, params_file: Path = PARAMS_FILE,
    params: V4Params | None = None,
) -> StageOutcome:
    """v4 on the held-out domains with the frozen parameters, beside v3 on the same data and
    the ablations of design §6.4. Smoke: one domain, and the smoke parameters if nothing is
    frozen."""
    held = tuple((split or load_split())[1])
    if smoke:
        held = held[:1]
    source = "given"
    if params is None:
        found = _resolve_params(params_file, results, smoke)
        if found is None:
            return _no_params(params_file, smoke)
        params, source = found
    data, error = _load(manifest, trajectories, held, tasks_root)
    if error is not None:
        return error
    main_rows: list[dict] = []
    v3_rows: list[list] = []
    ablation_rows: list[list] = []
    for domain in held:
        tasks, runs = data[domain]
        cd = _cells(domain, tasks, runs)
        models = _models(runs)
        raw = load_raw_tasks(domain, tasks_root)
        full = diagnose_domain_v4(domain, cd, models, raw, params)
        base = _v4_row(domain, full)
        main_rows.append(base)
        ablation_rows.append(_row_cells({**base, "domain": f"full v4 / {domain}"}))
        for label, p, kw in _variants(params):
            dx = diagnose_domain_v4(domain, cd, models, raw, p, **kw)
            ablation_rows.append(_row_cells({**_v4_row(domain, dx),
                                             "domain": f"{label} / {domain}"}))
        v3 = diagnose_domain(domain, cd, models)
        kinds = Counter(_kind(d) for d in v3.defects)
        lo, med = _theta(v3.fit)
        v3_rows.append([domain, len(v3.raters), kinds["hidden"], kinds["ambiguity"],
                        kinds["weak"], kinds["contradiction (unrefuted)"], lo, med,
                        "yes" if v3.fit.converged else "no"])
        ablation_rows.append([f"full v3 / {domain}", len(v3.raters), kinds["hidden"],
                              kinds["ambiguity"], kinds["weak"],
                              kinds["contradiction (unrefuted)"], *["n/a"] * 8, lo, med,
                              "yes" if v3.fit.converged else "no"])
        ablation_rows.append([f"no attractor / {domain}", "vacuous on S3 (binary cells)",
                              *["-"] * (len(V4_COLUMNS) - 2)])
    total = [sum(r[c] for r in main_rows) for c in V4_COLUMNS[2:14]]
    body = [f"Held-out domains: {', '.join(held)}. Parameters: {source}.", "", TAU_U_NOTE,
            REFUTATION_NOTE, "", "## v4 per domain", ""]
    body += _table(V4_COLUMNS, [*(_row_cells(r) for r in main_rows),
                                ["total", "", *total, "", "", ""]])
    body += ["", V4_NOTE, "", "## v3 on the same data", ""]
    body += _table(("domain", "raters", "hidden", "ambiguity", "weak",
                    "contradiction (unrefuted)", "θ min", "θ median", "converged"), v3_rows)
    body += ["", "## Ablations (design §6.4)", "",
             "Each row refits the held-out domain with one component removed. No per-cell β: "
             "β_c is set equal to its group's β_g in the profiles and in the fit. The "
             "attractor ablation has nothing to remove on S3: its cells are binary, so the "
             "error side has one slot and λ is not identified.", ""]
    body += _table(V4_COLUMNS, ablation_rows)
    payload = {"domains": list(held), "params": params_dict(params), "source": source,
               "v4": main_rows, "v3": v3_rows, "ablations": ablation_rows, "smoke": smoke}
    typed = sum(r["hidden"] + r["ambiguity"] + r["weak"] + r["contradiction (unrefuted)"]
                for r in main_rows)
    undet = sum(r["undetermined outcome"] + r["undetermined check"] for r in main_rows)
    key = f"{typed} typed defects, {undet} undetermined over {len(held)} domains"
    return StageOutcome(0, "DONE", key, "\n".join(body), payload)


# --- s3-placebo ---

def placebo_verdict(results: Sequence[PlaceboResult]) -> tuple[str, list[str], list[str]]:
    """(verdict, evaluable domains, domains over the bound). A domain whose real v4 count is 0
    is not evaluable and is excluded (ruling P-N). PASS needs at least PLACEBO_MIN_EVALUABLE
    evaluable domains and every evaluable v4 ratio at most PLACEBO_MAX_RATIO."""
    evaluable = [r for r in results if r.real_v4 > 0]
    over = [r.domain for r in evaluable if not r.ratio_v4 <= PLACEBO_MAX_RATIO]
    ok = len(evaluable) >= PLACEBO_MIN_EVALUABLE and not over
    return "PASS" if ok else "FAIL", [r.domain for r in evaluable], over


UNANIMOUS_FATES = ("defect", "shared", "undetermined", "witnessed_defect", "witnessed_shared",
                   "witnessed_error", "none")


def unanimous_fates(cd, dx: S3DiagnosisV4) -> dict[str, Counter]:
    """Cells all of whose readings are equal (not changeable), split into those that agree with
    the anchor and those that deviate, each counted by v4's fate for the cell (design §6.2)."""
    changeable = changeable_cells(cd)
    value: dict[str, str] = {}
    for r in cd.data.readings:
        value[r.cell] = r.value
    out: dict[str, Counter] = {"agreeing": Counter(), "deviating": Counter()}
    for cell, v in value.items():
        if cell in changeable:
            continue
        kind = "agreeing" if v == cd.data.cells[cell].anchor else "deviating"
        out[kind][dx.cell_outcomes.get(cell, "none")] += 1
    return out


def placebo_stage(
    *, manifest: Path, trajectories: Path, results: Path, split: tuple | None = None,
    tasks_root: Path = SOPBENCH_DIR, seeds: Sequence[int] = range(PLACEBO_SEEDS),
    smoke: bool = False, params_file: Path = PARAMS_FILE, params: V4Params | None = None,
) -> StageOutcome:
    """The permutation placebo on the held-out domains (smoke: one domain, 2 seeds), with the
    verdict of ruling P-N and the unanimous cells reported apart (design §6.2)."""
    held = tuple((split or load_split())[1])
    if smoke:
        held, seeds = held[:1], range(SMOKE_SEEDS)
    source = "given"
    if params is None:
        found = _resolve_params(params_file, results, smoke)
        if found is None:
            return _no_params(params_file, smoke)
        params, source = found
    data, error = _load(manifest, trajectories, held, tasks_root)
    if error is not None:
        return error
    outcomes: list[PlaceboResult] = []
    unanimous: dict[str, dict[str, Counter]] = {}
    for domain in held:
        tasks, runs = data[domain]
        models = _models(runs)
        cd = _cells(domain, tasks, runs)
        dx = diagnose_domain_v4(domain, cd, models, load_raw_tasks(domain, tasks_root), params)
        # The diagnosis's fit is the real v4 fit: the placebo does not fit the real data again.
        outcomes.append(placebo(domain, tasks, runs, models, params, V3Params(), list(seeds),
                                real_fit=dx.fit))
        unanimous[domain] = unanimous_fates(cd, dx)
    verdict, evaluable, over = placebo_verdict(outcomes)
    rows = []
    for r in outcomes:
        status = ("not evaluable (real v4 = 0)" if r.real_v4 == 0
                  else "over the bound" if r.domain in over else "ok")
        rows.append([r.domain, r.changeable, r.real_v4,
                     _fmt(statistics.fmean(r.placebo_v4)) if r.placebo_v4 else "n/a",
                     _fmt(r.ratio_v4), r.real_v3,
                     _fmt(statistics.fmean(r.placebo_v3)) if r.placebo_v3 else "n/a",
                     _fmt(r.ratio_v3), status])
    body = [f"Held-out domains: {', '.join(held)}; seeds: {len(list(seeds))}. Parameters: "
            f"{source}.", "", TAU_U_NOTE, "",
            f"Verdict: {verdict}. PASS needs at least {PLACEBO_MIN_EVALUABLE} evaluable "
            f"domains and every evaluable v4 ratio at most {PLACEBO_MAX_RATIO}. A domain "
            "whose real v4 count is 0 is not evaluable: it is reported and excluded.", "",
            f"Evaluable domains: {', '.join(evaluable) or 'none'}."
            + (f" Over the bound: {', '.join(over)}." if over else ""), "",
            "## Placebo per domain", "",
            "Counts are cells: the changeable cells (cells whose readings differ) that carry "
            "at least one typed defect (hidden, ambiguity or weak), each cell counted once, "
            "before witnesses. A placebo shuffles the act vectors among each task's runs "
            "without a tool error (a tool-error run gives no reading and keeps its own "
            "record), so every task's readings, and the changeable cells, are kept. Placebo "
            "columns are the mean over the seeds; a ratio is the mean placebo count over the "
            "real count.", ""]
    body += _table(("domain", "changeable cells", "real v4", "placebo v4 (mean)", "v4 ratio",
                    "real v3", "placebo v3 (mean)", "v3 ratio", "status"), rows)
    body += ["", "## Unanimous cells (reported apart)", "",
             "Cells with identical readings cannot change under the permutation, so they are "
             "reported apart, split by whether the unanimous reading agrees with the anchor, "
             "with v4's fate for each cell (design §6.2; a unanimous cell is as undecidable "
             "when it agrees as when it deviates). \"none\" on an agreeing cell is the "
             "candidate-value convention, not evidence: an unobserved value is not a "
             "candidate, so the cell has no defect form to test (theory note D5, Proposition "
             "T5.5(c)). Fates are mutually exclusive: witnessed_error is an undetermined "
             "outcome cell whose replay failed.",
             ""]
    urows = []
    for domain, kinds in unanimous.items():
        for kind in ("deviating", "agreeing"):
            c = kinds[kind]
            urows.append([domain, kind, sum(c.values()), *(c[f] for f in UNANIMOUS_FATES)])
    body += _table(("domain", "unanimous", "cells", *UNANIMOUS_FATES), urows)
    payload = {
        "verdict": verdict, "evaluable": evaluable, "over": over, "domains": list(held),
        "params": params_dict(params), "smoke": smoke,
        "results": [dataclasses.asdict(r) for r in outcomes],
        "unanimous": {d: {k: dict(c) for k, c in v.items()} for d, v in unanimous.items()}}
    key = (f"{len(evaluable)} of {len(held)} domains evaluable, "
           f"{len(over)} over {PLACEBO_MAX_RATIO}")
    return StageOutcome(0 if verdict == "PASS" else 1, verdict, key, "\n".join(body), payload)
