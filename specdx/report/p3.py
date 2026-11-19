"""The P3a stages: fetch SOPBench, replicate offline, and the local checks.

Each ``*_stage`` returns a StageOutcome; ``run_stage`` writes the report. Network access
is injected (``opener``, ``tree``); the defaults below are the only code that uses urllib.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from specdx.config import REPO_ROOT
from specdx.infer.cells import CellData
from specdx.report.stages import StageOutcome, _cell
from specdx.subjects.sopbench import SOPBENCH_COMMIT, SOPBENCH_DIR
from specdx.subjects.sopbench.cells import cells_for, prompt_variants
from specdx.subjects.sopbench.diagnose import (
    COMPOSITIONS,
    OFFLINE,
    PROVISIONAL,
    compose,
    diagnose_domain,
)
from specdx.subjects.sopbench.load import S3Run, S3Task, load_runs, load_tasks
from specdx.subjects.sopbench.load import TrajectoryMismatch as _Mismatch
from specdx.subjects.sopbench.trajectories import (
    MODEL_FAMILIES,
    TrajectoryError,
    fetch,
    parse_name,
    select,
)
from specdx.types import Defect

TREE_URL = (
    f"https://api.github.com/repos/Leezekun/SOPBench/git/trees/{SOPBENCH_COMMIT}?recursive=1"
)
TOP_DEFECTS = 10
SATURATED = 0.999
NA_OFFLINE = "n/a offline"
MISSING_EVALUATIONS = "no evaluations (unevaluated upstream)"
SHA_MISMATCH = "sha256 mismatch"
Runner = Callable[[list[str]], tuple[int, str]]


def http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned hosts
        return response.read()


def github_tree() -> list[dict]:
    """The pinned tree's files as {"path", "size"}; refuses a truncated listing."""
    payload = json.loads(http_get(TREE_URL))
    if payload.get("truncated"):
        raise OSError("the GitHub tree listing was truncated")
    return [{"path": e["path"], "size": e.get("size", 0)}
            for e in payload["tree"] if e.get("type") == "blob"]


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def fetch_stage(
    dest: Path,
    manifest: Path,
    *,
    opener: Callable[[str], bytes] = http_get,
    tree: Callable[[], list[dict]] = github_tree,
) -> StageOutcome:
    try:
        names = [n for e in tree() if (n := parse_name(e["path"], e["size"])) is not None]
        chosen = select(names)
        result = fetch(chosen, dest, manifest, opener=opener)
    except (OSError, ValueError, KeyError) as exc:  # URLError is an OSError
        kind = type(exc).__name__
        return StageOutcome(2, "FAIL", "", f"{kind}: {exc}")
    files = result["files"]
    by_domain = Counter(f["domain"] for f in files)
    by_family: dict[str, set[str]] = {}
    for f in files:
        by_family.setdefault(f["family"], set()).add(f["canonical"])
    models = {f["canonical"] for f in files}
    key = f"{len(files)} files, {len(models)} models, {len(by_family)} families"
    body = _table(("domain", "files"), sorted(by_domain.items()))
    body += ["", "Models by family:", ""]
    body += [f"- {fam}: {', '.join(sorted(ms))}" for fam, ms in sorted(by_family.items())]
    body += ["", f"Pinned at SOPBench commit `{SOPBENCH_COMMIT}`; manifest `{manifest}`."]
    return StageOutcome(0, "DONE", key, "\n".join(body))


def _task_of(defect) -> int:
    if defect.type == "contradiction":
        return int(defect.scenarios[0].rsplit(":", 1)[1])
    return int(defect.location[0].split(":")[2])


def _sha256(path: Path) -> str:
    """The file's sha256, read in 1 MiB blocks (trajectory files reach tens of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _kind(defect: Defect) -> str:
    if defect.type == "contradiction" and defect.note == OFFLINE:
        return "contradiction (unrefuted)"
    return defect.type


def _deviating(defect: Defect, data: CellData, by_cell: Mapping[str, list]) -> int:
    """The readings at the finding's cells that differ from the cell's anchor."""
    return sum(1 for c in defect.location for r in by_cell.get(c, ())
               if r.value != data.cells[c].anchor)


def rank_findings(defects: Sequence[Defect], deviating: Mapping[tuple, int]) -> list[Defect]:
    """By score, then by the number of deviating readings (both descending), then by key:
    most scores saturate at 1.000, so the score alone does not rank (M-2)."""
    return sorted(defects, key=lambda d: (-d.score, -deviating.get(d.key, 0), d.key))


def _load_domain(
    domain: str, listed: Sequence[dict], trajectories: Path, tasks: Sequence[S3Task],
    excluded_files: list[tuple[str, str]],
) -> tuple[list[S3Run], dict[str, int]]:
    """Every file of the domain that matches its manifest sha256 and aligns with the tasks.
    A file that is missing, differs from the manifest, is misaligned, or gives no run is
    excluded with its reason; nothing is dropped silently (rulings I-2, I-4)."""
    runs: list[S3Run] = []
    load = {"files": 0, "loaded": 0, "excluded": 0, "records": 0, "runs": 0,
            "missing_evaluation": 0}
    for entry in sorted((f for f in listed if f["domain"] == domain), key=lambda f: f["path"]):
        load["files"] += 1
        name = parse_name(entry["path"], entry.get("size", 0))
        family = entry.get("family") or MODEL_FAMILIES.get(entry.get("canonical", ""), "")
        path = trajectories / domain / Path(entry["path"]).name
        reason = ""
        try:
            if name is None:
                raise TrajectoryError(entry["path"])
            actual = _sha256(path)
            if actual != entry.get("sha256"):
                reason = (f"{SHA_MISMATCH} (manifest {str(entry.get('sha256'))[:12]}…, "
                          f"file {actual[:12]}…)")
            else:
                loaded, stats = load_runs(path, name, family, tasks)
                load["records"] += stats.records
                load["missing_evaluation"] += stats.missing_evaluation
                if not loaded:
                    reason = MISSING_EVALUATIONS
        except FileNotFoundError:
            reason = "missing file: run fetch-sopbench"
        except (OSError, _Mismatch, TrajectoryError, KeyError, IndexError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if reason:
            excluded_files.append((entry["path"], reason))
            load["excluded"] += 1
            continue
        load["loaded"] += 1
        load["runs"] += len(loaded)
        runs += loaded
    return runs, load


def _variant_split(runs: Sequence[S3Run], variants: Mapping[str, int]) -> dict:
    by_variant = Counter(variants[r.run_key] for r in runs)
    models = Counter(r.model for r in runs if variants[r.run_key] != 0)
    return {"runs": {str(v): n for v, n in sorted(by_variant.items())},
            "models": dict(sorted(models.items()))}


TABLE1 = ("domain", "composition", "raters", "runs", "excluded runs", "θ min", "θ median",
          "converged", "hidden", "ambiguity", "weak", "contradiction",
          "contradiction (unrefuted)")
KINDS = ("hidden", "ambiguity", "weak", "contradiction (unrefuted)")


def replication_stage(
    *,
    manifest: Path,
    trajectories: Path,
    domains: Sequence[str] | None = None,
    tasks_root: Path = SOPBENCH_DIR,
) -> StageOutcome:
    manifest, trajectories = Path(manifest), Path(trajectories)
    try:
        listed = json.loads(manifest.read_text())["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return StageOutcome(2, "FAIL", "", f"manifest {manifest} is missing or unreadable: {exc}")
    if not trajectories.is_dir():
        return StageOutcome(2, "FAIL", "", f"trajectory directory {trajectories} is missing; "
                            "run fetch-sopbench first")
    present = sorted({f["domain"] for f in listed})
    chosen = sorted(domains) if domains else present
    absent = [d for d in chosen if d not in present]
    if absent:
        return StageOutcome(2, "FAIL", "", f"domains not in the manifest: {', '.join(absent)}")

    table1: list[dict] = []
    saturation: dict[str, dict] = {}
    top: list[str] = []
    excluded_files: list[tuple[str, str]] = []
    loading: dict[str, dict[str, int]] = {}
    variant_data: dict[str, dict] = {}
    failures: list[str] = []
    all_defects: list[dict] = []
    theta_data: dict[str, dict] = {}
    n_all = 0
    for domain in chosen:
        tasks = load_tasks(domain, tasks_root)
        runs, loading[domain] = _load_domain(domain, listed, trajectories, tasks,
                                             excluded_files)
        if not runs:
            failures.append(f"{domain}: no runs (every file was excluded)")
            continue
        variants = prompt_variants(runs)
        variant_data[domain] = _variant_split(runs, variants)
        for which in COMPOSITIONS:
            part = compose(runs, which)
            cd = cells_for(tasks, part, variants)
            if not cd.data.readings:
                if which == "all":
                    failures.append(f"{domain}: no readings (every run has a tool error)")
                    break
                continue
            dx = diagnose_domain(domain, cd, {r.model: r.family for r in part})
            by_cell = cd.data.by_cell()
            dev = {d.key: _deviating(d, cd.data, by_cell) for d in dx.defects}
            ranked = rank_findings(dx.defects, dev)
            thetas = sorted(dx.fit.theta.values())
            kinds = Counter(_kind(d) for d in dx.defects)
            table1.append(dict(zip(TABLE1, (
                domain, which, len(dx.raters), len(part), cd.excluded_tool_error,
                f"{thetas[0]:.3f}" if thetas else "-",
                f"{statistics.median(thetas):.3f}" if thetas else "-",
                "yes" if dx.fit.converged else "no",
                kinds["hidden"], kinds["ambiguity"], kinds["weak"], NA_OFFLINE,
                kinds["contradiction (unrefuted)"]), strict=True)))
            saturation[f"{domain}/{which}"] = {
                "findings": len(dx.defects),
                "at_0.999": sum(d.score >= SATURATED for d in dx.defects),
                "distinct_tasks": {k: len({_task_of(d) for d in dx.defects if _kind(d) == k})
                                   for k in KINDS}}
            theta_data[f"{domain}/{which}"] = {
                "theta": dict(sorted(dx.fit.theta.items())), "converged": dx.fit.converged}
            if which == "all":
                n_all += len(dx.defects)
                goals = {t.index: t.user_goal for t in tasks}
                for d in ranked:
                    all_defects.append({
                        "domain": domain, "type": d.type, "location": list(d.location),
                        "scenarios": list(d.scenarios), "score": d.score, "note": d.note,
                        "subtype": d.subtype, "witnesses": list(d.witnesses),
                        "deviating": dev[d.key]})
                for d in ranked[:TOP_DEFECTS]:
                    top.append(f"| {domain} | {d.type} | {_cell(', '.join(d.location))} | "
                               f"{d.score:.3f} | {dev[d.key]} | "
                               f"{_cell(goals.get(_task_of(d), ''))} |")

    body = [PROVISIONAL, "",
            f"Hidden and contradiction findings were not refuted offline ({OFFLINE}); "
            "contradictions are counted apart as unrefuted candidates, and the plain "
            f"contradiction column, which needs a refutation, reads {NA_OFFLINE}.", ""]
    if failures:
        body += ["## Failures", ""] + [f"- {f}" for f in failures] + [""]
    body += ["## Table 1: findings by domain and rater composition", ""]
    body += _table(TABLE1, [[r[h] for h in TABLE1] for r in table1])
    body += ["", f"## Table 2: the {TOP_DEFECTS} highest-ranked findings per domain (all)", "",
             "Ranked by score, then by the number of readings that deviate from the anchor.",
             "", "| domain | type | location | score | deviating readings | user goal |",
             "|---|---|---|---|---|---|"]
    body += top
    body += ["", f"## Table 3: saturation and distinct tasks (findings at score ≥{SATURATED})",
             "", "A task has up to four cells, and an ambiguity on an anchored cell also "
             "brings its hidden pick, so the distinct tasks per type are the count to read.", ""]
    body += _table(
        ("domain", "composition", "findings", f"≥{SATURATED}",
         *(f"distinct tasks: {k}" for k in KINDS)),
        [[*key.split("/"), s["findings"], s["at_0.999"],
          *(s["distinct_tasks"][k] for k in KINDS)] for key, s in saturation.items()])
    body += ["", "## Loading", ""]
    body += _table(("domain", "files", "loaded", "excluded", "records", "runs",
                    "missing evaluations"),
                   [[d, *v.values()] for d, v in loading.items()])
    body += ["", "## Excluded files", ""]
    body += [f"- `{p}`: {why}" for p, why in excluded_files] or ["None."]
    body += ["", "## Prompt variants", "",
             "Within each task the most common rendered prompt is variant 0, and every other "
             "distinct prompt is variant 1, 2, … by decreasing frequency, decided per run "
             "(published files mix renders). T is built from variant 0. The variant is "
             "confounded with model generation: the earlier models saw an older render, so "
             "offline a variant effect cannot be separated from a rater's accuracy.", ""]
    for d, v in variant_data.items():
        other = sum(n for k, n in v["runs"].items() if k != "0")
        line = f"- {d}: variant 0: {v['runs'].get('0', 0)} runs; other variants: {other} runs"
        if other:
            per = ", ".join(f"{k}: {n}" for k, n in v["runs"].items() if k != "0")
            models = ", ".join(f"{m} {n}" for m, n in v["models"].items())
            line += f" (by variant {per}; models: {models})"
        body.append(line)
    data = {"domains": chosen, "defects": all_defects, "theta": theta_data,
            "table1": table1, "saturation": saturation, "loading": loading,
            "variants": variant_data, "failures": failures,
            "excluded_files": [p for p, _ in excluded_files],
            "excluded_reasons": dict(excluded_files)}
    key = f"{n_all} findings over {len(chosen)} domains"
    if failures:
        return StageOutcome(2, "FAIL", key, "\n".join(body), data)
    return StageOutcome(0, "DONE", key, "\n".join(body), data)




def _run(cmd: list[str]) -> tuple[int, str]:
    done = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-15:])
    return done.returncode, tail


def local_stage(
    *,
    runner: Runner = _run,
    manifest: Path,
    trajectories: Path,
) -> StageOutcome:
    bin_dir = Path(sys.executable).parent
    checks = (
        ("pytest", [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-m", "not db"]),
        ("ruff", [str(bin_dir / "ruff"), "check", "specdx", "tests", "scripts", "experiments"]),
    )
    failed = False
    body: list[str] = []
    for label, cmd in checks:
        code, tail = runner(cmd)
        failed |= code != 0
        body += [f"## {label}: {'ok' if code == 0 else f'FAILED (exit {code})'}", "",
                 "```", tail, "```", ""]
    if Path(manifest).exists():
        smoke = replication_stage(manifest=manifest, trajectories=trajectories,
                                  domains=["library"])
        body.append(f"Replication smoke (library): {smoke.verdict}, {smoke.key_number}")
        if smoke.exit_code != 0:
            body.append(smoke.body)
            failed = True
    else:
        body.append("replication smoke skipped: run fetch-sopbench first")
    return StageOutcome(1 if failed else 0, "FAIL" if failed else "PASS", "", "\n".join(body))


__all__ = ["fetch_stage", "local_stage", "rank_findings", "replication_stage"]
