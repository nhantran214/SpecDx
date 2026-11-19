"""The P3a stages: fetch SOPBench, replicate offline, and the local checks.

Each ``*_stage`` returns a StageOutcome; ``run_stage`` writes the report. Network access
is injected (``opener``, ``tree``); the defaults below are the only code that uses urllib.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from specdx.config import REPO_ROOT
from specdx.report.stages import StageOutcome, _cell
from specdx.subjects.sopbench import SOPBENCH_COMMIT, SOPBENCH_DIR
from specdx.subjects.sopbench.cells import cells_for, tell_for
from specdx.subjects.sopbench.diagnose import (
    COMPOSITIONS,
    OFFLINE,
    PROVISIONAL,
    compose,
    diagnose_domain,
)
from specdx.subjects.sopbench.load import S3Run, load_runs, load_tasks
from specdx.subjects.sopbench.load import TrajectoryMismatch as _Mismatch
from specdx.subjects.sopbench.trajectories import (
    MODEL_FAMILIES,
    TrajectoryError,
    fetch,
    parse_name,
    select,
)

TREE_URL = (
    f"https://api.github.com/repos/Leezekun/SOPBench/git/trees/{SOPBENCH_COMMIT}?recursive=1"
)
TOP_DEFECTS = 10
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

    rows: list[list[object]] = []
    top: list[str] = []
    excluded_files: list[tuple[str, str]] = []
    tell_lines: list[tuple[str, int]] = []
    all_defects: list[dict] = []
    theta_data: dict[str, dict] = {}
    n_all = 0
    for domain in chosen:
        tasks = load_tasks(domain, tasks_root)
        runs: list[S3Run] = []
        for entry in sorted((f for f in listed if f["domain"] == domain),
                            key=lambda f: f["path"]):
            name = parse_name(entry["path"], entry.get("size", 0))
            family = entry.get("family") or MODEL_FAMILIES.get(entry.get("canonical", ""), "")
            path = trajectories / domain / Path(entry["path"]).name
            try:
                if name is None:
                    raise TrajectoryError(entry["path"])
                loaded, _ = load_runs(path, name, family, tasks)
            except (_Mismatch, TrajectoryError, KeyError, IndexError) as exc:
                excluded_files.append((entry["path"], f"{type(exc).__name__}: {exc}"))
                continue
            runs += loaded
        tell_lines.append((domain, tell_for(domain, runs)[1]))
        for which in COMPOSITIONS:
            part = compose(runs, which)
            if not part:
                continue
            cd = cells_for(tasks, part)
            dx = diagnose_domain(domain, cd, {r.model: r.family for r in part})
            thetas = sorted(dx.fit.theta.values())
            kinds = Counter(
                "contradiction (unrefuted)" if d.type == "contradiction" and d.note == OFFLINE
                else d.type for d in dx.defects)
            rows.append([
                domain, which, len(dx.raters), len(part), cd.excluded_tool_error,
                f"{thetas[0]:.3f}" if thetas else "-",
                f"{statistics.median(thetas):.3f}" if thetas else "-",
                "yes" if dx.fit.converged else "no",
                kinds["hidden"], kinds["ambiguity"], kinds["weak"],
                kinds["contradiction"], kinds["contradiction (unrefuted)"],
            ])
            theta_data[f"{domain}/{which}"] = {
                "theta": dict(sorted(dx.fit.theta.items())), "converged": dx.fit.converged}
            if which == "all":
                n_all += len(dx.defects)
                goals = {t.index: t.user_goal for t in tasks}
                for d in dx.defects:
                    all_defects.append({
                        "domain": domain, "type": d.type, "location": list(d.location),
                        "scenarios": list(d.scenarios), "score": d.score, "note": d.note,
                        "subtype": d.subtype, "witnesses": list(d.witnesses)})
                for d in dx.defects[:TOP_DEFECTS]:
                    top.append(f"| {domain} | {d.type} | {_cell(', '.join(d.location))} | "
                               f"{d.score:.3f} | {_cell(goals.get(_task_of(d), ''))} |")

    body = [PROVISIONAL, "",
            f"Hidden and contradiction findings were not refuted offline ({OFFLINE}); "
            "contradictions are counted apart as unrefuted candidates.", "",
            "## Table 1: findings by domain and rater composition", ""]
    body += _table(
        ("domain", "composition", "raters", "runs", "excluded runs", "θ min", "θ median",
         "converged", "hidden", "ambiguity", "weak", "contradiction",
         "contradiction (unrefuted)"), rows)
    body += ["", f"## Table 2: the {TOP_DEFECTS} highest-scoring findings per domain (all)", "",
             "| domain | type | location | score | user goal |", "|---|---|---|---|---|"]
    body += top
    body += ["", "## Excluded files", ""]
    body += [f"- `{p}`: {why}" for p, why in excluded_files] or ["None."]
    body += ["", "## Prompt disagreement", "",
             "Runs whose prompt differs from the first run of the same task:", ""]
    body += [f"- {d}: {n}" for d, n in tell_lines]
    data = {"domains": chosen, "defects": all_defects, "theta": theta_data,
            "excluded_files": [p for p, _ in excluded_files]}
    return StageOutcome(0, "DONE", f"{n_all} findings over {len(chosen)} domains",
                        "\n".join(body), data)


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


__all__ = ["fetch_stage", "local_stage", "replication_stage"]
