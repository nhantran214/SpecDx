"""Stage reports and the status dashboard for the P3 experiment stages.

Every stage writes ``<results>/<name>.md`` (report), ``<results>/<name>.json``
(metadata) and optionally ``<results>/<name>.data.json``; ``STATUS.md`` is
rebuilt after each stage. All paths are explicit arguments.
"""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from specdx.config import REPO_ROOT


@dataclass(frozen=True)
class StageDef:
    name: str
    title: str
    plan: str
    implemented: bool


def _s(name: str, title: str, plan: str, implemented: bool = False) -> StageDef:
    return StageDef(name, title, plan, implemented)


STAGES: tuple[StageDef, ...] = (
    _s("local", "Local checks and tests", "P3a", True),
    _s("fetch-sopbench", "Fetch SOPBench data", "P3a", True),
    _s("replication", "Offline replication on SOPBench", "P3a", True),
    _s("s3-screen", "S3 screen", "P3b"),
    _s("s3-dev", "S3 development runs", "P3b"),
    _s("s3-calibrate", "S3 calibration", "P3b"),
    _s("s3-gate", "S3 gate", "P3b"),
    _s("s3-baselines-dev", "S3 baselines on dev", "P3b"),
    _s("s2-screen", "S2 screen", "P3c"),
    _s("s2-dev", "S2 development runs", "P3c"),
    _s("s2-calibrate", "S2 calibration", "P3c"),
    _s("s2-gate", "S2 gate", "P3c"),
    _s("s2-baselines-dev", "S2 baselines on dev", "P3c"),
    _s("freeze", "Freeze the preregistration", "P3b"),
    _s("s3-test", "S3 test split", "P3b"),
    _s("s2-test", "S2 test split", "P3c"),
    _s("s2-real", "S2 real-subject runs", "P3c"),
    _s("ablations", "Ablations", "P3d"),
    _s("status", "Status dashboard", "P3a", True),
)

VERDICTS = ("PASS", "FAIL", "STOP", "DONE", "PENDING")
GitFn = Callable[[list[str]], str]


@dataclass
class StageOutcome:
    exit_code: int
    verdict: str
    key_number: str
    body: str
    data: dict | None = None


def _real_git(args: list[str]) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout


def git_state(git: GitFn | None = None) -> tuple[str, str]:
    """Return (short SHA, 'clean' | 'dirty'); never mutates git."""
    git = git or _real_git
    try:
        sha = git(["rev-parse", "--short", "HEAD"]).strip() or "unknown"
        dirty = bool(git(["status", "--porcelain"]).strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "dirty"
    return sha, "dirty" if dirty else "clean"


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_report(
    name: str,
    outcome: StageOutcome,
    *,
    command: str,
    results: Path,
    started: datetime,
    ended: datetime,
    subject: str = "",
    split: str = "",
    git: GitFn | None = None,
) -> Path:
    """Write ``<name>.md``, ``<name>.json`` and ``<name>.data.json``; return the report path."""
    results.mkdir(parents=True, exist_ok=True)
    sha, state = git_state(git)
    duration = _duration((ended - started).total_seconds())
    rows = [
        ("Started", _iso(started)),
        ("Ended", _iso(ended)),
        ("Duration", duration),
        ("Commit", f"{sha} ({state})"),
        ("Command", f"`{command}`"),
        ("Exit code", str(outcome.exit_code)),
        ("Subject", subject or "-"),
        ("Split", split or "-"),
        ("Verdict", outcome.verdict),
    ]
    title = next((s.title for s in STAGES if s.name == name), name)
    lines = [f"# {name}: {title}", "", "| Field | Value |", "|---|---|"]
    lines += [f"| {k} | {_cell(v)} |" for k, v in rows]
    lines += ["", "## Results", "", outcome.body.rstrip(), ""]
    report = results / f"{name}.md"
    report.write_text("\n".join(lines))
    meta = {
        "stage": name,
        "started": _iso(started),
        "ended": _iso(ended),
        "duration": duration,
        "commit": sha,
        "tree": state,
        "command": command,
        "exit_code": outcome.exit_code,
        "subject": subject,
        "split": split,
        "verdict": outcome.verdict,
        "key_number": outcome.key_number,
        "report": report.name,
    }
    (results / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    if outcome.data is not None:
        (results / f"{name}.data.json").write_text(
            json.dumps(outcome.data, indent=2, sort_keys=True, default=str) + "\n"
        )
    return report


def run_stage(
    name: str,
    fn: Callable[[], StageOutcome],
    *,
    command: str,
    results: Path,
    work: Path | None = None,
    subject: str = "",
    split: str = "",
    git: GitFn | None = None,
) -> int:
    """Run ``fn``, write the stage report and rebuild STATUS.md; return the exit code."""
    started = _now()
    try:
        outcome = fn()
    except Exception:
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-30:])
        outcome = StageOutcome(2, "FAIL", "", f"```\n{tail.rstrip()}\n```")
    write_report(
        name, outcome, command=command, results=results, started=started,
        ended=_now(), subject=subject, split=split, git=git,
    )
    update_status(results, work if work is not None else results.parent / "work")
    return outcome.exit_code


def _meta(name: str, results: Path) -> dict | None:
    try:
        return json.loads((results / f"{name}.json").read_text())
    except (OSError, ValueError):
        return None


def already_done(name: str, results: Path) -> bool:
    meta = _meta(name, results)
    return meta is not None and meta.get("exit_code") == 0


def _pid_alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def update_status(results: Path, work: Path) -> Path:
    """Rebuild ``<results>/STATUS.md`` from the stage metadata and pid files."""
    results.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SpecDx P3 status",
        "",
        f"Built {_iso(_now())}",
        "",
        "| Stage | Plan | Status | Verdict | Key number | Ended | Duration | Report |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for st in STAGES:
        meta = _meta(st.name, results)
        verdict = key = ended = dur = report = "-"
        if meta is not None:
            verdict = meta.get("verdict") or "-"
            key = meta.get("key_number") or "-"
            ended = meta.get("ended") or "-"
            dur = meta.get("duration") or "-"
            fname = meta.get("report", f"{st.name}.md")
            report = f"[{fname}]({fname})"
        if not st.implemented:
            status = f"pending (plan {st.plan})"
        elif _pid_alive(work / f"p3-{st.name}.pid"):
            status = "running"
        elif meta is None:
            status = "not run"
        elif meta.get("exit_code") == 0:
            status = "done"
        else:
            status = f"failed (exit {meta.get('exit_code')})"
        cells = [st.name, st.plan, status, verdict, key, ended, dur, report]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    path = results / "STATUS.md"
    path.write_text("\n".join(lines) + "\n")
    return path
