"""specdx doctor (spec v2 §5.2): is this machine ready to run P1b?

Each check is ok, warn or fail. It checks the Python version and packages, PostgreSQL
(never on another project's port, and reachable), the vendor integrity script, every
Ollama host (reachable, each candidate model present, its native tool calling, and the
same digest on every host), free disk, and VRAM when nvidia-smi is local. It recommends a
hardware profile and prints the commands that pull missing models.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import psycopg

from specdx.config import REPO_ROOT
from specdx.db import PackDatabase
from specdx.eval.screen import load_profile

PACKAGES: tuple[str, ...] = ("numpy", "httpx", "psycopg", "tonomyx", "specdx")
TONOMYX_SRC = Path(os.environ.get("TONOMYX_SRC", "/home/ethan/AILab/TonomyX"))
MIN_FREE_GB = 10.0
# The largest GPU (GiB) each profile is meant for, smallest first; above the last: gpu80.
PROFILE_VRAM: tuple[tuple[str, float], ...] = (("gpu16", 22.0), ("gpu24", 44.0),
                                               ("gpu48", 76.0))

Run = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok", "warn" or "fail"
    detail: str


def check_python(version: tuple[int, int] = (sys.version_info.major, sys.version_info.minor)
                 ) -> list[Check]:
    status = "ok" if version >= (3, 13) else "warn" if version >= (3, 12) else "fail"
    out = [Check("python", status, f"{version[0]}.{version[1]} (3.13 is the tested version)")]
    for name in PACKAGES:
        try:
            out.append(Check(f"package {name}", "ok", importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            out.append(Check(f"package {name}", "fail", "not installed: run the setup step"))
    found = shutil.which("pdftotext")
    out.append(Check("pdftotext", "ok" if found else "warn",
                     found or "missing (poppler): needed unless work/tell is already built"))
    return out


def check_database(url: str) -> Check:
    try:
        PackDatabase(url)  # refuses another project's port before any connection
        with psycopg.connect(url, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except ValueError as exc:
        return Check("postgresql", "fail", str(exc))
    except psycopg.Error as exc:
        return Check("postgresql", "fail", f"unreachable: {exc}".strip())
    return Check("postgresql", "ok", "reachable")


def check_vendor(run: Run = subprocess.run, src: Path = TONOMYX_SRC) -> Check:
    if not src.is_dir():
        return Check("vendor", "warn", f"{src} is absent: vendor/tonomyx cannot be verified "
                                       "here (set TONOMYX_SRC to a TonomyX checkout)")
    done = run(["bash", str(REPO_ROOT / "scripts" / "verify_vendor.sh")], capture_output=True,
               text=True, env={**os.environ, "TONOMYX_SRC": str(src)})
    if done.returncode != 0:
        return Check("vendor", "fail", (done.stdout + done.stderr).strip()[-300:])
    return Check("vendor", "ok", done.stdout.strip())


def check_ollama(
    hosts: Sequence[str], models: Sequence[str], transport: httpx.BaseTransport | None = None
) -> tuple[list[Check], dict[str, list[str]]]:
    """The checks, and the candidate models each reachable host lacks."""
    out: list[Check] = []
    missing: dict[str, list[str]] = {}
    digests: dict[str, dict[str, str]] = {}
    with httpx.Client(transport=transport, timeout=30.0) as client:
        for host in hosts:
            try:
                response = client.get(f"{host}/api/tags")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                out.append(Check(f"ollama {host}", "fail", f"unreachable: {exc}"))
                continue
            listed = {str(m["name"]): str(m["digest"]) for m in response.json().get("models", [])}
            out.append(Check(f"ollama {host}", "ok", f"{len(listed)} models"))
            for model in models:
                name = model if ":" in model else f"{model}:latest"
                if name not in listed:
                    missing.setdefault(host, []).append(model)
                    continue
                digests.setdefault(model, {})[host] = listed[name]
                show = client.post(f"{host}/api/show", json={"model": model})
                tools = show.is_success and "tools" in show.json().get("capabilities", [])
                out.append(Check(f"tools {model} @ {host}", "ok" if tools else "warn",
                                 "native tool calling" if tools
                                 else "no native tool calling: the screen will not admit it"))
    for model, by_host in sorted(digests.items()):
        agree = len(set(by_host.values())) == 1
        detail = (next(iter(by_host.values())) if agree
                  else f"hosts disagree: {by_host}; pull the same tag on every host")
        out.append(Check(f"digest {model}", "ok" if agree else "fail", detail))
    for host, absent in sorted(missing.items()):
        out.append(Check(f"models @ {host}", "warn", "missing: " + ", ".join(absent)))
    return out, missing


def check_disk(path: Path, min_free_gb: float = MIN_FREE_GB) -> Check:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1024**3
    return Check("disk", "ok" if free >= min_free_gb else "warn", f"{free:.0f} GB free at {path}")


def read_vram(run: Run = subprocess.run) -> tuple[float | None, str]:
    """The largest local GPU's memory in GiB, or None, and what nvidia-smi said. A failing
    nvidia-smi (for example a driver/library version mismatch) is reported as such: Ollama
    may then be running on the CPU."""
    try:
        done = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True)
    except OSError:
        return None, "no local nvidia-smi"
    if done.returncode != 0:
        said = (done.stdout + done.stderr).strip().splitlines()
        return None, f"nvidia-smi failed: {said[0] if said else done.returncode}"
    try:
        sizes = [float(x) for x in done.stdout.split()]
    except ValueError:
        return None, f"nvidia-smi output not understood: {done.stdout.strip()[:80]}"
    if not sizes:
        return None, "nvidia-smi lists no GPU"
    return max(sizes) / 1024, f"{max(sizes) / 1024:.0f} GiB"


def recommend_profile(vram: float | None) -> str | None:
    if vram is None:
        return None
    return next((name for name, largest in PROFILE_VRAM if vram < largest), "gpu80")


def pull_commands(missing: Mapping[str, Sequence[str]], local: str) -> list[str]:
    """`ollama pull` for the first host; OLLAMA_HOST selects any other."""
    return [f"ollama pull {m}" if host == local else f"OLLAMA_HOST={host} ollama pull {m}"
            for host, models in sorted(missing.items()) for m in models]


def run_doctor(
    profile: str | None,
    *,
    database_url: str,
    hosts: Sequence[str],
    work_dir: Path,
    transport: httpx.BaseTransport | None = None,
    run: Run = subprocess.run,
) -> tuple[str, bool]:
    """The report, and whether every check passed (warnings allowed)."""
    vram, said = read_vram(run)
    recommended = recommend_profile(vram)
    chosen = load_profile(profile or recommended or "gpu16")
    models = [m for ms in chosen.candidates.values() for m in ms]
    checks = [*check_python(), check_database(database_url), check_vendor(run)]
    ollama, missing = check_ollama(hosts, models, transport)
    checks += [*ollama, check_disk(work_dir),
               Check("vram", "ok" if vram is not None else "warn", said)]
    lines = [f"# specdx doctor (candidates of profile {chosen.name})", "",
             "| check | status | detail |", "|---|---|---|"]
    lines += [f"| {c.name} | {c.status} | {c.detail.replace('|', '/')} |" for c in checks]
    lines += ["", "Recommended profile: " + (recommended or "unknown here; choose by the GPU of "
                                                         "the Ollama host (docs/RUNNING.md)")]
    pulls = pull_commands(missing, hosts[0])
    if pulls:
        lines += ["", "Pull the missing models:", "", *[f"    {p}" for p in pulls]]
    return "\n".join(lines) + "\n", not any(c.status == "fail" for c in checks)
