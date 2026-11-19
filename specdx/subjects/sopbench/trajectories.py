"""Select and fetch SOPBench's published trajectories (design §13d).

Network access is injected through ``opener``; only the CLI supplies a real one.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from specdx.subjects.sopbench import SOPBENCH_COMMIT

RAW_URL = "https://raw.githubusercontent.com/Leezekun/SOPBench/{commit}/{path}"
MIN_SIZE = 10_000
MODE_ORDER = ("fc", "react", "act-only", "react-v")

_NAME = re.compile(
    r"^output/(?P<domain>[^/]+)/(?:usr_(?P<user>.+?)-)?ast_(?P<model>.+?)"
    r"-mode_(?P<mode>.+?)-dep_full-fmt_structured-tool_(?P<tool>.+?)-shuffle_False\.json$"
)

MODEL_FAMILIES: dict[str, str] = {
    **dict.fromkeys(
        (
            "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini",
            "o1", "o4-mini", "o4-mini-high",
        ),
        "openai",
    ),
    **dict.fromkeys(
        (
            "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-20250219-thinking",
        ),
        "anthropic",
    ),
    **dict.fromkeys(
        (
            "gemini-1.5-pro", "gemini-2.0-flash-001", "gemini-2.0-flash-thinking-exp",
            "gemini-2.5-flash", "gemini-2.5-pro", "gemma-4-e2b-it", "gemma-4-e4b-it",
        ),
        "google",
    ),
    **dict.fromkeys(("llama3.1-8b-instruct", "llama3.1-70b-instruct"), "meta"),
    **dict.fromkeys(
        (
            "qwen2.5-7b-instruct", "qwen2.5-14b-instruct", "qwen2.5-32b-instruct",
            "qwen2.5-72b-instruct", "qwen3.5-2b", "qwen3.5-4b",
        ),
        "qwen",
    ),
    "deepseek-r1": "deepseek",
}

OPEN_WEIGHTS: frozenset[str] = frozenset(
    m for m in MODEL_FAMILIES if m.startswith(("gemma-", "llama", "qwen", "deepseek-r1"))
)


class UnknownModels(ValueError):
    """Published trajectories use model names missing from MODEL_FAMILIES."""


class TrajectoryError(ValueError):
    """A trajectory file is unreadable or is not a JSON list."""


@dataclass(frozen=True)
class TrajectoryName:
    path: str
    domain: str
    user: str | None
    model: str
    canonical: str
    mode: str
    tool_list: str
    size: int


def parse_name(path: str, size: int) -> TrajectoryName | None:
    m = _NAME.match(path)
    if m is None:
        return None
    model = m["model"]
    return TrajectoryName(
        path=path,
        domain=m["domain"],
        user=m["user"],
        model=model,
        canonical=model.replace("-it-fireworks", "-instruct"),
        mode=m["mode"],
        tool_list=m["tool"],
        size=size,
    )


def _rank(n: TrajectoryName) -> tuple[int, bool, str]:
    mode = MODE_ORDER.index(n.mode) if n.mode in MODE_ORDER else len(MODE_ORDER)
    return (mode, n.model != n.canonical, n.path)


def select(names: Sequence[TrajectoryName]) -> list[TrajectoryName]:
    kept = [
        n
        for n in names
        if n.user is None
        and n.tool_list == "full"
        and n.model != "human"
        and n.size >= MIN_SIZE
    ]
    unknown = sorted({n.canonical for n in kept} - set(MODEL_FAMILIES))
    if unknown:
        raise UnknownModels(f"unknown SOPBench models: {', '.join(unknown)}")
    best: dict[tuple[str, str], TrajectoryName] = {}
    for n in kept:
        key = (n.domain, n.canonical)
        if key not in best or _rank(n) < _rank(best[key]):
            best[key] = n
    return sorted(best.values(), key=lambda n: n.path)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(
    selected: Sequence[TrajectoryName],
    dest: Path,
    manifest: Path,
    *,
    opener: Callable[[str], bytes],
) -> dict:
    dest, manifest = Path(dest), Path(manifest)
    known: dict[str, str] = {}
    if manifest.exists():
        try:
            old = json.loads(manifest.read_text())
            known = {f["path"]: f["sha256"] for f in old.get("files", [])}
        except (ValueError, KeyError, TypeError, AttributeError):
            known = {}
    files = []
    for n in sorted(selected, key=lambda n: n.path):
        target = dest / n.domain / Path(n.path).name
        data = None
        if target.exists() and n.path in known:
            cached = target.read_bytes()
            if _sha(cached) == known[n.path]:
                data = cached
        if data is None:
            data = opener(RAW_URL.format(commit=SOPBENCH_COMMIT, path=n.path))
            try:
                parsed = json.loads(data)
            except ValueError as exc:
                raise TrajectoryError(n.path) from exc
            if not isinstance(parsed, list):
                raise TrajectoryError(n.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        files.append(
            {
                "path": n.path,
                "domain": n.domain,
                "model": n.model,
                "canonical": n.canonical,
                "family": MODEL_FAMILIES[n.canonical],
                "open_weights": n.canonical in OPEN_WEIGHTS,
                "mode": n.mode,
                "size": len(data),
                "sha256": _sha(data),
            }
        )
    result = {
        "commit": SOPBENCH_COMMIT,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result


__all__ = [
    "MODEL_FAMILIES",
    "OPEN_WEIGHTS",
    "TrajectoryError",
    "TrajectoryName",
    "UnknownModels",
    "fetch",
    "parse_name",
    "select",
]
