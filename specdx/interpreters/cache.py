"""Every run is cached on disk, so a study can be resumed and reproduced."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from specdx.types import Behaviour


def run_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


class RunCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Behaviour] = {}
        self._needs_newline = False
        if path.is_file():
            data = path.read_bytes()
            self._needs_newline = bool(data and not data.endswith(b"\n"))
            for chunk in data.split(b"\n"):
                if not chunk:
                    continue
                try:
                    line = chunk.decode("utf-8")
                    b = Behaviour.from_json(line)
                except (ValueError, KeyError, TypeError):
                    continue  # a line cut off by a crash; the run will simply be redone
                self._items[b.run_key] = b

    def get(self, key: str) -> Behaviour | None:
        return self._items.get(key)

    def put(self, behaviour: Behaviour) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            if self._needs_newline:
                handle.write("\n")
                self._needs_newline = False
            handle.write(behaviour.to_json() + "\n")
        self._items[behaviour.run_key] = behaviour

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Behaviour]:
        return iter(self._items.values())


def cache_summary(cache: RunCache) -> str:
    """specdx cache-info (spec v2 §5.2): cached runs per model and digest, and per
    specification. Runs cached before P1b do not record their specification."""
    by_model = Counter((b.interpreter.model, b.model_digest, b.status) for b in cache)
    by_spec = Counter(b.spec or "(not recorded: a P1 run)" for b in cache)
    lines = [f"# Run cache: {cache.path} ({len(cache)} runs)", "",
             "| model | digest | status | runs |", "|---|---|---|---:|"]
    lines += [f"| {m} | {d or '(none)'} | {s} | {n} |" for (m, d, s), n in sorted(by_model.items())]
    lines += ["", "| specification | runs |", "|---|---:|"]
    lines += [f"| {spec} | {n} |" for spec, n in sorted(by_spec.items())]
    return "\n".join(lines) + "\n"
