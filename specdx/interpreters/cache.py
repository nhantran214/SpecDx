"""Every run is cached on disk, so a study can be resumed and reproduced."""

from __future__ import annotations

import hashlib
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
