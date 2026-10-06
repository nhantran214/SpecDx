"""Every run is cached on disk, so a study can be resumed and reproduced."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from specdx.types import Behaviour


def run_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


class RunCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Behaviour] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    b = Behaviour.from_json(line)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # a line cut off by a crash; the run will simply be redone
                self._items[b.run_key] = b

    def get(self, key: str) -> Behaviour | None:
        return self._items.get(key)

    def put(self, behaviour: Behaviour) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(behaviour.to_json() + "\n")
        self._items[behaviour.run_key] = behaviour

    def __len__(self) -> int:
        return len(self._items)
