"""Load the playbooks a language-model executor is given, as data.

A playbook is the words a model needs where a scripted handler has code: what a
correct outcome for one intent looks like. They live in playbooks.json beside this
module, keyed by digital employee and intent, so a new intent is a data edit. They
are a projection kept outside the canon pack, which stays read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from tonomyx.de.executor import Playbook

_PLAYBOOKS = Path(__file__).with_name("playbooks.json")


def load_playbooks(path: Path = _PLAYBOOKS) -> dict[tuple[str, str], Playbook]:
    """Every playbook, keyed by (digital employee, intent). Keys starting '_' are notes."""
    raw: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    playbooks: dict[tuple[str, str], Playbook] = {}
    for de, intents in raw.items():
        if de.startswith("_") or not isinstance(intents, dict):
            continue
        for intent, entry in intents.items():
            playbooks[(de, str(intent))] = Playbook(
                steps=str(entry["steps"]),
                requires=tuple(str(name) for name in entry.get("requires", ())),
            )
    return playbooks
