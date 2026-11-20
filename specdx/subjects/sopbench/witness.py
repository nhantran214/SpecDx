"""S3 execution witness: replay a SOPBench task on SOPBench's own strict system.

The strict system is the code form of the task's constraints. If its outcome differs from the
task label, the label contradicts its own executable constraints (a specification defect);
if they agree, the agents' shared deviation is a shared agent error. Task data is replayed as
is, never repaired: the bank credit_cards list-vs-dict defect is what the witness must show.
Outcomes do not depend on hash() (ruling G4), so PYTHONHASHSEED is not enforced.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from specdx.subjects.sopbench import SOPBENCH_DIR


@dataclass(frozen=True)
class WitnessResult:
    outcome: str  # "succeeds" | "refused" | "error"
    calls: tuple[str, ...]
    error: str = ""


def load_raw_tasks(domain: str, root: Path = SOPBENCH_DIR) -> list[dict]:
    """Raw task dicts in load_tasks order, with user_goal set from the group key."""
    raw = json.loads((root / "data" / f"{domain}_tasks.json").read_text())
    tasks: list[dict] = []
    for goal, group in raw.items():
        for t in group:
            t["user_goal"] = goal
            tasks.append(t)
    return tasks


def _ok(r: object) -> bool:
    if isinstance(r, bool):
        return r
    if isinstance(r, (tuple, list)):
        return bool(r[0])
    return bool(r)


def _children(graph: dict) -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for a, b in graph["connections"]:
        kids.setdefault(a, []).append(b)
    return kids


def _replay(
    strict: object, graph: dict, user_known: dict, links: set[str], target: str
) -> WitnessResult:
    nodes = graph["nodes"]
    kids = _children(graph)
    calls: list[str] = []

    def call(name: str, mapping: dict) -> bool:
        if any(k not in user_known for k in mapping.values()):
            return False
        kwargs = {p: user_known[k] for p, k in mapping.items()}
        calls.append(name)
        return _ok(getattr(strict, name)(**kwargs))

    def all_of(children: list[int]) -> bool:
        results = [sat(c) for c in children]
        if all(results):
            return True
        return all([sat(c) for c in children])  # one more pass

    def sat(i: int) -> bool:
        node = nodes[i]
        children = kids.get(i, [])
        if node == "and":
            return all_of(children)
        if node == "or":
            return any(sat(c) for c in children)
        name, mapping = node[0], node[1]
        if name not in links:
            return True
        if not all_of(children):
            return False
        return call(name, mapping)

    try:
        if not all_of(kids.get(0, [])):
            return WitnessResult("refused", tuple(calls))
        ok = call(nodes[0][0], nodes[0][1])
    except Exception as exc:  # noqa: BLE001 - any strict failure is an "error" outcome
        return WitnessResult("error", tuple(calls), repr(exc))
    return WitnessResult("succeeds" if ok else "refused", tuple(calls))


def _env():
    root = str(SOPBENCH_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from env.task import get_default_dep_full
    from env.variables import domain_assistant_keys, domain_keys

    return get_default_dep_full, domain_assistant_keys, domain_keys


def witness(domain: str, task: dict) -> WitnessResult:
    get_default_dep_full, assistant_keys, keys = _env()
    try:
        keyset = assistant_keys[domain]
        dep_full = get_default_dep_full(domain, "full")
        dep_full[task["user_goal"]] = task["constraints"]
        strict = keys[domain + "_strict"](
            copy.deepcopy(task["initial_database"]),
            keyset.action_innate_dependencies,
            dep_full,
            task["constraint_parameters"],
        )
        links = {v[0] for v in keyset.constraint_links.values()}
    except Exception as exc:  # noqa: BLE001
        return WitnessResult("error", (), repr(exc))
    return _replay(
        strict, task["directed_action_graph"], task["user_known"], links, task["user_goal"]
    )
