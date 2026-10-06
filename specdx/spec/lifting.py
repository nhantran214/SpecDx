"""Clause lifting (spec §2.5 and §11, item 1).

Lifting is deliberately conservative: only integer cells whose value equals exactly one
non-structural anchor parameter are rewritten. Everything else stays tied to the scenario
it was written for.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tonomyx.canon.pack import CanonPack

from specdx.types import PARAMETERS, STRUCTURAL, Scenario

_INTEGER_TYPES = frozenset({"int", "bigint"})
_PLACEHOLDER = re.compile(r"^\{([a-z_]+)\}$")


def column_type(pack: CanonPack, table: str, column: str) -> str:
    return next(c.type for c in pack.columns[table] if c.name == column)


def lift_value(value: str, column_type: str, anchor: Scenario) -> str | None:
    if column_type not in _INTEGER_TYPES:
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    hits = [p for p, v in anchor.params().items() if p not in STRUCTURAL and v == number]
    return "{" + hits[0] + "}" if len(hits) == 1 else None


def lift_invariant(
    inv: Mapping[str, Any], anchor: Scenario, pack: CanonPack
) -> tuple[dict[str, Any], bool]:
    """A row-level eq invariant on an integer column, lifted if its literal binds."""
    spec = inv["assert"]
    if "where" in inv and "col" in spec and "eq" in spec:
        placeholder = lift_value(
            str(spec["eq"]), column_type(pack, inv["table"], spec["col"]), anchor
        )
        if placeholder is not None:
            lifted = copy.deepcopy(dict(inv))
            lifted["assert"]["eq"] = placeholder
            return lifted, True
    return copy.deepcopy(dict(inv)), False


def instantiate(obj: Any, scenario: Scenario) -> Any:
    """Replace every whole-string placeholder with the scenario's value, as text."""
    if isinstance(obj, str):
        match = _PLACEHOLDER.match(obj)
        return str(scenario.params()[match.group(1)]) if match else obj
    if isinstance(obj, dict):
        return {k: instantiate(v, scenario) for k, v in obj.items()}
    if isinstance(obj, list):
        return [instantiate(v, scenario) for v in obj]
    return obj


@dataclass(frozen=True)
class ExpectationRule:
    id: str
    key: str
    scope: str  # "lifted" | "anchored" | "all"
    anchor: str | None
    eq: str | None = None
    members: tuple[str, ...] = ()


def lift_expectations(
    canon: Mapping[str, Scenario], expects: Mapping[str, Mapping[str, object]]
) -> list[ExpectationRule]:
    """Numeric keys equal to one parameter in every scenario lift; categorical keys
    become a membership rule for all scenarios beside their exact per-scenario rules."""
    rules: list[ExpectationRule] = []
    keys = sorted({k for e in expects.values() for k in e})
    for key in keys:
        values = {name: expects[name][key] for name in sorted(canon) if key in expects[name]}
        exact = [
            ExpectationRule(f"expect:{key}:{name}", key, "anchored", name, eq=str(v))
            for name, v in values.items()
        ]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values.values()):
            params = [
                p for p in PARAMETERS
                if p not in STRUCTURAL
                and all(canon[name].params()[p] == v for name, v in values.items())
            ]
            if len(params) == 1:
                rules.append(
                    ExpectationRule(f"expect:{key}", key, "lifted", None, eq="{" + params[0] + "}")
                )
            else:
                rules += exact
            continue
        rules += exact
        distinct = tuple(sorted({str(v) for v in values.values()}))
        if len(distinct) > 1:
            rules.append(ExpectationRule(f"expect:{key}:in", key, "all", None, members=distinct))
        else:
            rules.append(ExpectationRule(f"expect:{key}:all", key, "all", None, eq=distinct[0]))
    return rules
