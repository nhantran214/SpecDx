"""Evaluate the canon pack's declarative invariants.

This is a port of the pack's own validate.py. Invariants are DATA, so this
evaluator stays small and closed to change: adding an invariant means editing
invariants.json, never this file.

One deliberate difference from validate.py: an unrecognised condition raises
instead of quietly counting as a failure. A typo in invariants.json should be a
loud error, not a red test that looks like a real violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Row = dict[str, str]


@dataclass(frozen=True)
class InvariantResult:
    """Outcome of one invariant, with enough detail to debug a failure."""

    invariant_id: str
    description: str
    passed: bool
    detail: str


def is_blank(value: object) -> bool:
    """A canon cell is blank when it is null or holds only whitespace."""
    return value is None or str(value).strip() == ""


def _cell_ok(value: str, condition: dict[str, Any]) -> bool:
    if "eq" in condition:
        return value == str(condition["eq"])
    if "ne" in condition:
        return value != str(condition["ne"])
    if "in" in condition:
        return value in [str(x) for x in condition["in"]]
    if "contains" in condition:
        return str(condition["contains"]) in value
    if "is_blank" in condition:
        return is_blank(value) == bool(condition["is_blank"])
    raise ValueError(f"unsupported invariant condition: {sorted(condition)}")


def _matches(row: Row, where: dict[str, Any] | None) -> bool:
    return all(str(row.get(k, "")).strip() == str(v) for k, v in (where or {}).items())


def _cell(row: Row, column: str) -> str:
    return str(row.get(column, "")).strip()


def evaluate(invariant: dict[str, Any], tables: dict[str, list[Row]]) -> InvariantResult:
    """Check one invariant against a set of tables."""
    selected = [
        r for r in tables.get(invariant["table"], []) if _matches(r, invariant.get("where"))
    ]
    spec = invariant["assert"]

    def result(passed: bool, detail: str) -> InvariantResult:
        return InvariantResult(invariant["id"], invariant["desc"], passed, detail)

    if "count" in spec:
        expected_count = spec["count"]
        return result(
            len(selected) == expected_count,
            f"count={len(selected)}, expected {expected_count}",
        )

    if "values_of" in spec:
        got = sorted({_cell(r, spec["values_of"]) for r in selected})
        expected = sorted(set(spec["equals_set"]))
        return result(got == expected, f"got {got}, expected {expected}")

    if "all" in spec:
        condition = spec["all"]
        bad = [r for r in selected if not _cell_ok(_cell(r, condition["col"]), condition)]
        return result(not bad, f"{len(bad)} of {len(selected)} rows violate the condition")

    # Row-level assertion: the first matching row carries the answer.
    if not selected:
        return result(False, "no row matched the where clause")
    column = str(spec.get("col", ""))
    value = _cell(selected[0], column)
    return result(_cell_ok(value, spec), f"{column}={value!r}")


def evaluate_all(
    invariants: list[dict[str, Any]], tables: dict[str, list[Row]]
) -> list[InvariantResult]:
    """Check every invariant and return one result each, in declaration order."""
    return [evaluate(inv, tables) for inv in invariants]
