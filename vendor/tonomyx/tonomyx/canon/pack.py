"""Read-only access to the canon pack.

Nothing in this module ever writes to the pack. The pack is the ground of truth;
a projection of it belongs outside its directory.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from tonomyx.config import Settings

# Data files are named "06_soline.csv" and the table is "soline". A file without
# a numeric prefix (de_scope.csv) keeps its stem.
_NUMBERED_PREFIX = re.compile(r"^\d+_")

# The canon CSVs are written with a byte-order mark.
_CSV_ENCODING = "utf-8-sig"


@dataclass(frozen=True)
class Column:
    """One column as schema.json declares it."""

    name: str
    type: str
    pk: bool = False
    nullable: bool = False
    # "table.column" this column references, or None when it names none. Read
    # from schema.json's own "fk" key -- adding a table's foreign keys there is
    # enough for anything (such as tonomyx/db/relations.py) that walks them.
    fk: str | None = None


@dataclass(frozen=True)
class KbDoc:
    """One knowledge-base article with its front matter."""

    kb_id: str
    version: str
    domain: str
    title: str
    effective_date: str
    body: str


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a '---' delimited front matter block from the body."""
    if not text.startswith("---"):
        raise ValueError("knowledge base document has no front matter block")
    _, raw, body = text.split("---", 2)
    front: dict[str, str] = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            front[key.strip()] = value.strip().strip('"')
    return front, body.strip()


class CanonPack:
    """Every fact the system is allowed to treat as ground truth."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def load(cls, settings: Settings) -> CanonPack:
        return cls(settings.canon_pack)

    def _json(self, *parts: str) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(
            self.root.joinpath(*parts).read_text(encoding="utf-8")
        )
        return payload

    @cached_property
    def meta(self) -> dict[str, Any]:
        return self._json("canon.meta.json")

    @cached_property
    def schema(self) -> dict[str, Any]:
        return self._json("schema.json")

    @cached_property
    def columns(self) -> dict[str, list[Column]]:
        return {
            table: [
                Column(
                    name=str(col["name"]),
                    type=str(col["type"]),
                    pk=bool(col.get("pk", False)),
                    nullable=bool(col.get("nullable", False)),
                    fk=str(col["fk"]) if col.get("fk") else None,
                )
                for col in cols
            ]
            for table, cols in self.schema["tables"].items()
        }

    @cached_property
    def append_only(self) -> frozenset[str]:
        return frozenset(self.schema["append_only"])

    @cached_property
    def classification(self) -> dict[str, str]:
        return dict(self.schema["classification"])

    @cached_property
    def invariants(self) -> list[dict[str, Any]]:
        return list(self._json("invariants.json")["invariants"])

    @cached_property
    def event_kinds(self) -> frozenset[str]:
        return frozenset(self._json("data", "event_kinds.json")["kinds"])

    @cached_property
    def scenarios(self) -> dict[str, dict[str, Any]]:
        return {s["name"]: s for s in self._json("data", "scenarios.json")["scenarios"]}

    @cached_property
    def tables(self) -> dict[str, list[dict[str, str]]]:
        """Every data CSV, keyed by table name, values kept as raw text."""
        out: dict[str, list[dict[str, str]]] = {}
        for path in sorted((self.root / "data").glob("*.csv")):
            stem = path.stem
            name = stem.split("_", 1)[1] if _NUMBERED_PREFIX.match(stem) else stem
            with path.open(encoding=_CSV_ENCODING, newline="") as handle:
                out[name] = [dict(row) for row in csv.DictReader(handle)]
        return out

    @cached_property
    def non_canon_columns(self) -> dict[str, frozenset[str]]:
        """Columns the canon deliberately leaves blank, keyed by table.

        The source books establish no price, amount, discount, supplier score,
        offer or money threshold, so the canon leaves those cells empty. The meta
        file lists them; a note after the column name (as on policy.gia_tri) is
        dropped, since the column is what a guard needs.
        """
        listed: dict[str, list[str]] = self.meta["cot_de_trong_khong_canon"]
        return {
            table: frozenset(entry.split()[0] for entry in entries)
            for table, entries in listed.items()
        }

    @cached_property
    def tool_catalog(self) -> dict[str, dict[str, str]]:
        """The nine tools the stack projection declares, keyed by name.

        Each entry carries the catalog's own input, effect and gating text, which is
        what a language model is shown when it is offered the tool.
        """
        entries = self._json("targets", "tonomyx", "tool-catalog.json")["tools"]
        return {
            str(e["name"]): {k: str(e[k]) for k in ("input", "effect", "gating")}
            for e in entries
        }

    @cached_property
    def illustrative_prices(self) -> dict[str, int]:
        """Unit prices that are explicitly NOT canon.

        The source books establish no money at all. These exist only so the demo
        can compute an amount; never present them as a business fact.
        """
        path = self.root / "data" / "illustrative" / "prices.csv"
        with path.open(encoding=_CSV_ENCODING, newline="") as handle:
            return {row["ma_sp"]: int(row["don_gia"]) for row in csv.DictReader(handle)}

    @cached_property
    def kb(self) -> dict[str, KbDoc]:
        """Knowledge base articles that have a markdown file.

        The kb table lists KB-0008 with no file behind it. That article is simply
        absent here; callers must handle a miss rather than assume coverage.
        """
        docs: dict[str, KbDoc] = {}
        for path in sorted((self.root / "kb").glob("*.md")):
            front, body = _split_front_matter(path.read_text(encoding="utf-8"))
            docs[front["kb_id"]] = KbDoc(
                kb_id=front["kb_id"],
                version=front.get("version", ""),
                domain=front.get("domain", ""),
                title=front.get("title", ""),
                effective_date=front.get("effective_date", ""),
                body=body,
            )
        return docs
