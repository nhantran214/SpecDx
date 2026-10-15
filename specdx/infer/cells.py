"""Anchored cells (spec v3 §2): the unit of evidence.

A cell is one (table, row, column) value, one golden event, or one cross-cutting clause,
read at the items whose step may write it. This module holds the syntactic normalisation
of readings (§2.5) and the matching of rows by natural key (§2.6).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from tonomyx.canon.pack import CanonPack

from specdx.db import golden_tables, is_background
from specdx.spec.columns import ColumnClass, ColumnModel
from specdx.spec.tell import TellCorpus

ABSENT = "absent"
_NUMERAL = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
_TRUE = frozenset({"t", "true", "1"})
_FALSE = frozenset({"f", "false", "0"})


def normalise(value: str, column_type: str = "text") -> str:
    """§2.5, in order: Unicode NFC, trim, inner whitespace collapsed, casefold, numerals
    canonicalised (35, 35.0 and 35,0 are 35). No semantic mapping: draft and Nháp stay
    distinct. A bool column's t/f (the golden CSVs) and true/false (end states read through
    the typed database) are one value, as C's own comparison already treats them."""
    text = " ".join(unicodedata.normalize("NFC", value).split()).casefold()
    if column_type == "bool":
        if text in _TRUE:
            return "true"
        if text in _FALSE:
            return "false"
    if column_type in ("int", "numeric") and _NUMERAL.match(text):
        number = Decimal(text.replace(",", "."))
        return "0" if number == 0 else format(number.normalize(), "f")
    return text


def tell_texts(corpus: TellCorpus) -> list[str]:
    """Every sentence of T, normalised once for token search."""
    return [normalise(corpus.sentence(s)) for s in corpus.sentence_ids()]


def occurs(value: str, texts: Iterable[str]) -> bool:
    """Whether the normalised value occurs in a normalised text as a whole token: neither
    preceded nor followed by a word character or a hyphen, so SOL-1 is not found in SOL-10
    and PO-0015 is not found in PO-0015-01."""
    needle = normalise(value)
    if not needle:
        return False
    # The boundary class [\w-] excludes only the ASCII hyphen, on purpose.
    pattern = re.compile(rf"(?<![\w-]){re.escape(needle)}(?![\w-])")
    return any(pattern.search(text) for text in texts)


def _case_rows(pack: CanonPack, table: str) -> list[dict[str, str]]:
    return [r for r in golden_tables(pack).get(table, []) if not is_background(table, r)]


def executor_generated(
    pack: CanonPack, columns: ColumnModel, corpus: TellCorpus
) -> frozenset[str]:
    """§2.6: the tables whose canon key T never states. A key is executor-generated when
    none of its golden values occurs anywhere in T; the test reads T only. Events have no
    key of their own (they are identified by kind, ref and actor)."""
    texts = tell_texts(corpus)
    out: set[str] = set()
    for table in sorted(golden_tables(pack)):
        keys = columns.keys.get(table, ())
        rows = _case_rows(pack, table)
        if table == "evt" or not keys or not rows:
            continue
        # The composite-key join is latent: only evt has a composite key and it is excluded.
        if not any(occurs(":".join(r[k] for k in keys), texts) for r in rows):
            out.add(table)
    return frozenset(out)


def match_columns(
    table: str, pack: CanonPack, columns: ColumnModel, generated: frozenset[str]
) -> tuple[str, ...]:
    """The columns that identify a row: its key, or for an executor-generated key the
    remaining LOCKED foreign-key columns (soline rows match on so and ma_sp)."""
    keys = columns.keys[table]
    if table not in generated:
        return keys
    return tuple(c.name for c in pack.columns[table]
                 if c.fk and c.name not in keys
                 and columns.classes[(table, c.name)] is ColumnClass.LOCKED)


def identity(row: Mapping[str, str], match: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalise(str(row.get(c, ""))) for c in match)


@dataclass(frozen=True)
class TableRows:
    """How one table's rows are named. golden: a golden row's identity → its canon key
    (SOLINE-0031-1). A row matching no golden row is named by its normalised identity."""

    table: str
    keys: tuple[str, ...]
    match: tuple[str, ...]
    golden: dict[tuple[str, ...], str]

    def name(self, row: Mapping[str, str]) -> tuple[str, bool]:
        """The row's name and whether it matches a golden row."""
        ident = identity(row, self.match)
        if ident in self.golden:
            return self.golden[ident], True
        return "/".join(ident), False


def table_rows(
    pack: CanonPack, columns: ColumnModel, corpus: TellCorpus, tables: Iterable[str]
) -> dict[str, TableRows]:
    generated = executor_generated(pack, columns, corpus)
    out: dict[str, TableRows] = {}
    for table in tables:
        keys = columns.keys[table]
        match = match_columns(table, pack, columns, generated)
        golden = {identity(r, match): ":".join(r[k] for k in keys)
                  for r in _case_rows(pack, table)}
        out[table] = TableRows(table, keys, match, golden)
    return out
