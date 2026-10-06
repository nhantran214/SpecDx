"""Copies of a canon pack, edited in place. The base pack is never written."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_NUMBERED = re.compile(r"^\d+_")
Rows = list[dict[str, str]]


def copy_pack(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def regenerate_sql(pack: Path) -> None:
    """Re-project data/*.csv to SQL with the pack's own generator.

    On an unedited copy the output is byte-identical to the shipped SQL.
    """
    subprocess.run([sys.executable, "generate_sql.py"], cwd=pack, check=True, capture_output=True)


def csv_path(pack: Path, table: str) -> Path:
    for path in sorted((pack / "data").glob("*.csv")):
        name = path.stem.split("_", 1)[1] if _NUMBERED.match(path.stem) else path.stem
        if name == table:
            return path
    raise KeyError(f"no CSV for table {table}")


def edit_csv_rows(pack: Path, table: str, fn: Callable[[Rows], Rows]) -> None:
    """Rewrite one CSV, keeping its byte-order mark (or lack of one) and line endings."""
    path = csv_path(pack, table)
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    rows = fn(rows)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)


def edit_json(pack: Path, rel: str, fn: Callable[[Any], Any]) -> None:
    path = pack / rel
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(fn(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replace_text(pack: Path, rel: str, old: str, new: str) -> bool:
    path = pack / rel
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def append_text(pack: Path, rel: str, text: str) -> None:
    with (pack / rel).open("a", encoding="utf-8") as handle:
        handle.write(text)
