"""Tell variants and the system prompt an interpreter reads."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tonomyx.canon.pack import CanonPack
from tonomyx.contracts.envelope import TaskEnvelope

VARIANTS_PATH = Path(__file__).with_name("variants.json")
PLACEHOLDERS: tuple[str, ...] = (
    "{role}", "{actor}", "{case_id}", "{statement}", "{inputs}", "{read}", "{draft}",
)
_NOTE = (
    "Tell variants: the standing instruction and each task statement. Variant 0 is "
    "hand-written and carries no business rule: every rule must come from the documents "
    "through search_spec. Variants 1-2 are paraphrases produced once by "
    "`specdx make-variants` and then frozen."
)


@dataclass(frozen=True)
class TellVariants:
    standing: tuple[str, ...]
    statements: dict[str, tuple[str, ...]]

    @property
    def count(self) -> int:
        return len(self.standing)


def load_variants(path: Path = VARIANTS_PATH) -> TellVariants:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TellVariants(
        standing=tuple(raw["standing"]),
        statements={k: tuple(v) for k, v in raw["statements"].items()},
    )


def save_variants(variants: TellVariants, path: Path = VARIANTS_PATH) -> None:
    payload = {
        "_note": _NOTE,
        "standing": list(variants.standing),
        "statements": {k: list(v) for k, v in variants.statements.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def role_of(pack: CanonPack, actor: str) -> str:
    roles = {r["ma_de"]: r["vai"] for r in pack.tables.get("de", [])}
    roles |= {r["ma_user"]: r["ho_ten"] for r in pack.tables.get("app_user", [])}
    return roles.get(actor, "không rõ vai")


def _describe(pack: CanonPack, tables: Iterable[str]) -> str:
    """Render tables with their columns: table(col1, col2, ...) joined by "; "."""
    described = []
    for table in sorted(tables):
        if table not in pack.columns:
            continue
        cols = pack.columns[table]
        if cols:
            col_names = ", ".join(col.name for col in cols)
            described.append(f"{table}({col_names})")
        else:
            described.append(table)
    return "; ".join(described) if described else "(không có)"


def system_prompt(
    variants: TellVariants,
    variant: int,
    *,
    pack: CanonPack,
    task: TaskEnvelope,
    read: Iterable[str],
    draft: Iterable[str],
    suffix: str = "",
) -> str:
    text = variants.standing[variant].format(
        role=role_of(pack, task.de),
        actor=task.de,
        case_id=task.case_id,
        statement=variants.statements[task.intent][variant],
        inputs=json.dumps(task.inputs, ensure_ascii=False),
        read=_describe(pack, read),
        draft=_describe(pack, draft),
    )
    return f"{text}\n{suffix}" if suffix else text
