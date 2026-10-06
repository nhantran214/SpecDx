"""Least privilege per digital employee, read from the de_scope table.

A digital employee with no de_scope row has no capability at all. Six of the
fifteen in the canon are in exactly that position; that is the canon's statement,
not a gap to be filled in.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from tonomyx.db.repo import Repository

# de_scope stores list-valued cells as pipe-separated text.
_SEPARATOR = "|"

# The value of de_scope.send_email that grants the capability.
_SEND_AFTER_APPROVAL = "after_apr"


class NoScope(PermissionError):
    """The digital employee has no de_scope row and therefore no authority."""


def _split(cell: str | None) -> frozenset[str]:
    return frozenset(part for part in (cell or "").split(_SEPARATOR) if part)


@dataclass(frozen=True)
class DeScope:
    """One row of de_scope, parsed."""

    ma_de: str
    tools: frozenset[str]
    read_tables: frozenset[str]
    draft_tables: frozenset[str]
    create_apr: bool
    kb_domain: str
    autonomy: str
    budget_usd: Decimal
    send_email: str

    @property
    def may_send_email(self) -> bool:
        return self.send_email == _SEND_AFTER_APPROVAL

    @property
    def token_tools(self) -> frozenset[str]:
        """The tool set a task token carries.

        send_email lives in its own de_scope column rather than in the tools list,
        so it is folded in here. Every gate downstream can then ask one question:
        is this tool in the token?
        """
        if self.may_send_email:
            return self.tools | {"send_email"}
        return self.tools

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> DeScope:
        return cls(
            ma_de=str(row["ma_de"]),
            tools=_split(row["tools"]),
            read_tables=_split(row["read_tables"]),
            draft_tables=_split(row["draft_tables"]),
            create_apr=str(row["create_apr"]).strip() == "yes",
            kb_domain=str(row["kb_domain"] or ""),
            autonomy=str(row["autonomy"] or ""),
            budget_usd=Decimal(str(row["budget_usd"])),
            send_email=str(row["send_email"] or "no").strip(),
        )


class ScopeRegistry:
    """Scope lookup for every digital employee that has one."""

    def __init__(self, scopes: dict[str, DeScope]) -> None:
        self._scopes = scopes

    @classmethod
    def from_repo(cls, repo: Repository) -> ScopeRegistry:
        rows = repo.select("de_scope")
        return cls({str(r["ma_de"]): DeScope.from_row(r) for r in rows})

    def has(self, ma_de: str) -> bool:
        return ma_de in self._scopes

    def get(self, ma_de: str) -> DeScope:
        try:
            return self._scopes[ma_de]
        except KeyError:
            raise NoScope(
                f"{ma_de} has no de_scope row, so it holds no capability"
            ) from None

    def codes(self) -> frozenset[str]:
        return frozenset(self._scopes)
