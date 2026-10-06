"""Single-use task tokens and short-lived user tokens.

Claim names and lifetimes come from the canon pack's
targets/tonomyx/task-token.claims.json and user-token.claims.json.

The issuer mints tokens from data passed by the caller. The caller is responsible
for assembling the complete tool set before issuing a task token — note that some
capabilities like send_email are stored in their own columns and must be added to
the tools set by the caller before calling issue_task_token.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol

from tonomyx.contracts.clock import Clock

ISSUER = "tonomyx-token-issuer"
TASK_TOKEN_TTL = timedelta(seconds=300)
USER_TOKEN_TTL = timedelta(minutes=15)

# No real personal data in this system: every address is a demo address.
DEMO_EMAIL_SUFFIX = "@demo.example"


class TokenExpired(PermissionError):
    """The token's deadline has passed."""


class TokenReplayed(PermissionError):
    """A single-use token was presented twice."""


@dataclass(frozen=True)
class TaskToken:
    """Authority for one digital-employee task, valid once and briefly."""

    iss: str
    sub: str
    task_id: str
    case_id: str
    tools: frozenset[str]
    read: frozenset[str]
    draft: frozenset[str]
    kb_domain: str
    autonomy: str
    budget_usd: Decimal
    jti: str
    exp: datetime


@dataclass(frozen=True)
class UserToken:
    """Authority for a human persona, used when a person decides an approval."""

    iss: str
    sub: str
    persona: str
    groups: frozenset[str]
    kb_domains: frozenset[str]
    exp: datetime


class JtiStore(Protocol):
    """Records spent token identifiers so none can be used twice."""

    def consume(self, jti: str) -> None: ...


class InMemoryJtiStore:
    """Good enough for one process; swap in a shared store to scale out."""

    def __init__(self) -> None:
        self._spent: set[str] = set()

    def consume(self, jti: str) -> None:
        if jti in self._spent:
            raise TokenReplayed(f"task token {jti} has already been used")
        self._spent.add(jti)


class TokenIssuer:
    """Mints and checks tokens."""

    def __init__(self, clock: Clock, jti_store: JtiStore) -> None:
        self.clock = clock
        self.jti_store = jti_store

    def issue_task_token(
        self,
        *,
        de: str,
        task_id: str,
        case_id: str,
        tools: set[str] | frozenset[str],
        read: set[str] | frozenset[str],
        draft: set[str] | frozenset[str],
        kb_domain: str,
        autonomy: str,
        budget_usd: Decimal,
    ) -> TaskToken:
        return TaskToken(
            iss=ISSUER,
            sub=de,
            task_id=task_id,
            case_id=case_id,
            tools=frozenset(tools),
            read=frozenset(read),
            draft=frozenset(draft),
            kb_domain=kb_domain,
            autonomy=autonomy,
            budget_usd=budget_usd,
            jti=str(uuid.uuid4()),
            exp=self.clock.now() + TASK_TOKEN_TTL,
        )

    def issue_user_token(
        self, *, sub: str, persona: str, kb_domains: set[str] | frozenset[str]
    ) -> UserToken:
        if not sub.endswith(DEMO_EMAIL_SUFFIX):
            raise ValueError(f"user subject must end in {DEMO_EMAIL_SUFFIX}: {sub}")
        return UserToken(
            iss=ISSUER,
            sub=sub,
            persona=persona,
            groups=frozenset({persona}),
            kb_domains=frozenset(kb_domains),
            exp=self.clock.now() + USER_TOKEN_TTL,
        )

    def redeem_task_token(self, token: TaskToken) -> TaskToken:
        """Check the deadline, then spend the token. Expiry is checked first so an
        expired token reports expiry every time instead of turning into a replay."""
        if self.clock.now() > token.exp:
            raise TokenExpired(f"task token {token.jti} expired at {token.exp.isoformat()}")
        self.jti_store.consume(token.jti)
        return token

    def check_user_token(self, token: UserToken) -> UserToken:
        """User tokens are reusable within their window, unlike task tokens."""
        if self.clock.now() > token.exp:
            raise TokenExpired(f"user token for {token.persona} expired at {token.exp.isoformat()}")
        return token
