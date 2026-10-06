"""Run one task envelope from scope lookup to result envelope.

Two kinds of actor pass through here. A digital employee gets a single-use task
token built from its de_scope row, and nothing beyond it. A person gets a context
carrying a persona and the three tools human context needs: decide_approval and
update_status, which belong to a person alone, and log_event, which recording a
decision also requires and which digital employees hold too. No de_scope row in
the canon grants update_status or decide_approval to any digital employee, and
this module is where that stays true.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from decimal import Decimal
from typing import Any

import psycopg

from tonomyx.canon.pack import CanonPack
from tonomyx.config import Settings
from tonomyx.contracts.clock import Clock, SystemClock
from tonomyx.contracts.envelope import (
    ResultEnvelope,
    ResultStatus,
    TaskEnvelope,
    TaskScope,
    denied,
)
from tonomyx.contracts.token import InMemoryJtiStore, JtiStore, TaskToken, TokenIssuer, UserToken
from tonomyx.db.repo import Repository
from tonomyx.de.executor import DeExecutor
from tonomyx.de.scope import NoScope, ScopeRegistry
from tonomyx.policy.store import PolicyStore
from tonomyx.tools.catalog import build_registry
from tonomyx.tools.ports import IdempotencyStore, InMemoryIdempotencyStore, MailSender
from tonomyx.tools.registry import ToolContext, ToolRegistry

# The tools a human context carries. decide_approval and update_status belong to
# people alone; log_event is shared with digital employees, but deciding a gate
# also means recording that it happened, so the human context needs it too.
HUMAN_TOOLS: frozenset[str] = frozenset({"decide_approval", "update_status", "log_event"})

# Trust gate level a human session operates at.
_HUMAN_AUTONOMY = "L3"


class DeRuntime:
    """The seam between an intent and the work that carries it out."""

    def __init__(
        self,
        *,
        repo: Repository,
        pack: CanonPack,
        scopes: ScopeRegistry,
        issuer: TokenIssuer,
        tools: ToolRegistry,
        executor: DeExecutor,
        clock: Clock,
        idempotency: IdempotencyStore,
        mail: MailSender | None = None,
    ) -> None:
        self.repo = repo
        self.pack = pack
        self.scopes = scopes
        self.issuer = issuer
        self.tools = tools
        self.executor = executor
        self.clock = clock
        self.idempotency = idempotency
        self.mail = mail
        self.policy = PolicyStore(repo)

    def run(
        self,
        de: str,
        intent: str,
        case_id: str,
        inputs: dict[str, Any],
        deadline_seconds: int = 300,
    ) -> ResultEnvelope:
        """Carry out one task, or say plainly why it could not run."""
        task_id = f"T-{uuid.uuid4().hex[:12]}"
        task = TaskEnvelope(
            task_id=task_id,
            case_id=case_id,
            de=de,
            intent=intent,
            inputs=inputs,
            scope=TaskScope(),
            idempotency_key=f"{de}:{intent}:{case_id}",
            deadline=self.clock.now() + timedelta(seconds=deadline_seconds),
        )

        try:
            scope = self.scopes.get(de)
        except NoScope as exc:
            return denied(task, str(exc))

        task = task.model_copy(
            update={
                "scope": TaskScope(
                    tools=sorted(scope.token_tools),
                    read=sorted(scope.read_tables),
                    draft=sorted(scope.draft_tables),
                    kb_domain=scope.kb_domain,
                )
            }
        )

        token = self.issuer.issue_task_token(
            de=de,
            task_id=task_id,
            case_id=case_id,
            tools=scope.token_tools,
            read=scope.read_tables,
            draft=scope.draft_tables,
            kb_domain=scope.kb_domain,
            autonomy=scope.autonomy,
            budget_usd=scope.budget_usd,
        )
        token = self.issuer.redeem_task_token(token)

        ctx = self._context(token)
        try:
            # A savepoint, not a bare call: a tool that raises must roll back only
            # its own work, not abort the whole connection. Repository.conn.transaction()
            # opens a SAVEPOINT here because build_runtime's own ScopeRegistry.from_repo
            # query already started an outer transaction before the first task ever
            # runs -- at the true outermost level this call would commit instead, which
            # is why tests/de/test_runtime.py proves the savepoint behaviour directly
            # rather than assuming it.
            with ctx.repo.conn.transaction():
                return self.executor.run(task, ctx)
        except Exception as exc:  # noqa: BLE001 - a failure must surface, never pass as done
            return ResultEnvelope(
                task_id=task_id,
                case_id=case_id,
                status=ResultStatus.ERROR,
                next_hint=f"{type(exc).__name__}: {exc}",
            )

    def human_context(self, persona: str) -> ToolContext:
        """A context for a person deciding a gate."""
        user = self.issuer.issue_user_token(
            sub=f"{persona.lower()}@demo.example",
            persona=persona,
            kb_domains=frozenset(),
        )
        token = self.issuer.issue_task_token(
            de=persona,
            task_id=f"H-{uuid.uuid4().hex[:12]}",
            case_id="",
            tools=HUMAN_TOOLS,
            read=frozenset(self.pack.columns),
            draft=frozenset(),
            kb_domain="",
            autonomy=_HUMAN_AUTONOMY,
            budget_usd=Decimal("0"),
        )
        return self._context(token, user_token_override=user)

    def _context(
        self, token: TaskToken, user_token_override: UserToken | None = None
    ) -> ToolContext:
        return ToolContext(
            repo=self.repo,
            pack=self.pack,
            policy=self.policy,
            token=token,
            clock=self.clock,
            idempotency=self.idempotency,
            user_token=user_token_override,
            mail=self.mail,
        )


def build_runtime(
    conn: psycopg.Connection[dict[str, Any]],
    settings: Settings,
    *,
    executor: DeExecutor,
    clock: Clock | None = None,
    mail: MailSender | None = None,
    idempotency: IdempotencyStore | None = None,
    jti_store: JtiStore | None = None,
) -> DeRuntime:
    """Assemble a runtime over one database connection.

    A caller that outlives one connection -- the HTTP app -- passes its own
    idempotency and jti stores so a retried request is recognised across
    requests. Anything else gets fresh in-memory stores scoped to this runtime.
    """
    the_clock = clock or SystemClock()
    pack = CanonPack.load(settings)
    repo = Repository(conn, pack)
    return DeRuntime(
        repo=repo,
        pack=pack,
        scopes=ScopeRegistry.from_repo(repo),
        issuer=TokenIssuer(clock=the_clock, jti_store=jti_store or InMemoryJtiStore()),
        tools=build_registry(),
        executor=executor,
        clock=the_clock,
        idempotency=idempotency or InMemoryIdempotencyStore(),
        mail=mail,
    )
