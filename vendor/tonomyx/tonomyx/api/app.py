"""The HTTP surface: the approval queue, one case, and its audit trail.

There is no endpoint that creates or edits a business record directly. Everything
that changes state goes through a tool, so the gates apply to an HTTP caller exactly
as they apply to a digital employee.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from tonomyx.canon.pack import CanonPack
from tonomyx.config import Settings
from tonomyx.contracts.token import InMemoryJtiStore, JtiStore
from tonomyx.db.bootstrap import seed_operational, truncate_operational
from tonomyx.db.engine import connect
from tonomyx.db.relations import case_scoped_rows
from tonomyx.db.repo import Repository
from tonomyx.de.executor import ScriptedExecutor
from tonomyx.de.runtime import DeRuntime, build_runtime
from tonomyx.flow.handlers import register_handlers
from tonomyx.flow.scenarios import check, expectations_for, run_scenario
from tonomyx.tools.impl.decide_approval import decide_approval
from tonomyx.tools.ports import IdempotencyStore, InMemoryIdempotencyStore
from tonomyx.tools.registry import ToolDenied

# Tables shown alongside a case.
CASE_TABLES = ("so", "soline", "po", "grn", "insp", "ncr", "ship", "inv", "vinv", "pay", "apr")


class DecideRequest(BaseModel):
    decision: str
    # A blank persona could otherwise reach the authority check and come back as a
    # 403 there, alongside the missing-field case's 422 from pydantic. Both are one
    # malformed request; requiring a non-empty persona here gives them one status.
    persona: str = Field(min_length=1)
    reason: str = ""
    idempotency_key: str


@dataclass
class RequestState:
    """What one request needs from its database dependency.

    A runtime alone is not enough for /scenarios/{name}/run: that route must
    truncate operational tables on the same connection before it runs, so the
    connection itself is carried alongside the runtime built over it.
    """

    conn: psycopg.Connection[dict[str, Any]]
    runtime: DeRuntime


@dataclass
class SharedStores:
    """Stores that must outlive a single request.

    A client that retries a decision after a dropped response has to be recognised
    on the second request, so the idempotency store lives as long as the app, not
    the request. Its lifetime is also tied to the data it describes: a scenario run
    wipes and rebuilds the operational tables, so it resets the store too, or a key
    recorded before the wipe would replay a decision that no longer exists.
    """

    idempotency: IdempotencyStore = field(default_factory=InMemoryIdempotencyStore)
    jti: JtiStore = field(default_factory=InMemoryJtiStore)

    def forget_decisions(self) -> None:
        self.idempotency = InMemoryIdempotencyStore()


def _scripted_executor() -> ScriptedExecutor:
    executor = ScriptedExecutor()
    register_handlers(executor)
    return executor


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application over one settings object."""
    resolved = settings or Settings.from_env()
    pack = CanonPack.load(resolved)
    app = FastAPI(title="TonomyX — Hệ Doanh nghiệp Tự hành", version=pack.meta["canon_version"])

    stores = SharedStores()

    def request_state() -> Iterator[RequestState]:
        with connect(resolved) as conn:
            runtime = build_runtime(
                conn,
                resolved,
                executor=_scripted_executor(),
                idempotency=stores.idempotency,
                jti_store=stores.jti,
            )
            yield RequestState(conn=conn, runtime=runtime)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "canon_version": str(pack.meta["canon_version"])}

    @app.get("/cases/{case_id}")
    def read_case(case_id: str, state: RequestState = Depends(request_state)) -> dict[str, Any]:
        repo: Repository = state.runtime.repo
        case = repo.select_one("cases", where={"case_id": case_id})
        if case is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id}")
        records: dict[str, list[dict[str, Any]]] = {
            table: case_scoped_rows(repo, table, case_id) for table in CASE_TABLES
        }
        return {"case": case, "records": records}

    @app.get("/cases/{case_id}/events")
    def read_events(
        case_id: str, state: RequestState = Depends(request_state)
    ) -> dict[str, Any]:
        if state.runtime.repo.select_one("cases", where={"case_id": case_id}) is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id}")
        events = sorted(
            state.runtime.repo.select("evt", where={"case_id": case_id}),
            key=lambda r: int(r["id"]),
        )
        return {"events": events}

    @app.get("/approvals")
    def read_approvals(
        trang_thai: str | None = None, state: RequestState = Depends(request_state)
    ) -> dict[str, Any]:
        where = {"trang_thai": trang_thai} if trang_thai else None
        return {"approvals": state.runtime.repo.select("apr", where=where)}

    @app.post("/approvals/{ma_apr}/decide")
    def decide(
        ma_apr: str, request: DecideRequest, state: RequestState = Depends(request_state)
    ) -> dict[str, Any]:
        if state.runtime.repo.select_one("apr", where={"ma_apr": ma_apr}) is None:
            raise HTTPException(status_code=404, detail=f"no approval {ma_apr}")
        ctx = state.runtime.human_context(request.persona)
        try:
            result = decide_approval(
                ctx,
                apr=ma_apr,
                decision=request.decision,
                reason=request.reason,
                idempotency_key=request.idempotency_key,
            )
        except ToolDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return {"approval": result["row"]}

    @app.post("/scenarios/{name}/run")
    def run(
        name: str, illustrative: bool = False, state: RequestState = Depends(request_state)
    ) -> dict[str, Any]:
        if name not in pack.scenarios:
            raise HTTPException(status_code=404, detail=f"no scenario {name}")
        # A scenario run starts from master data only; the request's connection
        # may still hold the seeded case from an earlier request or test fixture.
        truncate_operational(state.conn)
        # The wipe invalidates every key recorded against the old rows, and the flow
        # itself records keys such as draft:SO-0031 that a second run must not replay
        # against an empty table. Start this run, and every request after it, fresh.
        stores.forget_decisions()
        runtime = build_runtime(
            state.conn,
            resolved,
            executor=_scripted_executor(),
            idempotency=stores.idempotency,
            jti_store=stores.jti,
        )
        try:
            outcome = run_scenario(runtime, name, use_illustrative=illustrative)
        except Exception:
            # A run that does not finish must not leave the database with master
            # data only: restore the canon operational state before propagating.
            seed_operational(state.conn, resolved)
            raise
        problems = check(outcome, expectations_for(pack, name))
        return {
            "outcome": {
                "apr_0102": outcome.apr_0102,
                "three_way": outcome.three_way,
                "pay_units": outcome.pay_units,
                "open_units": outcome.open_units,
            },
            "problems": problems,
            "illustrative": illustrative,
        }

    return app
