"""Command line for the demo: set up, check, run, inspect, serve."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import psycopg

from tonomyx.canon.invariants import evaluate_all
from tonomyx.canon.pack import CanonPack
from tonomyx.config import Settings
from tonomyx.contracts.envelope import ResultStatus
from tonomyx.db.bootstrap import (
    bootstrap,
    drop_everything,
    seed_operational,
    truncate_operational,
)
from tonomyx.db.engine import connect
from tonomyx.db.repo import Repository, tables_as_text
from tonomyx.de.executor import DeExecutor, OllamaExecutor, ScriptedExecutor
from tonomyx.de.runtime import build_runtime
from tonomyx.flow.case_0031 import CASE_ID, open_case
from tonomyx.flow.handlers import register_handlers
from tonomyx.flow.playbooks import load_playbooks
from tonomyx.flow.scenarios import check, expectations_for, run_scenario
from tonomyx.tools.catalog import build_registry

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


DEFAULT_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MAX_STEPS = 12

_THINK = {"auto": None, "on": True, "off": False}


def _llm_executor(
    model: str, *, base_url: str, max_steps: int, think: bool | None
) -> DeExecutor:
    """The executor run-task uses. A seam so tests can stand in for the model."""
    return OllamaExecutor(
        model,
        build_registry(),
        base_url=base_url,
        max_steps=max_steps,
        think=think,
        playbooks=load_playbooks(),
    )


def _parse_inputs(pairs: list[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--input expects KEY=VALUE, got {pair!r}")
        inputs[key] = value
    return inputs


def _run_task(
    args: argparse.Namespace, conn: psycopg.Connection[dict[str, Any]], settings: Settings
) -> int:
    """Run one digital employee and print its envelope as JSON on stdout."""
    executor = _llm_executor(
        args.model, base_url=args.base_url, max_steps=args.max_steps, think=_THINK[args.think]
    )
    runtime = build_runtime(conn, settings, executor=executor)
    if args.from_master:
        truncate_operational(conn)
        open_case(runtime.repo)
    try:
        result = runtime.run(args.de, args.intent, args.case, _parse_inputs(args.input))
    except Exception:
        if args.from_master:
            seed_operational(conn, settings)
        raise
    print(result.model_dump_json(indent=2))
    calls = result.usage.get("tool_calls", [])
    print(
        f"  {args.de} {args.intent}: {result.status.value} · "
        f"{len(calls)} lời gọi tool · {len(result.drafts)} nháp · "
        f"{len(result.approvals)} cổng · {len(result.events)} sự kiện",
        file=sys.stderr,
    )
    return 0 if result.status in (ResultStatus.DONE, ResultStatus.NEEDS_GATE) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tonomyx")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "bootstrap",
        help="reset the database and load the canon pack (destructive: this is the demo reset)",
    )
    sub.add_parser("validate", help="check every canon invariant against the database")
    run = sub.add_parser("run-scenario", help="run one scenario from master data")
    run.add_argument("name")
    run.add_argument(
        "--illustrative",
        action="store_true",
        help="compute an amount from non-canon illustrative prices (default: off)",
    )
    task = sub.add_parser(
        "run-task", help="run one digital employee on a local model, through the gated tools"
    )
    task.add_argument("de", help="digital employee code, e.g. DE-OP-01")
    task.add_argument("intent", help="what to do, e.g. check_availability")
    task.add_argument("--case", default=CASE_ID, help=f"case id (default: {CASE_ID})")
    task.add_argument(
        "--input", action="append", default=[], metavar="KEY=VALUE", help="task input; repeatable"
    )
    task.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})"
    )
    task.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    task.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    task.add_argument(
        "--think",
        choices=("auto", "on", "off"),
        default="auto",
        help="reasoning mode for models that support it (default: model's own)",
    )
    task.add_argument(
        "--from-master",
        action="store_true",
        help="clear operational tables and open the case first (destructive, like run-scenario)",
    )
    sub.add_parser("queue", help="list approvals waiting for a decision")
    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_env()
    pack = CanonPack.load(settings)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("tonomyx.api.app:create_app", host=args.host, port=args.port, factory=True)
        return 0

    with connect(settings) as conn:
        if args.command == "bootstrap":
            drop_everything(conn)
            bootstrap(conn, settings)
            print(f"Đã nạp Canon Pack {pack.meta['canon_version']} vào PostgreSQL.")
            return 0

        repo = Repository(conn, pack)

        if args.command == "validate":
            results = evaluate_all(pack.invariants, tables_as_text(repo))
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                detail = "" if result.passed else f"   ({result.detail})"
                print(f"  [{status}] {result.invariant_id:<22} {result.description}{detail}")
            passed = sum(1 for r in results if r.passed)
            print(f"KẾT QUẢ: {passed}/{len(results)} bất biến XANH")
            return 0 if passed == len(results) else 1

        if args.command == "queue":
            for row in repo.select("apr", where={"trang_thai": "Chờ duyệt"}):
                print(f"  {row['ma_apr']:<10} {row['loai']:<40} hồ sơ {row['hoso_ref']}")
            return 0

        if args.command == "run-task":
            return _run_task(args, conn, settings)

        if args.command == "run-scenario":
            if args.name not in pack.scenarios:
                print(f"Không có kịch bản {args.name!r}.", file=sys.stderr)
                return 2
            truncate_operational(conn)
            executor = ScriptedExecutor()
            register_handlers(executor)
            runtime = build_runtime(conn, settings, executor=executor)
            try:
                outcome = run_scenario(runtime, args.name, use_illustrative=args.illustrative)
            except Exception:
                # A run that does not finish must not leave the database with
                # master data only: restore the canon operational state before
                # letting the failure propagate.
                seed_operational(conn, settings)
                raise
            problems = check(outcome, expectations_for(pack, args.name))
            if args.illustrative:
                print("  (minh họa: số tiền hiển thị KHÔNG phải canon)")
            print(f"  apr_0102   = {outcome.apr_0102}")
            print(f"  three_way  = {outcome.three_way}")
            print(f"  pay_units  = {outcome.pay_units}")
            print(f"  open_units = {outcome.open_units}")
            for problem in problems:
                print(f"  LỆCH: {problem}", file=sys.stderr)
            return 0 if not problems else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
