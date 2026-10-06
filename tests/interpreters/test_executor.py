import json
from datetime import UTC, datetime

import httpx
import pytest
from tonomyx.contracts.clock import FixedClock
from tonomyx.contracts.envelope import ResultStatus
from tonomyx.de.runtime import build_runtime
from tonomyx.flow.case_0031 import open_case
from tonomyx.tools.catalog import build_registry

from specdx.interpreters.executor import SpecInterpreterExecutor
from specdx.interpreters.prompt import load_variants
from specdx.orchestrate.flow import run_case
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import canon_scenarios
from specdx.spec.tell import TellCorpus

pytestmark = pytest.mark.db
CLOCK = FixedClock(datetime(2026, 9, 1, 2, 0, tzinfo=UTC))


def _call(name, arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def _say(text):
    return {"role": "assistant", "content": text}


def _scripted(*replies):
    queue, seen = list(replies), []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        reply = queue.pop(0) if queue else _say("Xong.")
        return httpx.Response(200, json={"message": reply, "done": True})

    return httpx.MockTransport(handle), seen


@pytest.fixture(scope="module")
def corpus(pack) -> TellCorpus:
    return TellCorpus.build(pack, [])


@pytest.fixture
def run_de(db, corpus):
    def run(transport, de="DE-OP-01", intent="check_availability", inputs=None):
        executor = SpecInterpreterExecutor(
            "scripted", build_registry(), corpus, load_variants(), transport=transport, seed=7
        )
        with db.isolated() as conn:
            runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
            open_case(runtime.repo)
            envelope = runtime.run(de, intent, "CASE-0031", inputs or {"ma_so": "SO-0031"})
            rows = conn.execute("SELECT kind, ref FROM evt").fetchall()
        return envelope, rows
    return run


def test_the_token_tools_and_search_spec_are_offered(run_de) -> None:
    transport, seen = _scripted(_say("Xong."))
    run_de(transport)
    assert {t["function"]["name"] for t in seen[0]["tools"]} == {
        "read_records", "log_event", "search_spec"
    }


def test_sampling_options_and_a_prompt_without_playbooks(run_de) -> None:
    transport, seen = _scripted(_say("Xong."))
    run_de(transport)
    assert seen[0]["options"] == {"temperature": 0.7, "seed": 7}
    system = seen[0]["messages"][0]["content"]
    assert "search_spec" in system and "kiểm tra khả năng đáp ứng" in system
    assert "Hướng dẫn nghiệp vụ" not in system


def test_search_spec_answers_from_the_corpus_and_tools_write_for_real(run_de) -> None:
    transport, seen = _scripted(
        _call("search_spec", {"query": "chiết khấu tối đa"}),
        _call("log_event", {"case_id": "CASE-0031", "kind": "FLAGGED", "ref": "ITEM-0012",
                            "actor": "AI", "basis": "thiếu"}),
        _say("Đã gắn cờ."),
    )
    envelope, rows = run_de(transport)
    tool_reply = [m for m in seen[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "KB:KB-0003" in tool_reply
    assert envelope.status is ResultStatus.DONE
    assert {"kind": "FLAGGED", "ref": "ITEM-0012"} in rows


def test_a_number_the_canon_leaves_blank_is_refused(run_de) -> None:
    transport, seen = _scripted(
        _call("write_draft", {"table": "pay", "idempotency_key": "k",
                              "payload": {"ma_pay": "PAY-X", "ma_vinv": "VINV-0015",
                                          "so_tien": 0, "trang_thai": "Chờ duyệt"}}),
        _say("Xong."),
    )
    run_de(transport, de="DE-FN-03", intent="draft_payment", inputs={"ma_pay": "PAY-X"})
    reply = [m for m in seen[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "denied" in reply and "pay.so_tien" in reply


def test_a_timeout_ends_the_step_not_the_case(db, pack, corpus) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), corpus, load_variants(),
        transport=httpx.MockTransport(handle),
    )
    steps = load_work_orders()
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        case = run_case(runtime, canon_scenarios(pack)["grn12"], steps)
    assert len(case.statuses) == len(steps)
    assert case.statuses["DE-SM-02:draft_quote"] == "error"
    assert "ollama call failed" in case.claims["DE-SM-02:draft_quote"]
    assert case.statuses["P2:decide_quote"] == "error"
