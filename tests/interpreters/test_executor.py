import json
from datetime import UTC, datetime

import httpx
import pytest
from tonomyx.contracts.clock import FixedClock
from tonomyx.contracts.envelope import ResultStatus
from tonomyx.de.executor import _Record
from tonomyx.de.runtime import build_runtime
from tonomyx.flow.case_0031 import open_case
from tonomyx.tools.catalog import build_registry

from specdx.interpreters.executor import SpecInterpreterExecutor
from specdx.interpreters.harness import NUDGE
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
    def run(transport, de="DE-OP-01", intent="check_availability", inputs=None, rows=(),
            **options):
        executor = SpecInterpreterExecutor(
            "scripted", build_registry(), corpus, load_variants(), transport=transport, seed=7,
            **options,
        )
        with db.isolated() as conn:
            runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
            open_case(runtime.repo)
            for table, row in rows:  # what the earlier steps of the case would have written
                runtime.repo.insert(table, row)
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
    assert seen[0]["options"] == {"temperature": 0.7, "seed": 7, "num_ctx": 16384}
    system = seen[0]["messages"][0]["content"]
    assert "search_spec" in system and "kiểm tra khả năng đáp ứng" in system
    assert "Hướng dẫn nghiệp vụ" not in system
    # DE-OP-01's read scope includes soline; verify columns are shown
    assert "soline(ma_soline, so, ma_sp, sl, don_gia, chiet_khau, du_hang)" in system


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


def test_a_read_timeout_is_data_and_is_not_retried(db, corpus) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Xong."}})

    sleeps: list[float] = []
    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), corpus, load_variants(),
        transport=httpx.MockTransport(handle), sleep=sleeps.append,
    )
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        open_case(runtime.repo)
        slow = runtime.run("DE-OP-01", "check_availability", "CASE-0031", {"ma_so": "SO-0031"})
        assert executor.infra_failure is not None  # so far every call timed out
        later = runtime.run("DE-OP-01", "check_availability", "CASE-0031", {"ma_so": "SO-0031"})
    assert "ollama call failed: ReadTimeout" in (slow.next_hint or "")
    assert "ollama call failed" not in (later.next_hint or "")  # the model answered
    assert len(calls) == 2 and sleeps == [] and executor.infra_failure is None


def test_an_exhausted_server_error_flags_the_executor_and_stops_calling(db, corpus) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={"error": "CUDA error: out of memory"})

    sleeps: list[float] = []
    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), corpus, load_variants(),
        transport=httpx.MockTransport(handle), sleep=sleeps.append,
    )
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        open_case(runtime.repo)
        first = runtime.run("DE-OP-01", "check_availability", "CASE-0031", {"ma_so": "SO-0031"})
        second = runtime.run("DE-OP-01", "check_availability", "CASE-0031", {"ma_so": "SO-0031"})
    assert first.status is ResultStatus.ERROR and second.status is ResultStatus.ERROR
    assert len(calls) == 5 and sleeps == [2.0, 4.0, 8.0, 16.0]
    assert executor.infra_failure is not None and "500" in executor.infra_failure


def test_a_malformed_tool_call_ollama_reports_as_500_is_behaviour(db, corpus) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={
            "error": "error parsing tool call: raw='{\"name\": \"log_ev', "
                     "err=unexpected end of JSON input"})

    sleeps: list[float] = []
    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), corpus, load_variants(),
        transport=httpx.MockTransport(handle), sleep=sleeps.append,
    )
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        open_case(runtime.repo)
        envelope = runtime.run("DE-OP-01", "check_availability", "CASE-0031", {"ma_so": "SO-0031"})
    assert envelope.status is ResultStatus.ERROR and "error parsing tool call" in (
        envelope.next_hint or "")
    assert len(calls) == 1 and sleeps == [] and executor.infra_failure is None


def test_search_reply_includes_all_passages_when_they_fit(pack) -> None:
    """A corpus of 5 chunks each ~1500 chars → the reply parses as JSON and holds all 5 passages."""
    from specdx.interpreters.executor import SEARCH_REPLY_LIMIT

    # Create a mock corpus with small passages
    class MockChunk:
        def __init__(self, id_str, text):
            self.id = id_str
            self.text = text

    class MockCorpus:
        def search(self, query, k):
            # Return 5 chunks of ~1500 chars each
            text_1500 = "A" * 1500
            return [
                MockChunk("chunk-1", text_1500),
                MockChunk("chunk-2", text_1500),
                MockChunk("chunk-3", text_1500),
                MockChunk("chunk-4", text_1500),
                MockChunk("chunk-5", text_1500),
            ]

    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), MockCorpus(), load_variants()
    )

    # Simulate a search_spec call
    call = {"function": {"name": "search_spec", "arguments": json.dumps({"query": "test"})}}
    record = _Record()

    class MockToolContext:
        class MockPack:
            non_canon_columns = {}
        class MockRepo:
            pass
        pack = MockPack()
        repo = MockRepo()

    result = executor._execute(call, MockToolContext(), record)

    # Verify the response parses as JSON
    reply_json = json.loads(result["content"])
    # Verify all 5 passages are included
    assert len(reply_json["passages"]) == 5
    # Verify the reply is within limits
    assert len(result["content"]) <= SEARCH_REPLY_LIMIT


def test_search_reply_handles_large_passages(pack) -> None:
    """A corpus of 5 chunks each ~5000 chars → includes top passage, stays within limit."""
    from specdx.interpreters.executor import SEARCH_REPLY_LIMIT

    class MockChunk:
        def __init__(self, id_str, text):
            self.id = id_str
            self.text = text

    class MockCorpus:
        def search(self, query, k):
            # Return 5 chunks of ~5000 chars each
            text_5000 = "B" * 5000
            return [
                MockChunk("chunk-1", text_5000),
                MockChunk("chunk-2", text_5000),
                MockChunk("chunk-3", text_5000),
                MockChunk("chunk-4", text_5000),
                MockChunk("chunk-5", text_5000),
            ]

    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), MockCorpus(), load_variants()
    )

    # Simulate a search_spec call
    call = {"function": {"name": "search_spec", "arguments": json.dumps({"query": "test"})}}
    record = _Record()

    class MockToolContext:
        class MockPack:
            non_canon_columns = {}
        class MockRepo:
            pass
        pack = MockPack()
        repo = MockRepo()

    result = executor._execute(call, MockToolContext(), record)

    # Verify the response parses as JSON
    reply_json = json.loads(result["content"])
    # Verify at least the top passage is included
    assert len(reply_json["passages"]) >= 1
    # Verify the reply is within limits
    assert len(result["content"]) <= SEARCH_REPLY_LIMIT


def test_malformed_gated_call_arguments_are_recorded(run_de) -> None:
    """Malformed JSON in gated tool arguments is recorded in usage."""
    transport, seen = _scripted(
        _call("log_event", "{not json"),
        _say("Xong."),
    )
    envelope, rows = run_de(transport)
    # Check that the malformed call was recorded in usage
    assert envelope.usage["tool_calls"][0]["name"] == "log_event"
    assert envelope.usage["tool_calls"][0]["outcome"] == "error"


def test_search_spec_corpus_error_is_recorded(db, pack) -> None:
    """When corpus.search raises, the error is recorded in usage."""
    class FailingCorpus:
        """A corpus that raises on search."""
        def search(self, query, k):
            raise RuntimeError("corpus is broken")

    transport, seen = _scripted(
        _call("search_spec", {"query": "test"}),
        _say("Xong."),
    )
    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), FailingCorpus(),
        load_variants(), transport=transport, seed=7
    )
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        open_case(runtime.repo)
        envelope = runtime.run("DE-OP-01", "check_availability", "CASE-0031",
                              {"ma_so": "SO-0031"})
    # Check that the search_spec error was recorded in usage
    assert envelope.usage["tool_calls"][0]["name"] == "search_spec"
    assert envelope.usage["tool_calls"][0]["outcome"] == "error"


def test_oversized_top_passage_is_trimmed_with_ellipsis(pack) -> None:
    """A corpus whose top chunk is ~20000 chars → reply is valid JSON, within limit, ends with …."""
    from specdx.interpreters.executor import SEARCH_REPLY_LIMIT

    class MockChunk:
        def __init__(self, id_str, text):
            self.id = id_str
            self.text = text

    class MockCorpus:
        def search(self, query, k):
            # Return a chunk much larger than SEARCH_REPLY_LIMIT
            text_20000 = "X" * 20000
            return [MockChunk("big-chunk", text_20000)]

    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), MockCorpus(), load_variants()
    )

    # Simulate a search_spec call
    call = {"function": {"name": "search_spec", "arguments": json.dumps({"query": "test"})}}
    record = _Record()

    class MockToolContext:
        class MockPack:
            non_canon_columns = {}
        class MockRepo:
            pass
        pack = MockPack()
        repo = MockRepo()

    result = executor._execute(call, MockToolContext(), record)

    # Verify the response parses as JSON
    reply_json = json.loads(result["content"])
    # Verify the reply is within limits
    assert len(result["content"]) <= SEARCH_REPLY_LIMIT
    # Verify the passage text ends with "…"
    assert reply_json["passages"][0]["text"].endswith("…")


INVOICE = {"ma_inv": "INV-0031", "ma_so": "SO-0031", "ma_kh": "CUST-0007"}
FLAG = {"case_id": "CASE-0031", "kind": "FLAGGED", "ref": "ITEM-0012", "actor": "AI",
        "basis": "thiếu"}


def test_a_drafting_step_that_wrote_nothing_is_nudged_at_most_twice(run_de) -> None:
    transport, seen = _scripted(_say("Xong."), _say("xong"), _say("xong"))
    envelope, _ = run_de(transport, de="DE-FN-01", intent="draft_sales_invoice", inputs=INVOICE)
    assert len(seen) == 3
    assert seen[1]["messages"][-1] == {"role": "user", "content": NUDGE}
    assert [m["content"] for m in seen[2]["messages"] if m["role"] == "user"][-2:] == [NUDGE,
                                                                                      NUDGE]
    assert envelope.usage["harness"]["nudges"] == 2
    assert envelope.status is ResultStatus.ERROR  # it never wrote: not done


def test_a_step_that_drafts_nothing_is_never_nudged(run_de) -> None:
    transport, seen = _scripted(_say("Xong."))
    envelope, _ = run_de(transport)
    assert len(seen) == 1 and envelope.usage["harness"]["nudges"] == 0


def test_a_successful_write_ends_the_step_without_a_nudge(run_de) -> None:
    transport, seen = _scripted(_call("log_event", FLAG), _say("Xong."))
    envelope, _ = run_de(transport, de="DE-FN-01", intent="draft_sales_invoice", inputs=INVOICE)
    assert len(seen) == 2 and envelope.usage["harness"]["nudges"] == 0
    assert envelope.status is ResultStatus.DONE


def test_a_tool_call_written_as_text_is_executed_and_marked(run_de) -> None:
    text = "Tôi sẽ gắn cờ.\n" + json.dumps({"name": "log_event", "parameters": FLAG})
    transport, seen = _scripted(_say(text), _say("Đã gắn cờ."))
    envelope, rows = run_de(transport)
    assert {"kind": "FLAGGED", "ref": "ITEM-0012"} in rows
    assert envelope.usage["harness"] == {"nudges": 0, "parsed_from_text": 1, "tool_calls": 1,
                                         "valid_calls": 1, "requests": 2, "prompt_tokens": 0,
                                         "eval_tokens": 0, "duration_ms": 0}
    assert envelope.usage["tool_calls"][0]["parsed_from_text"] is True
    assert envelope.status is ResultStatus.DONE
    # the history shows the parsed call as a native one, answered by its tool message
    asked, answered = seen[1]["messages"][-2:]
    assert asked["tool_calls"] == [{"function": {"name": "log_event", "arguments": FLAG}}]
    assert answered["role"] == "tool" and answered["tool_name"] == "log_event"


def test_a_text_call_to_a_tool_the_step_lacks_is_not_executed(run_de) -> None:
    text = json.dumps({"name": "write_draft", "arguments": {"table": "so", "payload": {},
                                                            "idempotency_key": "k"}})
    transport, seen = _scripted(_say(text))
    envelope, rows = run_de(transport)
    assert envelope.usage["harness"]["parsed_from_text"] == 0 and rows == []


def test_native_calls_are_counted_valid_only_when_they_fit_the_schema(run_de) -> None:
    transport, seen = _scripted(_call("read_records", {"table": "so", "colour": "red"}),
                                _say("Xong."))
    envelope, _ = run_de(transport)
    assert envelope.usage["harness"]["tool_calls"] == 1
    assert envelope.usage["harness"]["valid_calls"] == 0


def test_a_drafting_step_that_only_read_is_nudged(run_de) -> None:
    transport, seen = _scripted(_call("match_three_way", {"po": "PO-0015"}), _say("Xong."),
                                _say("xong"), _say("xong"))
    po = {"ma_po": "PO-0015", "ma_ncc": "VEND-0003", "case_id": "CASE-0031", "sl_dat": 15,
          "trang_thai": "Đã duyệt"}
    envelope, _ = run_de(transport, de="DE-FN-02", intent="reconcile",
                         inputs={"ma_po": "PO-0015", "ma_vinv": "VINV-0015"}, rows=[("po", po)])
    assert envelope.usage["tool_calls"][0] == {"name": "match_three_way",
                                               "args": {"po": "PO-0015"}, "outcome": "ok"}
    assert len(seen) == 4 and seen[2]["messages"][-1] == {"role": "user", "content": NUDGE}
    assert envelope.usage["harness"]["nudges"] == 2  # a reconciliation read is no write


def test_the_last_round_of_a_step_is_never_spent_on_a_nudge(run_de) -> None:
    transport, seen = _scripted(_say("Xong."), _say("xong"), _say("xong"))
    envelope, _ = run_de(transport, de="DE-FN-01", intent="draft_sales_invoice", inputs=INVOICE,
                         max_steps=2)
    assert len(seen) == 2 and envelope.usage["harness"]["nudges"] == 1


def test_an_outage_during_a_nudge_round_trip_is_flagged_and_keeps_the_counts(db, corpus) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json={"message": _say("Xong.")})
        raise httpx.ConnectError("connection refused", request=request)

    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), corpus, load_variants(),
        transport=httpx.MockTransport(handle), sleep=lambda s: None,
    )
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        open_case(runtime.repo)
        envelope = runtime.run("DE-FN-01", "draft_sales_invoice", "CASE-0031", INVOICE)
    assert len(calls) == 6  # the answer, then the nudge's five attempts
    assert executor.infra_failure is not None and "ConnectError" in executor.infra_failure
    assert envelope.status is ResultStatus.ERROR
    # Every attempt is a request, the five failed ones too.
    assert envelope.usage["harness"] == {"nudges": 1, "parsed_from_text": 0, "tool_calls": 0,
                                         "valid_calls": 0, "requests": 6, "prompt_tokens": 0,
                                         "eval_tokens": 0, "duration_ms": 0}


def test_retried_and_failed_attempts_are_requests_with_usage_only_on_success(run_de) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= 2:
            return httpx.Response(500, json={"error": "CUDA error: out of memory"})
        return httpx.Response(200, json={"message": _say("Xong."), "prompt_eval_count": 120,
                                         "eval_count": 30, "total_duration": 2_500_000_000})

    envelope, _ = run_de(httpx.MockTransport(handle), sleep=lambda s: None)
    h = envelope.usage["harness"]
    assert len(calls) == 3
    assert (h["requests"], h["prompt_tokens"], h["eval_tokens"], h["duration_ms"]) == (
        3, 120, 30, 2500)


def test_a_read_timeout_is_a_request(run_de) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    envelope, _ = run_de(httpx.MockTransport(handle))
    assert envelope.usage["harness"]["requests"] == 1


def test_each_request_records_its_usage(run_de) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": _say("Xong."), "done": True,
                                         "prompt_eval_count": 120, "eval_count": 30,
                                         "total_duration": 2_500_000_000})

    envelope, _ = run_de(httpx.MockTransport(handle))
    h = envelope.usage["harness"]
    assert (h["requests"], h["prompt_tokens"], h["eval_tokens"], h["duration_ms"]) == (
        1, 120, 30, 2500)
