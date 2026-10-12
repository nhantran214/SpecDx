import json

import pytest
from tonomyx.tools.catalog import build_registry

from specdx.interpreters.executor import SEARCH_SPEC
from specdx.interpreters.harness import (
    NUDGE,
    READ_ONLY_TOOLS,
    WRITE_TOOLS,
    parse_text_calls,
    tool_parameters,
    valid_call,
    validate_arguments,
    wrote,
)

SCHEMAS = [
    {"type": "function", "function": {"name": "log_event", "parameters": {
        "type": "object",
        "properties": {"case_id": {"type": "string"}, "kind": {"type": "string"},
                       "ref": {"type": "string"}, "actor": {"type": "string"},
                       "basis": {"type": "string"}},
        "required": ["case_id", "kind", "ref", "actor", "basis"]}}},
    {"type": "function", "function": {"name": "read_records", "parameters": {
        "type": "object",
        "properties": {"table": {"type": "string"}, "limit": {"type": "integer"},
                       "where": {"type": "object"}},
        "required": ["table"]}}},
]
TOOLS = tool_parameters(SCHEMAS)
ARGS = {"case_id": "CASE-0031", "kind": "FLAGGED", "ref": "ITEM-0012", "actor": "AI",
        "basis": "thiếu hàng"}
EXPECTED = [{"function": {"name": "log_event", "arguments": ARGS}}]


@pytest.mark.parametrize("text", [
    json.dumps({"name": "log_event", "arguments": ARGS}),
    json.dumps({"name": "log_event", "parameters": ARGS}),
    json.dumps({"function": "log_event", "parameters": ARGS}),
    json.dumps({"name": "log_event", "arguments": json.dumps(ARGS)}),
    "```json\n" + json.dumps({"name": "log_event", "arguments": ARGS}) + "\n```",
    "<tool_call>" + json.dumps({"name": "log_event", "arguments": ARGS}) + "</tool_call>",
    "Tôi sẽ gắn cờ.\n\n" + json.dumps({"name": "log_event", "parameters": ARGS}) + "\nXong.",
])
def test_every_accepted_text_form_is_parsed(text) -> None:
    assert parse_text_calls(text, TOOLS) == EXPECTED


def test_several_calls_in_one_reply_are_parsed_in_order() -> None:
    first = {"name": "read_records", "arguments": {"table": "so"}}
    text = json.dumps(first) + "\n" + json.dumps({"name": "log_event", "arguments": ARGS})
    assert [c["function"]["name"] for c in parse_text_calls(text, TOOLS)] == [
        "read_records", "log_event"]


@pytest.mark.parametrize("obj", [
    {"name": "send_email", "arguments": {"to": "x"}},  # not one of the step's tools
    {"name": "log_event", "arguments": {"case_id": "CASE-0031"}},  # required keys missing
    {"name": "log_event", "arguments": ARGS | {"extra": 1}},  # unknown key
    {"name": "read_records", "arguments": {"table": "so", "limit": "5"}},  # wrong type
    {"name": "read_records", "arguments": {"table": "so", "limit": True}},  # bool is no int
    {"name": "log_event"},  # no arguments key
    {"table": "po", "payload": {"ma_po": "PO-0015"}},  # arguments without a tool name
])
def test_invalid_names_and_arguments_are_rejected(obj) -> None:
    assert parse_text_calls(json.dumps(obj), TOOLS) == []


def test_text_that_is_not_json_yields_no_call() -> None:
    assert parse_text_calls("Đã xong {không phải json}", TOOLS) == []


def test_arguments_validate_against_the_schema() -> None:
    params = TOOLS["read_records"]
    assert validate_arguments(params, {"table": "so"})
    assert validate_arguments(params, {"table": "so", "limit": None})  # optional may be null
    assert not validate_arguments(params, {"table": None})  # a required one may not
    assert not validate_arguments(params, ["so"])


def test_a_native_call_is_valid_only_with_a_known_tool_and_valid_arguments() -> None:
    assert valid_call({"function": {"name": "read_records", "arguments": '{"table": "so"}'}},
                      TOOLS)
    assert not valid_call({"function": {"name": "read_records", "arguments": "{oops"}}, TOOLS)
    assert not valid_call({"function": {"name": "nope", "arguments": {}}}, TOOLS)


def test_only_a_successful_state_changing_call_counts_as_a_write() -> None:
    assert not wrote([{"name": "search_spec", "outcome": "ok"},
                      {"name": "write_draft", "outcome": "denied"}])
    assert not wrote([{"name": "match_three_way", "outcome": "ok"}])  # it reads only
    assert wrote([{"name": "log_event", "outcome": "ok"}])


def test_every_tool_is_classified_as_reading_or_writing() -> None:
    assert not READ_ONLY_TOOLS & WRITE_TOOLS
    assert build_registry().names() | {SEARCH_SPEC} == READ_ONLY_TOOLS | WRITE_TOOLS


def test_the_nudge_restates_the_standing_rule_and_allows_confirming_done() -> None:
    assert NUDGE == ("Bạn chưa thực hiện hành động nào bằng công cụ. Nếu còn việc phải làm, "
                     "hãy dùng công cụ; nếu thật sự đã xong, hãy trả lời lại 'xong'.")
