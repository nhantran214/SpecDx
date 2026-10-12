"""The neutral execution harness (spec v2 §2.1), at the executor's chat boundary.

Its texts are fixed constants, the same for every specification, and carry no business
value or rule. It does two things:

- a tool call the model wrote as text (a JSON object naming one of the step's tools,
  anywhere in its reply, fenced or in <tool_call> tags) is executed like a native call,
  provided its arguments validate against the tool's schema;
- a model that ends a step that drafts tables without any successful write call is asked,
  at most MAX_NUDGES times, to act or to confirm that it is done.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tonomyx.de.executor import _arguments

NUDGE = (
    "Bạn chưa thực hiện hành động nào bằng công cụ. Nếu còn việc phải làm, hãy dùng công cụ; "
    "nếu thật sự đã xong, hãy trả lời lại 'xong'."
)
MAX_NUDGES = 2  # spec v2 §3.10
# Every tool is classified, by reading its implementation under vendor/tonomyx/tonomyx/tools/impl.
# A step that only called read-only tools has not acted; only a WRITE_TOOLS call counts as
# acting. A tool added later fails tests/interpreters/test_harness.py until it is classified.
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {"read_records", "search_kb", "search_spec", "match_three_way"}
)
WRITE_TOOLS: frozenset[str] = frozenset({
    "write_draft", "update_status", "log_event", "create_approval", "decide_approval",
    "send_email",
})
_NAME_KEYS = ("name", "function")
_ARGUMENT_KEYS = ("arguments", "parameters")
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "object": (dict,), "array": (list,),
}

Parameters = Mapping[str, Any]


def tool_parameters(schemas: Sequence[Mapping[str, Any]]) -> dict[str, Parameters]:
    """Tool name → its JSON parameter schema, from the schemas offered to the model."""
    return {str(s["function"]["name"]): s["function"].get("parameters", {}) for s in schemas}


def validate_arguments(parameters: Parameters, args: Any) -> bool:
    """Whether args fit the schema: an object with every required key, no unknown key, and
    each value of its declared JSON type (null only where the key is optional)."""
    if not isinstance(args, dict):
        return False
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    if not required <= set(args):
        return False
    for name, value in args.items():
        if name not in properties:
            return False
        expected = properties[name].get("type")
        if value is None:
            if name in required:
                return False
            continue
        kinds = _JSON_TYPES.get(str(expected)) if expected else None
        if kinds is None:
            continue
        if isinstance(value, bool) and expected in ("integer", "number"):
            return False
        if not isinstance(value, kinds):
            return False
    return True


def valid_call(call: Mapping[str, Any], tools: Mapping[str, Parameters]) -> bool:
    """A native call names one of the step's tools with arguments that validate."""
    function = call.get("function") or {}
    name = str(function.get("name", ""))
    if name not in tools:
        return False
    try:
        args = _arguments(function.get("arguments"))
    except (ValueError, TypeError):
        return False
    return validate_arguments(tools[name], args)


def _as_call(obj: Any, tools: Mapping[str, Parameters]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name_key = next((k for k in _NAME_KEYS if k in obj), None)
    args_key = next((k for k in _ARGUMENT_KEYS if k in obj), None)
    if name_key is None or args_key is None or not isinstance(obj[name_key], str):
        return None
    call = {"function": {"name": obj[name_key], "arguments": obj[args_key]}}
    if not valid_call(call, tools):
        return None
    call["function"]["arguments"] = _arguments(obj[args_key])
    return call


def parse_text_calls(content: str, tools: Mapping[str, Parameters]) -> list[dict[str, Any]]:
    """The valid tool calls written in a reply's text, in order, shaped as native calls.
    Every top-level JSON object in the text is a candidate, so fences, <tool_call> tags and
    surrounding prose are all accepted."""
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    start = content.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            start = content.find("{", start + 1)
            continue
        call = _as_call(obj, tools)
        if call is not None:
            found.append(call)
        start = content.find("{", end)
    return found


def wrote(calls: Sequence[Mapping[str, Any]]) -> bool:
    """Whether any call of the step succeeded at changing something."""
    return any(c.get("outcome") == "ok" and c.get("name") in WRITE_TOOLS for c in calls)
