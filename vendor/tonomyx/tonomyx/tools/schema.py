"""Describe tools to a language model, derived from the tools themselves.

A schema is built from the tool function's own signature and the canon catalog's
description of it. Nothing here names a particular tool, so a tool added to the
registry is describable the moment it is registered.

The first parameter of every tool is the ToolContext the runtime injects. It is
never offered: a model that could supply its own context could supply its own
authority.
"""

from __future__ import annotations

import inspect
import types
import typing
from decimal import Decimal
from typing import Any

from tonomyx.canon.pack import CanonPack
from tonomyx.tools.registry import ToolFn, ToolRegistry

_SCALARS: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    Decimal: "number",
    bool: "boolean",
}


def _json_type(annotation: Any) -> dict[str, Any]:
    """Map one resolved Python annotation onto a JSON Schema fragment."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _json_type(inner[0])
        return {}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if origin is list or annotation is list:
        args = typing.get_args(annotation)
        items = _json_type(args[0]) if args else {}
        return {"type": "array", "items": items} if items else {"type": "array"}
    if isinstance(annotation, type) and annotation in _SCALARS:
        return {"type": _SCALARS[annotation]}
    return {}


def tool_schema(name: str, fn: ToolFn, entry: dict[str, str]) -> dict[str, Any]:
    """One tool described in the function-calling shape Ollama and OpenAI accept."""
    hints = typing.get_type_hints(fn)
    params = list(inspect.signature(fn).parameters.values())[1:]  # drop ctx
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in params:
        properties[param.name] = _json_type(hints.get(param.name, Any))
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    description = f"{entry['effect']}. Input: {entry['input']}. Gating: {entry['gating']}."
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def tool_schemas(
    registry: ToolRegistry, pack: CanonPack, *, names: frozenset[str]
) -> list[dict[str, Any]]:
    """Schemas for the named tools that are both registered and in the catalog."""
    catalog = pack.tool_catalog
    return [
        tool_schema(name, registry.get(name), catalog[name])
        for name in sorted(names)
        if name in registry.names() and name in catalog
    ]
