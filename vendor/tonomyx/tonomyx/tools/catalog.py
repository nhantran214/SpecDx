"""Wire the nine canon tools into one registry.

Adding a tenth tool means adding a line to the canon's tool-catalog.json, a name to
CANON_TOOL_NAMES, a module under impl/, and a line here. Nothing else changes.
"""

from __future__ import annotations

from tonomyx.tools.impl.create_approval import create_approval
from tonomyx.tools.impl.decide_approval import decide_approval
from tonomyx.tools.impl.log_event import log_event
from tonomyx.tools.impl.match_three_way import match_three_way
from tonomyx.tools.impl.read_records import read_records
from tonomyx.tools.impl.search_kb import search_kb
from tonomyx.tools.impl.send_email import send_email
from tonomyx.tools.impl.update_status import update_status
from tonomyx.tools.impl.write_draft import write_draft
from tonomyx.tools.registry import ToolRegistry


def build_registry() -> ToolRegistry:
    """A registry holding every tool the canon catalog declares."""
    registry = ToolRegistry()
    registry.register("read_records", read_records)
    registry.register("search_kb", search_kb)
    registry.register("write_draft", write_draft)
    registry.register("update_status", update_status)
    registry.register("create_approval", create_approval)
    registry.register("decide_approval", decide_approval)
    registry.register("log_event", log_event)
    registry.register("match_three_way", match_three_way)
    registry.register("send_email", send_email)
    return registry
