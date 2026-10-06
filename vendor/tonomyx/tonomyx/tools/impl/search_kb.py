"""search_kb — look up knowledge base articles inside the token's domain.

Retrieval here is a plain case-insensitive substring scan over the article body
and title. That is enough for the canon's three articles, and it keeps the answer
reproducible. Swapping in embeddings later means replacing this one function.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from tonomyx.tools.registry import ToolContext, ToolDenied, require_tool

# A digital employee whose de_scope kb_domain is this may read every domain.
ALL_DOMAINS = "all"


def _fold(text: str) -> str:
    """Case-fold and strip accents so a query matches regardless of diacritics."""
    stripped = unicodedata.normalize("NFD", text)
    return "".join(c for c in stripped if unicodedata.category(c) != "Mn").casefold()


def search_kb(ctx: ToolContext, *, query: str, domain: str | None = None) -> dict[str, Any]:
    """Return matching passages with their citation, or an explicit miss."""
    require_tool(ctx, "search_kb")
    wanted = domain or (ctx.token.kb_domain if ctx.token.kb_domain != ALL_DOMAINS else None)
    if ctx.token.kb_domain != ALL_DOMAINS and wanted != ctx.token.kb_domain:
        raise ToolDenied(
            f"{ctx.token.sub} may only search the {ctx.token.kb_domain} domain, not {wanted}"
        )

    needle = _fold(query)
    passages: list[dict[str, str]] = []
    for doc in ctx.pack.kb.values():
        if wanted is not None and doc.domain != wanted:
            continue
        if needle in _fold(doc.body) or needle in _fold(doc.title):
            passages.append(
                {
                    "kb_id": doc.kb_id,
                    "version": doc.version,
                    "title": doc.title,
                    "text": doc.body,
                }
            )
    return {"found": bool(passages), "passages": passages}
