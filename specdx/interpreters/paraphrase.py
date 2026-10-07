"""Tell variants 1..n: paraphrases of variant 0 by a local model, checked and then frozen."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from tonomyx.de.executor import _plain

from specdx.config import OLLAMA_URL
from specdx.interpreters.prompt import TellVariants

REWRITE = (
    "Viết lại đoạn văn sau bằng tiếng Việt với cách diễn đạt khác nhưng giữ nguyên ý nghĩa. "
)
KEEP_PLACEHOLDERS = (
    "Giữ nguyên từng ký hiệu trong ngoặc nhọn ({items}) đúng như cũ và không thêm ký hiệu nào "
    "khác. "
)
KEEP_CODES = "Giữ nguyên từng mã ({items}) đúng như cũ và không thêm mã nào khác. "
ONLY = "Chỉ trả về đoạn văn đã viết lại."
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_CODE = re.compile(r"[A-Z]{2,}-\d+")
_LEAKS = ("/no_th", "<think", "</think")
# CJK symbols and ideographs, Hangul syllables, Cyrillic: never in a Vietnamese paraphrase.
_FOREIGN = re.compile("[　-鿿가-힯Ѐ-ӿ]")
log = logging.getLogger(__name__)


def prompt_for(text: str) -> str:
    """The request for one text: placeholders and codes are mentioned only when the text
    has them, since naming them otherwise makes models invent some."""
    parts = [REWRITE]
    placeholders = sorted(set(_PLACEHOLDER.findall(text)))
    if placeholders:
        parts.append(KEEP_PLACEHOLDERS.format(items=", ".join(placeholders)))
    codes = sorted(set(_CODE.findall(text)))
    if codes:
        parts.append(KEEP_CODES.format(items=", ".join(codes)))
    parts.append(ONLY)
    return "".join(parts) + "\n\n" + text


def rejection(text: str, out: str) -> str | None:
    """Why a paraphrase cannot replace the text, or None when it can."""
    if not out.strip():
        return "empty"
    if set(_PLACEHOLDER.findall(out)) != set(_PLACEHOLDER.findall(text)):
        return "placeholders differ"
    if "{" in _PLACEHOLDER.sub("", out) or "}" in _PLACEHOLDER.sub("", out):
        return "a stray brace"  # str.format would fail on it
    if set(_CODE.findall(out)) != set(_CODE.findall(text)):
        return "codes differ"
    lowered = out.lower()
    if any(leak in lowered for leak in _LEAKS):
        return "a thinking marker leaked"
    if _FOREIGN.search(out):
        return "CJK or Cyrillic characters"
    return None


def paraphrase(
    text: str, *, model: str, seed: int, client: httpx.Client, base_url: str, retries: int = 6
) -> str:
    reasons: list[str] = []
    for attempt in range(retries):
        body: dict[str, Any] = {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt_for(text)}],
            "options": {"temperature": 0.7, "seed": seed * 10 + attempt},
        }
        if model.startswith("qwen3"):
            body["think"] = False
        reply = client.post(f"{base_url}/api/chat", json=body).json()["message"]
        out = _plain(reply)
        reason = rejection(text, out)
        if reason is None:
            return out
        reasons.append(reason)
        log.warning("paraphrase by %s rejected (%s): %r", model, reason, out[:120])
    raise ValueError(f"{model} could not paraphrase {text[:60]!r} in {retries} attempts: "
                     f"{'; '.join(reasons)}")


def make_variants(
    variants: TellVariants,
    *,
    model: str,
    count: int,
    base_url: str = OLLAMA_URL,
    transport: httpx.BaseTransport | None = None,
) -> TellVariants:
    standing = [variants.standing[0]]
    statements = {k: [v[0]] for k, v in variants.statements.items()}
    with httpx.Client(transport=transport, timeout=300.0) as client:
        for n in range(1, count + 1):
            standing.append(paraphrase(variants.standing[0], model=model, seed=n,
                                       client=client, base_url=base_url))
            for key in statements:
                statements[key].append(paraphrase(variants.statements[key][0], model=model,
                                                  seed=n, client=client, base_url=base_url))
    return TellVariants(tuple(standing), {k: tuple(v) for k, v in statements.items()})
