"""Tell variants 1..n: paraphrases of variant 0 by a local model, checked and then frozen."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

import httpx
from tonomyx.de.executor import _plain

from specdx.config import OLLAMA_URL
from specdx.interpreters.prompt import TellVariants
from specdx.orchestrate.work_orders import TEMPLATE_NAMES

REWRITE = (
    "Viết lại đoạn văn sau bằng tiếng Việt với cách diễn đạt khác nhưng giữ nguyên ý nghĩa. "
    "Không thêm, bớt hay thay đổi chi tiết, điều kiện hay đối tượng nào; "
    "không dùng thuật ngữ của lĩnh vực khác. "
)
KEEP_LINES = "Chép nguyên văn từng dòng sau, mỗi dòng trên một dòng riêng: {items}. "
KEEP_PLACEHOLDERS = (
    "Giữ nguyên từng ký hiệu trong ngoặc nhọn ({items}) đúng như cũ và không thêm ký hiệu nào "
    "khác. "
)
KEEP_CODES = "Giữ nguyên từng mã ({items}) đúng như cũ và không thêm mã nào khác. "
KEEP_IDENTIFIERS = (
    "Giữ nguyên từng định danh ({items}) đúng như cũ và không thêm định danh nào khác. "
)
ONLY = "Chỉ trả về đoạn văn đã viết lại."
HAND_WRITTEN = "hand-written"
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_CODE = re.compile(r"[A-Z]{2,}-\d+")
# Tool and input names such as search_spec and discount_pct, plus the input keys a
# scenario fills in (deliveries, received, ...), which need no underscore.
_IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_INPUT_KEY = re.compile(r"\b(?:" + "|".join(sorted(TEMPLATE_NAMES)) + r")\b")
# A label line is made only of "Label: {placeholder}." segments; it must stay verbatim.
_SEGMENT = r"[^:{}\n.]+:\s*\{[a-z_]+\}\."
_LABEL_LINE = re.compile(rf"\s*{_SEGMENT}(?:\s+{_SEGMENT})*\s*")
_LEAKS = ("/no_th", "<think", "</think")
# CJK symbols and ideographs, Hangul syllables, Cyrillic: never in a Vietnamese paraphrase.
_FOREIGN = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\u0400-\u04ff]")
log = logging.getLogger(__name__)


def label_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if _LABEL_LINE.fullmatch(line)]


def protected_names(text: str) -> set[str]:
    """The identifiers a paraphrase must keep: snake_case names and scenario input keys."""
    return set(_IDENTIFIER.findall(text)) | set(_INPUT_KEY.findall(text))


def prompt_for(text: str) -> str:
    """The request for one text: label lines, placeholders, codes and identifiers are
    mentioned only when the text has them, since naming them otherwise makes models invent
    some. An identifier inside a placeholder ({case_id}) is named with the placeholders."""
    parts = [REWRITE]
    labels = label_lines(text)
    if labels:
        parts.append(KEEP_LINES.format(items="; ".join(f"“{line}”" for line in labels)))
    placeholders = sorted(set(_PLACEHOLDER.findall(text)))
    if placeholders:
        parts.append(KEEP_PLACEHOLDERS.format(items=", ".join(placeholders)))
    codes = sorted(set(_CODE.findall(text)))
    if codes:
        parts.append(KEEP_CODES.format(items=", ".join(codes)))
    names = sorted(protected_names(_PLACEHOLDER.sub(" ", text)))
    if names:
        parts.append(KEEP_IDENTIFIERS.format(items=", ".join(names)))
    parts.append(ONLY)
    return "".join(parts) + "\n\n" + text


def rejection(text: str, out: str, avoid: Sequence[str] = ()) -> str | None:
    """Why a paraphrase cannot replace the text, or None when it can. avoid holds the
    other variants of the same text, which a paraphrase must not repeat."""
    if not out.strip():
        return "empty"
    if set(_PLACEHOLDER.findall(out)) != set(_PLACEHOLDER.findall(text)):
        return "placeholders differ"
    if "{" in _PLACEHOLDER.sub("", out) or "}" in _PLACEHOLDER.sub("", out):
        return "a stray brace"  # str.format would fail on it
    lines = {line.strip() for line in out.splitlines()}
    if any(label not in lines for label in label_lines(text)):
        return "a label line changed"
    if set(_CODE.findall(out)) != set(_CODE.findall(text)):
        return "codes differ"
    if protected_names(out) != protected_names(text):
        return "identifiers differ"
    lowered = out.lower()
    if any(leak in lowered for leak in _LEAKS):
        return "a thinking marker leaked"
    if _FOREIGN.search(out):
        return "CJK or Cyrillic characters"
    if out.strip() in {text.strip(), *(a.strip() for a in avoid)}:
        return "duplicate"
    return None


def paraphrase(
    text: str,
    *,
    model: str,
    seed: int,
    client: httpx.Client,
    base_url: str,
    retries: int = 6,
    avoid: Sequence[str] = (),
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
        reason = rejection(text, out, avoid)
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
            standing.append(paraphrase(standing[0], model=model, seed=n, client=client,
                                       base_url=base_url, avoid=standing[1:]))
            for texts in statements.values():
                texts.append(paraphrase(texts[0], model=model, seed=n, client=client,
                                        base_url=base_url, avoid=texts[1:]))
    marks = (HAND_WRITTEN, *([model] * count))
    return TellVariants(
        tuple(standing), {k: tuple(v) for k, v in statements.items()},
        {key: marks for key in ["standing", *statements]},
    )


def regenerate(
    variants: TellVariants,
    keys: Sequence[str],
    index: int,
    *,
    model: str,
    base_url: str = OLLAMA_URL,
    transport: httpx.BaseTransport | None = None,
) -> TellVariants:
    """Variant `index` of the named entries ("standing" or statement keys) made again,
    different from every variant it has now; everything else is left as it is."""
    if not 1 <= index < variants.count:
        raise ValueError(f"variant {index} is not a paraphrase: choose 1 to "
                         f"{variants.count - 1} (variant 0 is hand-written)")
    unknown = [k for k in keys if k != "standing" and k not in variants.statements]
    if unknown or not keys:
        raise ValueError(f"unknown entries {unknown}: choose standing or one of "
                         f"{sorted(variants.statements)}")
    standing = list(variants.standing)
    statements = {k: list(v) for k, v in variants.statements.items()}
    provenance = {k: list(v) for k, v in variants.provenance.items()}
    with httpx.Client(transport=transport, timeout=300.0) as client:
        for key in keys:
            texts = standing if key == "standing" else statements[key]
            texts[index] = paraphrase(texts[0], model=model, seed=index, client=client,
                                      base_url=base_url, avoid=texts[1:])
            marks = provenance.setdefault(key, [HAND_WRITTEN] + [None] * (variants.count - 1))
            marks[index] = model
    return TellVariants(tuple(standing), {k: tuple(v) for k, v in statements.items()},
                        {k: tuple(v) for k, v in provenance.items()})
