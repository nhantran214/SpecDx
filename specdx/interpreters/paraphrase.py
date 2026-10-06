"""Tell variants 1..n: paraphrases of variant 0 by a local model, checked and then frozen."""

from __future__ import annotations

from typing import Any

import httpx
from tonomyx.de.executor import _plain

from specdx.config import OLLAMA_URL
from specdx.interpreters.prompt import PLACEHOLDERS, TellVariants

PROMPT = (
    "Viết lại đoạn văn sau bằng tiếng Việt với cách diễn đạt khác nhưng giữ nguyên ý nghĩa. "
    "Giữ nguyên từng ký hiệu trong ngoặc nhọn (như {role}) và mọi mã (như APR-0102). "
    "Chỉ trả về đoạn văn đã viết lại.\n\n"
)


def paraphrase(
    text: str, *, model: str, seed: int, client: httpx.Client, base_url: str, retries: int = 3
) -> str:
    required = [p for p in PLACEHOLDERS if p in text]
    for attempt in range(retries):
        body: dict[str, Any] = {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": PROMPT + text}],
            "options": {"temperature": 0.7, "seed": seed * 10 + attempt},
        }
        if model.startswith("qwen3"):
            body["think"] = False
        reply = client.post(f"{base_url}/api/chat", json=body).json()["message"]
        out = _plain(reply)
        if out and all(p in out for p in required) and out.count("{") == text.count("{"):
            return out
    raise ValueError(f"{model} could not paraphrase while keeping {required}")


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
