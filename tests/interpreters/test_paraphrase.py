import json

import httpx
import pytest

from specdx.interpreters.paraphrase import make_variants, paraphrase
from specdx.interpreters.prompt import load_variants


def _model(*answers):
    queue = list(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = queue.pop(0) if queue else body["messages"][0]["content"].split("\n\n", 1)[1]
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    return httpx.MockTransport(handle)


def test_a_paraphrase_must_keep_every_placeholder() -> None:
    with httpx.Client(transport=_model("mất {role}", "giữ {role} và {actor}")) as client:
        out = paraphrase("A {role} B {actor}", model="m", seed=1, client=client, base_url="http://x")
    assert out == "giữ {role} và {actor}"


def test_a_paraphrase_that_never_keeps_them_is_refused() -> None:
    with httpx.Client(transport=_model("x", "y", "z")) as client, pytest.raises(ValueError):
        paraphrase("A {role}", model="m", seed=1, client=client, base_url="http://x")


def test_make_variants_adds_count_versions_of_everything() -> None:
    base = load_variants()
    made = make_variants(base, model="m", count=2, base_url="http://x", transport=_model())
    assert made.count == 3 and made.standing[0] == base.standing[0]
    assert all(len(v) == 3 for v in made.statements.values())
