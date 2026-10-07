import json

import httpx
import pytest

from specdx.interpreters.paraphrase import make_variants, paraphrase
from specdx.interpreters.prompt import load_variants


def _model(*answers, seen=None):
    """Answers in turn; once they run out, echoes the text it was asked to rewrite."""
    queue = list(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if seen is not None:
            seen.append(prompt)
        text = queue.pop(0) if queue else prompt.split("\n\n", 1)[1]
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    return httpx.MockTransport(handle)


def _paraphrase(text, *answers, seen=None):
    with httpx.Client(transport=_model(*answers, seen=seen)) as client:
        return paraphrase(text, model="m", seed=1, client=client, base_url="http://x")


def test_a_paraphrase_must_keep_every_placeholder() -> None:
    assert _paraphrase("A {role} B {actor}", "mất {role}", "giữ {role} và {actor}") == (
        "giữ {role} và {actor}")


def test_a_text_without_placeholders_or_codes_gets_no_instruction_about_them() -> None:
    seen: list[str] = []
    _paraphrase("soạn hóa đơn bán hàng cho khách", "lập hóa đơn bán cho khách hàng", seen=seen)
    instructions = seen[0].split("\n\n", 1)[0]
    assert "{" not in instructions and "ngoặc nhọn" not in instructions
    assert "APR-0102" not in instructions and "mã" not in instructions.split()
    assert "định danh" not in instructions
    assert "tiếng Việt" in instructions and "Chỉ trả về" in instructions


def test_the_instructions_name_the_placeholders_codes_and_identifiers_the_text_has() -> None:
    seen: list[str] = []
    _paraphrase("Bạn là {role}; dùng search_spec rồi duyệt APR-0102.",
                "Với vai {role}, tra search_spec và duyệt APR-0102.", seen=seen)
    instructions = seen[0].split("\n\n", 1)[0]
    assert "{role}" in instructions and "APR-0102" in instructions
    assert "định danh (search_spec)" in instructions


def test_a_renamed_or_dropped_identifier_is_rejected() -> None:
    text = "soạn báo giá (discount_pct là phần trăm chiết khấu khách đề nghị)"
    assert _paraphrase(text, "lập báo giá (discount_percent là mức chiết khấu khách xin)",
                       "lập báo giá (chiết khấu khách xin)",
                       "lập báo giá (discount_pct là mức chiết khấu khách xin)") == (
        "lập báo giá (discount_pct là mức chiết khấu khách xin)")


def test_an_invented_placeholder_is_rejected_and_the_retry_accepted() -> None:
    assert _paraphrase("soạn đơn mua hàng", "soạn đơn mua {order_id}", "lập đơn mua hàng") == (
        "lập đơn mua hàng")


def test_an_invented_or_lost_code_is_rejected() -> None:
    assert _paraphrase("duyệt báo giá", "duyệt APR-0102 báo giá", "phê duyệt báo giá") == (
        "phê duyệt báo giá")
    assert _paraphrase("dừng ở cổng APR-0102", "dừng ở cổng duyệt", "chờ ở cổng APR-0102") == (
        "chờ ở cổng APR-0102")


def test_a_stray_brace_is_rejected() -> None:
    assert _paraphrase("A {role}", "giữ {role} và {", "giữ {role}") == "giữ {role}"


@pytest.mark.parametrize("leak", ["duyệt 报价 nhanh", "duyệt 견적", "duyệt счёт"])
def test_a_cjk_or_cyrillic_leak_is_rejected(leak) -> None:
    assert _paraphrase("duyệt báo giá", leak, "phê duyệt báo giá") == "phê duyệt báo giá"


@pytest.mark.parametrize("leak", ["<think>cần viết lại", "</think> phê duyệt", "/no_think duyệt"])
def test_a_thinking_leak_is_rejected(leak) -> None:
    assert _paraphrase("duyệt báo giá", leak, "phê duyệt báo giá") == "phê duyệt báo giá"


def test_an_empty_answer_is_rejected() -> None:
    assert _paraphrase("duyệt báo giá", "   ", "phê duyệt báo giá") == "phê duyệt báo giá"


def test_a_valid_output_is_returned_unchanged() -> None:
    out = "Với vai trò {role} ({actor}), hãy duyệt APR-0102 cho {case_id}."
    assert _paraphrase("Bạn là {role} ({actor}); duyệt APR-0102 cho {case_id}.", out) == out


def test_a_model_that_never_complies_is_refused_after_six_attempts() -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x {y}"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client, pytest.raises(ValueError):
        paraphrase("A {role}", model="m", seed=1, client=client, base_url="http://x")
    assert len(calls) == 6


def test_make_variants_adds_count_versions_of_everything() -> None:
    base = load_variants()
    made = make_variants(base, model="m", count=2, base_url="http://x", transport=_model())
    assert made.count == 3 and made.standing[0] == base.standing[0]
    assert all(len(v) == 3 for v in made.statements.values())
