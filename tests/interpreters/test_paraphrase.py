import difflib
import json

import httpx
import pytest

from specdx.interpreters.paraphrase import (
    ORIGINAL,
    label_lines,
    make_variants,
    paraphrase,
    prompt_for,
    regenerate,
    rejection,
)
from specdx.interpreters.prompt import TellVariants, load_variants, save_variants


def _model(*answers, seen=None):
    """Answers in turn; once they run out, returns the text it was asked to rewrite with a
    numbered line appended: a different, valid paraphrase every call."""
    queue = list(answers)
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        calls.append(prompt)
        if seen is not None:
            seen.append(prompt)
        text = queue.pop(0) if queue else f"{prompt.split(chr(10) * 2, 1)[1]}\nBản {len(calls)}."
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    return httpx.MockTransport(handle)


def _paraphrase(text, *answers, seen=None, avoid=()):
    with httpx.Client(transport=_model(*answers, seen=seen)) as client:
        return paraphrase(text, model="m", seed=1, client=client, base_url="http://x",
                          avoid=avoid)


STANDING = load_variants().standing[0]
LABELS = ["Vụ việc: {case_id}. Nhiệm vụ: {statement}.", "Dữ liệu vào: {inputs}.",
          "Bảng được đọc: {read}.", "Bảng được soạn nháp: {draft}."]


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
    assert made.provenance["standing"] == ("hand-written", "m", "m")
    assert set(made.provenance) == {"standing", *base.statements}


def test_an_input_key_without_an_underscore_is_protected_too() -> None:
    text = "ghi nhận phiếu nhập kho theo các lần giao thực tế (deliveries liệt kê từng lần giao)"
    seen: list[str] = []
    kept = "ghi phiếu nhập theo từng lần giao (deliveries kê từng lần)"
    assert _paraphrase(text, "ghi phiếu nhập theo từng lần giao thực tế", kept,
                       seen=seen) == kept
    assert "định danh (deliveries)" in seen[0].split("\n\n", 1)[0]
    assert rejection(text, "ghi phiếu nhập theo các lần giao") == "identifiers differ"


def test_the_rewrite_instruction_forbids_new_details_and_foreign_jargon() -> None:
    assert ("Không thêm, bớt hay thay đổi chi tiết, điều kiện hay đối tượng nào; "
            "không dùng thuật ngữ của lĩnh vực khác.") in prompt_for("duyệt báo giá")


def test_a_paraphrase_identical_to_variant_zero_or_an_earlier_one_is_rejected() -> None:
    text = "soạn đề nghị chi cho hóa đơn nhà cung cấp"
    earlier = "lập đề nghị thanh toán cho hóa đơn nhà cung cấp"
    assert rejection(text, f"  {text} ") == "duplicate"
    assert rejection(text, earlier, avoid=[earlier]) == "duplicate"
    fresh = "chuẩn bị đề nghị chi cho hóa đơn của nhà cung cấp"
    assert _paraphrase(text, text, earlier, fresh, avoid=[earlier]) == fresh


def test_make_variants_never_repeats_an_earlier_variant() -> None:
    small = TellVariants(("Bạn là {role}.",), {"pay": ("soạn đề nghị chi",)})
    answers = {"Bạn là {role}.": ["Với vai {role}.", "Với vai {role}.", "Trong vai {role}."],
               "soạn đề nghị chi": ["lập đề nghị chi", "lập đề nghị chi", "chuẩn bị đề nghị chi"]}

    def handle(request: httpx.Request) -> httpx.Response:
        source = json.loads(request.content)["messages"][0]["content"].split("\n\n", 1)[1]
        reply = answers[source].pop(0)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": reply}})

    made = make_variants(small, model="m", count=2, base_url="http://x",
                         transport=httpx.MockTransport(handle))
    assert made.standing == ("Bạn là {role}.", "Với vai {role}.", "Trong vai {role}.")
    assert made.statements["pay"] == ("soạn đề nghị chi", "lập đề nghị chi",
                                      "chuẩn bị đề nghị chi")


def test_label_lines_are_found_listed_and_kept_verbatim() -> None:
    assert label_lines(STANDING) == LABELS
    instructions = prompt_for(STANDING).split("\n\n", 1)[0]
    assert all(f"“{line}”" in instructions for line in LABELS)
    changed = STANDING.replace("Bảng được đọc: {read}.", "Bảng đã đọc: {read}.")
    assert rejection(STANDING, changed) == "a label line changed"
    reworded = STANDING.replace("Quy tắc làm việc:", "Nguyên tắc làm việc:")
    assert rejection(STANDING, reworded) is None


def test_regenerate_remakes_only_the_named_entries_at_one_index() -> None:
    base = TellVariants(("A {role}.", "B {role}.", "C {role}."),
                        {"x": ("x0", "x1", "x2"), "y": ("y0", "y1", "y2")},
                        {"x": ("hand-written", "old", "old")})
    made = regenerate(base, ["standing", "x"], 2, model="new", base_url="http://x",
                      transport=_model("D {role}.", "x1", "x2", "x-new"))
    assert made.standing == ("A {role}.", "B {role}.", "D {role}.")
    assert made.statements == {"x": ("x0", "x1", "x-new"), "y": ("y0", "y1", "y2")}
    assert made.provenance == {"x": ("hand-written", "old", "new"),
                               "standing": ("hand-written", None, "new")}


def test_regenerate_can_sample_another_seed_or_keep_the_original_wording() -> None:
    base = TellVariants(("A {role}.", "B {role}.", "C {role}."),
                        {"x": ("x0", "x1", "x2"), "y": ("y0", "y1", "y2")})
    seeds: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seeds.append(json.loads(request.content)["options"]["seed"])
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x-new"}})

    made = regenerate(base, ["x"], 1, model="m", seed=7, base_url="http://x",
                      transport=httpx.MockTransport(handle))
    assert made.statements["x"] == ("x0", "x-new", "x2") and seeds == [70]
    assert made.provenance["x"] == ("hand-written", "m seed 7", None)

    def no_call(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no model call expected")

    kept = regenerate(made, ["y"], 2, model="m", original=True, base_url="http://x",
                      transport=httpx.MockTransport(no_call))
    assert kept.statements["y"] == ("y0", "y1", "y0")
    assert kept.provenance["y"] == ("hand-written", None, ORIGINAL)
    assert kept.statements["x"] == made.statements["x"]


@pytest.mark.parametrize(("keys", "index"), [(["x"], 0), (["x"], 3), (["nope"], 1)])
def test_regenerate_refuses_a_bad_index_or_key_before_calling_a_model(keys, index) -> None:
    base = TellVariants(("A {role}.", "B {role}.", "C {role}."), {"x": ("x0", "x1", "x2")})

    def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no model call expected")

    with pytest.raises(ValueError):
        regenerate(base, keys, index, model="m", base_url="http://x",
                   transport=httpx.MockTransport(handle))


def test_a_regenerated_entry_changes_only_its_own_lines_in_the_file(tmp_path) -> None:
    path = tmp_path / "variants.json"
    base = make_variants(TellVariants(("A {role}.",), {"x": ("x0",), "y": ("y0",)}),
                         model="old", count=2, base_url="http://x", transport=_model())
    save_variants(base, path)
    before = path.read_text(encoding="utf-8").splitlines()
    loaded = load_variants(path)
    assert loaded == base
    save_variants(regenerate(loaded, ["x"], 1, model="new", base_url="http://x",
                             transport=_model("x-new")), path)
    after = path.read_text(encoding="utf-8").splitlines()
    diff = list(difflib.ndiff(before, after))
    removed = [line[2:] for line in diff if line.startswith("- ")]
    added = [line[2:] for line in diff if line.startswith("+ ")]
    assert len(before) == len(after)
    assert sorted(added) == sorted(['      "new",', '      "x-new",'])  # the entry, its model
    assert len(removed) == 2 and '      "old",' in removed
