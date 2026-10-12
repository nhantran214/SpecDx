import pytest

from specdx.scenarios.space import boundary_domain, canon_scenarios, enumerate_scenarios
from specdx.scenarios.tguided import TLiteral, extract_literals, tboundary_scenarios
from specdx.spec.tell import Chunk, Sentence, TellCorpus


def _corpus(*texts: str, source: str = "KB") -> TellCorpus:
    return TellCorpus([Chunk("KB:T", source, tuple(Sentence(f"KB:T:{i:02d}", t)
                                                   for i, t in enumerate(texts)))])


def _bound(*texts: str) -> list[tuple[str, int]]:
    return [x.bound for x in extract_literals(_corpus(*texts))]


@pytest.mark.parametrize(("text", "expected"), [
    ("Mức chiết khấu tối đa là 7%;", [("discount_pct", 7)]),
    ("Yêu cầu vượt trần (vd 8%) phải gắn cờ ngoại lệ.", [("discount_pct", 8)]),
    ("Chiết khấu không vượt quá 10 phần trăm.", [("discount_pct", 10)]),
    ("Đơn có số lượng trên 100 đơn vị phải duyệt.", [("received", 100)]),
    ("Nhận ít nhất 12 cái theo số lượng đặt.", [("received", 12)]),
    ("Refunds of at most 10% are automatic.", [("discount_pct", 10)]),
    ("Orders that exceed 20 units need a manager.", [("received", 20)]),
    ("A quantity above 5 is split in two shipments.", [("received", 5)]),
])
def test_thresholds_in_vietnamese_and_english_are_extracted_and_bound(text, expected) -> None:
    assert _bound(text) == expected


@pytest.mark.parametrize("text", [
    "Liên hệ trong 24 giờ.",  # no comparison keyword
    "Trên 5 khách hàng đã hỏi.",  # a keyword, but neither a percentage nor a quantity
    "APR-0102 vượt trần của KB-0003.",  # identifiers are not numbers
    "Hiệu lực từ 2026-09-01, mức trần không đổi.",  # nor are dates
    "Chiết khấu 5% được ghi nhận trong hóa đơn của khách hàng, và sau đó mức tối đa áp dụng.",
    "Tỷ lệ tối đa 2,5% cho đơn nhỏ.",  # a decimal binds to no integer parameter
])
def test_numbers_without_a_nearby_keyword_or_binding_are_dropped(text) -> None:
    assert _bound(text) == []


def test_a_keyword_in_another_sentence_does_not_count() -> None:
    assert _bound("Mức tối đa được nêu dưới đây.", "Chiết khấu 7% cho khách quen.") == []


def test_literals_are_ordered_by_source_chunk_sentence_and_offset() -> None:
    corpus = TellCorpus([
        Chunk("PRINCIPLES", "CANON", (Sentence("PRINCIPLES:00", "Tối đa 9%."),)),
        Chunk("KB:B", "KB", (Sentence("KB:B:00", "Tối đa 4% hoặc tối đa 6%."),)),
        Chunk("KB:A", "KB", (Sentence("KB:A:00", "Tối đa 3%."),)),
    ])
    assert [(x.chunk, x.value) for x in extract_literals(corpus)] == [
        ("PRINCIPLES", 9), ("KB:A", 3), ("KB:B", 4), ("KB:B", 6)]


def test_the_canon_tell_text_yields_its_discount_thresholds(pack) -> None:
    literals = extract_literals(TellCorpus.build(pack, []))
    assert [(x.sentence, x.bound) for x in literals] == [
        ("KB:KB-0003:03", ("discount_pct", 5)), ("KB:KB-0003:04", ("discount_pct", 8))]


def test_bound_literals_widen_the_domain_within_physical_limits(pack) -> None:
    domain = boundary_domain(pack, [("discount_pct", 0), ("discount_pct", 12),
                                    ("received", 100)])
    assert -1 not in domain["discount_pct"]
    assert {11, 12, 13} <= set(domain["discount_pct"]) and 101 in domain["received"]
    assert boundary_domain(pack) == boundary_domain(pack, [])
    assert all(s.received <= s.ordered + 1 for s in enumerate_scenarios(domain))


def _literal(parameter: str, value: int, offset: int = 0) -> TLiteral:
    return TLiteral(parameter, value, "", "trên", "KB", "KB:T", "KB:T:00", offset)


def test_t_boundary_scenarios_vary_one_parameter_of_the_anchor(pack) -> None:
    canon = canon_scenarios(pack)
    found = tboundary_scenarios(canon, [_literal("discount_pct", 7), _literal("received", 15)])
    # received 15 on the anchor is the canon scenario happy: skipped
    assert [s.id for s in found] == ["o15-r12-d6-g1", "o15-r12-d7-g1", "o15-r12-d8-g1",
                                     "o15-r14-d0-g1", "o15-r16-d0-g1"]
    assert all(s.signature() not in {c.signature() for c in canon.values()} for s in found)


def test_at_most_twelve_t_boundary_scenarios_are_made(pack) -> None:
    literals = [_literal("discount_pct", v, i) for i, v in enumerate(range(10, 60, 3))]
    assert len(tboundary_scenarios(canon_scenarios(pack), literals)) == 12
