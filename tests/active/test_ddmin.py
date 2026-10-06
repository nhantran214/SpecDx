from specdx.active.ddmin import ddmin, localize_ambiguity
from specdx.spec.tell import Chunk, Sentence, TellCorpus


def test_ddmin_finds_the_minimal_failing_subset() -> None:
    assert sorted(ddmin(list(range(10)), lambda s: {3, 7} <= set(s))) == [3, 7]


def test_ddmin_returns_nothing_when_the_whole_does_not_fail() -> None:
    assert ddmin(list(range(5)), lambda s: False) == []


def test_ddmin_stops_at_the_test_budget_with_a_failing_superset() -> None:
    result = ddmin(list(range(64)), lambda s: {3, 7} <= set(s), max_tests=3)
    assert {3, 7} <= set(result)


def test_localisation_names_the_sentence_that_keeps_the_split() -> None:
    sentences = tuple(Sentence(f"A:{i:02d}", f"s{i}") for i in range(6))
    corpus = TellCorpus([Chunk("A", "KB", sentences)])
    found = localize_ambiguity(
        corpus, corpus.sentence_ids(), lambda c: "A:02" in c.sentence_ids()
    )
    assert found == ["A:02"]
