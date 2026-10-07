from specdx.active.ddmin import OutOfBudget, ddmin, localize_ambiguity
from specdx.spec.tell import Chunk, Sentence, TellCorpus


def test_ddmin_finds_the_minimal_failing_subset() -> None:
    assert sorted(ddmin(list(range(10)), lambda s: {3, 7} <= set(s))) == [3, 7]


def test_ddmin_is_unlocalised_when_the_whole_does_not_fail() -> None:
    assert ddmin(list(range(5)), lambda s: False) is None


def test_ddmin_is_unlocalised_when_the_empty_set_already_fails() -> None:
    calls = []

    def fails(s):
        calls.append(tuple(s))
        return True

    assert ddmin(list(range(8)), fails) is None
    assert calls == [()]


def test_a_ddmin_budget_that_runs_out_leaves_the_result_unlocalised() -> None:
    calls = []

    def fails(s):
        calls.append(tuple(s))
        return {3, 7} <= set(s)

    assert ddmin(list(range(64)), fails, max_tests=3) is None
    assert len(calls) == 3


def test_an_oracle_out_of_budget_leaves_the_result_unlocalised() -> None:
    calls = []

    def fails(s):
        if len(calls) == 4:
            raise OutOfBudget
        calls.append(tuple(s))
        return {3, 7} <= set(s)

    assert ddmin(list(range(64)), fails) is None


def test_ddmin_never_re_probes_a_subset() -> None:
    calls = []

    def fails(s):
        calls.append(tuple(s))
        return {3, 7} <= set(s)

    result = ddmin(list(range(64)), fails)
    assert len(calls) == len(set(calls))
    assert sorted(result) == [3, 7]


def test_localisation_names_the_sentence_that_keeps_the_split() -> None:
    sentences = tuple(Sentence(f"A:{i:02d}", f"s{i}") for i in range(6))
    corpus = TellCorpus([Chunk("A", "KB", sentences)])
    found = localize_ambiguity(
        corpus, corpus.sentence_ids(), lambda c: "A:02" in c.sentence_ids()
    )
    assert found == ["A:02"]


def test_a_split_that_persists_without_the_pool_is_unlocalised() -> None:
    sentences = tuple(Sentence(f"A:{i:02d}", f"s{i}") for i in range(6))
    corpus = TellCorpus([Chunk("A", "KB", sentences)])
    assert localize_ambiguity(corpus, corpus.sentence_ids(), lambda c: True) is None
