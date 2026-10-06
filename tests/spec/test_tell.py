import unicodedata

import pytest

from specdx.config import BASE_PACK, PDF_DIR
from specdx.spec import tell
from specdx.spec.tell import Bm25, Chunk, Sentence, TellCorpus, pdf_chunks, split_sentences
from specdx.variant import SpecVariant


@pytest.fixture(scope="module")
def corpus(pack, tmp_path_factory) -> TellCorpus:
    return TellCorpus.build(pack, pdf_chunks(PDF_DIR, tmp_path_factory.mktemp("tell")))


def test_the_corpus_holds_every_source(corpus) -> None:
    sources = {c.source for c in corpus.chunks}
    assert sources == {"DEMO", "BA", "SAD", "KB", "CANON"}
    assert len([c for c in corpus.chunks if c.source == "BA"]) > 50


def test_sentences_split_on_semicolons_full_stops_and_newlines() -> None:
    assert split_sentences("a; b.\nc! d") == ["a;", "b.", "c!", "d"]


def test_the_discount_cap_sentence_is_found(corpus) -> None:
    ids = corpus.find("**mức tối đa 5%**.")
    assert ids and all(TellCorpus.chunk_of(i) == "KB:KB-0003" for i in ids)


def test_find_matches_across_unicode_normal_forms(corpus) -> None:
    nfd = unicodedata.normalize("NFD", "**mức tối đa 5%**.")
    assert corpus.find(nfd) == corpus.find("**mức tối đa 5%**.")


def test_search_ranks_the_pricing_article_for_a_discount_query(corpus) -> None:
    assert "KB:KB-0003" in [c.id for c in corpus.search("chiết khấu tối đa", k=5)]


def test_removing_a_sentence_keeps_other_ids_and_changes_the_digest(corpus) -> None:
    target = corpus.find("**mức tối đa 5%**.")[0]
    others = [i for i in corpus.sentence_ids(["KB:KB-0003"]) if i != target]
    smaller = corpus.without([target])
    assert target not in smaller.sentence_ids()
    assert all(smaller.sentence(i) == corpus.sentence(i) for i in others)
    assert smaller.digest() != corpus.digest()


def test_replace_and_with_chunk(corpus) -> None:
    target = corpus.find("**mức tối đa 5%**.")[0]
    assert corpus.replace(target, "**mức tối đa hợp lý**.").sentence(target) == (
        "**mức tối đa hợp lý**."
    )
    extra = Chunk("KB:KB-0009", "KB", (Sentence("KB:KB-0009:00", "x."),))
    assert "KB:KB-0009:00" in corpus.with_chunk(extra).sentence_ids()


def test_json_round_trip(corpus) -> None:
    assert TellCorpus.from_json(corpus.to_json()).digest() == corpus.digest()


def test_bm25_prefers_the_document_with_the_term() -> None:
    index = Bm25([["chiet", "khau"], ["giao", "hang"]])
    scores = index.scores(["chiet"])
    assert scores[0] > scores[1]


def test_a_variant_digest_covers_pack_and_corpus(corpus) -> None:
    a = SpecVariant("base", BASE_PACK, corpus)
    b = SpecVariant("base", BASE_PACK, corpus.without(corpus.sentence_ids()[:1]))
    assert a.digest() != b.digest()


def test_chunking_version_invalidates_cache(tmp_path) -> None:
    # First call with version 1
    cache = tmp_path / "cache"
    pdf_chunks(PDF_DIR, cache)
    files_v1 = list(cache.glob("*.json"))
    assert len(files_v1) == 3

    # Monkeypatch version to 2
    monkeypatch_version = tell.CHUNKING_VERSION
    tell.CHUNKING_VERSION = 2
    try:
        # Second call with version 2 should create new cache files
        pdf_chunks(PDF_DIR, cache)
        files_after = list(cache.glob("*.json"))
        assert len(files_after) == 6  # 3 files per version

        # Third call with version 2 should not create new files
        count_before = len(list(cache.glob("*.json")))
        pdf_chunks(PDF_DIR, cache)
        count_after = len(list(cache.glob("*.json")))
        assert count_before == count_after
    finally:
        tell.CHUNKING_VERSION = monkeypatch_version
