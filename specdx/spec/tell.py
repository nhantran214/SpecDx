"""The Tell side T (spec §2.1): requirement documents, KB articles and canon principles.

Sentence ids are stable: removing or replacing a sentence never renumbers the others,
so a localisation result stays meaningful across corpus edits.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tonomyx.canon.pack import CanonPack

CHUNK_WORDS = 300
CHUNKING_VERSION = 1  # Bump whenever chunking or sentence splitting changes
PDF_DOCS = {"1.": "DEMO", "2.": "BA", "3.": "SAD"}
_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    sentences: tuple[Sentence, ...]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def _norm(text: str) -> str:
    """Exact-match normal form: composed Unicode, case-folded, single spaces."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _fold(text: str) -> str:
    """Search normal form: accents stripped, as TonomyX's search_kb does."""
    stripped = unicodedata.normalize("NFD", text)
    return "".join(c for c in stripped if unicodedata.category(c) != "Mn").casefold()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_fold(text))


def _make_chunk(chunk_id: str, source: str, texts: Iterable[str]) -> Chunk:
    return Chunk(
        chunk_id, source,
        tuple(Sentence(f"{chunk_id}:{i:02d}", t) for i, t in enumerate(texts)),
    )


def chunk_document(doc: str, text: str, words: int = CHUNK_WORDS) -> list[Chunk]:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > words:
            groups.append(current)
            current, count = [], 0
        current.append(para)
        count += n
    if current:
        groups.append(current)
    return [
        _make_chunk(f"{doc}:{i:03d}", doc, [s for para in group for s in split_sentences(para)])
        for i, group in enumerate(groups)
    ]


def pdf_chunks(pdf_dir: Path, cache_dir: Path) -> list[Chunk]:
    """Chunks of the three requirement PDFs, extracted once and cached by file hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[Chunk] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = PDF_DOCS[pdf.name[:2]]
        pdf_bytes = pdf.read_bytes()
        cache_key = hashlib.sha256(
            pdf_bytes + f"|{doc}|{CHUNK_WORDS}|{CHUNKING_VERSION}".encode()
        ).hexdigest()
        cached = cache_dir / f"{cache_key}.json"
        if cached.is_file():
            out += TellCorpus.from_json(cached.read_text(encoding="utf-8")).chunks
            continue
        text = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
        ).stdout
        chunks = chunk_document(doc, text)
        cached.write_text(TellCorpus(chunks).to_json(), encoding="utf-8")
        out += chunks
    return out


class Bm25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg = (sum(self.lengths) / len(docs)) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Sequence[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.lengths, strict=True):
            s = 0.0
            for term in query:
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                    s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


class TellCorpus:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks: tuple[Chunk, ...] = tuple(chunks)
        self._index: Bm25 | None = None
        self._sentences = {s.id: s.text for c in self.chunks for s in c.sentences}

    @classmethod
    def build(cls, pack: CanonPack, pdf: Sequence[Chunk]) -> TellCorpus:
        kb = [
            _make_chunk(f"KB:{d.kb_id}", "KB", split_sentences(f"{d.title}.\n{d.body}"))
            for d in pack.kb.values()
        ]
        principles = _make_chunk("PRINCIPLES", "CANON", [str(p) for p in pack.meta["nguyen_tac"]])
        return cls([*pdf, *kb, principles])

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def sentence_ids(self, chunk_ids: Iterable[str] | None = None) -> list[str]:
        wanted = None if chunk_ids is None else set(chunk_ids)
        return [s.id for c in self.chunks if wanted is None or c.id in wanted for s in c.sentences]

    def sentence(self, sentence_id: str) -> str:
        return self._sentences[sentence_id]

    @staticmethod
    def chunk_of(sentence_id: str) -> str:
        return sentence_id.rsplit(":", 1)[0]

    def find(self, text: str) -> list[str]:
        target = _norm(text)
        return [sid for sid, t in self._sentences.items() if _norm(t) == target]

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if self._index is None:
            self._index = Bm25([_tokens(c.text) for c in self.chunks])
        scores = self._index.scores(_tokens(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]

    def without(self, sentence_ids: Iterable[str]) -> TellCorpus:
        drop = set(sentence_ids)
        return TellCorpus([
            Chunk(c.id, c.source, tuple(s for s in c.sentences if s.id not in drop))
            for c in self.chunks
        ])

    def replace(self, sentence_id: str, text: str) -> TellCorpus:
        return TellCorpus([
            Chunk(c.id, c.source, tuple(
                Sentence(s.id, text) if s.id == sentence_id else s for s in c.sentences
            ))
            for c in self.chunks
        ])

    def with_chunk(self, chunk: Chunk) -> TellCorpus:
        return TellCorpus([*self.chunks, chunk])

    def to_json(self) -> str:
        return json.dumps(
            [{"id": c.id, "source": c.source, "sentences": [[s.id, s.text] for s in c.sentences]}
             for c in self.chunks],
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> TellCorpus:
        return cls([
            Chunk(c["id"], c["source"], tuple(Sentence(i, t) for i, t in c["sentences"]))
            for c in json.loads(text)
        ])
