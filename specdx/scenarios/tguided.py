"""T-guided scenario synthesis (spec v2 §3.7): place scenarios at the thresholds T states.

Extraction runs on the Tell corpus of the specification under diagnosis, so a mutant uses
its own T. A literal is a number, with an optional unit, within 40 characters of a
comparison keyword in the same sentence. A percentage binds to discount_pct; a number
near a quantity word binds to received; anything else is dropped. Every bound literal v
adds v - 1, v and v + 1 to its parameter's domain, and T-boundary scenarios vary one
parameter of the anchor scenario to those values.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from specdx.scenarios.space import ANCHOR
from specdx.spec.tell import TellCorpus
from specdx.types import Scenario

WINDOW = 40  # characters between a number and a keyword or quantity word
MAX_TBOUNDARY = 12  # spec v2 §3.10
VIETNAMESE_KEYWORDS: tuple[str, ...] = (
    "tối đa", "tối thiểu", "không vượt quá", "vượt quá", "vượt", "trên", "dưới", "ít nhất",
    "nhiều nhất", "trần", "ngưỡng", "lớn hơn", "nhỏ hơn",
)
ENGLISH_KEYWORDS: tuple[str, ...] = (
    "at most", "at least", "exceed", "above", "below", "maximum", "minimum", "cap",
    "threshold", "more than", "less than",
)
QUANTITY_WORDS: tuple[str, ...] = ("số lượng", "đơn vị", "units", "quantity")
PERCENT_UNITS: frozenset[str] = frozenset({"%", "phần trăm"})
_NUMBER = re.compile(
    r"(?<![\w.,/\-:])(\d+(?:[.,]\d+)?)(?![\w/\-:])(\s*(?:%|phần trăm|đơn vị|cái|chiếc)(?!\w))?",
    re.IGNORECASE,
)


def _phrase(words: Iterable[str], suffix: str = "") -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(w).replace(r"\ ", r"\s+") for w in sorted(words, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives}){suffix}(?!\w)", re.IGNORECASE)


# English keywords also match their inflections (exceeds, exceeded, exceeding).
_KEYWORDS = (_phrase(VIETNAMESE_KEYWORDS), _phrase(ENGLISH_KEYWORDS, "(?:s|es|ed|ing)?"))
_QUANTITY = _phrase(QUANTITY_WORDS)


@dataclass(frozen=True)
class TLiteral:
    """A numeric threshold found in T, bound to a scenario parameter."""

    parameter: str
    value: int
    unit: str
    keyword: str
    source: str
    chunk: str
    sentence: str
    offset: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def bound(self) -> tuple[str, int]:
        return self.parameter, self.value


def _near(span: tuple[int, int], spans: Iterable[tuple[int, int, str]]) -> str | None:
    """The first marker within WINDOW characters of span, if any."""
    start, end = span
    for s, e, text in spans:
        gap = start - e if e <= start else s - end if s >= end else 0
        if gap <= WINDOW:
            return text
    return None


def _literals_in(text: str) -> list[tuple[str, int, str, str, int]]:
    """(parameter, value, unit, keyword, offset) for every bound literal of one sentence."""
    text = unicodedata.normalize("NFC", text)
    keywords = [(m.start(), m.end(), m.group(0).lower()) for p in _KEYWORDS
                for m in p.finditer(text)]
    quantities = [(m.start(), m.end(), m.group(0).lower()) for m in _QUANTITY.finditer(text)]
    found = []
    for m in _NUMBER.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").strip().lower()
        if not raw.isdigit():
            continue  # a decimal binds to no integer parameter
        span = (m.start(1), m.end())
        keyword = _near(span, sorted(keywords))
        if keyword is None:
            continue
        if unit in PERCENT_UNITS:
            parameter = "discount_pct"
        elif _near(span, sorted(quantities)) is not None:
            parameter = "received"
        else:
            continue
        found.append((parameter, int(raw), unit, keyword, m.start(1)))
    return found


def extract_literals(corpus: TellCorpus) -> list[TLiteral]:
    """Every bound literal of T, ordered by source, chunk, sentence and offset."""
    out = [
        TLiteral(parameter, value, unit, keyword, chunk.source, chunk.id, sentence.id, offset)
        for chunk in corpus.chunks for sentence in chunk.sentences
        for parameter, value, unit, keyword, offset in _literals_in(sentence.text)
    ]
    return sorted(out, key=lambda x: (x.source, x.chunk, x.sentence, x.offset))


def _physical(s: Scenario) -> bool:
    return (s.grn_count <= s.received <= s.ordered + 1) and 0 <= s.discount_pct <= 100


def tboundary_scenarios(
    canon: Mapping[str, Scenario], literals: Iterable[TLiteral], limit: int = MAX_TBOUNDARY
) -> list[Scenario]:
    """Scenarios that move one parameter of the anchor to v - 1, v or v + 1 of a literal,
    in literal order, skipping canon points and repeats, at most `limit` of them."""
    anchor = canon[ANCHOR]
    taken = {s.signature() for s in canon.values()}
    out: list[Scenario] = []
    for lit in literals:
        for value in (lit.value - 1, lit.value, lit.value + 1):
            params = anchor.params() | {lit.parameter: value}
            o, r, d, g = (params[k] for k in ("ordered", "received", "discount_pct",
                                              "grn_count"))
            s = Scenario(f"o{o}-r{r}-d{d}-g{g}", o, r, d, g)
            if _physical(s) and s.signature() not in taken:
                taken.add(s.signature())
                out.append(s)
                if len(out) == limit:
                    return out
    return out
