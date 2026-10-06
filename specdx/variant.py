"""A specification under test: a canon pack directory plus its Tell corpus."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from specdx.db import pack_digest
from specdx.spec.tell import TellCorpus


@dataclass(frozen=True)
class SpecVariant:
    name: str
    pack: Path
    corpus: TellCorpus

    def digest(self) -> str:
        return hashlib.sha256(
            (pack_digest(self.pack) + self.corpus.digest()).encode()
        ).hexdigest()
