"""The values the baselines spec fixes now, and the frozen choices once committed
(baselines spec §3.3, §7). Editing a frozen value after the test split ran breaks this."""

import hashlib
import json
from pathlib import Path

import pytest

from specdx.baselines import benchguard, whowhen
from specdx.baselines.api import TELL_CHARS
from specdx.eval import freeze, stats, study

ROOT = Path(__file__).parents[1]
DOC = ROOT / "experiments" / "p3-baselines-prereg.json"


def test_values_fixed_by_the_spec() -> None:
    assert whowhen.K_RUNS == 5 and TELL_CHARS == 8000
    assert study.CHECKPOINTS == (25, 50, 100, 150, 200)
    assert (stats.RESAMPLES, stats.SEED) == (10_000, 20261021)
    assert benchguard.SEVERITY == {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    assert benchguard.CONFIDENCE == {"CONFIRMED": 3, "LIKELY": 2, "POSSIBLE": 1}
    assert benchguard.MAX_TOKENS == 4096 and whowhen.MAX_TOKENS == 1024
    assert freeze.FIXED["benchguard_commit"].startswith("7d6a52e")
    assert whowhen.OPTIONS == {"temperature": 0.6, "top_p": 0.95, "seed": 1}
    assert freeze.FIXED["whowhen_options"] == whowhen.OPTIONS
    assert freeze.FIXED["untuned"] == ["b2_naive", "b3_2pl", "b5_mus", "dawid_skene", "glad"]


@pytest.mark.skipif(not DOC.exists(), reason="frozen by `specdx freeze-baselines`")
def test_the_frozen_choices_match_their_markdown_and_the_fixed_values() -> None:
    doc = json.loads(DOC.read_text(encoding="utf-8"))
    body = freeze.canonical_json(doc)
    md = DOC.with_suffix(".md").read_text(encoding="utf-8")
    assert hashlib.sha256(body.encode()).hexdigest() in md
    assert doc["fixed"] == freeze.FIXED
    assert set(doc["chosen"]) == set(freeze.REGISTRY)
