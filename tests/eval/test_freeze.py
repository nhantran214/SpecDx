import hashlib
import json

import pytest

from specdx.baselines.api import REGISTRY
from specdx.eval.freeze import (
    UNTUNED,
    canonical_json,
    choose,
    model_size,
    prereg,
    prereg_markdown,
    study_arms,
    thresholds,
)
from specdx.eval.study import Arm, SplitResult
from specdx.mutate.operators import GroundTruth
from specdx.types import Defect


def _d(loc, s):
    return Defect("hidden", (loc,), (), (), s)


def _result():
    truths = {"H1": GroundTruth("H1", "hidden", frozenset({"c1"}), "clause")}
    arms = {
        "whowhen@qwen2.5:14b/all_at_once": {"base": [], "H1": [_d("c1", 0.4)]},
        "whowhen@gpt-oss:20b/all_at_once": {"base": [], "H1": [_d("c1", 0.4)]},
        "whowhen@gpt-oss:20b/step_by_step": {"base": [], "H1": [_d("zz", 0.9)]},
        "jest": {"base": [], "H1": [_d("c1", 0.7), _d("x", 0.2)]},
        "glad": {"base": [], "H1": [_d("c1", 0.7), _d("x", 0.2)]},
    }
    return SplitResult("dev", truths, {"H1"}, {"specdx": {"base": [], "H1": []}, **arms},
                       {}, {}, {})


def test_model_size_reads_the_tag() -> None:
    assert model_size("qwen2.5:14b") == 14 and model_size("qwen3:30b-a3b") == 30
    assert model_size("weird") == float("inf")


def test_thresholds_are_deciles_plus_zero() -> None:
    assert thresholds(_result(), "jest")[0] == 0.0


def test_ties_go_to_the_smaller_model_then_the_higher_threshold() -> None:
    c = choose(_result())
    assert c["whowhen"]["model"] == "qwen2.5:14b"
    # The highest tied threshold: the grid's top value that still keeps c1 (0.7) and drops x.
    keeps_c1 = [t for t in thresholds(_result(), "jest") if t <= 0.7]
    assert c["jest"]["tau"] == max(keeps_c1) > 0.2


def test_untuned_baselines_keep_threshold_zero() -> None:
    r = _result()
    assert "glad" in UNTUNED and "jest" not in UNTUNED
    assert thresholds(r, "glad") == [0.0]
    # A higher threshold would score better for glad (drops x), but spec §2.9 has none.
    assert choose(r)["glad"]["tau"] == 0.0
    assert choose(r)["jest"]["tau"] > 0.0


def test_choice_is_deterministic() -> None:
    assert choose(_result()) == choose(_result())


def test_a_smaller_model_wins_over_a_larger_one_with_a_higher_threshold() -> None:
    truths = {"H1": GroundTruth("H1", "hidden", frozenset({"c1"}), "clause")}
    arms = {"whowhen@gpt-oss:20b/all_at_once": {"base": [], "H1": [_d("c1", 0.99)]},
            "whowhen@qwen2.5:14b/all_at_once": {"base": [], "H1": [_d("c1", 0.5)]}}
    r = SplitResult("dev", truths, {"H1"}, arms, {}, {}, {})
    c = choose(r)["whowhen"]
    assert c["model"] == "qwen2.5:14b" and c["tau"] == 0.5  # lower tau, but the smaller model


def _full():
    r = _result()
    for name in REGISTRY:
        if not any(a.partition("@")[0] == name for a in r.defects):
            r.defects[name] = {"base": [], "H1": []}
    return r


def test_prereg_records_the_cache_and_omits_it_when_absent(tmp_path) -> None:
    src = tmp_path / "dev.json"
    src.write_text("{}", encoding="utf-8")
    cache = tmp_path / "baseline-llm.jsonl"
    cache.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    doc = prereg(_full(), src, cache)
    assert doc["baseline_cache"] == {
        "lines": 2, "md5": hashlib.md5(b'{"a": 1}\n{"b": 2}\n').hexdigest()}
    assert doc["source_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert "baseline_cache" not in prereg(_full(), src, None)
    assert "baseline_cache" not in prereg(_full(), src, tmp_path / "missing.jsonl")


def test_markdown_prints_the_hash_of_the_canonical_json(tmp_path) -> None:
    src = tmp_path / "dev.json"
    src.write_text("{}", encoding="utf-8")
    doc = prereg(_full(), src)
    md = prereg_markdown(doc)
    assert hashlib.sha256(canonical_json(doc).encode()).hexdigest() in md
    assert json.loads(canonical_json(doc)) == doc


def test_prereg_refuses_an_incomplete_choice(tmp_path) -> None:
    src = tmp_path / "dev.json"
    src.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="benchguard"):
        prereg(_result(), src)


def test_study_arms_are_sorted_by_name_with_float_tau(tmp_path) -> None:
    src = tmp_path / "dev.json"
    src.write_text("{}", encoding="utf-8")
    doc = prereg(_full(), src)
    arms = study_arms(doc)
    assert [a.name for a, _ in arms] == sorted(REGISTRY)
    by = {a.name: (a, t) for a, t in arms}
    assert by["whowhen"][0] == Arm("whowhen", "qwen2.5:14b", "all_at_once")
    assert all(isinstance(t, float) for _, t in arms)
