from specdx.eval.freeze import UNTUNED, choose, model_size, thresholds
from specdx.eval.study import SplitResult
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
    assert c["jest"]["tau"] > 0.2  # the higher threshold that still keeps c1 removes x


def test_untuned_baselines_keep_threshold_zero() -> None:
    r = _result()
    assert "glad" in UNTUNED and "jest" not in UNTUNED
    assert thresholds(r, "glad") == [0.0]
    # A higher threshold would score better for glad (drops x), but spec §2.9 has none.
    assert choose(r)["glad"]["tau"] == 0.0
    assert choose(r)["jest"]["tau"] > 0.0


def test_choice_is_deterministic() -> None:
    assert choose(_result()) == choose(_result())
