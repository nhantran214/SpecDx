import random

from specdx.infer.items import ItemView
from specdx.infer.sensitivity import (
    Reading,
    benjamini_hochberg,
    eligible,
    paraphrase_sensitivity,
    permutation_test,
)

MODELS = ("m1", "m2")


def _readings(klass_of) -> list[Reading]:
    return [Reading(m, v, klass_of(m, v, s)) for m in MODELS for v in (0, 1, 2) for s in (1, 2, 3)]


def test_a_planted_variant_effect_is_detected() -> None:
    stat, p = permutation_test(_readings(lambda m, v, s: "X" if v == 0 else "Y"), seed=1)
    assert stat > 0.6 and p < 0.01


def test_the_type_one_error_is_controlled_under_the_null() -> None:
    rng = random.Random(7)
    pvalues = {}
    for i in range(200):
        readings = _readings(lambda m, v, s: rng.choice("XY"))
        if eligible(readings):
            pvalues[f"i{i}"] = permutation_test(readings, seed=1, item=f"i{i}")[1]
    rejected = sum(p <= 0.1 for p in pvalues.values()) / len(pvalues)
    assert len(pvalues) >= 190 and rejected <= 0.1 + 0.05
    assert len(benjamini_hochberg(pvalues, 0.1)) / len(pvalues) <= 0.1 + 0.05


def test_the_test_is_deterministic_for_a_seed() -> None:
    readings = _readings(lambda m, v, s: "X" if (v + s) % 2 else "Y")
    assert permutation_test(readings, seed=3, item="a") == permutation_test(readings, seed=3,
                                                                            item="a")


def test_benjamini_hochberg_steps_up() -> None:
    p = {"a": 0.001, "b": 0.01, "c": 0.03, "d": 0.2, "e": 0.5}
    assert benjamini_hochberg(p, 0.1) == {"a", "b", "c"}
    assert benjamini_hochberg({"a": 0.05, "b": 0.05}, 0.1) == {"a", "b"}
    assert benjamini_hochberg({}, 0.1) == frozenset()


def test_only_acted_views_of_competent_models_on_eligible_items_are_tested() -> None:
    def view(item, model, variant, klass, acted=True):
        return ItemView(item, f"{item}|{model}|{variant}|{klass}", model, "f", klass,
                        frozenset(), (), acted=acted, variant=variant)

    views = [view("split", m, v, "X" if v == 0 else "Y") for m in ("m1", "m2", "weak")
             for v in (0, 1, 2)]
    views += [view("split", "m1", 1, "Y", acted=False), view("few", "m1", 0, "X"),
              view("few", "m1", 1, "Y")]
    tested = paraphrase_sensitivity(views, {"m1": 0.9, "m2": 0.6, "weak": 0.2},
                                    ["split", "few"], seed=1)
    assert set(tested) == {"split"}  # "few" has 2 behaviours, fewer than 6
    assert not eligible([Reading("m1", v, "X") for v in (0, 1, 2)] * 2)  # one class
    assert not eligible([Reading("m1", 0, k) for k in "XYXYXY"])  # one variant
