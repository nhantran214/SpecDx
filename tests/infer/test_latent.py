import random

import pytest

from specdx.infer.items import ItemView, observations
from specdx.infer.latent import RARE, Observation, fit

FAMILY = {"g1": "fa", "g2": "fb", "g3": "fc", "b1": "fa", "b2": "fb", "b3": "fc"}
GOOD = ("g1", "g2", "g3")
RUNS = 4


def _obs(item, model, i, klass):
    return Observation(item, f"{item}|{model}|{i}", model, FAMILY[model], klass)


def _clean_item(rng, item, truth):
    out = []
    for model in FAMILY:
        for i in range(RUNS):
            if model in GOOD:
                klass = truth if rng.random() < 0.9 else f"err-{rng.randrange(6)}"
            else:
                klass = f"err-{rng.randrange(6)}"
            out.append(_obs(item, model, i, klass))
    return out


def _ambiguous_item(rng, item):
    out = []
    for model in FAMILY:
        for i in range(RUNS):
            klass = ("X" if i % 2 == 0 else "Y") if model in GOOD else f"err-{rng.randrange(6)}"
            out.append(_obs(item, model, i, klass))
    return out


def _dataset():
    rng = random.Random(0)
    obs = []
    for j in range(8):
        obs += _clean_item(rng, f"clean{j}", f"A{j}")
    obs += _ambiguous_item(rng, "amb")
    return obs


def test_clean_items_recover_their_single_admissible_class() -> None:
    post = fit(_dataset())
    for j in range(8):
        item = post.items[f"clean{j}"]
        assert item.map_set == frozenset({f"A{j}"})
        assert item.p_ambiguous < 0.2


def test_an_ambiguous_item_is_recognised() -> None:
    item = fit(_dataset()).items["amb"]
    assert item.p_ambiguous > 0.5
    assert item.p_member["X"] > 0.5 and item.p_member["Y"] > 0.5


def test_competence_separates_good_from_bad_models() -> None:
    theta = fit(_dataset()).theta
    assert min(theta[m] for m in GOOD) > max(theta[m] for m in FAMILY if m not in GOOD)


def test_responsibility_is_high_for_admissible_outputs() -> None:
    post = fit(_dataset())
    item = post.items["clean0"]
    good = [o for o in _dataset() if o.item == "clean0" and o.klass == "A0"]
    assert all(item.responsibility[o.run_key] > 0.8 for o in good)


def test_a_family_of_bad_models_gets_a_lower_effect() -> None:
    family = {"g1": "fa", "g2": "fa", "b1": "fb", "b2": "fb"}
    rng = random.Random(1)
    obs = []
    for j in range(6):
        for model, fam in family.items():
            for i in range(RUNS):
                good = model.startswith("g")
                klass = f"A{j}" if good and rng.random() < 0.9 else f"err-{rng.randrange(6)}"
                obs.append(Observation(f"i{j}", f"i{j}|{model}|{i}", model, fam, klass))
    effect = fit(obs).family_effect
    assert effect["fa"] > effect["fb"]


def test_classes_beyond_k_max_are_pooled_and_never_admissible() -> None:
    obs = [Observation("i", f"r{n}", "m", "f", klass)
           for n, klass in enumerate(["a"] * 5 + ["b"] * 4 + ["c"] * 3)]
    item = fit(obs, k_max=2).items["i"]
    assert item.classes == ("a", "b")
    assert all(RARE not in s for s in item.set_posterior)


def test_no_observations_give_an_empty_posterior() -> None:
    post = fit([])
    assert post.theta == {} and post.items == {}


def test_theta_is_the_beta_map_of_the_expected_admissible_behaviours() -> None:
    obs = _dataset()
    post = fit(obs)
    for model in FAMILY:
        mine = [o for o in obs if o.model == model]
        admissible = sum(post.items[o.item].responsibility[o.run_key] for o in mine)
        assert post.theta[model] == pytest.approx((admissible + 6) / (len(mine) + 8), abs=1e-4)


def test_a_listed_model_that_never_acted_keeps_the_prior_mode() -> None:
    post = fit(_dataset(), models={"silent": "fz"})
    assert post.theta["silent"] == pytest.approx(0.75)
    assert fit([], models={"silent": "fz"}).theta == {"silent": pytest.approx(0.75)}


def _p1_shaped(seed: int = 3) -> list[ItemView]:
    """After the P1 measurements (72 items, 9 runs per model and item): two qwen models act
    in about a quarter of their runs and then mostly agree; their failed steps scatter over
    several failure modes. llama acts in none of its runs and always fails the same way."""
    rng = random.Random(seed)
    views = []
    for j in range(72):
        for model, family in (("qwen3", "qwen"), ("qwen2.5", "qwen"), ("llama", "llama")):
            for r in range(9):
                key = f"{model}|{j}|{r}"
                acting = model != "llama" and rng.random() < 0.25
                if acting:
                    klass = f"A{j}" if rng.random() < 0.9 else f"noise-{rng.randrange(4)}"
                elif model == "llama":
                    klass = "stuck"
                else:
                    klass = f"fail-{rng.randrange(6)}"
                views.append(ItemView(f"i{j}", key, model, family, klass, frozenset(), (),
                                      acted=acting))
    return views


def test_a_model_that_never_acts_cannot_pull_the_acting_models_below_the_prior_mean() -> None:
    views = _p1_shaped()
    population = {"qwen3": "qwen", "qwen2.5": "qwen", "llama": "llama"}
    unfiltered = fit([Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views])
    assert unfiltered.theta["qwen3"] < 0.7 < unfiltered.theta["llama"]  # the P1 inversion
    post = fit(observations(views), models=population)
    assert post.theta["qwen3"] >= 0.7 and post.theta["qwen2.5"] >= 0.7
    assert post.theta["qwen3"] >= post.theta["llama"]
