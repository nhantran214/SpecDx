import random

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
