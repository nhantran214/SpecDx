import random

from specdx.active.search import (
    CellBandit,
    NormalGamma,
    information_gain,
    should_escalate,
)
from specdx.infer.items import ItemView
from specdx.types import Scenario


def test_no_change_means_no_information() -> None:
    assert information_gain({"a": 0.3}, {"a": 0.3}) == 0.0
    assert information_gain({}, {"a": 0.9}) > information_gain({}, {"a": 0.2})


def test_the_normal_gamma_mean_follows_the_data() -> None:
    ng = NormalGamma()
    for _ in range(20):
        ng.update(2.0)
    assert abs(ng.mu - 2.0) < 0.2


def test_untried_cells_come_first_then_the_best_cell() -> None:
    rng = random.Random(0)
    a = Scenario("a", 15, 12, 0)
    b = Scenario("b", 15, 15, 8)
    c = Scenario("c", 15, 11, 0)
    bandit = CellBandit(5)
    first = bandit.choose([a, b], set(), rng)
    bandit.update(first, 0.0)
    second = bandit.choose([a, b], {first.signature()}, rng)
    assert {first.id, second.id} == {"a", "b"}
    bandit.update(second, 0.0)
    for _ in range(10):
        bandit.update(a, 5.0)
    assert bandit.choose([b, c], {a.signature()}, rng).id == "c"


def test_the_bandit_is_exhausted_when_every_open_cell_looks_useless() -> None:
    a = Scenario("a", 15, 12, 0)
    bandit = CellBandit(5)
    assert not bandit.exhausted([a], set(), 0.01)
    bandit.update(a, 0.0)
    assert bandit.exhausted([a], set(), 0.01)
    assert bandit.exhausted([a], {a.signature()}, 0.01)


def _view(item, klass):
    return ItemView(item, "r", "m", "f", klass, frozenset(), ())


def test_escalate_on_a_new_correction_set_or_a_pilot_split() -> None:
    seen = {frozenset()}
    agree = [_view("s|1", "K"), _view("s|1", "K")]
    split = [_view("s|1", "K"), _view("s|1", "L")]
    assert not should_escalate(agree, [frozenset(), frozenset()], seen)
    assert should_escalate(agree, [frozenset({"c"}), frozenset()], seen)
    assert should_escalate(split, [frozenset(), frozenset()], seen)
