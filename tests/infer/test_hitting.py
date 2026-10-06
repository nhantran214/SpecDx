from specdx.infer.hitting import soft_hitting_sets


def _f(*names):
    return frozenset(names)


def test_a_clause_every_run_violates_is_a_singleton() -> None:
    sets = [(_f("a", "x"), 1.0), (_f("a"), 1.0), (_f("a", "y"), 1.0)]
    assert soft_hitting_sets(sets)[0] == (_f("a"), 1.0)


def test_runs_split_between_two_clauses_give_a_pair() -> None:
    sets = [(_f("a"), 1.0)] * 5 + [(_f("b"), 1.0)] * 5
    assert soft_hitting_sets(sets) == [(_f("a", "b"), 1.0)]


def test_supersets_of_a_found_set_are_not_reported() -> None:
    sets = [(_f("a", "b"), 1.0)] * 4
    found = [h for h, _ in soft_hitting_sets(sets)]
    assert _f("a") in found and _f("b") in found and _f("a", "b") not in found


def test_enough_clean_runs_mean_no_hitting_set() -> None:
    sets = [(_f("a"), 1.0)] * 9 + [(_f(), 1.0)] * 1
    assert soft_hitting_sets(sets, tau=0.05) == []


def test_low_weight_runs_may_be_missed_within_tau() -> None:
    sets = [(_f("a"), 1.0)] * 20 + [(_f("z"), 0.5)]
    assert soft_hitting_sets(sets, tau=0.05)[0][0] == _f("a")


def test_no_weight_means_nothing() -> None:
    assert soft_hitting_sets([(_f("a"), 0.0)]) == []
