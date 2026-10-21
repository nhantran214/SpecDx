from specdx.cost import Cost, behaviours_cost, run_cost
from specdx.types import Behaviour, InterpreterId, Scenario, State

S = Scenario("happy", 15, 15, 0)
INTERP = InterpreterId("m", "f", 0, 1)


def _b(key: str, harness: dict, status: str = "ok") -> Behaviour:
    return Behaviour(key, INTERP, S, State.empty(), harness=harness, status=status)


def test_counted_runs_are_exact() -> None:
    b = _b("r1", {"s1": {"requests": 3, "prompt_tokens": 100, "eval_tokens": 20,
                         "duration_ms": 900, "tool_calls": 2, "nudges": 0}})
    assert run_cost(b) == Cost(3, 100, 20, 900, 0, True)


def test_old_runs_are_bounded_from_tool_calls_and_nudges() -> None:
    b = _b("r2", {"s1": {"tool_calls": 2, "nudges": 1}, "s2": {"tool_calls": 0, "nudges": 0}})
    c = run_cost(b)
    assert (c.requests, c.exact) == ((2 + 1 + 1) + (0 + 0 + 1), False)


def test_an_error_run_counts_one_error() -> None:
    assert run_cost(_b("r3", {}, status="error")).errors == 1


def test_probe_and_ddmin_runs_are_charged_at_the_mean_rounded_up() -> None:
    runs = [_b("a", {"s": {"requests": 2}}), _b("b", {"s": {"requests": 3}})]
    total = behaviours_cost(runs, extra_runs=4)
    assert total.requests == 5 + 4 * 3 and total.exact is False


def test_costs_add_and_exactness_is_sticky() -> None:
    assert (Cost(1, exact=True) + Cost(2, exact=False)) == Cost(3, exact=False)
