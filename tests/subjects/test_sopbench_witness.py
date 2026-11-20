"""The S3 execution witness: strict replay of a task's action graph."""

from __future__ import annotations

import pytest

from specdx.subjects.sopbench import DOMAINS
from specdx.subjects.sopbench.load import load_tasks
from specdx.subjects.sopbench.witness import (
    WitnessResult,
    _replay,
    load_raw_tasks,
    witness,
)

SIX = ("dmv", "healthcare", "hotel", "library", "online_market", "university")
CARD_GOALS = {
    "cancel_credit_card",
    "get_credit_card_info",
    "pay_bill_with_credit_card",
    "internal_check_credit_card_exist",
}
BANK_MISMATCHES = 6  # pinned from the first GREEN run (all cancel_credit_card)
BANK_ERRORS = 4


def _outcomes(domain: str) -> list[tuple[dict, WitnessResult]]:
    return [(t, witness(domain, t)) for t in load_raw_tasks(domain)]


def _label(task: dict) -> str:
    return "succeeds" if task["action_should_succeed"] else "refused"


def test_load_raw_tasks_order_matches_load_tasks():
    for domain in DOMAINS:
        raw = load_raw_tasks(domain)
        typed = load_tasks(domain)
        assert [t["user_goal"] for t in raw] == [t.user_goal for t in typed]


@pytest.mark.parametrize("domain", SIX)
def test_witness_matches_labels_on_six_domains(domain):
    for task, res in _outcomes(domain):
        assert res.outcome == _label(task)


def test_bank_mismatches_and_errors_are_credit_card_goals():
    results = _outcomes("bank")
    bad = [t["user_goal"] for t, r in results if r.outcome != _label(t)]
    errors = [r for _, r in results if r.outcome == "error"]
    assert set(bad) <= CARD_GOALS
    assert len(errors) == BANK_ERRORS
    assert len(bad) == BANK_MISMATCHES + BANK_ERRORS  # an error never equals a label


class _FakeStrict:
    def __init__(self, login_ok=True, boom=False):
        self.calls, self.login_ok, self.boom = [], login_ok, boom

    def login_user(self, username):
        self.calls.append("login_user")
        return self.login_ok

    def target(self, username):
        self.calls.append("target")
        if self.boom:
            raise KeyError("x")
        return True


_GRAPH = {
    "nodes": [["target", {"username": "u"}], "and", ["login_user", {"username": "u"}]],
    "connections": [[0, 1], [1, 2]],
}


def test_failed_login_stops_before_the_target():
    fake = _FakeStrict(login_ok=False)
    res = _replay(fake, _GRAPH, {"u": "x"}, {"login_user"}, "target")
    assert res.outcome == "refused"
    assert "target" not in fake.calls


def test_success_path_calls_target():
    fake = _FakeStrict()
    res = _replay(fake, _GRAPH, {"u": "x"}, {"login_user"}, "target")
    assert res.outcome == "succeeds"
    assert res.calls == ("login_user", "target")


def test_exception_is_an_error_outcome():
    fake = _FakeStrict(boom=True)
    res = _replay(fake, _GRAPH, {"u": "x"}, {"login_user"}, "target")
    assert res.outcome == "error"
    assert "KeyError" in res.error
