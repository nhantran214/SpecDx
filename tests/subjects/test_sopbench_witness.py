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
        assert [(t["user_goal"], bool(t["action_should_succeed"]), t["user_prompt"])
                for t in raw] == [(t.user_goal, t.should_succeed, t.user_prompt)
                                  for t in typed]


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
    mismatched = {t["user_goal"] for t, r in results
                  if r.outcome not in ("error", _label(t))}
    assert mismatched == {"cancel_credit_card"}


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


class _Scripted:
    """A strict system whose prerequisite answers are scripted per call."""

    def __init__(self, **answers):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[str] = []

    def __getattr__(self, name):
        if name.startswith("_") or name not in self.answers:
            raise AttributeError(name)

        def call(**kwargs):
            self.calls.append(name)
            queue = self.answers[name]
            return queue.pop(0) if len(queue) > 1 else queue[0]
        return call


def _graph(root_kind, leaves):
    nodes = [["target", {"username": "u"}], root_kind, *[[n, {"username": "u"}] for n in leaves]]
    return {"nodes": nodes,
            "connections": [[0, 1]] + [[1, i + 2] for i in range(len(leaves))]}


def test_an_or_node_needs_one_child():
    fake = _Scripted(a=[False], b=[True], target=[True])
    res = _replay(fake, _graph("or", ["a", "b"]), {"u": "x"}, {"a", "b"}, "target")
    assert res.outcome == "succeeds" and res.calls == ("a", "b", "target")
    none = _Scripted(a=[False], b=[False], target=[True])
    assert _replay(none, _graph("or", ["a", "b"]), {"u": "x"}, {"a", "b"},
                   "target").outcome == "refused"


def test_the_retry_pass_reruns_only_the_children_that_failed():
    # Task 6 Minor: the second pass of an "and" re-runs only what failed the first time, so a
    # prerequisite that already succeeded is not called (and its effect applied) twice.
    fake = _Scripted(a=[True], b=[False, True], target=[True])
    res = _replay(fake, _graph("and", ["a", "b"]), {"u": "x"}, {"a", "b"}, "target")
    assert res.outcome == "succeeds"
    assert res.calls == ("a", "b", "b", "target")


def test_a_leaf_outside_the_constraint_links_is_skipped():
    fake = _Scripted(a=[False], target=[True])
    res = _replay(fake, _graph("and", ["a"]), {"u": "x"}, set(), "target")
    assert res.outcome == "succeeds" and res.calls == ("target",)


def test_a_missing_user_known_key_refuses_the_target():
    # Ruling F-W: a leaf whose parameter the user does not know (no login credentials, say)
    # fails, so the target is refused: SOPBench's own semantics (no credentials, no action).
    fake = _Scripted(a=[True], target=[True])
    res = _replay(fake, _graph("and", ["a"]), {}, {"a"}, "target")
    assert res.outcome == "refused" and res.calls == ()


def test_an_imported_foreign_env_package_is_refused(monkeypatch):
    # Task 6 Minor: SOPBench's code lives in the top-level package `env`; another `env`
    # already imported is never silently used in its place.
    import sys
    import types

    from specdx.subjects.sopbench import witness as module

    monkeypatch.setitem(sys.modules, "env", types.ModuleType("env"))
    sys.modules["env"].__file__ = "/elsewhere/env/__init__.py"
    with pytest.raises(ImportError, match="env"):
        module._env()
