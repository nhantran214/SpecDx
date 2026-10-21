import hashlib
from types import SimpleNamespace

from specdx.baselines.whowhen import failure_log, parse_verdict, sample_runs
from specdx.types import Behaviour, InterpreterId, Scenario, State


def _h(k):
    return hashlib.sha256(k.encode()).hexdigest()


def test_sampling_takes_at_most_k_per_clause_and_reuses_runs() -> None:
    violated = {f"r{i}": frozenset({"a", "b"}) for i in range(8)}
    violated["solo"] = frozenset({"c"})
    picked = sample_runs(violated, set(violated), k=5)
    assert all(len(v) <= 5 for v in picked.values())
    assert picked["a"] == picked["b"]  # reuse: the same five runs serve both clauses
    assert picked["a"] == sorted((f"r{i}" for i in range(8)), key=_h)[:5]
    assert picked["c"] == ["solo"]


def test_sampling_skips_error_runs() -> None:
    picked = sample_runs({"r1": frozenset({"a"})}, ok=set(), k=5)
    assert picked == {}


def test_the_log_follows_step_order_with_claims_refusals_and_rows() -> None:
    steps = [SimpleNamespace(id="S1"), SimpleNamespace(id="S2")]
    b = Behaviour("k", InterpreterId("m", "f", 0, 1), Scenario("happy", 15, 15, 0),
                  State({"po": [{"ma_po": "PO-1", "trang_thai": "draft"}]}, steps={"S2": "done"}),
                  refused=(("S2", "write_draft"),), claims={"S2": "made PO-1", "S1": "checked"})
    log = failure_log(b, steps, {"S2": frozenset({"po"})})
    assert [e["name"] for e in log] == ["S1", "S2"]
    assert "write_draft" in log[1]["content"] and "PO-1" in log[1]["content"]


def test_verdict_parsing() -> None:
    v = frozenset({"inv:po-15"})
    assert parse_verdict("Verdict: agent_error", v) == "agent_error"
    assert parse_verdict("blah\nVerdict: spec_defect inv:po-15\n", v) == "inv:po-15"
    assert parse_verdict("Verdict: spec_defect inv:other", v) is None  # not a failed check
    assert parse_verdict("I think the agent was wrong", v) is None


def test_verdict_parsing_strips_quotes_brackets_and_trailing_punctuation() -> None:
    v = frozenset({"inv:po-15"})
    for written in ("`inv:po-15`", "`inv:po-15`.", "<inv:po-15>", '"inv:po-15",', "'inv:po-15';",
                    "inv:po-15.", "inv:po-15:", "inv:po-15)", "`inv:po-15`)."):
        assert parse_verdict(f"Verdict: spec_defect {written}", v) == "inv:po-15", written


class _Scripted:
    """A gateway answering from a function of the user prompt; records every call."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def chat(self, model, messages, **kw):
        from specdx.baselines.llm import Reply
        from specdx.cost import Cost

        self.calls.append((messages, kw))
        return Reply(self.answer(messages[-1]["content"]), Cost(1), "ok")


def _run(gateway, strategy):
    from specdx.baselines import whowhen

    return whowhen.REGISTRY["whowhen"].run(whowhen._toy_input(gateway), model="m",
                                           strategy=strategy)


def test_an_unparsable_reply_is_an_error_not_a_finding() -> None:
    out = _run(_Scripted(lambda p: "no idea"), "all_at_once")
    assert out.defects == [] and out.extra["unparsed"] >= 1


def test_a_spec_defect_verdict_scores_the_clause_for_every_strategy() -> None:
    def answer(prompt):
        if "Verdict:" in prompt:
            return "Verdict: spec_defect inv:x"
        if "Step Number" in prompt:
            return "Agent Name: S1\nStep Number: 0\nReason for Mistake: none"
        if "Respond ONLY in the format: 1. Yes/No" in prompt:
            return "1. Yes.\n2. Reason: bad"
        return "upper half"

    for strategy in ("all_at_once", "step_by_step", "binary_search"):
        gw = _Scripted(answer)
        out = _run(gw, strategy)
        assert [(d.type, d.location, d.score) for d in out.defects] == \
            [("hidden", ("inv:x",), 1.0)], strategy
        assert out.cost.requests == len(gw.calls) >= 2
        # The reference's open-model generation settings (Lib/local_model.py), seed 1.
        assert all(kw["max_tokens"] == 1024
                   and kw["options"] == {"temperature": 0.6, "top_p": 0.95, "seed": 1}
                   for _, kw in gw.calls)


def test_step_by_step_with_no_flagged_step_makes_no_prediction() -> None:
    gw = _Scripted(lambda p: "Verdict: spec_defect inv:x" if "Verdict:" in p
                   else "1. No.\n2. Reason: fine")
    out = _run(gw, "step_by_step")
    assert out.defects == [] and out.extra["unparsed"] == 1 and out.extra["judged"] == 0
    assert len(gw.calls) == 2  # one per step, and no verdict call


def test_the_verdict_prompt_shows_the_clamped_step() -> None:
    gw = _Scripted(lambda p: "Verdict: agent_error" if "Verdict:" in p else "Step Number: 7")
    _run(gw, "all_at_once")
    verdict = gw.calls[-1][0][-1]["content"]
    assert "Decisive step 1 by 'S2'" in verdict and "step 7" not in verdict


def test_an_agent_error_verdict_is_not_a_defect() -> None:
    out = _run(_Scripted(lambda p: "Verdict: agent_error" if "Verdict:" in p
                         else "Step Number: 0"), "all_at_once")
    assert out.defects == [] and out.extra["unparsed"] == 0 and out.extra["judged"] == 1


def test_a_specification_with_no_failing_runs_yields_an_empty_output() -> None:
    from specdx.baselines import whowhen

    inp = whowhen._toy_input(_Scripted(lambda p: ""))
    inp.violated = {}
    out = whowhen.REGISTRY["whowhen"].run(inp, model="m", strategy="all_at_once")
    assert out.defects == [] and out.failure is None and out.cost.requests == 0


def test_calls_near_the_context_window_are_counted() -> None:
    from specdx.baselines import whowhen
    from specdx.baselines.llm import Reply
    from specdx.cost import Cost

    class Big(_Scripted):
        def chat(self, model, messages, **kw):
            self.calls.append((messages, kw))
            tokens = 16384 - 1024 if "Verdict:" in messages[-1]["content"] else 15000
            return Reply(self.answer(messages[-1]["content"]), Cost(1, tokens), "ok")

    out = whowhen.REGISTRY["whowhen"].run(whowhen._toy_input(Big(
        lambda p: "Verdict: agent_error" if "Verdict:" in p else "Step Number: 0")),
        model="m", strategy="all_at_once")
    assert out.extra["near_ctx"] == 1  # 15000 + 1024 < 16384; 15360 + 1024 reaches it
    assert _run(_Scripted(lambda p: "Step Number: 0"), "all_at_once").extra["near_ctx"] == 0
