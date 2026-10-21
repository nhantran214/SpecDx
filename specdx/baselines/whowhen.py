"""B2: Who&When automated failure attribution (Zhang et al., ICML 2025; baselines spec
§2.5, §9.5), on SpecDx's failing runs.

The three localisation prompts are ported verbatim from
github.com/mingyin1/Agents_Failure_Attribution (Automated_FA/Lib/utils.py, commit b2bae5c,
MIT licence, Copyright (c) the Who&When authors). Only the problem, the answer and the chat
content are filled from SpecDx.

A run becomes a multi-agent log: one entry per work step, in flow order (agent = the step's
digital employee or actor). After a strategy names the decisive step, one added verdict call
(ours, not the paper's) asks whether the agent erred or a failed check is wrong. A clause's
score is the share of its sampled runs judged a specification defect at that clause."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from specdx.baselines.api import (
    REGISTRY,
    BaselineInput,
    BaselineOutput,
    register,
    scenario_text,
    tell_excerpt,
)
from specdx.baselines.llm import near_ctx
from specdx.cost import Cost
from specdx.spec.clauses import describe
from specdx.types import Behaviour, Defect

__all__ = ["REGISTRY", "failure_log", "parse_verdict", "sample_runs"]

K_RUNS = 5
MAX_TOKENS = 1024
# The reference's open-model generation (Lib/local_model.py, _run_local_generation:
# do_sample, temperature 0.6, top_p 0.95, max_new_tokens 1024), seeded (spec §9.8).
OPTIONS: dict[str, float | int] = {"temperature": 0.6, "top_p": 0.95, "seed": 1}
ROW_CHARS = 1200


def _h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def sample_runs(violated: Mapping[str, frozenset[str]], ok: set[str],
                k: int = K_RUNS) -> dict[str, list[str]]:
    """Per clause, at most k failing runs (spec §9.5). Runs already chosen for an earlier
    clause are reused first, so one log is judged once and serves every clause it fails."""
    by_clause: dict[str, list[str]] = {}
    for run, cs in violated.items():
        if run in ok:
            for c in cs:
                by_clause.setdefault(c, []).append(run)
    chosen: set[str] = set()
    out: dict[str, list[str]] = {}
    for c in sorted(by_clause):
        runs = sorted(by_clause[c], key=_h)
        reuse = [r for r in runs if r in chosen][:k]
        fresh = [r for r in runs if r not in chosen][:k - len(reuse)]
        out[c] = sorted(reuse + fresh, key=_h)
        chosen.update(out[c])
    return out


def failure_log(b: Behaviour, steps: Sequence[Any],
                drafts: Mapping[str, frozenset[str]]) -> list[dict[str, str]]:
    log = []
    for step in steps:
        if step.id not in b.claims and step.id not in b.state.steps:
            continue
        parts = [b.claims.get(step.id, "")]
        refused = [tool for s, tool in b.refused if s == step.id]
        if refused:
            parts.append("Refused tool calls: " + ", ".join(refused))
        rows = {t: b.state.tables.get(t, []) for t in sorted(drafts.get(step.id, ()))}
        if any(rows.values()):
            parts.append("Rows: " + json.dumps(rows, ensure_ascii=False)[:ROW_CHARS])
        log.append({"name": step.id, "content": "\n".join(p for p in parts if p)})
    return log


_VERDICT = re.compile(r"Verdict:\s*(agent_error|spec_defect\s+(\S+))", re.I)


_TRAILING, _QUOTES = ".,;:)", "`'\"<>"


def parse_verdict(text: str, violated: frozenset[str]) -> str | None:
    """agent_error, a failed check's id, or None (off-format, or not a failed check). The
    id may be wrapped in backticks, quotes or angle brackets and followed by .,;:)"""
    m = _VERDICT.search(text or "")
    if not m:
        return None
    if m.group(1).lower() == "agent_error":
        return "agent_error"
    cid = m.group(2).rstrip(_TRAILING).strip(_QUOTES).rstrip(_TRAILING)
    return cid if cid in violated else None


# --- prompts: verbatim from Lib/utils.py (b2bae5c) --------------------------------------
def _all_at_once_prompt(problem: str, answer: str, chat: str) -> str:
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history "
        "when solving a real world problem. "
        f"The problem is:  {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        "Identify which agent made an error, at which step, and explain the reason for the "
        "error. "
        "Here's the conversation:\n\n" + chat +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for "
        "the wrong solution to the real world problem. If there are no agents that make "
        "obvious mistakes, decide one single agent in your mind. Directly output the name of "
        "the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a "
        "conversation structured as follows: "
        """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes "
        "the speech of each agent. If the mistake is in agent c's speech, the step number is "
        "2. If the second speech by 'agent a' contains the mistake, the step number is 3, and "
        "so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your "
        "prediction)\n Reason for Mistake: \n"
    )


def _step_prompt(problem: str, answer: str, history: str, idx: int, agent: str) -> str:
    return (
        "You are an AI assistant tasked with evaluating the correctness of each step in an "
        "ongoing multi-agent conversation aimed at solving a real-world problem. The problem "
        f"being addressed is: {problem}. "
        f"The Answer for the problem is: {answer}\n"
        f"Here is the conversation history up to the current step:\n{history}\n"
        f"The most recent step ({idx}) was by '{agent}'.\n"
        "Your task is to determine whether this most recent agent's action (Step {idx}) "
        "contains an error that could hinder the problem-solving process or lead to an "
        "incorrect solution. "
        "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
        "Note: Please avoid being overly critical in your evaluation. Focus on errors that "
        "clearly derail the process."
        "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
    )


def _binary_prompt(problem: str, answer: str, chat: str, rng_desc: str, upper: str,
                   lower: str) -> str:
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent "
        "conversation. Multiple agents are collaborating to address a user query, with the "
        "goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within "
        "the provided segment. Determine which half of the segment contains the single step "
        "where this crucial error occurs, ultimately leading to the failure in resolving the "
        "user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        f"Review the following conversation segment {rng_desc}:\n\n{chat}\n\n"
        "Based on your analysis, predict whether the most critical error is more likely to "
        f"be located in the upper half ({upper}) or the lower half ({lower}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. "
        "Remember, your answer should be based on identifying the mistake that directly "
        "contributes to the failure in resolving the user's query. If no single clear error "
        "is evident, consider the step you believe is most responsible for the failure, "
        "allowing for subjective judgment, and base your answer on that."
    )
# --- end of verbatim prompts -------------------------------------------------------------

_SYSTEM = {
    "all_at_once": "You are a helpful assistant skilled in analyzing conversations.",
    "step_by_step": "You are a precise step-by-step conversation evaluator.",
    "binary_search": "You are an AI assistant specializing in localizing errors in "
                     "conversation segments.",
}

VERDICT_PROMPT = (
    "You are given a failed run of a business process and the step where the decisive "
    "mistake occurred.\nProblem: {problem}\nExpected outcome according to the checks: "
    "{answer}\nDecisive step {step} by '{agent}':\n{content}\n\nThe checks that failed are:\n"
    "{checks}\n\nDecide whether the agent made an error (it did not follow the instructions), "
    "or whether the agent followed the instructions and one of the failed checks is wrong.\n"
    "Respond ONLY in the format:\nVerdict: agent_error\nor\nVerdict: spec_defect <check id>"
)
_STEP = re.compile(r"Step Number:\s*\(?(\d+)")

Ask = Callable[[str, str], str | None]  # (system, user) -> reply text, None on a model error


@dataclass(frozen=True)
class WhoWhen:
    name: str = "whowhen"
    uses_llm: bool = True
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = ("all_at_once", "step_by_step", "binary_search")

    def run(self, inp: BaselineInput, *, model: str | None = None,
            strategy: str | None = None) -> BaselineOutput:
        assert inp.gateway is not None and model and strategy in self.strategies
        gateway = inp.gateway
        ok = {b.run_key for b in inp.report.behaviours if b.status == "ok"}
        runs = {b.run_key: b for b in inp.report.behaviours}
        picked = sample_runs(inp.violated, ok)
        excerpt = "\n".join(f"[T:{sid}] {t}" for sid, t in tell_excerpt(inp))
        cost, verdicts, unparsed, near = Cost(), {}, 0, 0

        def ask(system: str, prompt: str) -> str | None:
            nonlocal cost, near
            r = gateway.chat(model, [{"role": "system", "content": system},
                                     {"role": "user", "content": prompt}],
                             options=OPTIONS, max_tokens=MAX_TOKENS)
            cost = cost + r.cost
            near += near_ctx(r, MAX_TOKENS)
            return r.text if r.status == "ok" else None

        for key in sorted({r for rs in picked.values() for r in rs}, key=_h):
            b = runs[key]
            failed = inp.violated[key]
            log = failure_log(b, inp.steps, inp.drafts)
            if not log:
                unparsed += 1
                continue
            problem = f"{scenario_text(b.scenario)}\nInstructions (excerpt):\n{excerpt}"
            checks = "\n".join(f"- {c}: {describe(inp.clauses[c], b.scenario)}"
                               for c in sorted(failed))
            answer = "; ".join(describe(inp.clauses[c], b.scenario) for c in sorted(failed))
            step = self._localise(strategy, log, problem, answer, ask, key)
            if step is None:
                unparsed += 1
                continue
            step = min(step, len(log) - 1)
            entry = log[step]
            text = ask(_SYSTEM["all_at_once"], VERDICT_PROMPT.format(
                problem=problem, answer=answer, step=step, agent=entry["name"],
                content=entry["content"], checks=checks))
            v = parse_verdict(text or "", failed)
            if v is None:
                unparsed += 1
            verdicts[key] = v
        out = []
        for clause, keys in sorted(picked.items()):
            hits = [k for k in keys if verdicts.get(k) == clause]
            if hits:
                out.append(Defect("hidden", (clause,),
                                  tuple(sorted({runs[k].scenario.id for k in hits})),
                                  tuple(hits), len(hits) / len(keys)))
        return BaselineOutput(sorted(out, key=lambda d: (-d.score, d.location)), cost,
                              extra={"judged": len(verdicts), "unparsed": unparsed,
                                     "near_ctx": near})

    @staticmethod
    def _localise(strategy: str | None, log: list[dict[str, str]], problem: str, answer: str,
                  ask: Ask, key: str) -> int | None:
        """The decisive step index, or None when the model errored, answered off-format or
        (step_by_step) flagged no step."""
        chat = "\n".join(f"{e['name']}: {e['content']}" for e in log)
        if strategy == "all_at_once":
            reply = ask(_SYSTEM[strategy], _all_at_once_prompt(problem, answer, chat))
            m = _STEP.search(reply or "")
            return int(m.group(1)) if m else None
        if strategy == "step_by_step":
            history = ""
            for idx, e in enumerate(log):
                history += f"Step {idx} - {e['name']}: {e['content']}\n"
                reply = ask(_SYSTEM[strategy], _step_prompt(problem, answer, history, idx,
                                                            e["name"]))
                if reply is None:  # the original stops its analysis on a failed call
                    return None
                if reply.lower().strip().startswith("1. yes"):
                    return idx
            return None  # no step flagged: the original makes no prediction
        rng = random.Random(_h(key))
        lo, hi = 0, len(log) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            seg = "\n".join(f"{e['name']}: {e['content']}" for e in log[lo:hi + 1])
            reply = ask(_SYSTEM["binary_search"], _binary_prompt(
                problem, answer, seg, f"from step {lo} to step {hi}",
                f"from step {lo} to step {mid}", f"from step {mid + 1} to step {hi}"))
            if reply is None:
                return None
            a = reply.lower()
            if "upper half" in a:
                upper = True
            elif "lower half" in a:
                upper = False
            else:  # the original picks a half at random on an ambiguous reply
                upper = rng.randint(0, 1) == 0
            lo, hi = (lo, mid) if upper else (min(mid + 1, hi), hi)
        return lo


def _toy_input(gateway: Any) -> SimpleNamespace:
    """A one-run, two-step input for unit tests (no database, no corpus)."""
    from specdx.spec.clauses import Clause
    from specdx.types import InterpreterId, Scenario, State

    b = Behaviour("k1", InterpreterId("m", "f", 0, 1), Scenario("happy", 15, 15, 0),
                  State({}, steps={"S1": "done", "S2": "done"}),
                  claims={"S1": "checked the order", "S2": "wrote the draft"})
    clause = Clause("inv:x", "blank", "all", None, frozenset(), "x holds", {})
    return SimpleNamespace(
        gateway=gateway, report=SimpleNamespace(behaviours=[b]),
        violated={"k1": frozenset({"inv:x"})},
        steps=[SimpleNamespace(id="S1", intent="i1", tables=frozenset()),
               SimpleNamespace(id="S2", intent="i2", tables=frozenset())],
        drafts={}, statements={}, clauses={"inv:x": clause},
        variant=SimpleNamespace(corpus=SimpleNamespace()))


register(WhoWhen())
