# SpecDx Baselines and Comparison Study: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the seven baselines of the baselines spec, a fair dev/test comparison study against SpecDx, and the `run_all.sh` stages that run it end to end.

**Architecture:** Baselines are in-process modules under `specdx/baselines/` behind one interface. Each takes a `BaselineInput` built from SpecDx's own report on a specification and returns ranked `Defect`s and a `Cost`.
- **LLM calls.** The LLM-calling baselines (BenchGuard, Who&When) send every call through one cached `Gateway` to the same Ollama and digests as SpecDx.
- **Study.** `specdx/eval/study.py` runs a split, scores every arm with the existing `match_findings`, and computes the threshold-free metrics, paired statistics and the equal-cost table from SpecDx's anytime checkpoints.
- **Selection.** `freeze-baselines` picks models, strategies and thresholds on the dev split and writes a pinned pre-registration. `study` refuses the test split until the gate PASSED and that file is committed.

**Tech Stack:** Python 3.12, numpy, httpx (existing). New optional extra `[baselines]`: `crowd-kit`, `python-sat`, `benchguard` (git, pinned). pytest, ruff, bash and jq for `run_all.sh`.

**Spec:** `docs/superpowers/specs/2026-10-21-specdx-baselines-design.md`, including its §9 planning amendments. Also binding: `docs/superpowers/specs/2026-10-14-specdx-v3-design.md`, v2 and v1.

## Global Constraints

These are copied from the spec; every task must respect them.

**Repository rules**
- Never edit `vendor/`.
- Nothing is imported from `baseline/` clones, `/home/ethan/AILab/FAA` or `aebench`.
- Code, comments and docs are in English.
- ruff: `line-length = 100`; rules E, F, I, UP, B, SIM; target py312.

**Models and caches**
- Local open models only, through `SPECDX_OLLAMA_URL`; Ollama 0.30.4; the population is frozen in `experiments/p2-gate.json`. A model digest that differs from the population runs' digest stops the run.
- `work/runs.jsonl` is append-only and written only through `RunCache`; its first 6505 lines must never change.
- The baseline LLM cache is `<SPECDX_WORK>/baseline-llm.jsonl`, append-only and separate from `runs.jsonl`.
- Run keys must not change: `RUN_SCHEMA_VERSION` stays 3, and the new harness counters are observations only.

**Gate and test split**
- The gate v3 verdict (B2-naive and B3-2PL, `gate_verdict`) is unchanged.
- The test split runs only when the gate PASSED, `experiments/p3-baselines-prereg.json` is committed and the working tree is clean. Otherwise `run_all.sh` exits with code 4.

**Fixed values**

| Value | Setting |
|---|---|
| Who&When sampling | k = 5 failing runs per (specification, clause), with run reuse (spec §9.5) |
| T excerpt | `TELL_CHARS = 8000` |
| Context window | `num_ctx = 16384` (`specdx.config.NUM_CTX`) on every baseline call |
| Checkpoints | `(25, 50, 100, 150, 200)` population runs |
| Bootstrap | 10,000 resamples, seed 20261021 |
| Threshold grid | deciles of the arm's dev scores |
| Model tie-break | smaller parameter count (the `NNb` in the tag), then higher threshold |
| BenchGuard | commit `7d6a52e5627a1a734ecd895e0c73778ab54aca63`; `AuditMode.DEFINITION`; temperature 0.0; `max_tokens` 4096 |
| Who&When | prompts ported from `mingyin1/Agents_Failure_Attribution` at `b2bae5c5b06d681d04ea5e9b63b7a30525c04925` (MIT); `max_tokens` 1024; Ollama's default temperature; seed 1 |
| Severity rank | CRITICAL = 4, HIGH = 3, MEDIUM = 2, LOW = 1 |
| Confidence rank | CONFIRMED = 3, LIKELY = 2, POSSIBLE = 1 |

**Exit codes of `run_all.sh`:** 0 done, 1 check failed, 2 usage or environment error, 3 G2 stop, 4 gate NOT PASSED (test study locked).

## Review Focus

These are the input classes most likely to bite a user in practice; each has a test in the named task.
1. **A specification on which a baseline finds nothing to judge** (no anchored cells read, no failing runs, an empty clause set). The baseline must return an empty `BaselineOutput`, not crash, and scoring must count FNs. Tests: Tasks 5, 9, 11.
2. **An LLM reply that cannot be parsed** (a Who&When verdict outside the format; a BenchGuard protocol returning malformed JSON). It must be recorded as an error, never turned into a finding, and must not stop the spec. Tests: Tasks 9, 10.
3. **A configured model that is not pulled, or has another digest.** The gateway must stop before any chat request, with the model named in the message. Test: Task 3.
4. **Ties and all-zero F1 during selection on dev.** The tie rule must be deterministic (smaller model, then higher threshold) and give the same JSON on a rerun. Test: Task 13.
5. **A study interrupted and re-run.** The second run must be served from both caches and write a byte-identical result file. Test: Task 11 (rerun determinism with fakes).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | `[baselines]` extra |
| `specdx/cost.py` | create | `Cost`; `run_cost(Behaviour)`; `behaviours_cost(...)` (spec §9.1) |
| `specdx/interpreters/executor.py` | modify | harness counters `requests`, `prompt_tokens`, `eval_tokens`, `duration_ms` |
| `specdx/active/loop.py` | modify | `SpecDxConfig.checkpoints`, `Report.checkpoints` (spec §4.4, §9.2) |
| `specdx/baselines/llm.py` | create | `Gateway`, `Reply`, `DigestMismatch` (spec §4.2) |
| `specdx/baselines/api.py` | create | `BaselineInput`, `BaselineOutput`, `Baseline`, `REGISTRY`, `register`, `above`, `tell_excerpt`, `scenario_text` |
| `specdx/baselines/votes.py` | create | per-interpreter modal readings of each cell |
| `specdx/baselines/jest_rule.py` | create | JEST (spec §2.9) |
| `specdx/baselines/truth_inference.py` | create | GLAD and Dawid–Skene via crowd-kit (§2.9) |
| `specdx/baselines/irt_audit.py` | create | B3 forced-ceiling indicator (§2.6) |
| `specdx/baselines/semantic_entropy.py` | create | B4 plus shared ddmin localisation (§2.7) |
| `specdx/baselines/mus_c.py` | create | B5 by neighbourhood exploration plus PySAT Hitman (§9.3) |
| `specdx/baselines/whowhen.py` | create | B2 Who&When (§2.5, §9.5) |
| `specdx/baselines/benchguard.py` | create | B1 packaging, gateway client, finding mapping (§2.3, §9.4) |
| `specdx/baselines/__init__.py` | unchanged | stays empty; `api.load_all()` registers the baselines |
| `specdx/eval/study.py` | create | split runner, arms, scoring, AP / Detection@k, Table B |
| `specdx/eval/stats.py` | create | exact McNemar, Holm, bootstrap |
| `specdx/eval/freeze.py` | create | selection on dev; pre-registration writer |
| `specdx/eval/study_report.py` | create | Markdown tables A and B |
| `specdx/cli.py` | modify | `baselines`, `freeze-baselines`, `study`, `report` commands |
| `run_all.sh` | modify | new steps, `FOREGROUND`, exit 4, status |
| `docs/RUNNING.md` | modify | §12, running the comparison study |
| `tests/...` | create | one test file per module (named in each task) |

---

### Task 1: Request cost accounting (`specdx/cost.py`, executor counters)

**Files:**
- Create: `specdx/cost.py`
- Modify: `specdx/interpreters/executor.py` (`run` counts dict at about line 168; `_chat` at about lines 208–241)
- Test: `tests/test_cost.py`, `tests/interpreters/test_executor.py`

**Interfaces:**
- Produces:
  - `Cost(requests:int=0, prompt_tokens:int=0, eval_tokens:int=0, duration_ms:int=0, errors:int=0, exact:bool=True)`, with `__add__` and `as_dict()`;
  - `run_cost(b: Behaviour) -> Cost`;
  - `behaviours_cost(behaviours: Sequence[Behaviour], extra_runs: int = 0) -> Cost`.
- Harness counters on new runs: `requests`, `prompt_tokens`, `eval_tokens`, `duration_ms`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cost.py`:
```python
from specdx.cost import Cost, behaviours_cost, run_cost
from specdx.types import Behaviour, InterpreterId, Scenario, State

S = Scenario("happy", 15, 15, 0)
I = InterpreterId("m", "f", 0, 1)


def _b(key: str, harness: dict, status: str = "ok") -> Behaviour:
    return Behaviour(key, I, S, State.empty(), harness=harness, status=status)


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
```

Append to `tests/interpreters/test_executor.py`:
```python
def test_each_request_records_its_usage(run_de) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": _say("Xong."), "done": True,
                                         "prompt_eval_count": 120, "eval_count": 30,
                                         "total_duration": 2_500_000_000})

    envelope, _ = run_de(httpx.MockTransport(handle))
    h = envelope.usage["harness"]
    assert (h["requests"], h["prompt_tokens"], h["eval_tokens"], h["duration_ms"]) == (
        1, 120, 30, 2500)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_cost.py tests/interpreters/test_executor.py::test_each_request_records_its_usage -v`
Expected: FAIL. `specdx.cost` does not exist, and the key `requests` is missing.

- [ ] **Step 3: Implement**

`specdx/cost.py`:
```python
"""What a method spent, in model requests (baselines spec §9.1).

A request is one chat call to Ollama. Runs made since the executor counts requests carry
exact figures; older cached runs keep only tool_calls and nudges per step, and a step can
have made at most tool_calls + nudges + 1 requests, so their cost is an upper bound. The
bound overstates SpecDx's cost, which is the conservative direction for the comparison.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from specdx.types import Behaviour

COUNTERS: tuple[str, ...] = ("requests", "prompt_tokens", "eval_tokens", "duration_ms")


@dataclass(frozen=True)
class Cost:
    requests: int = 0
    prompt_tokens: int = 0
    eval_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0
    exact: bool = True

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.requests + other.requests, self.prompt_tokens + other.prompt_tokens,
                    self.eval_tokens + other.eval_tokens, self.duration_ms + other.duration_ms,
                    self.errors + other.errors, self.exact and other.exact)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run_cost(b: Behaviour) -> Cost:
    errors = int(b.status == "error")
    steps = list(b.harness.values())
    if steps and all("requests" in s for s in steps):
        totals = {k: sum(int(s.get(k, 0)) for s in steps) for k in COUNTERS}
        return Cost(**totals, errors=errors, exact=True)
    bound = sum(int(s.get("tool_calls", 0)) + int(s.get("nudges", 0)) + 1 for s in steps)
    return Cost(requests=bound, errors=errors, exact=False)


def behaviours_cost(behaviours: Sequence[Behaviour], extra_runs: int = 0) -> Cost:
    """The runs' cost, plus extra_runs (probe and ddmin runs a report only counts) charged
    at the mean requests per run, rounded up."""
    total = sum((run_cost(b) for b in behaviours), Cost())
    if extra_runs and behaviours:
        per = math.ceil(total.requests / len(behaviours))
        total = total + Cost(requests=per * extra_runs, exact=False)
    return total
```

In `specdx/interpreters/executor.py`:
- Change the counts initialiser in `run` to:
  ```python
  counts = {"nudges": 0, "parsed_from_text": 0, "tool_calls": 0, "valid_calls": 0,
            "requests": 0, "prompt_tokens": 0, "eval_tokens": 0, "duration_ms": 0}
  ```
- Pass `counts` into `_chat`, i.e. `reply = self._chat(client, messages, schemas, counts)`, and change its signature to `def _chat(self, client, messages, schemas, counts: dict[str, int])`.
- Replace the success branch inside `_chat`:
  ```python
                response.raise_for_status()
                data = response.json()
                counts["requests"] += 1
                counts["prompt_tokens"] += int(data.get("prompt_eval_count") or 0)
                counts["eval_tokens"] += int(data.get("eval_count") or 0)
                counts["duration_ms"] += int(data.get("total_duration") or 0) // 1_000_000
                message: dict[str, Any] = data["message"]
                return message
  ```

- [ ] **Step 4: Check that nothing assumes the old key set**

Run: `grep -rn "harness\[" specdx tests run_all.sh | grep -v "envelope.usage"`
Expected: only readers of named keys (`parsed_from_text`, `nudges`, `tool_calls`, `valid_calls`). Nothing iterates every harness key into a total. If something does, restrict it to the four old keys in this step.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_cost.py tests/interpreters/test_executor.py tests/test_run_keys.py tests/test_preregistration_v3.py -q`
Expected: PASS. Run keys are unchanged, because the counters are not interpreter-visible.

- [ ] **Step 6: Commit**

```bash
git add specdx/cost.py specdx/interpreters/executor.py tests/test_cost.py tests/interpreters/test_executor.py
git commit -m "feat: request cost accounting — exact counters on new runs, an upper bound for cached ones"
```

---

### Task 2: Anytime checkpoints in the loop

**Files:**
- Modify: `specdx/active/loop.py` (`SpecDxConfig` at about line 71, `Report` at about line 110, `Report.to_json`, and the `while runs < config.budget` body after `defects, spent = ...`)
- Test: `tests/active/test_loop.py`

**Interfaces:**
- Consumes: `behaviours_cost` (Task 1).
- Produces:
  - `SpecDxConfig.checkpoints: tuple[int, ...] = ()`;
  - `Report.checkpoints: list[dict[str, Any]]`, whose entries are `{"at": int, "runs": int, "cost": Cost.as_dict(), "defects": [asdict(Defect)]}`;
  - `Report.to_json()` emits `"checkpoints"` only when the list is non-empty.

- [ ] **Step 1: Write the failing test**

Append to `tests/active/test_loop.py`:
```python
def test_checkpoints_leave_the_report_unchanged() -> None:
    plain = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)), _config(budget=100))
    marked = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)),
                        _config(budget=100, checkpoints=(5, 20, 1000)))
    assert [c["at"] for c in marked.checkpoints] == [5, 20]
    assert all(c["runs"] >= c["at"] and c["cost"]["requests"] > 0 for c in marked.checkpoints)
    assert dataclasses.replace(marked, checkpoints=[]).to_json() == plain.to_json()
    assert "checkpoints" not in json.loads(plain.to_json())


def test_a_checkpoint_drops_the_base_ambiguities_like_the_final_report() -> None:
    base = run_specdx(VARIANT, _split_ctx(FakeRunner(_split_plan)), _config(budget=60))
    mutant = run_specdx(VARIANT, _split_ctx(FakeRunner(_split_plan)),
                        _config(budget=60, checkpoints=(10,)), base=base)
    kept = mutant.checkpoints[0]["defects"]
    assert not [d for d in kept if d["type"] == "ambiguity"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/active/test_loop.py -k checkpoint -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'checkpoints'`.

- [ ] **Step 3: Implement**

In `SpecDxConfig`, add after `params`:
```python
    # Anytime checkpoints (baselines spec §4.4, §9.2): population-run counts at which the
    # diagnosis so far is recorded. Output only: never read by any decision.
    checkpoints: tuple[int, ...] = ()
```

In `Report`, add the field `checkpoints: list[dict[str, Any]] = field(default_factory=list)`, and in `to_json` build the dict first, then:
```python
        out = {...existing keys...}
        if self.checkpoints:
            out["checkpoints"] = self.checkpoints
        return json.dumps(out, ensure_ascii=False, indent=2)
```
Keep the existing key order.

In `run_specdx`:
- before the loop, add `checkpoints: list[dict[str, Any]] = []`;
- immediately after `defects, spent = diagnosis.defects, diagnosis.spent`, add:
  ```python
        for at in config.checkpoints:
            if runs >= at and all(c["at"] != at for c in checkpoints):
                kept = [d for d in defects if d.type != "ambiguity" or base is None
                        or ambiguity_key(d) not in base.ambiguity_keys]
                checkpoints.append({"at": at, "runs": runs,
                                    "cost": behaviours_cost(behaviours).as_dict(),
                                    "defects": [asdict(d) for d in kept]})
  ```
- pass `checkpoints=checkpoints` to the `Report(...)` constructor;
- add `from specdx.cost import behaviours_cost` to the imports (`asdict` is already imported for `to_json`).

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/active -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add specdx/active/loop.py tests/active/test_loop.py
git commit -m "feat: anytime checkpoints in the loop (output only; the report is unchanged)"
```

---

### Task 3: LLM gateway

**Files:**
- Create: `specdx/baselines/llm.py`
- Test: `tests/baselines/test_llm.py`

**Interfaces:**
- Consumes: `Cost` (Task 1); `specdx.config.NUM_CTX`; `specdx.types.InfrastructureError`.
- Produces:
  - `Reply(text: str, cost: Cost, status: str, error: str = "", cached: bool = False)`;
  - `class DigestMismatch(RuntimeError)`;
  - `Gateway(base_url: str, cache_path: Path, pinned: Mapping[str, str] | None = None, *, transport: httpx.BaseTransport | None = None, timeout: float = 600.0)`, with methods `.digest(model) -> str` and `.chat(model, messages, *, options=None, max_tokens=None) -> Reply`, and the attribute `.total: Cost`.

- [ ] **Step 1: Write the failing tests**

`tests/baselines/test_llm.py`:
```python
import json

import httpx
import pytest

from specdx.baselines.llm import DigestMismatch, Gateway


def _server(digest="d1", fail=False):
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:7b", "digest": digest}]})
        if fail:
            raise httpx.ReadTimeout("slow", request=request)
        body = json.loads(request.content)
        assert body["options"]["num_ctx"] == 16384 and body["stream"] is False
        return httpx.Response(200, json={"message": {"content": "hi"}, "prompt_eval_count": 7,
                                         "eval_count": 3, "total_duration": 4_000_000})
    return httpx.MockTransport(handle), calls


def test_a_cache_hit_sends_no_chat_request(tmp_path) -> None:
    t, calls = _server()
    g = Gateway("http://x", tmp_path / "c.jsonl", transport=t)
    first = g.chat("m:7b", [{"role": "user", "content": "a"}], options={"seed": 1})
    again = Gateway("http://x", tmp_path / "c.jsonl", transport=t).chat(
        "m:7b", [{"role": "user", "content": "a"}], options={"seed": 1})
    assert (first.text, first.cost.requests, first.cost.prompt_tokens) == ("hi", 1, 7)
    assert again.cached and again.text == "hi"
    assert calls.count("/api/chat") == 1


def test_the_key_changes_with_the_digest(tmp_path) -> None:
    for d in ("d1", "d2"):
        t, _ = _server(digest=d)
        Gateway("http://x", tmp_path / "c.jsonl", transport=t).chat("m:7b", [])
    assert len((tmp_path / "c.jsonl").read_text().splitlines()) == 2


def test_a_timeout_is_cached_as_an_error_and_not_retried(tmp_path) -> None:
    t, calls = _server(fail=True)
    g = Gateway("http://x", tmp_path / "c.jsonl", transport=t)
    r1 = g.chat("m:7b", [])
    r2 = g.chat("m:7b", [])
    assert (r1.status, r1.cost.errors, r2.cached) == ("error", 1, True)
    assert calls.count("/api/chat") == 1


def test_a_missing_model_stops_before_any_chat(tmp_path) -> None:
    t, calls = _server()
    with pytest.raises(DigestMismatch, match="other:1b"):
        Gateway("http://x", tmp_path / "c.jsonl", transport=t).chat("other:1b", [])
    assert "/api/chat" not in calls


def test_a_pinned_digest_mismatch_stops(tmp_path) -> None:
    t, _ = _server(digest="new")
    with pytest.raises(DigestMismatch, match="m:7b"):
        Gateway("http://x", tmp_path / "c.jsonl", {"m:7b": "old"}, transport=t).chat("m:7b", [])
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/baselines/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: specdx.baselines.llm`.

- [ ] **Step 3: Implement**

`specdx/baselines/llm.py`:
```python
"""One door to Ollama for the LLM-calling baselines (baselines spec §4.2).

Every call is keyed by sha256(schema, model digest, request body), so a rerun is served
from the append-only cache and a changed model can never hit an old answer. A timeout or
HTTP error is cached as status "error": counted, never a finding, never retried.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from specdx.config import NUM_CTX
from specdx.cost import Cost
from specdx.types import InfrastructureError

SCHEMA = 1


class DigestMismatch(RuntimeError):
    """The model is not pulled, or its digest differs from the population runs'."""


@dataclass(frozen=True)
class Reply:
    text: str
    cost: Cost
    status: str
    error: str = ""
    cached: bool = False


class Gateway:
    def __init__(self, base_url: str, cache_path: Path,
                 pinned: Mapping[str, str] | None = None, *,
                 transport: httpx.BaseTransport | None = None, timeout: float = 600.0) -> None:
        self.base_url, self.cache_path = base_url.rstrip("/"), cache_path
        self.pinned = dict(pinned or {})
        self.transport, self.timeout = transport, timeout
        self.total = Cost()
        self._digests: dict[str, str] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec

    def _client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport, timeout=self.timeout)

    def digest(self, model: str) -> str:
        if model not in self._digests:
            try:
                with self._client() as c:
                    resp = c.get(f"{self.base_url}/api/tags")
                    resp.raise_for_status()
                    tags = resp.json()
            except httpx.HTTPError as exc:
                raise InfrastructureError(f"cannot list models at {self.base_url}: {exc}") from exc
            found = {m["name"]: m["digest"] for m in tags.get("models", [])}
            if model not in found:
                raise DigestMismatch(f"{model} is not pulled at {self.base_url}")
            if model in self.pinned and self.pinned[model] != found[model]:
                raise DigestMismatch(f"{model} has digest {found[model]}, the population runs "
                                     f"used {self.pinned[model]}")
            self._digests[model] = found[model]
        return self._digests[model]

    def chat(self, model: str, messages: Sequence[Mapping[str, str]], *,
             options: Mapping[str, Any] | None = None, max_tokens: int | None = None) -> Reply:
        digest = self.digest(model)
        opts: dict[str, Any] = {"num_ctx": NUM_CTX, **(options or {})}
        if max_tokens:
            opts["num_predict"] = max_tokens
        body: dict[str, Any] = {"model": model, "messages": [dict(m) for m in messages],
                                "stream": False, "options": opts}
        if model.startswith("qwen3"):  # as the interpreters call it (executor._chat)
            body["think"] = False
        key = hashlib.sha256(json.dumps({"schema": SCHEMA, "digest": digest, "body": body},
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self._cache:
            rec = self._cache[key]
            cost = Cost(**rec["cost"])
            self.total = self.total + cost
            return Reply(rec["text"], cost, rec["status"], rec.get("error", ""), cached=True)
        try:
            with self._client() as c:
                resp = c.post(f"{self.base_url}/api/chat", json=body)
                resp.raise_for_status()
                data = resp.json()
            cost = Cost(1, int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0),
                        int(data.get("total_duration") or 0) // 1_000_000)
            rec = {"text": str(data["message"].get("content") or ""), "status": "ok", "error": ""}
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            cost = Cost(requests=1, errors=1)
            rec = {"text": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        except httpx.TransportError as exc:  # connection refused, tunnel down: not data
            raise InfrastructureError(f"Ollama at {self.base_url} failed: {exc}") from exc
        rec.update({"key": key, "model": model, "digest": digest, "cost": cost.as_dict()})
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self._cache[key] = rec
        self.total = self.total + cost
        return Reply(rec["text"], cost, rec["status"], rec["error"])
```

`httpx.ReadTimeout` is a `TimeoutException`, so it is caught before `TransportError`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/baselines/test_llm.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add specdx/baselines/llm.py tests/baselines/test_llm.py
git commit -m "feat: cached LLM gateway for the baselines (digest-keyed, errors cached, digest pin)"
```

---

### Task 4: Baseline interface, registry, shared T excerpt, dependencies

**Files:**
- Create: `specdx/baselines/api.py`
- Modify: `pyproject.toml` (optional dependencies)
- Test: `tests/baselines/test_api.py`

**Interfaces:**
- Consumes: `Cost` (Task 1); `Gateway` (Task 3); `Report`, `LoopContext`, `SpecDxConfig` (`specdx.active.loop`); `CellData` (`specdx.infer.cells`); `ClauseSet` (`specdx.spec.clauses`); `WorkStep` (`specdx.orchestrate.work_orders`); `SpecVariant`; `State`; `Defect`; `Scenario`.
- Produces:
  ```python
  @dataclass
  class BaselineInput:
      variant: SpecVariant
      report: Report
      data: CellData
      violated: Mapping[str, frozenset[str]]       # run_key -> violated clause ids
      clauses: ClauseSet
      ctx: LoopContext
      config: SpecDxConfig
      steps: Sequence[WorkStep]
      statements: Mapping[str, Sequence[str]]      # intent -> statement variants
      drafts: Mapping[str, frozenset[str]]         # step id -> tables it drafts
      golden: State | None = None                  # canon end state, materialised (B5)
      gateway: Gateway | None = None               # LLM baselines

  @dataclass
  class BaselineOutput:
      defects: list[Defect]                        # every candidate, scored, unthresholded
      cost: Cost = Cost()
      failure: str | None = None
      extra: dict[str, Any] = field(default_factory=dict)

  class Baseline(Protocol):
      name: str
      uses_llm: bool
      reports: tuple[str, ...]                     # defect types it can report
      strategies: tuple[str | None, ...]
      def run(self, inp: BaselineInput, *, model: str | None = None,
              strategy: str | None = None) -> BaselineOutput: ...
  ```
  together with:
  - `REGISTRY: dict[str, Baseline]`;
  - `register(b: Baseline) -> Baseline`;
  - `above(defects, tau) -> list[Defect]`;
  - `ranked(defects) -> list[Defect]`;
  - `TELL_CHARS = 8000`;
  - `BASELINE_MODULES` and `load_all() -> dict[str, Baseline]`. These import every baseline module so that each registers. The package `__init__` stays empty on purpose: the gate imports `specdx.baselines.b2_single`, and it must not pull in the optional `[baselines]` packages;
  - `tell_excerpt(inp, step_ids: Sequence[str] | None = None) -> list[tuple[str, str]]`;
  - `scenario_text(s: Scenario) -> str`;
  - `families(config) -> dict[str, str]` (model → family).

- [ ] **Step 1: Add the dependencies and install them**

In `pyproject.toml` under `[project.optional-dependencies]` add:
```toml
baselines = [
    "crowd-kit>=1.4",
    "python-sat>=1.8.dev13",
    "benchguard @ git+https://github.com/XinmingTu/BenchGuard@7d6a52e5627a1a734ecd895e0c73778ab54aca63",
]
```

Run: `.venv/bin/python -m pip install -e ".[dev,baselines]"`
Expected: success. Then run `.venv/bin/python -c "import crowdkit, pysat, benchguard; print('ok')"` and expect `ok`.

- [ ] **Step 2: Write the failing tests**

`tests/baselines/test_api.py`:
```python
from pathlib import Path

from specdx.baselines.api import (
    REGISTRY, TELL_CHARS, BaselineOutput, above, ranked, register, scenario_text, tell_excerpt,
)
from specdx.types import Defect, Scenario


def _d(score: float, loc: str = "c") -> Defect:
    return Defect("hidden", (loc,), (), (), score)


def test_above_and_ranked() -> None:
    ds = [_d(0.2, "a"), _d(0.9, "b"), _d(0.5, "c")]
    assert [d.location for d in ranked(ds)] == [("b",), ("c",), ("a",)]
    assert [d.location for d in above(ds, 0.5)] == [("b",), ("c",)]


def test_register_keeps_one_per_name() -> None:
    class Dummy:
        name, uses_llm, reports, strategies = "dummy-test", False, ("hidden",), (None,)

        def run(self, inp, *, model=None, strategy=None):
            return BaselineOutput([])

    register(Dummy())
    assert REGISTRY["dummy-test"].reports == ("hidden",)
    del REGISTRY["dummy-test"]


def test_scenario_text_names_every_parameter() -> None:
    text = scenario_text(Scenario("x", 15, 12, 4, 2))
    assert all(s in text for s in ("15", "12", "4%", "2"))


def test_the_excerpt_is_capped_and_ordered(tell_corpus, pack) -> None:
    from types import SimpleNamespace

    from specdx.interpreters.prompt import load_variants
    from specdx.orchestrate.work_orders import load_work_orders
    from specdx.variant import SpecVariant

    steps = load_work_orders()
    inp = SimpleNamespace(variant=SpecVariant("base", Path("."), tell_corpus), steps=steps,
                          statements=load_variants().statements)
    out = tell_excerpt(inp)
    assert out and sum(len(t) for _, t in out) <= TELL_CHARS
    assert len({sid for sid, _ in out}) == len(out)
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/baselines/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

`specdx/baselines/api.py`:
```python
"""The baselines' shared interface (baselines spec §2.2, §4.1).

A baseline reads one specification's SpecDx evidence (BaselineInput) and returns every
candidate it would report, scored and unthresholded (BaselineOutput). Thresholds are
applied at scoring time with `above`, so selection on the dev split never re-runs a
baseline, and the ranked list gives average precision and Detection@k.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from specdx.cost import Cost
from specdx.types import Defect, Scenario, State

if TYPE_CHECKING:
    from specdx.active.loop import LoopContext, Report, SpecDxConfig
    from specdx.baselines.llm import Gateway
    from specdx.infer.cells import CellData
    from specdx.orchestrate.work_orders import WorkStep
    from specdx.spec.clauses import ClauseSet
    from specdx.variant import SpecVariant

TELL_CHARS = 8000  # spec §9.6


@dataclass
class BaselineInput:
    variant: SpecVariant
    report: Report
    data: CellData
    violated: Mapping[str, frozenset[str]]
    clauses: ClauseSet
    ctx: LoopContext
    config: SpecDxConfig
    steps: Sequence[WorkStep]
    statements: Mapping[str, Sequence[str]]
    drafts: Mapping[str, frozenset[str]]
    golden: State | None = None
    gateway: Gateway | None = None


@dataclass
class BaselineOutput:
    defects: list[Defect]
    cost: Cost = Cost()
    failure: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Baseline(Protocol):
    name: str
    uses_llm: bool
    reports: tuple[str, ...]
    strategies: tuple[str | None, ...]

    def run(self, inp: BaselineInput, *, model: str | None = None,
            strategy: str | None = None) -> BaselineOutput: ...


REGISTRY: dict[str, Baseline] = {}


def register(b: Baseline) -> Baseline:
    REGISTRY[b.name] = b
    return b


BASELINE_MODULES: tuple[str, ...] = ("jest_rule", "truth_inference", "irt_audit",
                                     "semantic_entropy", "mus_c", "whowhen", "benchguard")


def load_all() -> dict[str, Baseline]:
    """Import every baseline module (they register on import). Kept out of the package
    __init__ so the gate's imports never need the optional [baselines] extra."""
    import importlib

    for name in BASELINE_MODULES:
        importlib.import_module(f"specdx.baselines.{name}")
    return REGISTRY


def ranked(defects: Sequence[Defect]) -> list[Defect]:
    return sorted(defects, key=lambda d: (-d.score, d.type, d.location))


def above(defects: Sequence[Defect], tau: float) -> list[Defect]:
    return [d for d in ranked(defects) if d.score >= tau]


def families(config: SpecDxConfig) -> dict[str, str]:
    return {m: f for f, ms in config.models_by_family.items() for m in ms}


def scenario_text(s: Scenario) -> str:
    return (f"Case CASE-0031: the customer orders {s.ordered} units; {s.received} units are "
            f"received in {s.grn_count} goods receipt(s); the requested discount is "
            f"{s.discount_pct}%.")


def tell_excerpt(inp: Any, step_ids: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """The T sentences the localiser's query retrieves for the given steps (all steps by
    default): per step, top 5 chunks for its statement plus its tables; in step order, then
    sentence order; deduplicated; capped at TELL_CHARS characters (spec §9.6)."""
    corpus = inp.variant.corpus
    wanted = set(step_ids) if step_ids is not None else None
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    used = 0
    for step in inp.steps:
        if (wanted is not None and step.id not in wanted) or step.intent not in inp.statements:
            continue
        query = f"{inp.statements[step.intent][0]} {' '.join(sorted(step.tables))}"
        chunks = corpus.search(query, k=5)
        for sid in corpus.sentence_ids([c.id for c in chunks]):
            if sid in seen:
                continue
            text = corpus.sentence(sid)
            if used + len(text) > TELL_CHARS:
                return out
            seen.add(sid)
            used += len(text)
            out.append((sid, text))
    return out
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/baselines/test_api.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml specdx/baselines/api.py tests/baselines/test_api.py
git commit -m "feat: baseline interface, registry and shared T excerpt; [baselines] extra"
```

---

### Task 5: JEST rule and truth inference (GLAD, Dawid–Skene)

**Files:**
- Create: `specdx/baselines/votes.py`, `specdx/baselines/jest_rule.py`, `specdx/baselines/truth_inference.py`
- Test: `tests/baselines/test_statistical.py`

**Interfaces:**
- Consumes: `BaselineInput`, `BaselineOutput`, `register` (Task 4); `CellData`, `Cell`, `Reading` (`specdx.infer.cells`).
- Produces:
  - `votes.interpreter_votes(data: CellData, behaviours: Sequence[Behaviour]) -> dict[str, dict[str, str]]`, mapping cell → (interpreter key `model|vV|sS`) → modal value;
  - `votes.model_votes(data) -> dict[str, dict[str, str]]`, mapping cell → model → modal value;
  - registered baselines `"jest"`, `"glad"`, `"dawid_skene"`, each reporting `("hidden",)`.
- A hidden finding's location is `(cell.clause,)`. Several cells with the same clause merge, keeping the maximum score.

- [ ] **Step 1: Write the failing tests**

`tests/baselines/test_statistical.py`:
```python
from types import SimpleNamespace

import specdx.baselines.jest_rule  # noqa: F401 - registers "jest"
import specdx.baselines.truth_inference  # noqa: F401 - registers "glad", "dawid_skene"
from specdx.baselines.api import REGISTRY
from specdx.baselines.votes import model_votes
from specdx.infer.cells import Cell, CellData, Reading
from specdx.types import Behaviour, InterpreterId, Scenario, State

S = Scenario("happy", 15, 15, 0)


def _data(spec: dict[str, tuple[str, dict[str, list[str]]]]):
    """cell -> (anchor, model -> readings); one run per reading."""
    cells, readings, runs = {}, [], []
    for cid, (anchor, by_model) in spec.items():
        cells[cid] = Cell(cid, "value", anchor, f"golden:{cid}", f"t.{cid}", cid, ("s1",))
        for m, values in by_model.items():
            for i, v in enumerate(values):
                key = f"{cid}-{m}-{i}"
                readings.append(Reading(cid, key, m, "fam-" + m[0], "happy", v))
                runs.append(Behaviour(key, InterpreterId(m, "fam-" + m[0], 0, i + 1), S,
                                      State.empty()))
    return CellData(cells, readings), runs


def _inp(spec):
    data, runs = _data(spec)
    return SimpleNamespace(data=data, report=SimpleNamespace(behaviours=runs, theta={}))


MISKEY = {
    "c1": ("d", {"strong1": ["x", "x", "x"], "strong2": ["x", "x", "x"], "weak": ["d", "y", "x"]}),
    "c2": ("d", {"strong1": ["d", "d", "d"], "strong2": ["d", "d", "d"], "weak": ["d", "y", "d"]}),
}


def test_model_votes_take_each_models_mode() -> None:
    data, _ = _data(MISKEY)
    assert model_votes(data)["c1"] == {"strong1": "x", "strong2": "x", "weak": "d"}


def test_jest_scores_the_share_of_deviating_models() -> None:
    out = REGISTRY["jest"].run(_inp(MISKEY))
    scores = {d.location: d.score for d in out.defects}
    assert scores == {("golden:c1",): 2 / 3}


def test_truth_inference_flags_a_miskeyed_cell_only() -> None:
    for name in ("glad", "dawid_skene"):
        out = REGISTRY[name].run(_inp(MISKEY))
        assert [d.location for d in out.defects if d.score > 0.5] == [("golden:c1",)], name


def test_nothing_to_judge_gives_an_empty_output() -> None:
    for name in ("jest", "glad", "dawid_skene"):
        assert REGISTRY[name].run(_inp({})).defects == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/baselines/test_statistical.py -v`
Expected: FAIL with `ModuleNotFoundError: specdx.baselines.votes`.

- [ ] **Step 3: Implement**

`specdx/baselines/votes.py`:
```python
"""Modal readings per model or per interpreter (model, Tell variant, seed) of each cell."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from specdx.infer.cells import CellData
from specdx.types import Behaviour


def mode(values: Sequence[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def model_votes(data: CellData) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for cid, rs in data.by_cell().items():
        by: dict[str, list[str]] = {}
        for r in rs:
            by.setdefault(r.model, []).append(r.value)
        if by:
            out[cid] = {m: mode(vs) for m, vs in sorted(by.items())}
    return out


def interpreter_votes(data: CellData, behaviours: Sequence[Behaviour]) -> dict[str, dict[str, str]]:
    who = {b.run_key: f"{b.interpreter.model}|v{b.interpreter.variant}|s{b.interpreter.seed}"
           for b in behaviours}
    out: dict[str, dict[str, str]] = {}
    for cid, rs in data.by_cell().items():
        by: dict[str, list[str]] = {}
        for r in rs:
            by.setdefault(who.get(r.run_key, r.model), []).append(r.value)
        if by:
            out[cid] = {w: mode(vs) for w, vs in sorted(by.items())}
    return out
```

`specdx/baselines/jest_rule.py`:
```python
"""JEST's N+1-version rule (Park et al., ICSE 2021; baselines spec §2.9): a cell is a
specification defect when the share of models whose modal reading deviates from the
specification's declared value reaches τ_JEST. The score is that share."""

from __future__ import annotations

from dataclasses import dataclass

from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.baselines.votes import model_votes
from specdx.types import Defect


def merge_max(defects: list[Defect]) -> list[Defect]:
    best: dict[tuple[str, tuple[str, ...]], Defect] = {}
    for d in defects:
        if d.key not in best or d.score > best[d.key].score:
            best[d.key] = d
    return sorted(best.values(), key=lambda d: (-d.score, d.location))


@dataclass(frozen=True)
class JestRule:
    name: str = "jest"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        out: list[Defect] = []
        by_cell = inp.data.by_cell()
        for cid, votes in model_votes(inp.data).items():
            cell = inp.data.cells[cid]
            if cell.anchor is None or cell.clause is None:
                continue
            share = sum(v != cell.anchor for v in votes.values()) / len(votes)
            if share > 0:
                off = [r for r in by_cell[cid] if r.value != cell.anchor]
                out.append(Defect("hidden", (cell.clause,), tuple(sorted({r.scenario for r in off})),
                                  tuple(sorted(r.run_key for r in off)), share))
        return BaselineOutput(merge_max(out))


register(JestRule())
```

`specdx/baselines/truth_inference.py`:
```python
"""GLAD and Dawid–Skene on the cell readings (crowd-kit; baselines spec §2.9).
task = cell, worker = interpreter (model, Tell variant, seed), label = its modal reading.
A cell is a hidden-requirement candidate scored 1 − P(declared value)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from crowdkit.aggregation import GLAD, DawidSkene

from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.baselines.jest_rule import merge_max
from specdx.baselines.votes import interpreter_votes
from specdx.types import Defect


@dataclass(frozen=True)
class TruthInference:
    name: str
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def _proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.name == "glad":
            return GLAD(n_iter=100, silent=True).fit_predict_proba(df)
        return DawidSkene(n_iter=100).fit_predict_proba(df)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        votes = interpreter_votes(inp.data, inp.report.behaviours)
        anchored = {c: v for c, v in votes.items()
                    if inp.data.cells[c].anchor is not None and inp.data.cells[c].clause}
        rows = [(c, w, lab) for c, by in anchored.items() for w, lab in by.items()]
        if not rows:
            return BaselineOutput([])
        proba = self._proba(pd.DataFrame(rows, columns=["task", "worker", "label"]))
        by_cell = inp.data.by_cell()
        out: list[Defect] = []
        for cid in sorted(anchored):
            cell = inp.data.cells[cid]
            p_anchor = float(proba.loc[cid].get(cell.anchor, 0.0)) if cid in proba.index else 0.0
            score = 1.0 - p_anchor
            if score > 0:
                off = [r for r in by_cell[cid] if r.value != cell.anchor]
                out.append(Defect("hidden", (cell.clause,), tuple(sorted({r.scenario for r in off})),
                                  tuple(sorted(r.run_key for r in off)), score))
        return BaselineOutput(merge_max(out))


register(TruthInference("glad"))
register(TruthInference("dawid_skene"))
```

`specdx/baselines/__init__.py` stays empty (see `load_all` in Task 4).

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/baselines/test_statistical.py -v`
Expected: PASS. If crowd-kit's GLAD signature differs (for example, no `silent`), adapt the constructor arguments to the installed version: `python -c "import crowdkit.aggregation as a, inspect; print(inspect.signature(a.GLAD))"`. Keep `n_iter=100`.

- [ ] **Step 5: Commit**

```bash
git add specdx/baselines/votes.py specdx/baselines/jest_rule.py specdx/baselines/truth_inference.py tests/baselines/test_statistical.py
git commit -m "feat: JEST rule and GLAD/Dawid–Skene baselines on the cell readings"
```

---

### Task 6: B3, the forced-ceiling IRT indicator

**Files:**
- Create: `specdx/baselines/irt_audit.py`
- Test: `tests/baselines/test_irt_audit.py`

**Interfaces:**
- Consumes: `interpreter_votes` (Task 5); `BaselineInput` (Task 4).
- Produces:
  - `delta_loglik(y: np.ndarray, n: np.ndarray, theta: np.ndarray, c: float) -> float`;
  - the registered baseline `"irt_audit"`, reporting `("hidden",)` with score Δℓ.

- [ ] **Step 1: Read the paper's model**

Read `baseline/06_irt_audit/land2026auditing.pdf`, its method section, and confirm the likelihood used below:
- **Item response:** P_ij = c_i + (d_i − c_i)·σ(a_i(θ_j − b_i)), with a_i ≥ 0.
- **Constrained fit:** d_i = 1.
- **Free fit:** d_i ∈ [c_i, 1].
- **Score:** Δℓ_i = max ℓ_free − max ℓ_constrained.

If the paper differs, change `_ll` and the grid in Step 3, and record the change and the page number in the module docstring before continuing.

- [ ] **Step 2: Write the failing test**

`tests/baselines/test_irt_audit.py`:
```python
import numpy as np

from specdx.baselines.irt_audit import delta_loglik


def test_a_miskeyed_item_has_the_largest_ceiling_gain() -> None:
    rng = np.random.default_rng(0)
    theta = np.linspace(-2, 2, 12)
    n = np.full(12, 9)

    def item(a, b):
        p = 1 / (1 + np.exp(-a * (theta - b)))
        return rng.binomial(n, p)

    normal = [item(1.5, b) for b in (-1.0, 0.0, 1.0)]
    miskey = rng.binomial(n, np.full(12, 0.1))  # even the strongest fail it
    scores = [delta_loglik(y, n, theta, c=0.0) for y in [*normal, miskey]]
    assert int(np.argmax(scores)) == 3
    assert all(s >= -1e-9 for s in scores)
```

- [ ] **Step 3: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/baselines/test_irt_audit.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

`specdx/baselines/irt_audit.py`:
```python
"""B3: the forced-ceiling IRT mislabel indicator (Land & Bikel, arXiv 2605.30504;
baselines spec §2.6). Subjects are interpreters (model, Tell variant, seed); items are
anchored cells, scored 1 when an interpreter's modal reading equals the declared value.
θ_j is the logit of the subject's overall accuracy, standardised. Each item is fitted
twice by deterministic grid search: with the ceiling forced to 1, and with a free ceiling
d ∈ [c, 1]. Δℓ = ℓ_free − ℓ_forced is large when even strong subjects miss the declared
value: the signature of a mis-keyed item. c is the chance floor 1/|values read|."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.baselines.jest_rule import merge_max
from specdx.baselines.votes import interpreter_votes
from specdx.types import Defect

A_GRID = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
B_GRID = np.linspace(-3.0, 3.0, 25)
D_STEPS = 21
EPS = 1e-6


def _ll(y: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (n - y) * np.log(1 - p)))


def _best(y, n, theta, c, ceilings) -> float:
    best = -np.inf
    for a in A_GRID:
        for b in B_GRID:
            s = 1 / (1 + np.exp(-a * (theta - b)))
            for d in ceilings:
                best = max(best, _ll(y, n, c + (d - c) * s))
    return best


def delta_loglik(y: np.ndarray, n: np.ndarray, theta: np.ndarray, c: float) -> float:
    forced = _best(y, n, theta, c, [1.0])
    free = _best(y, n, theta, c, np.linspace(c, 1.0, D_STEPS))
    return max(0.0, free - forced)


@dataclass(frozen=True)
class IrtAudit:
    name: str = "irt_audit"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        votes = interpreter_votes(inp.data, inp.report.behaviours)
        items = [c for c in sorted(votes)
                 if inp.data.cells[c].anchor is not None and inp.data.cells[c].clause]
        subjects = sorted({w for c in items for w in votes[c]})
        if len(items) < 2 or len(subjects) < 2:
            return BaselineOutput([])
        y = np.array([[float(votes[c].get(s) == inp.data.cells[c].anchor) for c in items]
                      for s in subjects])
        seen = np.array([[float(s in votes[c]) for c in items] for s in subjects])
        acc = (y.sum(1) + 0.5) / (seen.sum(1) + 1.0)
        theta = np.log(acc / (1 - acc))
        theta = (theta - theta.mean()) / (theta.std() or 1.0)
        by_cell = inp.data.by_cell()
        out: list[Defect] = []
        for j, cid in enumerate(items):
            mask = seen[:, j] > 0
            values = {r.value for r in by_cell[cid]}
            c = 1.0 / max(len(values), 1)
            score = delta_loglik(y[mask, j], seen[mask, j], theta[mask], c)
            if score > 0:
                cell = inp.data.cells[cid]
                off = [r for r in by_cell[cid] if r.value != cell.anchor]
                out.append(Defect("hidden", (cell.clause,), tuple(sorted({r.scenario for r in off})),
                                  tuple(sorted(r.run_key for r in off)), score))
        return BaselineOutput(merge_max(out))


register(IrtAudit())
```


- [ ] **Step 5: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/baselines/test_irt_audit.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add specdx/baselines/irt_audit.py tests/baselines/test_irt_audit.py
git commit -m "feat: B3 forced-ceiling IRT indicator (Land & Bikel), grid-fitted"
```

---

### Task 7: B4 semantic entropy with the shared ddmin localisation

**Files:**
- Create: `specdx/baselines/semantic_entropy.py`
- Test: `tests/baselines/test_semantic_entropy.py`

**Interfaces:**
- Consumes:
  - `BaselineInput` (Task 4);
  - `oracle_interpreters` (`specdx.active.loop`);
  - `RunBudget` (`specdx.active.budget`);
  - `run_cost` (Task 1);
  - `inp.ctx.localizer(variant, defect, interps, budget, persists) -> tuple[str, ...] | None`;
  - `inp.ctx.cells.extract(runs) -> CellData`.
- Produces:
  - `entropy(values: Sequence[str]) -> float`;
  - `cell_score(readings) -> float`, the maximum of the pooled entropy and the per-family entropies;
  - the registered baseline `"semantic_entropy"`, reporting `("ambiguity",)`. Each defect's location is the localised sentence ids, or `()` when it stays unlocalised. Its `note` is `"scenario|step"`.

- [ ] **Step 1: Write the failing tests**

`tests/baselines/test_semantic_entropy.py`:
```python
import math
from types import SimpleNamespace

from specdx.active.loop import SpecDxConfig
from specdx.baselines.semantic_entropy import REGISTRY, cell_score, entropy
from specdx.infer.cells import Cell, CellData, Reading


def test_entropy() -> None:
    assert entropy(["a", "a"]) == 0.0
    assert math.isclose(entropy(["a", "b"]), math.log(2))


def test_a_split_cell_scores_above_a_consensus_cell() -> None:
    split = [Reading("c", f"r{i}", "m", "f", "s", v) for i, v in enumerate("abab")]
    same = [Reading("c", f"r{i}", "m", "f", "s", "a") for i in range(4)]
    assert cell_score(split) > cell_score(same) == 0.0


def _inp(localizer):
    cells = {"c1": Cell("c1", "value", "d", "golden:c1", "t.col1", "t.col1", ("S1",)),
             "c2": Cell("c2", "value", "d", "golden:c2", "t.col2", "t.col2", ("S1",))}
    readings = ([Reading("c1", f"a{i}", "m1", "f", "happy", v) for i, v in enumerate("abab")]
                + [Reading("c2", f"b{i}", "m1", "f", "happy", v) for i, v in enumerate("aaab")])
    ctx = SimpleNamespace(localizer=localizer, cells=None)
    cfg = SpecDxConfig({"f": ("m1",)}, localize_budget=80)
    return SimpleNamespace(data=CellData(cells, readings), ctx=ctx, config=cfg,
                           variant="V", report=SimpleNamespace(theta={"m1": 0.9}))


def test_candidates_are_localised_highest_score_first() -> None:
    order = []

    def localizer(variant, defect, interps, budget, persists):
        order.append(defect.note)
        return ("KB:KB-0003:02",)

    out = REGISTRY["semantic_entropy"].run(_inp(localizer))
    assert [d.location for d in out.defects] == [("KB:KB-0003:02",), ("KB:KB-0003:02",)]
    assert out.defects[0].score > out.defects[1].score
    assert order == ["happy|S1", "happy|S1"]


def test_an_unlocalised_candidate_keeps_an_empty_location() -> None:
    out = REGISTRY["semantic_entropy"].run(_inp(lambda *a: None))
    assert all(d.location == () for d in out.defects)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/baselines/test_semantic_entropy.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`specdx/baselines/semantic_entropy.py`:
```python
"""B4: discrete semantic entropy (Farquhar et al., Nature 2024; baselines spec §2.7).

Readings are clustered by exact normalised value (cells hold short values, so meaning
clusters are value clusters). A cell's score is the larger of its pooled entropy and its
per-family entropies. Candidates are localised in T with the same ddmin localiser and
budget as SpecDx (v3 §5.3), highest score first until the budget runs out; the oracle
passes while the re-runs still read the cell two or more ways."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from specdx.active.budget import RunBudget
from specdx.active.loop import oracle_interpreters
from specdx.baselines.api import REGISTRY, BaselineInput, BaselineOutput, register
from specdx.cost import Cost, run_cost
from specdx.infer.cells import Reading
from specdx.types import Defect

__all__ = ["REGISTRY", "cell_score", "entropy"]


def entropy(values: Sequence[str]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return -sum((k / n) * math.log(k / n) for k in Counter(values).values())


def cell_score(readings: Sequence[Reading]) -> float:
    pooled = entropy([r.value for r in readings])
    fams: dict[str, list[str]] = {}
    for r in readings:
        fams.setdefault(r.family, []).append(r.value)
    return max([pooled, *(entropy(v) for v in fams.values())])


@dataclass(frozen=True)
class SemanticEntropy:
    name: str = "semantic_entropy"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("ambiguity",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        by_cell = inp.data.by_cell()
        scored = []
        for cid, rs in by_cell.items():
            s = cell_score(rs)
            if s > 0:
                cell = inp.data.cells[cid]
                scenario = Counter(r.scenario for r in rs).most_common(1)[0][0]
                note = f"{scenario}|{cell.steps[0]}" if cell.steps else f"{scenario}|"
                scored.append((s, cid, Defect("ambiguity", (cell.location,), (scenario,),
                                              tuple(sorted(r.run_key for r in rs)), s, note)))
        scored.sort(key=lambda t: (-t[0], t[1]))
        cfg = inp.config
        budget = RunBudget(0, cfg.localize_budget)
        acted = {r.model for r in inp.data.readings}
        interps = oracle_interpreters(cfg.models_by_family, inp.report.theta, cfg.variants, acted)
        spent: list[Cost] = []
        out: list[Defect] = []
        for _, cid, d in scored:
            def persists(runs, cid=cid) -> bool:
                spent.extend(run_cost(b) for b in runs)
                data = inp.ctx.cells.extract(runs)
                return len({r.value for r in data.by_cell().get(cid, [])}) >= 2

            where = None
            if not budget.exhausted("ddmin") and inp.ctx.localizer is not None:
                where = inp.ctx.localizer(inp.variant, d, interps, budget, persists)
            out.append(Defect(d.type, tuple(where or ()), d.scenarios, d.witnesses, d.score,
                              d.note))
        return BaselineOutput(out, sum(spent, Cost()), extra={"ddmin_runs": budget.ddmin})


register(SemanticEntropy())
```


- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/baselines/test_semantic_entropy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add specdx/baselines/semantic_entropy.py tests/baselines/test_semantic_entropy.py
git commit -m "feat: B4 semantic entropy, localised by SpecDx's ddmin under the same budget"
```

---

### Task 8: B5, MUS on C by golden-state neighbourhood

**Files:**
- Create: `specdx/baselines/mus_c.py`
- Test: `tests/baselines/test_mus_c.py`

**Interfaces:**
- Consumes:
  - `propose(clause, state, scenario) -> list[State]` (`specdx.active.refute`);
  - `ClauseSet.violations(state, scenario)`, `ClauseSet.__getitem__`, `ClauseSet.anchor`;
  - `pysat.examples.hitman.Hitman`.
- Produces:
  - `explore(start, violations: Callable[[S], frozenset[str]], neighbours: Callable[[S, str], list[S]], key: Callable[[S], Hashable], max_evals: int = 200, depth: int = 2) -> list[frozenset[str]]`;
  - `minimal_hitting_sets(sets, max_size=3) -> list[tuple[str, ...]]`;
  - the registered baseline `"b5_mus"`, reporting `("contradiction",)`.

- [ ] **Step 1: Write the failing tests**

`tests/baselines/test_mus_c.py`:
```python
import pytest

from specdx.baselines.mus_c import explore, minimal_hitting_sets


def test_a_planted_pair_conflict_is_found() -> None:
    # state "s0" violates a; repairing a gives "s1", which violates b; repairing b gives s0
    viol = {"s0": frozenset({"a"}), "s1": frozenset({"b"})}
    nxt = {("s0", "a"): ["s1"], ("s1", "b"): ["s0"]}
    sets = explore("s0", viol.__getitem__, lambda s, c: nxt.get((s, c), []), key=str)
    assert minimal_hitting_sets(sets) == [("a", "b")]


def test_a_consistent_specification_has_no_mus() -> None:
    sets = explore("ok", lambda s: frozenset(), lambda s, c: [], key=str)
    assert minimal_hitting_sets(sets) == []


def test_exploration_respects_the_evaluation_budget() -> None:
    count = {"n": 0}

    def viol(s):
        count["n"] += 1
        return frozenset({"a"})

    explore(0, viol, lambda s, c: [s + 1, s + 2], key=str, max_evals=10)
    assert count["n"] <= 10


@pytest.mark.db
def test_b5_runs_on_the_base_specification(db, pack) -> None:
    from types import SimpleNamespace

    from specdx.baselines.api import REGISTRY
    from specdx.orchestrate.work_orders import load_work_orders
    from specdx.spec.clauses import ClauseSet

    clauses = ClauseSet.from_pack(pack, load_work_orders())
    out = REGISTRY["b5_mus"].run(SimpleNamespace(clauses=clauses, golden=db.golden_state()))
    assert out.failure is None
    assert all(d.type == "contradiction" for d in out.defects)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/baselines/test_mus_c.py -v -m "not db"`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`specdx/baselines/mus_c.py`:
```python
"""B5: contradictions in C alone, with no interpreters (baselines spec §9.3; v1 §4.5 B5).

From the canon's golden end state (materialised by PostgreSQL), explore up to two rounds
of single-clause repairs (`propose`), at most 200 clause-set evaluations. Each explored
state's violation set is a correction set; the minimal hitting sets of size ≤ 3 of those
sets (PySAT Hitman, smallest first) are the MUS candidates, reported as contradictions."""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pysat.examples.hitman import Hitman

from specdx.active.refute import propose
from specdx.baselines.api import BaselineInput, BaselineOutput, register
from specdx.types import Defect, State

S = TypeVar("S")
MAX_EVALS, DEPTH, MAX_SIZE = 200, 2, 3


def explore(start: S, violations: Callable[[S], frozenset[str]],
            neighbours: Callable[[S, str], list[S]], key: Callable[[S], Hashable],
            max_evals: int = MAX_EVALS, depth: int = DEPTH) -> list[frozenset[str]]:
    first = violations(start)
    evals, seen, sets = 1, {key(start)}, [first]
    frontier = [(start, first)]
    for _ in range(depth):
        nxt = []
        for state, bad in frontier:
            for cid in sorted(bad):
                for cand in neighbours(state, cid):
                    k = key(cand)
                    if k in seen:
                        continue
                    if evals >= max_evals:
                        return sets
                    seen.add(k)
                    evals += 1
                    v = violations(cand)
                    sets.append(v)
                    nxt.append((cand, v))
        frontier = nxt
    return sets


def minimal_hitting_sets(sets: Sequence[frozenset[str]],
                         max_size: int = MAX_SIZE) -> list[tuple[str, ...]]:
    if not sets or any(not s for s in sets):  # a consistent state exists: no MUS
        return []
    out: list[tuple[str, ...]] = []
    with Hitman(bootstrap_with=[sorted(s) for s in sets], htype="sorted") as h:
        for hs in h.enumerate():
            if len(hs) > max_size:
                break
            out.append(tuple(sorted(hs)))
    return sorted(set(out), key=lambda t: (len(t), t))


def _key(state: State) -> str:
    return json.dumps([state.tables, state.derived, state.steps], sort_keys=True,
                      ensure_ascii=False)


@dataclass(frozen=True)
class MusOnC:
    name: str = "b5_mus"
    uses_llm: bool = False
    reports: tuple[str, ...] = ("contradiction",)
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        if inp.golden is None:
            return BaselineOutput([], failure="no golden state (needs PostgreSQL)")
        clauses, scenario = inp.clauses, inp.clauses.anchor
        sets = explore(inp.golden, lambda s: clauses.violations(s, scenario),
                       lambda s, cid: propose(clauses[cid], s, scenario), _key)
        hs = minimal_hitting_sets(sets)
        return BaselineOutput([Defect("contradiction", h, (scenario.id,), (), 1.0 / len(h))
                               for h in hs], extra={"explored": len(sets)})


register(MusOnC())
```


- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/baselines/test_mus_c.py -v`
Expected: PASS, including the db test, with PostgreSQL up (`docker start specdx-postgres`).

- [ ] **Step 5: Commit**

```bash
git add specdx/baselines/mus_c.py tests/baselines/test_mus_c.py
git commit -m "feat: B5 contradictions in C by golden-neighbourhood exploration and PySAT hitting sets"
```

---

### Task 9: B2, Who&When failure attribution

**Files:**
- Create: `specdx/baselines/whowhen.py`
- Test: `tests/baselines/test_whowhen.py`

**Interfaces:**
- Consumes:
  - `BaselineInput` with a `gateway` (Tasks 3–4);
  - `tell_excerpt`, `scenario_text` (Task 4);
  - `describe(clause, scenario)` (`specdx.spec.clauses`).
- Produces:
  - `sample_runs(violated: Mapping[str, frozenset[str]], ok: set[str], k=5) -> dict[str, list[str]]`, mapping clause → run keys;
  - `failure_log(b, steps, drafts) -> list[dict[str, str]]`, each entry `{"name": step id, "content": text}`;
  - `parse_verdict(text, violated) -> str | None`, returning `"agent_error"`, a clause id, or `None` when the text cannot be parsed;
  - the registered baseline `"whowhen"`, reporting `("hidden",)`, with `strategies = ("all_at_once", "step_by_step", "binary_search")`.

- [ ] **Step 1: Write the failing tests**

`tests/baselines/test_whowhen.py`:
```python
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


def test_an_unparsable_reply_is_an_error_not_a_finding(monkeypatch) -> None:
    from specdx.baselines import whowhen
    from specdx.baselines.llm import Reply
    from specdx.cost import Cost

    class G:
        def chat(self, model, messages, **kw):
            return Reply("no idea", Cost(1), "ok")

    inp = whowhen._toy_input(G())  # helper defined in the module for tests (see Step 3)
    out = whowhen.REGISTRY["whowhen"].run(inp, model="m", strategy="all_at_once")
    assert out.defects == [] and out.extra["unparsed"] >= 1
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/baselines/test_whowhen.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`specdx/baselines/whowhen.py`:

The three localisation prompts are copied **verbatim** from `baseline/05_who_and_when/code/Automated_FA/Lib/utils.py` at commit `b2bae5c` (MIT; keep its copyright notice in the docstring). Only `{problem}`, `{answer}` and the chat content are filled from SpecDx. The verdict prompt is new and is ours.

```python
"""B2: Who&When automated failure attribution (Zhang et al., ICML 2025; baselines spec
§2.5, §9.5), on SpecDx's failing runs.

Localisation prompts are ported verbatim from github.com/mingyin1/Agents_Failure_Attribution
(commit b2bae5c, MIT licence, Copyright (c) the Who&When authors). A run becomes a
multi-agent log: one entry per work step, in flow order (agent = the step's digital
employee or actor). After a strategy names the decisive step, one added verdict call asks
whether the agent erred or a failed check is wrong. A clause's score is the share of its
sampled runs judged a specification defect at that clause."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from specdx.baselines.api import (
    REGISTRY, BaselineInput, BaselineOutput, register, scenario_text, tell_excerpt,
)
from specdx.cost import Cost
from specdx.spec.clauses import describe
from specdx.types import Behaviour, Defect

__all__ = ["REGISTRY", "failure_log", "parse_verdict", "sample_runs"]

K_RUNS = 5
MAX_TOKENS = 1024
ROW_CHARS = 1200


def _h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def sample_runs(violated: Mapping[str, frozenset[str]], ok: set[str],
                k: int = K_RUNS) -> dict[str, list[str]]:
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


def parse_verdict(text: str, violated: frozenset[str]) -> str | None:
    m = _VERDICT.search(text or "")
    if not m:
        return None
    if m.group(1).lower() == "agent_error":
        return "agent_error"
    return m.group(2) if m.group(2) in violated else None


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


def _binary_prompt(problem, answer, chat, rng_desc, upper, lower) -> str:
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


@dataclass(frozen=True)
class WhoWhen:
    name: str = "whowhen"
    uses_llm: bool = True
    reports: tuple[str, ...] = ("hidden",)
    strategies: tuple[str | None, ...] = ("all_at_once", "step_by_step", "binary_search")

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        assert inp.gateway is not None and model and strategy in self.strategies
        ok = {b.run_key for b in inp.report.behaviours if b.status == "ok"}
        runs = {b.run_key: b for b in inp.report.behaviours}
        picked = sample_runs(inp.violated, ok)
        excerpt = "\n".join(f"[T:{sid}] {t}" for sid, t in tell_excerpt(inp))
        cost, verdicts, unparsed = Cost(), {}, 0

        def ask(system: str, prompt: str) -> str:
            nonlocal cost
            r = inp.gateway.chat(model, [{"role": "system", "content": system},
                                         {"role": "user", "content": prompt}],
                                 options={"seed": 1}, max_tokens=MAX_TOKENS)
            cost = cost + r.cost
            return r.text if r.status == "ok" else ""

        for key in sorted({r for rs in picked.values() for r in rs}, key=_h):
            b = runs[key]
            failed = inp.violated[key]
            log = failure_log(b, inp.steps, inp.drafts)
            if not log:
                unparsed += 1
                continue
            problem = f"{scenario_text(b.scenario)}\nInstructions (excerpt):\n{excerpt}"
            answer = "; ".join(describe(inp.clauses[c], b.scenario) for c in sorted(failed))
            step = self._localise(strategy, log, problem, answer, ask, key)
            entry = log[min(step, len(log) - 1)]
            checks = "\n".join(f"- {c}: {describe(inp.clauses[c], b.scenario)}"
                               for c in sorted(failed))
            text = ask(_SYSTEM["all_at_once"], VERDICT_PROMPT.format(
                problem=problem, answer=answer, step=step, agent=entry["name"],
                content=entry["content"], checks=checks))
            v = parse_verdict(text, failed)
            if v is None:
                unparsed += 1
            verdicts[key] = v
        out = []
        for clause, keys in sorted(picked.items()):
            share = sum(verdicts.get(k) == clause for k in keys) / len(keys)
            if share > 0:
                hits = [k for k in keys if verdicts.get(k) == clause]
                out.append(Defect("hidden", (clause,),
                                  tuple(sorted({runs[k].scenario.id for k in hits})),
                                  tuple(hits), share))
        return BaselineOutput(sorted(out, key=lambda d: (-d.score, d.location)), cost,
                              extra={"judged": len(verdicts), "unparsed": unparsed})

    def _localise(self, strategy, log, problem, answer, ask, key) -> int:
        chat = "\n".join(f"{e['name']}: {e['content']}" for e in log)
        if strategy == "all_at_once":
            m = _STEP.search(ask(_SYSTEM[strategy], _all_at_once_prompt(problem, answer, chat)))
            return int(m.group(1)) if m else 0
        if strategy == "step_by_step":
            history = ""
            for idx, e in enumerate(log):
                history += f"Step {idx} - {e['name']}: {e['content']}\n"
                a = ask(_SYSTEM[strategy], _step_prompt(problem, answer, history, idx, e["name"]))
                if a.lower().strip().startswith("1. yes"):
                    return idx
            return len(log) - 1
        rng = random.Random(_h(key))
        lo, hi = 0, len(log) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            seg = "\n".join(f"{e['name']}: {e['content']}" for e in log[lo:hi + 1])
            a = ask(_SYSTEM[strategy], _binary_prompt(
                problem, answer, seg, f"from step {lo} to step {hi}",
                f"from step {lo} to step {mid}", f"from step {mid + 1} to step {hi}")).lower()
            upper = "upper half" in a if ("upper half" in a) != ("lower half" in a) \
                else rng.randint(0, 1) == 0
            lo, hi = (lo, mid) if upper else (min(mid + 1, hi), hi)
        return lo


def _toy_input(gateway: Any) -> SimpleNamespace:
    """A one-run input for unit tests (no database, no corpus)."""
    from specdx.types import InterpreterId, Scenario, State

    b = Behaviour("k1", InterpreterId("m", "f", 0, 1), Scenario("happy", 15, 15, 0),
                  State({}, steps={"S1": "done"}), claims={"S1": "done it"})
    from specdx.spec.clauses import Clause

    clause = Clause("inv:x", "blank", "all", None, frozenset(), "x holds", {})
    return SimpleNamespace(
        gateway=gateway, report=SimpleNamespace(behaviours=[b]),
        violated={"k1": frozenset({"inv:x"})}, steps=[SimpleNamespace(id="S1", intent="i",
                                                                      tables=frozenset())],
        drafts={}, statements={}, clauses={"inv:x": clause},
        variant=SimpleNamespace(corpus=SimpleNamespace(search=lambda q, k=5: [],
                                                       sentence_ids=lambda c: [])))


register(WhoWhen())
```

The toy clause is a real `Clause` of a kind `describe` answers with its description, so `describe` needs no change.


- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/baselines/test_whowhen.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add specdx/baselines/whowhen.py tests/baselines/test_whowhen.py
git commit -m "feat: B2 Who&When failure attribution on SpecDx runs (verbatim prompts + verdict)"
```

---

### Task 10: B1, BenchGuard

**Files:**
- Create: `specdx/baselines/benchguard.py`
- Test: `tests/baselines/test_benchguard.py`

**Interfaces:**
- Consumes:
  - `BaselineInput` with a `gateway`;
  - `tell_excerpt`, `scenario_text`;
  - `describe`;
  - BenchGuard: `BenchmarkLoader`, `VerificationPipeline`, `AuditMode`, `LLMConfig`, `LLMClient`, `LLMResponse`, `ProtocolRunner`, `Finding`.
- Produces:
  - `package(inp, root: Path) -> dict[str, str]`, mapping each written test file name to its clause id;
  - `map_finding(f: Finding, files: Mapping[str, str], corpus_ids: set[str]) -> list[Defect]`;
  - `SEVERITY`, `CONFIDENCE`, `MAPPING`, `WEAK_RE` (fixed now);
  - `GatewayClient(LLMClient)`;
  - the registered baseline `"benchguard"`, reporting all four types.

- [ ] **Step 1: Write the failing tests**

`tests/baselines/test_benchguard.py`:
```python
from benchguard.core.finding import Evidence, Finding, Location
from benchguard.taxonomy.categories import ErrorCategory, Severity

from specdx.baselines.benchguard import MAPPING, map_finding


def _f(sub, cat, files=(), text="", conf=0.9, sev=Severity.HIGH):
    loc = Location(file=files[0]) if files else None
    loc2 = Location(file=files[1]) if len(files) > 1 else None
    return Finding(task_id="happy", category=cat, subcategory=sub, severity=sev, title="t",
                   description=text, confidence=conf,
                   evidence=Evidence(primary_source=loc, conflicting_source=loc2))


FILES = {"tests/inv__po-15.txt": "inv:po-15", "tests/inv__grn-12.txt": "inv:grn-12"}


def test_every_mapped_subcategory_is_known() -> None:
    assert set(MAPPING) >= {"INST-INCOMPLETE", "INST-CONTRADICT", "INST-INFEASIBLE", "GT-LOGIC",
                            "GT-DATA", "GT-FMT", "EVAL-MISMATCH", "EVAL-COVERAGE",
                            "EVAL-JUDGE-BIAS", "EVAL-TOLERANCE"}


def test_gt_maps_to_hidden_at_the_cited_clause() -> None:
    [d] = map_finding(_f("GT-LOGIC", ErrorCategory.GT, ["tests/inv__po-15.txt"]), FILES, set())
    assert (d.type, d.location) == ("hidden", ("inv:po-15",)) and d.score == 3 * 3


def test_contradiction_needs_two_cited_clauses() -> None:
    two = map_finding(_f("INST-CONTRADICT", ErrorCategory.INST,
                         ["tests/inv__po-15.txt", "tests/inv__grn-12.txt"]), FILES, set())
    one = map_finding(_f("INST-CONTRADICT", ErrorCategory.INST, ["tests/inv__po-15.txt"]),
                      FILES, set())
    assert two[0].type == "contradiction" and two[0].location == ("inv:grn-12", "inv:po-15")
    assert one[0].type == "hidden"


def test_a_lenient_eval_also_reports_a_weak_check_on_the_column() -> None:
    files = {"tests/golden__po__PO-0015__don_gia.txt": "golden:po:PO-0015:don_gia"}
    ds = map_finding(_f("EVAL-MISMATCH", ErrorCategory.EVAL,
                        ["tests/golden__po__PO-0015__don_gia.txt"],
                        text="The check accepts any price"), files, set())
    assert {(d.type, d.location) for d in ds} == {
        ("hidden", ("golden:po:PO-0015:don_gia",)), ("weak", ("po.don_gia",))}


def test_incomplete_instructions_map_to_cited_sentences() -> None:
    ds = map_finding(_f("INST-INCOMPLETE", ErrorCategory.INST,
                        text="[T:KB:KB-0003:02] does not say the cap"), {}, {"KB:KB-0003:02"})
    assert [(d.type, d.location) for d in ds] == [("ambiguity", ("KB:KB-0003:02",))]


def test_env_findings_are_dropped() -> None:
    assert map_finding(_f("ENV-DEP", ErrorCategory.ENV), FILES, set()) == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/baselines/test_benchguard.py -v`
Expected: FAIL with `ModuleNotFoundError: specdx.baselines.benchguard`.

- [ ] **Step 3: Implement**

`specdx/baselines/benchguard.py`:
```python
"""B1: BenchGuard (Tu et al., arXiv 2604.24955; COLM 2026), definition audit only
(baselines spec §2.3, §9.4). Each specification becomes a BenchGuard benchmark with one
task per canon scenario: instruction = scenario + numbered T excerpt; tests = one file per
clause that applies; solution = the golden state. Every LLM call goes through the
gateway via GatewayClient; BenchGuard's code is unchanged. Findings map to SpecDx types
by the frozen MAPPING."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from benchguard.agent.client import LLMClient, LLMConfig, LLMResponse
from benchguard.agent.runner import ProtocolRunner
from benchguard.core.finding import Finding
from benchguard.core.pipeline import AuditMode, VerificationPipeline
from benchguard.loader import BenchmarkLoader

from specdx.baselines.api import (
    BaselineInput, BaselineOutput, register, scenario_text, tell_excerpt,
)
from specdx.cost import Cost
from specdx.spec.clauses import describe
from specdx.types import Defect

SEVERITY = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
CONFIDENCE = {"CONFIRMED": 3, "LIKELY": 2, "POSSIBLE": 1}
MAPPING = {
    "INST-INCOMPLETE": "ambiguity", "INST-CONTRADICT": "contradiction_or_hidden",
    "INST-INFEASIBLE": "contradiction", "GT-LOGIC": "hidden", "GT-DATA": "hidden",
    "GT-FMT": "hidden", "EVAL-MISMATCH": "hidden+weak", "EVAL-COVERAGE": "hidden+weak",
    "EVAL-JUDGE-BIAS": "hidden+weak", "EVAL-TOLERANCE": "hidden+weak",
}
WEAK_RE = re.compile(r"\b(accept|accepts|pass(es)?|allow(s)?|lenient|loose|permissive)\b", re.I)
SENT_RE = re.compile(r"\[T:([^\]]+)\]")
MAX_TOKENS = 4096


def _fname(clause_id: str) -> str:
    return "tests/" + re.sub(r"[^A-Za-z0-9._-]", "_", clause_id.replace(":", "__")) + ".txt"


def _column(clause_id: str) -> str | None:
    """golden:<table>:<key>:<column> -> table.column; other clauses have no column."""
    parts = clause_id.split(":")
    return f"{parts[1]}.{parts[-1]}" if parts[0] == "golden" and len(parts) >= 4 else None


def package(inp: BaselineInput, root: Path) -> dict[str, str]:
    if root.exists():
        shutil.rmtree(root)
    files: dict[str, str] = {}
    excerpt = "\n".join(f"[T:{sid}] {t}" for sid, t in tell_excerpt(inp))
    for name, scenario in sorted(inp.ctx.canon.items()):
        task = root / name
        (task / "tests").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "task.toml").write_text(
            f'[metadata]\nid = "{name}"\ncategory = "business-process"\n'
            'benchmark_source = "TonomyX"\n', encoding="utf-8")
        (task / "instruction.md").write_text(
            f"{scenario_text(scenario)}\n\nRequirements (numbered sentences):\n{excerpt}\n",
            encoding="utf-8")
        for c in inp.clauses:
            if inp.clauses.applies(c, scenario):
                rel = _fname(c.id)
                (task / rel).write_text(f"{c.id}\n{describe(c, scenario)}\n"
                                        f"{json.dumps(c.payload, ensure_ascii=False)}\n",
                                        encoding="utf-8")
                files[rel] = c.id
        golden = inp.golden.tables if inp.golden is not None else {}
        (task / "solution" / "golden_state.json").write_text(
            json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")
    return files


def _cited(f: Finding, files: Mapping[str, str]) -> list[str]:
    out = []
    for src in (f.evidence.primary_source, f.evidence.conflicting_source):
        if src is not None and src.file:
            for rel, cid in files.items():
                if src.file.endswith(rel) and cid not in out:
                    out.append(cid)
    return out


def map_finding(f: Finding, files: Mapping[str, str], corpus_ids: set[str]) -> list[Defect]:
    rule = MAPPING.get(f.subcategory)
    if rule is None:
        return []
    score = float(SEVERITY[f.severity.value] * CONFIDENCE[f.confidence_level.value])
    text = " ".join([f.title, f.description, f.evidence.description])
    clauses = _cited(f, files)
    out: list[Defect] = []
    if rule == "ambiguity":
        sents = tuple(sorted(s for s in SENT_RE.findall(text) if s in corpus_ids))
        out.append(Defect("ambiguity", sents, (f.task_id,), (), score, f.subcategory))
    elif rule == "contradiction" or (rule == "contradiction_or_hidden" and len(clauses) >= 2):
        out.append(Defect("contradiction", tuple(sorted(clauses)), (f.task_id,), (), score,
                          f.subcategory))
    elif clauses:
        for cid in clauses:
            out.append(Defect("hidden", (cid,), (f.task_id,), (), score, f.subcategory))
            col = _column(cid)
            if rule == "hidden+weak" and col and WEAK_RE.search(text):
                out.append(Defect("weak", (col,), (f.task_id,), (), score, f.subcategory))
    return out


class GatewayClient(LLMClient):
    def __init__(self, gateway, model: str) -> None:
        super().__init__(LLMConfig(model=f"ollama/{model}", temperature=0.0,
                                   max_tokens=MAX_TOKENS))
        self.gateway, self.model, self.cost = gateway, model, Cost()

    def complete(self, prompt, system_prompt="", temperature=None, max_tokens=None):
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}]
        t = 0.0 if temperature is None else temperature
        r = self.gateway.chat(self.model, msgs, options={"temperature": t, "seed": 1},
                              max_tokens=max_tokens or MAX_TOKENS)
        self.cost = self.cost + r.cost
        if r.status != "ok":
            raise RuntimeError(f"gateway error: {r.error}")
        return LLMResponse(content=r.text, model=self.model, tokens_prompt=r.cost.prompt_tokens,
                           tokens_completion=r.cost.eval_tokens)

    async def acomplete(self, prompt, system_prompt="", temperature=None, max_tokens=None):
        return self.complete(prompt, system_prompt, temperature, max_tokens)


@dataclass(frozen=True)
class BenchGuard:
    name: str = "benchguard"
    uses_llm: bool = True
    reports: tuple[str, ...] = ("contradiction", "hidden", "ambiguity", "weak")
    strategies: tuple[str | None, ...] = (None,)

    def run(self, inp: BaselineInput, *, model=None, strategy=None) -> BaselineOutput:
        assert inp.gateway is not None and model
        root = inp.gateway.cache_path.parent / "baselines" / "benchguard" / inp.variant.name
        files = package(inp, root)
        client = GatewayClient(inp.gateway, model)
        pipe = VerificationPipeline(BenchmarkLoader(root), llm_config=client.config,
                                    audit_mode=AuditMode.DEFINITION, agent_evidence="off",
                                    min_confidence=None, generate_fixes=False)
        pipe.runner = ProtocolRunner(client=client)
        try:
            report = pipe.audit(max_concurrent=1)
        except Exception as exc:  # noqa: BLE001 - a broken audit fails this spec only
            return BaselineOutput([], client.cost, failure=f"{type(exc).__name__}: {exc}")
        ids = set(inp.variant.corpus.sentence_ids())
        out = [d for f in report.findings for d in map_finding(f, files, ids)]
        merged: dict[tuple[str, tuple[str, ...]], Defect] = {}
        for d in out:
            if d.key not in merged or d.score > merged[d.key].score:
                merged[d.key] = d
        return BaselineOutput(sorted(merged.values(), key=lambda d: (-d.score, d.location)),
                              client.cost, extra={"findings": len(report.findings)})


register(BenchGuard())
```


- [ ] **Step 4: Check that a malformed protocol reply does not stop the audit**

Append to `tests/baselines/test_benchguard.py`:
```python
def test_a_malformed_llm_reply_yields_no_finding(tmp_path, tell_corpus, pack, golden) -> None:
    from types import SimpleNamespace

    from specdx.baselines.api import REGISTRY
    from specdx.baselines.llm import Reply
    from specdx.config import BASE_PACK
    from specdx.cost import Cost
    from specdx.interpreters.prompt import load_variants
    from specdx.orchestrate.work_orders import load_work_orders
    from specdx.scenarios.space import canon_scenarios
    from specdx.spec.clauses import ClauseSet
    from specdx.variant import SpecVariant

    class G:
        cache_path = tmp_path / "c.jsonl"

        def chat(self, model, messages, **kw):
            return Reply("not json at all", Cost(1), "ok")

    steps = load_work_orders()
    inp = SimpleNamespace(
        gateway=G(), variant=SpecVariant("base", BASE_PACK, tell_corpus),
        ctx=SimpleNamespace(canon=canon_scenarios(pack)), clauses=ClauseSet.from_pack(pack, steps),
        steps=steps, statements=load_variants().statements, golden=golden)
    out = REGISTRY["benchguard"].run(inp, model="m")
    assert out.defects == [] and out.cost.requests >= 1
```
This test is marked `db` by its `golden` fixture.

Run: `.venv/bin/python -m pytest tests/baselines/test_benchguard.py -v`
Expected: PASS. BenchGuard's protocols log the parse failure and return no finding. If the pipeline instead raises, `run` must still return `failure` set and `defects == []`. Adjust the assertion to `out.defects == []` only, and keep the `failure` path.

- [ ] **Step 5: Commit**

```bash
git add specdx/baselines/benchguard.py tests/baselines/test_benchguard.py
git commit -m "feat: B1 BenchGuard definition audit via the gateway, with the frozen finding mapping"
```

---

### Task 11: The split runner and scoring (`specdx/eval/study.py`)

**Files:**
- Create: `specdx/eval/study.py`
- Test: `tests/eval/test_study.py`

**Interfaces:**
- Consumes:
  - `run_specdx`, `SpecDxConfig`, `Report` (loop);
  - `Workbench` (`bench.loop_context`, `bench.base`, `bench.steps`, `bench.variants`, `bench.db`);
  - `GateConfig`, `Mutant`, `GroundTruth`;
  - `evaluate_reports`, `match_findings`, `Prf`;
  - `touch_counts`, `scorable_ids` (`specdx.eval.observability`, as in `run_gate`);
  - `REGISTRY`, `BaselineInput`, `above`;
  - `behaviours_cost`;
  - `draft_tables` (`specdx.infer.cells`);
  - `ClauseSet`; `Gateway`.
- Produces:
  - `CHECKPOINTS = (25, 50, 100, 150, 200)`;
  - `Arm(name: str, model: str | None = None, strategy: str | None = None)` with `.label`;
  - `dev_arms(config) -> list[Arm]`;
  - `make_input(bench, variant, report, ctx, cfg, gateway) -> BaselineInput`;
  - `run_split(bench, config, mutants, arms, gateway, out_lines: Path, split: str) -> SplitResult`;
  - `SplitResult` with fields `truths`, `scorable`, `defects: dict[arm_label, dict[spec, list[Defect]]]` (where `"specdx"` is an arm), `costs`, `failures`, `checkpoints: dict[spec, list[dict]]`, and methods `to_json()` and `from_json()`;
  - `score(result, arm, tau=0.0, primary=True, typed=True) -> dict[str, Prf]`;
  - `rank_metrics(result, arm, tau=0.0, ks=(1, 3, 5)) -> dict[type, {"ap": float, "detect@k": float}]`;
  - `equal_cost(result, arm) -> dict[spec, list[Defect]]`, giving SpecDx at the checkpoint whose requests ≤ the arm's requests on that spec.

- [ ] **Step 1: Write the failing tests (pure scoring, on a fabricated `SplitResult`)**

`tests/eval/test_study.py`:
```python
import json

from specdx.eval.study import Arm, SplitResult, equal_cost, rank_metrics, score
from specdx.mutate.operators import GroundTruth
from specdx.types import Defect


def _t(type_, loc, kind, mid="M"):
    return GroundTruth(mid, type_, frozenset({loc}), kind)


def _d(type_, loc, s):
    return Defect(type_, (loc,), (), (), s)


def _result():
    truths = {"H1": _t("hidden", "c1", "clause", "H1"), "H2": _t("hidden", "c2", "clause", "H2")}
    defects = {
        "specdx": {"base": [], "H1": [_d("hidden", "c1", 0.9)], "H2": [_d("hidden", "zz", 0.8)]},
        "jest": {"base": [], "H1": [_d("hidden", "x", 0.9), _d("hidden", "c1", 0.5)],
                 "H2": []},
    }
    cps = {"H1": [{"at": 25, "runs": 25, "cost": {"requests": 40},
                   "defects": [{"type": "hidden", "location": ["c1"], "scenarios": [],
                                "witnesses": [], "score": 0.7, "note": "", "subtype": ""}]}],
           "H2": []}
    costs = {"specdx": {"H1": {"requests": 400}, "H2": {"requests": 400}},
             "jest": {"H1": {"requests": 0}, "H2": {"requests": 0}},
             "llm@m": {"H1": {"requests": 50}, "H2": {"requests": 10}}}
    defects["llm@m"] = {"base": [], "H1": [], "H2": []}
    return SplitResult("dev", truths, {"H1", "H2"}, defects, costs, {}, cps)


def test_typed_f1_uses_the_existing_matching() -> None:
    s = score(_result(), "specdx")
    assert (s["hidden"].tp, s["hidden"].fn, s["hidden"].fp) == (1, 1, 1)


def test_threshold_filters_before_scoring() -> None:
    assert score(_result(), "jest", tau=0.6)["hidden"].tp == 0
    assert score(_result(), "jest", tau=0.0)["hidden"].tp == 1


def test_ap_is_reciprocal_rank_and_detection_at_k() -> None:
    m = rank_metrics(_result(), "jest")["hidden"]
    assert m["ap"] == (1 / 2 + 0) / 2 and m["detect@1"] == 0 and m["detect@3"] == 0.5


def test_equal_cost_picks_the_checkpoint_within_the_arms_requests() -> None:
    eq = equal_cost(_result(), "llm@m")
    assert [d.location for d in eq["H1"]] == [("c1",)]   # checkpoint 40 ≤ 50 requests
    assert eq["H2"] == []                                  # no checkpoint within 10


def test_round_trip_is_byte_stable() -> None:
    r = _result()
    text = r.to_json()
    assert SplitResult.from_json(text).to_json() == text
    assert json.loads(text)["split"] == "dev"


def test_an_arm_with_no_findings_scores_all_fn() -> None:
    s = score(_result(), "llm@m")
    assert (s["hidden"].tp, s["hidden"].fn) == (0, 2)


def test_arm_labels() -> None:
    assert Arm("whowhen", "m:7b", "step_by_step").label == "whowhen@m:7b/step_by_step"
    assert Arm("jest").label == "jest"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_study.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`specdx/eval/study.py`:
```python
"""The comparison study (baselines spec §3, §4.1): SpecDx and every baseline arm on one
split, scored with the gate's own matching. A split's base is the differential's reference
for every arm. LLM arms on the dev split are (baseline, model[, strategy]) for each
population model (decision 6); on the test split, the frozen arm per baseline."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from specdx.active.loop import SpecDxConfig, run_specdx  # noqa: F401 - SpecDxConfig for tests
from specdx.baselines.api import REGISTRY, BaselineInput, BaselineOutput, above, load_all
from specdx.cost import behaviours_cost
from specdx.eval.matching import Prf, evaluate_reports, match_findings
from specdx.eval.observability import scorable_ids, touch_counts
from specdx.infer.cells import draft_tables
from specdx.mutate.operators import GroundTruth
from specdx.spec.clauses import ClauseSet
from specdx.types import DEFECT_TYPES, Defect

CHECKPOINTS: tuple[int, ...] = (25, 50, 100, 150, 200)
load_all()  # every baseline registers before arms are listed


@dataclass(frozen=True)
class Arm:
    name: str
    model: str | None = None
    strategy: str | None = None

    @property
    def label(self) -> str:
        out = self.name + (f"@{self.model}" if self.model else "")
        return out + (f"/{self.strategy}" if self.strategy else "")


def dev_arms(models_by_family: Mapping[str, Sequence[str]]) -> list[Arm]:
    models = sorted(m for ms in models_by_family.values() for m in ms)
    arms: list[Arm] = []
    for name in sorted(REGISTRY):
        b = REGISTRY[name]
        for strategy in b.strategies:
            if b.uses_llm:
                arms += [Arm(name, m, strategy) for m in models]
            else:
                arms.append(Arm(name, None, strategy))
    return arms


def _defect(raw: Mapping[str, Any]) -> Defect:
    return Defect(raw["type"], tuple(raw["location"]), tuple(raw["scenarios"]),
                  tuple(raw["witnesses"]), float(raw["score"]), raw.get("note", ""),
                  raw.get("subtype", ""))


@dataclass
class SplitResult:
    split: str
    truths: dict[str, GroundTruth]
    scorable: set[str]
    defects: dict[str, dict[str, list[Defect]]]
    costs: dict[str, dict[str, dict[str, Any]]]
    failures: dict[str, dict[str, str]]
    checkpoints: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "split": self.split,
            "truths": {k: {"type": t.type, "location": sorted(t.location),
                           "kind": t.location_kind} for k, t in sorted(self.truths.items())},
            # observed_at (scorability metadata) is not kept: scorable is stored instead.
            "scorable": sorted(self.scorable),
            "defects": {a: {s: [asdict(d) for d in ds] for s, ds in sorted(by.items())}
                        for a, by in sorted(self.defects.items())},
            "costs": {a: dict(sorted(by.items())) for a, by in sorted(self.costs.items())},
            "failures": {a: dict(sorted(by.items())) for a, by in sorted(self.failures.items())},
            "checkpoints": dict(sorted(self.checkpoints.items())),
        }, ensure_ascii=False, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SplitResult:
        raw = json.loads(text)
        return cls(raw["split"],
                   {k: GroundTruth(k, t["type"], frozenset(t["location"]), t["kind"])
                    for k, t in raw["truths"].items()},
                   set(raw["scorable"]),
                   {a: {s: [_defect(d) for d in ds] for s, ds in by.items()}
                    for a, by in raw["defects"].items()},
                   raw["costs"], raw["failures"], raw.get("checkpoints", {}))


def _reports(result: SplitResult, arm: str, tau: float, primary: bool):
    by = result.defects[arm]
    keep = (lambda k: k in result.scorable) if primary else (lambda k: True)
    reports = {k: above(v, tau) for k, v in by.items() if k != "base" and k in result.truths
               and keep(k) and k not in result.failures.get(arm, {})}
    return reports, above(by.get("base", []), tau)


def score(result: SplitResult, arm: str, tau: float = 0.0, primary: bool = True,
          typed: bool = True) -> dict[str, Prf]:
    reports, base = _reports(result, arm, tau, primary)
    return evaluate_reports(reports, result.truths, base, typed=typed)


def rank_metrics(result: SplitResult, arm: str, tau: float = 0.0,
                 ks: Sequence[int] = (1, 3, 5)) -> dict[str, dict[str, float]]:
    """One truth per mutant, so AP is the reciprocal rank of the first typed hit."""
    reports, base = _reports(result, arm, tau, primary=True)
    per: dict[str, list[int | None]] = {t: [] for t in DEFECT_TYPES}
    for k, ds in reports.items():
        truth = result.truths[k]
        new, hits = match_findings(ds, truth, base, typed=True)
        rank = next((i + 1 for i, d in enumerate(new) if d in hits), None)
        per[truth.type].append(rank)
    out = {}
    for t, ranks in per.items():
        n = len(ranks) or 1
        row = {"ap": sum(1 / r for r in ranks if r) / n}
        row.update({f"detect@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks})
        out[t] = row
    return out


def equal_cost(result: SplitResult, arm: str) -> dict[str, list[Defect]]:
    out: dict[str, list[Defect]] = {}
    for spec, cps in result.checkpoints.items():
        budget = int(result.costs.get(arm, {}).get(spec, {}).get("requests", 0))
        fit = [c for c in cps if int(c["cost"]["requests"]) <= budget]
        out[spec] = [_defect(d) for d in fit[-1]["defects"]] if fit else []
    return out


def make_input(bench, variant, report, ctx, cfg, gateway) -> BaselineInput:
    from tonomyx.canon.pack import CanonPack

    pack = CanonPack(variant.pack)
    bench.db.use(variant.pack)
    return BaselineInput(
        variant=variant, report=report, data=ctx.cells.extract(report.behaviours),
        violated={b.run_key: ctx.lens.violations(b) for b in report.behaviours},
        clauses=ClauseSet.from_pack(pack, bench.steps), ctx=ctx, config=cfg, steps=bench.steps,
        statements=bench.variants.statements, drafts=draft_tables(pack),
        golden=bench.db.golden_state(), gateway=gateway)


def run_split(bench, config, mutants, arms: Sequence[Arm], gateway, out_lines: Path,
              split: str) -> SplitResult:
    """SpecDx (from the run cache, with checkpoints) then every arm, per specification.
    One JSON line per specification is appended to out_lines as it finishes."""
    cfg = dataclasses.replace(config.specdx(), checkpoints=CHECKPOINTS)
    base_cfg = dataclasses.replace(cfg, localize=False)
    base_ctx = bench.loop_context(bench.base, cfg)
    base = run_specdx(bench.base, base_ctx, base_cfg)
    touched = touch_counts({m.id: m.truth.observed_at for m in mutants}, base.behaviours,
                           base_ctx.lens)
    result = SplitResult(split, {m.id: m.truth for m in mutants}, set(scorable_ids(touched)),
                         {"specdx": {}}, {"specdx": {}}, {}, {})
    out_lines.parent.mkdir(parents=True, exist_ok=True)
    with out_lines.open("w", encoding="utf-8") as fh:
        for spec, variant, report, ctx in [("base", bench.base, base, base_ctx)] + [
                (m.id, m.variant, None, None) for m in mutants]:
            if report is None:
                ctx = bench.loop_context(variant, cfg)
                report = run_specdx(variant, ctx, cfg, base=base)
            result.defects["specdx"][spec] = list(report.defects)
            result.costs["specdx"][spec] = behaviours_cost(
                report.behaviours, report.probe_runs + report.ddmin_runs).as_dict()
            result.checkpoints[spec] = report.checkpoints
            inp = make_input(bench, variant, report, ctx, cfg, gateway)
            for arm in arms:
                b = REGISTRY[arm.name]
                try:
                    out: BaselineOutput = b.run(inp, model=arm.model, strategy=arm.strategy)
                except Exception as exc:  # noqa: BLE001 - one arm, one spec
                    out = BaselineOutput([], failure=f"{type(exc).__name__}: {exc}")
                result.defects.setdefault(arm.label, {})[spec] = out.defects
                result.costs.setdefault(arm.label, {})[spec] = out.cost.as_dict()
                if out.failure:
                    result.failures.setdefault(arm.label, {})[spec] = out.failure
            fh.write(json.dumps({"spec": spec, "done": True}) + "\n")
            fh.flush()
    return result
```


- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_study.py -v`
Expected: PASS.

- [ ] **Step 5: Add a rerun-determinism test with fakes (Review Focus 5)**

Append to `tests/eval/test_study.py` a test that monkeypatches `specdx.eval.study.run_specdx`, `make_input` and `REGISTRY` with deterministic fakes. It calls `run_split` twice with a fake `bench` (`base`, `loop_context` returning a `SimpleNamespace(lens=..., cells=...)`) and asserts that `to_json()` is identical. Use these fakes:
```python
def test_run_split_is_deterministic(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    import specdx.eval.study as st
    from specdx.baselines.api import BaselineOutput
    from specdx.cost import Cost

    rep = SimpleNamespace(defects=[_d("hidden", "c1", 0.9)], behaviours=[], probe_runs=0,
                          ddmin_runs=0, checkpoints=[])
    monkeypatch.setattr(st, "run_specdx", lambda *a, **k: rep)
    monkeypatch.setattr(st, "make_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "touch_counts", lambda *a, **k: {"H1": 3})
    monkeypatch.setattr(st, "scorable_ids", lambda t: {"H1"})
    fake = SimpleNamespace(uses_llm=False, strategies=(None,),
                           run=lambda inp, **k: BaselineOutput([_d("hidden", "c1", 0.4)], Cost()))
    monkeypatch.setattr(st, "REGISTRY", {"jest": fake})
    cfg = SimpleNamespace(specdx=lambda: st.SpecDxConfig({"f": ("m",)}))
    bench = SimpleNamespace(base="B", loop_context=lambda v, c: SimpleNamespace(lens=None))
    mut = SimpleNamespace(id="H1", variant="V", truth=_t("hidden", "c1", "clause", "H1"))
    a = st.run_split(bench, cfg, [mut], [st.Arm("jest")], None, tmp_path / "a.jsonl", "dev")
    b = st.run_split(bench, cfg, [mut], [st.Arm("jest")], None, tmp_path / "b.jsonl", "dev")
    assert a.to_json() == b.to_json()
```
`specdx/eval/study.py` already imports `SpecDxConfig` (Step 3), so `st.SpecDxConfig` resolves.

Run: `.venv/bin/python -m pytest tests/eval/test_study.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add specdx/eval/study.py tests/eval/test_study.py
git commit -m "feat: comparison-study split runner and scoring (AP, Detection@k, equal-cost)"
```

---

### Task 12: Statistics and the study report

**Files:**
- Create: `specdx/eval/stats.py`, `specdx/eval/study_report.py`
- Test: `tests/eval/test_stats.py`

**Interfaces:**
- Consumes: `SplitResult`, `score`, `rank_metrics`, `equal_cost`, `match_findings`.
- Produces:
  - `mcnemar_exact(b: int, c: int) -> float`;
  - `holm(pvalues: Mapping[str, float]) -> dict[str, float]`, giving adjusted p-values;
  - `detected(result, arm, tau) -> dict[mutant, bool]`;
  - `paired(result, arm_a, arm_b, tau_a, tau_b) -> {"b", "c", "n", "risk_difference", "p"}`;
  - `bootstrap_ci(result, arm, tau, resamples=10000, seed=20261021) -> (lo, hi)` for the detection rate;
  - `study_markdown(result, arms: Mapping[str, float]) -> str`, where `arms` maps an arm label to its threshold, with `"specdx"` mapped to 0.0.

- [ ] **Step 1: Write the failing tests**

`tests/eval/test_stats.py`:
```python
import math

from specdx.eval.stats import holm, mcnemar_exact


def test_mcnemar_exact_matches_the_binomial() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(5, 0), 2 * 0.5 ** 5)
    assert math.isclose(mcnemar_exact(8, 2), 2 * sum(math.comb(10, i) for i in range(3)) / 2 ** 10)


def test_holm_step_down() -> None:
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06)
    assert math.isclose(adj["b"], 0.06)  # monotone: never below the previous adjusted value
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`specdx/eval/stats.py`:
```python
"""Paired statistics of the comparison study (v1 §4.7; baselines spec §3.6)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from specdx.eval.matching import match_findings

RESAMPLES, SEED = 10_000, 20261021


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def detected(result, arm: str, tau: float) -> dict[str, bool]:
    from specdx.eval.study import _reports

    reports, base = _reports(result, arm, tau, primary=True)
    return {k: bool(match_findings(ds, result.truths[k], base, typed=True)[1])
            for k, ds in reports.items()}


def paired(result, arm_a: str, arm_b: str, tau_a: float, tau_b: float) -> dict[str, float]:
    da, db = detected(result, arm_a, tau_a), detected(result, arm_b, tau_b)
    keys = sorted(set(da) & set(db))
    b = sum(da[k] and not db[k] for k in keys)
    c = sum(db[k] and not da[k] for k in keys)
    n = len(keys) or 1
    return {"b": b, "c": c, "n": len(keys), "risk_difference": (b - c) / n,
            "p": mcnemar_exact(b, c)}


def bootstrap_ci(result, arm: str, tau: float, resamples: int = RESAMPLES,
                 seed: int = SEED) -> tuple[float, float]:
    hits = np.array([float(v) for _, v in sorted(detected(result, arm, tau).items())])
    if hits.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = hits[rng.integers(0, hits.size, size=(resamples, hits.size))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
```

`specdx/eval/study_report.py`:
```python
"""Markdown for the comparison study: Table A (native protocols), Table B (equal cost)."""

from __future__ import annotations

from collections.abc import Mapping

from specdx.eval.matching import evaluate_reports
from specdx.eval.stats import bootstrap_ci, holm, paired
from specdx.eval.study import equal_cost, rank_metrics, score
from specdx.types import DEFECT_TYPES


def _cost(result, arm: str) -> str:
    c = result.costs.get(arm, {})
    req = sum(int(v.get("requests", 0)) for v in c.values())
    exact = all(v.get("exact", True) for v in c.values())
    return f"{req}" + ("" if exact else " (bound)")


def study_markdown(result, arms: Mapping[str, float]) -> str:
    lines = [f"# Comparison study: {result.split} split", "",
             "## Table A: each method at its native protocol (primary table, typed)", "",
             "| arm | τ | " + " | ".join(f"{t} F1" for t in DEFECT_TYPES)
             + " | AP (mean) | Detection@1 | requests | detection 95% CI | failed specs |",
             "|---|---:|" + "---:|" * (len(DEFECT_TYPES) + 6)]
    for arm, tau in arms.items():
        s, r = score(result, arm, tau), rank_metrics(result, arm, tau)
        ap = sum(r[t]["ap"] for t in DEFECT_TYPES) / len(DEFECT_TYPES)
        d1 = sum(r[t]["detect@1"] for t in DEFECT_TYPES) / len(DEFECT_TYPES)
        lo, hi = bootstrap_ci(result, arm, tau)
        lines.append(f"| {arm} | {tau:.3g} | " + " | ".join(f"{s[t].f1:.2f}" for t in DEFECT_TYPES)
                     + f" | {ap:.2f} | {d1:.2f} | {_cost(result, arm)} | [{lo:.2f}, {hi:.2f}] | "
                     + (f"{len(result.failures[arm])} (INCOMPLETE)" if result.failures.get(arm)
                        else "0") + " |")
    rivals = {a: t for a, t in arms.items() if a != "specdx"}
    tests = {a: paired(result, "specdx", a, 0.0, t) for a, t in rivals.items()}
    adj = holm({a: v["p"] for a, v in tests.items()})
    lines += ["", "## Paired comparison with SpecDx (exact McNemar, Holm-adjusted)", "",
              "| arm | SpecDx only | arm only | n | risk difference | p (Holm) |",
              "|---|---:|---:|---:|---:|---:|"]
    for a, v in tests.items():
        lines.append(f"| {a} | {v['b']} | {v['c']} | {v['n']} | {v['risk_difference']:+.2f} | "
                     f"{adj[a]:.3f} |")
    lines += ["", "## Table B: SpecDx at each LLM arm's request budget", "",
              "Ambiguity at intermediate checkpoints is unlocalised (spec §9.2).", "",
              "| arm | arm requests | " + " | ".join(f"SpecDx {t} F1" for t in DEFECT_TYPES) + " |",
              "|---|---:|" + "---:|" * len(DEFECT_TYPES)]
    for a in rivals:
        if "@" not in a:
            continue
        eq = equal_cost(result, a)
        reports = {k: v for k, v in eq.items() if k != "base" and k in result.truths
                   and k in result.scorable}
        s = evaluate_reports(reports, result.truths, result.defects["specdx"].get("base", []))
        lines.append(f"| {a} | {_cost(result, a)} | "
                     + " | ".join(f"{s[t].f1:.2f}" for t in DEFECT_TYPES) + " |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_stats.py tests/eval/test_study.py -v`
Expected: PASS.

- [ ] **Step 5: Add a smoke test of the Markdown on the fabricated result**

Append to `tests/eval/test_study.py`:
```python
def test_the_report_renders_both_tables() -> None:
    from specdx.eval.study_report import study_markdown

    md = study_markdown(_result(), {"specdx": 0.0, "jest": 0.0, "llm@m": 0.0})
    assert "Table A" in md and "Table B" in md and "| llm@m |" in md
```
Run: `.venv/bin/python -m pytest tests/eval/test_study.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add specdx/eval/stats.py specdx/eval/study_report.py tests/eval/test_stats.py tests/eval/test_study.py
git commit -m "feat: exact McNemar, Holm, bootstrap and the study's Markdown tables"
```

---

### Task 13: Selection on dev and the baseline pre-registration

**Files:**
- Create: `specdx/eval/freeze.py`
- Test: `tests/eval/test_freeze.py`; `tests/test_preregistration_baselines.py`

**Interfaces:**
- Consumes: `SplitResult`, `score`, `REGISTRY`, `CHECKPOINTS`, `K_RUNS`, `TELL_CHARS`, `RESAMPLES`, `SEED`.
- Produces:
  - `model_size(tag: str) -> float`, the `NNb` number (`inf` when absent);
  - `thresholds(result, arm) -> list[float]` (deciles of the arm's dev scores, plus 0.0);
  - `choose(result) -> dict[str, dict]`, giving per baseline `{"arm": label, "model", "strategy", "tau", "dev_f1"}`;
  - `prereg(result, source: Path, cache: Path | None = None) -> dict` (the JSON document, including `baseline_cache: {"lines", "md5"}` of the LLM cache at freeze time) and `prereg_markdown(doc) -> str`;
  - `FIXED: dict`, the values fixed now;
  - `study_arms(doc) -> list[tuple[Arm, float]]`.

- [ ] **Step 1: Write the failing tests**

`tests/eval/test_freeze.py`:
```python
from specdx.eval.freeze import choose, model_size, thresholds
from specdx.eval.study import SplitResult
from specdx.mutate.operators import GroundTruth
from specdx.types import Defect


def _d(loc, s):
    return Defect("hidden", (loc,), (), (), s)


def _result():
    truths = {"H1": GroundTruth("H1", "hidden", frozenset({"c1"}), "clause")}
    arms = {
        "whowhen@qwen2.5:14b/all_at_once": {"base": [], "H1": [_d("c1", 0.4)]},
        "whowhen@gpt-oss:20b/all_at_once": {"base": [], "H1": [_d("c1", 0.4)]},
        "whowhen@gpt-oss:20b/step_by_step": {"base": [], "H1": [_d("zz", 0.9)]},
        "jest": {"base": [], "H1": [_d("c1", 0.7), _d("x", 0.2)]},
    }
    return SplitResult("dev", truths, {"H1"}, {"specdx": {"base": [], "H1": []}, **arms},
                       {}, {}, {})


def test_model_size_reads_the_tag() -> None:
    assert model_size("qwen2.5:14b") == 14 and model_size("qwen3:30b-a3b") == 30
    assert model_size("weird") == float("inf")


def test_thresholds_are_deciles_plus_zero() -> None:
    assert thresholds(_result(), "jest")[0] == 0.0


def test_ties_go_to_the_smaller_model_then_the_higher_threshold() -> None:
    c = choose(_result())
    assert c["whowhen"]["model"] == "qwen2.5:14b"
    assert c["jest"]["tau"] > 0.2  # the higher threshold that still keeps c1 removes x


def test_choice_is_deterministic() -> None:
    assert choose(_result()) == choose(_result())
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_freeze.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`specdx/eval/freeze.py`:
```python
"""Selection on the dev split and the baseline pre-registration (baselines spec §3.3, §7).

Per baseline: the arm (model, strategy) and threshold τ maximising mean typed F1 over the
types it reports, on the primary (scorable) table. τ ranges over 0 and the deciles of the
arm's dev scores. Ties: smaller model (the NNb in the tag), then higher τ, then label."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from specdx.baselines.api import REGISTRY, TELL_CHARS
from specdx.baselines.whowhen import K_RUNS
from specdx.eval.stats import RESAMPLES, SEED
from specdx.eval.study import CHECKPOINTS, Arm, SplitResult, score

FIXED: dict[str, Any] = {
    "k_failing_runs": K_RUNS, "tell_chars": TELL_CHARS, "checkpoints": list(CHECKPOINTS),
    "bootstrap": {"resamples": RESAMPLES, "seed": SEED}, "grid": "deciles+0",
    "benchguard_commit": "7d6a52e5627a1a734ecd895e0c73778ab54aca63",
    "whowhen_commit": "b2bae5c5b06d681d04ea5e9b63b7a30525c04925",
}


def model_size(tag: str | None) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)b", tag or "")
    return float(m.group(1)) if m else float("inf")


def thresholds(result: SplitResult, arm: str) -> list[float]:
    scores = [d.score for ds in result.defects.get(arm, {}).values() for d in ds]
    if not scores:
        return [0.0]
    deciles = np.quantile(np.array(scores), np.linspace(0, 0.9, 10))
    return sorted({0.0, *(round(float(x), 12) for x in deciles)})


def _parse(label: str) -> tuple[str, str | None, str | None]:
    name, _, rest = label.partition("@")
    model, _, strategy = rest.partition("/")
    return name, model or None, strategy or None


def choose(result: SplitResult) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, b in sorted(REGISTRY.items()):
        cands = []
        for label in sorted(result.defects):
            n, model, strategy = _parse(label)
            if n != name:
                continue
            for tau in thresholds(result, label):
                s = score(result, label, tau)
                f1 = sum(s[t].f1 for t in b.reports) / len(b.reports)
                cands.append((-f1, model_size(model), -tau, label, model, strategy, tau, f1))
        if cands:
            *_, label, model, strategy, tau, f1 = min(cands)
            out[name] = {"arm": label, "model": model, "strategy": strategy, "tau": tau,
                         "dev_f1": f1}
    return out


def prereg(result: SplitResult, source: Path, cache: Path | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {"schema": 1, "source": str(source),
                           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "chosen": choose(result), "fixed": FIXED}
    if cache is not None and cache.exists():
        data = cache.read_bytes()
        lines = data.splitlines(keepends=True)
        doc["baseline_cache"] = {"lines": len(lines),
                                 "md5": hashlib.md5(b"".join(lines)).hexdigest()}
    return doc


def prereg_markdown(doc: dict[str, Any]) -> str:
    body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)
    lines = ["# Baseline pre-registration (baselines spec §3.3)", "",
             f"Chosen on the dev split from `{doc['source']}` (sha256 `{doc['source_sha256']}`).",
             f"JSON sha256: `{hashlib.sha256(body.encode()).hexdigest()}`.", "",
             "| baseline | arm | τ | dev mean typed F1 |", "|---|---|---:|---:|"]
    for name, c in doc["chosen"].items():
        lines.append(f"| {name} | {c['arm']} | {c['tau']:.6g} | {c['dev_f1']:.3f} |")
    lines += ["", "Fixed before selection: " + json.dumps(doc["fixed"], sort_keys=True), ""]
    return "\n".join(lines)


def study_arms(doc: dict[str, Any]) -> list[tuple[Arm, float]]:
    return [(Arm(n, c["model"], c["strategy"]), float(c["tau"]))
            for n, c in sorted(doc["chosen"].items())]
```

`tests/test_preregistration_baselines.py`:
```python
"""The values the baselines spec fixes now, and the frozen choices once committed
(baselines spec §3.3, §7). Editing a frozen value after the test split ran breaks this."""

import hashlib
import json
from pathlib import Path

import pytest

from specdx.baselines import benchguard, whowhen
from specdx.baselines.api import TELL_CHARS
from specdx.eval import freeze, stats, study

ROOT = Path(__file__).parents[1]
DOC = ROOT / "experiments" / "p3-baselines-prereg.json"


def test_values_fixed_by_the_spec() -> None:
    assert whowhen.K_RUNS == 5 and TELL_CHARS == 8000
    assert study.CHECKPOINTS == (25, 50, 100, 150, 200)
    assert (stats.RESAMPLES, stats.SEED) == (10_000, 20261021)
    assert benchguard.SEVERITY == {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    assert benchguard.CONFIDENCE == {"CONFIRMED": 3, "LIKELY": 2, "POSSIBLE": 1}
    assert benchguard.MAX_TOKENS == 4096 and whowhen.MAX_TOKENS == 1024
    assert freeze.FIXED["benchguard_commit"].startswith("7d6a52e")


@pytest.mark.skipif(not DOC.exists(), reason="frozen by `run_all.sh freeze-baselines`")
def test_the_frozen_choices_match_their_markdown_and_the_fixed_values() -> None:
    doc = json.loads(DOC.read_text(encoding="utf-8"))
    body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)
    md = DOC.with_suffix(".md").read_text(encoding="utf-8")
    assert hashlib.sha256(body.encode()).hexdigest() in md
    assert doc["fixed"] == freeze.FIXED
    assert set(doc["chosen"]) == set(freeze.REGISTRY)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_freeze.py tests/test_preregistration_baselines.py -v`
Expected: PASS. The frozen-choices test is skipped until the freeze step runs.

- [ ] **Step 5: Commit**

```bash
git add specdx/eval/freeze.py tests/eval/test_freeze.py tests/test_preregistration_baselines.py
git commit -m "feat: dev-split selection and the pinned baseline pre-registration"
```

---

### Task 14: CLI commands and guards

**Files:**
- Modify: `specdx/cli.py` (subparsers at about lines 49–101; dispatch before the existing `GateConfig` import at about line 252). `RunCache` already iterates its `Behaviour`s (`__iter__`), so no cache change is needed
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `run_split`, `dev_arms`, `SplitResult`, `Arm` (Task 11); `study_markdown` (Task 12); `prereg`, `prereg_markdown`, `study_arms` (Task 13); `Gateway` (Task 3); `GateConfig`, `dev_split`, `all_mutants`.
- Produces these commands:

  | Command | Arguments |
  |---|---|
  | `specdx baselines` | `--split {dev,base} --config PATH --out PATH [--models M ...] [--workers N]` |
  | `specdx freeze-baselines` | `--from PATH --out PREFIX [--cache PATH]` |
  | `specdx study` | `--split test --config PATH --baselines PATH --out PATH [--workers N]` |
  | `specdx report` | `--dev PATH [--test PATH] --out PATH` |

- Produces the guard `study_guard(root: Path, gate_json: Path, prereg_json: Path) -> str | None`, which returns the reason for refusal, or `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:
```python
def test_study_refuses_without_a_passed_gate(tmp_path) -> None:
    from specdx.cli import study_guard

    gate = tmp_path / "g.json"
    gate.write_text('{"passed": false}')
    reason = study_guard(tmp_path, gate, tmp_path / "p.json")
    assert reason and "gate" in reason


def test_study_refuses_an_uncommitted_preregistration(tmp_path) -> None:
    import subprocess

    from specdx.cli import study_guard

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    gate = tmp_path / "g.json"
    gate.write_text('{"passed": true}')
    pre = tmp_path / "p.json"
    pre.write_text("{}")
    reason = study_guard(tmp_path, gate, pre)
    assert reason and "commit" in reason


def test_the_new_commands_parse() -> None:
    from specdx.cli import build_parser

    p = build_parser()
    a = p.parse_args(["baselines", "--split", "dev", "--config", "c.json", "--out", "o.json"])
    assert (a.command, a.split) == ("baselines", "dev")
    assert p.parse_args(["report", "--dev", "d.json", "--out", "r.md"]).command == "report"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k "study or commands" -v`
Expected: FAIL with `ImportError: study_guard` or `build_parser`.

- [ ] **Step 3: Implement**

In `specdx/cli.py`:
1. Move the parser construction out of `main` into a new `def build_parser() -> argparse.ArgumentParser:`, which `main` calls. This is a pure refactor; all existing tests keep passing.
2. Add four subparsers inside `build_parser`:
   ```python
    bl = sub.add_parser("baselines", help="run every baseline arm on the dev split (or the base)")
    bl.add_argument("--split", choices=("dev", "base"), required=True)
    bl.add_argument("--config", type=Path, required=True)
    bl.add_argument("--out", type=Path, required=True)
    bl.add_argument("--models", nargs="*", default=None)
    bl.add_argument("--workers", type=int, default=1)
    fr = sub.add_parser("freeze-baselines", help="choose and pin each baseline's arm and τ")
    fr.add_argument("--from", dest="source", type=Path, required=True)
    fr.add_argument("--out", type=Path, required=True)
    fr.add_argument("--cache", type=Path, default=None, help="the baseline LLM cache to fingerprint")
    st = sub.add_parser("study", help="the test split, once the gate PASSED (spec §3.7)")
    st.add_argument("--split", choices=("test",), required=True)
    st.add_argument("--config", type=Path, required=True)
    st.add_argument("--baselines", type=Path, required=True)
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--workers", type=int, default=1)
    rp = sub.add_parser("report", help="tables A and B, statistics, for dev and test")
    rp.add_argument("--dev", type=Path, required=True)
    rp.add_argument("--test", type=Path)
    rp.add_argument("--out", type=Path, required=True)
   ```
3. Add the guard:
   ```python
   def study_guard(root: Path, gate_json: Path, prereg_json: Path) -> str | None:
       import subprocess

       if not gate_json.exists() or not json.loads(gate_json.read_text()).get("passed"):
           return f"the gate has not PASSED ({gate_json}); the test split stays locked"
       def git(*args: str) -> subprocess.CompletedProcess[str]:
           return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
       rel = str(prereg_json.resolve().relative_to(root.resolve()))
       if git("ls-files", "--error-unmatch", rel).returncode != 0:
           return f"{rel} is not committed: run ./run_all.sh freeze-baselines"
       if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
           return "the working tree has uncommitted changes"
       return None
   ```
4. Dispatch in `main`, after `bench` is opened and before `from specdx.eval.pilot import GateConfig, ...`:
   ```python
    if args.command in ("baselines", "study"):
        from specdx.baselines.llm import Gateway
        from specdx.eval.freeze import study_arms
        from specdx.eval.pilot import GateConfig
        from specdx.eval.study import Arm, dev_arms, run_split

        config = GateConfig.load(args.config)
        config.check_population()
        allm = all_mutants(bench.base, bench.pdf, bench.work_dir)
        dev = dev_split(allm, config.dev_per_type, config.rng_seed)
        pinned: dict[str, str] = {}
        for r in bench.cache:  # RunCache iterates Behaviours in file order: the last wins
            if r.model_digest:
                pinned[r.interpreter.model] = r.model_digest
        gateway = Gateway(OLLAMA_URL, bench.work_dir / "baseline-llm.jsonl", pinned)
        if args.command == "baselines":
            arms = dev_arms(config.models_by_family)
            if args.models is not None:
                arms = [a for a in arms if a.model is None or a.model in args.models]
            mutants = dev if args.split == "dev" else []
        else:
            reason = study_guard(REPO_ROOT, REPO_ROOT / "results" / "p2-gate.json", args.baselines)
            if reason:
                print(f"error: {reason}", file=sys.stderr)
                return 4
            doc = json.loads(args.baselines.read_text(encoding="utf-8"))
            arms = [a for a, _ in study_arms(doc)]
            ids = {m.id for m in dev}
            mutants = [m for m in allm if m.id not in ids]
        result = run_split(bench, config, mutants, arms, gateway,
                           args.out.with_suffix(".progress.jsonl"), args.split)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.to_json(), encoding="utf-8")
        print(f"{len(result.defects)} arms on {len(mutants) + 1} specifications -> {args.out}")
        return 0
   ```
   - Import `REPO_ROOT` and `OLLAMA_URL` from `specdx.config`.
5. Add the commands that need no workbench before `Workbench.open`, so they never take the database lock:
   ```python
    if args.command == "freeze-baselines":
        from specdx.eval.freeze import prereg, prereg_markdown
        from specdx.eval.study import SplitResult

        result = SplitResult.from_json(args.source.read_text(encoding="utf-8"))
        bad = {a: f for a, f in result.failures.items() if f}
        if bad:
            print(f"error: failed specifications {bad}; re-run baselines-dev", file=sys.stderr)
            return 1
        doc = prereg(result, args.source, args.cache)
        body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)
        args.out.with_suffix(".json").write_text(body, encoding="utf-8")
        args.out.with_suffix(".md").write_text(prereg_markdown(doc), encoding="utf-8")
        return 0
    if args.command == "report":
        from specdx.eval.study import SplitResult
        from specdx.eval.study_report import study_markdown

        dev = SplitResult.from_json(args.dev.read_text(encoding="utf-8"))
        text = study_markdown(dev, {"specdx": 0.0, **{a: 0.0 for a in dev.defects
                                                      if a != "specdx"}})
        if args.test is not None and args.test.exists():
            pre = json.loads((REPO_ROOT / "experiments" / "p3-baselines-prereg.json")
                             .read_text(encoding="utf-8"))
            test = SplitResult.from_json(args.test.read_text(encoding="utf-8"))
            taus = {c["arm"]: c["tau"] for c in pre["chosen"].values()}
            text += "\n" + study_markdown(test, {"specdx": 0.0, **taus})
        args.out.write_text(text, encoding="utf-8")
        print(text)
        return 0
   ```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add specdx/cli.py tests/test_cli.py
git commit -m "feat: baselines, freeze-baselines, study and report commands with the study guard"
```

---

### Task 15: `run_all.sh` stages and the running guide

**Files:**
- Modify: `run_all.sh`
- Modify: `docs/RUNNING.md` (add §12)
- Test: `tests/test_docs.py`, extended with checks on the script

**Interfaces:**
- Consumes the CLI commands of Task 14 and `results/p2-gate.json`.
- Produces these `run_all.sh` steps: `baselines-dev`, `freeze-baselines`, `study`, `report`, `s2`; extended `local`, `status` and `all`; `FOREGROUND`; exit 4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_docs.py`:
```python
def test_run_all_offers_the_comparison_steps() -> None:
    import subprocess

    text = (ROOT / "run_all.sh").read_text(encoding="utf-8")
    for step in ("baselines-dev)", "freeze-baselines)", "study)", "report)", "s2)"):
        assert step in text
    assert "exit 4" in text and "FOREGROUND" in text
    usage = subprocess.run(["bash", str(ROOT / "run_all.sh")], capture_output=True, text=True)
    assert usage.returncode == 2 and "baselines-dev" in usage.stdout + usage.stderr


def test_s2_says_it_is_not_built_and_exits_2() -> None:
    import subprocess

    r = subprocess.run(["bash", str(ROOT / "run_all.sh"), "s2"], capture_output=True, text=True)
    assert r.returncode == 2 and "τ²" in (r.stdout + r.stderr)
```
If `tests/test_docs.py` defines no `ROOT`, add `ROOT = Path(__file__).parents[1]` there.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_docs.py -k "run_all or s2" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the changes to `run_all.sh`**

**1. Header.** Extend the usage comment block, keeping its format:
```bash
#   ./run_all.sh baselines-dev       every baseline arm on the dev split (after the gate)
#   ./run_all.sh freeze-baselines    choose each baseline's arm and τ on dev; commit the pre-registration
#   ./run_all.sh study               SpecDx and the frozen baselines on the test split (gate PASSED only)
#   ./run_all.sh report              tables A and B and the statistics (dev, and test when present)
#   ./run_all.sh s2                  S2 (τ²-bench) is not built yet
# `all` now runs g2 -> preflight -> g3 -> gate -> baselines-dev -> freeze-baselines -> study ->
# report in the foreground (run it inside tmux). Exit 4: the gate is NOT PASSED, so the test
# study is locked (the dev comparison is still produced).
```

**2. Helpers**, after `gate_running`:
```bash
FOREGROUND="${FOREGROUND:-0}"
BASELINE_CACHE="$CACHE_DIR/baseline-llm.jsonl"
PREREG=experiments/p3-baselines-prereg

running() { [[ -f work/$1.pid ]] && kill -0 "$(cat "work/$1.pid")" 2> /dev/null; }

# launch NAME CMD...: foreground under `all` (FOREGROUND=1), else detached with a pid file.
launch() {
  local name=$1; shift
  if [[ $FOREGROUND == 1 ]]; then
    "$@" 2>&1 | tee -a "work/$name.log"
    return "${PIPESTATUS[0]}"
  fi
  setsid nohup "$@" > "work/$name.log" 2>&1 < /dev/null &
  echo $! > "work/$name.pid"
  ok "launched $name, pid $(cat "work/$name.pid"); log work/$name.log"
}

gate_passed() { [[ -f results/p2-gate.json ]] && jq -e '.passed == true' results/p2-gate.json > /dev/null; }

check_baseline_cache() {
  [[ -f $PREREG.json ]] || return 0
  local lines md5 want_lines want_md5
  read -r want_lines want_md5 < <(jq -r '.baseline_cache | "\(.lines) \(.md5)"' "$PREREG.json" 2> /dev/null || echo "null null")
  [[ $want_lines == null ]] && return 0
  md5=$(head -n "$want_lines" "$BASELINE_CACHE" | md5sum | cut -d' ' -f1)
  [[ $md5 == "$want_md5" ]] || fail "the first $want_lines lines of $BASELINE_CACHE changed since the freeze"
  ok "baseline cache prefix intact"
}
```

**3. Gate.** Make `stage_gate` honour `FOREGROUND`: replace its `setsid nohup ... &` block with:
```bash
  launch p2-gate env SPECDX_WORK="$CACHE_DIR" SPECDX_OLLAMA_URL="$SPECDX_OLLAMA_URL" \
    "$SPECDX" gate --config "$GATE_CONFIG" --out results/p2-gate.json --workers "$WORKERS"
```
Keep the `gate_running` check and the y/N prompt.

**4. New steps:**
```bash
stage_baselines_dev() {
  need_remote
  [[ -f results/p2-gate.json ]] || die "the gate has not finished (results/p2-gate.json): ./run_all.sh gate"
  running p2-gate && die "the gate is still running"
  running p3-baselines-dev && { ok "baselines-dev is already running"; return; }
  step "Baselines on the dev split (every arm, every population model)"
  launch p3-baselines-dev env SPECDX_WORK="$CACHE_DIR" SPECDX_OLLAMA_URL="$SPECDX_OLLAMA_URL" \
    "$SPECDX" baselines --split dev --config "$GATE_CONFIG" --out results/p3-baselines-dev.json \
    --workers "$WORKERS"
}

stage_freeze() {
  step "Freeze the baselines' arms and thresholds (dev split only)"
  [[ -f results/p3-baselines-dev.json ]] || die "run ./run_all.sh baselines-dev first"
  running p3-baselines-dev && die "baselines-dev is still running"
  if git ls-files --error-unmatch "$PREREG.json" > /dev/null 2>&1; then
    ok "already frozen: $PREREG.json"; return
  fi
  "$SPECDX" freeze-baselines --from results/p3-baselines-dev.json --out "$PREREG" \
    --cache "$BASELINE_CACHE" || fail "freeze-baselines failed (a failed dev specification?)"
  "$PY" -m pytest -q -o addopts="" tests/test_preregistration_baselines.py | tail -1
  git add "$PREREG.json" "$PREREG.md" results/p3-baselines-dev.json
  git commit -q -m "experiments: baseline pre-registration — arms and thresholds frozen on the dev split" \
    -- "$PREREG.json" "$PREREG.md" results/p3-baselines-dev.json
  ok "frozen and committed $(git rev-parse --short HEAD)"
}

stage_study() {
  need_remote
  if ! gate_passed; then
    echo "The gate is NOT PASSED (results/p2-gate.json): the test study stays locked (spec §3.7)."
    exit 4
  fi
  git ls-files --error-unmatch "$PREREG.json" > /dev/null 2>&1 || die "run ./run_all.sh freeze-baselines first"
  [[ -z $(git status --porcelain --untracked-files=no) ]] || die "the working tree has uncommitted changes"
  check_cache
  check_baseline_cache
  running p3-study && { ok "the study is already running"; return; }
  step "Study: SpecDx and the frozen baselines on the test split (days)"
  launch p3-study env SPECDX_WORK="$CACHE_DIR" SPECDX_OLLAMA_URL="$SPECDX_OLLAMA_URL" \
    "$SPECDX" study --split test --config "$GATE_CONFIG" --baselines "$PREREG.json" \
    --out results/p3-study.json --workers "$WORKERS"
}

stage_report() {
  step "Report"
  [[ -f results/p3-baselines-dev.json ]] || die "nothing to report: run ./run_all.sh baselines-dev"
  local test_args=()
  [[ -f results/p3-study.json ]] && test_args=(--test results/p3-study.json)
  "$SPECDX" report --dev results/p3-baselines-dev.json "${test_args[@]}" --out results/p3-report.md > /dev/null
  ok "results/p3-report.md"
}
```

**5. `stage_local`.** After the end-to-end smoke run, add:
```bash
    step "Baselines on the smoke base (one model, own caches)"
    SPECDX_WORK="$smoke/work" SPECDX_OLLAMA_URL=$url "$SPECDX" baselines --split base \
      --config "$smoke/smoke.json" --out "$smoke/baselines-base.json" \
      --models "$(jq -r '[.[][]][0]' <<< "$population")" 2>> "$smoke/smoke.log" \
      || { tail -20 "$smoke/smoke.log"; fail "the baseline smoke run crashed"; }
    jq -r '.failures | to_entries[] | "  \(.key): \(.value | keys | join(", "))"' "$smoke/baselines-base.json"
    ok "baselines ran on the base: $(jq '.defects | length' "$smoke/baselines-base.json") arms"
```

**6. `stage_status`.** Also print the `p3-baselines-dev` and `p3-study` pids and log tails (use `running`), plus `wc -l < "$BASELINE_CACHE"` when the file exists.

**7. Dispatch:**
```bash
  baselines-dev)    stage_baselines_dev ;;
  freeze-baselines) check_tools; stage_freeze ;;
  study)            stage_study ;;
  report)           stage_report ;;
  s2)               echo "S2 (τ²-bench) needs the τ² adapter, a later P3 step; nothing to run."; exit 2 ;;
  all)
    check_tools
    export FOREGROUND=1
    stage_g2
    stage_preflight
    stage_g3
    stage_gate 1
    stage_baselines_dev
    stage_freeze
    if ! gate_passed; then stage_report; echo "Gate NOT PASSED: test study locked."; exit 4; fi
    stage_study
    stage_report
    ;;
```
With `FOREGROUND=1`, `stage_gate 1` skips the prompt and runs in the foreground.

- [ ] **Step 4: Run the tests and a shellcheck pass, and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_docs.py -v && (command -v shellcheck > /dev/null && shellcheck -S warning run_all.sh || true)`
Expected: PASS, and no shellcheck warnings in the new functions.

- [ ] **Step 5: Add §12 to `docs/RUNNING.md`**

Write `docs/RUNNING.md` §12, "The comparison study (baselines)":
- the step order;
- what each step writes: `results/p3-baselines-dev.json`, `experiments/p3-baselines-prereg.{json,md}`, `results/p3-study.json`, `results/p3-report.md`;
- `pip install -e ".[baselines]"`;
- that the LLM arms reuse the population models and need nothing extra pulled;
- that `study` exits 4 unless the gate PASSED;
- how to resume: re-run the same step, and both caches serve the finished work;
- that `baseline/` clones are reference copies only.

- [ ] **Step 6: Run the whole suite and lint, and confirm they pass**

Run: `.venv/bin/python -m pytest -q -o addopts="" && .venv/bin/ruff check .`
Expected: all tests pass (db tests need `docker start specdx-postgres`), and ruff is clean.

- [ ] **Step 7: Commit**

```bash
git add run_all.sh docs/RUNNING.md tests/test_docs.py
git commit -m "feat: run_all.sh — baselines-dev, freeze-baselines, study (gate-locked), report, s2; FOREGROUND for all"
```

---

## Self-review notes

**Spec coverage**

| Spec section | Task(s) |
|---|---|
| §2.1 set | Tasks 5–10 |
| §2.3 / §9.4 B1 | Task 10 |
| §2.5 / §9.5 B2 | Task 9 |
| §2.6 B3 | Task 6 |
| §2.7 B4 | Task 7 |
| §2.8 / §9.3 B5 | Task 8 |
| §2.9 TI, JEST | Task 5 |
| §3.1–3.4 inputs and scoring | Task 11 |
| §3.3 selection | Task 13 |
| §3.5 / §9.1 cost and Table B | Tasks 1, 2, 11, 12 |
| §3.6 statistics | Task 12 |
| §3.7 integrity | Tasks 14, 15 |
| §4.2 gateway | Task 3 |
| §4.4 / §9.2 checkpoints | Task 2 |
| §4.5 dependencies | Task 4 |
| §5 `run_all.sh` | Task 15 |
| §6 tests | in each task |
| §9.6 T excerpt | Task 4 |

**Gate verdict untouched.** `run_gate`, `gate_verdict`, `b2_single.py` and `b3_irt.py` are not modified.

**Known judgement calls left to the implementer, each with a concrete check:**
- B3's likelihood must be checked against the paper (Task 6, Step 1).
- crowd-kit's constructor arguments may vary by version (Task 5, Step 4).
- `RunCache.all()` may need adding (Task 14).
- `describe` on the toy clause (Task 9).
