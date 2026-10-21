"""B1: BenchGuard (Tu et al., arXiv 2604.24955; COLM 2026), definition audit only
(baselines spec §2.3, §9.4). Each specification becomes a BenchGuard benchmark with one
task, the anchor canon scenario (the golden state is defined there alone, spec §9.8):
instruction = scenario + numbered T excerpt; tests = one file per clause that applies;
solution = the golden state. Every LLM call goes through the
gateway via GatewayClient; BenchGuard's code is unchanged. Findings map to SpecDx types
by the frozen MAPPING."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from benchguard.agent.client import LLMClient, LLMConfig, LLMResponse
from benchguard.agent.runner import ProtocolRunner
from benchguard.core.finding import Finding, Location
from benchguard.core.pipeline import AuditMode, VerificationPipeline
from benchguard.loader import BenchmarkLoader

from specdx.baselines.api import (
    BaselineInput,
    BaselineOutput,
    register,
    scenario_text,
    tell_excerpt,
)
from specdx.baselines.llm import DigestMismatch, near_ctx
from specdx.cost import Cost
from specdx.spec.clauses import describe
from specdx.types import Defect, InfrastructureError

if TYPE_CHECKING:
    from pathlib import Path

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
_BEFORE = r"(?<![A-Za-z0-9_.:-])"
_AFTER = r"(?![A-Za-z0-9_:-]|\.[A-Za-z0-9])"


def _fname(clause_id: str) -> str:
    return "tests/" + re.sub(r"[^A-Za-z0-9._-]", "_", clause_id.replace(":", "__")) + ".txt"


def _column(clause_id: str) -> str | None:
    """golden:<table>:<key>:<column> -> table.column; other clauses have no column."""
    parts = clause_id.split(":")
    return f"{parts[1]}.{parts[-1]}" if parts[0] == "golden" and len(parts) >= 4 else None


def package(inp: BaselineInput, root: Path) -> dict[str, str]:
    """One task, the anchor scenario: the golden state is defined there alone."""
    if root.exists():
        shutil.rmtree(root)
    files: dict[str, str] = {}
    excerpt = "\n".join(f"[T:{sid}] {t}" for sid, t in tell_excerpt(inp))
    golden = inp.golden.tables if inp.golden is not None else {}
    scenario = inp.clauses.anchor
    task = root / scenario.id
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "task.toml").write_text(
        f"[metadata]\nid = {json.dumps(scenario.id)}\ncategory = \"business-process\"\n"
        'benchmark_source = "TonomyX"\n', encoding="utf-8")
    (task / "instruction.md").write_text(
        f"{scenario_text(scenario)}\n\nRequirements (numbered sentences):\n{excerpt}\n",
        encoding="utf-8")
    for c in inp.clauses:
        if inp.clauses.applies(c, scenario):
            rel = _fname(c.id)
            (task / rel).write_text(
                f"{c.id}\n{describe(c, scenario)}\n"
                f"{json.dumps(c.payload, ensure_ascii=False, default=str)}\n",
                encoding="utf-8")
            files[rel] = c.id
    (task / "solution" / "golden_state.json").write_text(
        json.dumps(golden, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return files


def _names(text: str, token: str) -> bool:
    """token occurs in text as a whole token (not inside a longer file name or clause id)."""
    return re.search(_BEFORE + re.escape(token) + _AFTER, text) is not None


def _cites(src: Location, rel: str, cid: str) -> bool:
    """BenchGuard's loader shows each test file under its base name (`# === <name> ===`) and
    its prompt asks for `"file": "eval_script"`, so a clause is cited by its file's base name
    or by its id (each file's first line) in `file` or `snippet`."""
    base = PurePosixPath(rel).name
    return any(_names(text, token) for text in (src.file, src.snippet) for token in (base, cid))


def _cited(f: Finding, files: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for src in (f.evidence.primary_source, f.evidence.conflicting_source):
        if src is not None:
            for rel, cid in files.items():
                if cid not in out and _cites(src, rel, cid):
                    out.append(cid)
    return out


def map_finding(f: Finding, files: Mapping[str, str], corpus_ids: set[str]) -> list[Defect]:
    rule = MAPPING.get(f.subcategory)
    if rule is None:
        return []
    score = float(SEVERITY[f.severity.value] * CONFIDENCE[f.confidence_level.value])
    snippets = [src.snippet for src in (f.evidence.primary_source,
                                         f.evidence.conflicting_source) if src is not None]
    text = " ".join([f.title, f.description, f.evidence.description, *snippets])
    clauses = _cited(f, files)
    out: list[Defect] = []
    if rule == "ambiguity":
        sents = tuple(sorted({s for s in SENT_RE.findall(text) if s in corpus_ids}))
        out.append(Defect("ambiguity", sents, (f.task_id,), (), score, f.subcategory))
    elif rule == "contradiction" or (rule == "contradiction_or_hidden" and len(clauses) >= 2):
        out.append(Defect("contradiction", tuple(sorted(clauses)), (f.task_id,), (), score,
                          f.subcategory))
    else:
        for cid in clauses:
            out.append(Defect("hidden", (cid,), (f.task_id,), (), score, f.subcategory))
            col = _column(cid)
            if rule == "hidden+weak" and col and WEAK_RE.search(text):
                out.append(Defect("weak", (col,), (f.task_id,), (), score, f.subcategory))
    return out


class GatewayClient(LLMClient):
    """BenchGuard's LLM client with its completion methods routed through the gateway.

    ProtocolRunner turns any client exception into a per-protocol error, so a gateway error
    reply (status "error") is raised to become exactly that. Infrastructure failures (Ollama
    down, model missing or with another digest) are not data: they are kept in `fatal` and
    re-raised by the baseline after the audit, and every later call fails at once."""

    def __init__(self, gateway, model: str) -> None:
        super().__init__(LLMConfig(model=f"ollama/{model}", temperature=0.0,
                                   max_tokens=MAX_TOKENS))
        self.gateway, self.model, self.cost = gateway, model, Cost()
        self.fatal: Exception | None = None
        self.near_ctx = 0  # calls whose prompt plus output budget reached num_ctx

    def complete(self, prompt, system_prompt="", temperature=None, max_tokens=None):
        if self.fatal is not None:
            raise self.fatal
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}]
        t = self.config.resolved_temperature() if temperature is None else temperature
        budget = max_tokens or self.config.max_tokens
        try:
            r = self.gateway.chat(self.model, msgs, options={"temperature": t, "seed": 1},
                                  max_tokens=budget)
        except (InfrastructureError, DigestMismatch) as exc:
            self.fatal = exc
            raise
        self.cost = self.cost + r.cost
        self.near_ctx += near_ctx(r, budget)
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

    def run(self, inp: BaselineInput, *, model: str | None = None,
            strategy: str | None = None) -> BaselineOutput:
        if inp.gateway is None or not model:
            raise ValueError("benchguard needs a gateway and a model")
        root = (inp.gateway.cache_path.parent / "baselines" / "benchguard" / inp.variant.name
                / model.replace(":", "_"))
        files = package(inp, root)
        client = GatewayClient(inp.gateway, model)
        pipe = VerificationPipeline(BenchmarkLoader(root), llm_config=client.config,
                                    audit_mode=AuditMode.DEFINITION, agent_evidence="off",
                                    min_confidence=None, generate_fixes=False)
        pipe.runner = ProtocolRunner(client=client)
        try:
            report = pipe.audit(max_concurrent=1)
        except Exception as exc:  # noqa: BLE001 - a broken audit fails this spec only
            if client.fatal is not None:
                raise client.fatal from exc
            return BaselineOutput([], client.cost, failure=f"{type(exc).__name__}: {exc}",
                                  extra={"near_ctx": client.near_ctx})
        if client.fatal is not None:
            raise client.fatal
        ids = set(inp.variant.corpus.sentence_ids())
        merged: dict[tuple[str, tuple[str, ...]], Defect] = {}
        for f in report.findings:
            for d in map_finding(f, files, ids):
                if d.key not in merged or d.score > merged[d.key].score:
                    merged[d.key] = d
        extra: dict[str, object] = {"findings": len(report.findings),
                                    "near_ctx": client.near_ctx}
        if not getattr(inp, "golden_materialised", True):
            extra["golden"] = "unmaterialised"  # the solution has no derived values
        return BaselineOutput(sorted(merged.values(), key=lambda d: (-d.score, d.location)),
                              client.cost, extra=extra)


register(BenchGuard())
