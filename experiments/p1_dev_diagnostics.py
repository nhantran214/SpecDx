"""P1 development-split diagnostics: the evidence spec §4.8 asks for before §3.2–§3.4 change.

Read-only. It never calls Ollama, never executes a run, never opens the database and never
writes to work/ or results/. It reads the run cache (work/runs.jsonl), the gate's
per-mutant lines (results/p1-gate-mutants.jsonl) and the mutant packs (work/mutants/), and
prints Markdown tables on stdout (or to --out). Only the base and the 20 development
mutants of the gate are used.

How cached runs are mapped to a specification: exactly as Runner.run keys them,
run_key(variant digest, steps digest, context digest, scenario parameters, interpreter
key, suffix, model digest). The Workbench supplies the steps and context digests; the
model digest is the one each cached behaviour records (one per model); suffix "" is a
population run (probe runs carry the refutation goal as suffix). The mutant variants are
rebuilt from the packs in work/mutants exactly as all_mutants builds them, without
calling all_mutants, because all_mutants rewrites the packs.

Usage: .venv/bin/python experiments/p1_dev_diagnostics.py [--out PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from tonomyx.canon.invariants import is_blank
from tonomyx.canon.pack import CanonPack

from specdx.config import REPO_ROOT, WORK_DIR
from specdx.eval.pilot import GateConfig
from specdx.infer.items import ItemView, SpecLens
from specdx.infer.latent import RARE, Observation, Posterior, _subset_prior, fit
from specdx.infer.scores import _hidden_score, _weak
from specdx.interpreters.cache import run_key
from specdx.interpreters.executor import SEARCH_K, SEARCH_REPLY_LIMIT
from specdx.interpreters.population import committee
from specdx.mutate.operators import A1_SITES, A2_SITES
from specdx.scenarios.space import boundary_domain, canon_scenarios, enumerate_scenarios
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour
from specdx.variant import SpecVariant
from specdx.workbench import Workbench

RUNS = WORK_DIR / "runs.jsonl"
GATE_LINES = REPO_ROOT / "results" / "p1-gate-mutants.jsonl"
GATE_CONFIG = REPO_ROOT / "experiments" / "p1-gate.json"
A_IDS = ("A1-01", "A1-02", "A2-01", "A2-02", "A3-00")
W_IDS = ("W1-04", "W2-01", "W2-03", "W2-04", "W2-05")
H1_IDS = ("H1-02", "H1-03", "H1-04")
# Mutants whose edit touches only C (invariants, golden CSVs): no interpreter can see it,
# so their runs measure how far two runs of one prompt and seed drift apart.
CONTROL_IDS = ("C1-00", "C1-01", "C1-02", "H1-02", "H1-03", "H1-04", "H2-00", "H2-01",
               "W2-01", "W2-03", "W2-04", "W2-05")
MODEL_ORDER = ("qwen3:14b", "qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b")
SHORT = {"qwen3:14b": "qwen3-14b", "qwen2.5:7b-instruct-q4_K_M": "qwen2.5-7b",
         "llama3.1:8b": "llama3.1-8b"}


# ---------------------------------------------------------------------------- loading


def _never(*_: Any, **__: Any) -> Any:
    raise RuntimeError("this analysis never executes a run")


@dataclass
class Spec:
    id: str
    variant: SpecVariant
    lens: SpecLens
    runs: dict[tuple[str, str], Behaviour]  # (scenario id, interpreter key) -> behaviour
    line: dict[str, Any]
    violated: dict[str, frozenset[str]] = field(default_factory=dict)
    views: list[ItemView] = field(default_factory=list)
    posterior: Posterior | None = None

    @property
    def behaviours(self) -> list[Behaviour]:
        return list(self.runs.values())


class Study:
    def __init__(self) -> None:
        self.cache: dict[str, Behaviour] = {}
        with RUNS.open("rb") as handle:
            for raw in handle:
                if raw.strip():
                    b = Behaviour.from_json(raw.decode("utf-8"))
                    self.cache[b.run_key] = b
        digests: dict[str, set[str]] = defaultdict(set)
        for b in self.cache.values():
            digests[b.interpreter.model].add(b.model_digest)
        if any(len(v) != 1 for v in digests.values()):
            raise SystemExit(f"a model has several digests in the cache: {digests}")
        self.model_digest = {m: next(iter(v)) for m, v in digests.items()}
        # factory and model_digest never reach Ollama: no run is executed here.
        self.bench = Workbench.open(factory=_never, model_digest=self.model_digest.__getitem__)
        self.config = GateConfig.load(GATE_CONFIG)
        self.cfg = self.config.specdx()
        self.members = committee(self.config.models_by_family, self.config.variants,
                                 self.config.seeds)
        self.lines = {(ln.get("mutant") or "base"): ln
                      for ln in (json.loads(x) for x in GATE_LINES.read_text().splitlines())}
        self.specs: dict[str, Spec] = {}
        for sid in self.lines:
            self.specs[sid] = self._load(sid)
        mapped = sum(len(s.runs) for s in self.specs.values())
        self.unmapped = len(self.cache) - mapped

    def variant_of(self, sid: str) -> SpecVariant:
        """The SpecVariant all_mutants built for this id, rebuilt read-only."""
        base = self.bench.base
        if sid == "base":
            return base
        pack = WORK_DIR / "mutants" / sid / base.pack.name
        corpus = TellCorpus.build(CanonPack(pack), self.bench.pdf)
        op, index = sid.split("-")
        if op == "A1":
            corpus = corpus.without(corpus.find(A1_SITES[int(index)]))
        elif op == "A2":
            old, new = A2_SITES[int(index)]
            for sentence in corpus.find(old):
                corpus = corpus.replace(sentence, new)
        return SpecVariant(sid, pack, corpus)

    def _load(self, sid: str) -> Spec:
        variant = self.variant_of(sid)
        runner = self.bench.runner
        pack = CanonPack(variant.pack)
        canon = canon_scenarios(pack)
        # One scenario per parameter vector; a canon name wins over a synthesised id,
        # as in the loop, where a visited signature is never chosen again.
        by_signature = {s.signature(): s for s in enumerate_scenarios(boundary_domain(pack))}
        for s in canon.values():
            by_signature[s.signature()] = s
        digest = variant.digest()
        runs: dict[tuple[str, str], Behaviour] = {}
        for scenario in by_signature.values():
            params = json.dumps(scenario.params(), sort_keys=True)
            for interp in self.members:
                key = run_key(digest, runner._steps_digest, runner.context_digest, params,
                              interp.key, "", self.model_digest[interp.model])
                b = self.cache.get(key)
                if b is not None:
                    runs[(scenario.id, interp.key)] = dataclasses.replace(
                        b, scenario=scenario, interpreter=interp)
        ctx = self.bench.loop_context(variant, self.cfg)
        spec = Spec(sid, variant, ctx.lens, runs, self.lines[sid])  # type: ignore[arg-type]
        for b in spec.behaviours:
            v = spec.lens.violations(b)
            spec.violated[b.run_key] = v
            spec.views.extend(spec.lens.views(b, v))
        spec.posterior = fit_views(spec.views, self.cfg.k_max, self.cfg.p_single)
        return spec


def fit_views(views: Sequence[ItemView], k_max: int, p_single: float) -> Posterior:
    return fit([Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views],
               k_max=k_max, p_single=p_single)


# ---------------------------------------------------------------------------- helpers


def table(header: Sequence[str], rows: Iterable[Sequence[Any]], align: str = "") -> None:
    def cell(c: Any) -> str:
        return str(c).replace("|", "\\|")

    print("| " + " | ".join(cell(h) for h in header) + " |")
    marks = [("---:" if (align[i:i + 1] == "r") else "---") for i in range(len(header))]
    print("|" + "|".join(marks) + "|")
    for row in rows:
        print("| " + " | ".join(cell(c) for c in row) + " |")
    print()


def pct(n: float, d: float) -> str:
    return "–" if not d else f"{100 * n / d:.0f}%"


def fmt(x: float, digits: int = 3) -> str:
    if x == 0:
        return "0"
    if abs(x) < 10 ** (-digits):
        return f"{x:.1e}"
    return f"{x:.{digits}f}"


def steps_of(study: Study) -> list[str]:
    return [s.id for s in study.bench.steps]


def scenario_order(spec: Spec) -> list[str]:
    seen: list[str] = []
    for sid, _ in spec.runs:
        if sid not in seen:
            seen.append(sid)
    return seen


def views_by_item(spec: Spec) -> dict[str, list[ItemView]]:
    out: dict[str, list[ItemView]] = defaultdict(list)
    for v in spec.views:
        out[v.item].append(v)
    return out


def item_log_evidence(
    views: Sequence[ItemView], theta: dict[str, float], k_max: int
) -> tuple[int, dict[int, float]]:
    """The log marginal likelihood of an item's runs given |A| = size, summed over the sets
    of that size with the prior's within-size weights (latent.py's likelihood, recomputed
    in log space so that nothing underflows). Returns (m, {size: log evidence})."""
    counts = Counter(v.klass for v in views)
    top = tuple(k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k_max])
    n, m = len(views), len(top)
    eps = {k: (counts[k] + 1) / (n + m + 1) for k in top}
    eps[RARE] = (n - sum(counts[k] for k in top) + 1) / (n + m + 1)
    labelled = [(v, v.klass if v.klass in top else RARE) for v in views]
    by_size: dict[int, list[float]] = defaultdict(list)
    for size in range(1, m + 1):
        for combo in combinations(top, size):
            subset = frozenset(combo)
            lp = 0.0
            for v, k in labelled:
                th = theta[v.model]
                faith = th / size if k in subset else 0.0
                lp += math.log(faith + (1 - th) * eps[k])
            by_size[size].append(lp)
    out = {}
    for size, values in by_size.items():
        top_v = max(values)
        out[size] = top_v + math.log(sum(math.exp(x - top_v) for x in values) / len(values))
    return m, out


def log_bayes_factor(m: int, evidence: dict[int, float]) -> float:
    """log P(runs | |A| >= 2) - log P(runs | |A| = 1), with the prior's size weights
    (0.5^(s-1)) inside |A| >= 2. -inf when only one class was seen (no split possible)."""
    if m < 2:
        return -math.inf
    prior = _subset_prior(m, 0.5)  # any p_single: only the within-ambiguous weights matter
    weights = {s: prior[s] * math.comb(m, s) for s in range(2, m + 1)}
    total = sum(weights.values())
    terms = [math.log(weights[s] / total) + evidence[s] for s in range(2, m + 1)]
    top_t = max(terms)
    log_amb = top_t + math.log(sum(math.exp(t - top_t) for t in terms))
    return log_amb - evidence[1]


def p_single_needed(lbf: float) -> float:
    """The largest prior P(|A| = 1) at which P(|A| >= 2) reaches 0.5, θ held fixed."""
    if lbf == -math.inf:
        return 0.0
    return 1 / (1 + math.exp(-lbf))


def p_amb_at(lbf: float, p_single: float) -> float:
    if lbf == -math.inf:
        return 0.0
    log_odds = lbf + math.log((1 - p_single) / p_single)
    return 1 / (1 + math.exp(-log_odds))


# ---------------------------------------------------------------------------- section 0


def section_mapping(study: Study) -> None:
    print("## 0. Data, mapping and validation\n")
    base = study.specs["base"]
    keys_by_spec = {sid: {b.run_key for b in s.behaviours} for sid, s in study.specs.items()}
    shared = sum(1 for a, b in combinations(keys_by_spec, 2)
                 if keys_by_spec[a] & keys_by_spec[b])
    print(f"- Run cache: {len(study.cache)} behaviours, all with status `ok`: "
          f"{sum(1 for b in study.cache.values() if b.status == 'ok')}.")
    print(f"- Population runs recovered by recomputed run key: "
          f"{sum(len(s.runs) for s in study.specs.values())} = {len(study.specs)} specifications "
          f"× {sorted({len(s.runs) for s in study.specs.values()})} runs. Pairs of "
          f"specifications sharing a run key: {shared}.")
    print(f"- The other {study.unmapped} cached runs carry a refutation goal as key suffix "
          f"(directed probes): {study.unmapped / len(study.specs):.0f} per specification, "
          "as each gate line's `runs.probe` says. They are not used below.")
    probe_ok = all(s.line["runs"]["population"] == len(s.runs) for s in study.specs.values())
    print(f"- Every gate line's `runs.population` equals the recovered count: {probe_ok}.")
    wit_total = wit_found = 0
    for s in study.specs.values():
        keys = keys_by_spec[s.id]
        for d in s.line["defects"]["specdx"]:
            for w in d["witnesses"]:
                wit_total += 1
                wit_found += w in keys
    print(f"- SpecDx witness run keys in the gate lines found among the recovered population "
          f"runs of the same specification: {wit_found} of {wit_total}.")
    order = scenario_order(base)
    same = all(Counter(sc for sc, _ in s.runs) == Counter(sc for sc, _ in base.runs)
               for s in study.specs.values())
    print(f"- Every specification ran the same scenarios with the same run counts: {same}.\n")
    rows = []
    for sc in order:
        b = next(b for (x, _), b in base.runs.items() if x == sc)
        p = b.scenario
        rows.append((sc, p.ordered, p.received, p.discount_pct, p.grn_count,
                     sum(1 for x, _ in base.runs if x == sc)))
    table(("scenario", "ordered", "received", "discount %", "GRNs", "runs per spec"), rows,
          "lrrrrr")
    # Validation: the recomputed posterior reproduces the recorded base scores.
    by_item = views_by_item(base)
    post = base.posterior
    assert post is not None
    checked = matched = 0
    worst = 0.0
    for d in base.line["defects"]["specdx"]:
        if d["type"] != "hidden":
            continue
        clause = d["location"][0]
        scores = []
        for sc in d["scenarios"]:
            items = sorted(i for i in post.items if i.split("|", 1)[0] == sc)
            score = _hidden_score(clause, items, post, by_item)
            if score is not None:
                scores.append(score)
        if not scores:
            continue
        checked += 1
        err = abs(max(scores) - d["score"])
        worst = max(worst, err)
        matched += err < 1e-9
    weak = [d for d in base.line["defects"]["specdx"] if d["type"] == "weak"]
    weak_rows = []
    for d in weak:
        item = d["note"]
        got = _weak(post.items[item], by_item[item], 2)
        weak_rows.append(f"`{item}` recorded {d['score']:.6f}, recomputed "
                         f"{got[1] if got else float('nan'):.6f}")
    print(f"- Validation of the offline posterior (gate config: k_max {study.cfg.k_max}, "
          f"P(|A|=1) {study.cfg.p_single}, weak_min_runs 2): {matched} of {checked} base "
          f"Hidden scores whose clause is violated in some view are reproduced to 1e-9 "
          f"(largest error {worst:.1e}); weak finding {'; '.join(weak_rows)}; largest "
          f"P(|A|>=2) over base items {max(i.p_ambiguous for i in post.items.values()):.3f} "
          "(< 0.5, so no ambiguity, as recorded).\n")


# ---------------------------------------------------------------------------- Q1


def _corpus_diff(base: TellCorpus, other: TellCorpus) -> list[tuple[str, str, str]]:
    b = {s.id: s.text for c in base.chunks for s in c.sentences}
    o = {s.id: s.text for c in other.chunks for s in c.sentences}
    out = []
    for sid in b:
        if sid not in o:
            out.append((sid, b[sid], "(removed)"))
        elif b[sid] != o[sid]:
            out.append((sid, b[sid], o[sid]))
    for sid in o:
        if sid not in b:
            out.append((sid, "(absent)", o[sid]))
    return out


def _payload_ids(corpus: TellCorpus, query: str) -> tuple[list[str], list[str]]:
    """The chunk ids search_spec ranks top-k, and those that fit the reply limit
    (executor.py: the first passage always, then passages while the JSON fits)."""
    hits = corpus.search(query, k=SEARCH_K)
    kept: list[dict[str, str]] = []
    for i, hit in enumerate(hits):
        passage = {"id": hit.id, "text": hit.text}
        if i == 0:
            kept.append(passage)
            continue
        payload = json.dumps({"passages": [*kept, passage]}, ensure_ascii=False)
        if len(payload) <= SEARCH_REPLY_LIMIT:
            kept.append(passage)
        else:
            break
    return [h.id for h in hits], [p["id"] for p in kept]


def _rank(corpus: TellCorpus, query: str, chunk: str) -> int | None:
    from specdx.spec.tell import _tokens
    if corpus._index is None:
        corpus.search("x", k=1)
    assert corpus._index is not None
    scores = corpus._index.scores(_tokens(query))
    ranked = sorted(range(len(corpus.chunks)), key=lambda i: (-scores[i], i))
    for r, i in enumerate(ranked, 1):
        if corpus.chunks[i].id == chunk:
            return r if scores[i] > 0 else None
    return None


A_MARKERS: dict[str, tuple[str, ...]] = {
    "A1-01": ("vượt trần", "gắn cờ"),
    "A1-02": ("APR chuẩn", "xử lý ngoại lệ"),
    "A2-01": ("5%", "hợp lý"),
    "A2-02": ("3 cổng", "một số cổng"),
    "A3-00": ("7%", "KB-0009"),
}


def _base_lens_class(study: Study, b: Behaviour) -> dict[str, tuple[Any, ...]]:
    """Per step: (projection, step-relevant violations, status) under the BASE lens, so
    runs of different specifications are compared on one ≡_S and one C."""
    lens = study.specs["base"].lens
    violated = lens.violations(b)
    return {v.item.split("|", 1)[1]: (v.projection, tuple(sorted(v.violated)),
                                       b.state.steps.get(v.item.split("|", 1)[1], ""))
            for v in lens.views(b, violated)}


def _state_sig(b: Behaviour) -> str:
    return json.dumps([b.state.tables, b.state.steps, b.state.derived], sort_keys=True,
                      ensure_ascii=False)


def section_q1(study: Study) -> None:
    print("## Q1. Are the ambiguity mutants observable?\n")
    base = study.specs["base"]
    base_corpus = study.bench.base.corpus
    print("### Q1.1 The edits to T (base corpus against the mutant corpus, sentence by "
          "sentence)\n")
    targets: dict[str, set[str]] = {}
    for mid in A_IDS:
        spec = study.specs[mid]
        diff = _corpus_diff(base_corpus, spec.variant.corpus)
        targets[mid] = {TellCorpus.chunk_of(sid) for sid, _, _ in diff}
        print(f"**{mid}** ({spec.line['operator']}; truth "
              f"{', '.join(spec.line['truth']['location'])}, kind "
              f"{spec.line['truth']['kind']}):\n")
        table(("sentence id", "before", "after"),
              [(f"`{sid}`", before, after) for sid, before, after in diff])
    print("A1 deletes the sentence from the pack file itself, so the corpus is rebuilt "
          "with the following sentences of the chunk renumbered (A1-01: KB-0003:05 becomes "
          ":04; A1-02: PRINCIPLES:04 becomes :03). The truth for A1 is chunk-level, so "
          "scoring is unaffected.\n")

    print("### Q1.2 Could search_spec have retrieved the edited chunk?\n")
    print("Behaviours do not record tool calls or their results: a Behaviour holds the end "
          "state, the step statuses, the refused calls as (step, tool name) only, and each "
          "step's final claim. The queries the runs issued are therefore unknown. BM25 "
          f"(k = {SEARCH_K}, reply limit {SEARCH_REPLY_LIMIT} characters, as in executor.py) "
          "is rerun offline on the mutant's corpus for the 9 interpreted steps × 3 Tell "
          "variants of the work-order statement (27 queries), and for the 9 localiser "
          "queries (statement variant 0 plus the step's table names, workbench.py).\n")
    steps = [s for s in study.bench.steps if s.kind != "env"]
    n_chunks = len(base_corpus.chunks)
    rows = []
    for mid in A_IDS:
        corpus = study.specs[mid].variant.corpus
        for chunk in sorted(targets[mid]):
            top5 = fits = 0
            best: list[int] = []
            per_step = []
            for step in steps:
                ranks = []
                for v in range(study.bench.variants.count):
                    q = study.bench.variants.statements[step.intent][v]
                    ranked, kept = _payload_ids(corpus, q)
                    top5 += chunk in ranked
                    fits += chunk in kept
                    r = _rank(corpus, q, chunk)
                    if r is not None:
                        ranks.append(r)
                best += ranks
                per_step.append(min(ranks) if ranks else None)
            loc = 0
            for step in steps:
                q = (f"{study.bench.variants.statements[step.intent][0]} "
                     f"{' '.join(sorted(step.tables))}")
                ranked, _ = _payload_ids(corpus, q)
                loc += chunk in ranked
            scored = [(r, s.id) for r, s in zip(per_step, steps, strict=True) if r is not None]
            best_step: tuple[int | None, str] = min(scored) if scored else (None, "-")
            rows.append((mid, f"`{chunk}`", f"{top5}/27", f"{fits}/27", f"{loc}/9",
                         f"{best_step[0]} of {n_chunks} ({best_step[1]})"
                         if best_step[0] else "never scored"))
    table(("mutant", "edited chunk", "in top 5 (statements)", "in reply payload",
           "in top 5 (localiser)", "best rank (step)"), rows, "llrrrl")
    changed = 0
    total = 0
    for mid in A_IDS:
        corpus = study.specs[mid].variant.corpus
        for step in steps:
            for v in range(study.bench.variants.count):
                q = study.bench.variants.statements[step.intent][v]
                total += 1
                changed += _payload_ids(corpus, q)[0] != _payload_ids(base_corpus, q)[0]
    print(f"Side effect of the edit on BM25 itself: the top-5 list differs between base and "
          f"mutant corpus for {changed} of {total} statement queries (IDF and average "
          "length change with any edit).\n")

    print("### Q1.3 Did the edit reach the interpreters? Paired comparison with the base\n")
    print("Each population run of a mutant has a base twin with the same scenario, model, "
          "Tell variant and seed, hence the same system prompt. A step's conversation can "
          "only differ from its twin's if a search_spec reply differed, if the database "
          "state differed (an earlier step diverged), or if Ollama is not deterministic. "
          "The control group measures the last: the 12 dev mutants whose edit touches "
          "only C (invariants, golden CSVs), which no interpreter can see "
          f"({', '.join(CONTROL_IDS)}). Classes are recomputed under the base lens for "
          "every specification, so a class difference is a behavioural difference.\n")
    base_sig = {k: _state_sig(b) for k, b in base.runs.items()}
    base_cls = {k: _base_lens_class(study, b) for k, b in base.runs.items()}
    step_ids = steps_of(study)
    stats: dict[str, dict[str, Any]] = {}
    for mid in (*A_IDS, *CONTROL_IDS):
        spec = study.specs[mid]
        same_state = claim_any = class_any = 0
        first_div: Counter[str] = Counter()
        class_div: Counter[str] = Counter()
        claim_div: Counter[str] = Counter()
        for k, b in spec.runs.items():
            twin = base.runs[k]
            same_state += _state_sig(b) == base_sig[k]
            diff_steps = [s for s in step_ids if b.claims.get(s) != twin.claims.get(s)]
            claim_any += bool(diff_steps)
            for s in diff_steps:
                claim_div[s] += 1
            if diff_steps:
                first_div[diff_steps[0]] += 1
            mine = _base_lens_class(study, b)
            cdiff = [s for s in step_ids if mine.get(s) != base_cls[k].get(s)]
            class_any += bool(cdiff)
            for s in cdiff:
                class_div[s] += 1
        stats[mid] = {"same_state": same_state, "claim_any": claim_any,
                      "class_any": class_any, "first": first_div, "class": class_div,
                      "claim": claim_div, "n": len(spec.runs)}
    ctrl = [stats[c] for c in CONTROL_IDS]
    same_c = {mid: [(c.id, c.payload, c.scope) for c in study.specs[mid].lens.clauses]
              == [(c.id, c.payload, c.scope) for c in base.lens.clauses] for mid in A_IDS}
    print(f"C of every A mutant is identical to the base's (ids, payloads, scopes): "
          f"{same_c}. Their classes under their own lens therefore equal the classes "
          "under the base lens used here.\n")

    def ctrl_range(key: str) -> str:
        values = [c[key] for c in ctrl]
        return f"{min(values)}–{max(values)} (mean {statistics.mean(values):.1f})"

    rows = [(mid, stats[mid]["n"], stats[mid]["same_state"], stats[mid]["claim_any"],
             stats[mid]["class_any"]) for mid in A_IDS]
    rows.append(("control (12 C-only mutants)", 200, ctrl_range("same_state"),
                 ctrl_range("claim_any"), ctrl_range("class_any")))
    table(("specification", "pairs", "identical end state", "any step claim differs",
           "any step class differs (base lens)"), rows, "lrrrr")
    rows = []
    for model in MODEL_ORDER:
        def diverged(sid: str, model: str = model) -> int:
            return sum(1 for k, b in study.specs[sid].runs.items()
                       if b.interpreter.model == model and b.claims != base.runs[k].claims)
        rows.append((SHORT[model], *[diverged(m) for m in A_IDS],
                     f"{statistics.mean(diverged(c) for c in CONTROL_IDS):.1f}"))
    print("Runs whose claims differ from the twin's at some step, per model:\n")
    table(("model", *A_IDS, "control mean"), rows, "l" + "r" * (len(A_IDS) + 1))
    print("First step whose claim differs from the twin's (runs), per step in work-order "
          "order; control is the mean over the 12 control mutants:\n")
    rows = []
    for s in step_ids:
        if s.startswith("ENV:"):
            continue
        rows.append((s, *[stats[m]["first"][s] for m in A_IDS],
                     f"{statistics.mean(c['first'][s] for c in ctrl):.1f}"))
    table(("step", *A_IDS, "control mean"), rows, "l" + "r" * (len(A_IDS) + 1))
    print("Runs whose class at the step differs from the twin's (base lens):\n")
    rows = []
    for s in step_ids:
        rows.append((s, *[stats[m]["class"][s] for m in A_IDS],
                     f"{statistics.mean(c['class'][s] for c in ctrl):.1f}",
                     max(c["class"][s] for c in ctrl)))
    table(("step", *A_IDS, "control mean", "control max"), rows,
          "l" + "r" * (len(A_IDS) + 2))

    print("What changed in the A mutants' differing pairs (base lens): locked fields that "
          "differ between twin projections, or only status/violations when projections "
          "are equal:\n")
    lens = base.lens
    rows = []
    for mid in A_IDS:
        spec = study.specs[mid]
        fields: Counter[str] = Counter()
        cells: Counter[str] = Counter()
        for k, b in spec.runs.items():
            mine = _base_lens_class(study, b)
            for s in step_ids:
                a, t = mine.get(s), base_cls[k].get(s)
                if a == t or a is None or t is None:
                    continue
                cells[f"{k[0]}|{s}"] += 1
                f = lens.diff(a[0], t[0])
                if f:
                    fields.update(f)
                elif a[2] != t[2]:
                    fields["(status only)"] += 1
                else:
                    fields["(violations only)"] += 1
        top = sorted(fields.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        by_disc = Counter(int(c.split("-d")[1].split("-")[0]) if "-d" in c else
                          {"grn12": 0, "happy": 0, "discount8": 8}[c.split("|")[0]]
                          for c in cells.elements())
        rows.append((mid, ", ".join(f"{f} ×{n}" for f, n in top) or "–", len(cells),
                     ", ".join(f"d{k}: {v}" for k, v in sorted(by_disc.items()))))
    table(("mutant", "differences (top 8)", "items touched",
           "differing (run, step) pairs by scenario discount"), rows, "llrl")

    print("Class distribution per item (all 27 runs, 11 for the last scenario), mutant "
          "against base, under the base lens: total-variation distance.\n")
    rows = []
    for mid in (*A_IDS, "control"):
        ids = CONTROL_IDS if mid == "control" else (mid,)
        tvs = []
        for cid in ids:
            spec = study.specs[cid]
            per_item_b: dict[str, Counter[Any]] = defaultdict(Counter)
            per_item_m: dict[str, Counter[Any]] = defaultdict(Counter)
            for k, b in spec.runs.items():
                mine = _base_lens_class(study, b)
                for s in step_ids:
                    per_item_m[f"{k[0]}|{s}"][repr(mine.get(s))] += 1
                    per_item_b[f"{k[0]}|{s}"][repr(base_cls[k].get(s))] += 1
            for item in per_item_b:
                n = sum(per_item_b[item].values())
                keys = set(per_item_b[item]) | set(per_item_m[item])
                tvs.append(sum(abs(per_item_b[item][x] - per_item_m[item][x])
                               for x in keys) / (2 * n))
        nz = [t for t in tvs if t > 0]
        rows.append((mid, len(tvs), len(nz), f"{max(tvs):.3f}",
                     f"{statistics.mean(nz):.3f}" if nz else "–"))
    table(("specification", "items", "items with TV > 0", "max TV", "mean TV where > 0"),
          rows, "lrrrr")

    print("### Q1.4 Do the claims show the edited text was read?\n")
    print("Runs (of 200) whose claim at some step contains the marker, base against "
          "mutant (case-insensitive):\n")
    rows = []
    for mid in A_IDS:
        for marker in A_MARKERS[mid]:
            def count(spec: Spec, marker: str = marker) -> int:
                return sum(1 for b in spec.behaviours
                           if any(marker.casefold() in (c or "").casefold()
                                  for c in b.claims.values()))
            rows.append((mid, f"`{marker}`", count(base), count(study.specs[mid])))
    table(("mutant", "marker", "base runs", "mutant runs"), rows, "llrr")
    print("Scenario coverage of the discount edits: the scenarios run hold discount_pct "
          f"{sorted({b.scenario.discount_pct for b in base.behaviours})}. A3-00's article "
          "(cap 7%, no gate up to 7%) disagrees with KB-0003 (cap 5%) only for a discount "
          "of 6 or 7%; the boundary domain contains 6 "
          f"({boundary_domain(CanonPack(base.variant.pack))['discount_pct']}), but no "
          "scenario with it was run on any specification.\n")


# ---------------------------------------------------------------------------- Q2


def field_value(tables: dict[str, list[dict[str, str]]], field_id: str) -> str:
    """A truth field's value in an end state: for evt:KIND:REF the actors of the matching
    events, else the values of table.column over the table's case rows."""
    if field_id.startswith("evt:"):
        _, kind, ref = field_id.split(":", 2)
        actors = sorted(e["actor"] for e in tables.get("evt", [])
                        if e["kind"] == kind and e["ref"] == ref)
        return "absent" if not actors else "present (" + ", ".join(actors) + ")"
    table_name, column = field_id.split(".")
    rows = [r for r in tables.get(table_name, []) if r.get("case_id", "CASE-0031") != ""]
    if not rows:
        return "no row"
    return ", ".join(sorted("blank" if is_blank(r.get(column)) else repr(r[column])
                            for r in rows))


def _projection_value(projection: Any, field_id: str) -> Any:
    tables = dict(projection)
    if field_id.startswith("evt:"):
        key = field_id[4:]
        return tuple(sorted(c for k, c in tables.get("evt", ()) if k == key))
    table_name, column = field_id.split(".")
    return tuple(sorted(dict(c).get(column) for _, c in tables.get(table_name, ())))


def section_q2(study: Study) -> None:
    print("## Q2. Are the weak-check mutants observable?\n")
    base = study.specs["base"]
    gold = {t: [dict(r) for r in rows] for t, rows in CanonPack(base.variant.pack).tables.items()}
    steps = study.bench.steps
    print("§3.4's Weak(s) needs, at one item: (1) a MAP set none of whose classes violates "
          "a step-relevant clause; (2) a clean class outside it with >= weak_min_runs (2) "
          "runs; (3) P(that class not in A) >= 0.5; and, to match, (4) the locked fields "
          "between that class and the top admissible class include the truth field. "
          "Condition (1) over all items:\n")
    rows = []
    for sid in ("base", *W_IDS):
        spec = study.specs[sid]
        post = spec.posterior
        assert post is not None
        by_item = views_by_item(spec)
        by_key = {b.run_key: b for b in spec.behaviours}
        clean = err = 0
        for item, ip in post.items.items():
            step = item.split("|", 1)[1]
            violated = {v.klass: v.violated for v in by_item[item]}
            status = {v.klass: by_key[v.run_key].state.steps.get(step) for v in by_item[item]}
            clean += not any(violated[k] for k in ip.map_set)
            err += any(status[k] == "error" for k in ip.map_set)
        rows.append((sid, len(post.items), clean, err))
    table(("spec", "items", "MAP set clean (condition 1)", "MAP set holds an `error`-status class"),
          rows, "lrrr")
    logged: Counter[tuple[str, str, str, str]] = Counter()
    for b in base.behaviours:
        for e in b.state.tables.get("evt", []):
            logged[(SHORT[b.interpreter.model], e["kind"], e["ref"], e["actor"])] += 1
    print("Every event present in a base end state, as (model, kind, ref, actor) × runs: "
          + "; ".join(f"{m} {k} {r} '{a}' ×{n}" for (m, k, r, a), n in sorted(logged.items()))
          + ".\n")
    for mid in W_IDS:
        spec = study.specs[mid]
        field_id = spec.line["truth"]["location"][0]
        removed = sorted(base.lens.clauses.ids - spec.lens.clauses.ids)
        added = sorted(spec.lens.clauses.ids - base.lens.clauses.ids)
        if field_id.startswith("evt:"):
            _, kind, ref = field_id.split(":", 2)
            owners = [s.id for s in steps if any(e in (f"evt={ref}", "evt") for e in s.slice)]
        else:
            owners = [s.id for s in steps if field_id.split(".")[0] in s.tables]
        golden = field_value(gold, field_id)
        print(f"### {mid}: `{field_id}`\n")
        print(f"- Clauses removed from C: {', '.join(f'`{c}`' for c in removed) or 'none'}; "
              f"added: {', '.join(added) or 'none'}. Owning step(s): "
              f"{', '.join(f'`{o}`' for o in owners)}. Golden value in the base canon: "
              f"{golden}.")
        if field_id.startswith("evt:"):
            anchored = [c for c in removed if c.startswith("event:")]
            print(f"- The removed event clause is anchored (scope "
                  f"{base.lens.clauses[anchored[0]].scope if anchored else '?'}, "
                  "checked on grn12 only); on every other scenario the event was already "
                  "unguarded in the base.")
        print()
        rows = []
        for model in MODEL_ORDER:
            for label, s in (("base", base), (mid, spec)):
                runs = [b for b in s.behaviours if b.interpreter.model == model]
                values = Counter(field_value(b.state.tables, field_id) for b in runs)
                anchor = [field_value(b.state.tables, field_id) for b in runs
                          if b.scenario.id == "grn12"]
                # For an event the field IS its presence: absence deviates. For a column,
                # a run without the row cannot show the field at all.
                observed = sum(n for v, n in values.items() if v != "no row")
                dev = sum(n for v, n in values.items() if v not in (golden, "no row"))
                dev_a = sum(1 for v in anchor if v not in (golden, "no row"))
                rows.append((SHORT[model], label, len(runs), observed, values[golden], dev,
                             f"{dev_a}/{len(anchor)}",
                             "; ".join(f"{v} ×{n}" for v, n in values.most_common())))
        table(("model", "spec", "runs", "field observable", "= golden", "deviate from golden",
               "deviate on grn12", "values seen"), rows, "llrrrrrl")
        post = spec.posterior
        assert post is not None
        by_item = views_by_item(spec)
        rows = []
        met = 0
        for sc in scenario_order(spec):
            for owner in owners:
                item = f"{sc}|{owner}"
                ip, views = post.items[item], by_item[item]
                counts = Counter(v.klass for v in views)
                violated = {v.klass: v.violated for v in views}
                proj = {v.klass: v.projection for v in views}
                map_set = ip.map_set
                map_clean = not any(violated[k] for k in map_set)
                clean = {k for k in counts if not violated[k]}
                eligible = sorted(k for k in clean if counts[k] >= 2 and k not in map_set)
                ranked = sorted(ip.p_member, key=lambda k: -ip.p_member[k])
                ref_value = _projection_value(proj[ranked[0]], field_id)
                deviant_clean_runs = sum(
                    counts[k] for k in clean
                    if _projection_value(proj[k], field_id) != ref_value)
                golden_runs = [b for b in spec.behaviours if b.scenario.id == sc]
                deviant_any = sum(1 for b in golden_runs
                                  if field_value(b.state.tables, field_id)
                                  not in (golden, "no row"))
                eligible_dev = [k for k in eligible
                                if _projection_value(proj[k], field_id) != ref_value]
                weak = _weak(ip, views, 2)
                fields = (sorted(spec.lens.diff(proj[weak[0]], proj[ranked[0]]))
                          if weak else [])
                hit = weak is not None and weak[1] >= 0.5 and any(
                    f == field_id for f in fields)
                met += hit
                rows.append((item, len(views), deviant_any, len(counts), len(map_set),
                             "yes" if map_clean else
                             f"no ({sum(1 for k in map_set if violated[k])} of {len(map_set)} "
                             f"violate; min {min(len(violated[k]) for k in map_set)})",
                             len(clean), deviant_clean_runs, len(eligible), len(eligible_dev),
                             fmt(weak[1]) if weak else "–",
                             ", ".join(fields) if weak and weak[1] >= 0.5 else "–"))
        table(("item", "runs", "runs deviating on field", "classes", "MAP set size",
               "MAP set clean", "clean classes",
               "clean runs whose field differs from the top class",
               "eligible (clean, >=2 runs, outside MAP)", "eligible differing on field",
               "Weak score", "finding fields (>= 0.5)"), rows, "lrrrrlrrrrrl")
        print(f"Items where the detector would report the truth field: {met}.\n")


# ---------------------------------------------------------------------------- Q3


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _amb_fields(spec: Spec, item: str, members: Sequence[str]) -> frozenset[str]:
    proj = {v.klass: v.projection for v in views_by_item(spec)[item]}
    return frozenset().union(*(spec.lens.diff(proj[a], proj[b])
                               for a, b in combinations(members, 2)))


def _counterfactual_keys(spec: Spec, lbf: dict[str, float], p_single: float
                         ) -> set[tuple[str, frozenset[str]]]:
    """Ambiguity identities (step, fields) that would be reported with the prior
    P(|A|=1) = p_single, θ held fixed: p_amb >= 0.5 and the two classes with the highest
    membership differ on some locked field (diagnose's rule when the MAP set is a
    singleton; an approximation otherwise)."""
    post = spec.posterior
    assert post is not None
    out = set()
    for item, value in lbf.items():
        if p_amb_at(value, p_single) < 0.5:
            continue
        ip = post.items[item]
        ranked = sorted(ip.p_member, key=lambda k: -ip.p_member[k])
        fields = _amb_fields(spec, item, ranked[:2])
        if fields:
            out.add((item.split("|", 1)[1], fields))
    return out


def section_q3(study: Study) -> None:
    print("## Q3. Why does ambiguity never fire?\n")
    print("The posterior is recomputed with latent.fit and the gate configuration "
          f"(`experiments/p1-gate.json`: K_max {study.cfg.k_max}, P(|A|=1) "
          f"{study.cfg.p_single}) from each specification's 200 population runs; it is the "
          "final posterior of the gate's loop (validated in section 0).\n")
    ids = ("base", *A_IDS)
    rows = []
    for sid in ids:
        post = study.specs[sid].posterior
        assert post is not None
        rows.append((sid, *[f"{post.theta[m]:.3f}" for m in MODEL_ORDER],
                     ", ".join(f"{f} {e:+.2f}" for f, e in sorted(post.family_effect.items()))))
    table(("spec", *[f"θ {SHORT[m]}" for m in MODEL_ORDER], "family effects u_f"), rows,
          "lrrrl")

    print("### Q3.1 P(|A_s| >= 2) over items, and classes per item\n")
    rows = []
    lbfs: dict[str, dict[str, float]] = {}
    for sid in ids:
        spec = study.specs[sid]
        post = spec.posterior
        assert post is not None
        by_item = views_by_item(spec)
        amb = [ip.p_ambiguous for ip in post.items.values()]
        ncls = [len({v.klass for v in by_item[i]}) for i in post.items]
        lbfs[sid] = {}
        for item in post.items:
            m, ev = item_log_evidence(by_item[item], post.theta, study.cfg.k_max)
            lbfs[sid][item] = log_bayes_factor(m, ev)
        rows.append((sid, len(amb), fmt(max(amb)), fmt(quantile(amb, 0.5)),
                     fmt(quantile(amb, 0.9)), fmt(quantile(amb, 0.99)),
                     sum(a >= 0.1 for a in amb), sum(a >= 0.5 for a in amb),
                     f"{statistics.median(ncls):.0f}", max(ncls),
                     sum(n >= 2 for n in ncls)))
    table(("spec", "items", "max", "median", "p90", "p99", "items >= 0.1", "items >= 0.5",
           "median classes", "max classes", "items with >= 2 classes"), rows,
          "lrrrrrrrrrr")

    print("### Q3.2 Who is 'competent', and do the competent models split?\n")
    base = study.specs["base"]
    post = base.posterior
    assert post is not None
    by_item = views_by_item(base)
    rows = []
    for model in MODEL_ORDER:
        runs = [b for b in base.behaviours if b.interpreter.model == model]
        statuses = Counter(st for b in runs for st in b.state.steps.values())
        wrote = sum(1 for b in runs if any(
            b.state.tables.get(t) for t in ("so", "soline", "inv", "po", "grn", "vinv", "pay")))
        events = sum(1 for b in runs if any(
            e["actor"] != "" for e in b.state.tables.get("evt", [])))
        viol = statistics.mean(len(base.violated[b.run_key]) for b in runs)
        in_map = sum(
            1 for item, ip in post.items.items()
            if (c := Counter(v.klass for v in by_item[item] if v.model == model))
            and c.most_common(1)[0][0] in ip.map_set)
        split = robust = 0
        for item in post.items:
            c = Counter(v.klass for v in by_item[item] if v.model == model)
            if len(c) >= 2:
                split += 1
                if c.most_common(2)[1][1] >= 2:
                    robust += 1
        rows.append((SHORT[model], f"{post.theta[model]:.3f}", len(runs),
                     f"{viol:.1f}", pct(wrote, len(runs)), pct(events, len(runs)),
                     pct(statuses["error"], sum(statuses.values())),
                     f"{in_map}/{len(post.items)}", f"{split}/{len(post.items)}",
                     f"{robust}/{len(post.items)}"))
    table(("model (base)", "θ", "runs", "mean violated clauses per run",
           "runs writing any DE row (so, soline, inv, po, grn, vinv, pay)",
           "runs with any event row", "step statuses `error`",
           "items whose MAP set holds the model's modal class",
           "items where the model's runs fall in >= 2 classes",
           "... with the second class >= 2 runs"), rows, "lrrrrrrrrr")
    trajectory = []
    obs = [Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in base.views]
    for iters in (1, 3, 5, 8, 12, 20, 50):
        p = fit(obs, k_max=study.cfg.k_max, p_single=study.cfg.p_single, iters=iters)
        trajectory.append((iters, *[f"{p.theta[m]:.3f}" for m in MODEL_ORDER]))
    print("EM trajectory of θ on the base (latent.fit with `iters` capped; 50 is the "
          "default):\n")
    table(("EM iterations", *[f"θ {SHORT[m]}" for m in MODEL_ORDER]), trajectory, "rrrr")
    previous: dict[str, float] | None = None
    falling = {m: True for m in MODEL_ORDER}
    for iters in range(1, 51):
        theta = fit(obs, k_max=study.cfg.k_max, p_single=study.cfg.p_single, iters=iters).theta
        if previous is not None:
            for m in MODEL_ORDER:
                falling[m] &= theta[m] <= previous[m] + 1e-12
        previous = theta
    print("θ never rises between consecutive iterations 1..50: "
          + ", ".join(f"{SHORT[m]} {falling[m]}" for m in MODEL_ORDER) + ".\n")

    print("What separates the classes: over all base items with >= 2 classes, the pairs "
          "of the two most frequent classes, by what differs between them.\n")
    reasons: Counter[str] = Counter()
    for item in post.items:
        c = Counter(v.klass for v in by_item[item])
        if len(c) < 2:
            continue
        (a, _), (b, _) = c.most_common(2)
        va = next(v for v in by_item[item] if v.klass == a)
        vb = next(v for v in by_item[item] if v.klass == b)
        step = item.split("|", 1)[1]
        sa = base.runs[next(k for k, x in base.runs.items() if x.run_key == va.run_key)]
        sb = base.runs[next(k for k, x in base.runs.items() if x.run_key == vb.run_key)]
        fields = base.lens.diff(va.projection, vb.projection)
        if fields:
            reasons["locked fields differ"] += 1
        elif sa.state.steps.get(step) != sb.state.steps.get(step):
            reasons["only step status differs (done / error / needs_gate)"] += 1
        else:
            reasons["only the violation set differs"] += 1
    table(("difference between the two most frequent classes", "items"),
          reasons.most_common(), "lr")

    print("### Q3.3 The items closest to firing (base)\n")
    top = sorted(post.items, key=lambda i: -post.items[i].p_ambiguous)[:10]
    rows = []
    for item in top:
        ip = post.items[item]
        c = Counter(v.klass for v in by_item[item])
        names = {k: f"k{i}" for i, (k, _) in enumerate(c.most_common())}
        per_model = []
        for model in MODEL_ORDER:
            mc = Counter(names[v.klass] for v in by_item[item] if v.model == model)
            per_model.append(" ".join(f"{k}×{n}" for k, n in sorted(mc.items())))
        ranked = sorted(ip.p_member, key=lambda k: -ip.p_member[k])
        fields = _amb_fields(base, item, ranked[:2])
        lbf = lbfs["base"][item]
        step = item.split("|", 1)[1]
        by_key = {b.run_key: b for b in base.behaviours}
        status = {v.klass: by_key[v.run_key].state.steps.get(step) for v in by_item[item]}
        rows.append((item, *per_model, ",".join(names[k] for k in sorted(ip.map_set,
                                                                         key=names.get)),
                     f"{ip.p_ambiguous:.3f}", f"{lbf:.2f}", f"{p_single_needed(lbf):.3f}",
                     ", ".join(sorted(fields)) or "none",
                     ", ".join(f"{names[k]} {status[k]}" for k in ranked[:2])))
    table(("item", *[SHORT[m] for m in MODEL_ORDER], "MAP set", "P(|A|>=2)",
           "log BF (|A|>=2 vs 1)", "P(|A|=1) needed", "locked fields between top 2",
           "step status of top 2"), rows, "lllllrrrll")

    print("### Q3.4 Counterfactual: what prior or threshold would make an item cross\n")
    print("θ held at the fitted values. The posterior odds of |A| >= 2 are the Bayes factor "
          "times the prior odds (1 - P(|A|=1)) / P(|A|=1), so an item crosses 0.5 exactly "
          "when P(|A|=1) <= BF / (1 + BF). 'Reportable' adds diagnose's second condition: "
          "the two top classes differ on some locked field. 'New against base' are "
          "reportable (step, fields) identities the base would not also report at that "
          "prior (the differential removes those). For understanding only; no value is "
          "recommended.\n")
    rows = []
    for sid in ids:
        spec = study.specs[sid]
        post_s = spec.posterior
        assert post_s is not None
        values = lbfs[sid]
        needed = [p_single_needed(v) for v in values.values()]
        cells = [sid, f"{max(needed):.3f}",
                 fmt(max(ip.p_ambiguous for ip in post_s.items.values()))]
        for p in (0.75, 0.5, 0.25):
            crossing = sum(p_amb_at(v, p) >= 0.5 for v in values.values())
            keys = _counterfactual_keys(spec, values, p)
            new = keys - _counterfactual_keys(study.specs["base"], lbfs["base"], p)
            cells.append(f"{crossing} / {len(keys)} / {len(new)}")
        rows.append(cells)
    table(("spec", "largest P(|A|=1) at which any item crosses",
           "threshold needed (max P(|A|>=2))",
           "P(|A|=1)=0.75: crossing / reportable / new", "0.5: c / r / n",
           "0.25: c / r / n"), rows, "lrrlll")
    print("The same with θ re-estimated (a full EM refit at the counterfactual prior):\n")
    rows = []
    for p in (0.5, 0.25):
        base_keys: set[tuple[str, frozenset[str]]] | None = None
        for sid in ids:
            spec = study.specs[sid]
            refit = fit_views(spec.views, study.cfg.k_max, p)
            alt = dataclasses.replace(spec, posterior=refit)
            items = views_by_item(spec)
            values = {}
            for item in refit.items:
                m, ev = item_log_evidence(items[item], refit.theta, study.cfg.k_max)
                values[item] = log_bayes_factor(m, ev)
            keys = _counterfactual_keys(alt, values, p)
            if sid == "base":
                base_keys = keys
            assert base_keys is not None
            rows.append((p, sid, *[f"{refit.theta[m]:.3f}" for m in MODEL_ORDER],
                         f"{max(i.p_ambiguous for i in refit.items.values()):.3f}",
                         sum(i.p_ambiguous >= 0.5 for i in refit.items.values()),
                         len(keys), len(keys - base_keys)))
    table(("P(|A|=1)", "spec", *[f"θ {SHORT[m]}" for m in MODEL_ORDER], "max P(|A|>=2)",
           "items >= 0.5", "reportable", "new against base"), rows, "rlrrrrrrr")
    print("A-mutant items whose P(|A|>=2) differs from the base's by more than 0.01:\n")
    rows = []
    base_post = study.specs["base"].posterior
    assert base_post is not None
    for sid in A_IDS:
        post_s = study.specs[sid].posterior
        assert post_s is not None
        moved = [(i, base_post.items[i].p_ambiguous, post_s.items[i].p_ambiguous)
                 for i in post_s.items
                 if abs(post_s.items[i].p_ambiguous - base_post.items[i].p_ambiguous) > 0.01]
        rows.append((sid, len(moved), "; ".join(f"{i} {a:.3f}→{b:.3f}"
                                                for i, a, b in sorted(moved)[:6])
                     + (" …" if len(moved) > 6 else "")))
    table(("mutant", "items moved", "examples (base → mutant)"), rows, "lrl")


# ---------------------------------------------------------------------------- Q4


def _rate_by_model(spec: Spec, clauses: Iterable[str], scenarios: Iterable[str]
                   ) -> dict[str, tuple[int, int]]:
    """Per model: (runs violating any of the clauses, runs where one of them applies),
    over the given scenarios."""
    wanted = set(scenarios)
    ids = set(clauses)
    out: dict[str, tuple[int, int]] = {}
    for model in MODEL_ORDER:
        hit = n = 0
        for b in spec.behaviours:
            if b.interpreter.model != model or b.scenario.id not in wanted:
                continue
            if not any(spec.lens.clauses.applies(spec.lens.clauses[c], b.scenario) for c in ids):
                continue
            n += 1
            hit += bool(spec.violated[b.run_key] & ids)
        out[model] = (hit, n)
    return out


def _row_missing(spec: Spec, clause_id: str, b: Behaviour) -> bool | None:
    clause = spec.lens.clauses[clause_id]
    if clause.kind != "golden":
        return None
    p = clause.payload
    return not any(all(r.get(k) == v for k, v in p["key"].items())
                   for r in b.state.tables.get(p["table"], []))


def section_q4(study: Study) -> None:
    from specdx.infer.scores import run_weights

    print("## Q4. Where does the base noise come from?\n")
    base = study.specs["base"]
    post = base.posterior
    assert post is not None
    findings = base.line["defects"]["specdx"]
    print(f"The base report holds {len(findings)} findings: "
          + ", ".join(f"{n} {t}" for t, n in Counter(d["type"] for d in findings).most_common())
          + ". Every location is a single clause except the weak one: "
          f"{Counter(len(d['location']) for d in findings)}.\n")

    print("### Q4.1 Competence and the weight each model carries\n")
    weights = run_weights(base.views, post)
    by_model: dict[str, float] = defaultdict(float)
    model_of = {b.run_key: b.interpreter.model for b in base.behaviours}
    for k, w in weights.items():
        by_model[model_of[k]] += w
    total = sum(by_model.values())
    rows = [(SHORT[m], f"{post.theta[m]:.3f}",
             sum(1 for b in base.behaviours if b.interpreter.model == m),
             f"{100 * by_model[m] / total:.1f}%") for m in MODEL_ORDER]
    table(("model", "θ", "runs", "share of the §3.3 hitting-set weight (responsibility × θ)"),
          rows, "lrrr")

    print("### Q4.2 Failure rates per step (base, all 200 runs)\n")
    print("'error' is the step status the runtime records when the interpreter's step did "
          "not finish (no successful tool call, the step limit, a failed environment "
          "recorder). 'Wrote' means the step's slice holds at least one row at the end.\n")
    rows = []
    for step in study.bench.steps:
        cells: list[Any] = [step.id]
        for model in MODEL_ORDER:
            runs = [b for b in base.behaviours if b.interpreter.model == model]
            err = sum(1 for b in runs if b.state.steps.get(step.id) == "error")
            vio = sum(1 for b in runs if base.violated[b.run_key] & base.lens.relevant[step.id])
            proj = [base.lens.columns.project(b.state.tables, step.slice) for b in runs]
            wrote = sum(1 for p in proj if any(rows_ for _, rows_ in p))
            cells.append(f"{pct(err, len(runs))} / {pct(vio, len(runs))} / {pct(wrote, len(runs))}")
        cells.append(len(base.lens.relevant[step.id]))
        rows.append(cells)
    table(("step", *[f"{SHORT[m]}: error / any relevant violation / wrote" for m in MODEL_ORDER],
           "relevant clauses"), rows, "llllr")

    print("Why interpreted steps write so little: each interpreted step's final claim "
          "(9 interpreted steps per run), and the refused tool calls the behaviours record "
          "(name only; the refusal reason is not recorded). The two claim columns "
          "overlap.\n")
    rows = []
    for model in MODEL_ORDER:
        runs = [b for b in base.behaviours if b.interpreter.model == model]
        claims = [c for b in runs for s_, c in b.claims.items() if not s_.startswith("ENV:")]
        textual = sum(1 for c in claims if '"name"' in c
                      and ('"arguments"' in c or '"parameters"' in c))
        nothing = sum(1 for c in claims if "no tool call that succeeded" in c)
        refused = Counter(tool for b in runs for _, tool in b.refused)
        rows.append((SHORT[model], len(claims), f"{textual} ({pct(textual, len(claims))})",
                     f"{nothing} ({pct(nothing, len(claims))})", sum(refused.values()),
                     ", ".join(f"{t} {n}" for t, n in sorted(
                         refused.items(), key=lambda kv: (-kv[1], kv[0]))[:5])))
    table(("model", "interpreted step claims", "claim is a tool call written as text",
           "'finished with no tool call that succeeded'", "refused calls",
           "refused, by tool (top 5)"), rows, "lrrrrl")

    print("### Q4.3 Failure rates per clause (base)\n")
    kinds: dict[str, list[float]] = defaultdict(list)
    per_clause: dict[str, tuple[int, int]] = {}
    for clause in base.lens.clauses:
        n = hit = 0
        for b in base.behaviours:
            if base.lens.clauses.applies(clause, b.scenario):
                n += 1
                hit += clause.id in base.violated[b.run_key]
        if n:
            per_clause[clause.id] = (hit, n)
            kinds[clause.kind].append(hit / n)
    rows = []
    for kind, rates in sorted(kinds.items()):
        rows.append((kind, len(rates), sum(r == 1 for r in rates), sum(0.5 <= r < 1 for r in rates),
                     sum(0 < r < 0.5 for r in rates), sum(r == 0 for r in rates),
                     f"{statistics.mean(rates):.2f}"))
    allr = [r for rs in kinds.values() for r in rs]
    rows.append(("all", len(allr), sum(r == 1 for r in allr), sum(0.5 <= r < 1 for r in allr),
                 sum(0 < r < 0.5 for r in allr), sum(r == 0 for r in allr),
                 f"{statistics.mean(allr):.2f}"))
    table(("clause kind", "clauses applicable somewhere", "violated by every applicable run",
           "50–99%", "1–49%", "never", "mean rate"), rows, "lrrrrrr")

    print("### Q4.4 The 91 findings: shared failure or model-specific?\n")
    print("For each finding, per model: the share of its runs (in the finding's scenarios, "
          "where the clause applies) that violate the location. 'All fail' = every run of "
          "every model violates it; 'every model >= 50%'; 'model-specific' = some model "
          "violates it in under half its runs.\n")
    cats: Counter[tuple[str, str]] = Counter()
    detail: list[tuple[Any, ...]] = []
    missing_rows = missing_total = 0
    for d in findings:
        if d["type"] == "weak":
            cats[(d["type"], "weak (field location)")] += 1
            continue
        rates = _rate_by_model(base, d["location"], d["scenarios"])
        shares = {m: (h / n if n else 0.0) for m, (h, n) in rates.items()}
        if all(s == 1 for s in shares.values()):
            cat = "all fail"
        elif all(s >= 0.5 for s in shares.values()):
            cat = "every model >= 50%"
        else:
            cat = "model-specific"
        cats[(d["type"], cat)] += 1
        clause = d["location"][0]
        for b in base.behaviours:
            if b.scenario.id in d["scenarios"] and clause in base.violated[b.run_key]:
                missing = _row_missing(base, clause, b)
                if missing is not None:
                    missing_total += 1
                    missing_rows += missing
        detail.append((d["type"], clause, ", ".join(d["scenarios"]) if len(d["scenarios"]) < 3
                       else f"{len(d['scenarios'])} scenarios", cat,
                       *[f"{h}/{n}" for h, n in rates.values()], d["note"][:40]))
    rows = [(t, c, n) for (t, c), n in sorted(cats.items())]
    table(("type", "category", "findings"), rows, "llr")
    hidden = [d for d in findings if d["type"] == "hidden"]
    buckets = Counter("0" if d["score"] == 0 else ("1" if d["score"] >= 1 - 1e-9 else
                      ("[0.5, 1)" if d["score"] >= 0.5 else "(0, 0.5)")) for d in hidden)
    notes = Counter(d["note"].split(" after")[0].split(":")[0] for d in findings
                    if d["type"] != "weak")
    print(f"Hidden(c) scores of the {len(hidden)} hidden findings: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(buckets.items()))
          + ". diagnose() reports every singleton hitting set whatever its Hidden(c) "
          "score; no threshold applies to it. Notes: "
          + ", ".join(f"{k}: {v}" for k, v in notes.most_common()) + ".\n")
    rows = []
    for d in findings:
        if d["type"] != "contradiction":
            continue
        clause = d["location"][0]
        runs = [b for b in base.behaviours if b.scenario.id in d["scenarios"]]
        violating = [b for b in runs if clause in base.violated[b.run_key]]
        keyed = [_row_missing(base, clause, b) for b in violating]
        rows.append((clause, ", ".join(d["scenarios"]), f"{len(violating)}/{len(runs)}",
                     f"{sum(bool(k) for k in keyed)}" if keyed and keyed[0] is not None
                     else "–",
                     sum(1 for b in violating if not b.state.tables.get("vinv")),
                     d["note"]))
    print("The base contradictions:\n")
    table(("clause", "scenario", "runs violating", "keyed row missing", "no vinv row",
           "note"), rows, "llrrrl")
    print(f"Of the violations behind golden-state findings, {missing_rows} of {missing_total} "
          f"({pct(missing_rows, missing_total)}) are a missing row (the run never wrote the "
          "keyed row), not a wrong value.\n")
    print("Model-specific findings (the rest are listed in the script's data):\n")
    table(("type", "clause", "scenarios", "category",
           *[f"{SHORT[m]} violating/runs" for m in MODEL_ORDER], "note"),
          [r for r in detail if r[3] == "model-specific"], "lllllll")

    print("### Q4.5 Concentration of the findings\n")
    by_step: Counter[str] = Counter()
    by_table: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    by_scn: Counter[str] = Counter()
    for d in findings:
        if d["type"] == "weak":
            continue
        clause = base.lens.clauses[d["location"][0]]
        owners = [s for s, rel in base.lens.relevant.items() if clause.id in rel]
        for s in owners or ["(no step)"]:
            by_step[s] += 1
        by_table[",".join(sorted(clause.tables))] += 1
        by_kind[clause.kind] += 1
        by_scn["grn12 only" if d["scenarios"] == ["grn12"] else
               ("one other scenario" if len(d["scenarios"]) == 1 else "several scenarios")] += 1
    table(("owning step (a clause can belong to several)", "findings"), by_step.most_common(), "lr")
    table(("clause kind", "findings"), by_kind.most_common(), "lr")
    table(("tables", "findings"), by_table.most_common(12), "lr")
    table(("scenarios of the finding", "findings"), by_scn.most_common(), "lr")

    print("### Q4.6 The masked H1 truths\n")
    gold = CanonPack(base.variant.pack).tables
    rows = []
    for mid in H1_IDS:
        spec = study.specs[mid]
        clause_id = spec.line["truth"]["location"][0]
        _, table_name, ident, column = clause_id.split(":")
        key = base.lens.columns.keys[table_name][0]
        golden = next(r[column] for r in gold[table_name] if r[key] == ident)
        mutant_golden = next(r[column] for r in CanonPack(spec.variant.pack).tables[table_name]
                             if r[key] == ident)
        bf = [d for d in findings if d["location"] == [clause_id]]
        mf = [d for d in spec.line["defects"]["specdx"] if d["location"] == [clause_id]]
        scope = base.lens.clauses[clause_id].scope
        for model in MODEL_ORDER:
            for scope_name, scns in (("grn12", {"grn12"}),
                                     ("all", set(scenario_order(base)))):
                values: Counter[str] = Counter()
                for b in base.behaviours:
                    if b.interpreter.model != model or b.scenario.id not in scns:
                        continue
                    rowsv = [r for r in b.state.tables.get(table_name, []) if r.get(key) == ident]
                    values[rowsv[0][column] if rowsv else "no row"] += 1
                rows.append((mid, f"`{clause_id}` ({scope})", f"{golden} → {mutant_golden}",
                             SHORT[model], scope_name, values[golden], values[mutant_golden],
                             "; ".join(f"{v} ×{n}" for v, n in values.most_common())))
        print(f"- **{mid}** `{clause_id}`: base report "
              + ("; ".join(f"{d['type']} ({d['note']}, score {d['score']:.3f})" for d in bf)
                 or "nothing")
              + "; mutant report "
              + ("; ".join(f"{d['type']} ({d['note']}, score {d['score']:.3f})" for d in mf)
                 or "nothing")
              + f"; the mutant line's SpecDx matches: {spec.line['matches']['specdx']}.")
    print()
    table(("mutant", "truth clause (scope)", "golden base → mutant", "model", "runs",
           "= base golden", "= mutant golden", "values in the base runs"), rows, "lllllrrl")


# ---------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the Markdown here instead of stdout")
    args = parser.parse_args(argv)
    study = Study()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("<!-- generated by experiments/p1_dev_diagnostics.py -->\n")
        for section in (section_mapping, section_q1, section_q2, section_q3, section_q4):
            section(study)
    text = buffer.getvalue()
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
