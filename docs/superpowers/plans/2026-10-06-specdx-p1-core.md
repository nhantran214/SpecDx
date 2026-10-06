# SpecDx Phase P1 (Core on S1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SpecDx core on the TonomyX canon (subject S1): it diagnoses defects in an executable business specification by running a population of local LLM interpreters, then passes the pilot go/no-go gate of spec §4.8.

**Architecture:**
- TonomyX is vendored byte-identical and frozen. SpecDx drives it only through its public runtime and gated tools, on its own PostgreSQL (port 55434).
- One run of one interpreter on one scenario yields a `Behaviour`, meaning the end state of CASE-0031 plus per-step statuses.
- The `infer/` layer turns behaviours into a posterior (a set-valued latent-class model) and into typed defects (soft minimal hitting sets plus refutation).
- The `active/` layer chooses scenarios (Thompson sampling), refutes contradiction candidates, and localises ambiguity with ddmin.
- `mutate/` builds seeded-defect copies of the specification. B2, B3 and the matching rules score the gate.

**Tech stack:** Python 3.13, PostgreSQL 16 (Docker), psycopg 3, httpx, numpy, pytest, ruff, Ollama (local), poppler `pdftotext`.

**Spec:** `docs/superpowers/specs/2026-10-06-specdx-design.md`. Read §2–§4, §6, §9 and §11 before starting. §11 records clarifications made while writing this plan.

**Pre-check (2026-10-06):** every code block of Tasks 1–19 was materialised in a scratch directory and run:
- all 130 tests passed, the non-database ones and the database ones (against a temporary PostgreSQL 16). One of them, the default-port check in `tests/test_vendor.py`, had to be deselected because the pre-check deliberately ran on another port;
- `ruff check` is clean.

Expected test counts in each task come from that run. If a count differs, re-read the task before changing code.

## Global Constraints

- `vendor/tonomyx/` is a byte-identical `git archive` of TonomyX commit `34c1d2ce70583b2de54573d28ba4f49e0af4cc7f`. Never edit anything under `vendor/`.
- Never write into the canon directory. Mutants are copies under `work/mutants/`.
- Nothing is imported from `/home/ethan/AILab/FAA` or `aebench`.
- Database: `postgresql://tonomyx:tonomyx@localhost:55434/tonomyx` (env `SPECDX_DATABASE_URL`). TonomyX uses 55432 and FAA 55433; never touch those.
- Ollama: `http://localhost:11434` (env `SPECDX_OLLAMA_URL`). Local models only, no hosted APIs.
- Code, comments, identifiers and test names in English. Vietnamese only in canon data, Tell text and prompt strings.
- Tell playbooks from TonomyX (`tonomyx/flow/playbooks.json`) are never shown to an interpreter.
- Defaults (spec §3.10):
  - temperature 0.7, seeds (1, 2, 3);
  - BM25 over 300-word chunks, k = 5;
  - P(|A_s| = 1) = 0.9, K_max = 6;
  - τ = 0.05, MHS size ≤ 3;
  - refutation: 3 interpreters × 3 attempts, ≤ 200 repair evaluations;
  - ε = 0.01 nats, Ambig threshold 0.5.
- Anchor scenario is `grn12`. Its parameters are ordered 15, received 12, discount 0, GRNs per PO 1. `grn_count` is structural.
- Every DB test is marked `@pytest.mark.db` and needs `docker compose up -d`.
- Run tests with `.venv/bin/python -m pytest`; lint with `.venv/bin/ruff check .`

## Review Focus

Five conditions the spec implies but the main tests do not exercise. Each has a test added in the task that owns the code:

1. **A model call that times out mid-run.** The step must end `error` and the case must continue to the next step. The behaviour is still recorded. *(Task 9, `test_a_timeout_ends_the_step_not_the_case`.)*
2. **A mutant whose SQL fails to bootstrap.** The gate must record that mutant as failed and carry on with the others. *(Task 19, `test_gate_records_a_broken_mutant_and_continues`.)*
3. **A run cache whose last line was cut off by a crash.** Loading must skip the partial line, not refuse the whole file. *(Task 10, `test_a_truncated_last_line_is_skipped`.)*
4. **Vietnamese text extracted from a PDF in decomposed Unicode (NFD)** while the KB file is composed (NFC). `TellCorpus.find` must still match. *(Task 7, `test_find_matches_across_unicode_normal_forms`.)*
5. **An environment step that follows a failed digital-employee step** (inspection when no GRN-0015 exists). It must be recorded as `error`, and the database transaction must stay usable for the remaining steps. *(Task 8, `test_a_failed_step_does_not_poison_later_steps`.)*

---

## File Structure

```
pyproject.toml · docker-compose.yml · .gitignore · scripts/verify_vendor.sh
vendor/VENDORED.md · vendor/tonomyx/                      (frozen)
specdx/
  config.py            paths, URLs, settings_for()
  types.py             Scenario, State, InterpreterId, Behaviour, Defect, RefuteResult, Refuter
  variant.py           SpecVariant (pack dir + Tell corpus)
  db.py                PackDatabase: load a pack, isolated runs, materialise states
  scenarios/space.py   canon scenarios, boundary domain, cells
  spec/columns.py      locked/blank/ignored columns, projections, field diffs
  spec/lifting.py      clause lifting and expectation rules
  spec/clauses.py      Clause, ClauseSet, evaluation, derived outcome
  spec/tell.py         Tell corpus from PDFs + KB + principles, BM25
  orchestrate/work_orders.json · work_orders.py   the CASE-0031 work order
  orchestrate/flow.py  parametric case runner on TonomyX's public runtime
  orchestrate/reference.py  scripted persona handlers for reproduction
  interpreters/variants.json · prompt.py · population.py · executor.py · cache.py · paraphrase.py
  observe/state.py     capture_state, fn_reconcile probe
  observe/run.py       Runner: one cached run → Behaviour
  infer/items.py       SpecLens: violations, item views, class keys
  infer/latent.py      set-valued latent-class EM
  infer/hitting.py     soft minimal hitting sets
  infer/scores.py      diagnose → typed defects
  active/refute.py     SymbolicRepair, DirectedProbe, CombinedRefuter
  active/search.py     information gain, Normal–Gamma bandit, escalation
  active/ddmin.py      ddmin and ambiguity localisation
  active/loop.py       SpecDx main loop
  mutate/pack_copy.py  copy a pack, edit CSV/JSON/text, regenerate SQL
  mutate/operators.py  mutation operators with ground truth and kill witnesses
  baselines/b2_single.py · baselines/b3_irt.py
  eval/matching.py     location matching, typed P/R/F1, gate verdict
  eval/pilot.py        capability check, gate runner
  workbench.py         wires db, cache, corpus, runner, loop context
  cli.py
experiments/p1-gate.json
results/               gate outputs (committed)
tests/                 mirrors specdx/
```

---

### Task 1: Repository scaffold, vendoring, database container

**Files:**
- Create: `pyproject.toml`, `docker-compose.yml`, `.gitignore`, `scripts/verify_vendor.sh`, `vendor/VENDORED.md`, `specdx/__init__.py`, `specdx/config.py`, `tests/__init__.py`, `tests/test_vendor.py`
- Create (by archive): `vendor/tonomyx/`

**Interfaces:**
- Produces: `specdx.config.REPO_ROOT`, `VENDOR`, `BASE_PACK`, `PDF_DIR`, `WORK_DIR`, `DATABASE_URL`, `OLLAMA_URL`, and `settings_for(pack: Path, url: str = DATABASE_URL) -> tonomyx.config.Settings`.

- [ ] **Step 1: Vendor TonomyX at the pinned commit**

```bash
cd /home/ethan/AILab/TonomyX-paper
mkdir -p vendor/tonomyx
git -C /home/ethan/AILab/TonomyX archive 34c1d2ce70583b2de54573d28ba4f49e0af4cc7f \
  tonomyx pyproject.toml data/canon_pack_CASE-0031_v2.0 demo | tar -x -C vendor/tonomyx
ls vendor/tonomyx
```
Expected: `data  demo  pyproject.toml  tonomyx`.

- [ ] **Step 2: Write the project files**

`pyproject.toml`:
```toml
[project]
name = "specdx"
version = "0.1.0"
description = "SpecDx: probabilistic diagnosis of executable business specifications with populations of LLM interpreters"
requires-python = ">=3.12"
dependencies = ["numpy>=2.1", "httpx>=0.27", "psycopg[binary]>=3.2"]

[project.optional-dependencies]
dev = ["pytest>=8.3", "ruff>=0.7"]

[project.scripts]
specdx = "specdx.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["specdx*"]

[tool.setuptools.package-data]
specdx = ["orchestrate/*.json", "interpreters/*.json"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["db: needs the SpecDx PostgreSQL on port 55434 (docker compose up -d)"]

[tool.ruff]
line-length = 100
target-version = "py312"
exclude = ["vendor/"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

`docker-compose.yml`:
```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: specdx-postgres
    environment:
      POSTGRES_USER: tonomyx
      POSTGRES_PASSWORD: tonomyx
      POSTGRES_DB: tonomyx
      LANG: C.UTF-8
    ports:
      - "55434:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tonomyx -d tonomyx"]
      interval: 2s
      timeout: 3s
      retries: 30
```

`.gitignore`:
```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
work/
.claude/
```

`scripts/verify_vendor.sh`:
```bash
#!/usr/bin/env bash
# Prove vendor/tonomyx is byte-identical to the pinned TonomyX commit.
set -euo pipefail
SRC=/home/ethan/AILab/TonomyX
COMMIT=34c1d2ce70583b2de54573d28ba4f49e0af4cc7f
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
git -C "$SRC" archive "$COMMIT" tonomyx pyproject.toml data/canon_pack_CASE-0031_v2.0 demo | tar -x -C "$TMP"
diff -rq -x __pycache__ -x '*.egg-info' "$TMP" "$HERE/vendor/tonomyx"
echo "vendor/tonomyx matches $COMMIT"
```

`vendor/VENDORED.md`:
````markdown
# Vendored artefacts

## tonomyx

**Source:** `/home/ethan/AILab/TonomyX`, commit `34c1d2ce70583b2de54573d28ba4f49e0af4cc7f` (master, 2026-09-30). Its `tonomyx/` and `data/` trees are identical to commit `9de7c28`, the one FAA vendored.

**Copied with:** `git archive 34c1d2c tonomyx pyproject.toml data/canon_pack_CASE-0031_v2.0 demo`. That is the package, its build file, the canon pack, and the three requirement PDFs that form the Tell corpus.

**Rule:** nothing under `vendor/` is ever edited. Mutants are copies under `work/mutants/`.

**Check:** `bash scripts/verify_vendor.sh` prints `vendor/tonomyx matches 34c1d2c…` and nothing else.
````

`specdx/__init__.py`:
```python
"""SpecDx: probabilistic diagnosis of executable business specifications."""
```

`tests/__init__.py`: empty file.

`specdx/config.py`:
```python
"""Paths and endpoints, read once from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from tonomyx.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR = REPO_ROOT / "vendor" / "tonomyx"
BASE_PACK = VENDOR / "data" / "canon_pack_CASE-0031_v2.0" / "canon_pack_CASE-0031"
PDF_DIR = VENDOR / "demo"
WORK_DIR = Path(os.environ.get("SPECDX_WORK", str(REPO_ROOT / "work")))
DATABASE_URL = os.environ.get(
    "SPECDX_DATABASE_URL", "postgresql://tonomyx:tonomyx@localhost:55434/tonomyx"
)
OLLAMA_URL = os.environ.get("SPECDX_OLLAMA_URL", "http://localhost:11434")


def settings_for(pack: Path, url: str = DATABASE_URL) -> Settings:
    """TonomyX settings pointing at one specification and the SpecDx database."""
    if not (pack / "canon.meta.json").is_file():
        raise FileNotFoundError(f"not a canon pack: {pack} has no canon.meta.json")
    return Settings(canon_pack=pack, database_url=url, smtp_host="localhost", smtp_port=1025)
```

- [ ] **Step 3: Write the failing smoke test**

`tests/test_vendor.py`:
```python
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK, PDF_DIR, settings_for


def test_the_vendored_package_reads_the_vendored_pack() -> None:
    pack = CanonPack(BASE_PACK)
    assert len(pack.invariants) == 42
    assert pack.meta["canon_version"] == "CASE-0031-ADV-2.0"


def test_the_three_requirement_documents_are_vendored() -> None:
    assert len(sorted(PDF_DIR.glob("*.pdf"))) == 3


def test_settings_point_at_the_specdx_database() -> None:
    assert settings_for(BASE_PACK).database_url.endswith(":55434/tonomyx")
```

- [ ] **Step 4: Run it to see it fail**

Run: `python3 -m pytest tests/test_vendor.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tonomyx'` (no environment yet).

- [ ] **Step 5: Create the environment, start the database, run again**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"
docker compose up -d
bash scripts/verify_vendor.sh
.venv/bin/python -m pytest tests/test_vendor.py -q
```
Expected: `vendor/tonomyx matches 34c1d2ce…`, then `3 passed`.

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/verify_vendor.sh
git add pyproject.toml docker-compose.yml .gitignore scripts vendor specdx tests
git commit -m "chore: scaffold SpecDx and vendor TonomyX at 34c1d2c"
```

---

### Task 2: Core value types and the scenario space

**Files:**
- Create: `specdx/types.py`, `specdx/scenarios/__init__.py`, `specdx/scenarios/space.py`
- Test: `tests/test_types.py`, `tests/scenarios/__init__.py`, `tests/scenarios/test_space.py`

**Interfaces:**
- Produces:
  - `PARAMETERS`, `STRUCTURAL`, `DEFECT_TYPES`, `FIRST_GRN`, `SECOND_GRN`.
  - `Scenario(id, ordered, received, discount_pct, grn_count=1)`, with `.params() -> dict[str, int]`, `.signature() -> tuple[int, ...]` and `.deliveries() -> list[tuple[str, int]]`.
  - `State(tables, derived={}, steps={})`, with `.empty()`.
  - `InterpreterId(model, family, variant, seed)`, with `.key`.
  - `Behaviour(run_key, interpreter, scenario, state, refused=(), claims={}, status="ok", error="")`, with `.to_json()` and `.from_json()`.
  - `Defect(type, location, scenarios, witnesses, score, note="")`, with `.key`.
  - `RefuteResult(refuted, witness, attempts)` and the `Refuter` protocol `refute(targets, scenario, behaviours) -> RefuteResult`.
  - `space.ANCHOR`, `canon_scenarios(pack)`, `discount_cap_pct(pack)`, `boundary_domain(pack)`, `enumerate_scenarios(domain)`, `cell_of(scenario, cap_pct)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_types.py`:
```python
from specdx.types import Behaviour, Defect, InterpreterId, Scenario, State


def test_deliveries_split_a_receipt_in_two_when_asked() -> None:
    assert Scenario("x", 15, 12, 0).deliveries() == [("GRN-0015", 12)]
    assert Scenario("x", 15, 13, 0, grn_count=2).deliveries() == [
        ("GRN-0015", 7), ("GRN-0016", 6)
    ]


def test_signature_ignores_the_name() -> None:
    assert Scenario("happy", 15, 15, 0).signature() == Scenario("o15", 15, 15, 0).signature()


def test_a_behaviour_survives_a_json_round_trip() -> None:
    b = Behaviour(
        run_key="k",
        interpreter=InterpreterId("qwen3:14b", "qwen", 0, 1),
        scenario=Scenario("grn12", 15, 12, 0),
        state=State(
            {"so": [{"ma_so": "SO-0031"}]}, {"fn_reconcile:PO-0015": "Khớp"}, {"s": "done"}
        ),
        refused=(("s", "write_draft"),),
        claims={"s": "Xong."},
    )
    assert Behaviour.from_json(b.to_json()) == b


def test_a_defect_key_is_its_type_and_location() -> None:
    d = Defect("hidden", ("golden:so:SO-0031:trang_thai",), ("grn12",), ("k",), 0.9)
    assert d.key == ("hidden", ("golden:so:SO-0031:trang_thai",))
```

`tests/scenarios/__init__.py`: empty.

`tests/scenarios/test_space.py`:
```python
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.scenarios.space import (
    boundary_domain,
    canon_scenarios,
    cell_of,
    discount_cap_pct,
    enumerate_scenarios,
)
from specdx.types import Scenario

PACK = CanonPack(BASE_PACK)


def test_the_canon_scenarios_carry_the_canon_parameters() -> None:
    canon = canon_scenarios(PACK)
    assert canon["grn12"] == Scenario("grn12", 15, 12, 0)
    assert canon["happy"] == Scenario("happy", 15, 15, 0)
    assert canon["discount8"] == Scenario("discount8", 15, 15, 8)


def test_the_discount_cap_is_read_from_the_policy_table() -> None:
    assert discount_cap_pct(PACK) == 5


def test_the_boundary_domain_brackets_every_literal() -> None:
    assert boundary_domain(PACK) == {
        "ordered": [14, 15, 16],
        "received": [11, 12, 13, 14, 15, 16],
        "discount_pct": [0, 4, 5, 6, 8],
        "grn_count": [1, 2],
    }


def test_enumeration_keeps_only_physical_scenarios() -> None:
    scenarios = enumerate_scenarios(boundary_domain(PACK))
    assert all(s.received <= s.ordered + 1 and s.received >= s.grn_count for s in scenarios)
    assert any(s.received == s.ordered + 1 for s in scenarios)
    assert len({s.id for s in scenarios}) == len(scenarios)


def test_cells_group_by_relation_not_by_value() -> None:
    assert cell_of(Scenario("a", 15, 12, 0), 5) == cell_of(Scenario("b", 16, 11, 0), 5)
    assert cell_of(Scenario("a", 15, 15, 8), 5) == "recv-eq|disc-gt|grn-1"
    assert cell_of(Scenario("a", 15, 16, 4, 2), 5) == "recv-gt|disc-le|grn-2"
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_types.py tests/scenarios -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.types'`.

- [ ] **Step 3: Implement**

`specdx/types.py`:
```python
"""Value types shared by every SpecDx unit."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

# Scenario parameters (spec §2.1). grn_count changes which rows exist, so it is structural.
PARAMETERS: tuple[str, ...] = ("ordered", "received", "discount_pct", "grn_count")
STRUCTURAL: frozenset[str] = frozenset({"grn_count"})
DEFECT_TYPES: tuple[str, ...] = ("contradiction", "hidden", "ambiguity", "weak")

FIRST_GRN = "GRN-0015"
SECOND_GRN = "GRN-0016"


@dataclass(frozen=True)
class Scenario:
    """One point of the scenario space."""

    id: str
    ordered: int
    received: int
    discount_pct: int
    grn_count: int = 1

    def params(self) -> dict[str, int]:
        return {
            "ordered": self.ordered,
            "received": self.received,
            "discount_pct": self.discount_pct,
            "grn_count": self.grn_count,
        }

    def signature(self) -> tuple[int, ...]:
        """The parameters alone: two scenarios with equal signatures are the same scenario."""
        return tuple(self.params()[p] for p in PARAMETERS)

    def deliveries(self) -> list[tuple[str, int]]:
        """How the received units arrive: one receipt, or two when grn_count is 2."""
        if self.grn_count == 1:
            return [(FIRST_GRN, self.received)]
        if self.grn_count == 2:
            first = self.received - self.received // 2
            return [(FIRST_GRN, first), (SECOND_GRN, self.received // 2)]
        raise ValueError(f"grn_count must be 1 or 2, got {self.grn_count}")


@dataclass(frozen=True)
class State:
    """A database end state as text, plus values only the database can compute."""

    tables: dict[str, list[dict[str, str]]]
    derived: dict[str, str] = field(default_factory=dict)
    steps: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> State:
        return cls(tables={})


@dataclass(frozen=True)
class InterpreterId:
    """One member of the population: a model read through one Tell variant with one seed."""

    model: str
    family: str
    variant: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}|v{self.variant}|s{self.seed}"


@dataclass(frozen=True)
class Behaviour:
    """What one interpreter did on one scenario."""

    run_key: str
    interpreter: InterpreterId
    scenario: Scenario
    state: State
    refused: tuple[tuple[str, str], ...] = ()
    claims: dict[str, str] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Behaviour:
        raw = json.loads(text)
        return cls(
            run_key=raw["run_key"],
            interpreter=InterpreterId(**raw["interpreter"]),
            scenario=Scenario(**raw["scenario"]),
            state=State(**raw["state"]),
            refused=tuple((str(a), str(b)) for a, b in raw["refused"]),
            claims=dict(raw["claims"]),
            status=raw["status"],
            error=raw["error"],
        )


@dataclass(frozen=True)
class Defect:
    """One typed finding about the specification."""

    type: str
    location: tuple[str, ...]
    scenarios: tuple[str, ...]
    witnesses: tuple[str, ...]
    score: float
    note: str = ""

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return (self.type, self.location)


@dataclass(frozen=True)
class RefuteResult:
    """Whether a state satisfying the targets was found (spec §3.6)."""

    refuted: bool
    witness: str | None
    attempts: int


class Refuter(Protocol):
    """Anything that can try to satisfy a set of clauses on one scenario."""

    def refute(
        self, targets: frozenset[str], scenario: Scenario, behaviours: Sequence[Behaviour]
    ) -> RefuteResult: ...
```

`specdx/scenarios/__init__.py`: empty.

`specdx/scenarios/space.py`:
```python
"""The scenario space Π for S1 (spec §2.1, §3.5)."""

from __future__ import annotations

import itertools

from tonomyx.canon.pack import CanonPack
from tonomyx.flow.handlers import PO
from tonomyx.policy.rules import DISCOUNT_CAP_KEY

from specdx.types import Scenario

# The canon scenario whose end state the invariants and the golden state describe.
ANCHOR = "grn12"


def canon_scenarios(pack: CanonPack) -> dict[str, Scenario]:
    """The canon's own scenarios. The ordered quantity is the canon's purchase order."""
    ordered = int(next(r for r in pack.tables["po"] if r["ma_po"] == PO)["sl_dat"])
    return {
        name: Scenario(
            id=name,
            ordered=ordered,
            received=int(s["grn_received"]),
            discount_pct=int(s["intake_discount_pct"]),
        )
        for name, s in pack.scenarios.items()
    }


def discount_cap_pct(pack: CanonPack) -> int:
    """The discount cap as a whole percentage, from the policy table."""
    row = next(r for r in pack.tables["policy"] if r["ma_nguong"] == DISCOUNT_CAP_KEY)
    return round(float(row["gia_tri"]) * 100)


def boundary_domain(pack: CanonPack) -> dict[str, list[int]]:
    """Canon values plus the values on either side of every literal that matters."""
    canon = canon_scenarios(pack)
    anchor = canon[ANCHOR]
    ordered = anchor.ordered
    cap = discount_cap_pct(pack)
    return {
        "ordered": sorted({ordered - 1, ordered, ordered + 1}),
        "received": sorted(
            {s.received for s in canon.values()}
            | {ordered - 1, ordered, ordered + 1}
            | {anchor.received - 1, anchor.received + 1}
        ),
        "discount_pct": sorted({s.discount_pct for s in canon.values()} | {cap - 1, cap, cap + 1}),
        "grn_count": [1, 2],
    }


def enumerate_scenarios(domain: dict[str, list[int]]) -> list[Scenario]:
    """Every physical combination: at most one unit over-delivered, one unit per receipt."""
    out: list[Scenario] = []
    for o, r, d, g in itertools.product(
        domain["ordered"], domain["received"], domain["discount_pct"], domain["grn_count"]
    ):
        if r > o + 1 or r < g:
            continue
        out.append(Scenario(f"o{o}-r{r}-d{d}-g{g}", o, r, d, g))
    return out


def cell_of(scenario: Scenario, cap_pct: int) -> str:
    """The relational cell a scenario falls in: what is compared with what, not the values."""
    if scenario.received < scenario.ordered:
        rel = "lt"
    elif scenario.received == scenario.ordered:
        rel = "eq"
    else:
        rel = "gt"
    if scenario.discount_pct == 0:
        disc = "zero"
    elif scenario.discount_pct <= cap_pct:
        disc = "le"
    else:
        disc = "gt"
    return f"recv-{rel}|disc-{disc}|grn-{scenario.grn_count}"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_types.py tests/scenarios -q`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/types.py specdx/scenarios tests/test_types.py tests/scenarios
git commit -m "feat: core value types and the S1 scenario space"
```

---

### Task 3: The database layer and state capture

**Files:**
- Create: `specdx/db.py`, `specdx/observe/__init__.py`, `specdx/observe/state.py`, `tests/conftest.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `State` (Task 2), `settings_for` (Task 1).
- Produces:
  - `db.pack_digest(path) -> str`, `golden_tables(pack) -> dict[str, list[dict[str, str]]]`, `split_background(tables) -> (background, case)`, `write_rows(conn, pack, tables)`, `BACKGROUND_TABLES`.
  - `PackDatabase(url=DATABASE_URL)`, with `.use(pack_dir) -> Settings`, `.settings`, `.pack`, `.isolated()` (a context manager yielding a connection), `.materialize(state) -> State | None` and `.golden_state() -> State`.
  - `observe.state.ERROR`, `capture_state(conn, pack, steps=None) -> State`, `reconcile_value(conn, po) -> str`.
  - Fixtures `pack` and `db` (which re-selects the base pack on every use), and `golden`.

- [ ] **Step 1: Write the fixtures and the failing tests**

`tests/conftest.py`:
```python
"""Shared fixtures. The database fixture always hands back the base specification."""

from __future__ import annotations

import pytest
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.db import PackDatabase
from specdx.types import State

_DB = PackDatabase()


@pytest.fixture(scope="session")
def pack() -> CanonPack:
    return CanonPack(BASE_PACK)


@pytest.fixture
def db() -> PackDatabase:
    """The one database, loaded with the base pack (cheap when it already is)."""
    _DB.use(BASE_PACK)
    return _DB


@pytest.fixture
def golden(db: PackDatabase) -> State:
    return db.golden_state()
```

`tests/test_db.py`:
```python
import shutil

import pytest

from specdx.config import BASE_PACK
from specdx.db import pack_digest, split_background
from specdx.types import State

pytestmark = pytest.mark.db


def test_a_run_starts_from_master_data_and_background_rows(db) -> None:
    with db.isolated() as conn:
        assert conn.execute("SELECT count(*) AS n FROM so").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM cust").fetchone()["n"] > 0
        assert conn.execute("SELECT count(*) AS n FROM cand").fetchone()["n"] == 3
        aprs = conn.execute("SELECT ma_apr FROM apr").fetchall()
        assert [r["ma_apr"] for r in aprs] == ["APR-TH-01"]


def test_whatever_a_run_writes_is_rolled_back(db) -> None:
    with db.isolated() as conn:
        conn.execute("INSERT INTO cases (case_id, tieu_de, trang_thai) VALUES ('C-X', 't', 's')")
    with db.isolated() as conn:
        assert conn.execute("SELECT count(*) AS n FROM cases").fetchone()["n"] == 0


def test_the_golden_state_is_the_canon_end_state(golden) -> None:
    po = golden.tables["po"][0]
    assert (po["ma_po"], po["sl_dat"], po["trang_thai"]) == ("PO-0015", "15", "Đã duyệt")
    assert golden.tables["grn"][0]["sl_nhan"] == "12"
    assert golden.derived["fn_reconcile:PO-0015"] == "Cần kiểm"
    assert {r["ma_apr"] for r in golden.tables["apr"]} == {
        "APR-0102", "APR-0155", "APR-0210", "APR-TH-01"
    }


def test_two_receipts_for_one_order_make_fn_reconcile_error(db, golden) -> None:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["grn"].append({"ma_grn": "GRN-0016", "po": "PO-0015", "sl_nhan": "3"})
    real = db.materialize(State(tables))
    assert real is not None
    assert real.derived["fn_reconcile:PO-0015"] == "ERROR"


def test_a_state_the_database_refuses_does_not_materialise(db, golden) -> None:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["evt"].append({"case_id": "CASE-0031", "kind": "BOGUS", "ref": "SO-0031",
                          "actor": "AI", "basis": "x", "at": "2026-09-01 02:00:00+00"})
    assert db.materialize(State(tables)) is None


def test_background_rows_are_the_th0_branch(golden) -> None:
    background, case = split_background(golden.tables)
    assert {r["ma_apr"] for r in background["apr"]} == {"APR-TH-01"}
    assert "cand" not in case and len(background["cand"]) == 3


def test_the_digest_changes_with_any_file(tmp_path) -> None:
    copy = tmp_path / "pack"
    shutil.copytree(BASE_PACK, copy)
    before = pack_digest(copy)
    (copy / "kb" / "KB-0001.md").write_text("changed", encoding="utf-8")
    assert pack_digest(copy) != before
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_db.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.db'`.

- [ ] **Step 3: Implement**

`specdx/observe/__init__.py`: empty.

`specdx/observe/state.py`:
```python
"""Read a run's end state out of the database, as text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import psycopg
from tonomyx.canon.pack import CanonPack
from tonomyx.db.bootstrap import OPERATIONAL_TABLES
from tonomyx.db.repo import Repository

from specdx.types import State

# What fn_reconcile "returns" when it raises (two receipts make its scalar subquery fail).
ERROR = "ERROR"


def reconcile_value(conn: psycopg.Connection[dict[str, Any]], po: str) -> str:
    """The canon's own fn_reconcile verdict for one order, or ERROR if it cannot answer."""
    try:
        with conn.transaction():
            row = conn.execute("SELECT fn_reconcile(%s) AS r", (po,)).fetchone()
    except psycopg.Error:
        return ERROR
    return "" if row is None or row["r"] is None else str(row["r"])


def capture_state(
    conn: psycopg.Connection[dict[str, Any]],
    pack: CanonPack,
    steps: Mapping[str, str] | None = None,
) -> State:
    """Every operational and audit table as text, plus fn_reconcile for every order."""
    repo = Repository(conn, pack)
    tables = {t: repo.as_text(t) for t in OPERATIONAL_TABLES}
    orders = {r["ma_po"] for r in tables["po"]} | {r["po"] for r in tables["vinv"] if r["po"]}
    derived = {f"fn_reconcile:{po}": reconcile_value(conn, po) for po in sorted(orders)}
    return State(tables=tables, derived=derived, steps=dict(steps or {}))
```

`specdx/db.py`:
```python
"""One PostgreSQL, loaded with whichever specification is under test.

Every run starts from master data plus background rows, the operational rows that do
not belong to CASE-0031 (spec §11, item 2). seed_operational.sql is never loaded: the
golden end state is read from the pack's CSVs instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from tonomyx.canon.invariants import is_blank
from tonomyx.canon.pack import CanonPack
from tonomyx.config import Settings
from tonomyx.db.bootstrap import OPERATIONAL_TABLES, drop_everything, resync_sequences

from specdx.config import DATABASE_URL, settings_for
from specdx.observe.state import capture_state
from specdx.types import State

MASTER_LOAD_ORDER: tuple[str, ...] = ("ddl.sql", "guardrails.sql", "seed_master.sql")
BACKGROUND_TABLES: tuple[str, ...] = ("cand", "att", "esg")
_CAST = {
    "text": "text", "int": "int", "bigint": "bigint", "numeric": "numeric",
    "bool": "boolean", "date": "date", "timestamp": "timestamptz",
}
_MARKER = "specdx_marker"

Tables = Mapping[str, Sequence[Mapping[str, str]]]


def pack_digest(pack: Path) -> str:
    """A hash of every file in the pack: any edit makes it a different specification."""
    h = hashlib.sha256()
    for path in sorted(p for p in pack.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(pack)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def golden_tables(pack: CanonPack) -> dict[str, list[dict[str, str]]]:
    """The canon's end state for the anchor scenario, straight from its CSVs."""
    return {t: [dict(r) for r in pack.tables.get(t, [])] for t in OPERATIONAL_TABLES}


def is_background(table: str, row: Mapping[str, str]) -> bool:
    return table in BACKGROUND_TABLES or (table == "apr" and is_blank(row.get("case_id")))


def split_background(tables: Tables) -> tuple[dict[str, list[dict[str, str]]], ...]:
    """Split rows into (background, case) by the rule in is_background."""
    background: dict[str, list[dict[str, str]]] = {}
    case: dict[str, list[dict[str, str]]] = {}
    for table, rows in tables.items():
        for row in rows:
            target = background if is_background(table, row) else case
            target.setdefault(table, []).append(dict(row))
    return background, case


def write_rows(conn: psycopg.Connection[dict[str, Any]], pack: CanonPack, tables: Tables) -> None:
    """Insert text rows in foreign-key order, casting each cell to its schema type."""
    for table in OPERATIONAL_TABLES:
        columns = [c for c in pack.columns[table] if c.type != "serial"]
        for row in tables.get(table, []):
            present = [c for c in columns if c.name in row]
            query = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({vals})").format(
                t=sql.Identifier(table),
                cols=sql.SQL(", ").join(sql.Identifier(c.name) for c in present),
                vals=sql.SQL(", ").join(sql.SQL("%s::" + _CAST[c.type]) for c in present),
            )
            conn.execute(query, [None if is_blank(row[c.name]) else row[c.name] for c in present])


def _marker(conn: psycopg.Connection[dict[str, Any]]) -> str | None:
    try:
        row = conn.execute(f"SELECT digest FROM {_MARKER}").fetchone()
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return None
    return None if row is None else str(row["digest"])


class PackDatabase:
    """The SpecDx database and the specification currently loaded in it."""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.url = database_url
        self._digest: str | None = None
        self._settings: Settings | None = None
        self._pack: CanonPack | None = None
        self._background: dict[str, list[dict[str, str]]] = {}

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            raise RuntimeError("no specification loaded: call use() first")
        return self._settings

    @property
    def pack(self) -> CanonPack:
        if self._pack is None:
            raise RuntimeError("no specification loaded: call use() first")
        return self._pack

    def use(self, pack_dir: Path) -> Settings:
        """Load a specification's schema, guardrails and master data, unless already loaded."""
        pack_dir = pack_dir.resolve()
        digest = pack_digest(pack_dir)
        if digest != self._digest:
            with psycopg.connect(self.url, row_factory=dict_row) as conn:
                if _marker(conn) != digest:
                    drop_everything(conn)
                    for name in MASTER_LOAD_ORDER:
                        path = pack_dir / "targets" / "postgres" / name
                        conn.execute(path.read_text(encoding="utf-8"))
                    resync_sequences(conn)
                    conn.execute(f"CREATE TABLE {_MARKER} (digest text)")
                    conn.execute(f"INSERT INTO {_MARKER} VALUES (%s)", (digest,))
                    conn.commit()
            self._digest = digest
            self._settings = settings_for(pack_dir, self.url)
            self._pack = CanonPack(pack_dir)
            self._background, _ = split_background(golden_tables(self._pack))
        return self.settings

    @contextmanager
    def isolated(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """A connection on the run's starting state. Everything is rolled back afterwards."""
        conn = psycopg.connect(self.url, row_factory=dict_row)
        try:
            names = sql.SQL(", ").join(sql.Identifier(t) for t in OPERATIONAL_TABLES)
            conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(names))
            write_rows(conn, self.pack, self._background)
            yield conn
        finally:
            conn.rollback()
            conn.close()

    def materialize(self, state: State) -> State | None:
        """Write a state's case rows into a fresh run and read it back, or None if refused."""
        _, case = split_background(state.tables)
        with self.isolated() as conn:
            try:
                with conn.transaction():
                    write_rows(conn, self.pack, case)
            except psycopg.Error:
                return None
            return capture_state(conn, self.pack, state.steps)

    def golden_state(self) -> State:
        """The canon's end state, materialised so that derived values are real."""
        state = self.materialize(State(golden_tables(self.pack)))
        if state is None:
            raise RuntimeError("the database refuses the canon's own golden state")
        return state
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_db.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/db.py specdx/observe tests/conftest.py tests/test_db.py
git commit -m "feat: PackDatabase with isolated runs, background rows and state capture"
```

---

### Task 4: Column classes, projections and field differences

**Files:**
- Create: `specdx/spec/__init__.py`, `specdx/spec/columns.py`
- Test: `tests/spec/__init__.py`, `tests/spec/test_columns.py`

**Interfaces:**
- Consumes: `State` (Task 2).
- Produces:
  - `ColumnClass` (LOCKED / BLANK / IGNORED), `FREE_TEXT`, `FILLED`, and `Projection` (a hashable nested tuple).
  - `ColumnModel.of(pack)`, with fields `.classes` and `.keys`.
  - `.project(tables, entries) -> Projection`. Each entry is `"table"` or `"table=KEY"`; for `evt`, `"evt=REF"` filters by ref.
  - `.diff(a, b) -> frozenset[str]`. Field ids are `"table.column"` or `"evt:KIND:REF"`.

- [ ] **Step 1: Write the failing tests**

`tests/spec/__init__.py`: empty.

`tests/spec/test_columns.py`:
```python
from specdx.spec.columns import FILLED, ColumnClass, ColumnModel

BLANK = {
    ("soline", "don_gia"), ("soline", "chiet_khau"), ("po", "don_gia"), ("inv", "so_tien"),
    ("vinv", "so_tien"), ("pay", "so_tien"), ("apr", "gia_tri"), ("cand", "muc_offer"),
    ("cand", "khung_max"),
}
IGNORED = {("apr", "noi_dung"), ("apr", "created_at"), ("evt", "basis"), ("evt", "at"),
           ("evt", "id")}
CASE_TABLES = ("cases", "so", "soline", "inv", "po", "grn", "insp", "ncr", "ship", "vinv",
               "pay", "cand", "apr", "evt")


def test_case_table_columns_are_classified_as_pinned(pack) -> None:
    model = ColumnModel.of(pack)
    for (table, column), cls in model.classes.items():
        if table not in CASE_TABLES:
            continue
        if (table, column) in BLANK:
            assert cls is ColumnClass.BLANK, (table, column)
        elif (table, column) in IGNORED:
            assert cls is ColumnClass.IGNORED, (table, column)
        else:
            assert cls is ColumnClass.LOCKED, (table, column)


def _tables(**changes):
    base = {
        "so": [{"ma_so": "SO-0031", "ma_kh": "CUST-0007", "case_id": "CASE-0031",
                "trang_thai": "Chờ duyệt"}],
        "pay": [{"ma_pay": "PAY-0031", "ma_vinv": "VINV-0015", "so_tien": "",
                 "trang_thai": "Chờ duyệt"}],
        "evt": [{"id": "1", "case_id": "CASE-0031", "kind": "FLAGGED", "ref": "ITEM-0012",
                 "actor": "AI", "basis": "a", "at": "t1"}],
    }
    for table, row in changes.items():
        base[table] = [row]
    return base


def test_free_text_and_timestamps_do_not_separate_behaviours(pack) -> None:
    model = ColumnModel.of(pack)
    other = _tables(evt={"id": "9", "case_id": "CASE-0031", "kind": "FLAGGED",
                         "ref": "ITEM-0012", "actor": "AI", "basis": "zzz", "at": "t9"})
    entries = ["so", "pay", "evt=ITEM-0012"]
    assert model.project(_tables(), entries) == model.project(other, entries)


def test_a_locked_value_separates_behaviours_and_is_named(pack) -> None:
    model = ColumnModel.of(pack)
    other = _tables(so={"ma_so": "SO-0031", "ma_kh": "CUST-0007", "case_id": "CASE-0031",
                        "trang_thai": "Đã duyệt"})
    a = model.project(_tables(), ["so"])
    b = model.project(other, ["so"])
    assert a != b
    assert model.diff(a, b) == frozenset({"so.trang_thai"})


def test_a_filled_blank_column_is_one_value_whatever_it_holds(pack) -> None:
    model = ColumnModel.of(pack)
    one = _tables(pay={"ma_pay": "PAY-0031", "ma_vinv": "VINV-0015", "so_tien": "0",
                       "trang_thai": "Chờ duyệt"})
    two = _tables(pay={"ma_pay": "PAY-0031", "ma_vinv": "VINV-0015", "so_tien": "999",
                       "trang_thai": "Chờ duyệt"})
    assert model.project(one, ["pay"]) == model.project(two, ["pay"])
    assert ("so_tien", FILLED) in model.project(one, ["pay"])[0][1][0][1]
    assert model.diff(model.project(_tables(), ["pay"]),
                      model.project(one, ["pay"])) == frozenset({"pay.so_tien"})


def test_a_missing_event_is_named_by_kind_and_ref(pack) -> None:
    model = ColumnModel.of(pack)
    without = _tables()
    without["evt"] = []
    a = model.project(_tables(), ["evt=ITEM-0012"])
    b = model.project(without, ["evt=ITEM-0012"])
    assert model.diff(a, b) == frozenset({"evt:FLAGGED:ITEM-0012"})


def test_a_keyed_entry_keeps_only_that_row(pack) -> None:
    model = ColumnModel.of(pack)
    tables = {"apr": [
        {"ma_apr": "APR-0102", "loai": "q", "trang_thai": "Đã duyệt"},
        {"ma_apr": "APR-0155", "loai": "p", "trang_thai": "Chờ duyệt"},
    ]}
    projection = model.project(tables, ["apr=APR-0102"])
    assert [key for key, _ in projection[0][1]] == ["APR-0102"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/spec/test_columns.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.spec'`.

- [ ] **Step 3: Implement**

`specdx/spec/__init__.py`: empty.

`specdx/spec/columns.py`:
```python
"""Spec-induced equivalence ≡_S (spec §2.2): which columns separate two behaviours."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from tonomyx.canon.invariants import is_blank
from tonomyx.canon.pack import CanonPack


class ColumnClass(StrEnum):
    LOCKED = "locked"
    BLANK = "blank"
    IGNORED = "ignored"


# Free text, as TonomyX's own golden comparison excludes it.
FREE_TEXT: frozenset[tuple[str, str]] = frozenset({("apr", "noi_dung"), ("evt", "basis")})
_IGNORED_TYPES = frozenset({"serial", "timestamp"})
# Any value in a blank column: the specification says only "this must stay empty".
FILLED = "<filled>"

Cells = tuple[tuple[str, str], ...]
Projection = tuple[tuple[str, tuple[tuple[str, Cells], ...]], ...]
Tables = Mapping[str, Sequence[Mapping[str, str]]]


def _parse(entries: Iterable[str]) -> dict[str, set[str] | None]:
    """'table' keeps every row; 'table=KEY' keeps only those keys (refs, for evt)."""
    wanted: dict[str, set[str] | None] = {}
    for entry in entries:
        table, _, key = entry.partition("=")
        if not key:
            wanted[table] = None
        elif wanted.get(table, set()) is not None:
            wanted.setdefault(table, set()).add(key)  # type: ignore[union-attr]
    return wanted


@dataclass(frozen=True)
class ColumnModel:
    classes: dict[tuple[str, str], ColumnClass]
    keys: dict[str, tuple[str, ...]]

    @classmethod
    def of(cls, pack: CanonPack) -> ColumnModel:
        blank = pack.non_canon_columns
        classes: dict[tuple[str, str], ColumnClass] = {}
        keys: dict[str, tuple[str, ...]] = {}
        for table, columns in pack.columns.items():
            keys[table] = tuple(c.name for c in columns if c.pk)
            for c in columns:
                if c.name in blank.get(table, frozenset()):
                    classes[(table, c.name)] = ColumnClass.BLANK
                elif c.type in _IGNORED_TYPES or (table, c.name) in FREE_TEXT:
                    classes[(table, c.name)] = ColumnClass.IGNORED
                else:
                    classes[(table, c.name)] = ColumnClass.LOCKED
        return cls(classes, keys)

    def _row(self, table: str, row: Mapping[str, str]) -> tuple[str, Cells]:
        if table == "evt":
            key = f"{row.get('kind', '')}:{row.get('ref', '')}"
            return key, (("actor", row.get("actor", "")), ("case_id", row.get("case_id", "")))
        key = ":".join(str(row.get(k, "")) for k in self.keys[table])
        cells: list[tuple[str, str]] = []
        for column in sorted(row):
            cls = self.classes.get((table, column))
            if cls is ColumnClass.LOCKED:
                cells.append((column, str(row[column])))
            elif cls is ColumnClass.BLANK:
                cells.append((column, "" if is_blank(row[column]) else FILLED))
        return key, tuple(cells)

    def project(self, tables: Tables, entries: Iterable[str]) -> Projection:
        out: list[tuple[str, tuple[tuple[str, Cells], ...]]] = []
        for table, keys in sorted(_parse(entries).items()):
            rows: list[tuple[str, Cells]] = []
            for row in tables.get(table, []):
                selector = row.get("ref", "") if table == "evt" else ":".join(
                    str(row.get(k, "")) for k in self.keys[table]
                )
                if keys is None or selector in keys:
                    rows.append(self._row(table, row))
            out.append((table, tuple(sorted(rows))))
        return tuple(out)

    def diff(self, a: Projection, b: Projection) -> frozenset[str]:
        """The fields on which two projections disagree."""
        fields: set[str] = set()
        tables_a, tables_b = dict(a), dict(b)
        for table in set(tables_a) | set(tables_b):
            rows_a = Counter(tables_a.get(table, ()))
            rows_b = Counter(tables_b.get(table, ()))
            changed = (rows_a - rows_b) + (rows_b - rows_a)
            if table == "evt":
                fields |= {f"evt:{key}" for key, _ in changed}
                continue
            by_key_a = {k: dict(c) for k, c in tables_a.get(table, ())}
            by_key_b = {k: dict(c) for k, c in tables_b.get(table, ())}
            for key, _ in changed:
                ca, cb = by_key_a.get(key, {}), by_key_b.get(key, {})
                for column in set(ca) | set(cb):
                    if ca.get(column) != cb.get(column):
                        fields.add(f"{table}.{column}")
        return frozenset(fields)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/spec/test_columns.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/spec tests/spec
git commit -m "feat: spec-induced equivalence with projections and field differences"
```

---
### Task 5: Clause lifting

**Files:**
- Create: `specdx/spec/lifting.py`
- Test: `tests/spec/test_lifting.py`

**Interfaces:**
- Consumes: `Scenario`, `PARAMETERS`, `STRUCTURAL` (Task 2).
- Produces:
  - `column_type(pack, table, column) -> str`.
  - `lift_value(value: str, column_type: str, anchor: Scenario) -> str | None`, returning a placeholder such as `"{ordered}"`.
  - `lift_invariant(inv: dict, anchor: Scenario, pack) -> tuple[dict, bool]`.
  - `instantiate(obj, scenario) -> obj`, which replaces whole-string placeholders.
  - `ExpectationRule(id, key, scope, anchor, eq=None, members=())` and `lift_expectations(canon: Mapping[str, Scenario], expects: Mapping[str, Mapping[str, object]]) -> list[ExpectationRule]`.

- [ ] **Step 1: Write the failing tests**

`tests/spec/test_lifting.py`:
```python
from specdx.scenarios.space import canon_scenarios
from specdx.spec.lifting import (
    ExpectationRule,
    instantiate,
    lift_expectations,
    lift_invariant,
    lift_value,
)
from specdx.types import Scenario

ANCHOR = Scenario("grn12", 15, 12, 0)


def test_only_unique_integer_literals_lift() -> None:
    assert lift_value("15", "int", ANCHOR) == "{ordered}"
    assert lift_value("12", "int", ANCHOR) == "{received}"
    assert lift_value("20", "int", ANCHOR) is None
    assert lift_value("15", "text", ANCHOR) is None
    assert lift_value("Đã duyệt", "text", ANCHOR) is None


def test_exactly_two_canon_invariants_lift(pack) -> None:
    lifted = {}
    for inv in pack.invariants:
        new, ok = lift_invariant(inv, ANCHOR, pack)
        if ok:
            lifted[inv["id"]] = new["assert"]["eq"]
    assert lifted == {"po-15": "{ordered}", "grn-12": "{received}"}


def test_lifting_never_mutates_the_pack(pack) -> None:
    before = [dict(i["assert"]) for i in pack.invariants]
    for inv in pack.invariants:
        lift_invariant(inv, ANCHOR, pack)
    assert [dict(i["assert"]) for i in pack.invariants] == before


def test_instantiate_fills_placeholders_recursively() -> None:
    s = Scenario("x", 16, 14, 4)
    assert instantiate({"a": "{ordered}", "b": ["{received}", "x"], "c": 3}, s) == {
        "a": "16", "b": ["14", "x"], "c": 3
    }


def test_expectations_lift_per_the_rules_in_spec_section_11(pack) -> None:
    canon = canon_scenarios(pack)
    expects = {name: dict(pack.scenarios[name]["expect"]) for name in canon}
    rules = {r.id: r for r in lift_expectations(canon, expects)}
    assert rules["expect:pay_units"] == ExpectationRule(
        "expect:pay_units", "pay_units", "lifted", None, eq="{received}"
    )
    assert rules["expect:three_way:in"].members == ("Cần kiểm", "Khớp")
    assert rules["expect:apr_0102:in"].members == ("normal", "over_cap")
    assert rules["expect:open_units:grn12"].eq == "3"
    assert rules["expect:three_way:happy"].anchor == "happy"
    assert len(rules) == 12
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/spec/test_lifting.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.spec.lifting'`.

- [ ] **Step 3: Implement**

`specdx/spec/lifting.py`:
```python
"""Clause lifting (spec §2.5 and §11, item 1).

Lifting is deliberately conservative: only integer cells whose value equals exactly one
non-structural anchor parameter are rewritten. Everything else stays tied to the scenario
it was written for.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tonomyx.canon.pack import CanonPack

from specdx.types import PARAMETERS, STRUCTURAL, Scenario

_INTEGER_TYPES = frozenset({"int", "bigint"})
_PLACEHOLDER = re.compile(r"^\{([a-z_]+)\}$")


def column_type(pack: CanonPack, table: str, column: str) -> str:
    return next(c.type for c in pack.columns[table] if c.name == column)


def lift_value(value: str, column_type: str, anchor: Scenario) -> str | None:
    if column_type not in _INTEGER_TYPES:
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    hits = [p for p, v in anchor.params().items() if p not in STRUCTURAL and v == number]
    return "{" + hits[0] + "}" if len(hits) == 1 else None


def lift_invariant(
    inv: Mapping[str, Any], anchor: Scenario, pack: CanonPack
) -> tuple[dict[str, Any], bool]:
    """A row-level eq invariant on an integer column, lifted if its literal binds."""
    spec = inv["assert"]
    if "where" in inv and "col" in spec and "eq" in spec:
        placeholder = lift_value(
            str(spec["eq"]), column_type(pack, inv["table"], spec["col"]), anchor
        )
        if placeholder is not None:
            lifted = copy.deepcopy(dict(inv))
            lifted["assert"]["eq"] = placeholder
            return lifted, True
    return copy.deepcopy(dict(inv)), False


def instantiate(obj: Any, scenario: Scenario) -> Any:
    """Replace every whole-string placeholder with the scenario's value, as text."""
    if isinstance(obj, str):
        match = _PLACEHOLDER.match(obj)
        return str(scenario.params()[match.group(1)]) if match else obj
    if isinstance(obj, dict):
        return {k: instantiate(v, scenario) for k, v in obj.items()}
    if isinstance(obj, list):
        return [instantiate(v, scenario) for v in obj]
    return obj


@dataclass(frozen=True)
class ExpectationRule:
    id: str
    key: str
    scope: str  # "lifted" | "anchored" | "all"
    anchor: str | None
    eq: str | None = None
    members: tuple[str, ...] = ()


def lift_expectations(
    canon: Mapping[str, Scenario], expects: Mapping[str, Mapping[str, object]]
) -> list[ExpectationRule]:
    """Numeric keys equal to one parameter in every scenario lift; categorical keys
    become a membership rule for all scenarios beside their exact per-scenario rules."""
    rules: list[ExpectationRule] = []
    keys = sorted({k for e in expects.values() for k in e})
    for key in keys:
        values = {name: expects[name][key] for name in sorted(canon) if key in expects[name]}
        exact = [
            ExpectationRule(f"expect:{key}:{name}", key, "anchored", name, eq=str(v))
            for name, v in values.items()
        ]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values.values()):
            params = [
                p for p in PARAMETERS
                if p not in STRUCTURAL
                and all(canon[name].params()[p] == v for name, v in values.items())
            ]
            if len(params) == 1:
                rules.append(
                    ExpectationRule(f"expect:{key}", key, "lifted", None, eq="{" + params[0] + "}")
                )
            else:
                rules += exact
            continue
        rules += exact
        distinct = tuple(sorted({str(v) for v in values.values()}))
        if len(distinct) > 1:
            rules.append(ExpectationRule(f"expect:{key}:in", key, "all", None, members=distinct))
        else:
            rules.append(ExpectationRule(f"expect:{key}:all", key, "all", None, eq=distinct[0]))
    return rules
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/spec/test_lifting.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/spec/lifting.py tests/spec/test_lifting.py
git commit -m "feat: conservative clause lifting and expectation rules"
```

---

### Task 6: The work order and the clause set

**Files:**
- Create: `specdx/orchestrate/__init__.py`, `specdx/orchestrate/work_orders.json`, `specdx/orchestrate/work_orders.py`, `specdx/spec/clauses.py`
- Test: `tests/orchestrate/__init__.py`, `tests/orchestrate/test_work_orders.py`, `tests/spec/test_clauses.py`

**Interfaces:**
- Consumes:
  - `ColumnModel`, `ColumnClass` (Task 4);
  - `lift_invariant`, `lift_value`, `column_type`, `instantiate`, `lift_expectations` (Task 5);
  - `golden_tables` (Task 3);
  - `canon_scenarios`, `ANCHOR` (Task 2).
- Produces:
  - `WorkStep(id, kind, actor, intent, slice, inputs)`, with `.tables -> frozenset[str]`.
  - `load_work_orders(path=WORK_ORDERS) -> tuple[WorkStep, ...]` and `render_inputs(step, scenario) -> dict[str, str]`.
  - `Clause(id, kind, scope, anchor, tables, description, payload)`.
  - `extract_clauses(pack, canon, steps, columns, anchor=ANCHOR) -> list[Clause]`.
  - `master_tables(pack)`, `derive_outcome(state) -> dict[str, str]` and `evaluate(clause, state, scenario, master) -> bool | None`.
  - `ClauseSet(clauses, master, canon, anchor=ANCHOR)`, with `.from_pack(pack, steps)`, `.applies(clause, scenario)`, `.violations(state, scenario) -> frozenset[str]`, `[id]`, iteration, `.ids`.

- [ ] **Step 1: Write the work order data**

`specdx/orchestrate/__init__.py`: empty.

`specdx/orchestrate/work_orders.json`:
```json
{
  "_note": "The CASE-0031 work order (spec §11, items 3 and 7). kind is de (a digital employee via DeRuntime.run), persona (a person's interpreted consequences via DeRuntime.human_context) or env (recorded by the environment). slice lists the rows a step is answerable for: 'table', 'table=KEY', or 'evt=REF'. Input values may hold {ordered}, {received}, {discount_pct}, {grn_count} or {deliveries}.",
  "steps": [
    {"id": "DE-SM-02:draft_quote", "kind": "de", "actor": "DE-SM-02", "intent": "draft_quote",
     "slice": ["so", "soline", "apr=APR-0102", "evt=SO-0031"],
     "inputs": {"ma_so": "SO-0031", "ma_kh": "CUST-0007", "ma_apr": "APR-0102",
                "lines": "ITEM-0005:20,ITEM-0012:{ordered}", "discount_pct": "{discount_pct}"}},
    {"id": "DE-OP-01:check_availability", "kind": "de", "actor": "DE-OP-01", "intent": "check_availability",
     "slice": ["evt=ITEM-0005", "evt=ITEM-0012"],
     "inputs": {"ma_so": "SO-0031"}},
    {"id": "P2:decide_quote", "kind": "persona", "actor": "P2", "intent": "decide_quote",
     "slice": ["apr=APR-0102", "so", "evt=APR-0102"],
     "inputs": {"ma_apr": "APR-0102", "quyet_dinh": "Đã duyệt"}},
    {"id": "DE-FN-04:draft_purchase_order", "kind": "de", "actor": "DE-FN-04", "intent": "draft_purchase_order",
     "slice": ["po", "apr=APR-0155", "evt=PO-0015"],
     "inputs": {"ma_so": "SO-0031", "ma_po": "PO-0015", "ma_ncc": "VEND-0003", "ma_apr": "APR-0155"}},
    {"id": "P6:decide_po", "kind": "persona", "actor": "P6", "intent": "decide_po",
     "slice": ["apr=APR-0155", "po", "evt=APR-0155"],
     "inputs": {"ma_apr": "APR-0155", "quyet_dinh": "Đã duyệt"}},
    {"id": "DE-OP-02:record_goods_receipt", "kind": "de", "actor": "DE-OP-02", "intent": "record_goods_receipt",
     "slice": ["grn", "evt=GRN-0015", "evt=GRN-0016"],
     "inputs": {"ma_po": "PO-0015", "received": "{received}", "deliveries": "{deliveries}"}},
    {"id": "ENV:inspection", "kind": "env", "actor": "P4", "intent": "inspection",
     "slice": ["insp", "ncr", "evt=NCR-0007"], "inputs": {}},
    {"id": "ENV:shipment", "kind": "env", "actor": "P4", "intent": "shipment",
     "slice": ["ship", "evt=SHIP-0031"], "inputs": {}},
    {"id": "DE-FN-01:draft_sales_invoice", "kind": "de", "actor": "DE-FN-01", "intent": "draft_sales_invoice",
     "slice": ["inv", "evt=INV-0031"],
     "inputs": {"ma_inv": "INV-0031", "ma_so": "SO-0031", "ma_kh": "CUST-0007"}},
    {"id": "DE-FN-02:reconcile", "kind": "de", "actor": "DE-FN-02", "intent": "reconcile",
     "slice": ["vinv", "evt=VINV-0015"],
     "inputs": {"ma_po": "PO-0015", "ma_vinv": "VINV-0015"}},
    {"id": "DE-FN-03:draft_payment", "kind": "de", "actor": "DE-FN-03", "intent": "draft_payment",
     "slice": ["pay", "apr=APR-0210", "evt=PAY-0031"],
     "inputs": {"ma_pay": "PAY-0031", "ma_vinv": "VINV-0015", "ma_apr": "APR-0210", "pay_units": "{received}"}}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

`tests/orchestrate/__init__.py`: empty.

`tests/orchestrate/test_work_orders.py`:
```python
from specdx.orchestrate.work_orders import load_work_orders, render_inputs
from specdx.types import Scenario


def test_the_work_order_has_eleven_steps_in_canon_order() -> None:
    steps = load_work_orders()
    assert [s.id for s in steps][:3] == [
        "DE-SM-02:draft_quote", "DE-OP-01:check_availability", "P2:decide_quote"
    ]
    assert len(steps) == 11
    assert {s.kind for s in steps} == {"de", "persona", "env"}


def test_inputs_are_rendered_from_the_scenario() -> None:
    steps = {s.id: s for s in load_work_orders()}
    s = Scenario("x", 16, 13, 4, grn_count=2)
    assert render_inputs(steps["DE-SM-02:draft_quote"], s)["lines"] == "ITEM-0005:20,ITEM-0012:16"
    assert render_inputs(steps["DE-SM-02:draft_quote"], s)["discount_pct"] == "4"
    receipt = render_inputs(steps["DE-OP-02:record_goods_receipt"], s)
    assert receipt == {"ma_po": "PO-0015", "received": "13",
                       "deliveries": "GRN-0015:7,GRN-0016:6"}


def test_slice_tables_drop_the_keys() -> None:
    steps = {s.id: s for s in load_work_orders()}
    assert steps["DE-SM-02:draft_quote"].tables == frozenset({"so", "soline", "apr", "evt"})
```

`tests/spec/test_clauses.py`:
```python
import pytest

from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet, derive_outcome
from specdx.types import Scenario, State

STEPS = load_work_orders()


@pytest.fixture(scope="module")
def clauses(pack) -> ClauseSet:
    return ClauseSet.from_pack(pack, STEPS)


def _done(state: State) -> State:
    return State(state.tables, state.derived, {s.id: "done" for s in STEPS})


def test_clause_families_and_counts(clauses) -> None:
    kinds = [c.kind for c in clauses]
    assert kinds.count("invariant") == 42
    assert kinds.count("expectation") == 12
    assert kinds.count("step") == 11
    assert kinds.count("golden_event") == 10
    assert {"reconcile:consistency", "guard:pay", "vocab:evt", "blank:pay.so_tien"} <= clauses.ids
    assert len(clauses.ids) == len(list(clauses))


@pytest.mark.db
def test_the_canon_end_state_violates_nothing_on_its_own_scenario(clauses, golden) -> None:
    grn12 = Scenario("grn12", 15, 12, 0)
    assert clauses.violations(_done(golden), grn12) == frozenset()


@pytest.mark.db
def test_the_canon_end_state_is_wrong_for_happy(clauses, golden) -> None:
    happy = Scenario("happy", 15, 15, 0)
    v = clauses.violations(_done(golden), happy)
    assert {"expect:three_way:happy", "expect:pay_units", "golden:grn:GRN-0015:sl_nhan",
            "inv:grn-12"} <= v
    assert "expect:three_way:grn12" not in v


@pytest.mark.db
def test_a_changed_status_violates_its_golden_clause(clauses, golden) -> None:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["so"][0]["trang_thai"] = "Đã duyệt"
    v = clauses.violations(_done(State(tables, golden.derived)), Scenario("grn12", 15, 12, 0))
    assert v == frozenset({"golden:so:SO-0031:trang_thai"})


def test_lifted_clauses_do_not_apply_to_two_receipts(clauses) -> None:
    po = clauses["golden:po:PO-0015:sl_dat"]
    assert po.scope == "lifted"
    assert clauses.applies(po, Scenario("x", 16, 13, 0))
    assert not clauses.applies(po, Scenario("x", 16, 13, 0, grn_count=2))


def test_anchored_clauses_follow_parameters_not_names(clauses) -> None:
    rule = clauses["expect:three_way:happy"]
    assert clauses.applies(rule, Scenario("o15-r15-d0-g1", 15, 15, 0))
    assert not clauses.applies(rule, Scenario("grn12", 15, 12, 0))


def test_the_illustrative_flag_does_not_count_as_over_cap() -> None:
    state = State({"evt": [{"kind": "FLAGGED", "ref": "SO-0031", "actor": "AI",
                            "basis": "(Minh họa, không canon) x"}],
                   "grn": [], "po": [], "vinv": []})
    assert derive_outcome(state)["apr_0102"] == "normal"


def test_a_missing_step_is_not_applicable_and_an_error_violates(clauses) -> None:
    rule = clauses["step:DE-FN-02:reconcile"]
    grn12 = Scenario("grn12", 15, 12, 0)
    assert rule.id not in clauses.violations(State({}, {}, {}), grn12)
    assert rule.id in clauses.violations(State({}, {}, {rule.payload["step"]: "error"}), grn12)
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/orchestrate/test_work_orders.py tests/spec/test_clauses.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.orchestrate.work_orders'`.

- [ ] **Step 4: Implement the work order loader**

`specdx/orchestrate/work_orders.py`:
```python
"""The CASE-0031 work order, as data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from specdx.types import Scenario

WORK_ORDERS = Path(__file__).with_name("work_orders.json")
_WHOLE = re.compile(r"^\{(\w+)\}$")


@dataclass(frozen=True)
class WorkStep:
    id: str
    kind: str  # "de" | "persona" | "env"
    actor: str
    intent: str
    slice: tuple[str, ...]
    inputs: dict[str, str] = field(default_factory=dict)

    @property
    def tables(self) -> frozenset[str]:
        return frozenset(entry.partition("=")[0] for entry in self.slice)


def load_work_orders(path: Path = WORK_ORDERS) -> tuple[WorkStep, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        WorkStep(
            id=s["id"], kind=s["kind"], actor=s["actor"], intent=s["intent"],
            slice=tuple(s["slice"]), inputs=dict(s.get("inputs", {})),
        )
        for s in raw["steps"]
    )


def render_inputs(step: WorkStep, scenario: Scenario) -> dict[str, str]:
    values = {k: str(v) for k, v in scenario.params().items()}
    values["deliveries"] = ",".join(f"{grn}:{n}" for grn, n in scenario.deliveries())
    out: dict[str, str] = {}
    for key, template in step.inputs.items():
        whole = _WHOLE.match(template)
        out[key] = values[whole.group(1)] if whole else template.format(**values)
    return out
```

- [ ] **Step 5: Implement the clause set**

`specdx/spec/clauses.py`:
```python
"""The Check side C of an executable business specification (spec §2.1, §11)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tonomyx.canon.invariants import evaluate as evaluate_invariant
from tonomyx.canon.invariants import is_blank
from tonomyx.canon.pack import CanonPack
from tonomyx.db.bootstrap import OPERATIONAL_TABLES
from tonomyx.flow.handlers import PO, SO, VINV

from specdx.db import golden_tables
from specdx.orchestrate.work_orders import WorkStep
from specdx.scenarios.space import ANCHOR, canon_scenarios
from specdx.spec.columns import ColumnClass, ColumnModel
from specdx.spec.lifting import (
    column_type,
    instantiate,
    lift_expectations,
    lift_invariant,
    lift_value,
)
from specdx.types import STRUCTURAL, Scenario, State

GOLDEN_TABLES: tuple[str, ...] = (
    "cases", "so", "soline", "inv", "po", "grn", "insp", "ncr", "ship", "vinv", "pay", "apr",
)
# The canon labels one seeded event as illustrative (it belongs to discount8).
ILLUSTRATIVE_PREFIX = "(Minh họa"
DONE_STATUSES = frozenset({"done", "needs_gate"})
PAID = "Đã chi"
APPROVED = "Đã duyệt"
NEEDS_CHECK = "Cần kiểm"
_EXPECTATION_TABLES: dict[str, frozenset[str]] = {
    "apr_0102": frozenset({"evt"}),
    "three_way": frozenset({"vinv"}),
    "pay_units": frozenset({"grn"}),
    "open_units": frozenset({"po", "grn"}),
}
_TRUE = frozenset({"t", "true", "1"})
_FALSE = frozenset({"f", "false", "0"})


@dataclass(frozen=True)
class Clause:
    id: str
    kind: str
    scope: str  # "all" | "lifted" | "anchored"
    anchor: str | None
    tables: frozenset[str]
    description: str
    payload: dict[str, Any] = field(default_factory=dict)


def master_tables(pack: CanonPack) -> dict[str, list[dict[str, str]]]:
    return {t: rows for t, rows in pack.tables.items() if t not in OPERATIONAL_TABLES}


def _same(a: str, b: str) -> bool:
    la, lb = a.strip().lower(), b.strip().lower()
    if la in _TRUE | _FALSE and lb in _TRUE | _FALSE:
        return (la in _TRUE) == (lb in _TRUE)
    return a.strip() == b.strip()


def extract_clauses(
    pack: CanonPack,
    canon: Mapping[str, Scenario],
    steps: Sequence[WorkStep],
    columns: ColumnModel,
    anchor: str = ANCHOR,
) -> list[Clause]:
    anchor_s = canon[anchor]
    out: list[Clause] = []
    for inv in pack.invariants:
        lifted, ok = lift_invariant(inv, anchor_s, pack)
        out.append(Clause(
            f"inv:{inv['id']}", "invariant", "lifted" if ok else "anchored",
            None if ok else anchor, frozenset({inv["table"]}), str(inv["desc"]),
            {"invariant": lifted},
        ))
    gold = golden_tables(pack)
    for table in GOLDEN_TABLES:
        keys = columns.keys[table]
        for row in gold[table]:
            if table == "apr" and is_blank(row.get("case_id")):
                continue
            ident = ":".join(row[k] for k in keys)
            for col, value in row.items():
                if col in keys or is_blank(value):
                    continue
                if columns.classes[(table, col)] is not ColumnClass.LOCKED:
                    continue
                placeholder = lift_value(value, column_type(pack, table, col), anchor_s)
                out.append(Clause(
                    f"golden:{table}:{ident}:{col}", "golden",
                    "lifted" if placeholder else "anchored", None if placeholder else anchor,
                    frozenset({table}), f"{table} {ident}: {col} = {value}",
                    {"table": table, "key": {k: row[k] for k in keys}, "column": col,
                     "value": placeholder or value},
                ))
    for row in gold["evt"]:
        if row["basis"].startswith(ILLUSTRATIVE_PREFIX):
            continue
        out.append(Clause(
            f"event:{row['kind']}:{row['ref']}:{row['actor']}", "golden_event", "anchored",
            anchor, frozenset({"evt"}), f"sự kiện {row['kind']} {row['ref']} ({row['actor']})",
            {"event": [row["kind"], row["ref"], row["actor"]]},
        ))
    expects = {name: dict(pack.scenarios[name]["expect"]) for name in canon}
    for rule in lift_expectations(canon, expects):
        out.append(Clause(
            rule.id, "expectation", rule.scope, rule.anchor,
            _EXPECTATION_TABLES.get(rule.key, frozenset()), f"kỳ vọng {rule.key}",
            {"key": rule.key, "eq": rule.eq, "members": list(rule.members)},
        ))
    for table, cols in sorted(pack.non_canon_columns.items()):
        if table not in OPERATIONAL_TABLES:
            continue
        for col in sorted(cols):
            out.append(Clause(
                f"blank:{table}.{col}", "blank", "all", None, frozenset({table}),
                f"{table}.{col} để trống", {"table": table, "column": col},
            ))
    for step in steps:
        out.append(Clause(
            f"step:{step.id}", "step", "all", None, step.tables,
            f"bước {step.id} hoàn thành", {"step": step.id},
        ))
    out.append(Clause("reconcile:consistency", "reconcile", "all", None,
                      frozenset({"vinv", "po", "grn"}), "vinv.doi_soat = fn_reconcile(po)", {}))
    out.append(Clause("guard:pay", "guard_pay", "all", None, frozenset({"pay", "apr", "vinv"}),
                      "không chi khi chưa duyệt hoặc đối soát còn Cần kiểm", {}))
    out.append(Clause("vocab:evt", "vocabulary", "all", None, frozenset({"evt"}),
                      "evt.kind thuộc danh mục canon", {"kinds": sorted(pack.event_kinds)}))
    return out


def derive_outcome(state: State) -> dict[str, str]:
    """The scenarios.json vocabulary, read from an end state."""
    t = state.tables
    flagged = any(
        e["kind"] == "FLAGGED" and e["ref"] == SO
        and not e.get("basis", "").startswith(ILLUSTRATIVE_PREFIX)
        for e in t.get("evt", [])
    )
    vinv = next((r for r in t.get("vinv", []) if r["ma_vinv"] == VINV), None)
    received = sum(int(r["sl_nhan"]) for r in t.get("grn", [])
                   if r["po"] == PO and not is_blank(r["sl_nhan"]))
    po = next((r for r in t.get("po", []) if r["ma_po"] == PO), None)
    ordered = int(po["sl_dat"]) if po is not None and not is_blank(po["sl_dat"]) else 0
    return {
        "apr_0102": "over_cap" if flagged else "normal",
        "three_way": vinv["doi_soat"] if vinv is not None else "",
        "pay_units": str(received),
        "open_units": str(ordered - received),
    }


def evaluate(
    clause: Clause, state: State, scenario: Scenario,
    master: Mapping[str, list[dict[str, str]]],
) -> bool | None:
    """True or False on this state; None when the clause cannot be judged on it."""
    p = instantiate(clause.payload, scenario)
    t = state.tables
    if clause.kind == "invariant":
        return evaluate_invariant(p["invariant"], {**master, **t}).passed
    if clause.kind == "golden":
        rows = [r for r in t.get(p["table"], []) if all(r.get(k) == v for k, v in p["key"].items())]
        return bool(rows) and _same(str(rows[0].get(p["column"], "")), str(p["value"]))
    if clause.kind == "golden_event":
        kind, ref, actor = p["event"]
        return any(e["kind"] == kind and e["ref"] == ref and e["actor"] == actor
                   for e in t.get("evt", []))
    if clause.kind == "expectation":
        value = derive_outcome(state)[p["key"]]
        return value in p["members"] if p["members"] else value == p["eq"]
    if clause.kind == "blank":
        return all(is_blank(r.get(p["column"])) for r in t.get(p["table"], []))
    if clause.kind == "step":
        status = state.steps.get(p["step"])
        return None if status is None else status in DONE_STATUSES
    if clause.kind == "reconcile":
        rows = t.get("vinv", [])
        if not rows:
            return None
        return all(state.derived.get(f"fn_reconcile:{r['po']}") == r["doi_soat"] for r in rows)
    if clause.kind == "guard_pay":
        approved = {r["hoso_ref"] for r in t.get("apr", []) if r["trang_thai"] == APPROVED}
        checks = {r["ma_vinv"]: r["doi_soat"] for r in t.get("vinv", [])}
        return all(
            r["ma_pay"] in approved and checks.get(r["ma_vinv"]) != NEEDS_CHECK
            for r in t.get("pay", []) if r["trang_thai"] == PAID
        )
    if clause.kind == "vocabulary":
        return all(e["kind"] in p["kinds"] for e in t.get("evt", []))
    raise ValueError(f"unknown clause kind {clause.kind!r}")


class ClauseSet:
    """Every clause of one specification, with the rule for where each applies."""

    def __init__(
        self,
        clauses: Sequence[Clause],
        master: Mapping[str, list[dict[str, str]]],
        canon: Mapping[str, Scenario],
        anchor: str = ANCHOR,
    ) -> None:
        self._clauses = list(clauses)
        self._by_id = {c.id: c for c in self._clauses}
        self.master = master
        self.canon = dict(canon)
        self.anchor = canon[anchor]

    @classmethod
    def from_pack(cls, pack: CanonPack, steps: Sequence[WorkStep]) -> ClauseSet:
        canon = canon_scenarios(pack)
        clauses = extract_clauses(pack, canon, steps, ColumnModel.of(pack))
        return cls(clauses, master_tables(pack), canon)

    def __iter__(self) -> Iterator[Clause]:
        return iter(self._clauses)

    def __len__(self) -> int:
        return len(self._clauses)

    def __getitem__(self, clause_id: str) -> Clause:
        return self._by_id[clause_id]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def applies(self, clause: Clause, scenario: Scenario) -> bool:
        if clause.scope == "all":
            return True
        if clause.scope == "lifted":
            return all(scenario.params()[p] == self.anchor.params()[p] for p in STRUCTURAL)
        assert clause.anchor is not None
        return scenario.signature() == self.canon[clause.anchor].signature()

    def violations(self, state: State, scenario: Scenario) -> frozenset[str]:
        return frozenset(
            c.id for c in self._clauses
            if self.applies(c, scenario) and evaluate(c, state, scenario, self.master) is False
        )
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/orchestrate/test_work_orders.py tests/spec/test_clauses.py -q`
Expected: `11 passed`.

- [ ] **Step 7: Commit**

```bash
git add specdx/orchestrate specdx/spec/clauses.py tests/orchestrate tests/spec/test_clauses.py
git commit -m "feat: CASE-0031 work order and the clause set with lifting and applicability"
```

---

### Task 7: The Tell corpus

**Files:**
- Create: `specdx/spec/tell.py`, `specdx/variant.py`
- Test: `tests/spec/test_tell.py`

**Interfaces:**
- Consumes: `CanonPack`, `PDF_DIR`, `WORK_DIR` (Task 1), `pack_digest` (Task 3).
- Produces:
  - `Sentence(id, text)`, and `Chunk(id, source, sentences)` with `.text`.
  - `split_sentences(text)`, `chunk_document(doc, text, words=300)`, `pdf_chunks(pdf_dir, cache_dir) -> list[Chunk]` and `Bm25`.
  - `TellCorpus(chunks)`, with:
    - `.build(pack, pdf)` and `.digest()`;
    - `.sentence_ids(chunk_ids=None)`, `.sentence(id)` and `TellCorpus.chunk_of(id)`;
    - `.find(text) -> list[str]` and `.search(query, k=5) -> list[Chunk]`;
    - `.without(ids)`, `.replace(id, text)` and `.with_chunk(chunk)`;
    - `.to_json()` and `.from_json(text)`.
  - `SpecVariant(name, pack, corpus)`, with `.digest()`.

- [ ] **Step 1: Write the failing tests**

`tests/spec/test_tell.py`:
```python
import unicodedata

import pytest

from specdx.config import BASE_PACK, PDF_DIR
from specdx.spec.tell import Bm25, Chunk, Sentence, TellCorpus, pdf_chunks, split_sentences
from specdx.variant import SpecVariant


@pytest.fixture(scope="module")
def corpus(pack, tmp_path_factory) -> TellCorpus:
    return TellCorpus.build(pack, pdf_chunks(PDF_DIR, tmp_path_factory.mktemp("tell")))


def test_the_corpus_holds_every_source(corpus) -> None:
    sources = {c.source for c in corpus.chunks}
    assert sources == {"DEMO", "BA", "SAD", "KB", "CANON"}
    assert len([c for c in corpus.chunks if c.source == "BA"]) > 50


def test_sentences_split_on_semicolons_full_stops_and_newlines() -> None:
    assert split_sentences("a; b.\nc! d") == ["a;", "b.", "c!", "d"]


def test_the_discount_cap_sentence_is_found(corpus) -> None:
    ids = corpus.find("**mức tối đa 5%**.")
    assert ids and all(TellCorpus.chunk_of(i) == "KB:KB-0003" for i in ids)


def test_find_matches_across_unicode_normal_forms(corpus) -> None:
    nfd = unicodedata.normalize("NFD", "**mức tối đa 5%**.")
    assert corpus.find(nfd) == corpus.find("**mức tối đa 5%**.")


def test_search_ranks_the_pricing_article_for_a_discount_query(corpus) -> None:
    assert "KB:KB-0003" in [c.id for c in corpus.search("chiết khấu tối đa", k=5)]


def test_removing_a_sentence_keeps_other_ids_and_changes_the_digest(corpus) -> None:
    target = corpus.find("**mức tối đa 5%**.")[0]
    others = [i for i in corpus.sentence_ids(["KB:KB-0003"]) if i != target]
    smaller = corpus.without([target])
    assert target not in smaller.sentence_ids()
    assert all(smaller.sentence(i) == corpus.sentence(i) for i in others)
    assert smaller.digest() != corpus.digest()


def test_replace_and_with_chunk(corpus) -> None:
    target = corpus.find("**mức tối đa 5%**.")[0]
    assert corpus.replace(target, "**mức tối đa hợp lý**.").sentence(target) == (
        "**mức tối đa hợp lý**."
    )
    extra = Chunk("KB:KB-0009", "KB", (Sentence("KB:KB-0009:00", "x."),))
    assert "KB:KB-0009:00" in corpus.with_chunk(extra).sentence_ids()


def test_json_round_trip(corpus) -> None:
    assert TellCorpus.from_json(corpus.to_json()).digest() == corpus.digest()


def test_bm25_prefers_the_document_with_the_term() -> None:
    index = Bm25([["chiet", "khau"], ["giao", "hang"]])
    scores = index.scores(["chiet"])
    assert scores[0] > scores[1]


def test_a_variant_digest_covers_pack_and_corpus(corpus) -> None:
    a = SpecVariant("base", BASE_PACK, corpus)
    b = SpecVariant("base", BASE_PACK, corpus.without(corpus.sentence_ids()[:1]))
    assert a.digest() != b.digest()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/spec/test_tell.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.spec.tell'`.

- [ ] **Step 3: Implement**

`specdx/spec/tell.py`:
```python
"""The Tell side T (spec §2.1): requirement documents, KB articles and canon principles.

Sentence ids are stable: removing or replacing a sentence never renumbers the others,
so a localisation result stays meaningful across corpus edits.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tonomyx.canon.pack import CanonPack

CHUNK_WORDS = 300
PDF_DOCS = {"1.": "DEMO", "2.": "BA", "3.": "SAD"}
_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    sentences: tuple[Sentence, ...]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def _norm(text: str) -> str:
    """Exact-match normal form: composed Unicode, case-folded, single spaces."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _fold(text: str) -> str:
    """Search normal form: accents stripped, as TonomyX's search_kb does."""
    stripped = unicodedata.normalize("NFD", text)
    return "".join(c for c in stripped if unicodedata.category(c) != "Mn").casefold()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_fold(text))


def _make_chunk(chunk_id: str, source: str, texts: Iterable[str]) -> Chunk:
    return Chunk(
        chunk_id, source,
        tuple(Sentence(f"{chunk_id}:{i:02d}", t) for i, t in enumerate(texts)),
    )


def chunk_document(doc: str, text: str, words: int = CHUNK_WORDS) -> list[Chunk]:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > words:
            groups.append(current)
            current, count = [], 0
        current.append(para)
        count += n
    if current:
        groups.append(current)
    return [
        _make_chunk(f"{doc}:{i:03d}", doc, [s for para in group for s in split_sentences(para)])
        for i, group in enumerate(groups)
    ]


def pdf_chunks(pdf_dir: Path, cache_dir: Path) -> list[Chunk]:
    """Chunks of the three requirement PDFs, extracted once and cached by file hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[Chunk] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = PDF_DOCS[pdf.name[:2]]
        cached = cache_dir / f"{hashlib.sha256(pdf.read_bytes()).hexdigest()}.json"
        if cached.is_file():
            out += TellCorpus.from_json(cached.read_text(encoding="utf-8")).chunks
            continue
        text = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
        ).stdout
        chunks = chunk_document(doc, text)
        cached.write_text(TellCorpus(chunks).to_json(), encoding="utf-8")
        out += chunks
    return out


class Bm25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg = (sum(self.lengths) / len(docs)) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Sequence[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.lengths, strict=True):
            s = 0.0
            for term in query:
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                    s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


class TellCorpus:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks: tuple[Chunk, ...] = tuple(chunks)
        self._index: Bm25 | None = None
        self._sentences = {s.id: s.text for c in self.chunks for s in c.sentences}

    @classmethod
    def build(cls, pack: CanonPack, pdf: Sequence[Chunk]) -> TellCorpus:
        kb = [
            _make_chunk(f"KB:{d.kb_id}", "KB", split_sentences(f"{d.title}.\n{d.body}"))
            for d in pack.kb.values()
        ]
        principles = _make_chunk("PRINCIPLES", "CANON", [str(p) for p in pack.meta["nguyen_tac"]])
        return cls([*pdf, *kb, principles])

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def sentence_ids(self, chunk_ids: Iterable[str] | None = None) -> list[str]:
        wanted = None if chunk_ids is None else set(chunk_ids)
        return [s.id for c in self.chunks if wanted is None or c.id in wanted for s in c.sentences]

    def sentence(self, sentence_id: str) -> str:
        return self._sentences[sentence_id]

    @staticmethod
    def chunk_of(sentence_id: str) -> str:
        return sentence_id.rsplit(":", 1)[0]

    def find(self, text: str) -> list[str]:
        target = _norm(text)
        return [sid for sid, t in self._sentences.items() if _norm(t) == target]

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if self._index is None:
            self._index = Bm25([_tokens(c.text) for c in self.chunks])
        scores = self._index.scores(_tokens(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]

    def without(self, sentence_ids: Iterable[str]) -> TellCorpus:
        drop = set(sentence_ids)
        return TellCorpus([
            Chunk(c.id, c.source, tuple(s for s in c.sentences if s.id not in drop))
            for c in self.chunks
        ])

    def replace(self, sentence_id: str, text: str) -> TellCorpus:
        return TellCorpus([
            Chunk(c.id, c.source, tuple(
                Sentence(s.id, text) if s.id == sentence_id else s for s in c.sentences
            ))
            for c in self.chunks
        ])

    def with_chunk(self, chunk: Chunk) -> TellCorpus:
        return TellCorpus([*self.chunks, chunk])

    def to_json(self) -> str:
        return json.dumps(
            [{"id": c.id, "source": c.source, "sentences": [[s.id, s.text] for s in c.sentences]}
             for c in self.chunks],
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> TellCorpus:
        return cls([
            Chunk(c["id"], c["source"], tuple(Sentence(i, t) for i, t in c["sentences"]))
            for c in json.loads(text)
        ])
```

`specdx/variant.py`:
```python
"""A specification under test: a canon pack directory plus its Tell corpus."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from specdx.db import pack_digest
from specdx.spec.tell import TellCorpus


@dataclass(frozen=True)
class SpecVariant:
    name: str
    pack: Path
    corpus: TellCorpus

    def digest(self) -> str:
        return hashlib.sha256(
            (pack_digest(self.pack) + self.corpus.digest()).encode()
        ).hexdigest()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/spec/test_tell.py -q`
Expected: `10 passed`. If `test_the_corpus_holds_every_source` fails on the BA chunk count, print `len(...)` and check that `pdftotext` extracted text, not images. The BA PDF holds about 32k words, so more than 50 chunks of 300 words.

- [ ] **Step 5: Commit**

```bash
git add specdx/spec/tell.py specdx/variant.py tests/spec/test_tell.py
git commit -m "feat: Tell corpus from requirement PDFs, KB and canon principles, with BM25"
```

---
### Task 8: The parametric case runner and the reference flow

**Files:**
- Create: `specdx/orchestrate/flow.py`, `specdx/orchestrate/reference.py`
- Test: `tests/orchestrate/test_flow.py`

**Interfaces:**
- Consumes: `WorkStep`, `render_inputs` (Task 6); TonomyX `DeRuntime`, `open_case`, `_record_inspection`, `_record_shipment` and the scripted handlers.
- Produces:
  - `CaseRun(statuses, claims, refused)`.
  - `run_case(runtime, scenario, steps) -> CaseRun`. Persona steps run through `runtime.executor` with `runtime.human_context(actor)`. Environment steps reuse TonomyX's own recorders.
  - `reference_executor() -> ScriptedExecutor`: TonomyX's seven handlers plus `decide_quote` (P2) and `decide_po` (P6).

- [ ] **Step 1: Write the failing tests**

`tests/orchestrate/test_flow.py`:
```python
from datetime import UTC, datetime

import pytest
from tonomyx.contracts.clock import FixedClock
from tonomyx.contracts.envelope import ResultEnvelope, TaskEnvelope, denied
from tonomyx.de.runtime import build_runtime
from tonomyx.tools.registry import ToolContext

from specdx.observe.state import capture_state
from specdx.orchestrate.flow import run_case
from specdx.orchestrate.reference import reference_executor
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import canon_scenarios
from specdx.spec.clauses import GOLDEN_TABLES, ILLUSTRATIVE_PREFIX, ClauseSet
from specdx.spec.columns import ColumnModel

pytestmark = pytest.mark.db
STEPS = load_work_orders()
CLOCK = FixedClock(datetime(2026, 9, 1, 2, 0, tzinfo=UTC))


def _run(db, scenario, executor):
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        case = run_case(runtime, scenario, STEPS)
        return case, capture_state(conn, db.pack, case.statuses)


def _chain(state):
    events = sorted(state.tables["evt"], key=lambda e: int(e["id"]))
    return [(e["kind"], e["ref"], e["actor"]) for e in events
            if not e["basis"].startswith(ILLUSTRATIVE_PREFIX)]


@pytest.mark.parametrize("name", ["grn12", "happy", "discount8"])
def test_the_reference_flow_satisfies_every_clause(db, pack, name) -> None:
    scenario = canon_scenarios(pack)[name]
    case, state = _run(db, scenario, reference_executor())
    assert set(case.statuses.values()) <= {"done", "needs_gate"}
    assert ClauseSet.from_pack(pack, STEPS).violations(state, scenario) == frozenset()


def test_the_reference_flow_reproduces_the_golden_state(db, pack, golden) -> None:
    _, state = _run(db, canon_scenarios(pack)["grn12"], reference_executor())
    model = ColumnModel.of(pack)
    assert model.project(state.tables, GOLDEN_TABLES) == model.project(golden.tables, GOLDEN_TABLES)
    assert _chain(state) == _chain(golden)


class _Without:
    """The reference flow with one intent refused."""

    def __init__(self, intent: str) -> None:
        self.inner = reference_executor()
        self.intent = intent

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        if task.intent == self.intent:
            return denied(task, "removed for the test")
        return self.inner.run(task, ctx)


def test_a_failed_step_does_not_poison_later_steps(db, pack) -> None:
    case, state = _run(db, canon_scenarios(pack)["grn12"], _Without("record_goods_receipt"))
    assert case.statuses["DE-OP-02:record_goods_receipt"] == "denied"
    assert case.statuses["ENV:inspection"] == "error"
    assert case.statuses["DE-FN-03:draft_payment"] == "needs_gate"
    assert [r["ma_pay"] for r in state.tables["pay"]] == ["PAY-0031"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/orchestrate/test_flow.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.orchestrate.flow'`.

- [ ] **Step 3: Implement**

`specdx/orchestrate/flow.py`:
```python
"""Run CASE-0031 for any scenario through TonomyX's public runtime.

The vendored run_case takes two parameters and hard-codes the rest; this runner reads
the work order instead. Digital-employee steps go through DeRuntime.run, so every gate
applies. Persona steps go through the same executor with a human context. Environment
steps reuse TonomyX's own recorders, which write the canon's fixed facts.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope, TaskScope
from tonomyx.de.runtime import DeRuntime
from tonomyx.flow.case_0031 import CASE_ID, _record_inspection, _record_shipment, open_case

from specdx.orchestrate.work_orders import WorkStep, render_inputs
from specdx.types import Scenario


@dataclass
class CaseRun:
    statuses: dict[str, str] = field(default_factory=dict)
    claims: dict[str, str] = field(default_factory=dict)
    refused: list[tuple[str, str]] = field(default_factory=list)


def _environment(runtime: DeRuntime, step: WorkStep, scenario: Scenario) -> tuple[str, str]:
    try:
        with runtime.repo.conn.transaction():
            if step.intent == "inspection":
                _record_inspection(runtime, scenario.received)
            elif step.intent == "shipment":
                _record_shipment(runtime)
            else:
                raise ValueError(f"unknown environment step {step.intent!r}")
    except Exception as exc:  # noqa: BLE001 - a failed step is recorded, never swallowed
        return ResultStatus.ERROR.value, f"{type(exc).__name__}: {exc}"
    return ResultStatus.DONE.value, ""


def _persona(runtime: DeRuntime, step: WorkStep, inputs: dict[str, str]) -> ResultEnvelope:
    ctx = runtime.human_context(step.actor)
    task = TaskEnvelope(
        task_id=f"H-{uuid.uuid4().hex[:12]}",
        case_id=CASE_ID,
        de=step.actor,
        intent=step.intent,
        inputs=dict(inputs),
        scope=TaskScope(tools=sorted(ctx.token.tools), read=sorted(ctx.token.read)),
        idempotency_key=f"{step.actor}:{step.intent}:{CASE_ID}",
        deadline=runtime.clock.now() + timedelta(seconds=300),
    )
    try:
        with runtime.repo.conn.transaction():
            return runtime.executor.run(task, ctx)
    except Exception as exc:  # noqa: BLE001 - same contract as DeRuntime.run
        return ResultEnvelope(
            task_id=task.task_id, case_id=CASE_ID, status=ResultStatus.ERROR,
            next_hint=f"{type(exc).__name__}: {exc}",
        )


def run_case(runtime: DeRuntime, scenario: Scenario, steps: Sequence[WorkStep]) -> CaseRun:
    run = CaseRun()
    open_case(runtime.repo)
    for step in steps:
        if step.kind == "env":
            status, hint = _environment(runtime, step, scenario)
        else:
            inputs = render_inputs(step, scenario)
            envelope = (
                runtime.run(step.actor, step.intent, CASE_ID, dict(inputs))
                if step.kind == "de"
                else _persona(runtime, step, inputs)
            )
            status, hint = envelope.status.value, envelope.next_hint
            run.refused += [
                (step.id, str(call.get("name", "")))
                for call in envelope.usage.get("tool_calls", [])
                if call.get("outcome") != "ok"
            ]
        run.statuses[step.id] = status
        run.claims[step.id] = hint
    return run
```

`specdx/orchestrate/reference.py`:
```python
"""The scripted reference flow: TonomyX's handlers plus the two persona decisions.

Used only to prove the runner reproduces the canon. It is never an interpreter.
"""

from __future__ import annotations

from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope
from tonomyx.de.executor import ScriptedExecutor
from tonomyx.flow.handlers import PO, register_handlers
from tonomyx.tools.impl.decide_approval import decide_approval
from tonomyx.tools.impl.log_event import log_event
from tonomyx.tools.impl.update_status import update_status
from tonomyx.tools.registry import ToolContext


def _decide(ctx: ToolContext, task: TaskEnvelope, basis: str) -> dict[str, object]:
    apr = str(task.inputs["ma_apr"])
    decide_approval(
        ctx, apr=apr, decision=str(task.inputs["quyet_dinh"]), reason="",
        idempotency_key=f"decide:{apr}",
    )
    return log_event(
        ctx, case_id=task.case_id, kind="APPROVED", ref=apr, actor="Người", basis=basis
    )["row"]


def decide_quote(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """P2 approves the quote. The canon leaves SO-0031 pending afterwards."""
    event = _decide(ctx, task, "P2 duyệt báo giá")
    return ResultEnvelope(
        task_id=task.task_id, case_id=task.case_id, status=ResultStatus.DONE, events=[event]
    )


def decide_po(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """P6 approves the purchase order, which then advances."""
    event = _decide(ctx, task, f"P6 duyệt {PO} (maker-checker) trước khi phát hành")
    update_status(ctx, table="po", key={"ma_po": PO}, new_status="Đã duyệt")
    return ResultEnvelope(
        task_id=task.task_id, case_id=task.case_id, status=ResultStatus.DONE, events=[event]
    )


def reference_executor() -> ScriptedExecutor:
    executor = ScriptedExecutor()
    register_handlers(executor)
    executor.register("P2", "decide_quote", decide_quote)
    executor.register("P6", "decide_po", decide_po)
    return executor
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/orchestrate/test_flow.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/orchestrate/flow.py specdx/orchestrate/reference.py tests/orchestrate/test_flow.py
git commit -m "feat: parametric CASE-0031 runner that reproduces the canon end state"
```

---

### Task 9: Interpreters (prompt, population, executor)

**Files:**
- Create: `specdx/interpreters/__init__.py`, `specdx/interpreters/variants.json`, `specdx/interpreters/prompt.py`, `specdx/interpreters/population.py`, `specdx/interpreters/executor.py`
- Test: `tests/interpreters/__init__.py`, `tests/interpreters/test_population.py`, `tests/interpreters/test_executor.py`

**Interfaces:**
- Consumes: `TellCorpus` (Task 7), `run_case` (Task 8), `InterpreterId` (Task 2); TonomyX `tool_schemas`, `ToolRegistry` and the executor helpers `_Record`, `_arguments`, `_invented_numbers`, `_plain` (private but frozen in `vendor/`).
- Produces:
  - `TellVariants(standing, statements)`, with `.count`.
  - `load_variants(path=VARIANTS_PATH)` and `save_variants(variants, path=VARIANTS_PATH)`.
  - `role_of(pack, actor) -> str`, `system_prompt(variants, variant, *, pack, task, read, draft, suffix="") -> str` and `PLACEHOLDERS`.
  - `family_of(model) -> str`, `committee(models_by_family, variants, seeds) -> list[InterpreterId]` and `pilot(models_by_family, theta) -> list[InterpreterId]`. Both `committee` and `pilot` take families from the configuration, so a model always carries the same family label.
  - `SEARCH_SPEC`, `SEARCH_SCHEMA`, and `SpecInterpreterExecutor(model, tools, corpus, variants, *, variant=0, seed=1, temperature=0.7, base_url=OLLAMA_URL, max_steps=16, timeout=180.0, transport=None, suffix="")`, which implements `DeExecutor`.

- [ ] **Step 1: Write the Tell variants file (variant 0 only; Task 19 adds variants 1–2)**

`specdx/interpreters/__init__.py`: empty.

`specdx/interpreters/variants.json`:
```json
{
  "_note": "Tell variants: the standing instruction and each task statement. Variant 0 is hand-written and carries no business rule: every rule must come from the documents through search_spec. Variants 1-2 are paraphrases produced once by `specdx make-variants` and then frozen.",
  "standing": [
    "Bạn là {role} ({actor}) trong Hệ Doanh nghiệp Tự hành.\nQuy tắc làm việc:\n1. Chỉ hành động bằng các công cụ được cấp; việc không làm qua công cụ coi như chưa làm.\n2. Trước khi làm, dùng công cụ search_spec để tra tài liệu nghiệp vụ, rồi làm đúng như tài liệu quy định.\n3. Nếu công cụ từ chối, đọc lý do rồi điều chỉnh; không lặp lại đúng lời gọi đó.\n4. Khi xong, trả lời một câu tóm tắt bằng tiếng Việt và không gọi thêm công cụ.\nVụ việc: {case_id}. Nhiệm vụ: {statement}.\nDữ liệu vào: {inputs}.\nBảng được đọc: {read}.\nBảng được soạn nháp: {draft}."
  ],
  "statements": {
    "draft_quote": ["soạn báo giá cho đơn hàng của khách (discount_pct là phần trăm chiết khấu khách đề nghị)"],
    "check_availability": ["kiểm tra khả năng đáp ứng tồn kho cho các dòng của đơn hàng"],
    "decide_quote": ["bạn là người duyệt; quyết định của bạn về cổng phê duyệt nằm trong dữ liệu vào (quyet_dinh). Hãy ghi nhận quyết định đó và thực hiện các hệ quả theo tài liệu nghiệp vụ"],
    "draft_purchase_order": ["soạn đơn mua hàng bổ sung cho phần hàng thiếu của đơn hàng"],
    "decide_po": ["bạn là người duyệt; quyết định của bạn về cổng phê duyệt nằm trong dữ liệu vào (quyet_dinh). Hãy ghi nhận quyết định đó và thực hiện các hệ quả theo tài liệu nghiệp vụ"],
    "record_goods_receipt": ["ghi nhận phiếu nhập kho theo các lần giao thực tế (deliveries liệt kê mã phiếu nhập:số lượng)"],
    "draft_sales_invoice": ["soạn hóa đơn bán hàng cho khách"],
    "reconcile": ["đối soát ba chiều giữa đơn mua, phiếu nhập và hóa đơn nhà cung cấp, rồi ghi kết quả"],
    "draft_payment": ["soạn đề nghị chi cho hóa đơn nhà cung cấp"]
  }
}
```

- [ ] **Step 2: Write the failing tests**

`tests/interpreters/__init__.py`: empty.

`tests/interpreters/test_population.py`:
```python
from specdx.interpreters.population import committee, family_of, pilot
from specdx.types import InterpreterId


def test_families_come_from_the_model_name() -> None:
    assert family_of("qwen3:14b") == "qwen"
    assert family_of("qwen2.5:7b-instruct-q4_K_M") == "qwen"
    assert family_of("llama3.1:8b") == "llama"
    assert family_of("mistral-nemo:12b") == "mistral"
    assert family_of("command-r7b") == "cohere"


def test_the_committee_is_the_full_product_with_configured_families() -> None:
    ids = committee({"x": ("a:1",), "llama": ("llama3.1:8b",)}, variants=(0, 1), seeds=(1, 2, 3))
    assert len(ids) == 12
    assert InterpreterId("llama3.1:8b", "llama", 1, 3) in ids
    assert InterpreterId("a:1", "x", 0, 1) in ids


def test_the_pilot_takes_the_best_model_per_family_or_the_first_listed() -> None:
    by_family = {"qwen": ("qwen3:14b", "qwen2.5:7b"), "llama": ("llama3.1:8b",)}
    assert pilot(by_family, None) == [
        InterpreterId("qwen3:14b", "qwen", 0, 1), InterpreterId("llama3.1:8b", "llama", 0, 1)
    ]
    assert pilot(by_family, {"qwen2.5:7b": 0.9, "qwen3:14b": 0.4})[0].model == "qwen2.5:7b"
```

`tests/interpreters/test_executor.py`:
```python
import json
from datetime import UTC, datetime

import httpx
import pytest
from tonomyx.contracts.clock import FixedClock
from tonomyx.contracts.envelope import ResultStatus
from tonomyx.de.runtime import build_runtime
from tonomyx.flow.case_0031 import open_case
from tonomyx.tools.catalog import build_registry

from specdx.interpreters.executor import SpecInterpreterExecutor
from specdx.interpreters.prompt import load_variants
from specdx.orchestrate.flow import run_case
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import canon_scenarios
from specdx.spec.tell import TellCorpus

pytestmark = pytest.mark.db
CLOCK = FixedClock(datetime(2026, 9, 1, 2, 0, tzinfo=UTC))


def _call(name, arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def _say(text):
    return {"role": "assistant", "content": text}


def _scripted(*replies):
    queue, seen = list(replies), []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        reply = queue.pop(0) if queue else _say("Xong.")
        return httpx.Response(200, json={"message": reply, "done": True})

    return httpx.MockTransport(handle), seen


@pytest.fixture(scope="module")
def corpus(pack) -> TellCorpus:
    return TellCorpus.build(pack, [])


@pytest.fixture
def run_de(db, corpus):
    def run(transport, de="DE-OP-01", intent="check_availability", inputs=None):
        executor = SpecInterpreterExecutor(
            "scripted", build_registry(), corpus, load_variants(), transport=transport, seed=7
        )
        with db.isolated() as conn:
            runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
            open_case(runtime.repo)
            envelope = runtime.run(de, intent, "CASE-0031", inputs or {"ma_so": "SO-0031"})
            rows = conn.execute("SELECT kind, ref FROM evt").fetchall()
        return envelope, rows
    return run


def test_the_token_tools_and_search_spec_are_offered(run_de) -> None:
    transport, seen = _scripted(_say("Xong."))
    run_de(transport)
    assert {t["function"]["name"] for t in seen[0]["tools"]} == {
        "read_records", "log_event", "search_spec"
    }


def test_sampling_options_and_a_prompt_without_playbooks(run_de) -> None:
    transport, seen = _scripted(_say("Xong."))
    run_de(transport)
    assert seen[0]["options"] == {"temperature": 0.7, "seed": 7}
    system = seen[0]["messages"][0]["content"]
    assert "search_spec" in system and "kiểm tra khả năng đáp ứng" in system
    assert "Hướng dẫn nghiệp vụ" not in system


def test_search_spec_answers_from_the_corpus_and_tools_write_for_real(run_de) -> None:
    transport, seen = _scripted(
        _call("search_spec", {"query": "chiết khấu tối đa"}),
        _call("log_event", {"case_id": "CASE-0031", "kind": "FLAGGED", "ref": "ITEM-0012",
                            "actor": "AI", "basis": "thiếu"}),
        _say("Đã gắn cờ."),
    )
    envelope, rows = run_de(transport)
    tool_reply = [m for m in seen[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "KB:KB-0003" in tool_reply
    assert envelope.status is ResultStatus.DONE
    assert {"kind": "FLAGGED", "ref": "ITEM-0012"} in rows


def test_a_number_the_canon_leaves_blank_is_refused(run_de) -> None:
    transport, seen = _scripted(
        _call("write_draft", {"table": "pay", "idempotency_key": "k",
                              "payload": {"ma_pay": "PAY-X", "ma_vinv": "VINV-0015",
                                          "so_tien": 0, "trang_thai": "Chờ duyệt"}}),
        _say("Xong."),
    )
    run_de(transport, de="DE-FN-03", intent="draft_payment", inputs={"ma_pay": "PAY-X"})
    reply = [m for m in seen[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "denied" in reply and "pay.so_tien" in reply


def test_a_timeout_ends_the_step_not_the_case(db, pack, corpus) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    executor = SpecInterpreterExecutor(
        "scripted", build_registry(), corpus, load_variants(),
        transport=httpx.MockTransport(handle),
    )
    steps = load_work_orders()
    with db.isolated() as conn:
        runtime = build_runtime(conn, db.settings, executor=executor, clock=CLOCK)
        case = run_case(runtime, canon_scenarios(pack)["grn12"], steps)
    assert len(case.statuses) == len(steps)
    assert case.statuses["DE-SM-02:draft_quote"] == "error"
    assert "ollama call failed" in case.claims["DE-SM-02:draft_quote"]
    assert case.statuses["P2:decide_quote"] == "error"
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/interpreters -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.interpreters.population'`.

- [ ] **Step 4: Implement**

`specdx/interpreters/prompt.py`:
```python
"""Tell variants and the system prompt an interpreter reads."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tonomyx.canon.pack import CanonPack
from tonomyx.contracts.envelope import TaskEnvelope

VARIANTS_PATH = Path(__file__).with_name("variants.json")
PLACEHOLDERS: tuple[str, ...] = (
    "{role}", "{actor}", "{case_id}", "{statement}", "{inputs}", "{read}", "{draft}",
)
_NOTE = (
    "Tell variants: the standing instruction and each task statement. Variant 0 is "
    "hand-written and carries no business rule: every rule must come from the documents "
    "through search_spec. Variants 1-2 are paraphrases produced once by "
    "`specdx make-variants` and then frozen."
)


@dataclass(frozen=True)
class TellVariants:
    standing: tuple[str, ...]
    statements: dict[str, tuple[str, ...]]

    @property
    def count(self) -> int:
        return len(self.standing)


def load_variants(path: Path = VARIANTS_PATH) -> TellVariants:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TellVariants(
        standing=tuple(raw["standing"]),
        statements={k: tuple(v) for k, v in raw["statements"].items()},
    )


def save_variants(variants: TellVariants, path: Path = VARIANTS_PATH) -> None:
    payload = {
        "_note": _NOTE,
        "standing": list(variants.standing),
        "statements": {k: list(v) for k, v in variants.statements.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def role_of(pack: CanonPack, actor: str) -> str:
    roles = {r["ma_de"]: r["vai"] for r in pack.tables.get("de", [])}
    roles |= {r["ma_user"]: r["ho_ten"] for r in pack.tables.get("app_user", [])}
    return roles.get(actor, "không rõ vai")


def _names(tables: Iterable[str]) -> str:
    return ", ".join(sorted(tables)) or "(không có)"


def system_prompt(
    variants: TellVariants,
    variant: int,
    *,
    pack: CanonPack,
    task: TaskEnvelope,
    read: Iterable[str],
    draft: Iterable[str],
    suffix: str = "",
) -> str:
    text = variants.standing[variant].format(
        role=role_of(pack, task.de),
        actor=task.de,
        case_id=task.case_id,
        statement=variants.statements[task.intent][variant],
        inputs=json.dumps(task.inputs, ensure_ascii=False),
        read=_names(read),
        draft=_names(draft),
    )
    return f"{text}\n{suffix}" if suffix else text
```

`specdx/interpreters/population.py`:
```python
"""Who is in the population, and who runs the pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from specdx.types import InterpreterId

_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("qwen", "qwen"), ("llama", "llama"), ("mistral", "mistral"), ("granite", "granite"),
    ("phi", "phi"), ("command-r", "cohere"), ("gemma", "gemma"), ("exaone", "exaone"),
    ("solar", "solar"),
)


def family_of(model: str) -> str:
    for prefix, family in _FAMILY_PREFIXES:
        if model.startswith(prefix):
            return family
    return model.split(":", 1)[0]


def committee(
    models_by_family: Mapping[str, Sequence[str]], variants: Sequence[int], seeds: Sequence[int]
) -> list[InterpreterId]:
    return [
        InterpreterId(m, family, v, s)
        for family, models in models_by_family.items()
        for m in models for v in variants for s in seeds
    ]


def pilot(
    models_by_family: Mapping[str, Sequence[str]], theta: Mapping[str, float] | None
) -> list[InterpreterId]:
    """One interpreter per family: the most competent model so far, else the first listed."""
    out: list[InterpreterId] = []
    for family, models in models_by_family.items():
        known = [m for m in models if theta and m in theta]
        best = max(known, key=lambda m: theta[m]) if known and theta else models[0]
        out.append(InterpreterId(best, family, 0, 1))
    return out
```

`specdx/interpreters/executor.py`:
```python
"""An interpreter: a local model that reads T through search_spec and acts through gated tools.

It follows OllamaExecutor's contract: the envelope comes from what the tools did, and every
call runs in its own savepoint. It differs in what the model is told: no playbook and no
business rule in the prompt, only the Tell corpus behind search_spec.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope
from tonomyx.de.executor import _arguments, _invented_numbers, _plain, _Record
from tonomyx.tools.registry import ToolContext, ToolDenied, ToolNotFound, ToolRegistry
from tonomyx.tools.schema import tool_schemas

from specdx.config import OLLAMA_URL
from specdx.interpreters.prompt import TellVariants, system_prompt
from specdx.spec.tell import TellCorpus

SEARCH_SPEC = "search_spec"
SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_SPEC,
        "description": (
            "Tra cứu tài liệu nghiệp vụ (tài liệu BA, mô tả demo, SAD, KB, nguyên tắc canon). "
            "Trả về các đoạn liên quan nhất."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
_REPLY_LIMIT = 4000


class SpecInterpreterExecutor:
    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        corpus: TellCorpus,
        variants: TellVariants,
        *,
        variant: int = 0,
        seed: int = 1,
        temperature: float = 0.7,
        base_url: str = OLLAMA_URL,
        max_steps: int = 16,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
        suffix: str = "",
    ) -> None:
        self.model, self.tools, self.corpus, self.variants = model, tools, corpus, variants
        self.variant, self.seed, self.temperature = variant, seed, temperature
        self.base_url, self.max_steps, self.timeout = base_url, max_steps, timeout
        self.transport, self.suffix = transport, suffix

    def run(self, task: TaskEnvelope, ctx: ToolContext) -> ResultEnvelope:
        schemas = [*tool_schemas(self.tools, ctx.pack, names=ctx.token.tools), SEARCH_SCHEMA]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(
                self.variants, self.variant, pack=ctx.pack, task=task,
                read=ctx.token.read, draft=ctx.token.draft, suffix=self.suffix,
            )},
            {"role": "user", "content": f"Thực hiện nhiệm vụ {task.intent} cho {task.case_id}."},
        ]
        record = _Record()
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
                for step in range(1, self.max_steps + 1):
                    reply = self._chat(client, messages, schemas)
                    messages.append(reply)
                    calls = reply.get("tool_calls") or []
                    if not calls:
                        return record.finish(task, self.model, step, _plain(reply))
                    for call in calls:
                        messages.append(self._execute(call, ctx, record))
        except httpx.HTTPError as exc:
            return ResultEnvelope(
                task_id=task.task_id, case_id=task.case_id, status=ResultStatus.ERROR,
                next_hint=f"ollama call failed: {exc}",
            )
        return record.envelope(
            task, self.model, self.max_steps, ResultStatus.ERROR,
            f"{self.model} did not finish within {self.max_steps} steps",
        )

    def _chat(
        self, client: httpx.Client, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model, "messages": messages, "tools": schemas, "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        if self.model.startswith("qwen3"):
            body["think"] = False
        response = client.post(f"{self.base_url}/api/chat", json=body)
        response.raise_for_status()
        message: dict[str, Any] = response.json()["message"]
        return message

    def _execute(self, call: dict[str, Any], ctx: ToolContext, record: _Record) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        args: dict[str, Any] = {}
        payload: Any
        try:
            args = _arguments(function.get("arguments"))
            if name == SEARCH_SPEC:
                hits = self.corpus.search(str(args.get("query", "")), k=5)
                payload = {"passages": [{"id": c.id, "text": c.text} for c in hits]}
                record.calls.append({"name": name, "args": args, "outcome": "ok"})
            else:
                invented = _invented_numbers(name, args, ctx.pack.non_canon_columns)
                if invented:
                    raise ToolDenied(
                        f"the canon leaves {', '.join(invented)} blank: pass null instead"
                    )
                with ctx.repo.conn.transaction():
                    result = self.tools.invoke(name, ctx, **args)
                record.ok(name, args, result)
                payload = result
        except ToolDenied as exc:
            record.failed(name, args, "denied")
            payload = {"error": "denied", "tool": name, "reason": str(exc)}
        except ToolNotFound:
            record.failed(name, args, "unknown")
            payload = {"error": "unknown tool", "tool": name}
        except Exception as exc:  # noqa: BLE001 - the model is told what broke
            record.failed(name, args, "error")
            payload = {"error": type(exc).__name__, "tool": name, "reason": str(exc)}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        return {"role": "tool", "tool_name": name, "content": content[:_REPLY_LIMIT]}
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/interpreters -q`
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add specdx/interpreters tests/interpreters
git commit -m "feat: LLM interpreters that read the Tell corpus and act through gated tools"
```

---

### Task 10: Run cache, Runner, and item views

**Files:**
- Create: `specdx/interpreters/cache.py`, `specdx/observe/run.py`, `specdx/infer/__init__.py`, `specdx/infer/items.py`
- Test: `tests/interpreters/test_cache.py`, `tests/observe/__init__.py`, `tests/observe/test_run.py`, `tests/infer/__init__.py`, `tests/infer/test_items.py`

**Interfaces:**
- Consumes: `PackDatabase` (Task 3), `run_case` (Task 8), `capture_state` (Task 3), `SpecVariant` (Task 7), `ClauseSet` (Task 6), `ColumnModel` (Task 4).
- Produces:
  - `run_key(*parts) -> str` and `RunCache(path)`, with `.get(key)`, `.put(behaviour)` and `len()`.
  - `ExecutorFactory = Callable[[InterpreterId, TellCorpus, str], DeExecutor]` and `RUN_CLOCK`.
  - `Runner(db, cache, factory, steps, *, model_digest=None)`, with `.run(variant, scenario, interp, suffix="") -> Behaviour` and `.executed`, the count of runs not served from cache.
  - `ItemView(item, run_key, model, family, klass, violated, projection)`.
  - `SpecLens(clauses, steps, columns)`, with `.violations(behaviour)`, `.views(behaviour, violated) -> list[ItemView]`, `.diff(a, b)` and `.relevant[step_id]`.

- [ ] **Step 1: Write the failing tests**

`tests/interpreters/test_cache.py`:
```python
from specdx.interpreters.cache import RunCache, run_key
from specdx.types import Behaviour, InterpreterId, Scenario, State


def _b(key: str) -> Behaviour:
    return Behaviour(key, InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0), State({}))


def test_keys_are_stable_and_distinct() -> None:
    assert run_key("a", "b") == run_key("a", "b") != run_key("a", "c")


def test_the_cache_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    RunCache(path).put(_b("k1"))
    assert RunCache(path).get("k1") == _b("k1")


def test_a_truncated_last_line_is_skipped(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    cache = RunCache(path)
    cache.put(_b("k1"))
    cache.put(_b("k2"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_key": "k3", "interp')
    reloaded = RunCache(path)
    assert len(reloaded) == 2 and reloaded.get("k3") is None
```

`tests/observe/__init__.py`: empty.

`tests/observe/test_run.py`:
```python
import pytest

from specdx.config import BASE_PACK
from specdx.interpreters.cache import RunCache
from specdx.observe.run import Runner
from specdx.orchestrate.reference import reference_executor
from specdx.orchestrate.work_orders import load_work_orders
from specdx.scenarios.space import canon_scenarios
from specdx.spec.clauses import ClauseSet
from specdx.spec.tell import TellCorpus
from specdx.types import InterpreterId, Scenario
from specdx.variant import SpecVariant

pytestmark = pytest.mark.db
STEPS = load_work_orders()
REF = InterpreterId("reference", "reference", 0, 1)


@pytest.fixture
def variant(pack) -> SpecVariant:
    return SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))


def test_a_run_is_executed_once_then_served_from_cache(db, pack, variant, tmp_path) -> None:
    runner = Runner(db, RunCache(tmp_path / "r.jsonl"), lambda i, c, s: reference_executor(),
                    STEPS, model_digest=lambda m: "d")
    grn12 = canon_scenarios(pack)["grn12"]
    first = runner.run(variant, grn12, REF)
    assert first.status == "ok"
    assert ClauseSet.from_pack(pack, STEPS).violations(first.state, grn12) == frozenset()
    renamed = runner.run(variant, Scenario("o15-r12-d0-g1", 15, 12, 0), REF)
    assert runner.executed == 1
    assert renamed.scenario.id == "o15-r12-d0-g1" and renamed.state == first.state


def test_a_crashing_factory_yields_an_error_behaviour(db, pack, variant, tmp_path) -> None:
    def broken(i, c, s):
        raise RuntimeError("no model")

    runner = Runner(db, RunCache(tmp_path / "r.jsonl"), broken, STEPS, model_digest=lambda m: "d")
    b = runner.run(variant, canon_scenarios(pack)["grn12"], REF)
    assert b.status == "error" and "no model" in b.error
```

`tests/infer/__init__.py`: empty.

`tests/infer/test_items.py`:
```python
import pytest

from specdx.infer.items import SpecLens
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.types import Behaviour, InterpreterId, Scenario, State

STEPS = load_work_orders()
GRN12 = Scenario("grn12", 15, 12, 0)


@pytest.fixture(scope="module")
def lens(pack) -> SpecLens:
    return SpecLens(ClauseSet.from_pack(pack, STEPS), STEPS, ColumnModel.of(pack))


def _behaviour(golden: State, key: str, so_status: str) -> Behaviour:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["so"][0]["trang_thai"] = so_status
    state = State(tables, golden.derived, {s.id: "done" for s in STEPS})
    return Behaviour(key, InterpreterId("m", "f", 0, 1), GRN12, state)


def test_relevance_is_local_to_a_step(lens) -> None:
    assert "golden:so:SO-0031:trang_thai" in lens.relevant["P2:decide_quote"]
    assert "golden:so:SO-0031:trang_thai" not in lens.relevant["DE-FN-02:reconcile"]
    assert "event:FLAGGED:ITEM-0012:AI" in lens.relevant["DE-OP-01:check_availability"]
    assert "inv:apr0155-approved" not in lens.relevant["P2:decide_quote"]


@pytest.mark.db
def test_equal_behaviours_share_classes_and_a_difference_splits_one_step(lens, golden) -> None:
    a = _behaviour(golden, "a", "Chờ duyệt")
    b = _behaviour(golden, "b", "Chờ duyệt")
    c = _behaviour(golden, "c", "Đã duyệt")
    va, vb, vc = (lens.views(x, lens.violations(x)) for x in (a, b, c))
    assert [v.klass for v in va] == [v.klass for v in vb]
    differing = [x.item for x, z in zip(va, vc, strict=True) if x.klass != z.klass]
    assert set(differing) == {"grn12|DE-SM-02:draft_quote", "grn12|P2:decide_quote"}
    p2 = next(v for v in vc if v.item == "grn12|P2:decide_quote")
    assert p2.violated == frozenset({"golden:so:SO-0031:trang_thai"})


def test_an_error_behaviour_has_no_views(lens) -> None:
    b = Behaviour("e", InterpreterId("m", "f", 0, 1), GRN12, State.empty(), status="error")
    assert lens.violations(b) == frozenset() and lens.views(b, frozenset()) == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/interpreters/test_cache.py tests/observe tests/infer/test_items.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.interpreters.cache'`.

- [ ] **Step 3: Implement**

`specdx/interpreters/cache.py`:
```python
"""Every run is cached on disk, so a study can be resumed and reproduced."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from specdx.types import Behaviour


def run_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


class RunCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Behaviour] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    b = Behaviour.from_json(line)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # a line cut off by a crash; the run will simply be redone
                self._items[b.run_key] = b

    def get(self, key: str) -> Behaviour | None:
        return self._items.get(key)

    def put(self, behaviour: Behaviour) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(behaviour.to_json() + "\n")
        self._items[behaviour.run_key] = behaviour

    def __len__(self) -> int:
        return len(self._items)
```

`specdx/observe/run.py`:
```python
"""One interpreter, one scenario, one specification: a cached Behaviour."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from tonomyx.contracts.clock import FixedClock
from tonomyx.de.executor import DeExecutor
from tonomyx.de.runtime import build_runtime

from specdx.db import PackDatabase
from specdx.interpreters.cache import RunCache, run_key
from specdx.observe.state import capture_state
from specdx.orchestrate.flow import run_case
from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InterpreterId, Scenario, State
from specdx.variant import SpecVariant

ExecutorFactory = Callable[[InterpreterId, TellCorpus, str], DeExecutor]
RUN_CLOCK = datetime(2026, 9, 1, 2, 0, tzinfo=UTC)


class Runner:
    def __init__(
        self,
        db: PackDatabase,
        cache: RunCache,
        factory: ExecutorFactory,
        steps: Sequence[WorkStep],
        *,
        model_digest: Callable[[str], str] | None = None,
    ) -> None:
        self.db, self.cache, self.factory, self.steps = db, cache, factory, tuple(steps)
        self.model_digest = model_digest or (lambda model: model)
        self.executed = 0

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = ""
    ) -> Behaviour:
        key = run_key(
            variant.digest(), json.dumps(scenario.params(), sort_keys=True), interp.key,
            suffix, self.model_digest(interp.model),
        )
        cached = self.cache.get(key)
        if cached is not None:
            return dataclasses.replace(cached, scenario=scenario)
        self.executed += 1
        try:
            settings = self.db.use(variant.pack)
            executor = self.factory(interp, variant.corpus, suffix)
            with self.db.isolated() as conn:
                runtime = build_runtime(conn, settings, executor=executor,
                                        clock=FixedClock(RUN_CLOCK))
                case = run_case(runtime, scenario, self.steps)
                state = capture_state(conn, runtime.pack, case.statuses)
            behaviour = Behaviour(
                run_key=key, interpreter=interp, scenario=scenario, state=state,
                refused=tuple(case.refused), claims=dict(case.claims),
            )
        except Exception as exc:  # noqa: BLE001 - a failed run is data, never dropped
            behaviour = Behaviour(
                run_key=key, interpreter=interp, scenario=scenario, state=State.empty(),
                status="error", error=f"{type(exc).__name__}: {exc}",
            )
        self.cache.put(behaviour)
        return behaviour
```

`specdx/infer/__init__.py`: empty.

`specdx/infer/items.py`:
```python
"""Per-step views of a behaviour: the class an interpreter's output falls in at each step."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from specdx.orchestrate.work_orders import WorkStep
from specdx.spec.clauses import Clause, ClauseSet
from specdx.spec.columns import ColumnModel, Projection
from specdx.types import Behaviour


@dataclass(frozen=True)
class ItemView:
    item: str
    run_key: str
    model: str
    family: str
    klass: str
    violated: frozenset[str]
    projection: Projection


def _keyed(step: WorkStep, table: str) -> set[str] | None:
    """None when the step owns the whole table; else the keys it owns (maybe none)."""
    keys: set[str] = set()
    for entry in step.slice:
        name, _, key = entry.partition("=")
        if name == table:
            if not key:
                return None
            keys.add(key)
    return keys


def _relevant(step: WorkStep, clause: Clause) -> bool:
    if clause.kind == "step":
        return bool(clause.payload["step"] == step.id)
    if clause.kind == "vocabulary":
        return False
    if clause.kind == "golden_event":
        keys = _keyed(step, "evt")
        return keys is None or clause.payload["event"][1] in keys
    if clause.kind == "golden":
        keys = _keyed(step, clause.payload["table"])
        ident = ":".join(str(v) for v in clause.payload["key"].values())
        return keys is None or ident in keys
    if clause.kind == "invariant":
        inv = clause.payload["invariant"]
        keys = _keyed(step, inv["table"])
        if keys is None:
            return True
        return bool(keys & {str(v) for v in (inv.get("where") or {}).values()})
    return bool(clause.tables & step.tables)


class SpecLens:
    def __init__(
        self, clauses: ClauseSet, steps: Sequence[WorkStep], columns: ColumnModel
    ) -> None:
        self.clauses, self.steps, self.columns = clauses, tuple(steps), columns
        self.relevant = {
            s.id: frozenset(c.id for c in clauses if _relevant(s, c)) for s in self.steps
        }

    def violations(self, behaviour: Behaviour) -> frozenset[str]:
        if behaviour.status != "ok":
            return frozenset()
        return self.clauses.violations(behaviour.state, behaviour.scenario)

    def views(self, behaviour: Behaviour, violated: frozenset[str]) -> list[ItemView]:
        if behaviour.status != "ok":
            return []
        out: list[ItemView] = []
        for step in self.steps:
            mine = violated & self.relevant[step.id]
            projection = self.columns.project(behaviour.state.tables, step.slice)
            status = behaviour.state.steps.get(step.id, "")
            klass = hashlib.sha1(repr((projection, sorted(mine), status)).encode()).hexdigest()[:16]
            out.append(ItemView(
                item=f"{behaviour.scenario.id}|{step.id}", run_key=behaviour.run_key,
                model=behaviour.interpreter.model, family=behaviour.interpreter.family,
                klass=klass, violated=mine, projection=projection,
            ))
        return out

    def diff(self, a: Projection, b: Projection) -> frozenset[str]:
        return self.columns.diff(a, b)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/interpreters/test_cache.py tests/observe tests/infer/test_items.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/interpreters/cache.py specdx/observe/run.py specdx/infer tests/interpreters/test_cache.py tests/observe tests/infer
git commit -m "feat: cached Runner and per-step item views with local clause relevance"
```

---
### Task 11: The set-valued latent-class model

**Files:**
- Create: `specdx/infer/latent.py`
- Test: `tests/infer/test_latent.py`

**Interfaces:**
- Produces:
  - `RARE` and `Observation(item, run_key, model, family, klass)`.
  - `ItemPosterior(classes, set_posterior, p_member, p_ambiguous, responsibility)`, with `.map_set`.
  - `Posterior(theta, family_effect, items)`.
  - `fit(observations, *, k_max=6, p_single=0.9, sigma_v=1.0, sigma_u=1.0, iters=50, tol=1e-6, theta0=0.7) -> Posterior`.
- Competence θ is indexed by **model**, pooled over seeds and Tell variants (spec §11, item 5).
- **Known risk, found while pre-checking this plan:** when there are few items per run, the family effect pulls a good model towards a weak model of the same family. The model may then pick one class of a genuinely ambiguous step as "admissible", and report a spurious *weak* defect on the other. Real runs have eleven steps per behaviour, which is what the Task 16 test now mimics. If the gate (Task 20) shows weak findings that mirror ambiguities, report it and try `sigma_v=2.0` on the development split before anything else.

- [ ] **Step 1: Write the failing tests**

`tests/infer/test_latent.py`:
```python
import random

from specdx.infer.latent import RARE, Observation, fit

FAMILY = {"g1": "fa", "g2": "fb", "g3": "fc", "b1": "fa", "b2": "fb", "b3": "fc"}
GOOD = ("g1", "g2", "g3")
RUNS = 4


def _obs(item, model, i, klass):
    return Observation(item, f"{item}|{model}|{i}", model, FAMILY[model], klass)


def _clean_item(rng, item, truth):
    out = []
    for model in FAMILY:
        for i in range(RUNS):
            if model in GOOD:
                klass = truth if rng.random() < 0.9 else f"err-{rng.randrange(6)}"
            else:
                klass = f"err-{rng.randrange(6)}"
            out.append(_obs(item, model, i, klass))
    return out


def _ambiguous_item(rng, item):
    out = []
    for model in FAMILY:
        for i in range(RUNS):
            klass = ("X" if i % 2 == 0 else "Y") if model in GOOD else f"err-{rng.randrange(6)}"
            out.append(_obs(item, model, i, klass))
    return out


def _dataset():
    rng = random.Random(0)
    obs = []
    for j in range(8):
        obs += _clean_item(rng, f"clean{j}", f"A{j}")
    obs += _ambiguous_item(rng, "amb")
    return obs


def test_clean_items_recover_their_single_admissible_class() -> None:
    post = fit(_dataset())
    for j in range(8):
        item = post.items[f"clean{j}"]
        assert item.map_set == frozenset({f"A{j}"})
        assert item.p_ambiguous < 0.2


def test_an_ambiguous_item_is_recognised() -> None:
    item = fit(_dataset()).items["amb"]
    assert item.p_ambiguous > 0.5
    assert item.p_member["X"] > 0.5 and item.p_member["Y"] > 0.5


def test_competence_separates_good_from_bad_models() -> None:
    theta = fit(_dataset()).theta
    assert min(theta[m] for m in GOOD) > max(theta[m] for m in FAMILY if m not in GOOD)


def test_responsibility_is_high_for_admissible_outputs() -> None:
    post = fit(_dataset())
    item = post.items["clean0"]
    good = [o for o in _dataset() if o.item == "clean0" and o.klass == "A0"]
    assert all(item.responsibility[o.run_key] > 0.8 for o in good)


def test_a_family_of_bad_models_gets_a_lower_effect() -> None:
    family = {"g1": "fa", "g2": "fa", "b1": "fb", "b2": "fb"}
    rng = random.Random(1)
    obs = []
    for j in range(6):
        for model, fam in family.items():
            for i in range(RUNS):
                good = model.startswith("g")
                klass = f"A{j}" if good and rng.random() < 0.9 else f"err-{rng.randrange(6)}"
                obs.append(Observation(f"i{j}", f"i{j}|{model}|{i}", model, fam, klass))
    effect = fit(obs).family_effect
    assert effect["fa"] > effect["fb"]


def test_classes_beyond_k_max_are_pooled_and_never_admissible() -> None:
    obs = [Observation("i", f"r{n}", "m", "f", klass)
           for n, klass in enumerate(["a"] * 5 + ["b"] * 4 + ["c"] * 3)]
    item = fit(obs, k_max=2).items["i"]
    assert item.classes == ("a", "b")
    assert all(RARE not in s for s in item.set_posterior)


def test_no_observations_give_an_empty_posterior() -> None:
    post = fit([])
    assert post.theta == {} and post.items == {}
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/infer/test_latent.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.infer.latent'`.

- [ ] **Step 3: Implement**

`specdx/infer/latent.py`:
```python
"""Set-valued latent-class model (spec §3.2).

Each item (a scenario step) has an unknown SET A of admissible output classes. A model
with competence θ emits an admissible class with probability θ (uniformly within A),
and otherwise a class drawn from the item's smoothed empirical error distribution ε.
Competence is shared across items, which is what separates a specification defect (every
competent model deviates at the same place) from an agent error (scattered).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

RARE = "__rare__"
_EDGE = 1e-4


@dataclass(frozen=True)
class Observation:
    item: str
    run_key: str
    model: str
    family: str
    klass: str


@dataclass
class ItemPosterior:
    classes: tuple[str, ...]
    set_posterior: dict[frozenset[str], float]
    p_member: dict[str, float]
    p_ambiguous: float
    responsibility: dict[str, float]

    @property
    def map_set(self) -> frozenset[str]:
        return max(
            self.set_posterior,
            key=lambda s: (self.set_posterior[s], -len(s), tuple(sorted(s))),
        )


@dataclass
class Posterior:
    theta: dict[str, float]
    family_effect: dict[str, float]
    items: dict[str, ItemPosterior]


def _sigmoid(x: float) -> float:
    p = 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))
    return min(max(p, _EDGE), 1 - _EDGE)


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _log_sum_exp(values: Sequence[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def _subset_prior(m: int, p_single: float) -> dict[int, float]:
    """Prior mass of one subset of each size: singletons share p_single."""
    if m == 1:
        return {1: 1.0}
    weights = {s: 0.5 ** (s - 1) for s in range(2, m + 1)}
    total = sum(weights.values())
    prior = {1: p_single / m}
    for size, w in weights.items():
        prior[size] = (1 - p_single) * (w / total) / math.comb(m, size)
    return prior


def _item_posterior(
    obs: Sequence[Observation], theta: dict[str, float], k_max: int, p_single: float
) -> ItemPosterior:
    counts = Counter(o.klass for o in obs)
    top = tuple(k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k_max])
    n, m = len(obs), len(top)
    eps = {k: (counts[k] + 1) / (n + m + 1) for k in top}
    eps[RARE] = (n - sum(counts[k] for k in top) + 1) / (n + m + 1)
    labelled = [(o, o.klass if o.klass in top else RARE) for o in obs]
    prior = _subset_prior(m, p_single)
    subsets = [frozenset(c) for size in range(1, m + 1) for c in combinations(top, size)]
    log_post: dict[frozenset[str], float] = {}
    for subset in subsets:
        lp = math.log(prior[len(subset)])
        for o, k in labelled:
            th = theta[o.model]
            faith = th / len(subset) if k in subset else 0.0
            lp += math.log(faith + (1 - th) * eps[k])
        log_post[subset] = lp
    z = _log_sum_exp(list(log_post.values()))
    post = {s: math.exp(v - z) for s, v in log_post.items()}
    p_member = {k: sum(p for s, p in post.items() if k in s) for k in top}
    p_ambiguous = sum(p for s, p in post.items() if len(s) >= 2)
    responsibility: dict[str, float] = {}
    for o, k in labelled:
        th = theta[o.model]
        r = 0.0
        for s, p in post.items():
            faith = th / len(s) if k in s else 0.0
            r += p * faith / (faith + (1 - th) * eps[k])
        responsibility[o.run_key] = r
    return ItemPosterior(top, post, p_member, p_ambiguous, responsibility)


def fit(
    observations: Sequence[Observation],
    *,
    k_max: int = 6,
    p_single: float = 0.9,
    sigma_v: float = 1.0,
    sigma_u: float = 1.0,
    iters: int = 50,
    tol: float = 1e-6,
    theta0: float = 0.7,
) -> Posterior:
    if not observations:
        return Posterior({}, {}, {})
    by_item: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_item[o.item].append(o)
    family = {o.model: o.family for o in observations}
    mu = _logit(theta0)
    eta = {m: mu for m in family}
    effect = {f: 0.0 for f in set(family.values())}
    for _ in range(iters):
        theta = {m: _sigmoid(e) for m, e in eta.items()}
        items = {i: _item_posterior(obs, theta, k_max, p_single) for i, obs in by_item.items()}
        rs: dict[str, list[float]] = defaultdict(list)
        for o in observations:
            rs[o.model].append(items[o.item].responsibility[o.run_key])
        new_eta: dict[str, float] = {}
        for model, values in rs.items():
            center = mu + effect[family[model]]
            e = eta[model]
            for _ in range(20):
                p = _sigmoid(e)
                grad = sum(values) - len(values) * p - (e - center) / sigma_v**2
                hess = -len(values) * p * (1 - p) - 1 / sigma_v**2
                step = grad / hess
                e -= step
                if abs(step) < 1e-9:
                    break
            new_eta[model] = e
        for fam in effect:
            members = [m for m in new_eta if family[m] == fam]
            effect[fam] = sigma_u**2 * sum(new_eta[m] - mu for m in members) / (
                sigma_u**2 * len(members) + sigma_v**2
            )
        mu = sum(new_eta[m] - effect[family[m]] for m in new_eta) / len(new_eta)
        delta = max(abs(_sigmoid(new_eta[m]) - _sigmoid(eta[m])) for m in new_eta)
        eta = new_eta
        if delta < tol:
            break
    theta = {m: _sigmoid(e) for m, e in eta.items()}
    items = {i: _item_posterior(obs, theta, k_max, p_single) for i, obs in by_item.items()}
    return Posterior(theta, effect, items)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/infer/test_latent.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/infer/latent.py tests/infer/test_latent.py
git commit -m "feat: set-valued latent-class EM with family effects"
```

---

### Task 12: Soft minimal hitting sets

**Files:**
- Create: `specdx/infer/hitting.py`
- Test: `tests/infer/test_hitting.py`

**Interfaces:**
- Produces `soft_hitting_sets(weighted_sets: Sequence[tuple[frozenset[str], float]], *, tau=0.05, max_size=3, min_mass=0.01) -> list[tuple[frozenset[str], float]]`. The result is sorted by support (descending), then size, then name. Each hitting set is minimal: no proper subset also qualifies.

- [ ] **Step 1: Write the failing tests**

`tests/infer/test_hitting.py`:
```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/infer/test_hitting.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.infer.hitting'`.

- [ ] **Step 3: Implement**

`specdx/infer/hitting.py`:
```python
"""Soft minimal hitting sets over sampled correction sets (spec §3.3).

Each behaviour b satisfies C minus V(b), so V(b) is a correction set. Every MUS is a
minimal hitting set of the minimal correction sets (Liffiton & Sakallah, 2008). With noisy
samples, a set H qualifies when it hits at least (1 - tau) of the weighted mass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations


def soft_hitting_sets(
    weighted_sets: Sequence[tuple[frozenset[str], float]],
    *,
    tau: float = 0.05,
    max_size: int = 3,
    min_mass: float = 0.01,
) -> list[tuple[frozenset[str], float]]:
    total = sum(w for _, w in weighted_sets)
    if total <= 0:
        return []
    mass: Counter[str] = Counter()
    for violated, weight in weighted_sets:
        for clause in violated:
            mass[clause] += weight
    universe = sorted(c for c, m in mass.items() if m / total >= min_mass)
    found: list[tuple[frozenset[str], float]] = []
    for size in range(1, max_size + 1):
        for combo in combinations(universe, size):
            candidate = frozenset(combo)
            if any(previous <= candidate for previous, _ in found):
                continue
            covered = sum(w for violated, w in weighted_sets if violated & candidate) / total
            if covered >= 1 - tau:
                found.append((candidate, covered))
    return sorted(found, key=lambda x: (-x[1], len(x[0]), sorted(x[0])))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/infer/test_hitting.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/infer/hitting.py tests/infer/test_hitting.py
git commit -m "feat: soft minimal hitting sets over sampled correction sets"
```

---

### Task 13: Refutation (symbolic repair and directed probe)

**Files:**
- Create: `specdx/active/__init__.py`, `specdx/active/refute.py`
- Test: `tests/active/__init__.py`, `tests/active/test_refute.py`

**Interfaces:**
- Consumes: `ClauseSet`, `Clause` (Task 6), `PackDatabase.materialize` (Task 3), `Runner` (Task 10), `RefuteResult` (Task 2), `instantiate` (Task 5).
- Produces:
  - `propose(clause, state, scenario) -> list[State]`.
  - `SymbolicRepair(db, variant, clauses, *, budget=200)`, with `.search(targets, scenario, starts) -> State | None`.
  - `DirectedProbe(runner, variant, clauses, interpreters, *, attempts=3, width=3)`, with `.search(targets, scenario, allowed) -> Behaviour | None`.
  - `CombinedRefuter(clauses, repair, probe)`, which implements `Refuter`; its results are memoised per (scenario signature, targets).
- **Refutation rule (spec §3.6, §11 item 4):** a witness satisfies every target clause and introduces no violation beyond those its starting point already had.

- [ ] **Step 1: Write the failing tests**

`tests/active/__init__.py`: empty.

`tests/active/test_refute.py`:
```python
import dataclasses

import pytest

from specdx.active.refute import CombinedRefuter, DirectedProbe, SymbolicRepair
from specdx.config import BASE_PACK
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import Clause, ClauseSet
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, InterpreterId, Scenario, State
from specdx.variant import SpecVariant

pytestmark = pytest.mark.db
STEPS = load_work_orders()
GRN12 = Scenario("grn12", 15, 12, 0)


@pytest.fixture
def clauses(pack) -> ClauseSet:
    return ClauseSet.from_pack(pack, STEPS)


@pytest.fixture
def variant(pack) -> SpecVariant:
    return SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))


def _done(golden: State, cells: dict[tuple[str, str], str] | None = None) -> State:
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    for (table, column), value in (cells or {}).items():
        tables[table][0][column] = value
    return State(tables, dict(golden.derived), {s.id: "done" for s in STEPS})


def test_a_hidden_requirement_is_refuted_by_repair(db, clauses, variant, golden) -> None:
    start = _done(golden, {("so", "trang_thai"): "Đã duyệt"})
    target = frozenset({"golden:so:SO-0031:trang_thai"})
    assert clauses.violations(start, GRN12) == target
    found = SymbolicRepair(db, variant, clauses).search(target, GRN12, [start])
    assert found is not None and found.tables["so"][0]["trang_thai"] == "Chờ duyệt"


def test_reconciliation_over_two_receipts_cannot_be_repaired(db, clauses, variant, golden) -> None:
    two = Scenario("x", 15, 12, 0, grn_count=2)
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    tables["grn"] = [{"ma_grn": "GRN-0015", "po": "PO-0015", "sl_nhan": "6"},
                     {"ma_grn": "GRN-0016", "po": "PO-0015", "sl_nhan": "6"}]
    start = db.materialize(State(tables, {}, {s.id: "done" for s in STEPS}))
    target = frozenset({"reconcile:consistency"})
    assert "reconcile:consistency" in clauses.violations(start, two)
    assert SymbolicRepair(db, variant, clauses).search(target, two, [start]) is None


def test_a_conflicting_pair_cannot_be_repaired_without_new_violations(
    db, clauses, variant, golden
) -> None:
    extra = Clause(
        "inv:mut", "invariant", "anchored", "grn12", frozenset({"po"}), "PO đặt 18",
        {"invariant": {"id": "mut", "desc": "x", "table": "po", "where": {"ma_po": "PO-0015"},
                       "assert": {"col": "sl_dat", "eq": "18"}}},
    )
    mutant = ClauseSet([*clauses, extra], clauses.master, clauses.canon)
    start = _done(golden)
    assert mutant.violations(start, GRN12) == frozenset({"inv:mut"})
    repair = SymbolicRepair(db, variant, mutant)
    assert repair.search(frozenset({"inv:mut"}), GRN12, [start]) is None


def test_a_state_the_database_refuses_is_no_witness(db, clauses, variant, golden) -> None:
    bogus = Clause("event:BOGUS:SO-0031:AI", "golden_event", "anchored", "grn12",
                   frozenset({"evt"}), "bogus", {"event": ["BOGUS", "SO-0031", "AI"]})
    mutant = ClauseSet([*clauses, bogus], clauses.master, clauses.canon)
    target = frozenset({bogus.id})
    assert SymbolicRepair(db, variant, mutant).search(target, GRN12, [_done(golden)]) is None


class _FakeRunner:
    def __init__(self, state: State) -> None:
        self.state, self.calls = state, 0

    def run(self, variant, scenario, interp, suffix=""):
        self.calls += 1
        assert "Yêu cầu bổ sung" in suffix
        return Behaviour(f"probe{self.calls}", interp, scenario, self.state)


def test_the_probe_refutes_and_results_are_memoised(clauses, variant, golden) -> None:
    runner = _FakeRunner(_done(golden))
    interp = InterpreterId("m", "f", 0, 1)
    probe = DirectedProbe(runner, variant, clauses, [interp])
    refuter = CombinedRefuter(clauses, None, probe)
    start = Behaviour("s", interp, GRN12, _done(golden, {("so", "trang_thai"): "Đã duyệt"}))
    target = frozenset({"golden:so:SO-0031:trang_thai"})
    first = refuter.refute(target, GRN12, [start])
    again = refuter.refute(target, dataclasses.replace(GRN12, id="other"), [start])
    assert first.refuted and first.witness == "probe1"
    assert again == first and runner.calls == 1
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/active/test_refute.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.active'`.

- [ ] **Step 3: Implement**

`specdx/active/__init__.py`: empty.

`specdx/active/refute.py`:
```python
"""Refutation of a set of clauses on one scenario (spec §3.6).

A witness must satisfy every target clause and add no violation beyond those its starting
point already had. Finding one is proof (sound); failing to find one is only evidence.
Symbolic repair edits states directly but keeps scenario facts (receipts, environment
rows) untouched, and every candidate is checked against the real database, guardrails
included.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol

from tonomyx.flow.case_0031 import CASE_ID
from tonomyx.flow.handlers import SO, VINV

from specdx.db import PackDatabase
from specdx.observe.state import ERROR
from specdx.spec.clauses import PAID, Clause, ClauseSet
from specdx.spec.lifting import instantiate
from specdx.types import Behaviour, InterpreterId, RefuteResult, Scenario, State
from specdx.variant import SpecVariant

REPAIR_BASIS = "specdx repair"
REPAIR_AT = "2026-09-01 02:00:00+00"
GOAL = "Yêu cầu bổ sung: kết quả cuối cùng phải thỏa mọi điều sau: {items}."


class RunnerLike(Protocol):
    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = ""
    ) -> Behaviour: ...


def _tables(state: State) -> dict[str, list[dict[str, str]]]:
    return {t: [dict(r) for r in rows] for t, rows in state.tables.items()}


def _event(kind: str, ref: str, actor: str) -> dict[str, str]:
    return {"case_id": CASE_ID, "kind": kind, "ref": ref, "actor": actor,
            "basis": REPAIR_BASIS, "at": REPAIR_AT}


def propose(clause: Clause, state: State, scenario: Scenario) -> list[State]:
    """Candidate edits that would make this one clause true, if any edit can."""
    p = instantiate(clause.payload, scenario)
    replace = dataclasses.replace
    if clause.kind == "golden":
        tables = _tables(state)
        rows = [r for r in tables.get(p["table"], [])
                if all(r.get(k) == v for k, v in p["key"].items())]
        if not rows:
            return []
        for r in rows:
            r[p["column"]] = str(p["value"])
        return [replace(state, tables=tables)]
    if clause.kind == "invariant":
        inv = p["invariant"]
        cond = inv["assert"].get("all", inv["assert"])
        if "col" not in cond:
            return []
        if "eq" in cond:
            values = [str(cond["eq"])]
        elif "in" in cond:
            values = [str(v) for v in cond["in"]][:2]
        elif cond.get("is_blank"):
            values = [""]
        else:
            return []
        out = []
        for value in values:
            tables = _tables(state)
            hit = False
            for r in tables.get(inv["table"], []):
                where = inv.get("where") or {}
                if all(str(r.get(k, "")).strip() == str(w) for k, w in where.items()):
                    r[cond["col"]] = value
                    hit = True
            if hit:
                out.append(replace(state, tables=tables))
        return out
    if clause.kind == "golden_event":
        tables = _tables(state)
        tables.setdefault("evt", []).append(_event(*p["event"]))
        return [replace(state, tables=tables)]
    if clause.kind == "expectation":
        targets = p["members"] or [p["eq"]]
        out = []
        for value in targets:
            tables = _tables(state)
            if p["key"] == "three_way":
                rows = [r for r in tables.get("vinv", []) if r["ma_vinv"] == VINV]
                if not rows:
                    return []
                rows[0]["doi_soat"] = value
            elif p["key"] == "apr_0102":
                tables["evt"] = [e for e in tables.get("evt", [])
                                 if not (e["kind"] == "FLAGGED" and e["ref"] == SO)]
                if value == "over_cap":
                    tables["evt"].append(_event("FLAGGED", SO, "AI"))
            else:
                return []
            out.append(replace(state, tables=tables))
        return out
    if clause.kind == "blank":
        tables = _tables(state)
        for r in tables.get(p["table"], []):
            r[p["column"]] = ""
        return [replace(state, tables=tables)]
    if clause.kind == "step":
        return [replace(state, steps={**state.steps, p["step"]: "done"})]
    if clause.kind == "reconcile":
        tables = _tables(state)
        changed = False
        for r in tables.get("vinv", []):
            value = state.derived.get(f"fn_reconcile:{r['po']}")
            if value and value != ERROR:
                r["doi_soat"] = value
                changed = True
        return [replace(state, tables=tables)] if changed else []
    if clause.kind == "guard_pay":
        tables = _tables(state)
        for r in tables.get("pay", []):
            if r["trang_thai"] == PAID:
                r["trang_thai"] = "Chờ duyệt"
        return [replace(state, tables=tables)]
    return []


class SymbolicRepair:
    def __init__(
        self, db: PackDatabase, variant: SpecVariant, clauses: ClauseSet, *, budget: int = 200
    ) -> None:
        self.db, self.variant, self.clauses, self.budget = db, variant, clauses, budget

    def _cost(
        self, state: State, scenario: Scenario, targets: frozenset[str], allowed: frozenset[str]
    ) -> int:
        bad = self.clauses.violations(state, scenario)
        return len(bad & targets) + len(bad - allowed - targets)

    def search(
        self, targets: frozenset[str], scenario: Scenario, starts: Sequence[State]
    ) -> State | None:
        self.db.use(self.variant.pack)
        spent = 0
        ordered = sorted(starts, key=lambda s: len(self.clauses.violations(s, scenario) & targets))
        for start in ordered:
            allowed = self.clauses.violations(start, scenario) - targets
            current = start
            cost = self._cost(current, scenario, targets, allowed)
            while spent < self.budget:
                if cost == 0:
                    spent += 1
                    real = self.db.materialize(current)
                    if real is not None and self._cost(real, scenario, targets, allowed) == 0:
                        return real
                    break
                bad = self.clauses.violations(current, scenario)
                focus = sorted((bad & targets) | (bad - allowed))
                proposals = [
                    s for cid in focus for s in propose(self.clauses[cid], current, scenario)
                ]
                best: State | None = None
                best_cost = cost
                for candidate in proposals:
                    if spent >= self.budget:
                        break
                    spent += 1
                    c = self._cost(candidate, scenario, targets, allowed)
                    if c < best_cost:
                        best, best_cost = candidate, c
                if best is None:
                    break
                current, cost = best, best_cost
        return None


class DirectedProbe:
    def __init__(
        self,
        runner: RunnerLike,
        variant: SpecVariant,
        clauses: ClauseSet,
        interpreters: Sequence[InterpreterId],
        *,
        attempts: int = 3,
        width: int = 3,
    ) -> None:
        self.runner, self.variant, self.clauses = runner, variant, clauses
        self.interpreters = list(interpreters)
        self.attempts, self.width = attempts, width

    def search(
        self, targets: frozenset[str], scenario: Scenario, allowed: frozenset[str]
    ) -> Behaviour | None:
        suffix = GOAL.format(items="; ".join(self.clauses[c].description for c in sorted(targets)))
        for interp in self.interpreters[: self.width]:
            for attempt in range(self.attempts):
                b = self.runner.run(
                    self.variant, scenario, dataclasses.replace(interp, seed=100 + attempt),
                    suffix=suffix,
                )
                if b.status != "ok":
                    continue
                bad = self.clauses.violations(b.state, scenario)
                if not (bad & targets) and bad <= allowed:
                    return b
        return None


class CombinedRefuter:
    """Repair first (cheap), then a directed probe. Memoised per scenario and targets."""

    def __init__(
        self, clauses: ClauseSet, repair: SymbolicRepair | None, probe: DirectedProbe | None
    ) -> None:
        self.clauses, self.repair, self.probe = clauses, repair, probe
        self._memo: dict[tuple[tuple[int, ...], frozenset[str]], RefuteResult] = {}

    def refute(
        self, targets: frozenset[str], scenario: Scenario, behaviours: Sequence[Behaviour]
    ) -> RefuteResult:
        key = (scenario.signature(), targets)
        if key in self._memo:
            return self._memo[key]
        starts = [b.state for b in behaviours if b.status == "ok"]
        allowed: frozenset[str] = frozenset().union(
            *(self.clauses.violations(s, scenario) for s in starts)
        ) - targets
        attempts = 0
        result: RefuteResult | None = None
        if self.repair is not None and starts:
            attempts += 1
            if self.repair.search(targets, scenario, starts) is not None:
                result = RefuteResult(True, "symbolic-repair", attempts)
        if result is None and self.probe is not None:
            found = self.probe.search(targets, scenario, allowed)
            attempts += self.probe.width * self.probe.attempts
            if found is not None:
                result = RefuteResult(True, found.run_key, attempts)
        if result is None:
            result = RefuteResult(False, None, attempts)
        self._memo[key] = result
        return result
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/active/test_refute.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/active tests/active
git commit -m "feat: sound refutation by symbolic repair and directed probes"
```

---

### Task 14: Typed diagnosis

**Files:**
- Create: `specdx/infer/scores.py`
- Test: `tests/infer/test_scores.py`

**Interfaces:**
- Consumes: `soft_hitting_sets` (Task 12), `Posterior` and `fit` (Task 11), `ItemView` (Task 10), `Refuter` (Task 2).
- Produces:
  - `run_weights(views, posterior) -> dict[str, float]`.
  - `merge_defects(defects) -> list[Defect]`.
  - `diagnose(behaviours, violated, views, posterior, refuter, diff, *, tau=0.05, ambig_threshold=0.5, weak_threshold=0.5, weak_min_runs=2) -> list[Defect]`, sorted by score (descending).
- Rules (spec §3.3–§3.4):
  - A singleton that refutation refutes is **hidden**. Any unrefuted hitting set is a **contradiction**. A refuted set of size ≥ 2 is dropped.
  - **Ambiguity:** P(|A| ≥ 2) ≥ threshold, located at the fields where the top two member classes differ.
  - **Weak:** the MAP admissible classes all pass every relevant clause, and some other class that also passes (at least `weak_min_runs` runs) is likely not admissible. It is located at the fields where it departs from the top admissible class.

- [ ] **Step 1: Write the failing tests**

`tests/infer/test_scores.py`:
```python
from specdx.infer.items import ItemView
from specdx.infer.latent import Observation, fit
from specdx.infer.scores import diagnose
from specdx.types import Behaviour, InterpreterId, RefuteResult, Scenario, State

GOOD = ("g1", "g2", "g3", "g4")
BAD = ("b1", "b2")
S = Scenario("s1", 15, 12, 0)


def _proj(name: str):
    return (("x", ((name, (("v", name),)),)),)


def _diff(a, b):
    return frozenset({"x.v"}) if a != b else frozenset()


class _Refuter:
    def __init__(self, refutes: bool) -> None:
        self.refutes = refutes

    def refute(self, targets, scenario, behaviours):
        return RefuteResult(self.refutes, "w" if self.refutes else None, 1)


def _world(outputs):
    """outputs: model -> list of (klass, violated set) per run, on one item."""
    behaviours, violated, views = [], {}, []
    for model, runs in outputs.items():
        for i, (klass, bad) in enumerate(runs):
            key = f"{model}|{i}"
            behaviours.append(Behaviour(key, InterpreterId(model, model[0], 0, i), S, State({})))
            violated[key] = frozenset(bad)
            views.append(ItemView("s1|step", key, model, model[0], klass, frozenset(bad),
                                  _proj(klass)))
    # Calibration items: competent models agree, each weak model errs in its own way.
    clean = [ItemView(f"c{j}|step", f"{m}|c{j}", m, m[0], "OK" if m in GOOD else f"E{j}-{m}",
                      frozenset(), _proj("OK")) for j in range(6) for m in outputs]
    obs = [Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views + clean]
    return behaviours, violated, views + clean, fit(obs)


def test_a_shared_violation_that_repair_refutes_is_hidden() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {f"n{i}"}) for i in range(3)]
                                                    for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Refuter(True), _diff)
    assert defects[0].type == "hidden" and defects[0].location == ("c",)


def test_the_same_pattern_unrefuted_is_a_contradiction() -> None:
    outputs = {m: [("K", {"c"})] * 3 for m in GOOD} | {m: [(f"E{i}", {"c", f"n{i}"})
                                                           for i in range(3)] for m in BAD}
    b, v, views, post = _world(outputs)
    defects = diagnose(b, v, views, post, _Refuter(False), _diff)
    assert defects[0].type == "contradiction" and defects[0].location == ("c",)


def test_competent_models_splitting_cleanly_is_ambiguity() -> None:
    outputs = {m: [("X", set()), ("Y", set()), ("X", set()), ("Y", set())] for m in GOOD}
    outputs |= {m: [(f"E{i}", {f"n{i}"}) for i in range(4)] for m in BAD}
    b, v, views, post = _world(outputs)
    found = [d for d in diagnose(b, v, views, post, _Refuter(True), _diff) if d.type == "ambiguity"]
    assert found and found[0].location == ("x.v",) and found[0].note == "s1|step"


def test_a_clean_deviant_class_beside_a_clean_admissible_one_is_weak() -> None:
    outputs = {m: [("A", set())] * 3 for m in GOOD} | {m: [("W", set())] * 3 for m in BAD}
    b, v, views, post = _world(outputs)
    found = [d for d in diagnose(b, v, views, post, _Refuter(True), _diff) if d.type == "weak"]
    assert found and found[0].location == ("x.v",)


def test_no_weak_when_the_admissible_class_itself_violates() -> None:
    outputs = {m: [("A", {"c"})] * 3 for m in GOOD} | {m: [("W", set())] * 3 for m in BAD}
    b, v, views, post = _world(outputs)
    assert not [d for d in diagnose(b, v, views, post, _Refuter(True), _diff) if d.type == "weak"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/infer/test_scores.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.infer.scores'`.

- [ ] **Step 3: Implement**

`specdx/infer/scores.py`:
```python
"""From behaviours and a posterior to typed, located defects (spec §3.3–§3.4)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

from specdx.infer.hitting import soft_hitting_sets
from specdx.infer.items import ItemView
from specdx.infer.latent import ItemPosterior, Posterior
from specdx.spec.columns import Projection
from specdx.types import Behaviour, Defect, Refuter

Diff = Callable[[Projection, Projection], frozenset[str]]


def run_weights(views: Iterable[ItemView], posterior: Posterior) -> dict[str, float]:
    """How faithful each run looks, averaged over its items."""
    acc: dict[str, list[float]] = defaultdict(list)
    for v in views:
        item = posterior.items.get(v.item)
        if item is not None:
            acc[v.run_key].append(item.responsibility.get(v.run_key, 0.0))
    return {k: sum(r) / len(r) for k, r in acc.items()}


def merge_defects(defects: Iterable[Defect]) -> list[Defect]:
    merged: dict[tuple[str, tuple[str, ...]], Defect] = {}
    for d in defects:
        old = merged.get(d.key)
        if old is None:
            merged[d.key] = d
            continue
        best = d if d.score > old.score else old
        merged[d.key] = Defect(
            d.type, d.location,
            tuple(sorted(set(old.scenarios) | set(d.scenarios))),
            tuple(sorted(set(old.witnesses) | set(d.witnesses))),
            best.score, best.note,
        )
    return sorted(merged.values(), key=lambda d: (-d.score, d.type, d.location))


def _scenario_of(item: str) -> str:
    return item.split("|", 1)[0]


def _weak(ip: ItemPosterior, views: Sequence[ItemView], min_runs: int) -> tuple[str, float] | None:
    violated = {v.klass: v.violated for v in views}
    if any(violated.get(k) for k in ip.map_set):
        return None  # the admissible output itself fails a check: that is not a weak check
    counts = Counter(v.klass for v in views)
    best: tuple[str, float] | None = None
    for klass in {v.klass for v in views if not v.violated}:
        if counts[klass] < min_runs or klass in ip.map_set:
            continue
        score = (
            sum(p for s, p in ip.set_posterior.items() if klass not in s)
            if klass in ip.classes else 1.0
        )
        if best is None or score > best[1]:
            best = (klass, score)
    return best


def diagnose(
    behaviours: Sequence[Behaviour],
    violated: Mapping[str, frozenset[str]],
    views: Sequence[ItemView],
    posterior: Posterior,
    refuter: Refuter,
    diff: Diff,
    *,
    tau: float = 0.05,
    ambig_threshold: float = 0.5,
    weak_threshold: float = 0.5,
    weak_min_runs: int = 2,
) -> list[Defect]:
    defects: list[Defect] = []
    weights = run_weights(views, posterior)
    groups: dict[str, list[Behaviour]] = defaultdict(list)
    for b in behaviours:
        if b.status == "ok":
            groups[b.scenario.id].append(b)
    for sid, group in groups.items():
        sets = [(violated[b.run_key], weights.get(b.run_key, 0.0)) for b in group]
        for hitting, support in soft_hitting_sets(sets, tau=tau):
            result = refuter.refute(hitting, group[0].scenario, group)
            witnesses = tuple(b.run_key for b in group if violated[b.run_key] & hitting)
            location = tuple(sorted(hitting))
            if not result.refuted:
                defects.append(Defect("contradiction", location, (sid,), witnesses, support,
                                      f"unrefuted after {result.attempts} attempts"))
            elif len(hitting) == 1:
                defects.append(Defect("hidden", location, (sid,), witnesses, support,
                                      f"refuted by {result.witness}"))
    by_item: dict[str, list[ItemView]] = defaultdict(list)
    for v in views:
        by_item[v.item].append(v)
    for item, ip in posterior.items.items():
        mine = by_item.get(item, [])
        projection = {v.klass: v.projection for v in mine}
        ranked = sorted(ip.p_member, key=lambda k: -ip.p_member[k])
        if ip.p_ambiguous >= ambig_threshold and len(ranked) >= 2:
            fields = diff(projection[ranked[0]], projection[ranked[1]])
            if fields:
                witnesses = tuple(v.run_key for v in mine if v.klass in ranked[:2])
                defects.append(Defect("ambiguity", tuple(sorted(fields)), (_scenario_of(item),),
                                      witnesses, ip.p_ambiguous, note=item))
        weak = _weak(ip, mine, weak_min_runs)
        if weak is not None and weak[1] >= weak_threshold and ranked:
            klass, score = weak
            fields = diff(projection[klass], projection[ranked[0]])
            if fields:
                witnesses = tuple(v.run_key for v in mine if v.klass == klass)
                defects.append(Defect("weak", tuple(sorted(fields)), (_scenario_of(item),),
                                      witnesses, score, note=item))
    return merge_defects(defects)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/infer/test_scores.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/infer/scores.py tests/infer/test_scores.py
git commit -m "feat: typed diagnosis of contradictions, hidden requirements, ambiguity and weak checks"
```

---
### Task 15: Scenario selection and ddmin

**Files:**
- Create: `specdx/active/search.py`, `specdx/active/ddmin.py`
- Test: `tests/active/test_search.py`, `tests/active/test_ddmin.py`

**Interfaces:**
- Consumes: `cell_of` (Task 2), `ItemView` (Task 10), `TellCorpus` (Task 7).
- Produces:
  - `binary_kl(p, q)` and `information_gain(before, after) -> float`.
  - `NormalGamma`, with `.update(x)` and `.sample(rng)`.
  - `CellBandit(cap_pct)`, with `.update(scenario, reward)`, `.choose(candidates, visited, rng) -> Scenario | None` and `.exhausted(candidates, visited, eps) -> bool`.
  - `should_escalate(pilot_views, pilot_sets, seen_sets) -> bool`.
  - `ddmin(items, fails, *, max_tests=None) -> list` and `localize_ambiguity(corpus, candidates, split_persists, *, max_tests=None) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/active/test_search.py`:
```python
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
```

`tests/active/test_ddmin.py`:
```python
from specdx.active.ddmin import ddmin, localize_ambiguity
from specdx.spec.tell import Chunk, Sentence, TellCorpus


def test_ddmin_finds_the_minimal_failing_subset() -> None:
    assert sorted(ddmin(list(range(10)), lambda s: {3, 7} <= set(s))) == [3, 7]


def test_ddmin_returns_nothing_when_the_whole_does_not_fail() -> None:
    assert ddmin(list(range(5)), lambda s: False) == []


def test_ddmin_stops_at_the_test_budget_with_a_failing_superset() -> None:
    result = ddmin(list(range(64)), lambda s: {3, 7} <= set(s), max_tests=3)
    assert {3, 7} <= set(result)


def test_localisation_names_the_sentence_that_keeps_the_split() -> None:
    sentences = tuple(Sentence(f"A:{i:02d}", f"s{i}") for i in range(6))
    corpus = TellCorpus([Chunk("A", "KB", sentences)])
    found = localize_ambiguity(
        corpus, corpus.sentence_ids(), lambda c: "A:02" in c.sentence_ids()
    )
    assert found == ["A:02"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/active/test_search.py tests/active/test_ddmin.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.active.search'`.

- [ ] **Step 3: Implement**

`specdx/active/search.py`:
```python
"""Information-directed scenario selection with sequential escalation (spec §3.5)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from specdx.infer.items import ItemView
from specdx.scenarios.space import cell_of
from specdx.types import Scenario

_EDGE = 1e-6


def binary_kl(p: float, q: float) -> float:
    p = min(max(p, _EDGE), 1 - _EDGE)
    q = min(max(q, _EDGE), 1 - _EDGE)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def information_gain(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    """Realised information: how far the defect scores moved, summed over hypotheses."""
    return sum(binary_kl(after.get(k, 0.0), before.get(k, 0.0)) for k in set(before) | set(after))


@dataclass
class NormalGamma:
    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0

    def update(self, x: float) -> None:
        mu, kappa = self.mu, self.kappa
        self.mu = (kappa * mu + x) / (kappa + 1)
        self.kappa = kappa + 1
        self.alpha += 0.5
        self.beta += kappa * (x - mu) ** 2 / (2 * (kappa + 1))

    def sample(self, rng: random.Random) -> float:
        tau = rng.gammavariate(self.alpha, 1 / self.beta)
        return rng.gauss(self.mu, 1 / math.sqrt(self.kappa * tau))


class CellBandit:
    def __init__(self, cap_pct: int) -> None:
        self.cap_pct = cap_pct
        self.cells: dict[str, NormalGamma] = defaultdict(NormalGamma)
        self.tried: set[str] = set()

    def update(self, scenario: Scenario, reward: float) -> None:
        cell = cell_of(scenario, self.cap_pct)
        self.cells[cell].update(reward)
        self.tried.add(cell)

    def _open(
        self, candidates: Iterable[Scenario], visited: set[tuple[int, ...]]
    ) -> dict[str, list[Scenario]]:
        open_: dict[str, list[Scenario]] = defaultdict(list)
        for s in candidates:
            if s.signature() not in visited:
                open_[cell_of(s, self.cap_pct)].append(s)
        return open_

    def choose(
        self, candidates: Sequence[Scenario], visited: set[tuple[int, ...]], rng: random.Random
    ) -> Scenario | None:
        open_ = self._open(candidates, visited)
        if not open_:
            return None
        untried = sorted(c for c in open_ if c not in self.tried)
        if untried:
            cell = rng.choice(untried)
        else:
            draws = {c: self.cells[c].sample(rng) for c in sorted(open_)}
            cell = max(draws, key=lambda c: draws[c])
        return rng.choice(open_[cell])

    def exhausted(
        self, candidates: Sequence[Scenario], visited: set[tuple[int, ...]], eps: float
    ) -> bool:
        open_ = self._open(candidates, visited)
        if not open_:
            return True
        if any(c not in self.tried for c in open_):
            return False
        return max(self.cells[c].mu for c in open_) < eps


def should_escalate(
    pilot_views: Sequence[ItemView],
    pilot_sets: Sequence[frozenset[str]],
    seen_sets: set[frozenset[str]],
) -> bool:
    """Run the full committee only when the pilot shows something new or disagrees."""
    if any(s not in seen_sets for s in pilot_sets):
        return True
    classes: dict[str, set[str]] = defaultdict(set)
    for v in pilot_views:
        classes[v.item].add(v.klass)
    return any(len(k) >= 2 for k in classes.values())
```

`specdx/active/ddmin.py`:
```python
"""Delta debugging (Zeller & Hildebrandt 2002) to localise ambiguity in T (spec §3.7)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from specdx.spec.tell import TellCorpus


def ddmin[T](
    items: Sequence[T], fails: Callable[[Sequence[T]], bool], *, max_tests: int | None = None
) -> list[T]:
    """A minimal subset on which `fails` still holds; anytime under max_tests."""
    tests = 0

    def check(subset: Sequence[T]) -> bool:
        nonlocal tests
        tests += 1
        return fails(subset)

    def out_of_budget() -> bool:
        return max_tests is not None and tests >= max_tests

    current = list(items)
    if not current or not check(current):
        return []
    n = 2
    while len(current) >= 2 and not out_of_budget():
        size = math.ceil(len(current) / n)
        chunks = [current[i:i + size] for i in range(0, len(current), size)]
        reduced = False
        for chunk in chunks:
            if out_of_budget():
                break
            if check(chunk):
                current, n, reduced = chunk, 2, True
                break
        if not reduced:
            for chunk in chunks:
                if out_of_budget():
                    break
                complement = [x for x in current if x not in chunk]
                if complement and check(complement):
                    current, n, reduced = complement, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(current):
                break
            n = min(len(current), 2 * n)
    return current


def localize_ambiguity(
    corpus: TellCorpus,
    candidates: Sequence[str],
    split_persists: Callable[[TellCorpus], bool],
    *,
    max_tests: int | None = None,
) -> list[str]:
    """The fewest candidate sentences that, kept, still let interpreters split."""
    pool = list(candidates)

    def fails(keep: Sequence[str]) -> bool:
        kept = set(keep)
        return split_persists(corpus.without([s for s in pool if s not in kept]))

    return ddmin(pool, fails, max_tests=max_tests)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/active/test_search.py tests/active/test_ddmin.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/active/search.py specdx/active/ddmin.py tests/active/test_search.py tests/active/test_ddmin.py
git commit -m "feat: Thompson scenario selection, escalation rule and ddmin localisation"
```

---

### Task 16: The SpecDx loop

**Files:**
- Create: `specdx/active/loop.py`
- Test: `tests/active/test_loop.py`

**Interfaces:**
- Consumes:
  - `fit`, `Observation` (Task 11); `diagnose` (Task 14);
  - `CellBandit`, `information_gain`, `should_escalate` (Task 15);
  - `committee`, `pilot` (Task 9).
- Produces:
  - `SpecDxConfig(models_by_family, variants=(0,), seeds=(1, 2, 3), budget=300, tau=0.05, eps=0.01, ambig_threshold=0.5, weak_threshold=0.5, k_max=6, p_single=0.9, rng_seed=20261006, synthesize=True, localize=True, localize_tests=40)`.
  - `LoopContext(runner, lens, canon, candidates, cap_pct, refuter_factory, localizer=None)`.
  - `Report(variant, defects, runs, executed, scenarios, theta, behaviours)`, with `.to_json()`.
  - `ranked_interpreters(models_by_family, theta, variants) -> list[InterpreterId]` and `run_specdx(variant, ctx, config) -> Report`.
- `runs` counts pilot and committee runs against `budget`. `executed` counts every run actually executed, probes and ddmin included.

- [ ] **Step 1: Write the failing tests**

`tests/active/test_loop.py`:
```python
from pathlib import Path

from specdx.active.loop import LoopContext, SpecDxConfig, run_specdx
from specdx.infer.items import ItemView
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, RefuteResult, Scenario, State
from specdx.variant import SpecVariant

CANON = {"discount8": Scenario("discount8", 15, 15, 8), "grn12": Scenario("grn12", 15, 12, 0),
         "happy": Scenario("happy", 15, 15, 0)}
MODELS = {"qwen": ("q1", "q2"), "llama": ("l1",)}
GOOD = ("q1", "l1")
VARIANT = SpecVariant("toy", Path("."), TellCorpus([]))


class FakeRunner:
    def __init__(self, plan) -> None:
        self.plan, self.executed, self.calls = plan, 0, []

    def run(self, variant, scenario, interp, suffix=""):
        self.executed += 1
        self.calls.append(scenario.id)
        klass, bad = self.plan(scenario, interp)
        return Behaviour(f"{scenario.id}|{interp.key}", interp, scenario,
                         State({}, {"k": klass, "v": ",".join(sorted(bad))}, {}))


class FakeLens:
    """One item under test plus five calibration items, as a real run has eleven steps:
    competence is learnt from the steps where competent models agree."""

    def violations(self, b):
        return frozenset(x for x in b.state.derived["v"].split(",") if x)

    def views(self, b, violated):
        k = b.state.derived["k"]
        model, family = b.interpreter.model, b.interpreter.family
        out = [ItemView(f"{b.scenario.id}|step", b.run_key, model, family, k, violated,
                        (("x", ((k, ()),)),))]
        for j in range(5):
            cal = "OK" if model in GOOD else f"E{j}-{b.run_key}"
            out.append(ItemView(f"{b.scenario.id}|cal{j}", b.run_key, model, family, cal,
                                frozenset(), (("x", ((cal, ()),)),)))
        return out

    def diff(self, a, b):
        return frozenset({"x.k"}) if a != b else frozenset()


class AlwaysRefute:
    def refute(self, targets, scenario, behaviours):
        return RefuteResult(True, "w", 1)


def _hidden_plan(scenario, interp):
    good = interp.model in GOOD
    if scenario.id == "grn12":
        return ("K", {"c1"}) if good else (f"E{interp.seed}", {f"n{interp.seed}"})
    return ("OK", set()) if good else (f"E{interp.seed}-{scenario.id}", {"noise"})


def _ctx(runner, localizer=None):
    return LoopContext(runner, FakeLens(), CANON, list(CANON.values()), 5,
                       lambda interps: AlwaysRefute(), localizer)


def test_a_planted_hidden_requirement_is_reported_first() -> None:
    report = run_specdx(VARIANT, _ctx(FakeRunner(_hidden_plan)),
                        SpecDxConfig(MODELS, budget=100, synthesize=False))
    top = report.defects[0]
    assert (top.type, top.location) == ("hidden", ("c1",)) and "grn12" in top.scenarios
    assert report.theta["q1"] > report.theta["q2"]


def test_an_agreeing_pilot_on_known_ground_does_not_escalate() -> None:
    runner = FakeRunner(_hidden_plan)
    run_specdx(VARIANT, _ctx(runner), SpecDxConfig(MODELS, budget=100, synthesize=False))
    assert runner.calls.count("discount8") == 9 and runner.calls.count("grn12") == 9
    assert runner.calls.count("happy") == 2


def test_the_budget_is_respected() -> None:
    runner = FakeRunner(_hidden_plan)
    report = run_specdx(VARIANT, _ctx(runner), SpecDxConfig(MODELS, budget=5, synthesize=False))
    assert report.runs <= 5 and runner.executed <= 5


def _split_plan(scenario, interp):
    good = interp.model in GOOD
    if scenario.id == "grn12" and good:
        # Two competent models read the same text two ways, so the pilot already disagrees.
        return ("X" if interp.model == "q1" else "Y", set())
    return ("OK", set()) if good else (f"E{interp.seed}-{scenario.id}", {"n"})


def test_ambiguity_locations_are_replaced_by_the_localiser() -> None:
    report = run_specdx(
        VARIANT, _ctx(FakeRunner(_split_plan), lambda v, d, i: ("A:01",)),
        SpecDxConfig(MODELS, budget=200, synthesize=False),
    )
    assert any(d.type == "ambiguity" and d.location == ("A:01",) for d in report.defects)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/active/test_loop.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.active.loop'`.

- [ ] **Step 3: Implement**

`specdx/active/loop.py`:
```python
"""The SpecDx main loop (spec §3.8)."""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from specdx.active.search import CellBandit, information_gain, should_escalate
from specdx.infer.items import ItemView
from specdx.infer.latent import Observation, Posterior, fit
from specdx.infer.scores import diagnose
from specdx.interpreters.population import committee, pilot
from specdx.spec.columns import Projection
from specdx.types import Behaviour, Defect, InterpreterId, Refuter, Scenario
from specdx.variant import SpecVariant


class RunnerLike(Protocol):
    executed: int

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = ""
    ) -> Behaviour: ...


class LensLike(Protocol):
    def violations(self, behaviour: Behaviour) -> frozenset[str]: ...

    def views(self, behaviour: Behaviour, violated: frozenset[str]) -> list[ItemView]: ...

    def diff(self, a: Projection, b: Projection) -> frozenset[str]: ...


Localizer = Callable[[SpecVariant, Defect, Sequence[InterpreterId]], tuple[str, ...]]


@dataclass(frozen=True)
class SpecDxConfig:
    models_by_family: dict[str, tuple[str, ...]]
    variants: tuple[int, ...] = (0,)
    seeds: tuple[int, ...] = (1, 2, 3)
    budget: int = 300
    tau: float = 0.05
    eps: float = 0.01
    ambig_threshold: float = 0.5
    weak_threshold: float = 0.5
    k_max: int = 6
    p_single: float = 0.9
    rng_seed: int = 20261006
    synthesize: bool = True
    localize: bool = True
    localize_tests: int = 40


@dataclass
class LoopContext:
    runner: RunnerLike
    lens: LensLike
    canon: dict[str, Scenario]
    candidates: list[Scenario]
    cap_pct: int
    refuter_factory: Callable[[Sequence[InterpreterId]], Refuter]
    localizer: Localizer | None = None


@dataclass
class Report:
    variant: str
    defects: list[Defect]
    runs: int
    executed: int
    scenarios: list[str]
    theta: dict[str, float]
    behaviours: list[Behaviour] = field(default_factory=list, repr=False)

    def to_json(self) -> str:
        return json.dumps(
            {"variant": self.variant, "defects": [asdict(d) for d in self.defects],
             "runs": self.runs, "executed": self.executed, "scenarios": self.scenarios,
             "theta": self.theta},
            ensure_ascii=False, indent=2,
        )


def ranked_interpreters(
    models_by_family: dict[str, tuple[str, ...]],
    theta: dict[str, float],
    variants: tuple[int, ...],
) -> list[InterpreterId]:
    """Every model once, most competent first: who refutes and who localises."""
    models = [(m, f) for f, ms in models_by_family.items() for m in ms]
    models.sort(key=lambda mf: -theta.get(mf[0], 0.0))
    return [InterpreterId(m, f, variants[0], 1) for m, f in models]


def run_specdx(variant: SpecVariant, ctx: LoopContext, config: SpecDxConfig) -> Report:
    rng = random.Random(config.rng_seed)
    bandit = CellBandit(ctx.cap_pct)
    queue = [ctx.canon[name] for name in sorted(ctx.canon)]
    visited: set[tuple[int, ...]] = set()
    behaviours: list[Behaviour] = []
    violated: dict[str, frozenset[str]] = {}
    views: list[ItemView] = []
    seen_sets: set[frozenset[str]] = set()
    posterior = Posterior({}, {}, {})
    defects: list[Defect] = []
    previous: dict[str, float] = {}
    order: list[str] = []
    runs = 0
    executed_before = ctx.runner.executed
    members = committee(config.models_by_family, config.variants, config.seeds)

    while runs < config.budget:
        if queue:
            scenario = queue.pop(0)
        elif config.synthesize and not bandit.exhausted(ctx.candidates, visited, config.eps):
            chosen = bandit.choose(ctx.candidates, visited, rng)
            if chosen is None:
                break
            scenario = chosen
        else:
            break
        visited.add(scenario.signature())
        order.append(scenario.id)

        pilot_ids = pilot(config.models_by_family, posterior.theta or None)
        fresh: list[Behaviour] = []
        for interp in pilot_ids:
            if runs >= config.budget:
                break
            fresh.append(ctx.runner.run(variant, scenario, interp))
            runs += 1
        sets = [ctx.lens.violations(b) for b in fresh]
        pilot_views = [v for b, s in zip(fresh, sets, strict=True) for v in ctx.lens.views(b, s)]
        ok_sets = [s for b, s in zip(fresh, sets, strict=True) if b.status == "ok"]
        if should_escalate(pilot_views, ok_sets, seen_sets):
            for interp in members:
                if interp in pilot_ids:
                    continue
                if runs >= config.budget:
                    break
                fresh.append(ctx.runner.run(variant, scenario, interp))
                runs += 1

        for b in fresh:
            v = ctx.lens.violations(b)
            violated[b.run_key] = v
            if b.status == "ok":
                seen_sets.add(v)
            behaviours.append(b)
            views.extend(ctx.lens.views(b, v))
        posterior = fit(
            [Observation(v.item, v.run_key, v.model, v.family, v.klass) for v in views],
            k_max=config.k_max, p_single=config.p_single,
        )
        refuter = ctx.refuter_factory(
            ranked_interpreters(config.models_by_family, posterior.theta, config.variants)
        )
        defects = diagnose(
            behaviours, violated, views, posterior, refuter, ctx.lens.diff,
            tau=config.tau, ambig_threshold=config.ambig_threshold,
            weak_threshold=config.weak_threshold,
        )
        scores = {f"{d.type}|{'|'.join(d.location)}": d.score for d in defects}
        bandit.update(scenario, information_gain(previous, scores))
        previous = scores

    if config.localize and ctx.localizer is not None:
        interps = ranked_interpreters(config.models_by_family, posterior.theta, config.variants)
        localised: list[Defect] = []
        for d in defects:
            if d.type == "ambiguity":
                sentences = ctx.localizer(variant, d, interps)
                if sentences:
                    d = dataclasses.replace(d, location=tuple(sentences))
            localised.append(d)
        defects = localised

    return Report(variant.name, defects, runs, ctx.runner.executed - executed_before, order,
                  dict(posterior.theta), behaviours)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/active/test_loop.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/active/loop.py tests/active/test_loop.py
git commit -m "feat: SpecDx loop with pilot escalation, posterior refits and localisation"
```

---

### Task 17: Specification mutation operators

**Files:**
- Create: `specdx/mutate/__init__.py`, `specdx/mutate/pack_copy.py`, `specdx/mutate/operators.py`
- Test: `tests/mutate/__init__.py`, `tests/mutate/test_pack_copy.py`, `tests/mutate/test_operators.py`

**Interfaces:**
- Consumes: `SpecVariant` (Task 7), `TellCorpus.build` (Task 7), `ClauseSet` (Task 6), `PackDatabase` (Task 3).
- Produces:
  - `copy_pack(src, dst) -> Path`, `regenerate_sql(pack)`, `csv_path(pack, table)`, `edit_csv_rows(pack, table, fn)`, `edit_json(pack, rel, fn)`, `replace_text(pack, rel, old, new) -> bool` and `append_text(pack, rel, text)`.
  - `OPERATOR_TYPES`, `GroundTruth(mutant_id, type, location, location_kind)` and `Mutant(id, operator, site, variant, truth)`.
  - `all_mutants(base, pdf, work_dir) -> list[Mutant]` (36 mutants), `dev_split(mutants, per_type, seed) -> list[Mutant]` and `witness_state(mutant, golden) -> State | None`.
- The `location_kind` values are `"clause"`, `"conflict"`, `"field"`, `"sentence"` and `"chunk"` (spec §4.6, §11).

- [ ] **Step 1: Write the failing tests**

`tests/mutate/__init__.py`: empty.

`tests/mutate/test_pack_copy.py`:
```python
from specdx.config import BASE_PACK
from specdx.mutate.pack_copy import copy_pack, csv_path, edit_csv_rows, regenerate_sql


def test_an_identity_csv_edit_leaves_every_file_byte_identical(tmp_path) -> None:
    pack = copy_pack(BASE_PACK, tmp_path / "pack")
    for path in sorted((pack / "data").glob("*.csv")):
        before = path.read_bytes()
        table = path.stem.split("_", 1)[1] if path.stem[0].isdigit() else path.stem
        edit_csv_rows(pack, table, lambda rows: rows)
        assert path.read_bytes() == before, path.name


def test_a_csv_edit_reaches_the_regenerated_seed(tmp_path) -> None:
    pack = copy_pack(BASE_PACK, tmp_path / "pack")

    def change(rows):
        for r in rows:
            if r["ma_vinv"] == "VINV-0015":
                r["trang_thai"] = "Nháp"
        return rows

    edit_csv_rows(pack, "vinv", change)
    regenerate_sql(pack)
    seed = (pack / "targets" / "postgres" / "seed_operational.sql").read_text(encoding="utf-8")
    assert "'Nháp'" in seed and csv_path(pack, "vinv").name == "13_vinv.csv"
```

`tests/mutate/test_operators.py`:
```python
import pytest
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.mutate.operators import all_mutants, dev_split, witness_state
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet
from specdx.spec.tell import TellCorpus
from specdx.types import Scenario, State
from specdx.variant import SpecVariant

STEPS = load_work_orders()
GRN12 = Scenario("grn12", 15, 12, 0)


@pytest.fixture(scope="module")
def mutants(pack, tmp_path_factory):
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))
    return {m.id: m for m in all_mutants(base, [], tmp_path_factory.mktemp("work"))}


@pytest.fixture(scope="module")
def base_clauses(pack) -> ClauseSet:
    return ClauseSet.from_pack(pack, STEPS)


def _clauses(m) -> ClauseSet:
    return ClauseSet.from_pack(CanonPack(m.variant.pack), STEPS)


def _done(golden: State) -> State:
    return State(golden.tables, golden.derived, {s.id: "done" for s in STEPS})


def test_there_are_thirty_six_mutants_and_a_stratified_dev_split(mutants) -> None:
    assert len(mutants) == 36
    dev = dev_split(list(mutants.values()), 5, 20261006)
    assert len(dev) == 20
    assert sorted({m.truth.type for m in dev}) == ["ambiguity", "contradiction", "hidden", "weak"]


@pytest.mark.db
def test_hidden_and_conflict_mutants_are_witnessed_by_the_canon_state(mutants, golden) -> None:
    for m in mutants.values():
        if m.operator not in ("C1", "H1", "H2"):
            continue
        v = _clauses(m).violations(_done(witness_state(m, golden)), GRN12)
        if m.operator == "C1":
            assert v == {f"inv:mut-c1-{m.site}"}, m.id
        else:
            assert v == m.truth.location, m.id


@pytest.mark.db
def test_weak_mutants_let_a_t_violating_state_pass(mutants, base_clauses, golden) -> None:
    for m in mutants.values():
        if m.operator not in ("W1", "W2"):
            continue
        witness = _done(witness_state(m, golden))
        assert _clauses(m).violations(witness, GRN12) == frozenset(), m.id
        assert base_clauses.violations(witness, GRN12), m.id


@pytest.mark.db
def test_guardrail_mutants_refuse_the_canon_state(db, mutants, golden) -> None:
    for m in mutants.values():
        if m.operator != "C2":
            continue
        db.use(m.variant.pack)
        assert db.materialize(golden) is None, m.id
    db.use(BASE_PACK)


def test_tell_mutants_change_the_corpus_and_the_kb(mutants) -> None:
    a1 = mutants["A1-00"]
    assert a1.variant.corpus.find("**mức tối đa 5%**.") == []
    assert a1.truth.location == frozenset({"KB:KB-0003"})
    kb = (a1.variant.pack / "kb" / "KB-0003.md").read_text(encoding="utf-8")
    assert "mức tối đa 5%" not in kb
    a2 = mutants["A2-01"]
    assert set(a2.variant.corpus.find("**mức tối đa hợp lý**.")) == a2.truth.location
    a3 = mutants["A3-00"]
    assert "KB-0009" in CanonPack(a3.variant.pack).kb
    assert all(TellCorpus.chunk_of(s) == "KB:KB-0009" for s in a3.truth.location)


def test_the_base_pack_is_never_touched(mutants) -> None:
    text = (BASE_PACK / "kb" / "KB-0003.md").read_text(encoding="utf-8")
    assert "**mức tối đa 5%**" in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/mutate -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.mutate'`.

- [ ] **Step 3: Implement the pack editing helpers**

`specdx/mutate/__init__.py`: empty.

`specdx/mutate/pack_copy.py`:
```python
"""Copies of a canon pack, edited in place. The base pack is never written."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_NUMBERED = re.compile(r"^\d+_")
Rows = list[dict[str, str]]


def copy_pack(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def regenerate_sql(pack: Path) -> None:
    """Re-project data/*.csv to SQL with the pack's own generator.

    On an unedited copy the output is byte-identical to the shipped SQL.
    """
    subprocess.run([sys.executable, "generate_sql.py"], cwd=pack, check=True, capture_output=True)


def csv_path(pack: Path, table: str) -> Path:
    for path in sorted((pack / "data").glob("*.csv")):
        name = path.stem.split("_", 1)[1] if _NUMBERED.match(path.stem) else path.stem
        if name == table:
            return path
    raise KeyError(f"no CSV for table {table}")


def edit_csv_rows(pack: Path, table: str, fn: Callable[[Rows], Rows]) -> None:
    """Rewrite one CSV, keeping its byte-order mark (or lack of one) and line endings."""
    path = csv_path(pack, table)
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    rows = fn(rows)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)


def edit_json(pack: Path, rel: str, fn: Callable[[Any], Any]) -> None:
    path = pack / rel
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(fn(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replace_text(pack: Path, rel: str, old: str, new: str) -> bool:
    path = pack / rel
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def append_text(pack: Path, rel: str, text: str) -> None:
    with (pack / rel).open("a", encoding="utf-8") as handle:
        handle.write(text)
```

- [ ] **Step 4: Run the pack-copy tests**

Run: `.venv/bin/python -m pytest tests/mutate/test_pack_copy.py -q`
Expected: `2 passed`. If the identity test fails on a file, compare its quoting with `csv.QUOTE_MINIMAL` output. Then make `edit_csv_rows` write back the original bytes when `fn` returns the rows unchanged, by comparing before writing.

- [ ] **Step 5: Implement the operators**

`specdx/mutate/operators.py`:
```python
"""Mutation operators for an executable business specification (spec §4.3, §11 item 6).

Each operator copies the base pack, applies one edit, and states its ground truth. State
operators also give a kill witness: a state that proves the defect is real.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tonomyx.canon.pack import CanonPack

from specdx.mutate.pack_copy import (
    append_text,
    copy_pack,
    edit_csv_rows,
    edit_json,
    regenerate_sql,
    replace_text,
)
from specdx.spec.tell import Chunk, TellCorpus
from specdx.types import State
from specdx.variant import SpecVariant

OPERATOR_TYPES: dict[str, str] = {
    "C1": "contradiction", "C2": "contradiction", "H1": "hidden", "H2": "hidden",
    "A1": "ambiguity", "A2": "ambiguity", "A3": "ambiguity", "W1": "weak", "W2": "weak",
}
C1_SITES = ("po-15", "grn-12", "vinv-cankiem", "apr0102-approved", "apr0155-approved",
            "apr0210-pending")
C2_SITES = (
    ("po", "ma_po", "PO-0015", "Đã duyệt", ("golden:po:PO-0015:trang_thai",)),
    ("apr", "ma_apr", "APR-0102", "Đã duyệt",
     ("golden:apr:APR-0102:trang_thai", "inv:apr0102-approved")),
    ("apr", "ma_apr", "APR-0155", "Đã duyệt",
     ("golden:apr:APR-0155:trang_thai", "inv:apr0155-approved")),
)
H1_SITES = (
    ("vinv", "ma_vinv", "VINV-0015", "trang_thai", "Nháp"),
    ("inv", "ma_inv", "INV-0031", "doi_soat", "Khớp"),
    ("inv", "ma_inv", "INV-0031", "trang_thai", "Chờ duyệt"),
    ("pay", "ma_pay", "PAY-0031", "trang_thai", "Nháp"),
    ("apr", "ma_apr", "APR-0210", "muc_tu_hanh", "L2"),
)
H2_SITES = (("PAID", "PAY-0031"), ("REJECTED", "APR-0155"))
A1_SITES = (
    "**mức tối đa 5%**.",
    "Yêu cầu vượt trần (vd 8%) phải gắn cờ ngoại lệ và dừng ở cổng P2 (APR-0102);",
    "GRN lệch / lô lỗi / giao trễ: P4 xử lý ngoại lệ — KHÔNG mở APR chuẩn.",
)
A2_SITES = (
    ("từ 100 sản phẩm giảm 5%;", "từ 100 sản phẩm giảm ở mức hợp lý;"),
    ("**mức tối đa 5%**.", "**mức tối đa hợp lý**."),
    ("CASE-0031 có ĐÚNG 3 cổng: APR-0102/P2, APR-0155/P6 (maker-checker), "
     "APR-0210/P5 (đối soát lệch).", "CASE-0031 có một số cổng phê duyệt phù hợp."),
)
A3_ARTICLE = (
    "---\nkb_id: KB-0009\nversion: 1\neffective_date: 2026-09-01\ndomain: sales\n"
    "title: Chính sách giá & chiết khấu (cập nhật)\n---\n"
    "Mức chiết khấu tối đa là 7%; yêu cầu đến 7% được áp dụng trực tiếp, "
    "không cần dừng ở cổng duyệt.\n"
)
W1_SITES = (
    ("soline", "chiet_khau", "chietkhau-0"), ("soline", "don_gia", "soline-dongia-blank"),
    ("po", "don_gia", "po-dongia-blank"), ("inv", "so_tien", "inv-sotien-blank"),
    ("vinv", "so_tien", "vinv-sotien-blank"), ("pay", "so_tien", "pay-sotien-blank"),
    ("apr", "gia_tri", "apr-giatri-blank"),
)
W2_SITES = (("DRAFTED", "SO-0031"), ("FLAGGED", "ITEM-0012"), ("DRAFTED", "PO-0015"),
            ("RECEIVED", "GRN-0015"), ("RECONCILED", "VINV-0015"), ("DRAFTED", "PAY-0031"))
_TRIGGER = """
-- specdx mutant C2: forbid a status the golden state requires
CREATE OR REPLACE FUNCTION fn_specdx_c2() RETURNS trigger AS $$
BEGIN
  IF NEW.{key} = '{value}' AND NEW.trang_thai = '{status}' THEN
    RAISE EXCEPTION 'specdx mutant: % may not be %', NEW.{key}, NEW.trang_thai;
  END IF;
  RETURN NEW;
END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_specdx_c2 ON {table};
CREATE TRIGGER trg_specdx_c2 BEFORE INSERT OR UPDATE ON {table}
  FOR EACH ROW EXECUTE FUNCTION fn_specdx_c2();
"""


@dataclass(frozen=True)
class GroundTruth:
    mutant_id: str
    type: str
    location: frozenset[str]
    location_kind: str


@dataclass(frozen=True)
class Mutant:
    id: str
    operator: str
    site: str
    variant: SpecVariant
    truth: GroundTruth


def _kb_files_with(pack: Path, text: str) -> list[str]:
    return [f"kb/{p.name}" for p in sorted((pack / "kb").glob("*.md"))
            if text in p.read_text(encoding="utf-8")]


def _edit_principles(pack: Path, old: str, new: str | None) -> None:
    def fn(meta: dict) -> dict:
        items = [p for p in meta["nguyen_tac"] if p != old]
        if new is not None and len(items) < len(meta["nguyen_tac"]):
            items = [new if p == old else p for p in meta["nguyen_tac"]]
        return {**meta, "nguyen_tac": items}
    edit_json(pack, "canon.meta.json", fn)


def _corpus(pack: Path, pdf: Sequence[Chunk]) -> TellCorpus:
    return TellCorpus.build(CanonPack(pack), pdf)


def _mutant(mid: str, op: str, site: str, pack: Path, corpus: TellCorpus,
            location: frozenset[str], kind: str) -> Mutant:
    return Mutant(mid, op, site, SpecVariant(mid, pack, corpus),
                  GroundTruth(mid, OPERATOR_TYPES[op], location, kind))


def all_mutants(base: SpecVariant, pdf: Sequence[Chunk], work_dir: Path) -> list[Mutant]:
    out: list[Mutant] = []

    def fresh(mid: str) -> Path:
        return copy_pack(base.pack, work_dir / "mutants" / mid / base.pack.name)

    for i, site in enumerate(C1_SITES):
        mid, pack = f"C1-{i:02d}", fresh(f"C1-{i:02d}")
        inv = next(x for x in CanonPack(pack).invariants if x["id"] == site)
        value = str(inv["assert"]["eq"])
        alt = str(int(value) + 3) if value.isdigit() else f"{value} (khác)"
        added = {"id": f"mut-c1-{site}", "desc": f"{inv['desc']} (đột biến)",
                 "table": inv["table"], "where": copy.deepcopy(inv["where"]),
                 "assert": {"col": inv["assert"]["col"], "eq": alt}}
        edit_json(pack, "invariants.json",
                  lambda d, a=added: {**d, "invariants": [*d["invariants"], a]})
        out.append(_mutant(mid, "C1", site, pack, _corpus(pack, pdf),
                           frozenset({f"inv:{site}", f"inv:mut-c1-{site}"}), "conflict"))

    for i, (table, key, value, status, location) in enumerate(C2_SITES):
        mid, pack = f"C2-{i:02d}", fresh(f"C2-{i:02d}")
        append_text(pack, "targets/postgres/guardrails.sql",
                    _TRIGGER.format(key=key, value=value, status=status, table=table))
        out.append(_mutant(mid, "C2", value, pack, _corpus(pack, pdf),
                           frozenset(location), "conflict"))

    for i, (table, key, ident, column, value) in enumerate(H1_SITES):
        mid, pack = f"H1-{i:02d}", fresh(f"H1-{i:02d}")

        def change(rows, key=key, ident=ident, column=column, value=value):
            for r in rows:
                if r[key] == ident:
                    r[column] = value
            return rows

        edit_csv_rows(pack, table, change)
        regenerate_sql(pack)
        out.append(_mutant(mid, "H1", f"{table}.{column}", pack, _corpus(pack, pdf),
                           frozenset({f"golden:{table}:{ident}:{column}"}), "clause"))

    for i, (kind, ref) in enumerate(H2_SITES):
        mid, pack = f"H2-{i:02d}", fresh(f"H2-{i:02d}")
        added = {"id": f"mut-h2-{kind}-{ref}", "desc": f"Có sự kiện {kind} cho {ref}",
                 "table": "evt", "where": {"kind": kind, "ref": ref}, "assert": {"count": 1}}
        edit_json(pack, "invariants.json",
                  lambda d, a=added: {**d, "invariants": [*d["invariants"], a]})
        out.append(_mutant(mid, "H2", f"{kind}:{ref}", pack, _corpus(pack, pdf),
                           frozenset({f"inv:mut-h2-{kind}-{ref}"}), "clause"))

    for i, text in enumerate(A1_SITES):
        mid, pack = f"A1-{i:02d}", fresh(f"A1-{i:02d}")
        for rel in _kb_files_with(pack, text):
            replace_text(pack, rel, text, "")
        _edit_principles(pack, text, None)
        corpus = _corpus(pack, pdf)
        corpus = corpus.without(corpus.find(text))
        chunks = frozenset(TellCorpus.chunk_of(s) for s in base.corpus.find(text))
        out.append(_mutant(mid, "A1", text, pack, corpus, chunks, "chunk"))

    for i, (old, new) in enumerate(A2_SITES):
        mid, pack = f"A2-{i:02d}", fresh(f"A2-{i:02d}")
        for rel in _kb_files_with(pack, old):
            replace_text(pack, rel, old, new)
        _edit_principles(pack, old, new)
        corpus = _corpus(pack, pdf)
        for sid in corpus.find(old):
            corpus = corpus.replace(sid, new)
        out.append(_mutant(mid, "A2", old, pack, corpus, frozenset(corpus.find(new)), "sentence"))

    mid, pack = "A3-00", fresh("A3-00")
    (pack / "kb" / "KB-0009.md").write_text(A3_ARTICLE, encoding="utf-8")
    corpus = _corpus(pack, pdf)
    out.append(_mutant(mid, "A3", "KB-0009", pack, corpus,
                       frozenset(corpus.sentence_ids(["KB:KB-0009"])), "sentence"))

    for i, (table, column, inv_id) in enumerate(W1_SITES):
        mid, pack = f"W1-{i:02d}", fresh(f"W1-{i:02d}")
        edit_json(pack, "invariants.json", lambda d, x=inv_id: {
            **d, "invariants": [v for v in d["invariants"] if v["id"] != x]})

        def unblank(meta, table=table, column=column):
            blank = dict(meta["cot_de_trong_khong_canon"])
            blank[table] = [c for c in blank[table] if c.split()[0] != column]
            return {**meta, "cot_de_trong_khong_canon": blank}

        edit_json(pack, "canon.meta.json", unblank)
        out.append(_mutant(mid, "W1", f"{table}.{column}", pack, _corpus(pack, pdf),
                           frozenset({f"{table}.{column}"}), "field"))

    for i, (kind, ref) in enumerate(W2_SITES):
        mid, pack = f"W2-{i:02d}", fresh(f"W2-{i:02d}")
        edit_csv_rows(pack, "evt", lambda rows, k=kind, r=ref: [
            x for x in rows if not (x["kind"] == k and x["ref"] == r)])
        regenerate_sql(pack)
        out.append(_mutant(mid, "W2", f"{kind}:{ref}", pack, _corpus(pack, pdf),
                           frozenset({f"evt:{kind}:{ref}"}), "field"))
    return out


def dev_split(mutants: Sequence[Mutant], per_type: int, seed: int) -> list[Mutant]:
    by_type: dict[str, list[Mutant]] = defaultdict(list)
    for m in sorted(mutants, key=lambda m: m.id):
        by_type[m.truth.type].append(m)
    rng = random.Random(seed)
    chosen: list[Mutant] = []
    for kind in sorted(by_type):
        pool = by_type[kind]
        rng.shuffle(pool)
        chosen += pool[:per_type]
    return sorted(chosen, key=lambda m: m.id)


def witness_state(mutant: Mutant, golden: State) -> State | None:
    """A state proving a state-level defect: the canon's own end state, edited if needed."""
    tables = {t: [dict(r) for r in rows] for t, rows in golden.tables.items()}
    if mutant.operator in ("C1", "H1", "H2"):
        return State(tables, dict(golden.derived), dict(golden.steps))
    if mutant.operator == "W1":
        table, column = mutant.site.split(".")
        for r in tables[table]:
            r[column] = "1"
        return State(tables, dict(golden.derived), dict(golden.steps))
    if mutant.operator == "W2":
        kind, ref = mutant.site.split(":")
        tables["evt"] = [e for e in tables["evt"] if not (e["kind"] == kind and e["ref"] == ref)]
        return State(tables, dict(golden.derived), dict(golden.steps))
    return None
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/mutate -q`
Expected: `8 passed`.

- [ ] **Step 7: Commit**

```bash
git add specdx/mutate tests/mutate
git commit -m "feat: specification mutation operators with ground truth and kill witnesses"
```

---
### Task 18: Baselines B2 and B3, and the matching rules

**Files:**
- Create: `specdx/baselines/__init__.py`, `specdx/baselines/b2_single.py`, `specdx/baselines/b3_irt.py`, `specdx/eval/__init__.py`, `specdx/eval/matching.py`
- Test: `tests/baselines/__init__.py`, `tests/baselines/test_baselines.py`, `tests/eval/__init__.py`, `tests/eval/test_matching.py`

**Interfaces:**
- Consumes: `Defect`, `DEFECT_TYPES` (Task 2), `GroundTruth` (Task 17), `TellCorpus.chunk_of` (Task 7).
- Produces:
  - `b2_report(behaviours, violated, theta, scenario_ids) -> list[Defect]` and `fit_2pl(y, *, iters=2000, lr=0.05, l2=0.01)`.
  - `b3_report(behaviours, violated) -> list[Defect]`.
  - `Prf(tp, fp, fn)`, with `.precision`, `.recall`, `.f1` and `.as_dict()`.
  - `location_matches(defect, truth) -> bool` and `evaluate_reports(reports, truths, base, *, typed=True, k=None) -> dict[str, Prf]`.
  - `gate_verdict(scores, *, method="specdx", baselines=("b2", "b3"), required=3) -> tuple[bool, dict[str, bool]]`.
- **Note for the gate:** B2 and B3 cannot type a defect, so they report every finding as `hidden`. The typed comparison of spec §4.8 is kept as the verdict. The untyped scores (`typed=False`, location only) are reported beside it so the gate result can be read fairly.

- [ ] **Step 1: Write the failing tests**

`tests/baselines/__init__.py`: empty. `tests/eval/__init__.py`: empty.

`tests/baselines/test_baselines.py`:
```python
import random

import numpy as np

from specdx.baselines.b2_single import b2_report
from specdx.baselines.b3_irt import b3_report, fit_2pl
from specdx.types import Behaviour, InterpreterId, Scenario, State

S = Scenario("grn12", 15, 12, 0)


def _b(key, model):
    return Behaviour(key, InterpreterId(model, "f", 0, 1), S, State({}))


def test_b2_flags_whatever_the_best_interpreter_violates() -> None:
    bs = [_b("a", "m1"), _b("b", "m1"), _b("c", "m2")]
    violated = {"a": frozenset({"x"}), "b": frozenset({"x", "y"}), "c": frozenset({"z"})}
    report = b2_report(bs, violated, {"m1": 0.9, "m2": 0.3}, {"grn12"})
    assert [(d.type, d.location, d.score) for d in report] == [
        ("hidden", ("x",), 1.0), ("hidden", ("y",), 0.5)
    ]


def test_fit_2pl_gives_negative_discrimination_to_a_miskeyed_item() -> None:
    rng = random.Random(0)
    rows = []
    for i in range(30):
        strong = i < 15
        row = [1.0 if (strong or rng.random() < 0.2) else 0.0 for _ in range(5)]
        row.append(0.0 if strong else 1.0)  # the miskeyed item: the strong get it "wrong"
        rows.append(row)
    _, a, _ = fit_2pl(np.array(rows))
    assert a[5] < 0 < min(a[:5])


def test_b3_reports_negatively_discriminating_clauses() -> None:
    rng = random.Random(1)
    bs, violated = [], {}
    for i in range(30):
        strong = i < 15
        bad = {f"c{j}" for j in range(5) if not (strong or rng.random() < 0.2)}
        if strong:
            bad.add("miskeyed")
        key = f"r{i}"
        bs.append(_b(key, "m"))
        violated[key] = frozenset(bad)
    report = b3_report(bs, violated)
    assert report and report[0].location == ("miskeyed",)
```

`tests/eval/test_matching.py`:
```python
from specdx.eval.matching import Prf, evaluate_reports, gate_verdict, location_matches
from specdx.mutate.operators import GroundTruth
from specdx.types import Defect


def _d(kind, *loc):
    return Defect(kind, tuple(loc), ("s",), (), 0.9)


def test_location_rules_per_kind() -> None:
    clause = GroundTruth("m", "hidden", frozenset({"c1"}), "clause")
    assert location_matches(_d("hidden", "c1"), clause)
    conflict = GroundTruth("m", "contradiction", frozenset({"a", "b"}), "conflict")
    assert location_matches(_d("contradiction", "a"), conflict)
    assert not location_matches(_d("contradiction", "a", "x", "y"), conflict)
    chunk = GroundTruth("m", "ambiguity", frozenset({"KB:KB-0003"}), "chunk")
    assert location_matches(_d("ambiguity", "KB:KB-0003:02"), chunk)
    assert not location_matches(_d("ambiguity", "soline.chiet_khau"), chunk)
    field = GroundTruth("m", "weak", frozenset({"pay.so_tien"}), "field")
    assert location_matches(_d("weak", "pay.so_tien", "pay.trang_thai"), field)


def test_base_findings_are_removed_and_errors_counted() -> None:
    truths = {"m1": GroundTruth("m1", "hidden", frozenset({"c1"}), "clause"),
              "m2": GroundTruth("m2", "weak", frozenset({"pay.so_tien"}), "field")}
    reports = {"m1": [_d("hidden", "c1"), _d("hidden", "noise")],
               "m2": [_d("hidden", "pay.so_tien")]}
    base = [_d("hidden", "noise")]
    typed = evaluate_reports(reports, truths, base)
    assert (typed["hidden"].tp, typed["hidden"].fp, typed["weak"].fn) == (1, 1, 1)
    untyped = evaluate_reports(reports, truths, base, typed=False)
    assert untyped["weak"].tp == 1


def test_f1_and_the_gate_verdict() -> None:
    assert Prf(tp=2, fp=2, fn=0).f1 == 2 * 0.5 * 1.0 / 1.5
    good, poor = Prf(tp=3, fp=1, fn=1), Prf(tp=1, fp=3, fn=3)
    types = ("contradiction", "hidden", "ambiguity", "weak")
    scores = {"specdx": dict.fromkeys(types, good), "b2": dict.fromkeys(types, poor),
              "b3": {t: (good if t == "hidden" else poor) for t in types}}
    passed, wins = gate_verdict(scores)
    assert passed and wins == {"contradiction": True, "hidden": False, "ambiguity": True,
                               "weak": True}
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/baselines tests/eval -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.baselines'`.

- [ ] **Step 3: Implement**

`specdx/baselines/__init__.py`, `specdx/eval/__init__.py`: empty.

`specdx/baselines/b2_single.py`:
```python
"""B2: the single most competent interpreter, every violated clause flagged (spec §4.5)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence

from specdx.types import Behaviour, Defect


def b2_report(
    behaviours: Sequence[Behaviour],
    violated: Mapping[str, frozenset[str]],
    theta: Mapping[str, float],
    scenario_ids: Collection[str],
) -> list[Defect]:
    if not theta:
        return []
    best = max(sorted(theta), key=lambda m: theta[m])
    mine = [b for b in behaviours
            if b.status == "ok" and b.interpreter.model == best and b.scenario.id in scenario_ids]
    if not mine:
        return []
    counts = Counter(c for b in mine for c in violated[b.run_key])
    out = []
    for clause, n in counts.items():
        hit = [b for b in mine if clause in violated[b.run_key]]
        out.append(Defect("hidden", (clause,), tuple(sorted({b.scenario.id for b in hit})),
                          tuple(b.run_key for b in hit), n / len(mine)))
    return sorted(out, key=lambda d: (-d.score, d.location))
```

`specdx/baselines/b3_irt.py`:
```python
"""B3: 2PL item response theory over runs × clauses (after arXiv:2605.30504).

A clause whose discrimination is negative is one that stronger runs fail more often: the
signature of a mis-keyed item. B3 sees only satisfied/violated, so a clause that does not
apply to a run's scenario counts as satisfied; that is the baseline's documented limit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from specdx.types import Behaviour, Defect


def fit_2pl(
    y: np.ndarray, *, iters: int = 2000, lr: float = 0.05, l2: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m = y.shape
    theta, a, b = np.zeros(n), np.ones(m), np.zeros(m)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a[None, :] * (theta[:, None] - b[None, :]))))
        r = y - p
        g_theta = (r * a[None, :]).sum(axis=1) - l2 * theta
        g_a = (r * (theta[:, None] - b[None, :])).sum(axis=0) - l2 * (a - 1.0)
        g_b = (-r * a[None, :]).sum(axis=0) - l2 * b
        theta += lr * g_theta / max(m, 1)
        a += lr * g_a / max(n, 1)
        b += lr * g_b / max(n, 1)
        theta -= theta.mean()
    return theta, a, b


def b3_report(
    behaviours: Sequence[Behaviour], violated: Mapping[str, frozenset[str]]
) -> list[Defect]:
    ok = [b for b in behaviours if b.status == "ok"]
    clauses = sorted({c for b in ok for c in violated[b.run_key]})
    if len(ok) < 2 or not clauses:
        return []
    y = np.array([[0.0 if c in violated[b.run_key] else 1.0 for c in clauses] for b in ok])
    _, a, _ = fit_2pl(y)
    out = []
    for j, clause in enumerate(clauses):
        if a[j] < 0:
            hit = [b for b in ok if clause in violated[b.run_key]]
            out.append(Defect("hidden", (clause,), tuple(sorted({b.scenario.id for b in hit})),
                              tuple(b.run_key for b in hit), float(-a[j])))
    return sorted(out, key=lambda d: (-d.score, d.location))
```

`specdx/eval/matching.py`:
```python
"""Scoring reports against ground truth (spec §4.6–§4.8)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from specdx.mutate.operators import GroundTruth
from specdx.spec.tell import TellCorpus
from specdx.types import DEFECT_TYPES, Defect


@dataclass
class Prf:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": self.precision,
                "recall": self.recall, "f1": self.f1}


def location_matches(defect: Defect, truth: GroundTruth) -> bool:
    found = set(defect.location)
    kind = truth.location_kind
    if kind in ("clause", "field", "sentence"):
        return bool(found & truth.location)
    if kind == "conflict":
        union = found | truth.location
        return bool(union) and len(found & truth.location) / len(union) >= 0.5
    if kind == "chunk":
        return any(TellCorpus.chunk_of(s) in truth.location for s in found)
    raise ValueError(f"unknown location kind {kind!r}")


def evaluate_reports(
    reports: Mapping[str, Sequence[Defect]],
    truths: Mapping[str, GroundTruth],
    base: Sequence[Defect],
    *,
    typed: bool = True,
    k: int | None = None,
) -> dict[str, Prf]:
    """One TP or FN per mutant under its true type; every unmatched new finding is an FP."""
    scores = {t: Prf() for t in DEFECT_TYPES}
    base_keys = {d.key for d in base}
    for mutant_id, defects in reports.items():
        truth = truths[mutant_id]
        new = [d for d in defects if d.key not in base_keys]
        if k is not None:
            new = new[:k]
        hits = [d for d in new
                if (not typed or d.type == truth.type) and location_matches(d, truth)]
        if hits:
            scores[truth.type].tp += 1
        else:
            scores[truth.type].fn += 1
        for d in new:
            if d not in hits:
                scores[d.type].fp += 1
    return scores


def gate_verdict(
    scores: Mapping[str, Mapping[str, Prf]],
    *,
    method: str = "specdx",
    baselines: Sequence[str] = ("b2", "b3"),
    required: int = 3,
) -> tuple[bool, dict[str, bool]]:
    wins = {t: scores[method][t].f1 > max(scores[b][t].f1 for b in baselines)
            for t in DEFECT_TYPES}
    return sum(wins.values()) >= required, wins
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/baselines tests/eval -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add specdx/baselines specdx/eval tests/baselines tests/eval
git commit -m "feat: baselines B2 and B3, location matching and the gate verdict"
```

---

### Task 19: Workbench, capability check, Tell paraphrases, gate runner and CLI

**Files:**
- Create: `specdx/workbench.py`, `specdx/interpreters/paraphrase.py`, `specdx/eval/pilot.py`, `specdx/cli.py`, `experiments/p1-gate.json`, `README.md`
- Test: `tests/interpreters/test_paraphrase.py`, `tests/eval/test_pilot.py`, `tests/test_workbench.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `ollama_digests(base_url=OLLAMA_URL, transport=None) -> Callable[[str], str]`.
  - `Workbench`, with:
    - `.open(work_dir=WORK_DIR, *, factory=None, model_digest=None)`;
    - fields `.db`, `.cache`, `.steps`, `.variants`, `.pdf`, `.base`, `.runner`, `.work_dir`;
    - `.llm_factory(interp, corpus, suffix)` and `.loop_context(variant, config) -> LoopContext`, which wires `CombinedRefuter` and the ddmin localiser.
  - `paraphrase(text, *, model, seed, client, base_url, retries=3) -> str` and `make_variants(variants, *, model, count, base_url=OLLAMA_URL, transport=None) -> TellVariants`.
  - `Capability(model, declares_tools, calls_tool)`, with `.usable`, and `check_model(model, *, base_url=OLLAMA_URL, transport=None) -> Capability`.
  - `GateConfig.load(path)`, with `.specdx() -> SpecDxConfig`.
  - `GateResult(scores, untyped, passed, wins, failed)`, with `.to_json()`; `run_gate(config, bench, mutants) -> GateResult`; and `gate_markdown(result) -> str`.
  - `specdx.cli.main(argv=None) -> int`, with the subcommands `build-tell`, `make-variants`, `capability`, `mutants`, `run` and `gate`.

- [ ] **Step 1: Write the experiment configuration**

`experiments/p1-gate.json`:
```json
{
  "_note": "Pilot go/no-go gate (spec §4.8): two tool-capable families, development mutants only. Task 20 may swap the second family after the capability pilot.",
  "models_by_family": {
    "qwen": ["qwen3:14b", "qwen2.5:7b-instruct-q4_K_M"],
    "llama": ["llama3.1:8b"]
  },
  "variants": [0, 1, 2],
  "seeds": [1, 2, 3],
  "budget": 200,
  "dev_per_type": 5,
  "rng_seed": 20261006
}
```

- [ ] **Step 2: Write the failing tests**

`tests/interpreters/test_paraphrase.py`:
```python
import json

import httpx
import pytest

from specdx.interpreters.paraphrase import make_variants, paraphrase
from specdx.interpreters.prompt import load_variants


def _model(*answers):
    queue = list(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = queue.pop(0) if queue else body["messages"][0]["content"].split("\n\n", 1)[1]
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    return httpx.MockTransport(handle)


def test_a_paraphrase_must_keep_every_placeholder() -> None:
    with httpx.Client(transport=_model("mất {role}", "giữ {role} và {actor}")) as client:
        out = paraphrase("A {role} B {actor}", model="m", seed=1, client=client, base_url="http://x")
    assert out == "giữ {role} và {actor}"


def test_a_paraphrase_that_never_keeps_them_is_refused() -> None:
    with httpx.Client(transport=_model("x", "y", "z")) as client, pytest.raises(ValueError):
        paraphrase("A {role}", model="m", seed=1, client=client, base_url="http://x")


def test_make_variants_adds_count_versions_of_everything() -> None:
    base = load_variants()
    made = make_variants(base, model="m", count=2, base_url="http://x", transport=_model())
    assert made.count == 3 and made.standing[0] == base.standing[0]
    assert all(len(v) == 3 for v in made.statements.values())
```

`tests/eval/test_pilot.py`:
```python
import json
from pathlib import Path

import httpx

from specdx.active.loop import LoopContext
from specdx.eval.pilot import GateConfig, check_model, gate_markdown, run_gate
from specdx.infer.items import ItemView
from specdx.mutate.operators import GroundTruth, Mutant
from specdx.spec.tell import TellCorpus
from specdx.types import Behaviour, RefuteResult, Scenario, State
from specdx.variant import SpecVariant


def test_capability_needs_a_declared_and_a_real_tool_call() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        call = {"function": {"name": "echo", "arguments": {"text": "ok"}}}
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "",
                                                     "tool_calls": [call]}})

    cap = check_model("m", base_url="http://x", transport=httpx.MockTransport(handle))
    assert cap.usable


def test_a_model_without_the_tools_capability_is_not_usable() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": ["completion"]})

    cap = check_model("m", base_url="http://x", transport=httpx.MockTransport(handle))
    assert not cap.declares_tools and not cap.usable


CANON = {"grn12": Scenario("grn12", 15, 12, 0)}


class _Runner:
    def __init__(self, broken: bool) -> None:
        self.broken, self.executed = broken, 0

    def run(self, variant, scenario, interp, suffix=""):
        self.executed += 1
        key = f"{variant.name}|{interp.key}"
        if self.broken:
            return Behaviour(key, interp, scenario, State.empty(), status="error", error="bad SQL")
        bad = "c1" if variant.name == "H1-00" else ""
        return Behaviour(key, interp, scenario, State({}, {"v": bad}, {}))


class _Lens:
    def violations(self, b):
        return frozenset(x for x in b.state.derived.get("v", "").split(",") if x)

    def views(self, b, violated):
        if b.status != "ok":
            return []
        return [ItemView(f"{b.scenario.id}|s", b.run_key, b.interpreter.model,
                         b.interpreter.family, "K", violated, ())]

    def diff(self, a, b):
        return frozenset()


class _Refute:
    def refute(self, targets, scenario, behaviours):
        return RefuteResult(True, "w", 1)


class _Bench:
    base = SpecVariant("base", Path("."), TellCorpus([]))

    def loop_context(self, variant, config):
        return LoopContext(_Runner(variant.name == "broken"), _Lens(), CANON,
                           list(CANON.values()), 5, lambda i: _Refute())


def _mutant(mid, kind, location):
    return Mutant(mid, mid[:2], "s", SpecVariant(mid, Path("."), TellCorpus([])),
                  GroundTruth(mid, kind, frozenset(location), "clause"))


def test_gate_records_a_broken_mutant_and_continues(tmp_path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"models_by_family": {"q": ["m1"], "l": ["m2"]},
                                "variants": [0], "seeds": [1], "budget": 10,
                                "dev_per_type": 1, "rng_seed": 1}), encoding="utf-8")
    mutants = [_mutant("H1-00", "hidden", {"c1"}), _mutant("broken", "weak", {"x.y"})]
    result = run_gate(GateConfig.load(path), _Bench(), mutants)
    assert "broken" in result.failed and "bad SQL" in result.failed["broken"]
    assert result.scores["specdx"]["hidden"].tp == 1
    assert "| hidden |" in gate_markdown(result)
```

`tests/test_workbench.py`:
```python
import pytest

from specdx.active.loop import SpecDxConfig
from specdx.orchestrate.reference import reference_executor
from specdx.types import InterpreterId
from specdx.workbench import Workbench

pytestmark = pytest.mark.db
REF = InterpreterId("reference", "ref", 0, 1)


def test_the_workbench_wires_a_runnable_context(tmp_path) -> None:
    bench = Workbench.open(tmp_path, factory=lambda i, c, s: reference_executor(),
                           model_digest=lambda m: m)
    ctx = bench.loop_context(bench.base, SpecDxConfig({"ref": ("reference",)}))
    assert set(ctx.canon) == {"grn12", "happy", "discount8"} and ctx.cap_pct == 5
    assert len(ctx.candidates) > 50
    b = ctx.runner.run(bench.base, ctx.canon["grn12"], REF)
    assert ctx.lens.violations(b) == frozenset()
```

`tests/test_cli.py`:
```python
import pytest

from specdx.cli import main


def test_help_lists_every_subcommand(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("build-tell", "make-variants", "capability", "mutants", "run", "gate"):
        assert command in out
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/interpreters/test_paraphrase.py tests/eval/test_pilot.py tests/test_workbench.py tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'specdx.interpreters.paraphrase'`.

- [ ] **Step 4: Implement the paraphraser**

`specdx/interpreters/paraphrase.py`:
```python
"""Tell variants 1..n: paraphrases of variant 0 by a local model, checked and then frozen."""

from __future__ import annotations

from typing import Any

import httpx
from tonomyx.de.executor import _plain

from specdx.config import OLLAMA_URL
from specdx.interpreters.prompt import PLACEHOLDERS, TellVariants

PROMPT = (
    "Viết lại đoạn văn sau bằng tiếng Việt với cách diễn đạt khác nhưng giữ nguyên ý nghĩa. "
    "Giữ nguyên từng ký hiệu trong ngoặc nhọn (như {role}) và mọi mã (như APR-0102). "
    "Chỉ trả về đoạn văn đã viết lại.\n\n"
)


def paraphrase(
    text: str, *, model: str, seed: int, client: httpx.Client, base_url: str, retries: int = 3
) -> str:
    required = [p for p in PLACEHOLDERS if p in text]
    for attempt in range(retries):
        body: dict[str, Any] = {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": PROMPT + text}],
            "options": {"temperature": 0.7, "seed": seed * 10 + attempt},
        }
        if model.startswith("qwen3"):
            body["think"] = False
        reply = client.post(f"{base_url}/api/chat", json=body).json()["message"]
        out = _plain(reply)
        if out and all(p in out for p in required) and out.count("{") == text.count("{"):
            return out
    raise ValueError(f"{model} could not paraphrase while keeping {required}")


def make_variants(
    variants: TellVariants,
    *,
    model: str,
    count: int,
    base_url: str = OLLAMA_URL,
    transport: httpx.BaseTransport | None = None,
) -> TellVariants:
    standing = [variants.standing[0]]
    statements = {k: [v[0]] for k, v in variants.statements.items()}
    with httpx.Client(transport=transport, timeout=300.0) as client:
        for n in range(1, count + 1):
            standing.append(paraphrase(variants.standing[0], model=model, seed=n,
                                       client=client, base_url=base_url))
            for key in statements:
                statements[key].append(paraphrase(variants.statements[key][0], model=model,
                                                  seed=n, client=client, base_url=base_url))
    return TellVariants(tuple(standing), {k: tuple(v) for k, v in statements.items()})
```

- [ ] **Step 5: Implement the workbench**

`specdx/workbench.py`:
```python
"""Wires the database, the run cache, the Tell corpus, the runner and the loop context."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from tonomyx.canon.pack import CanonPack
from tonomyx.de.executor import DeExecutor
from tonomyx.tools.catalog import build_registry

from specdx.active.ddmin import localize_ambiguity
from specdx.active.loop import LoopContext, SpecDxConfig
from specdx.active.refute import CombinedRefuter, DirectedProbe, SymbolicRepair
from specdx.config import BASE_PACK, OLLAMA_URL, PDF_DIR, WORK_DIR
from specdx.db import PackDatabase
from specdx.infer.items import SpecLens
from specdx.interpreters.cache import RunCache
from specdx.interpreters.executor import SpecInterpreterExecutor
from specdx.interpreters.prompt import TellVariants, load_variants
from specdx.observe.run import ExecutorFactory, Runner
from specdx.orchestrate.work_orders import WorkStep, load_work_orders
from specdx.scenarios.space import (
    boundary_domain,
    canon_scenarios,
    discount_cap_pct,
    enumerate_scenarios,
)
from specdx.spec.clauses import ClauseSet
from specdx.spec.columns import ColumnModel
from specdx.spec.tell import Chunk, TellCorpus, pdf_chunks
from specdx.types import Defect, InterpreterId, Scenario
from specdx.variant import SpecVariant


def ollama_digests(
    base_url: str = OLLAMA_URL, transport: httpx.BaseTransport | None = None
) -> Callable[[str], str]:
    """Model name → Ollama digest, so the run cache never mixes two builds of one model."""
    try:
        with httpx.Client(transport=transport, timeout=10.0) as client:
            models = client.get(f"{base_url}/api/tags").json().get("models", [])
        digests = {str(m["name"]): str(m["digest"]) for m in models}
    except httpx.HTTPError:
        digests = {}
    return lambda model: digests.get(model, model)


class Workbench:
    def __init__(
        self,
        db: PackDatabase,
        cache: RunCache,
        steps: tuple[WorkStep, ...],
        variants: TellVariants,
        pdf: list[Chunk],
        base: SpecVariant,
        work_dir: Path,
    ) -> None:
        self.db, self.cache, self.steps, self.variants = db, cache, steps, variants
        self.pdf, self.base, self.work_dir = pdf, base, work_dir
        self.runner: Runner

    @classmethod
    def open(
        cls,
        work_dir: Path = WORK_DIR,
        *,
        factory: ExecutorFactory | None = None,
        model_digest: Callable[[str], str] | None = None,
    ) -> Workbench:
        pdf = pdf_chunks(PDF_DIR, work_dir / "tell")
        base = SpecVariant("base", BASE_PACK, TellCorpus.build(CanonPack(BASE_PACK), pdf))
        bench = cls(PackDatabase(), RunCache(work_dir / "runs.jsonl"), load_work_orders(),
                    load_variants(), pdf, base, work_dir)
        bench.runner = Runner(bench.db, bench.cache, factory or bench.llm_factory, bench.steps,
                              model_digest=model_digest or ollama_digests())
        return bench

    def llm_factory(self, interp: InterpreterId, corpus: TellCorpus, suffix: str) -> DeExecutor:
        return SpecInterpreterExecutor(
            interp.model, build_registry(), corpus, self.variants,
            variant=interp.variant, seed=interp.seed, suffix=suffix,
        )

    def loop_context(self, variant: SpecVariant, config: SpecDxConfig) -> LoopContext:
        pack = CanonPack(variant.pack)
        clauses = ClauseSet.from_pack(pack, self.steps)
        lens = SpecLens(clauses, self.steps, ColumnModel.of(pack))
        canon = canon_scenarios(pack)
        candidates = enumerate_scenarios(boundary_domain(pack))
        by_id = {s.id: s for s in [*canon.values(), *candidates]}

        def refuter_factory(interps: Sequence[InterpreterId]) -> CombinedRefuter:
            return CombinedRefuter(
                clauses, SymbolicRepair(self.db, variant, clauses),
                DirectedProbe(self.runner, variant, clauses, interps),
            )

        def localizer(v: SpecVariant, defect: Defect, interps: Sequence[InterpreterId]
                      ) -> tuple[str, ...]:
            scenario_id, step_id = defect.note.split("|", 1)
            scenario: Scenario = by_id[scenario_id]
            step = next(s for s in self.steps if s.id == step_id)
            query = f"{self.variants.statements[step.intent][0]} {' '.join(sorted(step.tables))}"
            chunks = v.corpus.search(query, k=5)
            pool = v.corpus.sentence_ids([c.id for c in chunks])
            one_per_family: dict[str, InterpreterId] = {}
            for interp in interps:
                one_per_family.setdefault(interp.family, interp)

            def split_persists(corpus: TellCorpus) -> bool:
                probe = SpecVariant(f"{v.name}+ddmin", v.pack, corpus)
                classes = set()
                for interp in one_per_family.values():
                    b = self.runner.run(probe, scenario, interp)
                    for view in lens.views(b, lens.violations(b)):
                        if view.item == defect.note:
                            classes.add(view.klass)
                return len(classes) >= 2

            return tuple(localize_ambiguity(v.corpus, pool, split_persists,
                                            max_tests=config.localize_tests))

        return LoopContext(self.runner, lens, canon, candidates, discount_cap_pct(pack),
                           refuter_factory, localizer)
```

- [ ] **Step 6: Implement the capability check and the gate**

`specdx/eval/pilot.py`:
```python
"""The capability pilot (spec §4.2) and the go/no-go gate (spec §4.8)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from specdx.active.loop import LoopContext, SpecDxConfig, run_specdx
from specdx.baselines.b2_single import b2_report
from specdx.baselines.b3_irt import b3_report
from specdx.config import OLLAMA_URL
from specdx.eval.matching import Prf, evaluate_reports, gate_verdict
from specdx.mutate.operators import GroundTruth, Mutant
from specdx.types import DEFECT_TYPES, Defect
from specdx.variant import SpecVariant

ECHO = {"type": "function", "function": {
    "name": "echo", "description": "Trả lại văn bản.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                   "required": ["text"]}}}


@dataclass(frozen=True)
class Capability:
    model: str
    declares_tools: bool
    calls_tool: bool

    @property
    def usable(self) -> bool:
        return self.declares_tools and self.calls_tool


def check_model(
    model: str, *, base_url: str = OLLAMA_URL, transport: httpx.BaseTransport | None = None
) -> Capability:
    with httpx.Client(transport=transport, timeout=300.0) as client:
        show = client.post(f"{base_url}/api/show", json={"model": model}).json()
        if "tools" not in show.get("capabilities", []):
            return Capability(model, False, False)
        body: dict[str, Any] = {
            "model": model, "stream": False, "tools": [ECHO], "options": {"temperature": 0},
            "messages": [{"role": "user", "content": "Gọi công cụ echo với text là 'ok'."}],
        }
        if model.startswith("qwen3"):
            body["think"] = False
        message = client.post(f"{base_url}/api/chat", json=body).json().get("message", {})
    calls = message.get("tool_calls") or []
    return Capability(model, True, any((c.get("function") or {}).get("name") == "echo"
                                       for c in calls))


@dataclass(frozen=True)
class GateConfig:
    models_by_family: dict[str, tuple[str, ...]]
    variants: tuple[int, ...]
    seeds: tuple[int, ...]
    budget: int
    dev_per_type: int
    rng_seed: int

    @classmethod
    def load(cls, path: Path) -> GateConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            models_by_family={f: tuple(ms) for f, ms in raw["models_by_family"].items()},
            variants=tuple(raw["variants"]), seeds=tuple(raw["seeds"]), budget=int(raw["budget"]),
            dev_per_type=int(raw["dev_per_type"]), rng_seed=int(raw["rng_seed"]),
        )

    def specdx(self) -> SpecDxConfig:
        return SpecDxConfig(self.models_by_family, variants=self.variants, seeds=self.seeds,
                            budget=self.budget, rng_seed=self.rng_seed)


class BenchLike(Protocol):
    base: SpecVariant

    def loop_context(self, variant: SpecVariant, config: SpecDxConfig) -> LoopContext: ...


@dataclass
class GateResult:
    scores: dict[str, dict[str, Prf]]
    untyped: dict[str, dict[str, Prf]]
    passed: bool
    wins: dict[str, bool]
    failed: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        def table(s: dict[str, dict[str, Prf]]) -> dict[str, Any]:
            return {m: {t: p.as_dict() for t, p in by.items()} for m, by in s.items()}
        return json.dumps({"passed": self.passed, "wins": self.wins, "failed": self.failed,
                           "typed": table(self.scores), "untyped": table(self.untyped)},
                          ensure_ascii=False, indent=2)


def _baselines(report_behaviours, ctx: LoopContext, theta, canon_ids):
    violated = {b.run_key: ctx.lens.violations(b) for b in report_behaviours}
    return (b2_report(report_behaviours, violated, theta, canon_ids),
            b3_report(report_behaviours, violated))


def run_gate(config: GateConfig, bench: BenchLike, mutants: Sequence[Mutant]) -> GateResult:
    cfg = config.specdx()
    base_ctx = bench.loop_context(bench.base, cfg)
    base = run_specdx(bench.base, base_ctx, cfg)
    canon_ids = set(base_ctx.canon)
    base_b2, base_b3 = _baselines(base.behaviours, base_ctx, base.theta, canon_ids)
    reports: dict[str, dict[str, list[Defect]]] = {"specdx": {}, "b2": {}, "b3": {}}
    truths: dict[str, GroundTruth] = {}
    failed: dict[str, str] = {}
    for m in mutants:
        try:
            ctx = bench.loop_context(m.variant, cfg)
            report = run_specdx(m.variant, ctx, cfg)
        except Exception as exc:  # noqa: BLE001 - one broken mutant must not stop the gate
            failed[m.id] = f"{type(exc).__name__}: {exc}"
            continue
        if report.behaviours and all(b.status == "error" for b in report.behaviours):
            failed[m.id] = report.behaviours[0].error
            continue
        truths[m.id] = m.truth
        reports["specdx"][m.id] = report.defects
        reports["b2"][m.id], reports["b3"][m.id] = _baselines(
            report.behaviours, ctx, report.theta, canon_ids
        )
    bases = {"specdx": base.defects, "b2": base_b2, "b3": base_b3}
    scores = {n: evaluate_reports(reports[n], truths, bases[n]) for n in reports}
    untyped = {n: evaluate_reports(reports[n], truths, bases[n], typed=False) for n in reports}
    passed, wins = gate_verdict(scores)
    return GateResult(scores, untyped, passed, wins, failed)


def gate_markdown(result: GateResult) -> str:
    lines = [
        f"# P1 gate: {'PASSED' if result.passed else 'NOT PASSED'}",
        "",
        "Typed F1 (verdict) and untyped F1 (location only) per defect type.",
        "",
        "| type | SpecDx | B2 | B3 | SpecDx untyped | B2 untyped | B3 untyped | SpecDx wins |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in DEFECT_TYPES:
        lines.append(
            f"| {t} | {result.scores['specdx'][t].f1:.2f} | {result.scores['b2'][t].f1:.2f} | "
            f"{result.scores['b3'][t].f1:.2f} | {result.untyped['specdx'][t].f1:.2f} | "
            f"{result.untyped['b2'][t].f1:.2f} | {result.untyped['b3'][t].f1:.2f} | "
            f"{'yes' if result.wins[t] else 'no'} |"
        )
    if result.failed:
        lines += ["", "## Mutants that could not run", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(result.failed.items())]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 7: Implement the CLI and the README**

`specdx/cli.py`:
```python
"""specdx: the command line for phase P1."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from specdx.config import PDF_DIR, WORK_DIR


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="specdx", description="SpecDx phase P1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-tell", help="extract and cache the Tell corpus from the PDFs")
    mv = sub.add_parser("make-variants", help="paraphrase Tell variant 0 into variants 1..n")
    mv.add_argument("--model", default="qwen3:14b")
    mv.add_argument("--count", type=int, default=2)
    cap = sub.add_parser("capability", help="check native tool calling of local models")
    cap.add_argument("models", nargs="+")
    sub.add_parser("mutants", help="write every mutant and list its ground truth")
    run = sub.add_parser("run", help="run SpecDx on the base specification or one mutant")
    run.add_argument("--variant", default="base")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    gate = sub.add_parser("gate", help="run the P1 go/no-go gate on the development mutants")
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "build-tell":
        from specdx.spec.tell import pdf_chunks

        chunks = pdf_chunks(PDF_DIR, WORK_DIR / "tell")
        for source, n in sorted(Counter(c.source for c in chunks).items()):
            print(f"{source}: {n} chunks")
        return 0

    if args.command == "make-variants":
        from specdx.interpreters.paraphrase import make_variants
        from specdx.interpreters.prompt import load_variants, save_variants

        made = make_variants(load_variants(), model=args.model, count=args.count)
        save_variants(made)
        print(f"variants: {made.count}")
        return 0

    if args.command == "capability":
        from specdx.eval.pilot import check_model

        print("| model | declares tools | calls a tool | usable |\n|---|---|---|---|")
        for model in args.models:
            c = check_model(model)
            print(f"| {model} | {c.declares_tools} | {c.calls_tool} | {c.usable} |")
        return 0

    from specdx.mutate.operators import all_mutants, dev_split
    from specdx.workbench import Workbench

    bench = Workbench.open()
    if args.command == "mutants":
        for m in all_mutants(bench.base, bench.pdf, bench.work_dir):
            location = ", ".join(sorted(m.truth.location))
            print(f"{m.id}\t{m.truth.type}\t{m.truth.location_kind}\t{location}")
        return 0

    from specdx.eval.pilot import GateConfig, gate_markdown, run_gate

    config = GateConfig.load(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        from specdx.active.loop import run_specdx

        variant = bench.base
        if args.variant != "base":
            mutants = {m.id: m for m in all_mutants(bench.base, bench.pdf, bench.work_dir)}
            variant = mutants[args.variant].variant
        cfg = config.specdx()
        report = run_specdx(variant, bench.loop_context(variant, cfg), cfg)
        args.out.write_text(report.to_json(), encoding="utf-8")
        for d in report.defects[:10]:
            print(f"{d.score:.2f}\t{d.type}\t{', '.join(d.location)}")
        return 0

    mutants = dev_split(all_mutants(bench.base, bench.pdf, bench.work_dir),
                        config.dev_per_type, config.rng_seed)
    result = run_gate(config, bench, mutants)
    args.out.write_text(result.to_json(), encoding="utf-8")
    args.out.with_suffix(".md").write_text(gate_markdown(result), encoding="utf-8")
    print(gate_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`README.md`:
````markdown
# SpecDx: diagnosing executable business specifications with LLM interpreters

Phase P1 (core on the TonomyX canon). The design is in `docs/superpowers/specs/2026-10-06-specdx-design.md` and the plan in `docs/superpowers/plans/2026-10-06-specdx-p1-core.md`.

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e vendor/tonomyx -e ".[dev]"
docker compose up -d                       # PostgreSQL on port 55434
bash scripts/verify_vendor.sh              # vendor/tonomyx is byte-identical to 34c1d2c
.venv/bin/python -m pytest                 # unit and DB tests (no model needed)

.venv/bin/specdx build-tell                # cache the Tell corpus
.venv/bin/specdx capability qwen3:14b llama3.1:8b
.venv/bin/specdx make-variants --model qwen3:14b --count 2
.venv/bin/specdx mutants
.venv/bin/specdx gate --config experiments/p1-gate.json --out results/p1-gate.json
```

Rules: `vendor/` is never edited, mutants live in `work/`, nothing is imported from FAA.
````

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/interpreters/test_paraphrase.py tests/eval/test_pilot.py tests/test_workbench.py tests/test_cli.py -q`
Expected: `8 passed`.

- [ ] **Step 9: Run the whole suite and the linter**

Run: `.venv/bin/python -m pytest && .venv/bin/ruff check .`
Expected: every test passes; ruff prints `All checks passed!`.

- [ ] **Step 10: Commit**

```bash
git add specdx/workbench.py specdx/interpreters/paraphrase.py specdx/eval/pilot.py specdx/cli.py experiments README.md tests
git commit -m "feat: workbench, capability check, Tell paraphrases, gate runner and CLI"
```

---

### Task 20: Run the capability pilot and the P1 go/no-go gate

This task runs real models. There is no new code. Each step records its output in `results/`. **Do not run the test suite while a gate run is in progress:** both rebuild the database.

**Files:**
- Create: `results/p1-capability.md`, `results/p1-mutants.tsv`, `results/p1-gate.json`, `results/p1-gate.md`
- Modify: `specdx/interpreters/variants.json` (variants 1–2), and `experiments/p1-gate.json` (only if the capability pilot rules a family out)

- [ ] **Step 1: Pull candidate tool-capable models (spec §4.2)**

```bash
ollama pull mistral-nemo:12b && ollama pull granite3.3:8b && ollama pull phi4-mini && ollama pull command-r7b
```
Expected: each ends with `success`. About 25 GB are needed; check `df -h /home` first, which had 164 GB free on 2026-10-06.

- [ ] **Step 2: Capability pilot**

```bash
mkdir -p results
.venv/bin/specdx capability qwen3:14b qwen2.5:14b-instruct-q4_K_M qwen2.5:7b-instruct-q4_K_M \
  llama3.1:8b mistral-nemo:12b granite3.3:8b phi4-mini command-r7b | tee results/p1-capability.md
```
Expected: a Markdown table. `qwen3:14b` and `llama3.1:8b` must be `usable: True` for the configuration in `experiments/p1-gate.json`. If `llama3.1:8b` is not usable, replace the `llama` family in that file with the first other usable family (for example `"mistral": ["mistral-nemo:12b"]`) and note the swap in the commit message.

- [ ] **Step 3: Generate and freeze Tell variants 1–2**

```bash
.venv/bin/specdx make-variants --model qwen3:14b --count 2
git diff --stat specdx/interpreters/variants.json
```
Expected: `variants: 3`, and the file changes. Read the two new standing instructions. Each must still contain all seven placeholders (`{role}` … `{draft}`) and add no business rule. If a paraphrase adds a rule (for example "để trống số tiền"), delete that variant's entries and rerun with `--count 2`.

- [ ] **Step 4: Build the Tell corpus and list the mutants**

```bash
.venv/bin/specdx build-tell
.venv/bin/specdx mutants | tee results/p1-mutants.tsv
wc -l results/p1-mutants.tsv
```
Expected: chunk counts for BA, DEMO and SAD; then `36 results/p1-mutants.tsv`.

- [ ] **Step 5: Smoke-run SpecDx on the base specification with a small budget**

```bash
jq '.budget = 24' experiments/p1-gate.json > work/smoke.json
.venv/bin/specdx run --variant base --config work/smoke.json --out work/smoke-base.json
```
Expected: the run finishes, `work/smoke-base.json` has `"runs": 24` or fewer, and `"theta"` lists every configured model. If every behaviour has status `error`, read the first `error` field in `work/runs.jsonl` and fix the cause (usually Ollama not running) before going on.

- [ ] **Step 6: Run the gate**

```bash
nohup .venv/bin/specdx gate --config experiments/p1-gate.json --out results/p1-gate.json \
  > work/gate.log 2>&1 &
```
Expected: it runs for many hours (21 specifications × up to 200 runs). It is resumable: if interrupted, run the same command again, and cached runs are reused. When it ends, `results/p1-gate.md` holds the verdict table.

- [ ] **Step 7: Record the decision and commit**

```bash
cat results/p1-gate.md
git add results experiments specdx/interpreters/variants.json
git commit -m "results: P1 capability pilot and go/no-go gate"
```
- If the table says **PASSED**, P1 is complete; report to your human partner and stop.
- If it says **NOT PASSED**, do not tune on the test split. Report the table, including the untyped scores, to your human partner. The spec (§4.8) then calls for revisiting §3.2–§3.4 before any full run.
