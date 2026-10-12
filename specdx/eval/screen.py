"""The capability screen (spec v2 §2.2): which candidate models enter the population.

It measures the capability to act, never agreement with C: scoring against C would be
circular while the specification under diagnosis may itself be defective. Each candidate
runs the base specification on the canon scenario happy, Tell variants 0-2 x seeds 1-3,
through the harness. A model is admitted with act rate ≥ 0.6 and valid-call rate ≥ 0.8.
The population keeps the best admitted models, at most 2 per family and 4 in all, and
needs 2 families; with fewer, P1b stops and the thresholds are never lowered.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from specdx.config import REPO_ROOT
from specdx.infer.act import Act
from specdx.types import Behaviour, InterpreterId, Scenario
from specdx.variant import SpecVariant

SCREEN_SCENARIO = "happy"
SCREEN_VARIANTS: tuple[int, ...] = (0, 1, 2)
SCREEN_SEEDS: tuple[int, ...] = (1, 2, 3)
ACT_RATE_MIN = 0.6  # spec v2 §3.10
VALID_RATE_MIN = 0.8
PER_FAMILY = 2
MAX_MODELS = 4
MIN_FAMILIES = 2
PROFILES = REPO_ROOT / "experiments" / "profiles"


class ScreenRunner(Protocol):
    def model_digest(self, model: str) -> str: ...

    def run(
        self, variant: SpecVariant, scenario: Scenario, interp: InterpreterId, suffix: str = "",
        *, kind: str = "population",
    ) -> Behaviour: ...


class ActLens(Protocol):
    def acts(self, behaviour: Behaviour) -> list[Act]: ...


@dataclass(frozen=True)
class Profile:
    """A hardware profile (spec v2 §2.3): candidate models by family and the suggested
    parallelism. It only widens the candidate list; the screen decides admission."""

    name: str
    workers: int
    ollama_num_parallel: int
    candidates: dict[str, tuple[str, ...]]


def load_profile(name_or_path: str) -> Profile:
    """A profile by name (experiments/profiles/<name>.json) or by path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = PROFILES / f"{name_or_path}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Profile(raw["profile"], int(raw["workers"]), int(raw["ollama_num_parallel"]),
                   {f: tuple(ms) for f, ms in raw["candidates"].items()})


@dataclass(frozen=True)
class ScreenResult:
    model: str
    family: str
    runs: int
    act_rate: float
    valid_rate: float
    note: str = ""

    @property
    def admitted(self) -> bool:
        return self.act_rate >= ACT_RATE_MIN and self.valid_rate >= VALID_RATE_MIN


@dataclass
class Screen:
    results: list[ScreenResult]
    population: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def stop(self) -> bool:
        """The stop rule: fewer than MIN_FAMILIES families were admitted."""
        return len(self.population) < MIN_FAMILIES

    def to_json(self) -> str:
        return json.dumps({
            "population": {f: list(ms) for f, ms in self.population.items()},
            "stop": self.stop,
            "rules": {"act_rate_min": ACT_RATE_MIN, "valid_rate_min": VALID_RATE_MIN,
                      "per_family": PER_FAMILY, "max_models": MAX_MODELS,
                      "min_families": MIN_FAMILIES, "scenario": SCREEN_SCENARIO,
                      "variants": list(SCREEN_VARIANTS), "seeds": list(SCREEN_SEEDS)},
            "results": [asdict(r) | {"admitted": r.admitted} for r in self.results],
        }, ensure_ascii=False, indent=2)


def screen_model(
    runner: ScreenRunner, variant: SpecVariant, lens: ActLens, scenario: Scenario,
    model: str, family: str,
) -> ScreenResult:
    """Nine runs of one model; a model Ollama does not list is recorded, not admitted."""
    try:
        runner.model_digest(model)
    except KeyError:
        return ScreenResult(model, family, 0, 0.0, 0.0, "not installed")
    behaviours = [runner.run(variant, scenario, InterpreterId(model, family, v, s), kind="screen")
                  for v in SCREEN_VARIANTS for s in SCREEN_SEEDS]
    acts = [a for b in behaviours for a in lens.acts(b)]
    counts = [c for b in behaviours for c in b.harness.values()]
    calls = sum(c.get("tool_calls", 0) for c in counts)
    valid = sum(c.get("valid_calls", 0) for c in counts)
    act_rate = sum(a.acted for a in acts) / len(acts) if acts else 0.0
    return ScreenResult(model, family, len(behaviours), act_rate, valid / calls if calls else 0.0)


def choose_population(results: Sequence[ScreenResult]) -> dict[str, tuple[str, ...]]:
    """Admitted models by act rate, then valid-call rate, then name: at most PER_FAMILY
    per family and MAX_MODELS in all."""
    ranked = sorted((r for r in results if r.admitted),
                    key=lambda r: (-r.act_rate, -r.valid_rate, r.model))
    chosen: dict[str, list[str]] = {}
    per_family: Counter[str] = Counter()
    for r in ranked:
        if per_family[r.family] >= PER_FAMILY or sum(per_family.values()) >= MAX_MODELS:
            continue
        per_family[r.family] += 1
        chosen.setdefault(r.family, []).append(r.model)
    return {f: tuple(ms) for f, ms in chosen.items()}


def run_screen(
    runner: ScreenRunner, variant: SpecVariant, lens: ActLens, canon: Mapping[str, Scenario],
    candidates: Mapping[str, Sequence[str]],
) -> Screen:
    scenario = canon[SCREEN_SCENARIO]
    results = [screen_model(runner, variant, lens, scenario, m, f)
               for f, models in candidates.items() for m in models]
    return Screen(results, choose_population(results))


def screen_markdown(screen: Screen, profile: str) -> str:
    chosen = {m for ms in screen.population.values() for m in ms}
    lines = [
        f"# P1b capability screen ({profile})", "",
        f"Base specification, scenario `{SCREEN_SCENARIO}`, Tell variants {list(SCREEN_VARIANTS)}"
        f" x seeds {list(SCREEN_SEEDS)}. Admission: act rate ≥ {ACT_RATE_MIN} and valid-call "
        f"rate ≥ {VALID_RATE_MIN}. Population: at most {PER_FAMILY} per family and "
        f"{MAX_MODELS} in all, at least {MIN_FAMILIES} families.", "",
        "| model | family | runs | act rate | valid-call rate | admitted | chosen | note |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for r in screen.results:
        lines.append(f"| {r.model} | {r.family} | {r.runs} | {r.act_rate:.3f} | "
                     f"{r.valid_rate:.3f} | {'yes' if r.admitted else 'no'} | "
                     f"{'yes' if r.model in chosen else ''} | {r.note} |")
    lines += ["", "## Population", ""]
    if screen.stop:
        lines.append(f"STOP: fewer than {MIN_FAMILIES} families were admitted. P1b stops and "
                     "reports; the thresholds are never lowered.")
    else:
        lines += [f"- {family}: {', '.join(models)}"
                  for family, models in screen.population.items()]
    return "\n".join(lines) + "\n"
