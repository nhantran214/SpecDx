"""Every pre-registered value (spec v2 §3.10, experiments/p1b-preregistration.md) is the
value the code uses. Changing one after a P1b result is seen breaks this test. Spec v3
amends three: the screen's act rate (§7: 0.3), and S1's threshold and P(|A| = 1), which
left the loop with stage 2 (§6); v2's stage 2 keeps P(|A| = 1) = 0.9 as a baseline."""

import inspect
from pathlib import Path

import pytest

from specdx.active.loop import SpecDxConfig
from specdx.eval import matching, observability, screen
from specdx.eval.pilot import GateConfig
from specdx.infer import act, latent, scores, sensitivity
from specdx.interpreters import executor, harness
from specdx.scenarios import tguided

ROOT = Path(__file__).parents[1]
_CFG = SpecDxConfig.__dataclass_fields__


@pytest.mark.parametrize(("value", "registered"), [
    (harness.MAX_NUDGES, 2),
    (screen.ACT_RATE_MIN, 0.3), (screen.VALID_RATE_MIN, 0.8),
    (len(screen.SCREEN_VARIANTS) * len(screen.SCREEN_SEEDS), 9),
    (screen.PER_FAMILY, 2), (screen.MAX_MODELS, 4), (screen.MIN_FAMILIES, 2),
    (act.RASCH_L2, 1.0), (act.OMEGA, 0.3), (act.M_MIN, 3),
    ((latent.ALPHA, latent.BETA), (7.0, 3.0)),
    (sensitivity.COMPETENT, 0.5), (scores.TAU_H, 0.5),
    (sensitivity.PERMUTATIONS, 2000), (sensitivity.MIN_BEHAVIOURS, 6), (sensitivity.BH_Q, 0.1),
    (tguided.MAX_TBOUNDARY, 12), (tguided.WINDOW, 40),
    (observability.SCORABLE_K, 3), (matching.MIN_SCORABLE, 3),
    (matching.ABSOLUTE_BAR, 0.4),
    (inspect.signature(act.fit_rasch).parameters["tol"].default, 1e-6),
    (inspect.signature(act.fit_rasch).parameters["max_iter"].default, 100),
    (_CFG["k_max"].default, 6),
    (inspect.signature(latent.fit).parameters["p_single"].default, 0.9),
    (_CFG["tau"].default, 0.05),
    (executor.TEMPERATURE, 0.7),
    (inspect.signature(matching.gate_verdict).parameters["required"].default, 3),
    (screen.SCREEN_SCENARIO, "happy"),
    (screen.SCREEN_VARIANTS, (0, 1, 2)), (screen.SCREEN_SEEDS, (1, 2, 3)),
])
def test_the_code_uses_the_pre_registered_values(value, registered) -> None:
    assert value == registered


def test_the_gate_configuration_is_the_p1_split_and_budgets_on_the_screened_population(
    monkeypatch,
) -> None:
    monkeypatch.chdir(ROOT)  # the screen path is relative to the repository root
    config = GateConfig.load(ROOT / "experiments" / "p1b-gate.json")
    p1 = GateConfig.load(ROOT / "experiments" / "p1-gate.json")
    assert (config.dev_per_type, config.rng_seed) == (p1.dev_per_type, p1.rng_seed)
    assert (config.budget, config.probe_budget, config.localize_budget) == (200, 100, 80)
    assert (config.variants, config.seeds) == ((0, 1, 2), (1, 2, 3))
    assert config.screen == Path("results/p1b-screen.json")
    if config.models_by_family:
        config.check_population()  # frozen from the screen: it must equal its population
    else:
        with pytest.raises(ValueError, match="empty"):  # before the screen has run
            config.check_population()


def test_the_pre_registration_states_each_value() -> None:
    text = (ROOT / "experiments" / "p1b-preregistration.md").read_text(encoding="utf-8")
    for needle in ("Beta(7, 3)", "τ_H = 0.5", "ω = 0.3", "m_min = 3", "λ = 1.0", "2000",
                   "q = 0.1", "k = 3", "≥ 0.6", "≥ 0.8", "at most 12", "GO criterion",
                   "mechanism check", "θ ≥ 0.5", "θ ordering",
                   "population 200, probe 100, localise 80", "≤ 2 per family", "≤ 4 in all",
                   "temperature 0.7", "K_max = 6", "τ = 0.05", "at least one acted behaviour",
                   "Run weight", "NO-GO on all 21 specifications", "0.961", "0.743", "24 items",
                   "0.918 against 0.515", "0.75 against 0.507", "Proposition 2′",
                   "(a) On the base", "(c) At least 50%", "e66bfa5", "0d30803",
                   "counts acted views only"):
        assert needle in text, needle
