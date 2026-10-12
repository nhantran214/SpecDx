"""Every pre-registered value (spec v2 §3.10, experiments/p1b-preregistration.md) is the
value the code uses. Changing one after a P1b result is seen breaks this test."""

from pathlib import Path

import pytest

from specdx.eval import matching, observability, screen
from specdx.eval.pilot import GateConfig
from specdx.infer import act, latent, scores, sensitivity
from specdx.interpreters import harness
from specdx.scenarios import tguided

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(("value", "registered"), [
    (harness.MAX_NUDGES, 2),
    (screen.ACT_RATE_MIN, 0.6), (screen.VALID_RATE_MIN, 0.8),
    (len(screen.SCREEN_VARIANTS) * len(screen.SCREEN_SEEDS), 9),
    (screen.PER_FAMILY, 2), (screen.MAX_MODELS, 4), (screen.MIN_FAMILIES, 2),
    (act.RASCH_L2, 1.0), (act.OMEGA, 0.3), (act.M_MIN, 3),
    ((latent.ALPHA, latent.BETA), (7.0, 3.0)),
    (sensitivity.COMPETENT, 0.5), (scores.TAU_H, 0.5),
    (sensitivity.PERMUTATIONS, 2000), (sensitivity.MIN_BEHAVIOURS, 6), (sensitivity.BH_Q, 0.1),
    (tguided.MAX_TBOUNDARY, 12), (tguided.WINDOW, 40),
    (observability.SCORABLE_K, 3), (matching.MIN_SCORABLE, 3),
    (matching.ABSOLUTE_BAR, 0.4),
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
                   "mechanism check", "θ ≥ 0.5", "θ ordering"):
        assert needle in text, needle
