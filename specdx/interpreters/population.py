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
