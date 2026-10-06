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
