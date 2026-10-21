from pathlib import Path

from specdx.baselines.api import (
    REGISTRY,
    TELL_CHARS,
    BaselineOutput,
    above,
    ranked,
    register,
    scenario_text,
    tell_excerpt,
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
