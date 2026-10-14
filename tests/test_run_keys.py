"""Spec v3 §0: v3 changes nothing an interpreter sees, so run keys are unchanged. The P1b
screen and smoke runs (schema 3) stay cache hits for new runs, and P1's schema-2 runs stay
usable for offline replay only."""

from specdx.interpreters.executor import interpreter_settings
from specdx.interpreters.prompt import load_variants
from specdx.observe.run import RUN_SCHEMA_VERSION, context_digest
from specdx.workbench import tell_payload

# The context digest every P1b screen and smoke run was keyed under (work/runs.jsonl lines
# 6319-6505 map to the smoke population with it).
P1B_CONTEXT = "9c796b06f15fd7945d95c1e080b072a17ed2cfbb1ccc14ce28293e786d53ff7a"


def test_the_run_schema_and_the_context_digest_are_the_ones_p1b_cached_under() -> None:
    assert RUN_SCHEMA_VERSION == 3
    assert context_digest(tell_payload(load_variants()), interpreter_settings()) == P1B_CONTEXT
