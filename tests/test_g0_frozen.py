"""V3Params carries the values G0 chose (spec v3 §9), and G0 passed."""

import dataclasses
import json
from pathlib import Path

from specdx.infer import anchored
from specdx.infer.anchored import V3Params

ROOT = Path(__file__).parents[1]


def test_v3params_are_the_values_g0_chose_and_froze() -> None:
    chosen = json.loads((ROOT / "experiments" / "p2-g0-params.json").read_text(encoding="utf-8"))
    g0 = json.loads((ROOT / "results" / "p2-g0.json").read_text(encoding="utf-8"))
    assert chosen["verdict"] == g0["verdict"] == "PASS"
    assert anchored.PARAMS_FROZEN
    assert dataclasses.asdict(V3Params()) == chosen["params"] == g0["chosen"]["params"]
