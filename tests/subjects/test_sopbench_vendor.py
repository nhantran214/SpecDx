import json

from specdx.config import REPO_ROOT
from specdx.subjects.sopbench import DOMAINS, SOPBENCH_COMMIT, SOPBENCH_DIR

TASKS = {"bank": 134, "dmv": 97, "healthcare": 124, "hotel": 195, "library": 66,
         "online_market": 172, "university": 42}


def test_vendored_sopbench_has_every_domain_with_its_task_count():
    for d in DOMAINS:
        raw = json.loads((SOPBENCH_DIR / "data" / f"{d}_tasks.json").read_text(encoding="utf-8"))
        assert sum(len(v) for v in raw.values()) == TASKS[d]
    assert sum(TASKS.values()) == 830


def test_vendored_sopbench_carries_both_licences_and_no_output():
    assert "MIT" in (SOPBENCH_DIR / "LICENSE").read_text(encoding="utf-8")
    assert "CC BY 4.0" in (SOPBENCH_DIR / "DATA_LICENSE").read_text(encoding="utf-8")
    assert not (SOPBENCH_DIR / "output").exists()


def test_the_pin_is_recorded():
    assert SOPBENCH_COMMIT in (REPO_ROOT / "vendor" / "VENDORED.md").read_text(encoding="utf-8")
