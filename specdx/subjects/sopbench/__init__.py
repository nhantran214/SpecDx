"""SOPBench, vendored offline (design §13d)."""

from pathlib import Path

from specdx.config import REPO_ROOT

SOPBENCH_COMMIT = "8790ed7406acbd22270f90bd1c4038de3fc5db8f"
SOPBENCH_DIR: Path = REPO_ROOT / "vendor" / "sopbench"
DOMAINS: tuple[str, ...] = (
    "bank", "dmv", "healthcare", "hotel", "library", "online_market", "university",
)
