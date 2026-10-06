"""Runtime configuration, read once from the environment.

The canon pack is a read-only input. If it is missing the process must refuse
to start rather than fall back to invented data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CANON_PACK = (
    REPO_ROOT / "data" / "canon_pack_CASE-0031_v2.0" / "canon_pack_CASE-0031"
)

DEFAULT_DATABASE_URL = "postgresql://tonomyx:tonomyx@localhost:55432/tonomyx"


@dataclass(frozen=True)
class Settings:
    """Everything the process needs from its environment."""

    canon_pack: Path
    database_url: str
    smtp_host: str
    smtp_port: int

    @classmethod
    def from_env(cls) -> Settings:
        pack = Path(os.environ.get("TONOMYX_CANON_PACK", DEFAULT_CANON_PACK))
        # canon.meta.json is the marker file: a directory without it is not a pack.
        if not (pack / "canon.meta.json").is_file():
            raise FileNotFoundError(
                f"not a canon pack: {pack} has no canon.meta.json"
            )
        return cls(
            canon_pack=pack,
            database_url=os.environ.get("TONOMYX_DATABASE_URL", DEFAULT_DATABASE_URL),
            smtp_host=os.environ.get("TONOMYX_SMTP_HOST", "localhost"),
            smtp_port=int(os.environ.get("TONOMYX_SMTP_PORT", "1025")),
        )
