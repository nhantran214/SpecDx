"""Paths and endpoints, read once from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from tonomyx.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR = REPO_ROOT / "vendor" / "tonomyx"
BASE_PACK = VENDOR / "data" / "canon_pack_CASE-0031_v2.0" / "canon_pack_CASE-0031"
PDF_DIR = VENDOR / "demo"
WORK_DIR = Path(os.environ.get("SPECDX_WORK", str(REPO_ROOT / "work")))
DATABASE_URL = os.environ.get(
    "SPECDX_DATABASE_URL", "postgresql://tonomyx:tonomyx@localhost:55434/tonomyx"
)
# One Ollama host or several, comma-separated (spec v2 §5.1): workers use them round robin.
OLLAMA_URLS: tuple[str, ...] = tuple(
    u.strip().rstrip("/")
    for u in os.environ.get("SPECDX_OLLAMA_URL", "http://localhost:11434").split(",")
    if u.strip()
)
OLLAMA_URL = OLLAMA_URLS[0]


def settings_for(pack: Path, url: str = DATABASE_URL) -> Settings:
    """TonomyX settings pointing at one specification and the SpecDx database."""
    if not (pack / "canon.meta.json").is_file():
        raise FileNotFoundError(f"not a canon pack: {pack} has no canon.meta.json")
    return Settings(canon_pack=pack, database_url=url, smtp_host="localhost", smtp_port=1025)
