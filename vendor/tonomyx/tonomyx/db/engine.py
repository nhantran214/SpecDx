"""PostgreSQL connection handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from tonomyx.config import Settings


@contextmanager
def connect(settings: Settings) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    """Open one connection that yields rows as dicts keyed by canon column names."""
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        yield conn
