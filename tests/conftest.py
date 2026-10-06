"""Shared fixtures. The database fixture always hands back the base specification."""

from __future__ import annotations

import pytest
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.db import PackDatabase
from specdx.types import State

_DB = PackDatabase()


@pytest.fixture(scope="session")
def pack() -> CanonPack:
    return CanonPack(BASE_PACK)


@pytest.fixture
def db() -> PackDatabase:
    """The one database, loaded with the base pack (cheap when it already is)."""
    _DB.use(BASE_PACK)
    return _DB


@pytest.fixture
def golden(db: PackDatabase) -> State:
    return db.golden_state()
