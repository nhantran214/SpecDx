"""Shared fixtures. The database fixture always hands back the base specification."""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from tonomyx.canon.pack import CanonPack

from specdx import db as db_module
from specdx.config import BASE_PACK, PDF_DIR
from specdx.db import PackDatabase
from specdx.spec.tell import TellCorpus, pdf_chunks
from specdx.types import State

_DB = PackDatabase()


@pytest.fixture(scope="session")
def pack() -> CanonPack:
    return CanonPack(BASE_PACK)


@pytest.fixture(scope="session")
def tell_corpus(pack: CanonPack, tmp_path_factory: pytest.TempPathFactory) -> TellCorpus:
    """The base specification's real Tell corpus, its PDFs extracted into a temporary cache."""
    return TellCorpus.build(pack, pdf_chunks(PDF_DIR, tmp_path_factory.mktemp("tell")))


@pytest.fixture
def db() -> PackDatabase:
    """The one database, loaded with the base pack (cheap when it already is)."""
    _DB.use(BASE_PACK)
    return _DB


@pytest.fixture
def golden(db: PackDatabase) -> State:
    return db.golden_state()


@pytest.fixture
def lock_held_elsewhere(db: PackDatabase) -> Iterator[None]:
    """Another session holds the SpecDx lock, as a second process would. Never blocks:
    if the lock cannot be taken, something outside the tests holds it."""
    held = db_module._LOCKS.pop(db.url, None)  # this process lets go of it first
    if held is not None:
        held.close()
    with psycopg.connect(db.url, autocommit=True) as other:
        got = other.execute("SELECT pg_try_advisory_lock(%s)", (db_module.LOCK_KEY,)).fetchone()
        if not got or not got[0]:
            pytest.fail("could not take the SpecDx lock in a second session: another process "
                        "(a SpecDx run?) holds the database")
        yield
    db.use(BASE_PACK)  # the other session has ended: take the lock back
