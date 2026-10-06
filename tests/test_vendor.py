from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK, PDF_DIR, settings_for


def test_the_vendored_package_reads_the_vendored_pack() -> None:
    pack = CanonPack(BASE_PACK)
    assert len(pack.invariants) == 42
    assert pack.meta["canon_version"] == "CASE-0031-ADV-2.0"


def test_the_three_requirement_documents_are_vendored() -> None:
    assert len(sorted(PDF_DIR.glob("*.pdf"))) == 3


def test_settings_point_at_the_specdx_database() -> None:
    assert settings_for(BASE_PACK).database_url.endswith(":55434/tonomyx")
