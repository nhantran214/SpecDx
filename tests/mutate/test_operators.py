import pytest
from tonomyx.canon.pack import CanonPack

from specdx.config import BASE_PACK
from specdx.mutate.operators import all_mutants, dev_split, witness_state
from specdx.orchestrate.work_orders import load_work_orders
from specdx.spec.clauses import ClauseSet
from specdx.spec.tell import TellCorpus
from specdx.types import Scenario, State
from specdx.variant import SpecVariant

STEPS = load_work_orders()
GRN12 = Scenario("grn12", 15, 12, 0)


@pytest.fixture(scope="module")
def mutants(pack, tmp_path_factory):
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))
    return {m.id: m for m in all_mutants(base, [], tmp_path_factory.mktemp("work"))}


@pytest.fixture(scope="module")
def base_clauses(pack) -> ClauseSet:
    return ClauseSet.from_pack(pack, STEPS)


def _clauses(m) -> ClauseSet:
    return ClauseSet.from_pack(CanonPack(m.variant.pack), STEPS)


def _done(golden: State) -> State:
    return State(golden.tables, golden.derived, {s.id: "done" for s in STEPS})


def test_there_are_thirty_six_mutants_and_a_stratified_dev_split(mutants) -> None:
    assert len(mutants) == 36
    dev = dev_split(list(mutants.values()), 5, 20261006)
    assert len(dev) == 20
    assert sorted({m.truth.type for m in dev}) == ["ambiguity", "contradiction", "hidden", "weak"]


@pytest.mark.db
def test_hidden_and_conflict_mutants_are_witnessed_by_the_canon_state(mutants, golden) -> None:
    for m in mutants.values():
        if m.operator not in ("C1", "H1", "H2"):
            continue
        v = _clauses(m).violations(_done(witness_state(m, golden)), GRN12)
        if m.operator == "C1":
            assert v == {f"inv:mut-c1-{m.site}"}, m.id
        else:
            assert v == m.truth.location, m.id


@pytest.mark.db
def test_weak_mutants_let_a_t_violating_state_pass(mutants, base_clauses, golden) -> None:
    for m in mutants.values():
        if m.operator not in ("W1", "W2"):
            continue
        witness = _done(witness_state(m, golden))
        assert _clauses(m).violations(witness, GRN12) == frozenset(), m.id
        assert base_clauses.violations(witness, GRN12), m.id


@pytest.mark.db
def test_guardrail_mutants_refuse_the_canon_state(db, mutants, golden) -> None:
    for m in mutants.values():
        if m.operator != "C2":
            continue
        db.use(m.variant.pack)
        assert db.materialize(golden) is None, m.id
    db.use(BASE_PACK)


def test_tell_mutants_change_the_corpus_and_the_kb(mutants) -> None:
    a1 = mutants["A1-00"]
    assert a1.variant.corpus.find("**mức tối đa 5%**.") == []
    assert a1.truth.location == frozenset({"KB:KB-0003"})
    kb = (a1.variant.pack / "kb" / "KB-0003.md").read_text(encoding="utf-8")
    assert "mức tối đa 5%" not in kb
    a2 = mutants["A2-01"]
    kb = (a2.variant.pack / "kb" / "KB-0003.md").read_text(encoding="utf-8")
    assert "**mức tối đa hợp lý**." in kb
    found_ids = a2.variant.corpus.find("**mức tối đa hợp lý**.")
    assert a2.truth.location == frozenset(found_ids) and found_ids != []
    a3 = mutants["A3-00"]
    assert "KB-0009" in CanonPack(a3.variant.pack).kb
    assert a3.truth.location != frozenset()
    assert a3.truth.location == frozenset(a3.variant.corpus.sentence_ids(["KB:KB-0009"]))


def test_every_tell_mutant_is_real(pack, tmp_path_factory) -> None:
    """Every A-series mutant has non-empty ground truth and different corpus."""
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))
    base_digest = base.corpus.digest()
    mutants = all_mutants(base, [], tmp_path_factory.mktemp("work"))
    for m in mutants:
        if m.operator.startswith("A"):
            assert m.truth.location != frozenset(), f"{m.id} has empty truth"
            assert m.variant.corpus.digest() != base_digest, f"{m.id} has same corpus as base"


def test_a_drifted_site_is_refused(tmp_path_factory, monkeypatch) -> None:
    """Drifted site strings that match nothing raise ValueError."""
    import specdx.mutate.operators as ops_module

    monkeypatch.setattr(ops_module, "A1_SITES", ("câu không tồn tại.",))
    monkeypatch.setattr(ops_module, "A2_SITES", ())
    monkeypatch.setattr(ops_module, "A3_ARTICLE", "")

    from tonomyx.canon.pack import CanonPack
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(CanonPack(BASE_PACK), []))

    with pytest.raises(ValueError, match="A1 site matches no Tell sentence"):
        ops_module.all_mutants(base, [], tmp_path_factory.mktemp("work"))


def test_the_base_pack_is_never_touched(mutants) -> None:
    text = (BASE_PACK / "kb" / "KB-0003.md").read_text(encoding="utf-8")
    assert "**mức tối đa 5%**" in text


def _stamps(root) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_ino, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_second_build_leaves_every_existing_pack_untouched(pack, tmp_path) -> None:
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))
    first = all_mutants(base, [], tmp_path)
    before = _stamps(tmp_path / "mutants")
    again = all_mutants(base, [], tmp_path)
    assert _stamps(tmp_path / "mutants") == before  # nothing rewritten, nothing deleted
    assert [m.variant.digest() for m in again] == [m.variant.digest() for m in first]
    assert all(m.variant.pack.is_relative_to(tmp_path / "mutants") for m in again)


def test_a_pack_that_differs_from_what_the_code_builds_is_refused(pack, tmp_path) -> None:
    base = SpecVariant("base", BASE_PACK, TellCorpus.build(pack, []))
    all_mutants(base, [], tmp_path)
    edited = tmp_path / "mutants" / "C1-00" / BASE_PACK.name / "invariants.json"
    edited.write_text(edited.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        all_mutants(base, [], tmp_path)
    assert edited.read_text(encoding="utf-8").endswith(" ")  # left as it was
