import pytest
from benchguard.core.finding import Evidence, Finding, Location
from benchguard.taxonomy.categories import ErrorCategory, Severity

from specdx.baselines.benchguard import MAPPING, map_finding


def _f(sub, cat, files=(), text="", conf=0.9, sev=Severity.HIGH):
    loc = Location(file=files[0]) if files else None
    loc2 = Location(file=files[1]) if len(files) > 1 else None
    return Finding(task_id="happy", category=cat, subcategory=sub, severity=sev, title="t",
                   description=text, confidence=conf,
                   evidence=Evidence(primary_source=loc, conflicting_source=loc2))


FILES = {"tests/inv__po-15.txt": "inv:po-15", "tests/inv__grn-12.txt": "inv:grn-12"}


def test_every_mapped_subcategory_is_known() -> None:
    assert set(MAPPING) >= {"INST-INCOMPLETE", "INST-CONTRADICT", "INST-INFEASIBLE", "GT-LOGIC",
                            "GT-DATA", "GT-FMT", "EVAL-MISMATCH", "EVAL-COVERAGE",
                            "EVAL-JUDGE-BIAS", "EVAL-TOLERANCE"}


def test_gt_maps_to_hidden_at_the_cited_clause() -> None:
    [d] = map_finding(_f("GT-LOGIC", ErrorCategory.GT, ["tests/inv__po-15.txt"]), FILES, set())
    assert (d.type, d.location) == ("hidden", ("inv:po-15",)) and d.score == 3 * 3


def test_contradiction_needs_two_cited_clauses() -> None:
    two = map_finding(_f("INST-CONTRADICT", ErrorCategory.INST,
                         ["tests/inv__po-15.txt", "tests/inv__grn-12.txt"]), FILES, set())
    one = map_finding(_f("INST-CONTRADICT", ErrorCategory.INST, ["tests/inv__po-15.txt"]),
                      FILES, set())
    assert two[0].type == "contradiction" and two[0].location == ("inv:grn-12", "inv:po-15")
    assert one[0].type == "hidden"


def test_a_lenient_eval_also_reports_a_weak_check_on_the_column() -> None:
    files = {"tests/golden__po__PO-0015__don_gia.txt": "golden:po:PO-0015:don_gia"}
    ds = map_finding(_f("EVAL-MISMATCH", ErrorCategory.EVAL,
                        ["tests/golden__po__PO-0015__don_gia.txt"],
                        text="The check accepts any price"), files, set())
    assert {(d.type, d.location) for d in ds} == {
        ("hidden", ("golden:po:PO-0015:don_gia",)), ("weak", ("po.don_gia",))}


def test_incomplete_instructions_map_to_cited_sentences() -> None:
    ds = map_finding(_f("INST-INCOMPLETE", ErrorCategory.INST,
                        text="[T:KB:KB-0003:02] does not say the cap"), {}, {"KB:KB-0003:02"})
    assert [(d.type, d.location) for d in ds] == [("ambiguity", ("KB:KB-0003:02",))]


def test_env_findings_are_dropped() -> None:
    assert map_finding(_f("ENV-DEP", ErrorCategory.ENV), FILES, set()) == []


def test_a_clause_is_cited_by_its_file_name_or_its_id() -> None:
    files = {**FILES, "tests/inv__po-1.txt": "inv:po-1"}
    by_name = _f("GT-LOGIC", ErrorCategory.GT, ["inv__po-15.txt"])
    by_id = _f("GT-LOGIC", ErrorCategory.GT)
    by_id.evidence.primary_source = Location(file="eval_script", snippet="inv:po-15\npo ...")
    by_id_in_file = _f("GT-LOGIC", ErrorCategory.GT, ["eval_script (inv:po-15)"])
    for f in (by_name, by_id, by_id_in_file):
        assert [d.location for d in map_finding(f, files, set())] == [("inv:po-15",)]


def test_a_gateway_outage_is_not_swallowed_as_a_protocol_error(tmp_path) -> None:
    from benchguard.agent.runner import ProtocolRunner
    from benchguard.core.context import VerificationContext
    from benchguard.protocols import DefinitionProtocol

    from specdx.baselines.benchguard import GatewayClient
    from specdx.types import InfrastructureError

    class Down:
        cache_path = tmp_path / "c.jsonl"

        def chat(self, model, messages, **kw):
            raise InfrastructureError("Ollama down")

    client = GatewayClient(Down(), "m")
    ctx = VerificationContext(task_id="t", benchmark="b", task_instruction="do it",
                              gold_program_source="x", eval_script_source="y")
    result = ProtocolRunner(client=client).run(DefinitionProtocol(), ctx)
    assert result.error and result.findings == []  # BenchGuard swallows it ...
    assert isinstance(client.fatal, InfrastructureError)  # ... the client keeps it
    with pytest.raises(InfrastructureError):
        client.complete("again")


def _input(gateway, tell_corpus, pack, golden):
    from types import SimpleNamespace

    from specdx.config import BASE_PACK
    from specdx.interpreters.prompt import load_variants
    from specdx.orchestrate.work_orders import load_work_orders
    from specdx.scenarios.space import canon_scenarios
    from specdx.spec.clauses import ClauseSet
    from specdx.variant import SpecVariant

    steps = load_work_orders()
    return SimpleNamespace(
        gateway=gateway, variant=SpecVariant("base", BASE_PACK, tell_corpus),
        ctx=SimpleNamespace(canon=canon_scenarios(pack)), clauses=ClauseSet.from_pack(pack, steps),
        steps=steps, statements=load_variants().statements, golden=golden)


def test_a_malformed_llm_reply_yields_no_finding(tmp_path, tell_corpus, pack, golden) -> None:
    from specdx.baselines.api import REGISTRY
    from specdx.baselines.llm import Reply
    from specdx.cost import Cost

    class G:
        cache_path = tmp_path / "c.jsonl"

        def chat(self, model, messages, **kw):
            return Reply("not json at all", Cost(1), "ok")

    out = REGISTRY["benchguard"].run(_input(G(), tell_corpus, pack, golden), model="m:7b")
    assert out.defects == [] and out.cost.requests >= 1
    assert (tmp_path / "baselines" / "benchguard" / "base" / "m_7b" / "grn12").is_dir()


def test_only_the_anchor_scenario_is_packaged(tmp_path, tell_corpus, pack, golden) -> None:
    import json

    from specdx.baselines.benchguard import package

    inp = _input(None, tell_corpus, pack, golden)
    assert len(inp.ctx.canon) > 1  # the canon has other scenarios, which are not packaged
    files = package(inp, tmp_path / "pkg")
    assert [p.name for p in (tmp_path / "pkg").iterdir()] == [inp.clauses.anchor.id]
    task = tmp_path / "pkg" / inp.clauses.anchor.id
    assert json.loads((task / "solution" / "golden_state.json").read_text()) == golden.tables
    assert set(files.values()) == {c.id for c in inp.clauses
                                   if inp.clauses.applies(c, inp.clauses.anchor)}


@pytest.mark.parametrize("error", ["infrastructure", "digest"])
def test_an_outage_or_digest_mismatch_stops_the_run(tmp_path, tell_corpus, pack, golden,
                                                    error) -> None:
    from specdx.baselines.api import REGISTRY
    from specdx.baselines.llm import DigestMismatch
    from specdx.types import InfrastructureError

    exc = {"infrastructure": InfrastructureError("Ollama down"),
           "digest": DigestMismatch("m has digest x, the population runs used y")}[error]

    class G:
        cache_path = tmp_path / "c.jsonl"
        calls = 0

        def chat(self, model, messages, **kw):
            G.calls += 1
            raise exc

    with pytest.raises(type(exc)):
        REGISTRY["benchguard"].run(_input(G(), tell_corpus, pack, golden), model="m")
    assert G.calls == 1  # later tasks fail fast in the client, never reach the gateway
