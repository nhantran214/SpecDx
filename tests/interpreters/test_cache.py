from specdx.interpreters.cache import RunCache, run_key
from specdx.types import Behaviour, InterpreterId, Scenario, State


def _b(key: str) -> Behaviour:
    return Behaviour(key, InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0), State({}))


def test_keys_are_stable_and_distinct() -> None:
    assert run_key("a", "b") == run_key("a", "b") != run_key("a", "c")


def test_the_cache_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    RunCache(path).put(_b("k1"))
    assert RunCache(path).get("k1") == _b("k1")


def test_a_truncated_last_line_is_skipped(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    cache = RunCache(path)
    cache.put(_b("k1"))
    cache.put(_b("k2"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_key": "k3", "interp')
    reloaded = RunCache(path)
    assert len(reloaded) == 2 and reloaded.get("k3") is None
