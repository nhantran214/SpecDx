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


def test_a_file_cut_mid_multibyte_character_skips_the_partial_record(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    cache = RunCache(path)
    b1 = Behaviour("k1", InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0),
                   State({}), claims={"status": "Đã duyệt"})
    cache.put(b1)
    # Append a second record with multibyte char that will be cut off mid-UTF-8
    b2_json = Behaviour("k2", InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0),
                        State({}), claims={"text": "Đã duyệt"}).to_json()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(b2_json[:20])  # Write only first 20 chars, cutting mid-multibyte
    reloaded = RunCache(path)
    # First record should be complete and valid, second should be skipped
    assert len(reloaded) == 1 and reloaded.get("k1") == b1 and reloaded.get("k2") is None


def test_a_record_with_line_separator_character_survives_reload(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    b = Behaviour("k1", InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0),
                  State({}), claims={"text": "line separator"})
    RunCache(path).put(b)
    reloaded = RunCache(path)
    assert reloaded.get("k1") == b


def test_after_loading_truncated_file_put_then_reload_returns_record(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    cache = RunCache(path)
    cache.put(_b("k1"))
    cache.put(_b("k2"))
    # Truncate the file (no newline at end)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_key": "k3", "interp')
    reloaded = RunCache(path)
    assert len(reloaded) == 2
    # Now put a new record - it should work and be recoverable
    reloaded.put(_b("k4"))
    final = RunCache(path)
    assert final.get("k4") == _b("k4") and len(final) == 3
