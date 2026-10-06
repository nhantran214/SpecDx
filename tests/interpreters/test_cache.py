from specdx.interpreters.cache import RunCache, run_key
from specdx.types import Behaviour, InterpreterId, Scenario, State


def _b(key: str) -> Behaviour:
    return Behaviour(key, InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0), State({}))


def _vb(key: str, text: str) -> Behaviour:
    return Behaviour(key, InterpreterId("m", "f", 0, 1), Scenario("s", 15, 12, 0), State({}),
                     claims={"x": text})


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


def test_a_line_cut_inside_a_multibyte_character_is_skipped(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    RunCache(path).put(_b("k1"))
    raw = _vb("k2", "Đã duyệt").to_json().encode("utf-8")
    cut = raw.index("Đ".encode()) + 1  # inside the two-byte "Đ"
    with path.open("ab") as handle:
        handle.write(raw[:cut])
    reloaded = RunCache(path)
    assert len(reloaded) == 1 and reloaded.get("k1") is not None


def test_records_containing_unicode_line_separators_survive_a_reload(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    cache = RunCache(path)
    cache.put(_vb("k1", "a b"))
    cache.put(_vb("k2", "c\u0085d"))
    reloaded = RunCache(path)
    assert reloaded.get("k1").claims == {"x": "a b"}
    assert reloaded.get("k2").claims == {"x": "c\u0085d"}


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
