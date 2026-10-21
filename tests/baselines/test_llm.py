import json

import httpx
import pytest

from specdx.baselines.llm import DigestMismatch, Gateway
from specdx.types import InfrastructureError


def _server(digest="d1", fail=False):
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:7b", "digest": digest}]})
        if fail:
            raise httpx.ReadTimeout("slow", request=request)
        body = json.loads(request.content)
        assert body["options"]["num_ctx"] == 16384 and body["stream"] is False
        return httpx.Response(200, json={"message": {"content": "hi"}, "prompt_eval_count": 7,
                                         "eval_count": 3, "total_duration": 4_000_000})
    return httpx.MockTransport(handle), calls


def test_a_cache_hit_sends_no_chat_request(tmp_path) -> None:
    t, calls = _server()
    g = Gateway("http://x", tmp_path / "c.jsonl", transport=t)
    first = g.chat("m:7b", [{"role": "user", "content": "a"}], options={"seed": 1})
    again = Gateway("http://x", tmp_path / "c.jsonl", transport=t).chat(
        "m:7b", [{"role": "user", "content": "a"}], options={"seed": 1})
    assert (first.text, first.cost.requests, first.cost.prompt_tokens) == ("hi", 1, 7)
    assert again.cached and again.text == "hi"
    assert calls.count("/api/chat") == 1


def test_the_key_changes_with_the_digest(tmp_path) -> None:
    for d in ("d1", "d2"):
        t, _ = _server(digest=d)
        Gateway("http://x", tmp_path / "c.jsonl", transport=t).chat("m:7b", [])
    assert len((tmp_path / "c.jsonl").read_text().splitlines()) == 2


def test_a_timeout_is_cached_as_an_error_and_not_retried(tmp_path) -> None:
    t, calls = _server(fail=True)
    g = Gateway("http://x", tmp_path / "c.jsonl", transport=t)
    r1 = g.chat("m:7b", [])
    r2 = g.chat("m:7b", [])
    assert (r1.status, r1.cost.errors, r2.cached) == ("error", 1, True)
    assert calls.count("/api/chat") == 1


def test_a_missing_model_stops_before_any_chat(tmp_path) -> None:
    t, calls = _server()
    with pytest.raises(DigestMismatch, match="other:1b"):
        Gateway("http://x", tmp_path / "c.jsonl", transport=t).chat("other:1b", [])
    assert "/api/chat" not in calls


def test_a_pinned_digest_mismatch_stops(tmp_path) -> None:
    t, calls = _server(digest="new")
    with pytest.raises(DigestMismatch, match="m:7b"):
        Gateway("http://x", tmp_path / "c.jsonl", {"m:7b": "old"}, transport=t).chat("m:7b", [])
    assert "/api/chat" not in calls


def test_a_connect_timeout_is_infrastructure_and_never_cached(tmp_path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:7b", "digest": "d1"}]})
        raise httpx.ConnectTimeout("tunnel down", request=request)

    cache = tmp_path / "c.jsonl"
    g = Gateway("http://x", cache, transport=httpx.MockTransport(handle))
    with pytest.raises(InfrastructureError):
        g.chat("m:7b", [])
    assert not cache.exists() or cache.read_text() == ""


@pytest.mark.parametrize("content", [b"not json", b'{"no_message": 1}'])
def test_a_malformed_200_reply_is_cached_as_an_error(tmp_path, content) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:7b", "digest": "d1"}]})
        return httpx.Response(200, content=content)

    g = Gateway("http://x", tmp_path / "c.jsonl", transport=httpx.MockTransport(handle))
    r = g.chat("m:7b", [])
    assert (r.status, r.text, r.cost.requests, r.cost.errors) == ("error", "", 1, 1)
    assert r.error.startswith("malformed reply:")


def test_a_truncated_last_cache_line_is_skipped_but_an_earlier_one_raises(tmp_path) -> None:
    t, calls = _server()
    cache = tmp_path / "c.jsonl"
    Gateway("http://x", cache, transport=t).chat("m:7b", [])
    good = cache.read_text()
    cache.write_text(good + '{"key": "abc", "te')
    again = Gateway("http://x", cache, transport=t).chat("m:7b", [])
    assert again.cached and calls.count("/api/chat") == 1
    Gateway("http://x", cache, transport=t).chat("m:7b", [{"role": "user", "content": "b"}])
    assert len(Gateway("http://x", cache, transport=t)._cache) == 2  # reload is clean
    cache.write_text('{"key": "abc", "te\n' + good)
    with pytest.raises(ValueError, match="c.jsonl"):
        Gateway("http://x", cache, transport=t)


def test_a_reply_with_unicode_line_separators_survives_a_reload(tmp_path) -> None:
    text = "a\u2028b\u2029c\x85d"

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:7b", "digest": "d1"}]})
        return httpx.Response(200, json={"message": {"content": text}})

    t = httpx.MockTransport(handle)
    cache = tmp_path / "c.jsonl"
    assert Gateway("http://x", cache, transport=t).chat("m:7b", []).text == text
    again = Gateway("http://x", cache, transport=t).chat("m:7b", [])
    assert again.cached and again.text == text
