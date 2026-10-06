import httpx
import pytest

from specdx.workbench import ollama_digests


def _tags(calls: list[str]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"models": [
            {"name": "phi4-mini:latest", "digest": "aaa"},
            {"name": "qwen3:14b", "digest": "bbb"},
        ]})

    return httpx.MockTransport(handle)


def test_an_untagged_model_resolves_to_its_latest_digest() -> None:
    lookup = ollama_digests("http://x", _tags([]))
    assert lookup("phi4-mini") == "aaa" and lookup("qwen3:14b") == "bbb"


def test_an_unlisted_model_is_an_error() -> None:
    lookup = ollama_digests("http://x", _tags([]))
    with pytest.raises(KeyError, match="llama3.1:8b"):
        lookup("llama3.1:8b")


def test_the_tags_endpoint_is_read_once_and_only_on_first_use() -> None:
    calls: list[str] = []
    lookup = ollama_digests("http://x", _tags(calls))
    assert calls == []
    lookup("phi4-mini")
    lookup("qwen3:14b")
    assert calls == ["/api/tags"]


def test_an_unreachable_ollama_is_an_error() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):
        ollama_digests("http://x", httpx.MockTransport(down))("qwen3:14b")
