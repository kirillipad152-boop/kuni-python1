import asyncio
import json
import tomllib

import httpx
import pytest

from kuni import config as cfgmod
from kuni.config import Endpoint, EndpointAndModel
from kuni.llm import LLMError, OpenAIChat, Params, Session, backoff
from test_llm_tools_notifications import delta, sse


def pool():
    endpoint = cfgmod.config().llm.endpoint
    endpoint.bearerKey = "first"
    endpoint.extraBearerKeys = ("second", "first", "second")
    return cfgmod.config().resolved_llm().endpoint


def test_key_pool_config_roundtrip_inheritance_and_separate_provider():
    config = cfgmod.parse({"general": {"llm": {"endpoint": {
        "baseUrl": "https://example.com/v1/", "bearerKey": "main", "extraBearerKeys": ["other"]}}}})
    rendered = cfgmod.render(config)
    assert cfgmod.parse(tomllib.loads(rendered)).llm.endpoint.extraBearerKeys == ("other",)
    assert config.resolve(EndpointAndModel()).endpoint.bearer_keys() == ["main", "other"]
    assert config.resolve(EndpointAndModel(Endpoint("https://different.example/v1/"))).endpoint.bearer_keys() == []
    assert config.resolve(EndpointAndModel(Endpoint(bearerKey="own"))).endpoint.bearer_keys() == ["own"]
    assert Endpoint("", "same", ("same", "next", "next")).bearer_keys() == ["same", "next"]


@pytest.mark.parametrize("extra", ["a-key", [""], [1], ["line\nbreak"], ["space key"], None])
def test_invalid_pool_config_does_not_echo_credentials(extra):
    with pytest.raises(ValueError, match="extraBearerKeys must be an array") as error:
        cfgmod.parse({"general": {"llm": {"endpoint": {"extraBearerKeys": extra}}}})
    assert "line" not in str(error.value) and "space key" not in str(error.value)


@pytest.mark.parametrize("env,value", [("KUNI_API_KEY", "override"), ("KUNI_BASE_URL", "https://new.example/v1/")])
def test_env_override_does_not_silently_use_file_pool(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    config = cfgmod.parse({"general": {"llm": {"endpoint": {"bearerKey": "old", "extraBearerKeys": ["extra"]}}}})
    assert config.llm.endpoint.extraBearerKeys == ()


@pytest.mark.parametrize("streaming", [False, True])
async def test_429_tries_ready_key_without_sleeping_or_changing_request(monkeypatch, streaming, caplog):
    endpoint = pool()
    requests, events = [], []

    async def unexpected_sleep(_):
        pytest.fail("an unused key is ready")

    monkeypatch.setattr(asyncio, "sleep", unexpected_sleep)

    def handler(request):
        requests.append(request)
        if request.headers["authorization"] == "Bearer first":
            return httpx.Response(429, text="limit", headers={"retry-after": "120"})
        if streaming:
            return httpx.Response(200, content=sse(delta(content="ok"), "[DONE]"))
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        llm = OpenAIChat(http)
        llm.rate_limit_events.append(lambda: events.append("429"))
        llm.success_events.append(lambda: events.append("ok"))
        if streaming:
            await llm.chat(Params(), Session(session_id="stable"))
        else:
            await llm.post_json(endpoint, "chat/completions", {"model": "unchanged"}, "stable")
        assert len(requests) == 2 and events == ["429", "ok"]
        assert requests[0].content == requests[1].content
        assert requests[0].url == requests[1].url
        assert all(r.headers["x-session-id"] == "stable" for r in requests)
        assert "Bearer first" not in caplog.text and "Bearer second" not in caplog.text


async def test_all_keys_cooling_wait_and_exhaustion_are_bounded(monkeypatch):
    endpoint = pool()
    clock, sleeps, keys = [100.0], [], []
    monkeypatch.setattr("kuni.llm.time.monotonic", lambda: clock[0])

    async def advance(delay):
        assert delay > 0
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(asyncio, "sleep", advance)

    def handler(request):
        keys.append(request.headers["authorization"])
        return httpx.Response(429, text="limit", headers={"retry-after": "120"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        llm = OpenAIChat(http)
        with pytest.raises(LLMError, match="HTTP 429"):
            await llm.post_json(endpoint, "chat/completions", {})
        assert keys == ["Bearer first", "Bearer second"] * 3
        assert sleeps == [120, 120]
        await llm._request_headers(endpoint, None)  # the final failed attempt must also cool down
        assert sleeps == [120, 120, 120]


async def test_normal_calls_rotate_and_auth_errors_do_not_retry():
    endpoint = pool()
    keys = []

    def handler(request):
        keys.append(request.headers["authorization"])
        return httpx.Response(200, json={"ok": True}) if len(keys) <= 2 else httpx.Response(401, text="bad key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        llm = OpenAIChat(http)
        await llm.post_json(endpoint, "chat/completions", {})
        await llm.post_json(endpoint, "chat/completions", {})
        with pytest.raises(LLMError, match="HTTP 401"):
            await llm.post_json(endpoint, "chat/completions", {})
    assert keys == ["Bearer first", "Bearer second", "Bearer first"]


def test_explicit_retry_after_is_not_shortened(monkeypatch):
    assert backoff(429, 1, {"retry-after": "120"}) == 120
    monkeypatch.setattr("kuni.llm.time.time", lambda: 0)
    assert backoff(429, 1, {"retry-after": "Thu, 01 Jan 1970 00:02:00 GMT"}) == 120
    assert backoff(429, 1, {"retry-after": "invalid"}) == 5
    assert backoff(429, 1, {"retry-after": "inf"}) == 5
