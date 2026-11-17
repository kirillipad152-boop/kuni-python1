from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from kuni import config as cfgmod
from kuni.llm import (EMBEDDING_TAG, Function, LLMError, Message, OpenAIChat, Params, Session, ToolCall, Usage,
                      ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER)
from kuni.notifications import Notification, NotificationManager
from kuni.tools import Property, Tool, ToolError, ToolSet


# ---------------------------------------------------------------------------------------------------------------------
# session JSON
# ---------------------------------------------------------------------------------------------------------------------
def test_images_become_separate_content_parts():
    s = Session([Message(role=ROLE_USER, content=f"look\n<{EMBEDDING_TAG}>data:image/jpg;base64,AAA</{EMBEDDING_TAG}>\nwhat is it?")])
    j = s.to_json()
    assert [m["content"] if isinstance(m["content"], str) else "IMG" for m in j] == [
        "look\n", "<attachments>", "IMG", "</attachments>", "\nwhat is it?"]
    assert j[2]["content"] == [{"type": "image_url", "image_url": {"url": "data:image/jpg;base64,AAA"}}]
    assert all(m["role"] == "user" for m in j)


def test_two_images_in_one_message():
    c = f"<{EMBEDDING_TAG}>data:A</{EMBEDDING_TAG}><{EMBEDDING_TAG}>data:B</{EMBEDDING_TAG}>"
    j = Session([Message(role=ROLE_USER, content=c)]).to_json()
    urls = [m["content"][0]["image_url"]["url"] for m in j if isinstance(m["content"], list)]
    assert urls == ["data:A", "data:B"]


def test_message_json_roundtrip_keeps_reasoning_and_tool_calls():
    m = Message(role=ROLE_ASSISTANT, content="", reasoning_content="thinking",
                tool_calls=[ToolCall("c1", 0, "function", Function("wait", "{}"))])
    j = m.to_json()
    assert j["reasoning_content"] == "thinking"  # deepseek-like providers require it to be passed back
    assert j["tool_calls"][0] == {"id": "c1", "type": "function", "function": {"name": "wait", "arguments": "{}"}}
    back = Message.from_json(j)
    assert back.tool_calls[0].function.name == "wait" and back.reasoning_content == "thinking"
    assert "reasoning" not in Message(role=ROLE_USER, content="x").to_json()


def test_usage_parsing_variants():
    assert Usage.from_json(None) == Usage()
    u = Usage.from_json({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                         "prompt_tokens_details": {"cached_tokens": 4}})  # openrouter / agnes
    assert (u.prompt_cache_hit_tokens, u.prompt_cache_miss_tokens) == (4, 6)
    d = Usage.from_json({"prompt_tokens": 10, "prompt_cache_hit_tokens": 3, "prompt_cache_miss_tokens": 7})  # deepseek
    assert (d.prompt_cache_hit_tokens, d.prompt_cache_miss_tokens) == (3, 7)


# ---------------------------------------------------------------------------------------------------------------------
# OpenAIChat over a mocked transport
# ---------------------------------------------------------------------------------------------------------------------
def sse(*chunks: dict | str) -> bytes:
    out = ""
    for c in chunks:
        out += "data: " + (c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)) + "\n\n"
    return out.encode()


def delta(**kw) -> dict:
    finish = kw.pop("finish_reason", None)
    return {"id": "r1", "model": "m", "created": 1, "choices": [{"index": 0, "delta": kw, "finish_reason": finish}]}


def client_for(handler) -> OpenAIChat:
    return OpenAIChat(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_streaming_assembles_reasoning_content_tool_calls_and_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(
            delta(role="assistant", reasoning_content="hm"),
            delta(reasoning_content="mm"),
            delta(content="Приве"), delta(content="т!"),
            delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "send_telegram_message", "arguments": ""}}]),
            delta(tool_calls=[{"index": 0, "function": {"arguments": '{"text":'}}]),
            delta(tool_calls=[{"index": 0, "function": {"arguments": '"hi"}'}}]),
            delta(finish_reason="tool_calls"),
            {"id": "r1", "choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}},
            "[DONE]"), headers={"content-type": "text/event-stream"})

    llm = client_for(handler)
    changes = []
    s = llm.chat_streaming(Params(system_prompt="SYS", tools=[{"type": "function"}]), Session([Message(role=ROLE_USER, content="hi")], "sess-1"))
    s.on_changed(lambda r: changes.append(r.choices[0].message.content))
    await s.completed
    msg = s.response.choices[0].message
    assert msg.content == "Привет!" and msg.reasoning_content == "hmmm"
    assert len(msg.tool_calls) == 1 and msg.tool_calls[0].id == "call_1"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"text": "hi"}
    assert s.response.choices[0].finish_reason == "tool_calls"
    assert s.response.usage.total_tokens == 55
    assert changes[:3] == ["", "", "Приве"]  # observers see the partial response (typing indicator relies on it)

    body = seen["body"]
    assert seen["url"] == "https://apihub.agnes-ai.com/v1/chat/completions"
    assert body["model"] == "agnes-3.0-flash" and body["reasoning_effort"] == "high" and body["stream"] is True
    assert body["messages"][0] == {"role": "system", "content": "SYS"} and body["tools"] == [{"type": "function"}]
    assert body["temperature"] == 0.2 and "top_p" not in body
    assert seen["headers"]["authorization"] == "Bearer test-key" and seen["headers"]["x-session-id"] == "sess-1"


async def test_provider_that_repeats_the_tool_call_id_in_every_chunk_is_handled():
    def handler(_):
        tc = lambda args: delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "wait", "arguments": args}}])  # noqa: E731
        return httpx.Response(200, content=sse(tc("{"), tc("}"), delta(finish_reason="tool_calls"), "[DONE]"))

    r = await client_for(handler).chat(Params(), Session([Message(role=ROLE_USER, content="x")]))
    tc = r.choices[0].message.tool_calls[0]
    assert tc.id == "call_1" and tc.function.name == "wait" and tc.function.arguments == "{}"


async def test_retries_on_429_then_succeeds(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    calls = []

    def handler(_):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, content=sse(delta(content="ok"), "[DONE]"))

    r = await client_for(handler).chat(Params(), Session([Message(role=ROLE_USER, content="x")]))
    assert r.choices[0].message.content == "ok" and len(calls) == 2 and sleeps == [5]  # rate limits: 5,10,20,40,60s


async def test_retry_after_header_is_honored_and_retries_are_bounded(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(429, text="limit", headers={"retry-after": "7"})

    with pytest.raises(LLMError, match="HTTP 429"):
        await client_for(handler).chat(Params(), Session([Message(role=ROLE_USER, content="x")]))
    assert len(calls) == 6 and sleeps == [7.0] * 5  # first try + 5 retries


async def test_hard_http_error_raises_with_the_body():
    llm = client_for(lambda _: httpx.Response(400, text='{"error":"bad model"}'))
    with pytest.raises(LLMError, match="bad model"):
        await llm.chat(Params(), Session([Message(role=ROLE_USER, content="x")]))


async def test_error_object_inside_the_stream_raises():
    llm = client_for(lambda _: httpx.Response(200, content=sse({"error": {"message": "quota"}})))
    with pytest.raises(LLMError, match="quota"):
        await llm.chat(Params(), Session([Message(role=ROLE_USER, content="x")]))


async def test_reasoning_effort_is_omitted_when_empty_and_secondary_models_inherit():
    cfgmod.config().llm.reasoningEffort = ""
    bodies = []

    def handler(r):
        bodies.append(json.loads(r.content))
        return httpx.Response(200, content=sse(delta(content="ok"), "[DONE]"))

    llm = client_for(handler)
    await llm.chat(Params(), Session([Message(role=ROLE_USER, content="x")]))
    assert "reasoning_effort" not in bodies[0]
    await llm.chat(Params(config=cfgmod.config().llmImageToText), Session([Message(role=ROLE_USER, content="x")]))
    assert bodies[1]["model"] == "agnes-3.0-flash"  # empty vision model -> the main one


async def test_post_json_and_embeddings():
    def handler(r: httpx.Request):
        if r.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [1, 0, 0.5]}]})
        return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {"content": "hi"}}]})

    llm = client_for(handler)
    assert await llm.embedding(cfgmod.config().llm, "text") == [1.0, 0.0, 0.5]
    data = await llm.post_json(cfgmod.config().resolved_llm().endpoint, "chat/completions", {"messages": []})
    assert data["choices"][0]["message"]["content"] == "hi"


# ---------------------------------------------------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("streaming", [False, True])
async def test_llm_watchdog_hooks_report_each_429_and_only_success(monkeypatch, streaming):
    events, calls = [], []

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    def handler(_):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(429, text="limit")
        return httpx.Response(200, content=sse(delta(content="ok"), "[DONE]")) if streaming else httpx.Response(200, json={"ok": True})

    llm = client_for(handler)
    llm.rate_limit_events.append(lambda: events.append("429"))
    llm.success_events.append(lambda: events.append("ok"))
    if streaming:
        await llm.chat(Params(), Session())
    else:
        await llm.post_json(cfgmod.config().resolved_llm().endpoint, "chat/completions", {})
    assert events == ["429", "429", "ok"]
    await llm.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_exhausted_rate_limit_has_no_success_hook(monkeypatch, streaming):
    events = []

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    llm = client_for(lambda _: httpx.Response(429, text="limit"))
    llm.rate_limit_events.append(lambda: events.append("429"))
    llm.success_events.append(lambda: events.append("ok"))
    with pytest.raises(LLMError, match="HTTP 429"):
        if streaming:
            await llm.chat(Params(), Session())
        else:
            await llm.post_json(cfgmod.config().resolved_llm().endpoint, "chat/completions", {})
    assert events == ["429"] * 6
    await llm.aclose()


@pytest.mark.parametrize("status,body", [(403, '{}'), (200, 'not JSON'), (200, '[]'), (200, '{"error":"quota"}')])
async def test_json_error_is_not_watchdog_progress(status, body):
    events = []
    llm = client_for(lambda _: httpx.Response(status, text=body))
    llm.success_events.append(lambda: events.append("ok"))
    with pytest.raises(LLMError):
        await llm.post_json(cfgmod.config().resolved_llm().endpoint, "chat/completions", {})
    assert not events
    await llm.aclose()


async def test_empty_stream_is_not_watchdog_progress():
    events = []
    llm = client_for(lambda _: httpx.Response(200, content=sse("[DONE]")))
    llm.success_events.append(lambda: events.append("ok"))
    with pytest.raises(LLMError, match="empty stream"):
        await llm.chat(Params(), Session())
    assert not events
    await llm.aclose()


async def test_broken_observer_does_not_break_llm():
    llm = client_for(lambda _: httpx.Response(200, content=sse(delta(content="ok"), "[DONE]")))

    def broken():
        raise RuntimeError("observer broke")

    llm.success_events.append(broken)
    assert (await llm.chat(Params(), Session())).choices[0].message.content == "ok"
    await llm.aclose()


async def _noop(ctx):
    return "ok"


def test_tool_schema_matches_strict_mode():
    t = Tool("t", "d", _noop, {
        "a": Property("string", "A"),
        "b": Property("integer", "B", nullable=True),
        "c": Property("integer", "C", or_array=True, items=Property("integer")),
        "d": Property("array", "D", items=Property("string")),
    }, ["a", "b", "c", "d"])
    j = t.to_json()
    f = j["function"]
    assert j["type"] == "function" and f["strict"] is True
    assert f["parameters"]["additionalProperties"] is False and f["parameters"]["required"] == ["a", "b", "c", "d"]
    p = f["parameters"]["properties"]
    assert p["a"]["type"] == "string" and p["b"]["type"] == ["integer", "null"]
    assert p["c"]["type"] == ["integer", "array"] and p["c"]["items"]["type"] == "integer"
    assert p["d"]["items"]["type"] == "string"


def test_array_property_without_items_is_a_programming_error():
    with pytest.raises(AssertionError):
        Property("array").to_json()


async def test_tool_calls_results_errors_and_hooks():
    async def ok(ctx):
        return "fine\x00\x1b"

    async def boom(ctx):
        raise ToolError("nope")

    async def crash(ctx):
        raise KeyError("x")

    tools = ToolSet([Tool("ok", "", ok), Tool("boom", "", boom), Tool("crash", "", crash)])
    after = []
    tools.on_after_tool_call.append(after.append)
    calls = [ToolCall("1", 0, "function", Function("ok", "{}")), ToolCall("2", 1, "function", Function("boom", "{}")),
             ToolCall("3", 2, "function", Function("crash", "{}")), ToolCall("4", 3, "function", Function("missing", "{}")),
             ToolCall("5", 4, "function", Function("ok", "not json"))]
    res = await tools.handle_tool_calls(calls)
    assert [m.role for m in res] == [ROLE_TOOL] * 5 and [m.tool_call_id for m in res] == ["1", "2", "3", "4", "5"]
    assert res[0].content == "fine"  # control characters are stripped
    assert res[1].content == 'error while executing "boom" tool: nope'
    assert res[2].content.startswith('error while executing "crash" tool:')  # a bug in a tool must not kill the worker
    assert res[3].content == 'tool "missing" is not currently available. Please use another tool instead.'
    assert res[4].content.startswith('error while executing "ok" tool:')
    assert after == ["ok"]  # hooks fire on success only


async def test_replace_with_swaps_handlers_and_hooks_like_assigning_a_new_toolset():
    a = ToolSet([Tool("a", "", _noop)])
    a.on_after_tool_call.append(lambda n: None)
    a.replace_with(ToolSet([Tool("b", "", _noop)]))
    assert list(a.handlers()) == ["b"] and a.on_after_tool_call == []


# ---------------------------------------------------------------------------------------------------------------------
# notifications
# ---------------------------------------------------------------------------------------------------------------------
async def drain(nm: NotificationManager, count: int, pins=None) -> list[str]:
    got: list[str] = []

    async def worker(n: Notification) -> bool:
        got.append(n.message.split("\n")[0])
        return len(got) < count

    await asyncio.wait_for(nm.run(pins if pins is not None else set(), worker), 2)
    return got


async def test_priority_orders_notifications_fifo_within_equal_priority():
    nm = NotificationManager()
    for msg, prio in (("a", 0), ("b", 5), ("c", 5), ("papik", 1000), ("d", 0)):
        nm.pass_notification_to_ai(Notification(msg, priority=prio))
    assert await drain(nm, 5) == ["papik", "b", "c", "a", "d"]


async def test_current_time_is_appended_and_flags_fire():
    nm = NotificationManager()
    h = nm.pass_notification_to_ai(Notification("x"))
    assert not h.on_started_processing.is_set()
    seen = []

    async def worker(n):
        seen.append(n.message)
        assert h.on_started_processing.is_set() and not h.on_processed.is_set()
        return False

    await asyncio.wait_for(nm.run(set(), worker), 2)
    assert "Current time:" in seen[0] and seen[0].endswith("UTC") and h.on_processed.is_set()


async def test_pinned_notifications_stick_to_the_worker_that_has_the_pin():
    nm = NotificationManager()
    got1, got2 = [], []

    async def w1(n):
        got1.append(n.message.split("\n")[0])
        return len(got1) < 2

    async def w2(n):
        got2.append(n.message.split("\n")[0])
        return True

    pins1: set[str] = set()
    t1 = asyncio.create_task(nm.run(pins1, w1))
    t2 = asyncio.create_task(nm.run(set(), w2))
    await asyncio.sleep(0)
    nm.pass_notification_to_ai(Notification("first", pin='<chat id="1" />'))
    await asyncio.sleep(0.05)
    assert got1 == ["first"] and pins1 == {'<chat id="1" />'}
    nm.pass_notification_to_ai(Notification("second", pin='<chat id="1" />'))
    nm.pass_notification_to_ai(Notification("other", pin='<chat id="2" />'))
    await asyncio.wait_for(t1, 2)
    await asyncio.sleep(0.05)
    assert got1 == ["first", "second"]  # same chat -> same worker (it has the related context)
    assert got2 == ["other"]
    t2.cancel()


async def test_remove_notifications_by_substring():
    nm = NotificationManager()
    nm.pass_notification_to_ai(Notification('<notification chat_id="1">\nhi'))
    nm.pass_notification_to_ai(Notification('<notification chat_id="12">\nhi'))
    nm.remove_notifications('<notification chat_id="1">\n')
    assert await drain(nm, 1) == ['<notification chat_id="12">']
