from __future__ import annotations

import json

import aiohttp
import httpx
import pytest

from kuni.config import Endpoint
from kuni.llm import Function, Message, Session, ToolCall
from kuni.memory.graph import GraphMemory
from kuni.proxy.context_bridge import ContextBridge
from kuni.proxy.message_injector import MessageInjector, fingerprint
from kuni.proxy.server import PROXY_SYSTEM_PROMPT, ProxyServer
from kuni.proxy.streaming_filter import StreamingFilter
from kuni.tools import Tool, ToolSet


def chunk(**delta_and_finish) -> str:
    finish = delta_and_finish.pop("finish_reason", None)
    return "data: " + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                                  "choices": [{"index": 0, "delta": delta_and_finish, "finish_reason": finish}]})


def tool_chunk(name="", args="", idx=0, id_="call_1") -> str:
    return chunk(tool_calls=[{"index": idx, "id": id_ if name else None, "type": "function" if name else None,
                              "function": {"name": name, "arguments": args}}])


async def run_filter(lines, filtered):
    out, dispatched = [], []
    f = StreamingFilter(filtered)

    async def pass_through(line):
        out.append(line)

    async def handle(tc):
        dispatched.append((tc.function.name, tc.function.arguments))

    for line in lines:
        await f.process_line(line, pass_through, handle)
    return out, dispatched, f


# ---------------------------------------------------------------------------------------------------------------------
# streaming filter
# ---------------------------------------------------------------------------------------------------------------------
async def test_plain_content_passes_through_untouched():
    lines = [": keepalive", chunk(content="He"), chunk(content="llo"), chunk(finish_reason="stop"), "data: [DONE]", "garbage", "data: {not json"]
    out, dispatched, _ = await run_filter(lines, {"ask"})
    assert out == lines and dispatched == []


async def test_our_tool_call_is_hidden_and_dispatched_once_on_finish():
    lines = [chunk(content="thinking"), tool_chunk("ask", ""), tool_chunk(args='{"query":'), tool_chunk(args='"x"}'),
             chunk(finish_reason="tool_calls"), chunk(finish_reason="tool_calls")]  # a duplicate finish chunk
    out, dispatched, f = await run_filter(lines, {"ask"})
    assert dispatched == [("ask", '{"query":"x"}')]
    # the content before the tool call reached the client; the tool call chunks did not
    assert out[0] == chunk(content="thinking")
    assert len(out) == 2
    synthetic = json.loads(out[1][len("data: "):])
    assert synthetic["choices"][0]["delta"] == {"role": "assistant"} and "finish_reason" not in synthetic["choices"][0]
    assert f.choices[0].message.tool_calls[0].function.name == "ask"


async def test_clients_own_tool_calls_pass_and_are_reindexed_next_to_ours():
    lines = [tool_chunk("ask", '{"q":1}', idx=0, id_="mine"), tool_chunk("edit_file", '{"p":1}', idx=1, id_="theirs"),
             chunk(finish_reason="tool_calls")]
    out, dispatched, _ = await run_filter(lines, {"ask"})
    assert dispatched == [("ask", '{"q":1}')]
    final = json.loads(out[-1][len("data: "):])["choices"][0]
    assert final["finish_reason"] == "tool_calls"
    assert final["delta"]["tool_calls"] == [{"index": 0, "id": "theirs", "type": "function",
                                             "function": {"name": "edit_file", "arguments": '{"p":1}'}}]


async def test_only_clients_tool_calls_stream_through_as_they_are():
    lines = [tool_chunk("edit_file", "", id_="t"), tool_chunk(args='{"p":1}'), chunk(finish_reason="tool_calls")]
    out, dispatched, _ = await run_filter(lines, {"ask"})
    assert out == lines and dispatched == []


# ---------------------------------------------------------------------------------------------------------------------
# message injector
# ---------------------------------------------------------------------------------------------------------------------
def test_fingerprint_is_a_prefix_property():
    a = [{"role": "user", "content": "hi"}]
    b = a + [{"role": "assistant", "content": "yo"}]
    assert fingerprint(b).startswith(fingerprint(a)) and not fingerprint(a).startswith(fingerprint(b))
    assert "{" not in fingerprint(a) and "[" not in fingerprint(a)


def test_injector_reinserts_hidden_messages_after_the_message_they_followed():
    inj = MessageInjector()
    hi = {"role": "user", "content": "hi"}
    assert inj.merge([hi]) == [hi]
    hidden = [{"role": "assistant", "content": "", "tool_calls": []}, {"role": "tool", "content": "RESULT"}]
    inj.after(hi).extend(hidden)
    later = [{"role": "system", "content": "s"}, hi, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "again"}]
    merged = inj.merge(later)
    assert merged == [later[0], hi, *hidden, later[2], later[3]]


# ---------------------------------------------------------------------------------------------------------------------
# the server, end to end with a mocked upstream
# ---------------------------------------------------------------------------------------------------------------------
class Upstream:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "agnes-3.0-flash"}]})
        self.requests.append({"body": json.loads(request.content), "headers": dict(request.headers)})
        return httpx.Response(200, content=self.responses.pop(0).encode(), headers={"content-type": "text/event-stream"})


def sse(*lines: str) -> str:
    return "".join(line + "\n\n" for line in lines)


@pytest.fixture
async def proxy():
    created = []

    async def make(upstream: Upstream, tools: ToolSet | None = None):
        calls = []

        async def ask_handler(ctx):
            calls.append(ctx.args)
            return "MEMORY: Alex likes jazz"

        toolset = tools or ToolSet([Tool("ask", "ask memory", ask_handler)])
        server = ProxyServer(Endpoint("https://upstream.test/v1/", "up-key"), lambda session: toolset, port=0,
                             http=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
        await server.start()
        created.append(server)
        return server, calls

    yield make
    for s in created:
        await s.stop()


async def post(server: ProxyServer, body: dict) -> tuple[int, str]:
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{server.port}/v1/chat/completions", json=body) as r:
            return r.status, await r.text()


async def test_hidden_tool_loop_the_client_only_sees_the_final_answer(proxy):
    upstream = Upstream([
        sse(chunk(role="assistant"), tool_chunk("ask", ""), tool_chunk(args='{"query":"jazz"}'), chunk(finish_reason="tool_calls")),
        sse(chunk(role="assistant", content="Alex "), chunk(content="likes jazz"), chunk(finish_reason="stop"), "data: [DONE]"),
    ])
    server, ask_calls = await proxy(upstream)
    sent = []
    server.sent_request_to_llm.append(sent.append)
    body = {"model": "m", "stream": True, "messages": [{"role": "system", "content": "client system"}, {"role": "user", "content": "what does Alex like?"}],
            "tools": [{"type": "function", "function": {"name": "client_tool"}}], "top_k": 7}
    status, text = await post(server, body)

    assert status == 200
    assert "Alex " in text and "likes jazz" in text and '"ask"' not in text and "MEMORY" not in text  # hidden from the client
    assert ": kuni processing" in text and text.rstrip().endswith("data: [DONE]") and text.count("[DONE]") == 1
    assert ask_calls == [{"query": "jazz"}]

    first, second = upstream.requests
    assert first["body"]["messages"][0]["content"].startswith(PROXY_SYSTEM_PROMPT)  # the system prompt is hijacked
    assert first["body"]["top_k"] == 7 and first["body"]["session_id"] == "kuni_proxy_server"  # client's fields survive
    names = [t["function"]["name"] for t in first["body"]["tools"]]
    assert names == ["client_tool", "ask"]  # client's tools + ours
    assert first["headers"]["authorization"] == "Bearer up-key" and first["headers"]["x-session-id"] == "kuni_proxy_server"
    roles = [m["role"] for m in second["body"]["messages"]]
    assert roles[-2:] == ["assistant", "tool"]  # the follow-up carries the tool call + its result
    assert second["body"]["messages"][-1]["content"] == "MEMORY: Alex likes jazz"
    assert second["body"]["messages"][-2]["tool_calls"][0]["function"]["name"] == "ask"
    assert len(sent) == 1 and sent[0]["messages"][-1]["role"] == "tool"  # the bridge gets the session


async def test_hidden_messages_are_reinjected_into_the_next_request_of_the_dialog(proxy):
    upstream = Upstream([
        sse(tool_chunk("ask", '{"query":"x"}'), chunk(finish_reason="tool_calls")),
        sse(chunk(content="first answer"), chunk(finish_reason="stop")),
        sse(chunk(content="second answer"), chunk(finish_reason="stop")),
    ])
    server, _ = await proxy(upstream)
    system = {"role": "system", "content": "s"}
    q1 = {"role": "user", "content": "question one"}
    await post(server, {"stream": True, "messages": [system, q1]})
    # the client doesn't know about the hidden tool call: its next request has only the visible history
    await post(server, {"stream": True, "messages": [system, q1, {"role": "assistant", "content": "first answer"},
                                                      {"role": "user", "content": "question two"}]})
    msgs = upstream.requests[2]["body"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert msgs[3]["content"] == "MEMORY: Alex likes jazz" and msgs[-1]["content"] == "question two"


async def test_non_streaming_requests_are_rejected_politely(proxy):
    server, _ = await proxy(Upstream([]))
    status, text = await post(server, {"stream": False, "messages": [{"role": "user", "content": "x"}]})
    assert status == 400 and "stream=true" in text


async def test_upstream_errors_reach_the_client_as_an_sse_error(proxy):
    def failing(request):
        return httpx.Response(500, text="upstream exploded")

    server = ProxyServer(Endpoint("https://upstream.test/v1/", ""), lambda s: ToolSet(), port=0,
                         http=httpx.AsyncClient(transport=httpx.MockTransport(failing)))
    await server.start()
    try:
        status, text = await post(server, {"stream": True, "messages": [{"role": "user", "content": "x"}]})
    finally:
        await server.stop()
    assert status == 200 and "upstream exploded" in text and text.rstrip().endswith("data: [DONE]")


async def test_other_routes_are_reverse_proxied(proxy):
    server, _ = await proxy(Upstream([]))
    async with aiohttp.ClientSession() as s:
        async with s.get(f"http://127.0.0.1:{server.port}/v1/models") as r:
            assert r.status == 200 and (await r.json())["data"][0]["id"] == "agnes-3.0-flash"
        async with s.get(f"http://127.0.0.1:{server.port}/") as r:
            assert await r.text() == "Up and running"


# ---------------------------------------------------------------------------------------------------------------------
# context bridge
# ---------------------------------------------------------------------------------------------------------------------
def make_bridge(handler):
    from kuni.llm import OpenAIChat

    llm = OpenAIChat(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return ContextBridge(Endpoint("https://u.test/v1/", "k"), GraphMemory(), llm, timeout=3600)


def session_request(*texts: str) -> dict:
    return {"model": "m", "stream": True, "tools": [{"x": 1}], "session_id": "s1",
            "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)]}


def test_continuing_a_dialog_updates_the_session_and_a_new_dialog_starts_another():
    bridge = make_bridge(lambda r: httpx.Response(200))
    bridge.collect_request_to_llm(session_request("hello"))
    bridge.collect_request_to_llm(session_request("hello", "hi", "how are you"))  # same dialog, longer
    assert len(bridge.pending) == 1 and len(bridge.pending[0].request["messages"]) == 3
    bridge.collect_request_to_llm(session_request("something else"))
    assert len(bridge.pending) == 2


async def test_idle_sessions_are_dumped_to_the_graph_with_the_diary_prompt():
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        page = "Мы с Алексом полдня обсуждали проект про графовую память: он хочет заменить RAG на граф знаний."
        return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": page}}]})

    bridge = make_bridge(handler)
    bridge.collect_request_to_llm(session_request("давай обсудим графовую память", "давай"))
    assert await bridge.collect_and_save_sessions_not_newer_than(0) == []  # not idle yet
    assert len(bridge.pending) == 1
    items = await bridge.collect_and_save_sessions_not_newer_than(float("inf"))
    assert len(items) == 1 and "графовую память" in items[0] and not bridge.pending
    assert len(bridge.memory.episodes()) == 1
    req = seen[0]
    assert req["stream"] is False and "tools" not in req and req["session_id"] == "s1"
    assert "diary" in req["messages"][-1]["content"].lower()  # the diary prompt closes the session


async def test_bridge_gives_up_after_three_empty_answers():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "short"}}]})

    bridge = make_bridge(handler)
    bridge.collect_request_to_llm(session_request("a", "b"))
    assert await bridge.collect_and_save_sessions_not_newer_than(float("inf")) == [] and len(calls) == 3


async def test_work_items_are_prepended_for_the_owner_chat_only():
    from kuni.config import config
    from kuni.telegram.types import Chat, Message as TgMessage

    def handler(request):
        page = "Работали с Алексом над прокси-сервером: перехватываем запросы и подмешиваем инструмент ask."
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": page}}]})

    bridge = make_bridge(handler)
    bridge.collect_request_to_llm(session_request("work on the proxy", "ok"))
    stranger = Chat(5, "Stranger")
    assert await bridge.process_chat_history_message(stranger, TgMessage(1, 5, date=2**31), "FORMATTED") == "FORMATTED"
    assert len(bridge.pending) == 1  # nothing was touched
    owner = Chat(config().papikChatId, "Alex2772")
    out = await bridge.process_chat_history_message(owner, TgMessage(1, owner.id, date=2**31), "FORMATTED")
    assert out.startswith("<work_item>\n") and "прокси-сервером" in out and out.endswith("FORMATTED") and "Alex2772" in out
