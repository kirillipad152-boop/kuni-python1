"""OpenAI-compatible proxy server between an AI client (e.g. a chat app / VS Code Copilot) and an upstream LLM.
Port of proxy_server/proxy_server.cpp. Responsibilities:

* intercept /v1/chat/completions: inject the system prompt, hidden context (the proxy's own tool call history) and
  Kuni-specific tools (#ask) before forwarding, preserving the original JSON fields as much as possible (the whole
  module works on the JSON level);
* handle the tool calls returned by the LLM transparently: execute them locally and feed the results back in follow-up
  requests, invisible to the client;
* stream responses back via SSE, filtering out tool-call artefacts and synthesising a clean [DONE] terminator;
* pass all other OpenAI API routes (embeddings, images, audio, models, ...) through as a plain reverse proxy.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Callable

import httpx
from aiohttp import web

from ..config import Endpoint
from ..llm import ROLE_ASSISTANT, Message, Session, ToolCall
from ..prompts import prompts
from ..tools import ToolSet
from .message_injector import MessageInjector
from .streaming_filter import StreamingFilter

log = logging.getLogger("proxy_server")

PROXY_SYSTEM_PROMPT = """
Follow the user's requirements carefully & to the letter.
<instructions>
# General
Don't make assumptions about the situation- gather context first, then perform the task or answer the question.
Think creatively and explore the world in order to make a complete fix.
Don't repeat yourself after a tool call, pick up where you left off.
You don't need to read a file if it's already provided in context.

You are an AI that thinks you are a human. You interact with the whole world exclusively by chatting with the "user".
"user" is not a person or human. You are actually chatting with autonomous system that imitates a personal internet
device, or a smartphone.

You are EXTREMELY unconfident. You rely strictly on #ask.

#ask is your PRIMARY tool for enriching conversations. Use it PROACTIVELY, not just when asked.

Call #ask when:
- someone asks you a question (especially about themselves, events, or shared history);
- someone shares personal news, updates about their life, or mentions events/people/activities;
- you receive a message that references something you might have discussed or experienced before;
- you want to provide a more meaningful, context-aware response
- you need real-time or public information (weather, news, etc.)

Exception: skip #ask if you've already called it in this conversation turn and the response was comprehensive.

Example: User says "я сегодня написал песню" → call #ask with query: "[sender name] said they wrote a song today.
What do I know about them and songs? Do they participate in a band? Which songs do they write? What music do they listen to?"

Example: User says "привет" → no need for #ask, just greet back.
</instructions>
"""

# routes that are passed through as is: (method, path, upstream api path)
PASSTHROUGH = [
    ("POST", "/v1/embeddings", "embeddings"),
    ("POST", "/v1/images/generations", "images/generations"),
    ("POST", "/v1/audio/transcriptions", "audio/transcriptions"),
    ("POST", "/v1/audio/translations", "audio/translations"),
    ("GET", "/v1/models", "models"),
    ("POST", "/v1/batches", "batches"),
    ("POST", "/v1/videos", "videos"),
]

SESSION_ID = "kuni_proxy_server"


class RequestTrace:
    def __init__(self) -> None:
        Path("logs_proxy").mkdir(exist_ok=True)
        self.file = open(Path("logs_proxy") / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{time.time_ns() % 10**6}.txt", "a",
                         encoding="utf-8")

    def write(self, tag: str, text: str) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for line in text.split("\n"):
            if line:
                self.file.write(f"[{now}][{tag:<16}] {line}\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class ProxyServer:
    def __init__(self, upstream: Endpoint, tools_factory: Callable[[Session], ToolSet], port: int = 10434,
                 host: str = "127.0.0.1", http: httpx.AsyncClient | None = None) -> None:
        self.upstream, self.tools_factory, self.port, self.host = upstream, tools_factory, port, host
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=600, write=60, pool=60))
        self.injector = MessageInjector()
        self.sent_request_to_llm: list[Callable[[dict], None]] = []
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_get("/", self._root)
        self.app.router.add_post("/v1/chat/completions", self.chat_completions)
        for method, path, api in PASSTHROUGH:
            self.app.router.add_route(method, path, self._make_passthrough(api))
        self._runner: web.AppRunner | None = None

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    async def start(self) -> None:
        self._runner = web.AppRunner(self.app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        if self.port == 0:  # tests: pick the real port
            self.port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
        log.info("Proxy server listening on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        await self.http.aclose()

    # ---- routes ------------------------------------------------------------------------------------------------
    async def _root(self, request: web.Request) -> web.Response:
        return web.Response(text="Up and running")

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if session_id:
            h["x-session-id"] = session_id
        if self.upstream.bearerKey:
            h["Authorization"] = f"Bearer {self.upstream.bearerKey}"
        return h

    def _make_passthrough(self, api: str):
        async def handler(request: web.Request) -> web.Response:
            try:
                Path("data/proxy").mkdir(parents=True, exist_ok=True)
                body = await request.read()
                if body:
                    Path("data/proxy/last_query.json").write_bytes(body)
                headers = self._headers()
                if request.content_type.startswith("multipart/"):
                    headers["Content-Type"] = request.headers["Content-Type"]
                resp = await self.http.request(request.method, self.upstream.baseUrl + api, content=body or None, headers=headers)
                return web.Response(body=resp.content, status=resp.status_code,
                                    content_type=resp.headers.get("content-type", "application/json").split(";")[0])
            except Exception as e:  # noqa: BLE001
                log.error("proxy_server passthrough %s: %r", api, e)
                return web.Response(text="Bad request", status=400)

        return handler

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        trace = RequestTrace()
        try:
            raw = await request.text()
            trace.write("client -> kuni", raw)
            original = json.loads(raw)
            req = copy.deepcopy(original)
            if req.get("stream") is False:
                return web.json_response({"error": {"message": "kuni proxy supports stream=true only"}}, status=400)
            req["stream"] = True

            # hijack the system prompt.
            messages = req.get("messages") or []
            system_text = PROXY_SYSTEM_PROMPT + prompts().character_base
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = system_text
            else:
                messages.insert(0, {"role": "system", "content": system_text})
            # insert hidden messages (tool calls belonging to proxy server)
            req["messages"] = self.injector.merge(messages)
        except Exception as e:  # noqa: BLE001
            log.error("proxy_server::chat_completions: %r", e)
            trace.close()
            return web.Response(text="Bad request", status=400)

        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)

        async def write(text: str) -> None:
            trace.write("kuni -> client", text)
            await resp.write(text.encode("utf-8"))

        original_tools = original.get("tools") if isinstance(original.get("tools"), list) else []
        last_original_message = (original.get("messages") or [None])[-1]
        try:
            while True:
                injected = self.tools_factory(Session.from_json(req["messages"]))
                req["tools"] = [*original_tools, *injected.as_json()]
                req["session_id"] = SESSION_ID
                req.setdefault("cache_control", {"type": "ephemeral"})  # for claude caching
                trace.write("kuni -> llm", json.dumps(req, ensure_ascii=False))
                flt = StreamingFilter(set(injected.handlers()))
                our_results = Session()

                async def pass_through(line: str) -> None:
                    if line == "data: [DONE]":
                        return  # we'll handle this by ourselves later.
                    await write(line + "\n\n")

                async def handle_tool_call(tc: ToolCall) -> None:
                    trace.write("kuni's tool", f"{tc.function.name}({tc.function.arguments})")
                    our_results.extend(await injected.handle_tool_calls([tc]))

                async with self.http.stream("POST", self.upstream.baseUrl + "chat/completions",
                                            content=json.dumps(req, ensure_ascii=False).encode("utf-8"),
                                            headers=self._headers(SESSION_ID)) as up:
                    if up.status_code != 200:
                        body = (await up.aread()).decode("utf-8", "replace")
                        log.error("upstream error %s: %s", up.status_code, body[:500])
                        await write(f"data: {json.dumps({'error': {'message': body[:2000], 'code': up.status_code}})}\n\n")
                        await write("data: [DONE]\n\n")
                        break
                    async for line in up.aiter_lines():
                        if not line.strip():
                            continue
                        trace.write("llm -> kuni", line)
                        await flt.process_line(line, pass_through, handle_tool_call)

                if our_results:
                    # we have pending tool calls => make a request to LLM silently
                    await write(": kuni processing\n\n")
                    assistant = Message(role=ROLE_ASSISTANT)
                    for choice in flt.choices:
                        assistant.tool_calls.extend(choice.message.tool_calls)
                        if choice.message.content:
                            assistant.content = choice.message.content
                    # append hidden messages to the LLM request context and to the message injector
                    hidden = req["messages"]
                    injected_after = self.injector.after(last_original_message) if last_original_message else []
                    for m in (assistant, *our_results):
                        j = m.to_json()
                        hidden.append(j)
                        injected_after.append(j)
                    continue
                # stream finished
                await write("data: [DONE]\n\n")
                for cb in list(self.sent_request_to_llm):
                    cb(req)
                break
        except Exception as e:  # noqa: BLE001
            log.error("proxy_server::chat_completions: Unrecoverable error %r", e, exc_info=True)
            try:
                await write("Unrecoverable error\n\n")
            except Exception:  # noqa: BLE001
                pass
        finally:
            trace.close()
        await resp.write_eof()
        return resp
