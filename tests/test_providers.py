"""The LLM client must work with any OpenAI-compatible server, not only Agnes: real HTTP over a socket, no mocks of httpx."""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from kuni import config as cfgmod
from kuni.config import Endpoint, EndpointAndModel, config
from kuni.llm import LLMError, Message, OpenAIChat, Params, Session, ROLE_USER


def sse(*chunks) -> bytes:
    out = ""
    for chunk in chunks:
        out += (chunk if isinstance(chunk, str) and chunk.startswith(":") else "data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk))) + "\n\n"
    return out.encode()


def delta(**fields):
    finish = fields.pop("finish_reason", None)
    return {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": fields, "finish_reason": finish}]}


STYLES = {
    # OpenAI: usage arrives in a last chunk with empty choices, tool call announced with id/type/name first
    "openai": [delta(role="assistant", content=""), delta(content="Hel"), delta(content="lo"),
               delta(tool_calls=[{"index": 0, "id": "call_a", "type": "function", "function": {"name": "wait", "arguments": ""}}]),
               delta(tool_calls=[{"index": 0, "function": {"arguments": "{}"}}]), delta(finish_reason="tool_calls"),
               {"id": "x", "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}, "[DONE]"],
    # OpenRouter: SSE comments as keep-alives, a `reasoning` field, usage inside the final chunk
    "openrouter": [": OPENROUTER PROCESSING", delta(role="assistant", reasoning="thinking", content=None), delta(content="Hel"),
                   delta(content="lo", finish_reason="stop"),
                   {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
                   "[DONE]"],
    # a local server (Ollama / LM Studio / llama.cpp): no key, no usage at all, `reasoning_content` field
    "local": [delta(role="assistant", reasoning_content="hmm", content=""), delta(content="Hel"), delta(content="lo"),
              delta(finish_reason="stop"), "[DONE]"],
}


async def serve(style: str, seen: list):
    async def handler(request: web.Request) -> web.StreamResponse:
        seen.append({"path": request.path, "auth": request.headers.get("Authorization"), "body": await request.json()})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(sse(*STYLES[style]))
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = TestServer(app)
    await server.start_server()
    return server


@pytest.mark.parametrize("style,key,effort,model", [
    ("openai", "sk-openai-test-key", "", "gpt-4.1-mini"),
    ("openrouter", "sk-or-v1-test-key", "high", "anthropic/claude-sonnet-4"),
    ("local", "", "", "qwen3:8b"),
])
async def test_the_same_client_talks_to_any_openai_compatible_server(style, key, effort, model):
    seen: list = []
    server = await serve(style, seen)
    llm = OpenAIChat()
    try:
        base = str(server.make_url("/v1"))  # no trailing slash on purpose: it is added for you
        config().llm = EndpointAndModel(Endpoint(base, key), model, effort)
        answer = await llm.chat(Params(system_prompt="SYS", tools=[{"type": "function", "function": {"name": "wait"}}]),
                                Session([Message(role=ROLE_USER, content="hi")]))
    finally:
        await llm.aclose()
        await server.close()

    request = seen[0]
    assert request["path"] == "/v1/chat/completions" and request["body"]["model"] == model and request["body"]["stream"] is True
    assert request["auth"] == (f"Bearer {key}" if key else None)  # a keyless local server gets no Authorization header
    assert ("reasoning_effort" in request["body"]) == bool(effort)  # only sent when configured
    assert "agnes" not in json.dumps(request["body"]).lower()
    message = answer.choices[0].message
    assert message.content == "Hello"
    assert answer.usage.total_tokens in (0, 9, 15)  # providers without usage are fine
    if style == "openai":
        assert message.tool_calls[0].function.name == "wait" and message.tool_calls[0].id == "call_a" and answer.usage.prompt_tokens == 11
    if style == "openrouter":
        assert message.reasoning == "thinking"
    if style == "local":
        assert message.reasoning_content == "hmm"


async def test_a_provider_error_is_reported_not_swallowed():
    async def handler(request):
        return web.json_response({"error": {"message": "model not found", "type": "invalid_request_error"}}, status=404)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = TestServer(app)
    await server.start_server()
    llm = OpenAIChat()
    try:
        config().llm = EndpointAndModel(Endpoint(str(server.make_url("/v1")), "k"), "missing", "")
        with pytest.raises(LLMError, match="HTTP 404.*model not found"):
            await llm.chat(Params(), Session([Message(role=ROLE_USER, content="hi")]))
    finally:
        await llm.aclose()
        await server.close()


def test_every_part_can_use_its_own_provider_and_nothing_is_hardwired_to_one_vendor():
    cfg = cfgmod.parse({"general": {"llm": {"endpoint": {"baseUrl": "https://openrouter.ai/api/v1", "bearerKey": "A"}, "model": "m1"}},
                        "capabilities": {"vision": {"llm_image_to_text": {"endpoint": {"baseUrl": "http://localhost:11434/v1", "bearerKey": ""},
                                                                          "model": "llava"}}}})
    main = cfg.resolved_llm()
    assert main.endpoint.baseUrl == "https://openrouter.ai/api/v1/" and main.endpoint.bearerKey == "A" and main.model == "m1"
    vision = cfg.resolve(cfg.llmImageToText)  # a different host: the main key must not leak to it
    assert vision.endpoint.baseUrl == "http://localhost:11434/v1/" and vision.endpoint.bearerKey == "" and vision.model == "llava"
    assert "agnes" not in repr(main).lower() and "agnes" not in repr(vision).lower()
    # the environment switches the whole bot to another provider without editing the file
    import os

    os.environ.update(KUNI_BASE_URL="https://api.openai.com/v1", KUNI_API_KEY="sk-env", KUNI_MODEL="gpt-4.1", KUNI_REASONING_EFFORT="")
    try:
        env = cfgmod.parse({}).resolved_llm()
    finally:
        for name in ("KUNI_BASE_URL", "KUNI_API_KEY", "KUNI_MODEL", "KUNI_REASONING_EFFORT"):
            os.environ.pop(name, None)
    assert (env.endpoint.baseUrl, env.endpoint.bearerKey, env.model) == ("https://api.openai.com/v1/", "sk-env", "gpt-4.1")
