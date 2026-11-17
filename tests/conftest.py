"""Shared fixtures: no network, no Telegram, a scripted fake LLM."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kuni import config as cfgmod  # noqa: E402
from kuni import prompts as promptsmod  # noqa: E402
from kuni.llm import (Choice, Function, Message, OpenAIChat, Params, Response, Session, ToolCall, Usage,  # noqa: E402
                      ROLE_ASSISTANT)

logging.getLogger().setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test runs in its own cwd with default config and fresh prompts (they write to ./prompts, ./logs...)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KUNI_API_KEY", raising=False)
    monkeypatch.delenv("KUNI_BASE_URL", raising=False)
    cfg = cfgmod.parse({})
    cfg.llm.endpoint.bearerKey = "test-key"
    cfg.logLlmRequests = False
    cfg.randomlyGoSleep = False
    cfg.suggestIgnoreChance = 0.0
    cfg.toolReminderProbability = 0.0
    cfg.remindUseAsk = False
    cfg.typingSimulationMinWpm = cfg.typingSimulationMaxWpm = 10_000_000  # no real typing delays in tests
    cfg.lockdown = cfgmod.LockdownMode.NONE
    cfgmod.set_config(cfg)
    cfgmod.TESTING = True
    promptsmod.set_prompts(promptsmod.Prompts(tmp_path / "prompts"))
    from kuni.tools_impl import send_telegram_message as stm

    stm.rng = random.Random(1)
    stm._embeddings.clear()
    stm._give_a_head_start = 0.0
    stm._prev_message_send_time = 0.0
    yield
    cfgmod.TESTING = False
    promptsmod.set_prompts(None)


def assistant(content: str = "", tool_calls: list[tuple[str, dict]] | None = None, usage_total: int = 100) -> Response:
    """Builds the response the fake LLM returns."""
    msg = Message(role=ROLE_ASSISTANT, content=content)
    for i, (name, args) in enumerate(tool_calls or []):
        msg.tool_calls.append(ToolCall(id=f"call_{name}_{i}", index=i, type="function",
                                       function=Function(name, json.dumps(args, ensure_ascii=False))))
    return Response(id="x", model="fake", choices=[Choice(0, msg, "tool_calls" if tool_calls else "stop")],
                    usage=Usage(prompt_tokens=usage_total, completion_tokens=1, total_tokens=usage_total))


class FakeLLM(OpenAIChat):
    """Scripted LLM: `script` is a list of Responses or callables(params, messages) -> Response, consumed in order;
    when it runs out, `default` is used."""

    def __init__(self, script=None, default=None) -> None:
        super().__init__()
        self.script = list(script or [])
        self.default = default or (lambda p, m: assistant("{}"))
        self.calls: list[tuple[Params, list[Message]]] = []

    async def chat(self, params: Params, messages: Session) -> Response:
        self.calls.append((params, [Message(**m.__dict__) for m in messages]))
        item = self.script.pop(0) if self.script else self.default
        return item(params, messages) if callable(item) else item

    def chat_streaming(self, params: Params, messages: Session):
        import asyncio

        from kuni.llm import StreamingResponse

        s = StreamingResponse()

        async def run() -> None:
            s.response = await self.chat(params, messages)
            s._changed()

        s.completed = asyncio.ensure_future(run())
        return s

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_llm():
    return FakeLLM


async def wait_until(cond, timeout: float = 5.0, what: str = "condition"):
    """Polls an asyncio-world condition (workers run as tasks)."""
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for: {what}")
