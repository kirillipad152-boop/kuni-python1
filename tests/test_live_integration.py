"""Live tests against a real OpenAI-compatible endpoint. Skipped unless KUNI_API_KEY is set:

    KUNI_API_KEY=sk-... KUNI_BASE_URL=https://apihub.agnes-ai.com/v1 pytest -m integration -s

They cost a few cents and depend on the provider's rate limits.
"""

from __future__ import annotations

import json
import os

import pytest

from conftest import FakeLLM, wait_until
from kuni import config as cfgmod
from kuni.config import config
from kuni.llm import Message, OpenAIChat, Params, ROLE_USER, Session
from kuni.memory.extract import extract_graph, save_pages
from kuni.memory.graph import GraphMemory
from kuni.telegram.fake import FakeTelegram
from kuni.telegram_app import TelegramApp
from kuni.tools import Property, Tool, ToolSet
from kuni.tools_impl.ask import run_ask

# captured at import: the autouse `isolated` fixture scrubs these from os.environ for every test
LIVE = {k: os.environ.get(k) for k in ("KUNI_API_KEY", "KUNI_BASE_URL", "KUNI_MODEL", "KUNI_REASONING_EFFORT")}

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not os.environ.get("KUNI_API_KEY"), reason="KUNI_API_KEY is not set")]


@pytest.fixture(autouse=True)
def live_config(isolated, monkeypatch):
    for k, v in LIVE.items():
        if v:
            monkeypatch.setenv(k, v)
    cfg = cfgmod.parse({})  # env overrides: key / base url / model / reasoning effort
    cfg.logLlmRequests = False
    cfg.randomlyGoSleep = False
    cfg.suggestIgnoreChance = 0.0
    cfg.requestTimeoutSecs = 180
    cfgmod.set_config(cfg)
    cfgmod.TESTING = True


@pytest.fixture
async def llm():
    c = OpenAIChat()
    yield c
    await c.aclose()


async def test_chat_streaming_and_reasoning(llm):
    r = await llm.chat(Params(system_prompt="Answer in one word."), Session([Message(role=ROLE_USER, content="Say hello in Russian")]))
    assert "привет" in r.choices[0].message.content.lower() and r.usage.total_tokens > 0


async def test_tool_calling_roundtrip(llm):
    async def handler(ctx):
        return "Message sent successfully."

    tools = ToolSet([Tool("send_telegram_message", "Sends a message to the user", handler, {"text": Property("string", "text")}, ["text"])])
    s = Session([Message(role=ROLE_USER, content="Greet the user by calling the tool.")])
    r = await llm.chat(Params(system_prompt="You act only by calling tools.", tools=tools.as_json()), s)
    m = r.choices[0].message
    assert m.tool_calls and m.tool_calls[0].function.name == "send_telegram_message"
    results = await tools.handle_tool_calls(m.tool_calls)
    s << m << results  # the assistant message with its reasoning_content must be accepted back by the API
    r2 = await llm.chat(Params(system_prompt="You act only by calling tools.", tools=tools.as_json()), s)
    assert r2.choices


async def test_vision(llm):
    from PIL import Image, ImageDraw

    from kuni.llm import embed_image

    img = Image.new("RGB", (300, 200), (220, 30, 30))
    ImageDraw.Draw(img).ellipse((90, 40, 210, 160), fill=(30, 30, 220))
    r = await llm.chat(Params(system_prompt="Describe the image in one short sentence, mention colors."),
                       Session([Message(role=ROLE_USER, content="Photo:\n\n" + embed_image(img) + "\n\nDescribe the last photo.")]))
    text = r.choices[0].message.content.lower()
    assert "red" in text and "blue" in text


async def test_graph_extraction_ask_and_recall_end_to_end(llm):
    g = GraphMemory()
    page = ("Сегодня Алекс написал мне, что его повысили на работе в Яндексе. Он очень рад. "
            "Ещё он сказал, что играет на пианино и любит джаз. Я поздравила его и спросила про джаз.")
    saved = await save_pages(g, llm, page)
    assert saved and g.entities()
    names = " ".join(" ".join([n.name, *n.aliases]).lower() for n in g.entities())  # names AND aliases, in either script
    assert ("алекс" in names or "alex" in names) and ("яндекс" in names or "yandex" in names)
    assert "алекс" in names and "alex" in names  # the extraction prompt asks for both spellings (name + transliterated alias)
    answer = await run_ask(llm, g, "Что я знаю про Алекса: где он работает и что он любит?")
    low = answer.lower()
    assert ("яндекс" in low or "yandex" in low) and ("джаз" in low or "jazz" in low)


async def test_a_whole_conversation_with_the_real_model():
    cfg = config()
    cfg.papikChatId = 42
    tg = FakeTelegram(my_id=1000)
    tg.add_user(42, "Alex", "Smith", "alex")
    app = TelegramApp(tg, OpenAIChat())
    try:
        await tg.incoming(42, "привет! как дела?")
        await wait_until(lambda: ("set_online", False) in tg.actions, timeout=240, what="Kuni to finish the turn")
        sent = [a[2] for a in tg.actions if a[0] == "send_message"]
        assert sent, "she should have answered"
        print("\nKuni said:", sent)
    finally:
        await app.close()
