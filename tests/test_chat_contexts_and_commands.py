import asyncio
import json
from pathlib import Path

import pytest

from conftest import FakeLLM, assistant, wait_until
from kuni import config as cfgmod
from kuni.config import LockdownMode, config
from kuni.llm import Message, Session
from kuni.telegram.fake import FakeTelegram
from kuni.telegram.types import Message as TgMessage
from kuni.telegram_app import TelegramApp
from kuni.tools import ToolSet


@pytest.fixture
async def controlled():
    cfg = config()
    cfg.papikChatId = 42
    cfgmod.CONFIG_TOML.write_text(cfgmod.render(cfg))
    tg = FakeTelegram()
    tg.add_user(42, "Owner")
    tg.add_user(43, "Friend")
    llm = FakeLLM(default=lambda p, m: assistant("", [("wait", {})]))
    app = TelegramApp(tg, llm)
    yield app, tg, llm
    await app.close()


async def test_owner_commands_are_direct_even_when_muted_and_filtered(controlled):
    app, tg, llm = controlled
    cfg = config()
    cfg.chatNotificationFilter = LockdownMode.PAPIK_ONLY
    tg.chats[42].muted = True
    for text in ("/status", "/graph stats", "/pause on", "/status", "/pause off"):
        await tg.incoming(42, text)
    assert not llm.calls
    replies = [a[2] for a in tg.actions if a[0] == "send_message"]
    assert "работает" in replies[0] and "Узлов:" in replies[1]
    assert "Пауза включена" in replies[2] and "Куни: пауза" in replies[3]
    assert "Пауза выключена" in replies[4] and not config().paused
    assert cfgmod.parse(__import__("tomllib").loads(Path("config.toml").read_text())).paused is False
    assert Path("config.toml").stat().st_mode & 0o777 == 0o600


async def test_only_actual_owner_in_dm_can_change_settings(controlled):
    app, tg, llm = controlled
    tg.add_group(-100, "Group")
    for msg in (TgMessage(99, 43, 1, sender_id=43, text="/pause on"),
                TgMessage(100, -100, 1, sender_id=42, text="/pause on"),
                TgMessage(101, 42, 1, sender_id=43, text="/lockdown none"),
                TgMessage(102, 42, 1, sender_id=42, sender_is_chat=True, text="/pause on")):
        assert "только владельцу" in await app.try_handle_cmd(msg)
    assert not config().paused and not llm.calls


async def test_lockdown_persists_and_bad_arguments_do_not_change_it(controlled):
    app, tg, llm = controlled
    await tg.incoming(42, "/lockdown contacts_only")
    assert config().lockdown == LockdownMode.CONTACTS_ONLY
    assert 'contacts_only' in Path("config.toml").read_text()
    before = Path("config.toml").read_bytes()
    for text in ("/lockdown nope", "/lockdown none extra", "/pause nope", "/graph export"):
        await tg.incoming(42, text)
    assert Path("config.toml").read_bytes() == before and not llm.calls


async def test_pause_interrupts_inflight_llm_and_resume_drains_queue(controlled, monkeypatch):
    app, tg, llm = controlled
    started = asyncio.Event()
    cancelled = asyncio.Event()
    original = llm.chat

    async def slow(p, m):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(llm, "chat", slow)
    await tg.incoming(43, "Waiting for an answer")
    await asyncio.wait_for(started.wait(), 2)
    await tg.incoming(42, "/pause on")
    await asyncio.wait_for(cancelled.wait(), 2)
    await tg.incoming(42, "Owner is queued, too")
    assert config().paused
    monkeypatch.setattr(llm, "chat", original)
    await tg.incoming(42, "/resume")
    await wait_until(lambda: len(llm.calls) >= 2)
    assert not config().paused


async def test_chat_sessions_system_prompts_and_snapshots_are_isolated(controlled):
    app, tg, llm = controlled
    Path("data/working_memory.md").write_text("OWNER_ONLY_TASK")
    app.working_memory_path(43).parent.mkdir(exist_ok=True)
    app.working_memory_path(43).write_text("FRIEND_ONLY_TASK")
    await tg.incoming(42, "OWNER_SECRET")
    await wait_until(lambda: len(llm.calls) >= 1)
    await tg.incoming(43, "FRIEND_SECRET")
    await wait_until(lambda: len(llm.calls) >= 2)
    await tg.incoming(42, "Continue owner chat")
    await wait_until(lambda: len(llm.calls) >= 3)
    owner1, friend, owner2 = llm.calls[:3]
    assert "OWNER_ONLY_TASK" in owner1[0].system_prompt
    assert "OWNER_ONLY_TASK" not in friend[0].system_prompt and "FRIEND_ONLY_TASK" in friend[0].system_prompt
    assert "OWNER_SECRET" not in str([m.content for m in friend[1]])
    assert "FRIEND_SECRET" not in str([m.content for m in owner2[1]])
    assert "OWNER_SECRET" in str([m.content for m in owner2[1]])
    await wait_until(lambda: Path("data/contexts/43.json").is_file())
    snapshot = json.loads(Path("data/contexts/43.json").read_text())
    assert snapshot["chat_id"] == 43 and "OWNER_SECRET" not in str(snapshot)
    assert Path("data/contexts/43.json").stat().st_mode & 0o777 == 0o600
    app.contexts.pop(43)
    assert "FRIEND_SECRET" in str([m.content for m in app.get_context(43)])


async def test_opening_another_chat_does_not_change_private_context(controlled):
    app, tg, llm = controlled
    ctx = Session([Message(content="OWNER_SECRET")], chat_id=42)
    tools = ToolSet()
    result = await app.open_telegram_chat(tools, 43, ctx)
    assert "isolation" in result and not tools.handlers()
    assert not any(a[0] == "open_chat" for a in tg.actions)


async def test_diary_and_working_memory_dump_use_only_the_source_chat(controlled):
    app, tg, llm = controlled
    owner = app.get_context(42)
    owner << Message(content="OWNER_CONTEXT_SECRET")
    friend = app.get_context(43)
    friend << Message(content="FRIEND_CONTEXT_SECRET")
    Path("data/working_memory.md").write_text("OWNER_WORKING_SECRET")

    def summarize(params, messages):
        assert "OWNER_CONTEXT_SECRET" not in str([m.content for m in messages])
        assert "OWNER_WORKING_SECRET" not in params.system_prompt
        last = messages[-1].content
        if "What are important things" in last:
            return assistant("FRIEND_WORKING_TASK")
        if "memory cartographer" in params.system_prompt:
            return assistant('{"entities":[{"name":"Friend","summary":"FRIEND_MEMORY"}],"relations":[]}')
        return assistant("FRIEND_DIARY: We discussed the friend's plans and promises for next week.")

    llm.default = summarize
    await app.diary_dump_messages(friend)
    assert "OWNER_WORKING_SECRET" == Path("data/working_memory.md").read_text()
    assert app.working_memory_path(43).read_text() == "FRIEND_WORKING_TASK"
    assert owner and not friend
    episodes = app.memory.episodes(chat_id=43)
    assert episodes and all(p.source_chat_id == 43 and p.visibility == "private" for p in episodes)
    assert not app.memory.episodes(chat_id=42)


async def test_multiple_workers_do_not_mix_chat_contexts(controlled):
    app, tg, llm = controlled
    config().workerCount = 2
    app._sync_workers()
    await tg.incoming(42, "MULTI_OWNER_SECRET")
    await tg.incoming(43, "MULTI_FRIEND_SECRET")
    await wait_until(lambda: len(llm.calls) >= 2)
    assert len(app.workers) == 2
    for _, messages in llm.calls:
        text = str([m.content for m in messages])
        assert not ("MULTI_OWNER_SECRET" in text and "MULTI_FRIEND_SECRET" in text)


async def test_proactive_prompt_never_contains_private_diary(controlled):
    app, tg, llm = controlled
    Path("data/working_memory.md").write_text("OWNER_PRIVATE_TASK")
    app.memory.add_episode("OWNER_PRIVATE_DIARY: a highly personal conversation", source_chat_id=42)
    app.act_proactively()
    await wait_until(lambda: llm.calls)
    assert "OWNER_PRIVATE_DIARY" not in str([m.content for m in llm.calls[0][1]])
    assert "OWNER_PRIVATE_TASK" not in llm.calls[0][0].system_prompt


def test_session_copy_preserves_scope():
    context = Session([Message(content="private")], chat_id=43)
    assert context.copy().chat_id == 43


async def test_youtube_capability_registers_read_only_tools(controlled):
    app, tg, llm = controlled
    config().capabilityYouTube = True
    actions = ToolSet()
    app.update_tools(actions, Session(chat_id=43))
    assert {"search_youtube", "read_youtube_video"} <= actions.handlers().keys()
    config().capabilityYouTube = False
    actions = ToolSet()
    app.update_tools(actions, Session(chat_id=43))
    assert "search_youtube" not in actions.handlers()


def test_config_update_failure_keeps_old_file(monkeypatch):
    Path("config.toml").write_text(cfgmod.render(config()))
    original = Path("config.toml").read_bytes()
    import kuni.util as util

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(util.os, "replace", fail)
    with pytest.raises(OSError):
        cfgmod.update_general_settings(paused=True)
    assert Path("config.toml").read_bytes() == original and not config().paused


async def test_legacy_memory_is_quarantined_owner_only(controlled):
    app, tg, llm = controlled
    from kuni.memory.graph import GraphMemory

    memory = GraphMemory()
    old, _ = memory.upsert_entity("Legacy", summary="PRIVATE_OLD_DIARY")
    new = TelegramApp(tg, llm, working_dir="other_data", memory=memory)
    try:
        assert old.source_chat_id == 42 and old.visibility == "private"
        assert memory.recall("Legacy", chat_id=42).entities
        assert not memory.recall("Legacy", chat_id=43).entities
    finally:
        await new.close()
