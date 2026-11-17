"""The whole machine: Telegram event -> notification -> worker -> opened chat -> LLM -> tools -> memory."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import pytest

from conftest import FakeLLM, assistant, wait_until
from kuni import config as cfgmod
from kuni import worker as workermod
from kuni.config import config
from kuni.llm import Message, ROLE_USER, Session
from kuni.telegram.fake import FakeTelegram
from kuni.telegram.types import Chat, ChatType, MemberStatus, Message as TgMessage
from kuni.telegram_app import TelegramApp
from kuni.tools import Ctx, ToolSet
from kuni.tools_impl import telegram_tools

EXTRACTION = {"entities": [{"name": "Алекс", "type": "person", "summary": "друг"}, {"name": "джаз", "type": "topic"}],
              "relations": [{"source": "Алекс", "target": "джаз", "relation": "likes", "fact": "Алекс любит джаз.", "confidence": 0.5}]}

SEND = ("send_telegram_message", {"text": "привет~", "photo_filename": None, "audio_filename": None,
                                  "reply_to_message_id": None, "allow_typos": False})
WAIT = ("wait", {})


def smart_default(p, m):
    """What the LLM does when the script is exhausted: handles the maintenance prompts, otherwise `wait`."""
    last = m[-1].content if m else ""
    if "What are important things in timespan" in last:
        return assistant("- ответить Алексу про джаз — последнее обновление: Nov 15\nЭмоциональное состояние: joy")
    if "open diary and share your thoughts" in last:
        return assistant("Алекс рассказал мне про джаз, мы долго болтали о музыке и о его работе в Яндексе.\n---\n"
                         "Я пообещала Алексу подумать над рецептом пасты и написать ему в пятницу вечером.")
    if "memory cartographer" in p.system_prompt:
        return assistant(json.dumps(EXTRACTION, ensure_ascii=False))
    return assistant("", [WAIT])


@pytest.fixture
async def world():
    cfg = config()
    cfg.papikChatId = 42
    cfg.lockdown = cfgmod.LockdownMode.PAPIK_ONLY
    tg = FakeTelegram(my_id=1000)
    tg.add_user(42, "Alex", "Smith", "alex")
    made = []

    def make(script=(), **kw) -> tuple[TelegramApp, FakeLLM]:
        llm = FakeLLM(script, default=smart_default)
        app = TelegramApp(tg, llm, **kw)
        made.append(app)
        return app, llm

    class W:
        telegram = tg
        make_app = staticmethod(make)

    yield W
    for a in made:
        await a.close()


async def done(tg: FakeTelegram):
    await wait_until(lambda: ("set_online", False) in tg.actions, what="the notification to be fully processed")


def tool_names(llm: FakeLLM, call: int = 0) -> list[str]:
    return [t["function"]["name"] for t in llm.calls[call][0].tools]


# ---------------------------------------------------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------------------------------------------------
async def test_a_dm_from_the_owner_gets_read_answered_and_marked_read(world):
    tg = world.telegram
    app, llm = world.make_app([assistant("", [SEND]), assistant("", [WAIT])])
    msg = await tg.incoming(42, "привет, как дела?")
    await done(tg)

    # she opened the chat like a human: online -> open -> read marks -> typing -> reply -> offline
    kinds = [a[0] for a in tg.actions]
    assert kinds.index("set_online") < kinds.index("open_chat") < kinds.index("view_messages") < kinds.index("send_message")
    assert ("view_messages", 42, [msg.id]) in tg.actions and tg.chats[42].unread_count == 0
    assert ("close_chat", 42) in tg.actions and ("chat_action", 42, None) in tg.actions and ("set_online", False) in tg.actions
    sent = [a for a in tg.actions if a[0] == "send_message"]
    assert [a[2] for a in sent] == ["привет~"]

    # the LLM saw: the system prompt of the character, the chat tools + the global tools, the opened chat as text
    params, ctx = llm.calls[0]
    assert "You ACT EXCLUSIVELY BY CALLING TOOLS." in params.system_prompt and "I'm an AI. I think I'm a person." in params.system_prompt
    assert "lockdown mode" in params.system_prompt
    names = tool_names(llm)
    assert {"send_telegram_message", "get_chat_photo", "react_with_emoji", "remove_message", "edit_message_text",
            "forward_message", "ask", "get_telegram_chats", "search_chats", "search_messages", "view_messages_around",
            "remove_and_ban_chat", "open_chat_by_id", "pause", "wait"} == set(names)
    first = ctx[0].content
    assert first.startswith('You switched to the chat "Alex Smith" in Telegram. You see last messages:')
    assert f'<message message_id="{msg.id}"' in first and "привет, как дела?" in first and " unread" in first
    assert "You are in private chat with Alex Smith" in first and "Current time:" not in first.split("<message")[0]
    # after `wait` the work is done; the 2nd call saw the result of the send
    assert "Message sent successfully" in llm.calls[1][1][-1].content
    assert "What's your next action?" in llm.calls[1][1][-1].content


async def test_the_context_is_preserved_between_notifications_and_owner_wakes_her_first(world):
    tg = world.telegram
    app, llm = world.make_app([assistant("", [WAIT]), assistant("", [WAIT])])
    await tg.incoming(42, "first")
    await done(tg)
    tg.actions.clear()
    await tg.incoming(42, "second")
    await done(tg)
    ctx = llm.calls[1][1]
    assert any("first" in m.content for m in ctx) and "second" in ctx[-1].content or any("second" in m.content for m in ctx)
    assert len([m for m in ctx if m.role == "user"]) == 2  # the dialogue continues in the same context


async def test_notifications_from_strangers_are_ignored_under_lockdown(world):
    tg = world.telegram
    tg.add_user(43, "Eve", contact=False)
    app, llm = world.make_app()
    await tg.incoming(43, "hello? anyone?")
    await asyncio.sleep(0.05)  # give the worker a chance to (wrongly) pick something up
    assert not llm.calls and not app.notification_manager._notifications  # noqa: SLF001
    assert tg.chats[43].unread_count == 1  # nothing was read


# ---------------------------------------------------------------------------------------------------------------------
# notification texts / priorities
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def captured(world):
    """Captures notifications instead of letting the worker process them."""
    def install(app):
        got = []
        app.notification_manager.pass_notification_to_ai = lambda n: got.append(n) or None
        return got

    return install


async def test_notification_texts_for_dm_group_and_channel(world, captured):
    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_user(9, "Bob", "Builder")
    tg.add_group(-100, "Devs")
    tg.add_group(-200, "News", ChatType.CHANNEL)
    app, _ = world.make_app()
    got = captured(app)
    await tg.incoming(42, "hi")
    await tg.incoming(-100, "hey kuni", sender_id=9)
    await tg.incoming(-200, "breaking news", sender_id=-200, sender_is_chat=True)
    dm, group, channel = got
    assert dm.message == ('<notification chat_id="42">\nYou received a direct message from Alex Smith (chat_id = 42)\n</notification>\n'
                          "You don't have any chat open. Use #open tool to open the chat")
    assert 'Bob Builder (user_id = 9) sent a message in group chat "Devs" (chat_id = -100)' in group.message
    assert 'Channel "News" (chat_id=-200) created a new post' in channel.message
    assert dm.pin == '<chat id="42" />' and group.pin == '<chat id="-100" />'
    assert list(dm.actions.handlers()) == ["open"] and "Open \"Alex Smith\" chat" in dm.actions.handlers()["open"].description
    assert [n.priority for n in got] == [1000, 0, 0]  # the owner is processed ASAP


async def test_channel_posts_wait_to_be_read_when_notifications_are_off(world, captured, caplog):
    import logging

    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    config().channelNotifications = False
    tg.add_group(-100, "Devs")
    tg.add_group(-200, "News", ChatType.CHANNEL)
    app, llm = world.make_app()
    got = captured(app)
    caplog.set_level(logging.INFO, logger="MessageFlow")
    await tg.incoming(-200, "SECRET_POST", sender_id=-200, sender_is_chat=True)
    await tg.incoming(-100, "hey", sender_id=42)  # groups still notify
    assert [n.chat_id for n in got] == [-100]
    assert "reason=channel_read_on_demand" in caplog.text and "SECRET_POST" not in caplog.text
    assert tg.chats[-200].unread_count == 1  # untouched: she reads it when she decides to

    for proactive in (False, True):
        tool = telegram_tools.get_telegram_chats(tg, llm, proactive)
        text = await tool.handler(Ctx(tools=None, args={}, temporary_context=Session(chat_id=0), all_tool_calls=[]))
        assert ('chat_id="-200" title="News" type="channel" unread_count="1"' in text) is True
        assert "never send you notifications" in text


async def test_channel_notifications_default_keeps_the_original_behaviour(world, captured):
    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    assert config().channelNotifications is True
    tg.add_group(-200, "News", ChatType.CHANNEL)
    app, llm = world.make_app()
    got = captured(app)
    await tg.incoming(-200, "news", sender_id=-200, sender_is_chat=True)
    assert [n.chat_id for n in got] == [-200]
    tool = telegram_tools.get_telegram_chats(tg, llm, True)
    text = await tool.handler(Ctx(tools=None, args={}, temporary_context=Session(chat_id=0), all_tool_calls=[]))
    assert "type=\"channel\"" not in text and "never send you notifications" not in text


async def test_pinned_chats_only_wake_her_when_configured(world, captured):
    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_user(9, "Bob")
    tg.chats[9].pinned = True
    app, _ = world.make_app()
    got = captured(app)
    await tg.incoming(9, "a")
    config().wakeUpOnPinnedChat = True
    await tg.incoming(9, "b")
    assert [n.priority for n in got] == [0, 100]


async def test_muted_chats_are_ignored_with_a_probability(world, captured, monkeypatch):
    from kuni import telegram_app

    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_user(9, "Bob")
    tg.chats[9].muted = True
    app, _ = world.make_app()
    got = captured(app)
    monkeypatch.setattr(telegram_app.rng, "random", lambda: 0.5)  # < 0.8: ignored
    await tg.incoming(9, "spam")
    assert got == []
    monkeypatch.setattr(telegram_app.rng, "random", lambda: 0.95)  # the 20% that slips through
    await tg.incoming(9, "spam again")
    assert len(got) == 1


async def test_chat_notification_filter_is_independent_of_lockdown(world, captured):
    tg = world.telegram
    cfg = config()
    cfg.lockdown, cfg.chatNotificationFilter = cfgmod.LockdownMode.NONE, cfgmod.LockdownMode.CONTACTS_ONLY
    tg.add_user(9, "Bob")
    tg.add_user(8, "Stranger", contact=False)
    app, _ = world.make_app()
    got = captured(app)
    await tg.incoming(9, "from a contact")
    await tg.incoming(8, "from a stranger")
    assert len(got) == 1 and "chat_id=\"9\"" in got[0].message


async def test_version_command_is_answered_without_the_llm(world, captured):
    tg = world.telegram
    app, llm = world.make_app()
    got = captured(app)
    m = await tg.incoming(42, "/version")
    assert got == [] and not llm.calls
    reply = [a for a in tg.actions if a[0] == "send_message"][0]
    assert reply[1] == 42 and "Kernel: kuni" in reply[2] and "Project name: kuni" in reply[2] and reply[5] == m.id


async def test_unread_chats_become_notifications_on_startup_oldest_first(world, captured):
    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_user(9, "Bob")
    tg.add_user(8, "Cat")
    tg.add_user(7, "Quiet")
    app, _ = world.make_app()
    got = captured(app)
    for chat_id, text in ((8, "old unread"), (9, "newer unread")):
        m = await tg.incoming(chat_id, text)
        m.date = 1_700_000_000 + chat_id * -100
    got.clear()  # live notifications are not what we test here
    await app.send_notifications_on_init()
    assert [n.message.split('"')[1] for n in got] == ["8", "9"] or sorted(n.message.split('"')[1] for n in got) == ["8", "9"]
    config().checkChatsOnStartup = False
    got.clear()
    await app.send_notifications_on_init()
    assert got == []


# ---------------------------------------------------------------------------------------------------------------------
# opening chats: tools depend on what the account may do there
# ---------------------------------------------------------------------------------------------------------------------
async def open_chat(app: TelegramApp, chat_id: int, ctx=None) -> tuple[str, ToolSet]:
    tools = ToolSet()
    text = await app.open_telegram_chat(tools, chat_id, ctx if ctx is not None else Session())
    return text, tools


async def test_opening_a_group_where_she_is_not_a_member_offers_react_and_join_only(world):
    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    config().canJoinChats = True
    tg.add_group(-100, "Devs")
    tg.statuses[(-100, 1000)] = MemberStatus.LEFT
    app, _ = world.make_app()
    text, tools = await open_chat(app, -100)
    assert set(tools.handlers()) == {"react_with_emoji", "join_chat"}
    assert 'You are NOT a member of "Devs"' in text
    config().canJoinChats = False
    _, tools = await open_chat(app, -100)
    assert set(tools.handlers()) == {"react_with_emoji"}


async def test_member_admin_and_channel_toolsets(world):
    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_group(-100, "Devs")
    tg.add_group(-200, "News", ChatType.CHANNEL)
    app, _ = world.make_app()
    text, tools = await open_chat(app, -100)
    assert "leave_chat" in tools.handlers() and "group_admin_ban_user" not in tools.handlers()
    assert 'You are in group chat called "Devs".' in text and "sticker_send" not in tools.handlers()
    tg.statuses[(-100, 1000)] = MemberStatus.ADMIN
    _, tools = await open_chat(app, -100)
    assert {"group_admin_remove_message", "group_admin_ban_user", "group_admin_set_user_tag", "leave_chat"} <= set(tools.handlers())
    config().canLeaveChats, config().capabilityUseStickers = False, True
    _, tools = await open_chat(app, -100)
    assert "leave_chat" not in tools.handlers() and "sticker_send" in tools.handlers()
    text, tools = await open_chat(app, -200)
    assert set(tools.handlers()) == {"react_with_emoji", "forward_message"} and "You can't respond in telegram channels" in text


async def test_lockdown_hides_chats_from_open_chat_by_id(world):
    tg = world.telegram
    tg.add_user(9, "Bob")
    app, llm = world.make_app()
    actions = ToolSet()
    app.update_tools(actions, Session())
    handler = actions.handlers()["open_chat_by_id"].handler
    from kuni.llm import Function, ToolCall
    from kuni.tools import Ctx

    calls = [ToolCall("1", 0, "function", Function("open_chat_by_id", "{}"))]
    assert await handler(Ctx(tools=actions, args={"chat_id": 9}, temporary_context=Session(), all_tool_calls=calls)) == "No such chat"
    assert not any(a[0] == "open_chat" for a in tg.actions)
    two = calls * 2
    assert await handler(Ctx(tools=actions, args={"chat_id": 42}, temporary_context=Session(), all_tool_calls=two)) == "You can only call this tool once per turn."
    out = await handler(Ctx(tools=actions, args={"chat_id": 42}, temporary_context=Session(), all_tool_calls=calls))
    assert out.startswith('You switched to the chat "Alex Smith"') and "send_telegram_message" in actions.handlers()


async def test_owner_chat_can_send_her_to_read_channels_but_nothing_else(world, captured):
    from kuni.llm import Function, ToolCall

    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_user(9, "Bob")
    tg.add_group(-200, "News", ChatType.CHANNEL)
    tg.add_group(-300, "Devs")
    tg.chats[-200].unread_count = 3
    app, llm = world.make_app()
    got = captured(app)
    calls = [ToolCall("1", 0, "function", Function("open_chat_by_id", "{}"))]

    async def run(scope_chat, name, args):
        actions = ToolSet()
        app.update_tools(actions, Session(chat_id=scope_chat))
        return await actions.handlers()[name].handler(
            Ctx(tools=actions, args=args, temporary_context=Session(chat_id=scope_chat), all_tool_calls=calls))

    listing = await run(42, "get_telegram_chats", {})
    assert 'chat_id="-200" title="News" type="channel" unread_count="3"' in listing
    assert "Devs" not in listing and "Bob" not in listing  # groups and private chats stay hidden
    assert "queued separately" in await run(42, "open_chat_by_id", {"chat_id": -200})
    assert [n.chat_id for n in got] == [-200]
    assert "isolation" in await run(42, "open_chat_by_id", {"chat_id": -300})  # not a channel: still refused
    assert "isolation" in await run(42, "open_chat_by_id", {"chat_id": 9})
    assert [n.chat_id for n in got] == [-200]

    # another contact's chat gets neither the list nor the shortcut
    assert "News" not in await run(9, "get_telegram_chats", {})
    assert "isolation" in await run(9, "open_chat_by_id", {"chat_id": -200})
    assert [n.chat_id for n in got] == [-200]
    assert not any(a[0] == "open_chat" for a in tg.actions)


async def test_repeated_open_requests_add_one_turn_and_say_it_worked(world):
    from kuni.llm import Function, ToolCall
    from kuni.notifications import Notification

    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_user(9, "Bob")
    app, llm = world.make_app()
    for worker in app.workers:  # nothing may consume the queue behind the test's back
        worker.close()
    actions = ToolSet()
    app.update_tools(actions, Session(chat_id=0))
    calls = [ToolCall("1", 0, "function", Function("open_chat_by_id", "{}"))]
    handler = actions.handlers()["open_chat_by_id"].handler
    answers = [await handler(Ctx(tools=actions, args={"chat_id": 9}, temporary_context=Session(chat_id=0), all_tool_calls=calls))
               for _ in range(3)]
    assert all("queued separately" in a and "do not repeat the call" in a for a in answers)
    assert [h.notification.chat_id for h in app.notification_manager._notifications] == [9]
    app.notification_manager._notifications.clear()
    app.workers[0].current_notification = Notification("x", pin='<chat id="9" />')  # already being read
    app._queue_chat(9)
    assert not app.notification_manager._notifications


async def test_history_loading_stops_when_the_context_already_has_the_messages(world):
    tg = world.telegram
    msgs = [await tg.incoming(42, f"message number {i} " + "x" * 50) for i in range(40)]
    app, _ = world.make_app()
    text, _ = await open_chat(app, 42)
    # chatMaxHistoryLength = 2000 chars: 100-char overhead + ~65 chars of text per message => about a dozen messages
    shown = text.count("<message message_id=")
    assert 10 <= shown <= 16 and "message number 39" in text and "message number 0 " not in text

    already = Session([Message(role=ROLE_USER, content=f'<message message_id="{msgs[-3].id}" date="x">old</message>')])
    text, _ = await open_chat(app, 42, already)
    assert text.count("<message message_id=") == 3  # stops at the message she has already seen


async def test_empty_chat_and_new_person_instruction(world):
    tg = world.telegram
    tg.add_user(42, "Alex", "Smith")
    tg.messages[42] = []
    app, _ = world.make_app()
    text, _ = await open_chat(app, 42)
    assert "This chat is empty!" not in text  # can_write_to_a_new_person is off by default
    config().canWriteToANewPerson = True
    text, _ = await open_chat(app, 42)
    assert "This chat is empty! Only proceed if you looked up a @username" in text


# ---------------------------------------------------------------------------------------------------------------------
# the worker loop
# ---------------------------------------------------------------------------------------------------------------------
async def test_replies_without_tool_calls_are_corrected(world):
    tg = world.telegram
    app, llm = world.make_app([
        assistant("#send_telegram_message hello there"),
        assistant('<message message_id="1" sender="Kuni">hello</message>'),
        assistant("just thinking out loud"),
        assistant("", [WAIT]),
    ])
    await tg.incoming(42, "hi")
    await done(tg)
    nudges = [m.content for m in llm.calls[3][1] if m.role == "user" and m.content.startswith("Nice thoughts!")]
    assert len(nudges) == 3 and nudges[0].endswith("provided is not visible to anyone but you.")
    assert nudges[1].endswith("Call #wait if you are unsure.") and nudges[2].endswith("Call #wait if you are unsure.")
    assert not any(a[0] == "send_message" for a in tg.actions)


async def test_the_context_is_dumped_to_the_graph_when_the_token_budget_is_exceeded(world):
    tg = world.telegram
    app, llm = world.make_app([assistant("", [WAIT], usage_total=50_000)])
    await tg.incoming(42, "расскажи про джаз")
    await done(tg)
    await wait_until(lambda: app.memory.episodes(), what="the memory dump")
    await wait_until(lambda: not app.workers[0].temporary_context, what="the context to be cleared")
    assert len(app.memory.episodes()) == 2 and "джаз" in app.memory.episodes()[-1].summary + app.memory.episodes()[0].summary
    assert app.memory.resolve("Алекс") is not None and any(e.relation == "likes" for e in app.memory.edges.values())
    wm = (Path("data") / "working_memory.md").read_text(encoding="utf-8")
    assert "последнее обновление" in wm
    # the working memory goes to the system prompt of the next session, with the emotional-state instruction
    sp = app.get_system_prompt()
    assert "<things_to_remember>" in sp and "Эмоциональное состояние" in sp and "Your behaviour must be highly influenced" in sp
    # the diary prompt asked for pages of the whole context, with no tools
    diary_call = next(c for c in llm.calls if "open diary" in c[1][-1].content)
    assert diary_call[0].tools == [] and diary_call[1][0].content.startswith("You switched to the chat")


async def test_a_second_dump_preserves_unfinished_tasks_from_the_first(world):
    tg = world.telegram
    app, llm = world.make_app()
    (Path("data")).mkdir(exist_ok=True)
    (Path("data") / "working_memory.md").write_text("- купить хлеб — последнее обновление: Nov 14")
    ctx = app.workers[0].temporary_context
    ctx << Message(role=ROLE_USER, content="hello")
    await app.diary_dump_messages(ctx)
    prompt = next(c[1][-1].content for c in llm.calls if "What are important things" in c[1][-1].content)
    assert "<previous_things_to_remember>\n- купить хлеб — последнее обновление: Nov 14\n</previous_things_to_remember>" in prompt
    assert "You MUST preserve ALL items verbatim" in prompt and "Nov 15" not in prompt.split("Example format:")[0]
    assert not ctx and app.get_system_prompt().count("ответить Алексу про джаз") == 1  # the NEW working memory


async def test_a_failed_dump_keeps_the_context(world):
    tg = world.telegram
    app, llm = world.make_app()

    def boom(p, m):
        raise RuntimeError("llm is down")

    llm.default = boom
    ctx = app.workers[0].temporary_context
    ctx << Message(role=ROLE_USER, content="hello")
    with pytest.raises(RuntimeError):
        await app.diary_dump_messages(ctx)
    assert len(ctx) == 1  # nothing is lost
    assert not (Path("data") / "working_memory.md").exists()  # the old working memory isn't overwritten by garbage


async def test_a_tool_error_is_reported_to_the_llm_and_does_not_stop_the_turn(world):
    tg = world.telegram
    app, llm = world.make_app([assistant("", [("send_telegram_message", {"text": "", "photo_filename": None, "audio_filename": None,
                                                                         "reply_to_message_id": None})]), assistant("", [WAIT])])
    await tg.incoming(42, "hi")
    await done(tg)
    assert llm.calls[1][1][-1].content.startswith('Error: At least one of "text"')


async def test_ask_is_reminded_until_it_is_used(world):
    config().remindUseAsk = True
    tg = world.telegram
    app, llm = world.make_app([assistant("", [SEND]), assistant("", [WAIT])])
    await tg.incoming(42, "hi")
    await done(tg)
    assert "Have you called #ask yet this turn?" in llm.calls[0][1][-1].content
    assert "you sent a message without consulting #ask" in llm.calls[1][1][-1].content


async def test_begging_the_llm_to_be_lazy_keeps_the_api_valid(world, monkeypatch):
    cfgmod.TESTING = False  # the lazy-bait is disabled in tests, like in the C++ kernel
    config().suggestIgnoreChance = 0.5
    monkeypatch.setattr(workermod.rng, "random", lambda: 0.0)
    tg = world.telegram
    # first the chat is opened by the shortcut; the LLM wants to send; she begs it to think again; then it waits
    app, llm = world.make_app([assistant("", [SEND]), assistant("", [WAIT])])
    await tg.incoming(42, "hi")
    await done(tg)
    second = llm.calls[1][1]
    assistant_msgs = [i for i, m in enumerate(second) if m.role == "assistant" and m.tool_calls]
    tool_msgs = [i for i, m in enumerate(second) if m.role == "tool"]
    assert assistant_msgs and tool_msgs and assistant_msgs[0] < tool_msgs[0]  # no orphan tool messages: providers reject them
    assert "do you really want to continue" in second[tool_msgs[0]].content.lower()
    assert not any(a[0] == "send_message" for a in tg.actions)  # `wait` is never begged about, send was postponed


async def test_acting_proactively_reads_a_random_page_and_pushes_for_more_after_wait(world):
    tg = world.telegram
    app, llm = world.make_app([assistant("", [WAIT]), assistant("", [WAIT])])
    app.memory.add_episode("Страница дневника про поход в горы с Алексом в прошлые выходные.", visibility="public")
    app.act_proactively()
    await wait_until(lambda: len(llm.calls) >= 2, what="two LLM turns")
    first_user = llm.calls[0][1][0].content
    assert first_user.startswith("<your_diary_page just_for_reasoning no_plagiarism no_copy>\nСтраница дневника про поход в горы")
    assert "It's time to reflect on your thoughts!" in first_user and "Act proactively!" in first_user
    assert any("You should write someone else and be more proactive." in m.content for m in llm.calls[1][1])
    await wait_until(lambda: not app.is_acting_proactively, what="proactive mode to finish")


async def test_proactive_mode_hides_active_chats_from_the_chat_list(world):
    tg = world.telegram
    tg.chats[42].unread_count = 2
    tg.chats[42].last_message = TgMessage(1, 42, 1_700_000_000, sender_id=42, text="hello")
    app, llm = world.make_app([assistant("", [("get_telegram_chats", {})]), assistant("", [WAIT]), assistant("", [WAIT])])
    config().lockdown = cfgmod.LockdownMode.NONE
    app.act_proactively()
    await wait_until(lambda: len(llm.calls) >= 2, what="the chat list result")
    assert 'chat_id="42"' not in llm.calls[1][1][-1].content  # active chats come as notifications anyway


async def test_tool_call_events_feed_the_metrics(world):
    tg = world.telegram
    app, llm = world.make_app([assistant("", [SEND]), assistant("", [WAIT])])
    events = []
    app.tool_call_fired.append(events.append)
    await tg.incoming(42, "hi")
    await done(tg)
    assert [e.tool_name for e in events] == ["send_telegram_message"]  # wait/pause are not counted
    assert events[0].breadcrumb_labels["chat"] == "Alex Smith" and events[0].breadcrumb_labels["scenario"] == "reply to user"
    assert events[0].last_opened_chat_last_message_time is not None


async def test_metrics_exporter_records_usage_and_tool_calls():
    import urllib.request

    from kuni.llm import Usage
    from kuni.metrics import Exporter, MetricsBreadcrumbs

    bc = MetricsBreadcrumbs()
    exp = Exporter(bc, port=0)
    port = exp._server[0].server_address[1]
    try:
        llm = FakeLLM()
        exp.register_openai(llm)
        with bc.point("chat", "Alex"), bc.point("function", "notification processing loop"):
            for cb in llm.response_metrics:
                cb("agnes", Usage(100, 7, 107, 40, 60))
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode()
    finally:
        exp.close()
    assert 'llm_usage_input_total{chat="Alex",function="notification processing loop",model="agnes",scenario=""} 100.0' in body
    assert 'llm_usage_output_gauge{model="agnes"} 7.0' in body


def test_breadcrumbs_are_isolated_between_concurrent_tasks():
    import asyncio

    from kuni.metrics import MetricsBreadcrumbs

    bc = MetricsBreadcrumbs()
    seen = {}

    async def worker(name):
        bc.set("chat", name)
        await asyncio.sleep(0.01)
        seen[name] = bc.value()["chat"]

    async def main():
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(main())
    assert seen == {"a": "a", "b": "b"}  # the C++ kernel shared one mutable map between workers


async def test_typing_indicator_is_sent_in_chats_but_never_in_broadcast_channels(world):
    from kuni.llm import Function, ToolCall
    from conftest import assistant

    tg = world.telegram
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_group(-200, "News", ChatType.CHANNEL)
    app, _ = world.make_app()
    response = assistant("", [WAIT])

    await open_chat(app, 42)
    app.on_response_assembling(response)
    await wait_until(lambda: ("chat_action", 42, "typing") in tg.actions, what="the typing action in a DM")

    app._last_typing = 0.0
    tg.actions.clear()
    await open_chat(app, -200)
    app.on_response_assembling(response)
    await asyncio.sleep(0.05)
    assert not any(a[0] == "chat_action" and a[1] == -200 and a[2] == "typing" for a in tg.actions)


async def _join(app, scope, link="https://t.me/+AbCdEfGh"):
    session = Session(chat_id=scope)
    actions = ToolSet()
    app.update_tools(actions, session)
    tool = actions.handlers().get("join_chat_by_link")
    if tool is None:
        return None
    return await tool.handler(Ctx(tools=actions, args={"invite_link": link}, temporary_context=session, all_tool_calls=[]))


async def test_joining_by_link_is_for_owner_and_own_initiative_and_the_chat_is_queued_not_opened(world, captured):
    import time

    tg = world.telegram
    tg.add_user(43, "Friend")
    config().canJoinChats = True
    app, _ = world.make_app()
    got = captured(app)

    assert await _join(app, 43) is None  # another contact's chat does not even get the tool
    assert not any(a[0] == "join_chat_by_invite_link" for a in tg.actions)
    for scope in (42, 0):
        out = await _join(app, scope)
        assert out.startswith('Joined "Joined chat"') and "queued" in out
    assert [a for a in tg.actions if a[0] == "join_chat_by_invite_link"] == [("join_chat_by_invite_link", "https://t.me/+AbCdEfGh")] * 2
    assert not any(a[0] == "open_chat" for a in tg.actions)  # never read inside the chat that asked
    assert [n.chat_id for n in got] == [-1001234, -1001234]

    for bad in ("two words", "x" * 201, "  "):
        with pytest.raises(ValueError):
            await _join(app, 42, bad)

    config().canJoinChats = False
    assert await _join(app, 42) is None

    config().canJoinChats = True
    app._joins = [time.monotonic()] * 5  # the daily budget is used up
    before = len(tg.actions)
    assert (await _join(app, 42)).startswith("Error: you have already joined 5")
    assert len(tg.actions) == before


async def test_join_chat_tool_in_a_not_joined_channel_shares_the_daily_budget(world):
    import time

    tg = world.telegram
    config().canJoinChats = True
    config().lockdown = cfgmod.LockdownMode.NONE
    tg.add_group(-100, "News", ChatType.CHANNEL)
    tg.statuses[(-100, 1000)] = MemberStatus.LEFT
    app, _ = world.make_app()
    _, tools = await open_chat(app, -100)
    join = tools.handlers()["join_chat"]
    ctx = Ctx(tools=tools, args={}, temporary_context=Session(), all_tool_calls=[])
    app._joins = [time.monotonic()] * 5
    assert (await join.handler(ctx)).startswith("Error: you have already joined 5")
    assert ("join_chat", -100) not in tg.actions
    app._joins = []
    assert (await join.handler(ctx)).startswith('You joined "News"')
    assert ("join_chat", -100) in tg.actions and len(app._joins) == 1
