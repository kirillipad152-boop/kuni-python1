from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conftest import FakeLLM
from kuni.config import LockdownMode, config
from kuni.llm import Function, Session, ToolCall
from kuni.telegram.fake import FakeTelegram
from kuni.telegram.types import Chat, ChatType, Message
from kuni.tools import Ctx, ToolSet
from kuni.tools_impl import media_tools as mt
from kuni.tools_impl import send_telegram_message as stm
from kuni.tools_impl import telegram_tools as tt


def session(scope):
    result = Session()
    result.chat_id = scope
    return result


async def execute(tool, args, scope, *, direct=False):
    context = session(scope)
    if direct:
        return await tool.handler(Ctx(ToolSet([tool]), args, context, []))
    call = ToolCall("privacy", 0, "function", Function(tool.name, json.dumps(args)))
    result = await ToolSet([tool]).handle_tool_calls([call], temporary_context=context)
    return result[-1].content


@pytest.fixture
def tg():
    telegram = FakeTelegram()
    telegram.add_user(42, "Current", username="current")
    telegram.add_user(43, "Owner", username="owner")
    config().papikChatId = 43
    return telegram


def send_args(**extra):
    return {"text": "hello", "photo_filename": None, "audio_filename": None,
            "reply_to_message_id": None, "allow_typos": False, **extra}


OPENED_ACTIONS = [
    (lambda tg, chat: stm.send_telegram_message(tg, FakeLLM(), chat, []), send_args()),
    (tt.react_with_emoji, {"message_id": 1, "emoji": "👍"}),
    (tt.remove_message, {"message_id": 1}),
    (tt.edit_message_text, {"message_id": 1, "text": "new"}),
    (tt.forward_message, {"message_ids": [1], "to_chat_id": 42, "comment": "secret context"}),
    (tt.join_chat, {}),
    (tt.leave_chat, {}),
    (lambda tg, chat: tt.get_chat_photo(tg, FakeLLM(), chat, Session()), {}),
    (tt.group_admin_remove_message, {"message_id": 1}),
    (tt.group_admin_ban_user, {"user_id": 43}),
    (tt.group_admin_set_user_tag, {"user_id": 43, "tag": "new"}),
    (tt.sticker_send, {"sticker_id": 555, "reply_to_message_id": None}),
]


@pytest.mark.parametrize("factory,args", OPENED_ACTIONS)
@pytest.mark.parametrize("scope", [0, 42])
async def test_captured_foreign_chat_is_denied_even_by_direct_handler(tg, monkeypatch, factory, args, scope):
    foreign = Chat(43, "Owner", ChatType.SUPERGROUP, has_photo=True)
    reads = ("get_message", "get_chat", "get_user", "get_saved_stickers", "download_chat_photo")
    for name in reads:
        monkeypatch.setattr(tg, name, AsyncMock(side_effect=AssertionError("foreign network read")))
    result = await execute(factory(tg, foreign), args, scope, direct=True)
    assert "other chats" in result
    assert tg.actions == []
    for name in reads:
        getattr(tg, name).assert_not_awaited()


@pytest.mark.parametrize("factory,args", OPENED_ACTIONS)
async def test_crafted_chat_id_cannot_redirect_opened_tools(tg, factory, args):
    own = Chat(42, "Current", ChatType.SUPERGROUP, has_photo=True)
    result = await execute(factory(tg, own), {**args, "chat_id": 43}, 42)
    assert "other chats" in result
    assert tg.actions == []


async def test_implicit_raw_search_is_local_not_global(tg, monkeypatch):
    await tg.incoming(42, "needle in current chat")
    await tg.incoming(43, "needle OWNER PRIVATE SECRET")
    global_search = AsyncMock(side_effect=AssertionError("global raw search"))
    monkeypatch.setattr(tg, "search_messages_global", global_search)
    result = await execute(tt.search_messages(tg, FakeLLM(), Session()), {"query": "needle"}, 42)
    assert "needle in current chat" in result
    assert "OWNER PRIVATE SECRET" not in result and 'chat_id="43"' not in result
    assert 'total_count="1"' in result
    global_search.assert_not_awaited()


@pytest.mark.parametrize("scope,chat_id", [(42, 43), (0, 42), (0, None)])
async def test_cross_chat_and_public_raw_search_denied_before_sender_resolution(tg, monkeypatch, scope, chat_id):
    for name in ("search_public_chat", "search_chat_messages", "search_messages_global"):
        monkeypatch.setattr(tg, name, AsyncMock(side_effect=AssertionError("private network read")))
    args = {"query": "secret", "sender_id": "@owner", "chat_id": chat_id}
    result = await execute(tt.search_messages(tg, FakeLLM(), Session()), args, scope)
    assert "other chats" in result
    for name in ("search_public_chat", "search_chat_messages", "search_messages_global"):
        getattr(tg, name).assert_not_awaited()


@pytest.mark.parametrize("scope,chat_id", [(42, 43), (0, 42)])
async def test_messages_around_cannot_open_another_chat(tg, monkeypatch, scope, chat_id):
    lookup = AsyncMock(side_effect=AssertionError("private network read"))
    monkeypatch.setattr(tg, "get_messages_around", lookup)
    result = await execute(tt.view_messages_around(tg, FakeLLM(), Session()),
                           {"chat_id": chat_id, "message_id": 1}, scope)
    assert "other chats" in result
    lookup.assert_not_awaited()


@pytest.mark.parametrize("extra", [{"to_chat_id": 43}, {"from_chat_id": 43}])
async def test_forwarding_to_owner_and_crafted_source_are_denied_before_read(tg, monkeypatch, extra):
    lookup = AsyncMock(side_effect=AssertionError("foreign network read"))
    monkeypatch.setattr(tg, "get_message", lookup)
    args = {"message_ids": [1], "to_chat_id": 42, "comment": "private current-chat context", **extra}
    result = await execute(tt.forward_message(tg, tg.chats[42]), args, 42)
    assert "other chats" in result and tg.actions == []
    lookup.assert_not_awaited()


async def test_same_chat_forward_and_comment_remain_available(tg):
    message = await tg.incoming(42, "own message")
    result = await execute(tt.forward_message(tg, tg.chats[42]),
                           {"message_ids": [message.id], "to_chat_id": 42, "comment": "own comment"}, 42)
    assert result == "Forwarded 1 message(s) successfully to chat_id=42"
    assert tg.actions[0] == ("forward_messages", 42, 42, [message.id])
    assert tg.actions[1][:3] == ("send_message", 42, "own comment")


@pytest.mark.parametrize("scope,chat_id", [(42, 43), (0, 42)])
async def test_remove_and_ban_cannot_target_arbitrary_chat(tg, monkeypatch, scope, chat_id):
    lookup = AsyncMock(side_effect=AssertionError("foreign network read"))
    monkeypatch.setattr(tg, "get_chat", lookup)
    result = await execute(tt.remove_and_ban_chat(tg), {"chat_id": chat_id}, scope)
    assert "other chats" in result and tg.actions == []
    lookup.assert_not_awaited()


async def test_scoped_lists_and_chat_search_never_fetch_foreign_previews(tg, monkeypatch):
    await tg.incoming(42, "own preview")
    await tg.incoming(43, "OWNER PRIVATE SECRET")
    for name in ("get_chats", "search_chats_on_server", "search_public_chat"):
        monkeypatch.setattr(tg, name, AsyncMock(side_effect=AssertionError("foreign chat lookup")))
    for tool, args in ((tt.get_telegram_chats(tg, FakeLLM(), True), {}),
                       (tt.search_chats(tg), {"query": "@current"})):
        result = await execute(tool, args, 42)
        assert 'chat_id="42"' in result and "own preview" in result
        assert 'chat_id="43"' not in result and "OWNER PRIVATE SECRET" not in result
    assert await execute(tt.search_chats(tg), {"query": "Owner"}, 42) == "No chats found satisfying your query."
    for name in ("get_chats", "search_chats_on_server", "search_public_chat"):
        getattr(tg, name).assert_not_awaited()


@pytest.mark.parametrize("command", ["get_telegram_chats", "search_chats"])
async def test_public_destination_selection_is_metadata_only(tg, monkeypatch, command):
    await tg.incoming(42, "CURRENT PRIVATE SECRET")
    await tg.incoming(43, "OWNER PRIVATE SECRET")
    monkeypatch.setattr(tg, "get_user", AsyncMock(side_effect=AssertionError("preview sender lookup")))
    if command == "get_telegram_chats":
        tool, args = tt.get_telegram_chats(tg, FakeLLM(), False), {}
    else:
        tool, args = tt.search_chats(tg), {"query": ""}
    result = await execute(tool, args, 0)
    assert 'chat_id="42" title="Current" type="private"' in result
    assert 'chat_id="43" title="Owner" type="private"' in result
    assert "PRIVATE SECRET" not in result
    assert "preview=" not in result and "last_message=" not in result and "unread_count=" not in result
    tg.get_user.assert_not_awaited()


@pytest.mark.parametrize("from_id,to_id,extra", [(42, 43, {}), (43, 42, {}), (42, 42, {"from_chat_id": 43})])
async def test_forwarding_comment_cannot_leak_to_owner_or_from_foreign_chat(tg, monkeypatch, from_id, to_id, extra):
    lookup = AsyncMock(side_effect=AssertionError("forward source lookup"))
    monkeypatch.setattr(tg, "get_message", lookup)
    tool = tt.forward_message(tg, tg.chats[from_id])
    result = await execute(tool, {"message_ids": [1], "to_chat_id": to_id,
                                  "comment": "full private context", **extra}, 42)
    assert "other chats" in result and tg.actions == []
    lookup.assert_not_awaited()


async def test_same_chat_forward_and_comment_still_work(tg):
    message = await tg.incoming(42, "share locally")
    result = await execute(tt.forward_message(tg, tg.chats[42]),
                           {"message_ids": [message.id], "to_chat_id": 42, "comment": "local comment"}, 42)
    assert "successfully" in result
    assert tg.actions[0] == ("forward_messages", 42, 42, [message.id])
    assert tg.actions[1][:3] == ("send_message", 42, "local comment")


@pytest.mark.parametrize("scope,chat_id", [(42, 43), (0, 42)])
async def test_remove_and_ban_cannot_target_arbitrary_chat(tg, monkeypatch, scope, chat_id):
    lookup = AsyncMock(side_effect=AssertionError("ban target lookup"))
    monkeypatch.setattr(tg, "get_chat", lookup)
    result = await execute(tt.remove_and_ban_chat(tg), {"chat_id": chat_id}, scope)
    assert "other chats" in result and tg.actions == []
    lookup.assert_not_awaited()


@pytest.mark.parametrize("scope", [None, 42])
async def test_lockdown_change_during_typing_blocks_inflight_send(tg, monkeypatch, scope):
    async def narrow(_):
        config().lockdown = LockdownMode.PAPIK_ONLY

    monkeypatch.setattr(stm, "simulate_typing_delay", narrow)
    result = await execute(stm.send_telegram_message(tg, FakeLLM(), tg.chats[42], []), send_args(), scope)
    assert "Lockdown mode is enabled" in result
    assert not any(action[0] == "send_message" for action in tg.actions)


async def test_lockdown_is_rechecked_between_split_messages(tg, monkeypatch):
    original = tg.send_message

    async def send_then_narrow(*args, **kwargs):
        sent = await original(*args, **kwargs)
        config().lockdown = LockdownMode.PAPIK_ONLY
        return sent

    monkeypatch.setattr(tg, "send_message", send_then_narrow)
    result = await execute(stm.send_telegram_message(tg, FakeLLM(), tg.chats[42], []),
                           send_args(text="first\nsecond"), 42)
    assert "Lockdown mode is enabled" in result
    assert [a[2] for a in tg.actions if a[0] == "send_message"] == ["first"]


@pytest.mark.parametrize("command", ["edit_message_text", "sticker_send", "forward_message", "group_admin_ban_user",
                                     "group_admin_set_user_tag"])
async def test_outgoing_actions_recheck_lockdown_after_awaiting_read(tg, monkeypatch, command):
    message = await tg.incoming(42, "original")
    tg.saved_stickers = [Message(0, 0, kind="sticker", extra={"sticker_id": 555})]
    if command == "sticker_send":
        read, args = "get_saved_stickers", {"sticker_id": 555}
    elif command == "forward_message":
        read, args = "get_message", {"message_ids": [message.id], "to_chat_id": 42, "comment": "private"}
    elif command == "edit_message_text":
        read, args = "get_message", {"message_id": message.id, "text": "new"}
    else:
        read, args = "get_user", {"user_id": 43, "tag": "new"}
    original = getattr(tg, read)

    async def read_then_narrow(*args, **kwargs):
        result = await original(*args, **kwargs)
        config().lockdown = LockdownMode.PAPIK_ONLY
        return result

    monkeypatch.setattr(tg, read, read_then_narrow)
    result = await execute(getattr(tt, command)(tg, tg.chats[42]), args, 42)
    assert result == "Error: this chat is not accessible."
    assert tg.actions == []


async def test_lockdown_change_during_contact_lookup_cannot_use_stale_permission(tg, monkeypatch):
    config().lockdown = LockdownMode.CONTACTS_ONLY

    async def contacts_then_narrow():
        config().lockdown = LockdownMode.PAPIK_ONLY
        return [42]

    monkeypatch.setattr(tg, "get_contact_ids", contacts_then_narrow)
    result = await execute(tt.react_with_emoji(tg, tg.chats[42]), {"message_id": 1, "emoji": "👍"}, 42)
    assert result == "Error: this chat is not accessible." and tg.actions == []


async def test_remove_and_ban_rechecks_lockdown_after_delay(tg, monkeypatch):
    async def narrow(_):
        config().lockdown = LockdownMode.PAPIK_ONLY

    monkeypatch.setattr(tt.asyncio, "sleep", narrow)
    result = await execute(tt.remove_and_ban_chat(tg), {"chat_id": 42}, 42)
    assert result == "Failed"
    assert [a[0] for a in tg.actions] == ["send_message"]


async def test_forward_comment_rechecks_lockdown_after_forward(tg, monkeypatch):
    message = await tg.incoming(42, "own message")
    original = tg.forward_messages

    async def forward_then_narrow(*args):
        await original(*args)
        config().lockdown = LockdownMode.PAPIK_ONLY

    monkeypatch.setattr(tg, "forward_messages", forward_then_narrow)
    result = await execute(tt.forward_message(tg, tg.chats[42]),
                           {"message_ids": [message.id], "to_chat_id": 42, "comment": "not sent"}, 42)
    assert result == "Error: this chat is not accessible."
    assert [a[0] for a in tg.actions] == ["forward_messages"]


@pytest.mark.parametrize("scope,extra", [(0, {}), (42, {"chat_id": 43})])
async def test_sticker_save_respects_scope(tg, scope, extra):
    result = await execute(tt.sticker_save(tg), {"sticker_id": 555, **extra}, scope)
    assert "other chats" in result and tg.actions == []


@pytest.mark.parametrize("kind,filename,field", [("gallery", "private.png", "photo_filename"),
                                                ("voice_messages", "private.ogg", "audio_filename")])
async def test_guessed_foreign_and_shared_legacy_attachments_are_rejected(tg, kind, filename, field):
    for directory in (Path("data") / kind, Path("data") / kind / "43"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(b"private contents")
    result = await execute(stm.send_telegram_message(tg, FakeLLM(), tg.chats[42], []),
                           send_args(text=None, **{field: filename}), 42)
    assert "file not found" in result
    assert not any(a[0] == "send_message" for a in tg.actions)


@pytest.mark.parametrize("kind,filename,field", [("gallery", "private.png", "photo_filename"),
                                                ("voice_messages", "private.ogg", "audio_filename")])
@pytest.mark.parametrize("scope", [None, 42])
async def test_attachments_cannot_follow_symlinks_to_config(tg, kind, filename, field, scope):
    secret = Path("config.toml").absolute()
    secret.write_text("api_key = 'private'")
    directory = mt.media_directory(Ctx(None, {}, session(scope), []), kind)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).symlink_to(secret)
    result = await execute(stm.send_telegram_message(tg, FakeLLM(), tg.chats[42], []),
                           send_args(text=None, **{field: filename}), scope)
    assert "Invalid" in result and "path" in result
    assert not any(a[0] == "send_message" for a in tg.actions)


@pytest.mark.parametrize("kind,filename,field", [("gallery", "new.png", "photo_filename"),
                                                ("voice_messages", "new.ogg", "audio_filename")])
@pytest.mark.parametrize("scope", [None, 42, 43])
async def test_generation_stores_media_in_chat_and_survives_clean_context(tg, monkeypatch, kind, filename, field, scope):
    async def generate(*args):
        path = Path("data") / kind / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated media")
        return SimpleNamespace(path=path)

    if kind == "gallery":
        monkeypatch.setattr(mt, "ImageGenerator", lambda *a: SimpleNamespace(generate=generate))
        monkeypatch.setattr(mt.media, "image", AsyncMock(return_value="a generated image"))
        tool, args = mt.take_photo(None, FakeLLM()), {"photo_desc": "selfie"}
    else:
        monkeypatch.setattr(mt, "VoiceGenerator", lambda: SimpleNamespace(generate=generate))
        tool, args = mt.record_audio(), {"speech": "привет"}
    generated = await execute(tool, args, scope)
    assert f"Filename: {filename}" in generated
    directory = Path("data") / kind
    expected = directory / filename if scope is None else directory / str(scope) / filename
    assert expected.read_bytes() == b"generated media"
    if scope is not None:
        assert not (directory / filename).exists()
    chat_id = scope if scope is not None else 42
    sent = await execute(stm.send_telegram_message(tg, FakeLLM(), tg.chats[chat_id], []),
                         send_args(text=None, **{field: filename}), scope)
    assert "Message sent successfully" in sent
    action = next(a for a in tg.actions if a[0] == "send_message")
    assert action[3 if kind == "gallery" else 4] == str(expected)


async def test_scope_less_sequence_keeps_legacy_admin_access(tg):
    await tg.incoming(43, "legacy admin raw search")
    tool = tt.search_messages(tg, FakeLLM(), Session())
    result = await tool.handler(Ctx(None, {"query": "legacy admin"}, [], []))
    assert "legacy admin raw search" in result
