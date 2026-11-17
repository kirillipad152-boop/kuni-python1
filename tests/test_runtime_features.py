import asyncio
import logging
import socket
import time

import pytest

from conftest import FakeLLM, assistant, wait_until
from kuni import config as cfgmod
from kuni.config import config, LockdownMode
from kuni.llm import LLMError, Message, Session
from kuni.notifications import Notification
from kuni.reminders import fire_due
from kuni.telegram.fake import FakeTelegram
from kuni.telegram.types import ChatType, MemberStatus, TelegramError
from kuni.telegram_app import TelegramApp
from kuni.tools import Ctx, ToolError, ToolSet
from kuni import watchdog as wd


@pytest.fixture
async def app_world():
    config().papikChatId = 42
    tg = FakeTelegram()
    tg.add_user(42, "Owner")
    tg.add_user(43, "Friend")
    llm = FakeLLM(default=lambda p, m: assistant("", [("wait", {})]))
    app = TelegramApp(tg, llm)
    yield app, tg, llm
    await app.close()


async def test_message_audit_has_reasons_but_no_content(app_world, caplog, monkeypatch):
    app, tg, llm = app_world
    caplog.set_level(logging.INFO, logger="MessageFlow")
    config().lockdown = LockdownMode.PAPIK_ONLY
    await tg.incoming(43, "DO_NOT_LOG_PRIVATE_TEXT")
    assert "reason=lockdown" in caplog.text
    config().lockdown = LockdownMode.NONE
    config().chatNotificationFilter = LockdownMode.PAPIK_ONLY
    await tg.incoming(43, "DO_NOT_LOG_PRIVATE_TEXT")
    assert "reason=notification_filter" in caplog.text
    config().chatNotificationFilter = LockdownMode.NONE
    tg.chats[43].muted = True
    monkeypatch.setattr("kuni.telegram_app.rng.random", lambda: 0)
    await tg.incoming(43, "DO_NOT_LOG_PRIVATE_TEXT")
    assert "reason=muted" in caplog.text
    flow_records = [r.getMessage() for r in caplog.records if r.name == "MessageFlow"]
    assert all("DO_NOT_LOG" not in text for text in flow_records)
    assert not llm.calls


async def test_new_capabilities_registered_and_private_reminder_is_sent_without_llm(app_world):
    app, tg, llm = app_world
    config().capabilityWebReader = config().capabilityReminders = True
    tools = ToolSet()
    app.update_tools(tools, Session(chat_id=43))
    assert {"read_webpage", "set_reminder", "list_reminders", "cancel_reminder"} <= tools.handlers().keys()
    assert "read_webpage" in app.get_system_prompt(43) and "UTC" in app.get_system_prompt(43)
    item = app.reminders.add(43, 60, "PRIVATE_REMINDER", now=0)
    await fire_due(app.reminders, app._deliver_reminder, 90)
    sent = [a for a in tg.actions if a[0] == "send_message"]
    assert len(sent) == 1 and sent[0][1] == 43 and "PRIVATE_REMINDER" in sent[0][2]
    assert not app.reminders.list(43) and not llm.calls
    assert app.workers[0]._wake_up.is_set()
    tools = ToolSet()
    app.update_tools(tools, Session(chat_id=0))
    assert "set_reminder" not in tools.handlers()


async def test_reminder_respects_narrowed_lockdown(app_world):
    app, tg, llm = app_world
    config().capabilityReminders = True
    config().lockdown = LockdownMode.PAPIK_ONLY
    app.reminders.add(43, 60, "BLOCKED_NOTE", now=0)
    await fire_due(app.reminders, app._deliver_reminder, 60)
    assert app.reminders.list(43)[0].attempts == 1
    assert not any(a[0] == "send_message" for a in tg.actions)
    config().lockdown = LockdownMode.NONE
    await fire_due(app.reminders, app._deliver_reminder, 75)
    assert not app.reminders.list(43)
    assert any(a[0] == "send_message" and a[1] == 43 for a in tg.actions)


@pytest.mark.parametrize("kind,status", [(ChatType.CHANNEL, MemberStatus.MEMBER),
                                         (ChatType.SUPERGROUP, MemberStatus.LEFT),
                                         (ChatType.BASIC_GROUP, MemberStatus.BANNED)])
async def test_readonly_chat_cannot_set_reminders(app_world, monkeypatch, kind, status):
    app, tg, llm = app_world
    config().capabilityReminders = True
    tg.add_group(-100, "Read-only", kind)

    async def membership(*args):
        return status

    monkeypatch.setattr(tg, "get_member_status", membership)
    tools = ToolSet()
    context = Session(chat_id=-100)
    app.update_tools(tools, context)
    tool = tools.handlers()["set_reminder"]
    with pytest.raises(ToolError):
        await tool.handler(Ctx(tools, {"note": "note", "in_minutes": 1, "at_utc": None}, context, []))
    assert not app.reminders.list(-100)


async def test_telegram_delivery_forbidden_retains_reminder_and_pause_blocks_batch(app_world, monkeypatch):
    app, tg, llm = app_world
    config().capabilityReminders = True
    item = app.reminders.add(43, 60, "private", now=0)

    async def forbidden(*args, **kwargs):
        raise TelegramError("USER_IS_BLOCKED")

    monkeypatch.setattr(tg, "send_message", forbidden)
    await fire_due(app.reminders, app._deliver_reminder, 60)
    assert app.reminders.list(43) == [item] and item.attempts == 1
    config().paused = True
    await fire_due(app.reminders, app._deliver_reminder, item.retry_at)
    assert app.reminders.list(43) == [item] and not llm.calls


async def test_upstream_json_error_does_not_erase_chat_context(app_world, monkeypatch, caplog):
    app, tg, llm = app_world
    context = app.get_context(43)
    context.append(Message(content="KEEP_EXISTING_MEMORY"))

    async def broken(*args):
        raise LLMError("upstream returned invalid JSON")

    monkeypatch.setattr(llm, "chat", broken)
    caplog.set_level(logging.INFO, logger="MessageFlow")
    await app.workers[0].handle_notification(Notification("new", chat_id=43, message_id=7))
    assert any(m.content == "KEEP_EXISTING_MEMORY" for m in context)
    assert "event=processing_failed chat_id=43 message_id=7 error=LLMError" in caplog.text


async def test_wait_is_audited_without_message_contents(app_world, caplog):
    app, tg, llm = app_world
    caplog.set_level(logging.INFO, logger="MessageFlow")
    await tg.incoming(43, "PRIVATE_INPUT")
    await wait_until(lambda: any("event=turn_finished" in r.getMessage() for r in caplog.records))
    records = [r.getMessage() for r in caplog.records if r.name == "MessageFlow"]
    assert any("event=wait" in text and "reason=llm_decision" in text for text in records)
    assert all("PRIVATE_INPUT" not in text for text in records)


async def test_notify_is_direct_and_watchdog_sees_inflight_age(app_world, monkeypatch):
    app, tg, llm = app_world
    await app._notify_owner("Тест состояния")
    assert any(a[0] == "send_message" and a[1] == 42 for a in tg.actions) and not llm.calls
    app.workers[0].current_notification = Notification("secret", queued_at=time.monotonic() - 2000)
    app.workers[0].sleeping = True
    probe = await app._watchdog_probe()
    assert probe.queue_oldest_age >= 2000 and probe.sleeping
    await app.watchdog.check(probe)
    assert len([a for a in tg.actions if a[0] == "send_message"]) == 1  # no false stuck alert while asleep


def test_sd_notify_uses_local_socket(tmp_path, monkeypatch):
    address = str(tmp_path / "notify.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
        server.bind(address)
        server.settimeout(1)
        monkeypatch.setenv("NOTIFY_SOCKET", address)
        wd.notify_systemd("READY=1")
        assert server.recv(128) == b"READY=1"


async def test_heartbeat_sends_ready_and_periodic_watchdog(monkeypatch):
    sent, intervals = [], []
    monkeypatch.setattr(wd, "notify_systemd", sent.append)
    monkeypatch.setenv("WATCHDOG_USEC", "6000000")

    async def stop(interval):
        intervals.append(interval)
        raise asyncio.CancelledError

    monkeypatch.setattr(wd.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await wd.heartbeat()
    assert sent == ["READY=1", "WATCHDOG=1"] and intervals == [3]


async def test_watchdog_can_be_disabled(app_world):
    app, tg, llm = app_world
    config().watchdogEnabled = False

    async def bad_probe():
        raise AssertionError("disabled watchdog must not probe")

    async def stop(_):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await app.watchdog.run(bad_probe, sleep=stop, active=lambda: config().watchdogEnabled)
    assert not any(a[0] == "send_message" for a in tg.actions)
