import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from kuni import reminders as rem
from kuni.llm import Session
from kuni.tools import Ctx, ToolError
from kuni.tools_impl.reminder_tools import reminder_tools


def test_store_roundtrip_and_isolation(tmp_path):
    path = tmp_path / "reminders.json"
    store = rem.ReminderStore(path)
    first = store.add(42, 60, "first", now=0)
    second = store.add(43, 90, "private", now=0)
    loaded = rem.ReminderStore(path)
    assert loaded.list(42) == [first] and loaded.list(43) == [second]
    assert path.stat().st_mode & 0o777 == 0o600
    assert not loaded.remove(second.id, 42) and loaded.remove(first.id, 42)
    assert rem.ReminderStore(path).list(42) == []
    assert not loaded.due(89) and loaded.due(90) == [second]


@pytest.mark.parametrize("body", [b"not JSON", b'{"reminders": [{}]}', b'{"reminders": null}',
    b'{"reminders": [{"id":"r123456","chat_id":42,"due":NaN,"created":0,"note":"x"}]}'])
def test_corruption_is_quarantined_without_data_loss(tmp_path, body):
    path = tmp_path / "reminders.json"
    path.write_bytes(body)
    store = rem.ReminderStore(path)
    assert not store.list(42) and not path.exists()
    quarantined = list(tmp_path.glob("*.corrupt-*"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == body
    store.add(42, 60, "new", now=0)
    assert rem.ReminderStore(path).list(42)


def test_unquarantinable_file_cannot_be_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "reminders.json"
    path.write_bytes(b"original bad data")

    def denied(*args):
        raise PermissionError

    monkeypatch.setattr(Path, "rename", denied)
    store = rem.ReminderStore(path)
    with pytest.raises(OSError):
        store.add(42, 60, "new", now=0)
    assert path.read_bytes() == b"original bad data" and not store.list(42)


@pytest.mark.parametrize("due,note", [(4, "soon"), (rem.MAX_AHEAD_SECONDS + 1, "late"),
    (float("nan"), "nan"), (float("inf"), "inf"), (True, "bool"), (60, ""), (60, " "),
    (60, "x" * 501), (60, "bad\x00note")])
def test_limits(tmp_path, due, note):
    store = rem.ReminderStore(tmp_path / "r.json")
    with pytest.raises(ValueError):
        store.add(42, due, note, now=0)
    assert not store.list(42)


def test_caps_and_failed_writes_preserve_state(tmp_path, monkeypatch):
    store = rem.ReminderStore(tmp_path / "r.json")
    for _ in range(20):
        store.add(42, 60, "task", now=0)
    with pytest.raises(ValueError, match="pending reminders"):
        store.add(42, 60, "overflow", now=0)
    item = store.add(43, 60, "other", now=0)

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_save", fail)
    with pytest.raises(OSError):
        store.drop(item.id)
    assert store.list(43) == [item]
    with pytest.raises(OSError):
        store.bump_attempt(item.id, now=100)
    assert item.attempts == 0 and item.retry_at == 0
    with pytest.raises(OSError):
        store.add(44, 60, "unsaved", now=0)
    assert not store.list(44)


@pytest.mark.parametrize("value", [1, 10.0, "60", "60.0"])
def test_relative_time(value):
    assert rem.parse_when(value, None, 1000) == 1000 + float(value) * 60


@pytest.mark.parametrize("minutes,at", [(None, None), (1, "2026-11-16 12:00"), (True, None),
    (float("nan"), None), (float("inf"), None), (0, None), (525601, None), (1.5, None),
    (None, "garbage"), (None, "2026-13-01 12:00"), (None, "2026-11-16 12:00:99")])
def test_invalid_times(minutes, at):
    with pytest.raises(ValueError):
        rem.parse_when(minutes, at, 0)


@pytest.mark.parametrize("value", ["2026-11-16 12:00", "2026-11-16T12:00:00Z", "2026-11-16T17:00+05:00"])
def test_utc_and_offset_times(value):
    expected = datetime(2026, 11, 16, 12, tzinfo=timezone.utc).timestamp()
    assert rem.parse_when(None, value, 0) == expected


async def test_scheduler_late_delivery_once_and_failed_delivery_retained(tmp_path, caplog):
    store = rem.ReminderStore(tmp_path / "r.json")
    bad = store.add(42, 60, "PRIVATE_NOTE", now=0)
    good = store.add(43, 60, "fine", now=0)
    delivered = []

    async def fire(item, late):
        if item.id == bad.id:
            raise ConnectionError("temporary outage")
        delivered.append((item.id, late))

    await rem.fire_due(store, fire, 90)
    assert delivered == [(good.id, 30)]
    assert store.list(42) == [bad] and not store.list(43)
    for _ in range(4):
        await rem.fire_due(store, fire, bad.retry_at)
    assert bad.attempts == 5 and rem.ReminderStore(store.path).list(42)
    assert "PRIVATE_NOTE" not in caplog.text
    await rem.fire_due(store, fire, 90)  # retry_at has not arrived
    assert bad.attempts == 5


async def test_scheduler_cancellation_and_pause_keep_reminder(tmp_path):
    store = rem.ReminderStore(tmp_path / "r.json")
    item = store.add(42, 60, "task", now=0)

    async def cancelled(*args):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await rem.fire_due(store, cancelled, 60)
    assert store.list(42) == [item] and item.attempts == 0

    async def stop(_):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await rem.run_scheduler(store, cancelled, clock=lambda: 60, sleep=stop, active=lambda: False)
    assert store.list(42) == [item]


async def test_tools_are_strict_and_do_not_cross_chat_boundaries(tmp_path):
    store = rem.ReminderStore(tmp_path / "r.json")
    tools = {t.name: t for t in reminder_tools(store, clock=lambda: 0)}
    for tool in tools.values():
        assert set(tool.required) == set(tool.properties)
    ctx = Ctx(None, {"note": "private", "in_minutes": 1, "at_utc": None}, Session(chat_id=42), [])
    assert "UTC" in await tools["set_reminder"].handler(ctx)
    item = store.list(42)[0]
    ctx.temporary_context = Session(chat_id=43)
    assert "private" not in await tools["list_reminders"].handler(ctx)
    ctx.args = {"reminder_id": item.id}
    with pytest.raises(ToolError, match="not found"):
        await tools["cancel_reminder"].handler(ctx)
    ctx.temporary_context = Session(chat_id=42)
    assert "cancelled" in await tools["cancel_reminder"].handler(ctx)
    ctx.temporary_context = Session()
    ctx.args = {"note": "trusted", "in_minutes": "1", "at_utc": None}
    with pytest.raises(ToolError, match="require a specific chat"):
        await tools["set_reminder"].handler(ctx)
    assert not store.list(0)


async def test_cancelled_later_delivery_in_batch_is_not_sent(tmp_path):
    store = rem.ReminderStore(tmp_path / "r.json")
    first = store.add(42, 60, "first", now=0)
    second = store.add(42, 60, "second", now=0)
    sent = []

    async def fire(item, late):
        sent.append(item.id)
        if item.id == first.id:
            store.remove(second.id, 42)

    await rem.fire_due(store, fire, 60)
    assert sent == [first.id] and not store.list(42)
