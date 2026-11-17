"""Reminder tool factories; registration and the scheduler belong to the app."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from ..reminders import MAX_PER_CHAT, ReminderStore, parse_when
from ..tools import Ctx, Property, Tool, ToolError
from ..util import opt_str


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _scope(ctx: Ctx) -> int:
    chat_id = getattr(ctx.temporary_context, "chat_id", None)
    if not isinstance(chat_id, int) or isinstance(chat_id, bool) or chat_id == 0:
        raise ToolError("Reminders require a specific chat")
    return chat_id


def reminder_tools(store: ReminderStore, *, clock=time.time,
                   validate: Callable[[int], Awaitable[None]] | None = None) -> list[Tool]:
    async def set_handler(ctx: Ctx) -> str:
        chat_id = _scope(ctx)
        if validate is not None:
            await validate(chat_id)
        note = ctx.args.get("note")
        if not isinstance(note, str):
            raise ToolError("note is required string")
        in_minutes = ctx.args.get("in_minutes")
        if isinstance(in_minutes, str):  # some LLMs pass "null"/"None" instead of JSON null
            in_minutes = opt_str(in_minutes)
        now = clock()
        try:
            due = parse_when(in_minutes, opt_str(ctx.args.get("at_utc")), now)
            r = store.add(chat_id, due, note, now)
        except ValueError as e:
            raise ToolError(str(e)) from None
        return f"Reminder {r.id} set for {_utc(r.due)}."

    async def list_handler(ctx: Ctx) -> str:
        items = store.list(_scope(ctx))
        if not items:
            return "No pending reminders."
        return "\n".join(f"- {r.id}: {_utc(r.due)}{' (delivery retry pending)' if r.attempts else ''}: {r.note}"
                         for r in items)

    async def cancel_handler(ctx: Ctx) -> str:
        rid = ctx.args.get("reminder_id")
        if not isinstance(rid, str) or not store.remove(rid.strip(), _scope(ctx)):
            raise ToolError("reminder not found")
        return f"Reminder {rid.strip()} cancelled."

    return [
        Tool(
            "set_reminder",
            "Schedule a reminder for this chat. Its note is sent directly to this chat at the scheduled time, without "
            f"waiting for an LLM response. At most {MAX_PER_CHAT} pending reminders per chat. All times are UTC (the "
            "notifications show \"Current time: ... UTC\"): convert the user's local time yourself, or ask for their "
            "timezone if you don't know it. For relative requests (\"in 2 hours\") prefer in_minutes. Give exactly one "
            "of in_minutes / at_utc and null for the other.",
            set_handler,
            {"note": Property("string", "What to remind about, 1-500 characters, in your own words, with enough "
                              "context to act on it later; no control characters."),
             "in_minutes": Property("integer", "Fire after this many minutes from now, 1-525600. Null if at_utc is "
                                    "used.", nullable=True),
             "at_utc": Property("string", 'Absolute UTC time "YYYY-MM-DD HH:MM" (an optional Z or +HH:MM offset is '
                                "converted to UTC). Null if in_minutes is used.", nullable=True)},
            ["note", "in_minutes", "at_utc"]),
        Tool("list_reminders", "List this chat's pending reminders with their ids and due times (UTC).",
             list_handler, {}, []),
        Tool("cancel_reminder", "Cancel a pending reminder of this chat by its id (see list_reminders).",
             cancel_handler, {"reminder_id": Property("string", 'Reminder id such as "r1a2b3c".')}, ["reminder_id"]),
    ]
