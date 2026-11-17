"""Reminders: a small persistent store plus a polling scheduler. Firing is delegated to the app (`fire` callback)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import secrets
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from .util import atomic_write_text

log = logging.getLogger("reminders")

MIN_LEAD_SECONDS = 5
MAX_AHEAD_SECONDS = 365 * 86400
MAX_MINUTES = 525600  # 365 days
MAX_NOTE = 500
MAX_PER_CHAT = 20
MAX_TOTAL = 500


@dataclass
class Reminder:
    id: str
    chat_id: int
    due: float  # unix seconds, UTC
    note: str
    created: float
    attempts: int = 0
    retry_at: float = 0.0


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _from_json(d: dict) -> Reminder:
    r = Reminder(**d)
    if not (isinstance(r.id, str) and re.fullmatch(r"r[0-9a-f]{6}", r.id)
            and isinstance(r.note, str) and 1 <= len(r.note) <= MAX_NOTE and isinstance(r.chat_id, int)
            and not isinstance(r.chat_id, bool) and _is_num(r.due) and _is_num(r.created)
            and isinstance(r.attempts, int) and not isinstance(r.attempts, bool) and r.attempts >= 0
            and _is_num(r.retry_at) and all(math.isfinite(v) for v in (r.due, r.created, r.retry_at))):
        raise ValueError("bad reminder fields")
    return r


class ReminderStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._items: dict[str, Reminder] = {}
        self._writable = True
        self._load()

    # -- persistence ---------------------------------------------------------------------------------------------
    def _load(self) -> None:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return
        except OSError as e:
            log.error("can't read %s: %s", self.path, e)
            self._quarantine()
            return
        try:
            items = [_from_json(d) for d in json.loads(raw)["reminders"]]
            if len({r.id for r in items}) != len(items):
                raise ValueError("duplicate ids")
        except Exception as e:  # noqa: BLE001  (any malformed content: keep the bytes, start empty)
            log.error("reminders file %s is corrupt (%s)", self.path, type(e).__name__)
            self._quarantine()
            return
        self._items = {r.id: r for r in items}

    def _quarantine(self) -> None:
        target = self.path.with_name(f"{self.path.name}.corrupt-{time.time_ns()}")
        try:
            self.path.rename(target)
            target.chmod(0o600)
            log.error("moved the unreadable reminders file to %s; starting empty", target)
        except OSError as e:
            self._writable = False
            log.error("can't quarantine %s (%s); writes disabled to preserve the original", self.path, type(e).__name__)

    def _save(self) -> None:
        if not self._writable:
            raise OSError("reminders are read-only until the damaged file is safely recovered")
        atomic_write_text(self.path, json.dumps({"reminders": [asdict(r) for r in self._items.values()]},
                                                ensure_ascii=False, indent=1))

    # -- API -----------------------------------------------------------------------------------------------------
    def add(self, chat_id: int, due: float, note: str, now: float | None = None) -> Reminder:
        now = time.time() if now is None else now
        if not isinstance(chat_id, int) or isinstance(chat_id, bool) or chat_id == 0:
            raise ValueError("chat_id must be a nonzero integer")
        note = note.strip() if isinstance(note, str) else ""
        if not 1 <= len(note) <= MAX_NOTE:
            raise ValueError(f"note must be 1-{MAX_NOTE} characters")
        if any(c != "\n" and unicodedata.category(c) == "Cc" for c in note):
            raise ValueError("note must not contain control characters")
        if not _is_num(due) or not math.isfinite(due) or due < now + MIN_LEAD_SECONDS:
            raise ValueError(f"reminder time must be at least {MIN_LEAD_SECONDS} seconds in the future")
        if due > now + MAX_AHEAD_SECONDS:
            raise ValueError("reminder time must be within 365 days from now")
        if sum(r.chat_id == chat_id for r in self._items.values()) >= MAX_PER_CHAT:
            raise ValueError(f"too many pending reminders in this chat (max {MAX_PER_CHAT}); cancel some first")
        if len(self._items) >= MAX_TOTAL:
            raise ValueError("too many pending reminders overall; try again later")
        rid = "r" + secrets.token_hex(3)
        while rid in self._items:
            rid = "r" + secrets.token_hex(3)
        r = self._items[rid] = Reminder(rid, chat_id, float(due), note, now)
        try:
            self._save()
        except Exception:
            del self._items[rid]
            raise
        return r

    def all(self) -> list[Reminder]:
        return sorted(self._items.values(), key=lambda r: r.due)

    def list(self, chat_id: int) -> list[Reminder]:
        return sorted((r for r in self._items.values() if r.chat_id == chat_id), key=lambda r: r.due)

    def remove(self, reminder_id: str, chat_id: int) -> bool:
        r = self._items.get(reminder_id)
        if r is None or r.chat_id != chat_id:
            return False
        self.drop(reminder_id)
        return True

    def due(self, now: float) -> list[Reminder]:
        return sorted((r for r in self._items.values() if r.due <= now and r.retry_at <= now), key=lambda r: r.due)

    def drop(self, reminder_id: str) -> None:
        old = self._items.pop(reminder_id, None)
        if old is not None:
            try:
                self._save()
            except BaseException:
                self._items[reminder_id] = old
                raise

    def bump_attempt(self, reminder_id: str, now: float | None = None) -> int:
        r = self._items.get(reminder_id)
        if r is None:
            return 0
        previous = r.attempts, r.retry_at
        r.attempts += 1
        r.retry_at = (time.time() if now is None else now) + min(900, 15 * 2 ** min(r.attempts - 1, 6))
        try:
            self._save()
        except BaseException:
            r.attempts, r.retry_at = previous
            raise
        return r.attempts


# --------------------------------------------------------------------------------------------------------------------
_AT_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ Tt]\d{2}:\d{2}(:\d{2})?(Z|z|[+-]\d{2}:\d{2})?")


def parse_when(in_minutes, at_utc, now: float) -> float:
    """Absolute due time (unix seconds) from exactly one of `in_minutes` / `at_utc`."""
    if (in_minutes is None) == (at_utc is None):
        raise ValueError("provide exactly one of in_minutes or at_utc")
    if in_minutes is not None:
        try:
            if isinstance(in_minutes, bool) or not isinstance(in_minutes, (int, float, str)):
                raise ValueError
            m = float(in_minutes)
            if not math.isfinite(m) or m != int(m):
                raise ValueError
        except (ValueError, OverflowError):
            raise ValueError("in_minutes must be a whole number of minutes") from None
        if not 1 <= m <= MAX_MINUTES:
            raise ValueError(f"in_minutes must be between 1 and {MAX_MINUTES}")
        return now + int(m) * 60
    s = at_utc.strip() if isinstance(at_utc, str) else ""
    if not _AT_RE.fullmatch(s):
        raise ValueError('at_utc must look like "YYYY-MM-DD HH:MM" (UTC), optionally with Z or a +HH:MM offset')
    try:
        dt = datetime.fromisoformat(s.replace("z", "Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, OverflowError):
        raise ValueError("at_utc is not a valid date and time") from None


# --------------------------------------------------------------------------------------------------------------------
Fire = Callable[[Reminder, float], Awaitable[None]]


async def fire_due(store: ReminderStore, fire: Fire, now: float) -> None:
    """Failed delivery remains durable and retries with backoff, never disappears during an outage."""
    for r in store.due(now):
        if r not in store.list(r.chat_id):
            continue  # it may have been cancelled while an earlier delivery was awaiting Telegram
        try:
            async with asyncio.timeout(30):
                await fire(r, max(0.0, now - r.due))
        except Exception as e:  # noqa: BLE001  (CancelledError is a BaseException and propagates)
            attempts = store.bump_attempt(r.id, now)
            log.warning("reminder delivery failed id=%s chat_id=%d attempt=%d error=%s; retained for retry",
                        r.id, r.chat_id, attempts, type(e).__name__)
        else:
            store.drop(r.id)


async def run_scheduler(store: ReminderStore, fire: Fire, *, interval: float = 15.0, clock=time.time,
                         sleep=asyncio.sleep, active=lambda: True) -> None:
    while True:
        try:
            if active():
                await fire_due(store, fire, clock())
        except Exception:  # noqa: BLE001  (e.g. disk full while dropping: keep the scheduler alive)
            log.exception("reminder scheduler step failed")
        await sleep(interval)
