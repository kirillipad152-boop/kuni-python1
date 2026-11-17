"""Notification queue. Port of NotificationManager.

Think of it as your phone's notifications: you receive a notification, read it and (maybe) react to it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .tools import ToolSet
from .util import now_str

log = logging.getLogger("NotificationManager")


class Flag:
    """AFuture<> analog: a one-shot signal that can be awaited and observed, and stays set."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._callbacks: list[Callable[[], None]] = []

    def set(self) -> None:
        if self._event.is_set():
            return
        self._event.set()
        for cb in self._callbacks:
            cb()
        self._callbacks.clear()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def on_set(self, cb: Callable[[], None]) -> None:
        if self._event.is_set():
            cb()
        else:
            self._callbacks.append(cb)


@dataclass
class Notification:
    # Notification message in natural language passed to LLM.
    message: str
    # Related tools available to the LLM when processing this notification (e.g. the "open" tool).
    actions: ToolSet = field(default_factory=ToolSet)
    # Higher priority notifications will be processed earlier.
    priority: int = 0
    # Optional freeform pin token that routes the notification to the worker that processed the same pin before,
    # e.g. "<chat id=1235 />": the same chat is processed by the same worker, which has more related context.
    pin: str | None = None
    chat_id: int | None = None
    message_id: int | None = None
    queued_at: float = field(default_factory=time.monotonic)


@dataclass
class NotificationHandle:
    notification: Notification
    on_started_processing: Flag = field(default_factory=Flag)  # resolves when the notification was passed to a worker
    on_processed: Flag = field(default_factory=Flag)  # resolved by the worker when it's completely processed


@dataclass
class _WorkerReg:
    pins: set[str]
    wake_up: Flag = field(default_factory=Flag)


class NotificationManager:
    def __init__(self) -> None:
        self._notifications: deque[NotificationHandle] = deque()
        self._workers: list[_WorkerReg] = []

    def oldest_age(self) -> float | None:
        if not self._notifications:
            return None
        return max(0.0, time.monotonic() - min(h.notification.queued_at for h in self._notifications))

    def pass_notification_to_ai(self, notification: Notification) -> NotificationHandle:
        log.debug("passNotificationToAI")
        handle = NotificationHandle(notification)
        at = len(self._notifications)
        for i, h in enumerate(self._notifications):
            if notification.priority > h.notification.priority:
                at = i
                break
        self._notifications.insert(at, handle)

        if notification.pin:
            # wake up suitable worker based on pin.
            for worker in self._workers:
                if notification.pin in worker.pins:
                    worker.wake_up.set()
                    return handle

        # wake up first idle worker.
        for worker in self._workers:
            if not worker.wake_up.is_set():
                worker.wake_up.set()
                return handle
        return handle

    def clear(self) -> int:
        """Drops everything that is still waiting (owner's decision); a turn already running is not touched."""
        dropped = list(self._notifications)
        self._notifications.clear()
        for handle in dropped:
            n = handle.notification
            logging.getLogger("MessageFlow").info("event=discarded chat_id=%s message_id=%s reason=queue_cleared_by_owner",
                                                 n.chat_id, n.message_id)
            handle.on_processed.set()
        return len(dropped)

    def remove_notifications(self, substring: str) -> None:
        """Removes obsolete notifications from the queue by the given substring (must be unique enough)."""
        log.debug("removeNotifications: %s", substring)
        kept = deque()
        for handle in self._notifications:
            if substring in handle.notification.message:
                n = handle.notification
                logging.getLogger("MessageFlow").info("event=coalesced chat_id=%s message_id=%s reason=chat_history_refresh",
                                                     n.chat_id, n.message_id)
                handle.on_processed.set()
            else:
                kept.append(handle)
        self._notifications = kept

    def _next(self, pins: set[str]) -> NotificationHandle | None:
        def take(h: NotificationHandle) -> NotificationHandle:
            self._notifications.remove(h)
            if h.notification.pin:
                pins.add(h.notification.pin)
            return h

        for h in list(self._notifications):
            pin = h.notification.pin
            if not pin:
                return take(h)
            if pin in pins:
                return take(h)
            if any(pin in w.pins for w in self._workers):
                continue  # this notification is pinned to other worker, skip.
            return take(h)
        return None

    async def run(self, worker_pins: set[str], worker: Callable[[Notification], Awaitable[bool]]) -> None:
        """Registers a worker and runs it in a loop. `worker` returns False to break."""
        reg = _WorkerReg(worker_pins)
        self._workers.append(reg)
        try:
            while True:
                handle = self._next(worker_pins)
                if handle is None:
                    await reg.wake_up.wait()
                    reg.wake_up = Flag()  # reset
                    continue
                handle.notification.message += f"\nCurrent time: {now_str()} UTC"
                handle.on_started_processing.set()
                try:
                    ok = await worker(handle.notification)
                finally:
                    handle.on_processed.set()
                if not ok:
                    break
        finally:
            self._workers.remove(reg)
