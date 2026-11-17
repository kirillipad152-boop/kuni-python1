"""Pure watchdog logic: decides when the owner must be alerted. No I/O except the injected notify."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import socket
import time

from .util import atomic_write_text

log = logging.getLogger("Watchdog")

QUEUE_STUCK_SECONDS = 30 * 60
STALL_WINDOW = 10 * 60  # window for counting 429s
STALL_RATE_LIMITS = 5
NO_SUCCESS_SECONDS = 15 * 60
DISK_LOW_RATIO = 0.10
PROXY_DOWN_CHECKS = 3
ALERT_COOLDOWN = 3 * 60 * 60

_ALERTS = {
    "queue_stuck": "Очередь: самое старое уведомление ждёт уже {minutes} мин.",
    "llm_stalled": "Нейросеть не отвечает: сплошные 429 и ни одного успешного ответа за {minutes} мин.",
    "disk_low": "Мало места на диске: свободно {percent}%.",
    "proxy_down": "Прокси для YouTube не отвечает несколько проверок подряд.",
}
_RECOVERED = {
    "queue_stuck": "Очередь уведомлений снова движется.",
    "llm_stalled": "Нейросеть снова отвечает.",
    "disk_low": "Место на диске в порядке.",
    "proxy_down": "Прокси для YouTube снова работает.",
}


@dataclass
class Probe:
    queue_oldest_age: float | None  # None = queue empty
    worker_busy: bool
    sleeping: bool
    paused: bool
    disk_free_ratio: float
    youtube_proxy_ok: bool | None  # None = no proxy configured


class Watchdog:
    def __init__(self, notify: Callable[[str], Awaitable[None]], clock: Callable[[], float] = time.monotonic):
        self._notify = notify
        self._clock = clock
        self._rate_limits: deque[float] = deque()
        self._last_success = clock()  # startup counts as progress
        self._last_alert: dict[str, float] = {}
        self._active: set[str] = set()
        self._proxy_failures = 0

    def record_rate_limit(self) -> None:
        now = self._clock()
        self._rate_limits.append(now)
        while self._rate_limits and now - self._rate_limits[0] > STALL_WINDOW:
            self._rate_limits.popleft()

    def record_llm_success(self) -> None:
        self._last_success = self._clock()
        self._rate_limits.clear()

    async def check(self, probe: Probe) -> None:
        now = self._clock()
        while self._rate_limits and now - self._rate_limits[0] > STALL_WINDOW:
            self._rate_limits.popleft()
        self._proxy_failures = self._proxy_failures + 1 if probe.youtube_proxy_ok is False else 0
        since_success = now - self._last_success
        queue_age = probe.queue_oldest_age
        conditions = {
            "queue_stuck": (
                queue_age is not None and queue_age > QUEUE_STUCK_SECONDS and not probe.paused and not probe.sleeping,
                {"minutes": int((queue_age or 0) // 60)},
            ),
            "llm_stalled": (
                (len(self._rate_limits) >= STALL_RATE_LIMITS and since_success > NO_SUCCESS_SECONDS)
                or ("llm_stalled" in self._active and self._last_success < self._last_alert["llm_stalled"]),
                {"minutes": int(since_success // 60)},
            ),
            "disk_low": (
                probe.disk_free_ratio < DISK_LOW_RATIO,
                {"percent": round(probe.disk_free_ratio * 100)},
            ),
            "proxy_down": (self._proxy_failures >= PROXY_DOWN_CHECKS, {}),
        }
        for kind, (bad, fields) in conditions.items():
            if kind in {"queue_stuck", "llm_stalled"} and (probe.paused or probe.sleeping):
                continue  # deliberate inactivity is neither a fault nor proof of recovery
            if bad:
                last = self._last_alert.get(kind)
                if last is None or now - last >= ALERT_COOLDOWN:
                    if await self._send(_ALERTS[kind].format(**fields)):
                        self._last_alert[kind] = now
                        self._active.add(kind)
            elif kind in self._active:
                if await self._send(_RECOVERED[kind]):
                    self._active.discard(kind)

    async def _send(self, text: str) -> bool:
        try:
            await self._notify(text)
            return True
        except Exception:
            log.exception("watchdog notify failed, will retry")
            return False

    async def run(
        self,
        probe_fn: Callable[[], Awaitable[Probe]],
        interval: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        active: Callable[[], bool] = lambda: True,
    ) -> None:
        while True:
            try:
                if active():
                    await self.check(await probe_fn())
            except Exception:  # CancelledError is BaseException and propagates
                log.exception("watchdog check failed")
            await sleep(interval)


def was_unclean_shutdown(path: Path) -> bool:
    return Path(path).exists()


def mark_running(path: Path) -> bool:
    """Write the run marker; return True if the previous run left one (unclean shutdown)."""
    path = Path(path)
    unclean = path.exists()
    atomic_write_text(path, "running\n")
    return unclean


def mark_clean_shutdown(path: Path) -> None:
    Path(path).unlink(missing_ok=True)


def unclean_restart_message() -> str:
    return "Куни перезапустилась после аварийной остановки (в прошлый раз она не завершилась штатно)."


def notify_systemd(state: str) -> None:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1)
            sock.sendto(state.encode(), address)
    except OSError:
        log.warning("systemd notification could not be sent")


async def heartbeat() -> None:
    notify_systemd("READY=1")
    interval = max(1, int(os.environ.get("WATCHDOG_USEC", "120000000")) / 2_000_000)
    while True:
        notify_systemd("WATCHDOG=1")
        await asyncio.sleep(interval)
